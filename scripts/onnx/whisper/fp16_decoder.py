# ruff: noqa: T201

import argparse
import os
import time

import onnx
from onnxconverter_common import float16
from optimum.onnx.graph_transformations import check_and_save_model


def subgraph_free_names(graph: onnx.GraphProto) -> set[str]:
    local = {init.name for init in graph.initializer}
    local.update(value.name for value in graph.input)
    free = set()
    for node in graph.node:
        for name in node.input:
            if name and name not in local:
                free.add(name)
        for attribute in node.attribute:
            for sub in _attribute_graphs(attribute):
                free.update(subgraph_free_names(sub))
        local.update(node.output)
    return free - local


def _attribute_graphs(attribute: onnx.AttributeProto) -> list[onnx.GraphProto]:
    if attribute.type == onnx.AttributeProto.GRAPH:
        return [attribute.g]
    if attribute.type == onnx.AttributeProto.GRAPHS:
        return list(attribute.graphs)
    return []


def node_dependencies(node: onnx.NodeProto) -> set[str]:
    deps = {name for name in node.input if name}
    for attribute in node.attribute:
        for sub in _attribute_graphs(attribute):
            deps.update(subgraph_free_names(sub))
    return deps


def toposort_graph(graph: onnx.GraphProto) -> None:
    produced = {init.name for init in graph.initializer}
    produced.update(value.name for value in graph.input)
    deps = {id(node): node_dependencies(node) for node in graph.node}
    pending = list(graph.node)
    ordered = []
    while pending:
        ready = [node for node in pending if deps[id(node)] <= produced]
        if not ready:
            raise RuntimeError(f"cycle or missing producer among {len(pending)} nodes")
        for node in ready:
            produced.update(node.output)
        ordered.extend(ready)
        ready_ids = {id(node) for node in ready}
        pending = [node for node in pending if id(node) not in ready_ids]
    del graph.node[:]
    graph.node.extend(ordered)


def rename_value(graph: onnx.GraphProto, old: str, new: str) -> None:
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
        for attribute in node.attribute:
            for sub in _attribute_graphs(attribute):
                rename_value(sub, old, new)


def keep_fp32_inputs(graph: onnx.GraphProto, names: set[str]) -> int:
    patched = 0
    for value in graph.input:
        if value.name not in names:
            continue
        internal = f"{value.name}_fp16"
        rename_value(graph, value.name, internal)
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        graph.node.append(
            onnx.helper.make_node(
                "Cast",
                inputs=[value.name],
                outputs=[internal],
                to=onnx.TensorProto.FLOAT16,
                name=f"musetric_cast_in_{value.name}",
            )
        )
        patched += 1
    return patched


def keep_fp32_outputs(graph: onnx.GraphProto, prefixes: tuple[str, ...]) -> int:
    selected = [value for value in graph.output if value.name.startswith(prefixes)]
    for value in selected:
        internal = f"{value.name}_fp16"
        for node in graph.node:
            for index, name in enumerate(node.output):
                if name == value.name:
                    node.output[index] = internal
        graph.node.append(
            onnx.helper.make_node(
                "Cast",
                inputs=[internal],
                outputs=[value.name],
                to=onnx.TensorProto.FLOAT,
                name=f"musetric_cast_out_{value.name}",
            )
        )
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="fp32 graph to convert")
    parser.add_argument("--output", required=True, help="float16 graph to write")
    parser.add_argument(
        "--fp32-inputs",
        default="encoder_hidden_states",
        help="comma-separated inputs to keep float32 and cast on entry",
    )
    parser.add_argument(
        "--fp32-outputs",
        default="logits,cross_attentions.",
        help="comma-separated output name prefixes to cast back to float32",
    )
    args = parser.parse_args()

    started = time.time()
    model = onnx.load_model(args.input)
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=False,
        disable_shape_infer=model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF,
    )
    graph = converted.graph
    inputs = keep_fp32_inputs(graph, set(filter(None, args.fp32_inputs.split(","))))
    outputs = keep_fp32_outputs(
        graph, tuple(filter(None, args.fp32_outputs.split(",")))
    )
    toposort_graph(graph)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    check_and_save_model(converted, args.output)
    print(
        f"{inputs} float32 input(s), {outputs} float32 output(s), "
        f"{os.path.getsize(args.output)} bytes in {time.time() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
