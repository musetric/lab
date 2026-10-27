# ruff: noqa: T201

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

PREFIX = "scale_residual_stream"
FLOAT16_MAX = 65504.0
HEADROOM = 8.0
EMBED_GATHER = "/model/decoder/embed_tokens/Gather"
POSITIONS = "model.decoder.embed_positions.weight"
LAYER_NORM_EPS = "layer_norm/Add"
LAYER_NORM_OPSET = 17
SCALED_WEIGHTS = ("/self_attn/v_proj/MatMul", "/fc2/MatMul")
SCALED_BIASES = (
    "/self_attn/v_proj/Add",
    "/self_attn/out_proj/Add",
    "/encoder_attn/out_proj/Add",
    "/fc2/Add",
)
CROSS_SOFTMAX = "/encoder_attn/Softmax"
CROSS_OUTPUT_CAST = "musetric_cast_out_cross_attentions."


def scale_initializer(graph, initializers, name, factor):
    array = numpy_helper.to_array(initializers[name]).astype(np.float64) * factor
    if np.abs(array).max() * HEADROOM > FLOAT16_MAX:
        raise ValueError(
            f"{name}: {np.abs(array).max():.4g} leaves no float16 headroom"
        )
    scaled = numpy_helper.from_array(array.astype(np.float16), name)
    initializers[name].CopyFrom(scaled)


def initializer_input(node, initializers):
    names = [name for name in node.input if name in initializers]
    if len(names) != 1:
        raise ValueError(f"{node.name}: expected one initializer input, found {names}")
    return names[0]


def rewrite_cross_softmax(graph, node, scale_log2, index):
    consumers = [other for other in graph.node if node.output[0] in other.input]
    if len(consumers) != 1 or consumers[0].op_type != "Reshape":
        found = [c.op_type for c in consumers]
        raise ValueError(f"{node.name}: expected one Reshape reader, found {found}")
    reshape = consumers[0]
    readers = [other for other in graph.node if reshape.output[0] in other.input]
    gather = [other for other in readers if other.op_type == "Gather"]
    if len(gather) != 1:
        raise ValueError(f"{reshape.name}: expected one Gather reader of the heads")
    gather = gather[0]
    cast_out = [other for other in graph.node if gather.output[0] in other.input]
    if len(cast_out) != 1 or cast_out[0].op_type != "Cast":
        raise ValueError(f"{gather.name}: expected the output Cast to read it")
    cast_out = cast_out[0]
    prefix = f"{PREFIX}/{index}"
    scores32 = f"{prefix}/scores"
    probabilities32 = f"{prefix}/probabilities"
    full32 = f"{prefix}/full"
    scaled32 = f"{prefix}/scaled"
    axis = next(
        (helper.get_attribute_value(a) for a in node.attribute if a.name == "axis"), -1
    )
    new_nodes = [
        helper.make_node(
            "Cast",
            [node.input[0]],
            [scores32],
            name=f"{prefix}/cast_scores",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Softmax",
            [scores32],
            [probabilities32],
            name=f"{prefix}/softmax",
            axis=axis,
        ),
        helper.make_node(
            "Mul",
            [probabilities32, f"{PREFIX}/probability_scale"],
            [scaled32],
            name=f"{prefix}/scale",
        ),
        helper.make_node(
            "Cast",
            [scaled32],
            [node.output[0]],
            name=f"{prefix}/cast_probabilities",
            to=TensorProto.FLOAT16,
        ),
    ]
    gather.input[0] = full32
    gather.output[0] = cast_out.output[0]
    graph.node.remove(cast_out)
    position = list(graph.node).index(node)
    graph.node.remove(node)
    for offset, new in enumerate(new_nodes):
        graph.node.insert(position + offset, new)
    graph.node.insert(
        list(graph.node).index(reshape),
        helper.make_node(
            "Reshape",
            [probabilities32, reshape.input[1]],
            [full32],
            name=f"{prefix}/reshape",
        ),
    )


def replace_layer_norm(graph, initializers, eps_add, eps_factor):
    producers = {output: node for node in graph.node for output in node.output}
    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def only(nodes, op):
        return nodes[0] if len(nodes) == 1 and nodes[0].op_type == op else None

    var_mean = producers.get(eps_add.input[0])
    power = (
        producers.get(var_mean.input[0])
        if var_mean is not None and var_mean.op_type == "ReduceMean"
        else None
    )
    centered = (
        producers.get(power.input[0])
        if power is not None and power.op_type == "Pow"
        else None
    )
    mean = (
        producers.get(centered.input[1])
        if centered is not None and centered.op_type == "Sub"
        else None
    )
    sqrt = only(consumers.get(eps_add.output[0], []), "Sqrt")
    div = only(consumers.get(sqrt.output[0], []), "Div") if sqrt else None
    mul = only(consumers.get(div.output[0], []), "Mul") if div else None
    add = only(consumers.get(mul.output[0], []), "Add") if mul else None
    if (
        None in (var_mean, power, centered, mean, sqrt, div, mul, add)
        or mean.op_type != "ReduceMean"
        or mean.input[0] != centered.input[0]
        or div.input[0] != centered.output[0]
    ):
        raise ValueError(f"{eps_add.name}: not the layer norm pattern")
    epsilon = (
        float(
            numpy_helper.to_array(
                initializers[initializer_input(eps_add, initializers)]
            )
        )
        * eps_factor
    )
    gamma = initializer_input(mul, initializers)
    beta = initializer_input(add, initializers)
    replaced = [mean, centered, power, var_mean, eps_add, sqrt, div, mul, add]
    position = list(graph.node).index(mean)
    for node in replaced:
        graph.node.remove(node)
    graph.node.insert(
        position,
        helper.make_node(
            "LayerNormalization",
            [centered.input[0], gamma, beta],
            [add.output[0]],
            name=f"{PREFIX}/{add.name}",
            axis=-1,
            epsilon=epsilon,
            stash_type=1,
        ),
    )
    for name in (
        initializer_input(eps_add, initializers),
        initializer_input(power, initializers),
    ):
        if not any(name in node.input for node in graph.node):
            graph.initializer.remove(initializers.pop(name))


def scale_embedding(graph, gather):
    scaled = f"{PREFIX}/embedding"
    for other in graph.node:
        for index in range(len(other.input)):
            if other.input[index] == gather.output[0]:
                other.input[index] = scaled
    graph.node.insert(
        list(graph.node).index(gather) + 1,
        helper.make_node(
            "Mul",
            [gather.output[0], f"{PREFIX}/stream_scale"],
            [scaled],
            name=f"{PREFIX}/embedding",
        ),
    )


def apply(model, scale_log2):
    graph = model.graph
    factor = 2.0**scale_log2
    initializers = {init.name: init for init in graph.initializer}
    counts = {
        "positions": 0,
        "embedding": 0,
        "epsilon": 0,
        "weights": 0,
        "biases": 0,
        "softmax": 0,
    }
    for name in list(initializers):
        if name.startswith(POSITIONS):
            scale_initializer(graph, initializers, name, factor)
            counts["positions"] += 1
    for node in list(graph.node):
        if node.name == EMBED_GATHER:
            scale_embedding(graph, node)
            counts["embedding"] += 1
        elif node.op_type == "Add" and node.name.endswith(LAYER_NORM_EPS):
            replace_layer_norm(graph, initializers, node, factor * factor)
            counts["epsilon"] += 1
        elif node.op_type == "MatMul" and node.name.endswith(SCALED_WEIGHTS):
            scale_initializer(
                graph, initializers, initializer_input(node, initializers), factor
            )
            counts["weights"] += 1
        elif node.op_type == "Add" and node.name.endswith(SCALED_BIASES):
            scale_initializer(
                graph, initializers, initializer_input(node, initializers), factor
            )
            counts["biases"] += 1
    for node in [
        n
        for n in graph.node
        if n.op_type == "Softmax" and n.name.endswith(CROSS_SOFTMAX)
    ]:
        rewrite_cross_softmax(graph, node, scale_log2, counts["softmax"])
        counts["softmax"] += 1
    graph.initializer.append(
        numpy_helper.from_array(
            np.array(factor, dtype=np.float16), f"{PREFIX}/stream_scale"
        )
    )
    graph.initializer.append(
        numpy_helper.from_array(
            np.array(factor, dtype=np.float32), f"{PREFIX}/probability_scale"
        )
    )
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-log2", type=int, default=10)
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    if any(node.op_type == "If" for node in model.graph.node):
        raise SystemExit("the pass takes the flat step decoder, not a merged one")
    for opset in model.opset_import:
        if opset.domain == "" and opset.version < LAYER_NORM_OPSET:
            opset.version = LAYER_NORM_OPSET
    counts = apply(model, args.scale_log2)
    expected = {
        "positions": 1,
        "embedding": 1,
        "epsilon": 13,
        "weights": 8,
        "biases": 16,
        "softmax": 4,
    }
    if counts != expected:
        raise SystemExit(f"unexpected graph: rewrote {counts}, expected {expected}")
    print(f"scaled by 2^{args.scale_log2}: {counts}")
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
