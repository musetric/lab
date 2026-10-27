# ruff: noqa: T201

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

SPLIT_INPUT_COUNT = 2


def node_attribute(
    node: onnx.NodeProto,
    name: str,
) -> onnx.AttributeProto | None:
    return next(
        (attribute for attribute in node.attribute if attribute.name == name),
        None,
    )


def constant_values(producers: dict[str, onnx.NodeProto], name: str) -> list[int]:
    node = producers.get(name)
    if node is None or node.op_type != "Constant":
        raise ValueError(f"Split sizes must be a Constant node: {name}")
    attribute = node_attribute(node, "value")
    if attribute is None:
        raise ValueError(f"Constant node has no tensor value: {node.name}")
    return numpy_helper.to_array(attribute.t).astype(np.int64).reshape(-1).tolist()


def split_axis(node: onnx.NodeProto) -> int:
    attribute = node_attribute(node, "axis")
    return attribute.i if attribute is not None else 0


def unused_split_size_constants(
    graph: onnx.GraphProto,
    producers: dict[str, onnx.NodeProto],
) -> set[int]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    removed_node_ids: set[int] = set()
    for node in graph.node:
        if node.op_type != "Split" or len(node.input) != SPLIT_INPUT_COUNT:
            continue
        size_source = producers.get(node.input[1])
        if (
            size_source is not None
            and size_source.op_type == "Constant"
            and consumers.get(node.input[1]) == [node]
        ):
            removed_node_ids.add(id(size_source))
    return removed_node_ids


def make_slices(
    node: onnx.NodeProto,
    sizes: list[int],
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    axis = split_axis(node)
    offset = 0
    slices: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    for index, (size, output) in enumerate(zip(sizes, node.output, strict=True)):
        prefix = f"{node.name}__slice_{index}"
        starts = f"{prefix}_starts"
        ends = f"{prefix}_ends"
        axes = f"{prefix}_axes"
        initializers.extend(
            [
                numpy_helper.from_array(np.array([offset], dtype=np.int64), starts),
                numpy_helper.from_array(
                    np.array([offset + size], dtype=np.int64), ends
                ),
                numpy_helper.from_array(np.array([axis], dtype=np.int64), axes),
            ]
        )
        slices.append(
            helper.make_node(
                "Slice",
                [node.input[0], starts, ends, axes],
                [output],
                name=prefix,
            )
        )
        offset += size
    return slices, initializers


def replace_splits(model: onnx.ModelProto) -> int:
    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    removed_node_ids = unused_split_size_constants(graph, producers)
    new_nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    count = 0

    for node in graph.node:
        if id(node) in removed_node_ids:
            continue
        if node.op_type != "Split":
            new_nodes.append(node)
            continue
        if len(node.input) != SPLIT_INPUT_COUNT:
            raise ValueError(f"Split must have data and fixed sizes: {node.name}")
        sizes = constant_values(producers, node.input[1])
        if len(sizes) != len(node.output):
            raise ValueError(f"Split size/output mismatch: {node.name}")
        slices, slice_initializers = make_slices(node, sizes)
        new_nodes.extend(slices)
        initializers.extend(slice_initializers)
        count += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(initializers)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input)
    count = replace_splits(model)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.output)
    print(f"replaced {count} Split node(s)")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
