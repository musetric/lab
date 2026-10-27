# ruff: noqa: T201

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

PAD_PAIR = 2
WITH_BIAS = 3


def constant(graph, known, name, value):
    if name not in known:
        graph.initializer.append(
            numpy_helper.from_array(np.array(value, dtype=np.int64), name)
        )
        known.add(name)
    return name


def attributes(node):
    found = {}
    for attribute in node.attribute:
        found[attribute.name] = list(attribute.ints) if attribute.ints else attribute.i
    return found


def rewritable(node, initializers):
    got = attributes(node)
    kernel = got.get("kernel_shape", [])
    pads = got.get("pads", [0, 0])
    strides = got.get("strides", [1])
    dilations = got.get("dilations", [1])
    if len(kernel) != 1 or got.get("group", 1) != 1:
        return None
    if len(pads) != PAD_PAIR or pads[0] != pads[1] or dilations != [1]:
        return None
    if node.input[1] not in initializers:
        return None
    return kernel[0], pads[0], strides[0]


def rewrite(model, node, shape):
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    known = set(initializers)
    kernel, pad, stride = shape

    weight = numpy_helper.to_array(initializers[node.input[1]])
    out_channels, in_channels, taps = weight.shape
    matrix = weight.transpose(0, 2, 1).reshape(out_channels, taps * in_channels)
    prefix = (node.name or "conv").replace("/", "_").strip("_")
    matrix_name = f"{prefix}_matmul_weight"
    graph.initializer.append(numpy_helper.from_array(matrix, matrix_name))
    known.add(matrix_name)

    nodes = []
    source = node.input[0]
    if pad > 0:
        padded = f"{prefix}_padded"
        pads_name = constant(graph, known, f"{prefix}_pads", [0, 0, pad, 0, 0, pad])
        nodes.append(
            helper.make_node("Pad", [source, pads_name], [padded], name=f"{prefix}_pad")
        )
        source = padded

    axis = constant(graph, known, "conv_mm_axis_time", [2])
    tap_outputs = []
    for tap in range(kernel):
        sliced = f"{prefix}_tap{tap}"
        start = constant(graph, known, f"conv_mm_start_{tap}", [tap])
        tail = kernel - 1 - tap
        end = constant(
            graph,
            known,
            f"conv_mm_end_{tail}",
            [-tail if tail else np.iinfo(np.int64).max],
        )
        step = constant(graph, known, f"conv_mm_step_{stride}", [stride])
        nodes.append(
            helper.make_node(
                "Slice",
                [source, start, end, axis, step],
                [sliced],
                name=f"{prefix}_slice{tap}",
            )
        )
        tap_outputs.append(sliced)

    stacked = f"{prefix}_taps"
    nodes.append(
        helper.make_node(
            "Concat", tap_outputs, [stacked], name=f"{prefix}_concat", axis=1
        )
    )

    rows = f"{prefix}_rows"
    rows_shape = constant(
        graph, known, f"conv_mm_rows_{taps * in_channels}", [taps * in_channels, -1]
    )
    nodes.append(
        helper.make_node(
            "Reshape", [stacked, rows_shape], [rows], name=f"{prefix}_drop_batch"
        )
    )
    product = f"{prefix}_product"
    nodes.append(
        helper.make_node(
            "MatMul", [matrix_name, rows], [product], name=f"{prefix}_matmul"
        )
    )

    biased = product
    if len(node.input) >= WITH_BIAS:
        bias = numpy_helper.to_array(initializers[node.input[2]])
        column_name = f"{prefix}_bias_column"
        graph.initializer.append(
            numpy_helper.from_array(bias.reshape(-1, 1), column_name)
        )
        known.add(column_name)
        biased = f"{prefix}_biased"
        nodes.append(
            helper.make_node(
                "Add", [product, column_name], [biased], name=f"{prefix}_bias"
            )
        )

    out_shape = constant(
        graph, known, f"conv_mm_out_{out_channels}", [1, out_channels, -1]
    )
    nodes.append(
        helper.make_node(
            "Reshape",
            [biased, out_shape],
            [node.output[0]],
            name=f"{prefix}_restore_batch",
        )
    )
    return nodes


def prune(graph):
    used = {out.name for out in graph.output}
    for node in reversed(graph.node):
        if any(name in used for name in node.output):
            used.update(node.input)
    kept = [init for init in graph.initializer if init.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    for value in [v for v in graph.value_info if v.name not in used]:
        graph.value_info.remove(value)


def apply(model):
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}

    targets = []
    for position, node in enumerate(graph.node):
        if node.op_type != "Conv":
            continue
        shape = rewritable(node, initializers)
        if shape is None:
            raise SystemExit(f"{node.name}: not a rewritable 1-D convolution")
        targets.append((position, node, shape))

    if not targets:
        print("no Conv nodes in the graph; nothing to do")
        return

    replacements = {}
    for position, node, shape in targets:
        replacements[position] = rewrite(model, node, shape)
        print(
            f"rewrote {node.name}: kernel {shape[0]} pad {shape[1]} stride {shape[2]}"
        )

    rebuilt = []
    for position, node in enumerate(graph.node):
        rebuilt.extend(replacements.get(position, [node]))
    before = len(graph.node)
    del graph.node[:]
    graph.node.extend(rebuilt)
    prune(graph)
    print(f"nodes: {before} -> {len(graph.node)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    apply(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
