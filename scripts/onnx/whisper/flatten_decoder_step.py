"""Split the merged decoder into a flat cached step and a cross-attention projection.

The merged decoder holds two copies of the network behind an `If`: the first step
(no cache, the prompt, cross-attention keys and values computed from the encoder)
and every later step (one token, everything cached).

This writes two graphs:

* the decoder with the `If` replaced by the cached-step branch, the step graph.
  `use_cache_branch` is gone; `encoder_hidden_states` stays as an input nothing
  reads, so the pipeline keeps handing it over for the first step;
* `cross_kv`: the encoder states' key and value projections of every layer, the
  part of the first branch the step graph does not have.

The first step is then the cross projection plus the prompt fed through the step
graph one token at a time, which applies exactly the causal mask the first branch
builds for the whole prompt.

    uv run python scripts/onnx/whisper/flatten_decoder_step.py \
      --input  decoder_model_merged_fp16.onnx \
      --step   decoder_model_merged_fp16.onnx \
      --cross  cross_kv_fp16.onnx
"""

# ruff: noqa: T201

import argparse
import contextlib
import re
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

CACHED = "with_past"
FIRST = "no_past"
HEADS = 20
HEAD_SIZE = 64
BRANCH_SELECTOR = "use_cache_branch"
ENCODER_STATES = "encoder_hidden_states"
PROJECTION = re.compile(r"/model/decoder/layers\.(\d+)/encoder_attn/([kv])_proj/")


def branch_of(node, name):
    return next(a.g for a in node.attribute if a.g.name == name)


def prune(graph):
    """Drop nodes and initializers nothing reads any more."""
    while True:
        read = {name for node in graph.node for name in node.input}
        read |= {value.name for value in graph.output}
        alive = [node for node in graph.node if any(out in read for out in node.output)]
        if len(alive) == len(graph.node):
            break
        del graph.node[:]
        graph.node.extend(alive)
    read = {name for node in graph.node for name in node.input}
    initializers = [init for init in graph.initializer if init.name in read]
    del graph.initializer[:]
    graph.initializer.extend(initializers)


def flatten(model):
    """Replace the `If` with its cached branch, wired to the `If` outputs."""
    graph = model.graph
    branching = next(node for node in graph.node if node.op_type == "If")
    cached = branch_of(branching, CACHED)
    renamed = dict(
        zip((value.name for value in cached.output), branching.output, strict=True)
    )
    produced = {out for node in cached.node for out in node.output}
    nodes = []
    for node in cached.node:
        node.output[:] = [renamed.get(out, out) for out in node.output]
        node.input[:] = [renamed.get(name, name) for name in node.input]
        nodes.append(node)
    for inner, outer in renamed.items():
        if inner not in produced:
            nodes.append(
                helper.make_node("Identity", [inner], [outer], name=f"flatten/{outer}")
            )
    position = list(graph.node).index(branching)
    rebuilt = list(graph.node)[:position] + nodes + list(graph.node)[position + 1 :]
    del graph.node[:]
    graph.node.extend(rebuilt)
    graph.initializer.extend(cached.initializer)
    inputs = [value for value in graph.input if value.name != BRANCH_SELECTOR]
    del graph.input[:]
    graph.input.extend(inputs)
    prune(graph)
    return len(nodes)


def cross_projection(model):
    """The encoder states' key/value projections, as a graph of their own."""
    graph = model.graph
    branching = next(node for node in graph.node if node.op_type == "If")
    first = branch_of(branching, FIRST)
    weights = {
        init.name: init for init in list(graph.initializer) + list(first.initializer)
    }
    layers = {}
    for node in first.node:
        match = PROJECTION.match(node.name)
        if not match:
            continue
        layer, kind = int(match.group(1)), match.group(2)
        entry = layers.setdefault(layer, {})
        for name in node.input:
            if name in weights:
                entry[f"{kind}_{node.op_type}"] = weights[name]
    nodes = [
        helper.make_node(
            "Cast",
            [ENCODER_STATES],
            ["states"],
            name="states",
            to=onnx.TensorProto.FLOAT16,
        )
    ]
    initializers = [
        numpy_helper.from_array(
            np.array([0, -1, HEADS, HEAD_SIZE], dtype=np.int64), "heads_shape"
        )
    ]
    outputs = []
    for layer in sorted(layers):
        for kind in ("k", "v"):
            weight = layers[layer][f"{kind}_MatMul"]
            product = f"layer{layer}/{kind}/product"
            nodes.append(
                helper.make_node(
                    "MatMul", ["states", weight.name], [product], name=product
                )
            )
            initializers.append(weight)
            projected = product
            bias = layers[layer].get(f"{kind}_Add")
            if bias is not None:
                projected = f"layer{layer}/{kind}/biased"
                nodes.append(
                    helper.make_node(
                        "Add", [bias.name, product], [projected], name=projected
                    )
                )
                initializers.append(bias)
            split = f"layer{layer}/{kind}/heads"
            nodes.append(
                helper.make_node(
                    "Reshape", [projected, "heads_shape"], [split], name=split
                )
            )
            name = f"present.{layer}.encoder.{'key' if kind == 'k' else 'value'}"
            nodes.append(
                helper.make_node(
                    "Transpose", [split], [name], name=name, perm=[0, 2, 1, 3]
                )
            )
            outputs.append(
                helper.make_tensor_value_info(
                    name,
                    onnx.TensorProto.FLOAT16,
                    ["batch_size", HEADS, "encoder_positions", HEAD_SIZE],
                )
            )
    states = next(value for value in graph.input if value.name == ENCODER_STATES)
    cross = helper.make_graph(nodes, "cross_kv", [states], outputs, initializers)
    return helper.make_model(
        cross, opset_imports=model.opset_import, ir_version=model.ir_version
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--cross", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    cross = cross_projection(model)
    onnx.save(cross, str(args.cross))
    size = args.cross.stat().st_size
    print(f"wrote {args.cross} ({size} bytes), {len(cross.graph.output)} outputs")
    inlined = flatten(model)
    args.step.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.step))
    size = args.step.stat().st_size
    total = len(model.graph.node)
    print(f"wrote {args.step} ({size} bytes), {inlined} inlined, {total} in total")


if __name__ == "__main__":
    main()
