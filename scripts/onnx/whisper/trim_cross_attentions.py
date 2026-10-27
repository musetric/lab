"""Emit only the cross-attention heads the word times are read from.

This narrows each `cross_attentions.*` output to the heads `alignment_heads`
names in the generation config, inside both branches of the merged decoder, and
rewrites `alignment_heads` to the new positions. Layers no head is taken from
keep one.

    uv run python scripts/onnx/whisper/trim_cross_attentions.py \
      --input  decoder_model_merged_fp16.onnx \
      --output decoder_model_merged_fp16.onnx \
      --generation-config generation_config.json
"""

# ruff: noqa: T201

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_PREFIX = "cross_attentions."
HEAD_AXIS = 1
TRIM_PREFIX = "trim_cross_attentions"


def subgraphs(graph):
    """The graph itself and every graph nested in its nodes' attributes."""
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from subgraphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    yield from subgraphs(nested)


def layer_of(name):
    return int(name[len(OUTPUT_PREFIX) :])


def kept_heads(alignment_heads, layers):
    """Head indices to keep per layer, in the order the output will carry them."""
    kept = {layer: [] for layer in range(layers)}
    for layer, head in alignment_heads:
        if head not in kept[layer]:
            kept[layer].append(head)
    for layer, heads in kept.items():
        kept[layer] = sorted(heads) if heads else [0]
    return kept


def set_head_dim(value, count):
    dim = value.type.tensor_type.shape.dim[HEAD_AXIS]
    dim.ClearField("dim_param")
    dim.dim_value = count


PASS_THROUGH = ("Cast", "Identity")
BRANCHING = ("If", "Loop", "Scan")


def comes_from_branch(name, producers):
    """Whether the value is what a branching node handed the graph."""
    seen = set()
    while name not in seen:
        seen.add(name)
        node = producers.get(name)
        if node is None:
            return False
        if node.op_type in BRANCHING:
            return True
        if node.op_type not in PASS_THROUGH:
            return False
        name = node.input[0]
    return False


def rewrite(graph, index, kept):
    """Gather the kept heads out of each cross-attention output of one graph."""
    outputs = [value for value in graph.output if value.name.startswith(OUTPUT_PREFIX)]
    if not outputs:
        return 0
    producers = {output: node for node in graph.node for output in node.output}
    added = 0
    for value in outputs:
        heads = kept[layer_of(value.name)]
        set_head_dim(value, len(heads))
        producer = producers.get(value.name)
        if producer is None or comes_from_branch(value.name, producers):
            continue
        prefix = f"{TRIM_PREFIX}/{index}/{value.name}"
        source = f"{prefix}/full"
        indices = f"{prefix}/heads"
        producer.output[0] = source
        # The value is on the compute path too: attention reshapes it back for
        # the product with the values, and that reader keeps the whole tensor.
        for node in graph.node:
            for index_ in range(len(node.input)):
                if node.input[index_] == value.name:
                    node.input[index_] = source
        graph.initializer.append(
            numpy_helper.from_array(np.array(heads, dtype=np.int64), indices)
        )
        graph.node.append(
            helper.make_node(
                "Gather",
                [source, indices],
                [value.name],
                name=prefix,
                axis=HEAD_AXIS,
            )
        )
        added += 1
    return added


def apply(model, alignment_heads, layers):
    kept = kept_heads(alignment_heads, layers)
    total = 0
    # Both branches of the merged decoder are narrowed; what the `If` hands the
    # outer graph is then already narrow, so there the shape is only restated.
    for index, graph in enumerate(subgraphs(model.graph)):
        count = rewrite(graph, index, kept)
        if count:
            print(f"graph {index} ({graph.name}): {count} outputs narrowed")
        total += count
    remapped = [[layer, kept[layer].index(head)] for layer, head in alignment_heads]
    return total, kept, remapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    config = json.loads(args.generation_config.read_text(encoding="utf-8"))
    alignment_heads = config["alignment_heads"]
    model = onnx.load(str(args.input))
    total, kept, remapped = apply(model, alignment_heads, args.layers)
    if not total:
        print("no cross-attention output found; nothing to do")
        return
    print(f"kept per layer: {kept}")
    print(f"alignment_heads: {alignment_heads} -> {remapped}")
    config["alignment_heads"] = remapped
    args.generation_config.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
