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

KEY_TRANSPOSE = [0, 2, 1]
SCORE_RANK = 3
PAD_PREFIX = "pad_attention_scores"


def subgraphs(graph):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from subgraphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    yield from subgraphs(nested)


def attention_scores(graph):
    producers = {output: node for node in graph.node for output in node.output}
    found = []
    for node in graph.node:
        if node.op_type != "MatMul" or len(node.input) != 2:  # noqa: PLR2004
            continue
        producer = producers.get(node.input[1])
        if producer is None or producer.op_type != "Transpose":
            continue
        perm = next(
            (list(a.ints) for a in producer.attribute if a.name == "perm"), None
        )
        if perm == KEY_TRANSPOSE:
            found.append(node)
    return found


def count_affected(model):
    return sum(len(attention_scores(graph)) for graph in subgraphs(model.graph))


def rewrite(graph, index):
    targets = attention_scores(graph)
    if not targets:
        return 0
    prefix = f"{PAD_PREFIX}/{index}"
    query_pads = np.zeros(2 * SCORE_RANK, dtype=np.int64)
    query_pads[-1] = 1
    key_pads = np.zeros(2 * SCORE_RANK, dtype=np.int64)
    key_pads[-2] = 1
    for name, value in (("query_pads", query_pads), ("key_pads", key_pads)):
        graph.initializer.append(numpy_helper.from_array(value, f"{prefix}/{name}"))

    rewritten = []
    for node in graph.node:
        if node not in targets:
            rewritten.append(node)
            continue
        base = f"{prefix}/{node.name or node.output[0]}"
        query = f"{base}/query"
        keys = f"{base}/keys"
        rewritten.extend(
            [
                helper.make_node(
                    "Pad",
                    [node.input[0], f"{prefix}/query_pads"],
                    [query],
                    name=query,
                    mode="constant",
                ),
                helper.make_node(
                    "Pad",
                    [node.input[1], f"{prefix}/key_pads"],
                    [keys],
                    name=keys,
                    mode="constant",
                ),
                helper.make_node(
                    "MatMul", [query, keys], [node.output[0]], name=node.name
                ),
            ]
        )
    del graph.node[:]
    graph.node.extend(rewritten)
    return len(targets)


def apply(model):
    total = 0
    for index, graph in enumerate(list(subgraphs(model.graph))):
        count = rewrite(graph, index)
        if count:
            print(f"graph {index} ({graph.name}): {count} attention scores widened")
        total += count
    if not total:
        print("no attention score reaches the packed kernel; nothing to do")
    return total


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
