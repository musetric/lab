# ruff: noqa: T201

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

BINDING_LIMIT = 128 * 1024 * 1024
QUERY_BLOCK = 250
ELEMENT_BYTES = {TensorProto.FLOAT: 4, TensorProto.FLOAT16: 2}
SCORE_RANK = 3


class Attention:

    def __init__(self, score, softmax, reshape, context, shape):
        self.score = score
        self.softmax = softmax
        self.reshape = reshape
        self.context = context
        self.shape = shape

    @property
    def queries(self):
        return static_shape(self.shape)[1]

    def score_bytes(self, queries):
        dims = static_shape(self.shape)
        heads = dims[0]
        if not isinstance(heads, int):
            heads = int(str(heads).split("*")[0])
        element = ELEMENT_BYTES.get(self.shape.type.tensor_type.elem_type, 4)
        return heads * queries * dims[2] * element


def static_shape(value_info):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param)
    return dims


def scores_of(softmax, shapes, producers):
    score = producers.get(softmax.input[0])
    if score is None or score.op_type != "MatMul":
        return None, None
    shape = shapes.get(score.output[0])
    if shape is None or len(static_shape(shape)) != SCORE_RANK:
        return None, None
    if not isinstance(static_shape(shape)[1], int):
        return None, None
    return score, shape


def context_of(softmax, shape, shapes, single_consumer):
    after = single_consumer(softmax.output[0])
    reshape = None
    if after is not None and after.op_type == "Reshape":
        passthrough = shapes.get(after.output[0])
        if passthrough is None or static_shape(passthrough) != static_shape(shape):
            return None, None
        reshape, after = after, single_consumer(after.output[0])
    if after is None or after.op_type != "MatMul":
        return None, None
    return reshape, after


def find_attentions(model):
    inferred = SymbolicShapeInference.infer_shapes(
        model, auto_merge=True, guess_output_rank=True
    )
    shapes = {vi.name: vi for vi in inferred.graph.value_info}
    producers = {out: node for node in model.graph.node for out in node.output}
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def single_consumer(name):
        found = consumers.get(name, [])
        return found[0] if len(found) == 1 else None

    found = []
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax":
            continue
        score, shape = scores_of(softmax, shapes, producers)
        if score is None:
            continue
        reshape, context = context_of(softmax, shape, shapes, single_consumer)
        if context is None:
            continue
        found.append(Attention(score, softmax, reshape, context, shape))
    return found


def rewrite(model, attentions, block):
    graph = model.graph
    known = {init.name for init in graph.initializer}

    def constant(name, value):
        if name not in known:
            graph.initializer.append(
                numpy_helper.from_array(np.array(value, dtype=np.int64), name)
            )
            known.add(name)
        return name

    axis = constant("blocked_attn_axis", [1])
    replacements = {}
    for index, attention in enumerate(attentions):
        queries = attention.queries
        prefix = f"blocked_attn_{index}"
        query, keys = attention.score.input
        values = attention.context.input[1]
        nodes, parts = [], []
        for part, start in enumerate(range(0, queries, block)):
            end = min(start + block, queries)
            tag = f"{prefix}_{part}"
            sliced, scored, weighted, context = (
                f"{tag}_query",
                f"{tag}_score",
                f"{tag}_weight",
                f"{tag}_context",
            )
            nodes.append(
                helper.make_node(
                    "Slice",
                    [
                        query,
                        constant(f"blocked_attn_start_{start}", [start]),
                        constant(f"blocked_attn_end_{end}", [end]),
                        axis,
                    ],
                    [sliced],
                    name=f"{tag}_slice",
                )
            )
            nodes.append(
                helper.make_node(
                    "MatMul", [sliced, keys], [scored], name=f"{tag}_score"
                )
            )
            softmax = helper.make_node(
                "Softmax", [scored], [weighted], name=f"{tag}_softmax"
            )
            softmax.attribute.extend(attention.softmax.attribute)
            nodes.append(softmax)
            nodes.append(
                helper.make_node(
                    "MatMul", [weighted, values], [context], name=f"{tag}_context"
                )
            )
            parts.append(context)
        nodes.append(
            helper.make_node(
                "Concat",
                parts,
                [attention.context.output[0]],
                name=f"{prefix}_concat",
                axis=1,
            )
        )
        replacements[id(attention.context)] = nodes

    dropped = set()
    for attention in attentions:
        for node in (attention.score, attention.softmax, attention.reshape):
            if node is not None:
                dropped.add(id(node))
    rebuilt = []
    for node in graph.node:
        if id(node) in dropped:
            continue
        rebuilt.extend(replacements.get(id(node), [node]))
    del graph.node[:]
    graph.node.extend(rebuilt)
    prune(graph)


def prune(graph):
    used = {out.name for out in graph.output}
    while True:
        live = []
        reachable = set(used)
        for node in reversed(graph.node):
            if any(out in reachable for out in node.output):
                live.append(node)
                reachable.update(node.input)
        live.reverse()
        if len(live) == len(graph.node):
            used = reachable
            break
        del graph.node[:]
        graph.node.extend(live)
    kept = [init for init in graph.initializer if init.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    for vi in [vi for vi in graph.value_info if vi.name not in used]:
        graph.value_info.remove(vi)


def apply(model, query_block=QUERY_BLOCK):
    found = find_attentions(model)
    if not found:
        raise SystemExit("no MatMul/Softmax/MatMul attention chain in the graph")
    attentions = [a for a in found if a.queries > query_block]
    if not attentions:
        print(
            f"all {len(found)} attention chains already fit in {query_block} "
            f"query rows; nothing to do"
        )
        return

    before = max(a.score_bytes(a.queries) for a in attentions)
    after = max(a.score_bytes(query_block) for a in attentions)
    print(f"attention chains to block: {len(attentions)} of {len(found)}")
    print(
        f"largest score tensor: {before / 1048576:.1f} MiB -> {after / 1048576:.1f} MiB"
    )
    if after > BINDING_LIMIT:
        raise SystemExit(
            f"a {query_block}-row block still needs {after / 1048576:.1f} MiB, "
            f"over the {BINDING_LIMIT / 1048576:.0f} MiB WebGPU binding floor"
        )

    nodes = len(model.graph.node)
    rewrite(model, attentions, query_block)
    print(f"nodes: {nodes} -> {len(model.graph.node)}")


def largest_score_bytes(model):
    found = find_attentions(model)
    if not found:
        return 0
    return max(a.score_bytes(a.queries) for a in found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--query-block",
        type=int,
        default=QUERY_BLOCK,
        help="query rows per block; 1500/250 splits each score tensor into six",
    )
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    apply(model, args.query_block)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
