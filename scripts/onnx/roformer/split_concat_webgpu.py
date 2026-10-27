# ruff: noqa: T201 -- CLI build tool: stdout (progress/results) is its interface.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

GROUP = 8

parser = argparse.ArgumentParser(description="Patch wide Concat/Split nodes.")
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)

model = onnx.load(str(args.input), load_external_data=True)
graph = model.graph

for t in graph.initializer:
    if t.data_location == onnx.TensorProto.EXTERNAL:
        t.data_location = onnx.TensorProto.DEFAULT
        del t.external_data[:]

producer = {o: n for n in graph.node for o in n.output}
consumers: dict[str, list] = {}
for n in graph.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n)

removed_ids: set[int] = set()
removed_names: set[str] = set()
n_softmax = 0
for sm in [n for n in graph.node if n.op_type == "Softmax"]:
    c1 = producer.get(sm.input[0])
    c2s = consumers.get(sm.output[0], [])
    if (
        c1 is None
        or c1.op_type != "Cast"
        or len(consumers.get(c1.output[0], [])) != 1
        or len(c2s) != 1
        or c2s[0].op_type != "Cast"
    ):
        continue
    removed_ids.update({id(c1), id(c2s[0])})
    removed_names.update({c1.output[0], sm.output[0]})
    sm.input[0] = c1.input[0]
    sm.output[0] = c2s[0].output[0]
    n_softmax += 1

kept = [n for n in graph.node if id(n) not in removed_ids]
del graph.node[:]
graph.node.extend(kept)
stale_vi = [vi for vi in graph.value_info if vi.name in removed_names]
for vi in stale_vi:
    graph.value_info.remove(vi)
print(f"unwrapped {n_softmax} Softmax to fp16 (removed {len(removed_ids)} casts)")

inits = {t.name: t for t in graph.initializer}
static_dims = {
    vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
    for vi in list(graph.value_info) + list(graph.input)
}

new_nodes = []
new_inits = []
n_concat = 0
n_split = 0


def get_axis(node) -> int:
    return next((a.i for a in node.attribute if a.name == "axis"), 0)


def chunks(seq):
    for start in range(0, len(seq), GROUP):
        yield seq[start : start + GROUP]


def split_sizes(node) -> np.ndarray:
    if len(node.input) > 1 and node.input[1]:
        return numpy_helper.to_array(inits[node.input[1]]).astype(np.int64)
    dims = static_dims[node.input[0]]
    axis = get_axis(node)
    dim = dims[axis if axis >= 0 else axis + len(dims)]
    if dim <= 0:
        raise RuntimeError(
            f"{node.name}: Split input has no static size on axis {axis}"
        )
    count = len(node.output)
    step = -(-dim // count)
    return np.array([step] * (count - 1) + [dim - step * (count - 1)], dtype=np.int64)


for node in graph.node:
    if node.op_type == "Concat" and len(node.input) > GROUP:
        n_concat += 1
        axis = get_axis(node)
        base = node.name or f"concat_{n_concat}"
        group_outs = []
        for gi, chunk in enumerate(chunks(list(node.input))):
            if len(chunk) == 1:
                group_outs.append(chunk[0])
                continue
            out = f"{base}__grp{gi}"
            new_nodes.append(
                helper.make_node(
                    "Concat", chunk, [out], name=f"{base}__grp{gi}_node", axis=axis
                )
            )
            group_outs.append(out)
        new_nodes.append(
            helper.make_node(
                "Concat", group_outs, list(node.output), name=base, axis=axis
            )
        )
        print(f"concat {base}: {len(node.input)} inputs -> {len(group_outs)} groups")

    elif node.op_type == "Split" and len(node.output) > GROUP:
        n_split += 1
        axis = get_axis(node)
        base = node.name or f"split_{n_split}"
        data_in = node.input[0]
        sizes = split_sizes(node)
        outs = list(node.output)

        out_groups = list(chunks(outs))
        size_groups = list(chunks(sizes.tolist()))

        group_names = []
        for gi, og in enumerate(out_groups):
            group_names.append(og[0] if len(og) == 1 else f"{base}__grp{gi}")
        l1_sizes = f"{base}__l1_sizes"
        new_inits.append(
            numpy_helper.from_array(
                np.array([sum(sg) for sg in size_groups], dtype=np.int64), l1_sizes
            )
        )
        new_nodes.append(
            helper.make_node(
                "Split", [data_in, l1_sizes], group_names, name=f"{base}__l1", axis=axis
            )
        )
        for gi, (og, sg) in enumerate(zip(out_groups, size_groups, strict=False)):
            if len(og) == 1:
                continue
            l2_sizes = f"{base}__grp{gi}_sizes"
            new_inits.append(
                numpy_helper.from_array(np.array(sg, dtype=np.int64), l2_sizes)
            )
            new_nodes.append(
                helper.make_node(
                    "Split",
                    [group_names[gi], l2_sizes],
                    og,
                    name=f"{base}__grp{gi}",
                    axis=axis,
                )
            )
        print(f"split {base}: {len(outs)} outputs -> {len(out_groups)} groups")

    else:
        new_nodes.append(node)

del graph.node[:]
graph.node.extend(new_nodes)
graph.initializer.extend(new_inits)

print(f"rewrote {n_concat} Concat + {n_split} Split node(s)")
onnx.checker.check_model(model, full_check=False)

args.output.unlink(missing_ok=True)
Path(str(args.output) + ".data").unlink(missing_ok=True)

onnx.save_model(
    model,
    str(args.output),
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=args.output.name + ".data",
    size_threshold=4096,
)
print(f"saved {args.output}")
