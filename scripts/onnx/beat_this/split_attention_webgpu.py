# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ATTENTION_RANK = 4
FLOAT_BYTES = 4
MIB = 2**20


@dataclass
class Core:
    scores: onnx.NodeProto
    softmax: onnx.NodeProto
    weighted: onnx.NodeProto
    shape: list[int]


@dataclass
class SplitContext:
    initializers: list[onnx.TensorProto] = field(default_factory=list)
    names: dict[tuple[int, int, int], tuple[str, str, str]] = field(
        default_factory=dict
    )

    def slice_inputs(self, start: int, end: int, axis: int) -> tuple[str, str, str]:
        key = (start, end, axis)
        if key not in self.names:
            prefix = f"attention_split/{axis}_{start}_{end}"
            values = {"starts": start, "ends": end, "axes": axis}
            for suffix, value in values.items():
                self.initializers.append(
                    numpy_helper.from_array(
                        np.array([value], dtype=np.int64), f"{prefix}/{suffix}"
                    )
                )
            self.names[key] = (f"{prefix}/starts", f"{prefix}/ends", f"{prefix}/axes")
        return self.names[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="static rewritten .onnx")
    parser.add_argument(
        "--shapes", required=True, help="shapes .json of that frame count"
    )
    parser.add_argument("--out", required=True, help="output .onnx")
    parser.add_argument(
        "--budget-mib",
        type=float,
        default=72,
        help="largest scores tensor one head group may hold",
    )
    return parser.parse_args()


def softmax_axis(node: onnx.NodeProto) -> int:
    attribute = next((item for item in node.attribute if item.name == "axis"), None)
    return attribute.i if attribute is not None else -1


def find_cores(
    graph: onnx.GraphProto, shapes: dict[str, list[int]], budget: float
) -> list[Core]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    cores = []
    for node in graph.node:
        shape = shapes.get(node.output[0])
        if node.op_type != "MatMul" or not shape or len(shape) != ATTENTION_RANK:
            continue
        if shape[-1] != shape[-2] or math.prod(shape) * FLOAT_BYTES <= budget:
            continue
        (softmax,) = consumers[node.output[0]]
        (weighted,) = consumers[softmax.output[0]]
        if softmax.op_type != "Softmax" or softmax_axis(softmax) not in {-1, 3}:
            raise ValueError(f"{node.name}: scores do not feed a last-axis Softmax")
        if weighted.op_type != "MatMul" or weighted.input[0] != softmax.output[0]:
            raise ValueError(f"{node.name}: Softmax does not feed a MatMul")
        cores.append(Core(node, softmax, weighted, shape))
    return cores


def split_core(
    core: Core, shapes: dict[str, list[int]], budget: float, context: SplitContext
) -> list[onnx.NodeProto]:
    batch, heads, frames, _ = core.shape
    axis = 0 if batch > 1 else 1
    size = core.shape[axis]
    slice_bytes = math.prod(core.shape) // size * FLOAT_BYTES
    groups = math.ceil(size / max(1, int(budget // slice_bytes)))
    group = math.ceil(size / groups)
    query, keys = core.scores.input
    values = core.weighted.input[1]
    for name in (query, keys, values):
        if shapes[name][axis] != size:
            raise ValueError(f"{name}: axis {axis} is {shapes[name][axis]}, not {size}")
    nodes = []
    outputs = []
    base = core.scores.name or core.scores.output[0]
    for start in range(0, size, group):
        end = min(start + group, size)
        slice_args = context.slice_inputs(start, end, axis)
        part = f"{base}/heads_{start}_{end}"
        sliced = {}
        for role, name in (("q", query), ("k", keys), ("v", values)):
            sliced[role] = f"{part}/{role}"
            nodes.append(
                helper.make_node(
                    "Slice",
                    [name, *slice_args],
                    [sliced[role]],
                    name=f"{part}/slice_{role}",
                )
            )
        nodes.append(
            helper.make_node(
                "MatMul",
                [sliced["q"], sliced["k"]],
                [f"{part}/scores"],
                name=f"{part}/scores",
            )
        )
        nodes.append(
            helper.make_node(
                "Softmax",
                [f"{part}/scores"],
                [f"{part}/weights"],
                name=f"{part}/softmax",
                axis=softmax_axis(core.softmax),
            )
        )
        nodes.append(
            helper.make_node(
                "MatMul",
                [f"{part}/weights", sliced["v"]],
                [f"{part}/out"],
                name=f"{part}/out",
            )
        )
        outputs.append(f"{part}/out")
    nodes.append(
        helper.make_node(
            "Concat",
            outputs,
            [core.weighted.output[0]],
            name=f"{base}/concat",
            axis=axis,
        )
    )
    scores_mib = batch * heads * frames * frames * FLOAT_BYTES / MIB
    group_mib = scores_mib / size * group
    print(
        f"{base}: {core.shape} {scores_mib:.1f} MiB -> {groups} groups of {group} "
        f"along axis {axis}, {group_mib:.1f} MiB each"
    )
    return nodes


def main() -> None:
    args = parse_args()
    model = onnx.load(args.model)
    shapes: dict[str, list[int]] = json.loads(Path(args.shapes).read_text())
    budget = args.budget_mib * MIB
    graph = model.graph
    cores = find_cores(graph, shapes, budget)
    context = SplitContext()
    replaced = {core.scores.output[0] for core in cores}
    replaced |= {core.softmax.output[0] for core in cores}
    by_last = {core.weighted.output[0]: core for core in cores}
    nodes = []
    for node in graph.node:
        if node.output[0] in replaced:
            continue
        if node.output[0] in by_last:
            nodes.extend(split_core(by_last[node.output[0]], shapes, budget, context))
            continue
        nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    graph.initializer.extend(context.initializers)
    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"saved {args.out}: {len(cores)} attention cores split")


if __name__ == "__main__":
    main()
