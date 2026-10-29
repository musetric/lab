# ruff: noqa: T201 -- CLI audit tool: stdout is its interface.

import argparse
import math
import sys
from pathlib import Path

import onnx
from onnx import numpy_helper, shape_inference

WEBGPU_DISPATCH_ROWS = 65535
SOFTMAX_AXIS_OPSET = 13

LEADING_AXES_OPS = frozenset(
    {"LayerNormalization", "RMSNormalization", "SimplifiedLayerNormalization"}
)
LAST_AXIS_OPS = frozenset(
    {"SkipLayerNormalization", "SkipSimplifiedLayerNormalization"}
)
ONE_AXIS_OPS = frozenset({"LpNormalization", "TopK"})
REDUCE_OPS = frozenset(
    {
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
    }
)
ROW_DISPATCH_OPS = (
    LEADING_AXES_OPS
    | LAST_AXIS_OPS
    | ONE_AXIS_OPS
    | REDUCE_OPS
    | {"Softmax", "InstanceNormalization"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="audit a core for WebGPU dispatch rows"
    )
    parser.add_argument("models", nargs="+", type=Path)
    return parser.parse_args()


def walk_graphs(graph):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("g"):
                yield from walk_graphs(attribute.g)
            for sub in attribute.graphs:
                yield from walk_graphs(sub)


def static_shapes(graph) -> dict[str, list[int]]:
    shapes = {}
    for value in list(graph.value_info) + list(graph.input) + list(graph.output):
        dims = value.type.tensor_type.shape.dim
        if dims and all(d.HasField("dim_value") for d in dims):
            shapes[value.name] = [d.dim_value for d in dims]
    return shapes


def attribute(node, name: str, default):
    return next(
        (onnx.helper.get_attribute_value(a) for a in node.attribute if a.name == name),
        default,
    )


def reduce_axes(node, rank: int, constants: dict) -> list[int] | None:
    axes = attribute(node, "axes", None)
    if axes is None and len(node.input) > 1 and node.input[1]:
        if node.input[1] not in constants:
            return None
        axes = numpy_helper.to_array(constants[node.input[1]]).tolist()
    if axes is None:
        return list(range(rank))
    return [axis + rank if axis < 0 else axis for axis in axes]


def dispatch_rows(node, dims: list[int], opset: int, constants: dict) -> int | None:
    rank = len(dims)
    size = math.prod(dims)
    if node.op_type in LAST_AXIS_OPS:
        return size // dims[-1]
    if node.op_type == "InstanceNormalization":
        return dims[0] * dims[1]
    if node.op_type in REDUCE_OPS:
        axes = reduce_axes(node, rank, constants)
        if axes is None:
            return None
        return size // math.prod(dims[axis] for axis in axes)
    coerced = node.op_type == "Softmax" and opset < SOFTMAX_AXIS_OPSET
    axis = attribute(node, "axis", 1 if coerced else -1)
    axis = axis + rank if axis < 0 else axis
    if node.op_type in LEADING_AXES_OPS or coerced:
        return math.prod(dims[:axis])
    return size // dims[axis]


def audit_model(path: Path) -> tuple[int, int]:
    model = shape_inference.infer_shapes(
        onnx.load(str(path), load_external_data=False), data_prop=True
    )
    opset = next(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0
    )
    counted: dict[tuple[str, str, int], int] = {}
    unknown: dict[str, int] = {}
    for graph in walk_graphs(model.graph):
        shapes = static_shapes(graph)
        constants = {t.name: t for t in graph.initializer}
        for node in graph.node:
            if node.op_type not in ROW_DISPATCH_OPS:
                continue
            dims = shapes.get(node.input[0])
            rows = None if dims is None else dispatch_rows(node, dims, opset, constants)
            if rows is None:
                unknown[node.op_type] = unknown.get(node.op_type, 0) + 1
                continue
            key = (node.op_type, str(dims), rows)
            counted[key] = counted.get(key, 0) + 1

    failures = 0
    for (op, dims, rows), count in sorted(counted.items(), key=lambda kv: -kv[0][2]):
        head = f"{count:4d}x {op:<18} {dims:<22} rows={rows}"
        if rows > WEBGPU_DISPATCH_ROWS:
            failures += count
            print(
                f"  FAIL  {head} > {WEBGPU_DISPATCH_ROWS}: the last row is overwritten"
            )
        else:
            print(f"  ok    {head}")
    for op, count in sorted(unknown.items()):
        failures += count
        print(f"  FAIL  {count:4d}x {op:<18} rows unknown: no static shape or axes")
    if not counted and not unknown:
        print("  ok    no row-dispatched kernel in the graph")
    return failures, len(counted)


def main() -> None:
    args = parse_args()
    failures = 0
    for path in args.models:
        print(f"{path.name}:")
        model_failures, _ = audit_model(path)
        failures += model_failures
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
