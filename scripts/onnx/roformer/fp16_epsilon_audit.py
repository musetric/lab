"""Audit exported ONNX graphs for normalization epsilons that fp16 cannot carry.

1. A fused normalization (RMSNormalization, LayerNormalization, ...) on an fp16
   node casts the reciprocal 1/sqrt(var + epsilon) to fp16, so its epsilon must
   stay above 1/65504^2 = 2.33e-10.

2. A decomposed normalization (ReduceMean -> Add(eps) -> Sqrt -> Div) that keeps
   its epsilon as an fp16 initializer below the smallest normal fp16 value,
   6.10e-5, reads it as zero on a GPU that flushes denormals.

The rule this enforces is (1). (2) is reported as a warning: it only fires on a
flushing device and only for a row whose variance is exactly zero. Neither bounds
how large an epsilon may be.

Usage:
  uv run python scripts/onnx/roformer/fp16_epsilon_audit.py model.onnx [more.onnx ...]
  uv run python scripts/onnx/roformer/fp16_epsilon_audit.py --weights model.onnx
"""

# ruff: noqa: T201 -- CLI audit tool: stdout is its interface.

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

FP16_MAX = 65504.0
# below this an fp16 1/sqrt(var + epsilon) overflows to +inf for a zero row
FP16_RSQRT_FLOOR = 1.0 / FP16_MAX**2
# the floor is exact and data-independent, so this only covers f32 rounding
FP16_RSQRT_MARGIN = 2.0 * FP16_RSQRT_FLOOR
# smallest normal fp16; below it a denormal-flushing GPU reads the epsilon as 0
FP16_MIN_NORMAL = 6.103515625e-05
# an epsilon is never larger than this, so a bigger constant is something else
EPSILON_CEILING = 1e-3

# every op whose epsilon feeds a 1/sqrt(var + epsilon)
EPSILON_OPS = frozenset(
    {
        "BatchNormalization",
        "EmbedLayerNormalization",
        "GroupNormalization",
        "InstanceNormalization",
        "LayerNormalization",
        "RMSNormalization",
        "SimplifiedLayerNormalization",
        "SkipLayerNormalization",
        "SkipSimplifiedLayerNormalization",
    }
)
# the tail of a decomposed normalization, reached from the Add that holds the eps
NORMALIZE_TAILS = frozenset({"Sqrt", "Reciprocal", "Pow", "Rsqrt"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument(
        "--weights",
        action="store_true",
        help="also load the external data and report non-finite fp16 weights "
        "(an inf weight makes its whole MatMul column NaN).",
    )
    return parser.parse_args()


def walk_graphs(graph):
    """The graph and every subgraph an If/Loop/Scan attribute carries."""
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("g"):
                yield from walk_graphs(attribute.g)
            for sub in attribute.graphs:
                yield from walk_graphs(sub)


def scalar_constants(graph) -> dict[str, tuple[float, int]]:
    """Scalar float constants by name, from initializers and from Constant nodes."""
    tensors = [(t.name, t) for t in graph.initializer]
    for node in graph.node:
        if node.op_type == "Constant":
            value = next((a.t for a in node.attribute if a.name == "value"), None)
            if value is not None:
                tensors.append((node.output[0], value))

    found: dict[str, tuple[float, int]] = {}
    for name, tensor in tensors:
        if tensor.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
            continue
        # an epsilon is always inline; skipping external data keeps the audit
        # runnable without the weights file beside the graph
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            continue
        array = numpy_helper.to_array(tensor)
        if array.size == 1:
            found[name] = (float(array.reshape(-1)[0]), tensor.data_type)
    return found


def tensor_dtypes(graph) -> dict[str, int]:
    dtypes = {t.name: t.data_type for t in graph.initializer}
    for value in list(graph.value_info) + list(graph.input) + list(graph.output):
        dtypes[value.name] = value.type.tensor_type.elem_type
    return dtypes


def fused_epsilons(graph) -> list[tuple[str, int, float]]:
    """(label, dtype, epsilon) for every fused normalization in one graph."""
    dtypes = tensor_dtypes(graph)
    found = []
    for node in graph.node:
        if node.op_type not in EPSILON_OPS:
            continue
        epsilon = next((a.f for a in node.attribute if a.name == "epsilon"), None)
        if epsilon is None:
            continue
        dtype = next(
            (dtypes[n] for n in [*node.output, *node.input] if n in dtypes),
            onnx.TensorProto.UNDEFINED,
        )
        found.append((node.op_type, dtype, epsilon))
    return found


def decomposed_epsilons(
    graph, constants: dict[str, tuple[float, int]]
) -> list[tuple[str, int, float]]:
    """Same, for the ReduceMean -> Add(eps) -> Sqrt/Reciprocal form.

    `constants` spans every graph in the file, because a subgraph reads the outer
    scope: an If branch normalizes with an epsilon held by the parent graph.
    """
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    found = []
    for node in graph.node:
        if node.op_type != "Add":
            continue
        epsilon = next((constants[n] for n in node.input if n in constants), None)
        if epsilon is None:
            continue
        tail = [c.op_type for c in consumers.get(node.output[0], [])]
        normalize = next((op for op in tail if op in NORMALIZE_TAILS), None)
        if normalize is None:
            continue
        value, dtype = epsilon
        if 0 < value < EPSILON_CEILING:
            found.append((f"Add(eps) -> {normalize}", dtype, value))
    return found


def report(counted: dict[tuple[str, int, float], int]) -> tuple[int, int]:
    failures = 0
    warnings = 0
    for (label, dtype, epsilon), count in sorted(counted.items()):
        dtype_name = onnx.TensorProto.DataType.Name(dtype).lower()
        head = f"{count:4d}x {label:<26} {dtype_name:<9} epsilon={epsilon:g}"
        if dtype != onnx.TensorProto.FLOAT16:
            print(f"  ok    {head}")
        elif label in EPSILON_OPS:
            # a fused op keeps its epsilon in an f32 uniform, so only the
            # reciprocal it produces has to stay inside fp16
            if epsilon < FP16_RSQRT_FLOOR:
                failures += count
                print(f"  FAIL  {head} < {FP16_RSQRT_FLOOR:g}: a zero row becomes NaN")
            elif epsilon < FP16_RSQRT_MARGIN:
                warnings += count
                print(f"  warn  {head} is within 2x of the floor {FP16_RSQRT_FLOOR:g}")
            else:
                print(f"  ok    {head}")
        elif epsilon < FP16_MIN_NORMAL:
            warnings += count
            print(f"  warn  {head} is denormal in fp16: reads as 0 on Apple GPUs")
        else:
            print(f"  ok    {head}")
    if not counted:
        print("  ok    no normalization carries an epsilon")
    return failures, warnings


def count_nonfinite_weights(model) -> int:
    nonfinite = 0
    for graph in walk_graphs(model.graph):
        for tensor in graph.initializer:
            if tensor.data_type != onnx.TensorProto.FLOAT16:
                continue
            array = numpy_helper.to_array(tensor).astype(np.float32)
            nonfinite += int((~np.isfinite(array)).sum())
    return nonfinite


def audit_model(path: Path, weights: bool) -> tuple[int, int]:
    model = onnx.load(str(path), load_external_data=weights)
    constants: dict[str, tuple[float, int]] = {}
    for graph in walk_graphs(model.graph):
        constants.update(scalar_constants(graph))

    counted: dict[tuple[str, int, float], int] = {}
    for graph in walk_graphs(model.graph):
        for key in fused_epsilons(graph) + decomposed_epsilons(graph, constants):
            counted[key] = counted.get(key, 0) + 1

    failures, warnings = report(counted)
    if weights:
        nonfinite = count_nonfinite_weights(model)
        if nonfinite:
            failures += 1
            print(f"  FAIL  {nonfinite} non-finite fp16 weight value(s)")
        else:
            print("  ok    every fp16 weight is finite")
    return failures, warnings


def main() -> None:
    args = parse_args()
    failures = 0
    warnings = 0
    for path in args.models:
        print(f"{path.name}:")
        model_failures, model_warnings = audit_model(path, args.weights)
        failures += model_failures
        warnings += model_warnings
    print(f"\n{failures} failure(s), {warnings} warning(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
