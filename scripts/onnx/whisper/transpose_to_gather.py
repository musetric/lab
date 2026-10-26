"""Replace the one Transpose shape Adreno 600-series adapters get wrong with a Gather.

`Transpose` on the WebGPU execution provider returns a quarter of its output as
zeros there, deterministically and with no error. The cause is in the provider's
shader: it stages the tile in `array<array<T, tile_size + 1>, tile_size>`, and
on that hardware a workgroup array whose size is not a multiple of 512 bytes
loses the stores of one warp out of four across `workgroupBarrier`.

Only the shapes that reach that shader are affected: after leading extents of
one are squeezed away, a rank-2 permutation that swaps both axes. In this graph
that is one node - a rank-3 tensor with a leading one whose last two axes swap.
The permutations attention uses, rank 4 and rank 3 with a real leading batch,
take a different kernel and are exact. See musetric/musetric#975.

`Gather` against a precomputed index table returns the same permutation exactly
on the same device, so this rewrites

    y = Transpose(x, perm=[0, 2, 1])          x: [1, C, T]

into

    y = Reshape(Gather(Reshape(x, [-1]), idx), [1, T, C])

with `idx[t * C + c] = c * T + t`. The values are untouched; the cost is one
int32 index per element of the tensor.

The table is static, so the rewrite pins the extents it was built for: run it on
a graph whose shapes are already fixed.

    uv run python scripts/onnx/whisper/transpose_to_gather.py \
      --input  tmp/whisper-export/.../onnx/encoder_model_q4.onnx \
      --output tmp/whisper-export/.../onnx/encoder_model_q4.onnx
"""

# ruff: noqa: T201

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

BATCH_NAMES = {"batch_size", "batch", "N"}
# The broken form is rank three with a leading extent of one.
BROKEN_RANK = 3


def static_shape(value_info, batch):
    """Dimensions with the symbolic batch resolved, or None if anything else is free."""
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.dim_param in BATCH_NAMES:
            dims.append(batch)
        else:
            return None
    return dims


def affected(node, shapes, batch):
    """The rank-3, leading-one, last-two-axis swap - the only broken form."""
    perm = next((list(a.ints) for a in node.attribute if a.name == "perm"), None)
    if perm != [0, 2, 1]:
        return None
    value_info = shapes.get(node.input[0])
    if value_info is None:
        return None
    dims = static_shape(value_info, batch)
    if dims is None or len(dims) != BROKEN_RANK or dims[0] != 1:
        return None
    return dims


def rewrite(graph, node, dims, known):
    _, channels, frames = dims
    prefix = (node.name or "transpose").replace("/", "_").strip("_")

    # out[t * channels + c] reads in[c * frames + t]
    index = (
        np.arange(channels, dtype=np.int32)[None, :] * frames
        + np.arange(frames, dtype=np.int32)[:, None]
    ).reshape(-1)
    index_name = f"gather_index_{channels}x{frames}"
    if index_name not in known:
        graph.initializer.append(numpy_helper.from_array(index, index_name))
        known.add(index_name)

    flat_name = "gather_flat_shape"
    if flat_name not in known:
        graph.initializer.append(
            numpy_helper.from_array(np.array([-1], dtype=np.int64), flat_name)
        )
        known.add(flat_name)

    out_name = f"gather_out_{frames}x{channels}"
    if out_name not in known:
        graph.initializer.append(
            numpy_helper.from_array(
                np.array([1, frames, channels], dtype=np.int64), out_name
            )
        )
        known.add(out_name)

    return [
        helper.make_node(
            "Reshape",
            [node.input[0], flat_name],
            [f"{prefix}_flat"],
            name=f"{prefix}_flatten",
        ),
        helper.make_node(
            "Gather",
            [f"{prefix}_flat", index_name],
            [f"{prefix}_gathered"],
            name=f"{prefix}_gather",
            axis=0,
        ),
        helper.make_node(
            "Reshape",
            [f"{prefix}_gathered", out_name],
            [node.output[0]],
            name=f"{prefix}_restore",
        ),
    ]


def infer_shapes(model):
    inferred = SymbolicShapeInference.infer_shapes(
        model, auto_merge=True, guess_output_rank=True
    )
    shapes = {vi.name: vi for vi in inferred.graph.value_info}
    for value_info in list(model.graph.input) + list(model.graph.output):
        shapes[value_info.name] = value_info
    return shapes


def apply(model, batch=1):
    shapes = infer_shapes(model)
    graph = model.graph
    known = {init.name for init in graph.initializer}
    total = sum(1 for node in graph.node if node.op_type == "Transpose")

    # Keyed by position: a protobuf wrapper's id() is recycled, so matching on
    # it silently replaces unrelated nodes.
    replacements = {}
    indices = 0
    for position, node in enumerate(graph.node):
        if node.op_type != "Transpose":
            continue
        dims = affected(node, shapes, batch)
        if dims is None:
            continue
        replacements[position] = rewrite(graph, node, dims, known)
        indices += dims[1] * dims[2]
        print(f"rewrote {node.name}: {dims} perm=[0, 2, 1]")

    if not replacements:
        print(f"no affected Transpose among {total}; nothing to do")
        return

    rebuilt = []
    for position, node in enumerate(graph.node):
        rebuilt.extend(replacements.get(position, [node]))
    before = len(graph.node)
    del graph.node[:]
    graph.node.extend(rebuilt)

    print(f"rewrote {len(replacements)} of {total} Transpose nodes")
    print(f"nodes: {before} -> {len(graph.node)}")
    print(f"index tables: {indices} int32 = {indices * 4 / 1048576:.1f} MiB")


def count_affected(model, batch=1):
    shapes = infer_shapes(model)
    return sum(
        1
        for node in model.graph.node
        if node.op_type == "Transpose" and affected(node, shapes, batch) is not None
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="value to assume for a symbolic batch dimension",
    )
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    apply(model, args.batch)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
