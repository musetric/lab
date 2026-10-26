# ruff: noqa: T201
"""Static rewrite of ChordNet that avoids the broken Adreno WebGPU kernels.

- Transpose families that select the broken shared-tile kernel
  ((0,2,1), (0,1,3,2), (0,3,1,2), (0,3,2,1), (2,0,1)) become
  Reshape(1D) -> Gather(const int32 idx) -> Reshape(target). The primitive was
  verified bit-exact against the untouched graph on both Adreno phones.
  Rank-5 (2,0,3,1,4) head-split transposes and rank-4 (0,2,1,3) are kept.
- Conv becomes Pad + Reshape + Gather(im2col idx) + MatMul(reshaped weights,
  bias added back). Verified faithful to the original Conv within fp32
  accumulation noise.
- A four-dimensional MatMul folds its two batch axes into one, and a MatMul
  whose constant weight has a width that is not a multiple of four is widened
  with zero columns and sliced back. Both shapes return wrong values on Adreno
  750 while the shapes they become are exact there. With --vec4-attention a
  four-dimensional MatMul of more than eight rows is padded to the vec4 kernel
  instead of folded.

Tensor shapes come from `capture_shapes.py` because ONNX shape inference
cannot resolve runtime-computed Reshape shapes.

Every Gather index and Reshape target is baked for the captured batch and
frames, so the rewritten graph runs at that shape and no other. Its input and
output dims are pinned to it: a graph that still declares `windows` looks
batchable and fails inside a Reshape on any other batch, and a symbolic dim
keeps the shape arithmetic of the source graph from folding, which leaves it on
the CPU and blocks WebGPU graph capture.

The rewrite is faithful. The Adreno defects it avoids, and those it does not,
are listed with their workarounds in musetric/musetric#975.

Usage:
    uv run python capture_shapes.py --model beat_this.onnx --frames 1500 \
        --out shapes1500.json
    uv run python rewrite_static_adreno.py --model beat_this.onnx \
        --out beat_this_t1500.onnx --shapes shapes1500.json \
        [--no-conv] [--no-transpose]
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPLACE_PERMS = {
    (0, 2, 1),
    (0, 1, 3, 2),
    (0, 3, 1, 2),
    (0, 3, 2, 1),
    (2, 0, 1),
    (1, 0, 2),
}
CONV_BIAS_INPUT = 2
RANK_2D = 2
RANK_4D = 4
MATMUL_WIDTH_MULTIPLE = 4
VEC4_MIN_ROWS = 8


@dataclass
class RewriteContext:
    shapes: dict[str, list[int]]
    initializers: dict[str, onnx.TensorProto]
    new_nodes: list[onnx.NodeProto] = field(default_factory=list)
    new_inits: list[onnx.TensorProto] = field(default_factory=list)
    gather_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], str] = field(
        default_factory=dict
    )
    replaced_transposes: int = 0
    replaced_convs: int = 0
    replaced_matmuls: int = 0
    padded_matmuls: int = 0
    vec4_matmuls: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="dynamic source .onnx")
    parser.add_argument("--out", required=True, help="rewritten .onnx output")
    parser.add_argument(
        "--shapes", required=True, help="shapes .json from capture_shapes"
    )
    parser.add_argument("--no-conv", action="store_true", help="keep Conv nodes")
    parser.add_argument(
        "--no-matmul", action="store_true", help="keep four-dimensional MatMul nodes"
    )
    parser.add_argument(
        "--no-transpose", action="store_true", help="keep Transpose nodes"
    )
    parser.add_argument(
        "--vec4-attention",
        action="store_true",
        help="pad four-dimensional MatMuls of more than eight rows to the vec4 "
        "kernel instead of folding them into three dimensions",
    )
    return parser.parse_args()


def gather_index(
    in_dims: list[int], perm: tuple[int, ...]
) -> tuple[np.ndarray, list[int]]:
    out_dims = [in_dims[p] for p in perm]
    coords = np.unravel_index(np.arange(int(np.prod(out_dims))), out_dims)
    iin: list[np.ndarray | None] = [None] * len(in_dims)
    for k in range(len(perm)):
        iin[perm[k]] = coords[k]
    q = np.ravel_multi_index(iin, in_dims)
    return q.astype(np.int32), out_dims


def rewrite_transpose(
    ctx: RewriteContext, node: onnx.NodeProto, perm: tuple[int, ...]
) -> bool:
    shape = ctx.shapes.get(node.input[0])
    if shape is None or len(shape) != len(perm):
        return False
    key = (tuple(shape), perm)
    idx_name = ctx.gather_cache.get(key)
    if idx_name is None:
        q, out_dims = gather_index(shape, perm)
        idx_name = f"gi_{len(ctx.gather_cache)}_{abs(hash(key)) % 99999}"
        ctx.new_inits.append(
            helper.make_tensor(idx_name, TensorProto.INT32, [q.size], q)
        )
        ctx.gather_cache[key] = idx_name
    numel = int(np.prod(shape))
    out_dims = [shape[p] for p in perm]
    ctx.new_inits.append(
        helper.make_tensor(f"{node.name}/gshape", TensorProto.INT64, [1], [numel])
    )
    ctx.new_inits.append(
        helper.make_tensor(
            f"{node.name}/oshape", TensorProto.INT64, [len(out_dims)], out_dims
        )
    )
    ctx.new_nodes.extend(
        [
            helper.make_node(
                "Reshape",
                [node.input[0], f"{node.name}/gshape"],
                [f"{node.name}/gflat"],
                name=node.name + "_ri",
            ),
            helper.make_node(
                "Gather",
                [f"{node.name}/gflat", idx_name],
                [f"{node.name}/gg"],
                axis=0,
                name=node.name + "_g",
            ),
            helper.make_node(
                "Reshape",
                [f"{node.name}/gg", f"{node.name}/oshape"],
                [node.output[0]],
                name=node.name + "_ro",
            ),
        ]
    )
    return True


def rewrite_matmul(ctx: RewriteContext, node: onnx.NodeProto) -> bool:
    """Compute a four-dimensional MatMul as a three-dimensional one.

    On Adreno 750 the WebGPU MatMul kernel returns wrong values for some
    four-dimensional extents while the same extents in three dimensions are
    exact, which is what makes the chord attention disagree with wasm there.
    Folding the two batch axes into one is exact: a batched MatMul multiplies
    each matrix of the batch on its own, whatever the batch is shaped like.
    """
    left = ctx.shapes.get(node.input[0])
    right = ctx.shapes.get(node.input[1])
    if left is None or right is None:
        return False
    if len(left) != RANK_4D or len(right) != RANK_4D or left[:2] != right[:2]:
        return False
    batch = left[0] * left[1]
    folded = [
        (f"{node.name}/lshape", [batch, left[2], left[3]]),
        (f"{node.name}/rshape", [batch, right[2], right[3]]),
        (f"{node.name}/oshape", [left[0], left[1], left[2], right[3]]),
    ]
    for name, dims in folded:
        ctx.new_inits.append(
            helper.make_tensor(name, TensorProto.INT64, [len(dims)], dims)
        )
    ctx.new_nodes.extend(
        [
            helper.make_node(
                "Reshape",
                [node.input[0], f"{node.name}/lshape"],
                [f"{node.name}/left3d"],
                name=node.name + "_rl",
            ),
            helper.make_node(
                "Reshape",
                [node.input[1], f"{node.name}/rshape"],
                [f"{node.name}/right3d"],
                name=node.name + "_rr",
            ),
            helper.make_node(
                "MatMul",
                [f"{node.name}/left3d", f"{node.name}/right3d"],
                [f"{node.name}/mm3d"],
                name=node.name + "_mm",
            ),
            helper.make_node(
                "Reshape",
                [f"{node.name}/mm3d", f"{node.name}/oshape"],
                [node.output[0]],
                name=node.name + "_ro",
            ),
        ]
    )
    return True


def pad_matmul_vec4(ctx: RewriteContext, node: onnx.NodeProto) -> bool:
    """Pad a four-dimensional MatMul of two activations to the vec4 kernel.

    onnxruntime's WebGPU MatMul takes its vec4 kernel when K and N are
    multiples of four and a scalar one otherwise, and the Adreno 750 driver
    miscompiles the scalar one for two- and four-dimensional products. Zero
    rows and columns along K add nothing to the product, and the extra output
    columns along N are sliced off, so every real value keeps its arithmetic.

    Only products with more than eight rows: with fewer, onnxruntime runs the
    vec4 kernel with one row per thread, which is wrong on Adreno 660 for most
    N; those keep the fold into three dimensions.
    """
    left = ctx.shapes.get(node.input[0])
    right = ctx.shapes.get(node.input[1])
    if left is None or right is None:
        return False
    if len(left) != RANK_4D or len(right) != RANK_4D or left[:2] != right[:2]:
        return False
    rows, inner, width = left[2], left[3], right[3]
    if rows <= VEC4_MIN_ROWS:
        return False
    padded_inner = -(-inner // MATMUL_WIDTH_MULTIPLE) * MATMUL_WIDTH_MULTIPLE
    padded_width = -(-width // MATMUL_WIDTH_MULTIPLE) * MATMUL_WIDTH_MULTIPLE
    if padded_inner == inner and padded_width == width:
        return False
    left_name, right_name = node.input[0], node.input[1]
    zero = f"{node.name}/zero"
    ctx.new_inits.append(helper.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
    if padded_inner != inner:
        pads = [0, 0, 0, 0, 0, 0, 0, padded_inner - inner]
        ctx.new_inits.append(
            helper.make_tensor(f"{node.name}/lpads", TensorProto.INT64, [8], pads)
        )
        left_name = f"{node.name}/left4"
        ctx.new_nodes.append(
            helper.make_node(
                "Pad",
                [node.input[0], f"{node.name}/lpads", zero],
                [left_name],
                name=node.name + "_pl",
            )
        )
    pads = [0, 0, 0, 0, 0, 0, padded_inner - inner, padded_width - width]
    ctx.new_inits.append(
        helper.make_tensor(f"{node.name}/rpads", TensorProto.INT64, [8], pads)
    )
    right_name = f"{node.name}/right4"
    ctx.new_nodes.append(
        helper.make_node(
            "Pad",
            [node.input[1], f"{node.name}/rpads", zero],
            [right_name],
            name=node.name + "_pr",
        )
    )
    if padded_width == width:
        ctx.new_nodes.append(
            helper.make_node(
                "MatMul", [left_name, right_name], [node.output[0]], name=node.name
            )
        )
        return True
    for name, dims in (
        (f"{node.name}/from", [0]),
        (f"{node.name}/to", [width]),
        (f"{node.name}/axis", [-1]),
    ):
        ctx.new_inits.append(helper.make_tensor(name, TensorProto.INT64, [1], dims))
    ctx.new_nodes.extend(
        [
            helper.make_node(
                "MatMul",
                [left_name, right_name],
                [f"{node.name}/wide"],
                name=node.name + "_mm4",
            ),
            helper.make_node(
                "Slice",
                [
                    f"{node.name}/wide",
                    f"{node.name}/from",
                    f"{node.name}/to",
                    f"{node.name}/axis",
                ],
                [node.output[0]],
                name=node.name + "_slice",
            ),
        ]
    )
    return True


def pad_matmul_width(ctx: RewriteContext, node: onnx.NodeProto) -> bool:
    """Widen a constant MatMul weight to a multiple of four columns.

    On Adreno 750 the WebGPU MatMul kernel returns wrong values when the output
    width is not a multiple of four: the classifier of this model,
    [16, 108, 144] x [144, 170], differs from wasm by 4.2 there, while the same
    product at 168 or 172 columns is exact. The extra columns are zeros and the
    result is sliced back, so every real column keeps its arithmetic.
    """
    weight = ctx.initializers.get(node.input[1])
    if weight is None or len(weight.dims) != RANK_2D:
        return False
    width = weight.dims[1]
    if width % MATMUL_WIDTH_MULTIPLE == 0:
        return False
    padded_width = width + MATMUL_WIDTH_MULTIPLE - width % MATMUL_WIDTH_MULTIPLE
    values = numpy_helper.to_array(weight)
    padded = np.zeros((weight.dims[0], padded_width), dtype=values.dtype)
    padded[:, :width] = values
    ctx.new_inits.append(numpy_helper.from_array(padded, f"{node.name}/wide"))
    for name, dims in (
        (f"{node.name}/from", [0]),
        (f"{node.name}/to", [width]),
        (f"{node.name}/axis", [-1]),
    ):
        ctx.new_inits.append(helper.make_tensor(name, TensorProto.INT64, [1], dims))
    ctx.new_nodes.extend(
        [
            helper.make_node(
                "MatMul",
                [node.input[0], f"{node.name}/wide"],
                [f"{node.name}/widened"],
                name=node.name + "_mmw",
            ),
            helper.make_node(
                "Slice",
                [
                    f"{node.name}/widened",
                    f"{node.name}/from",
                    f"{node.name}/to",
                    f"{node.name}/axis",
                ],
                [node.output[0]],
                name=node.name + "_slice",
            ),
        ]
    )
    return True


def rewrite_conv(ctx: RewriteContext, node: onnx.NodeProto) -> bool:
    weight_name = node.input[1]
    if weight_name not in ctx.initializers:
        return False
    weight = numpy_helper.to_array(ctx.initializers[weight_name])
    shape = ctx.shapes.get(node.input[0], [])
    if any(d is None for d in shape):
        return False
    if shape[0] != 1:
        raise SystemExit(
            f"{node.name}: the Conv rewrite is batch-1 only, captured batch {shape[0]}"
        )
    c_out, _, kh, kw = weight.shape
    c_in, hin, win = shape[1], shape[2], shape[3]
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    pt, pl, pb, pr = attrs.get("pads", [0, 0, 0, 0])
    sh, sw = attrs.get("strides", [1, 1])
    h_pad, w_pad = hin + pt + pb, win + pl + pr
    h_out = (h_pad - kh) // sh + 1
    w_out = (w_pad - kw) // sw + 1
    cc = np.arange(c_in).reshape(c_in, 1, 1, 1, 1)
    ii = np.arange(kh).reshape(1, kh, 1, 1, 1)
    jj = np.arange(kw).reshape(1, 1, kw, 1, 1)
    uu = (np.arange(h_out) * sh).reshape(1, 1, 1, h_out, 1)
    vv = (np.arange(w_out) * sw).reshape(1, 1, 1, 1, w_out)
    q = (cc * h_pad + uu + ii) * w_pad + vv + jj
    q = np.ascontiguousarray(np.broadcast_to(q, (c_in, kh, kw, h_out, w_out))).ravel()
    ctx.new_inits.extend(
        [
            helper.make_tensor(
                f"{node.name}/pads",
                TensorProto.INT64,
                [8],
                [0, 0, pt, pl, 0, 0, pb, pr],
            ),
            helper.make_tensor(f"{node.name}/zero", TensorProto.FLOAT, [1], [0.0]),
            helper.make_tensor(f"{node.name}/q", TensorProto.INT32, [q.size], q),
            helper.make_tensor(
                f"{node.name}/cshape", TensorProto.INT64, [1], [c_in * h_pad * w_pad]
            ),
            helper.make_tensor(
                f"{node.name}/pshape",
                TensorProto.INT64,
                [2],
                [c_in * kh * kw, h_out * w_out],
            ),
            helper.make_tensor(
                f"{node.name}/oshape", TensorProto.INT64, [4], [1, c_out, h_out, w_out]
            ),
            numpy_helper.from_array(
                np.ascontiguousarray(weight.reshape(c_out, -1)), f"{node.name}/w2"
            ),
        ]
    )
    ctx.new_nodes.extend(
        [
            helper.make_node(
                "Pad",
                [node.input[0], f"{node.name}/pads", f"{node.name}/zero"],
                [f"{node.name}/pad"],
                mode="constant",
                name=node.name + "_pad",
            ),
            helper.make_node(
                "Reshape",
                [f"{node.name}/pad", f"{node.name}/cshape"],
                [f"{node.name}/flat"],
                name=node.name + "_ri",
            ),
            helper.make_node(
                "Gather",
                [f"{node.name}/flat", f"{node.name}/q"],
                [f"{node.name}/patches"],
                axis=0,
                name=node.name + "_g",
            ),
            helper.make_node(
                "Reshape",
                [f"{node.name}/patches", f"{node.name}/pshape"],
                [f"{node.name}/pmat"],
                name=node.name + "_rp",
            ),
            helper.make_node(
                "MatMul",
                [f"{node.name}/w2", f"{node.name}/pmat"],
                [f"{node.name}/mm"],
                name=node.name + "_mm",
            ),
        ]
    )
    if len(node.input) > CONV_BIAS_INPUT and node.input[CONV_BIAS_INPUT]:
        bias = numpy_helper.to_array(ctx.initializers[node.input[CONV_BIAS_INPUT]])
        ctx.new_inits.append(
            numpy_helper.from_array(bias.reshape(c_out, 1), f"{node.name}/b2")
        )
        ctx.new_nodes.append(
            helper.make_node(
                "Add",
                [f"{node.name}/mm", f"{node.name}/b2"],
                [f"{node.name}/biased"],
                name=node.name + "_add",
            )
        )
        ctx.new_nodes.append(
            helper.make_node(
                "Reshape",
                [f"{node.name}/biased", f"{node.name}/oshape"],
                [node.output[0]],
                name=node.name + "_ro",
            )
        )
    else:
        ctx.new_nodes.append(
            helper.make_node(
                "Reshape",
                [f"{node.name}/mm", f"{node.name}/oshape"],
                [node.output[0]],
                name=node.name + "_ro",
            )
        )
    return True


def pin_io_dims(graph: onnx.GraphProto, shapes: dict[str, list[int]]) -> dict:
    """Pin symbolic graph input/output dims to the captured input shapes."""
    values: dict[str, int] = {}
    for value in graph.input:
        captured = shapes.get(value.name)
        if captured is None:
            continue
        for dim, size in zip(value.type.tensor_type.shape.dim, captured, strict=True):
            if dim.HasField("dim_param"):
                values[dim.dim_param] = size
    for value in list(graph.input) + list(graph.output):
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_param") and dim.dim_param in values:
                dim.dim_value = values[dim.dim_param]
    return values


def main() -> None:
    args = parse_args()
    model = onnx.load(args.model)
    graph = model.graph
    ctx = RewriteContext(
        shapes=json.loads(Path(args.shapes).read_text(encoding="utf-8")),
        initializers={init.name: init for init in graph.initializer},
        new_inits=list(graph.initializer),
    )
    for node in graph.node:
        if node.op_type == "Transpose" and not args.no_transpose:
            perm = tuple(next(a.ints for a in node.attribute if a.name == "perm"))
            if perm in REPLACE_PERMS and rewrite_transpose(ctx, node, perm):
                ctx.replaced_transposes += 1
                continue
        if (
            node.op_type == "MatMul"
            and not args.no_matmul
            and args.vec4_attention
            and pad_matmul_vec4(ctx, node)
        ):
            ctx.vec4_matmuls += 1
            continue
        if (
            node.op_type == "MatMul"
            and not args.no_matmul
            and rewrite_matmul(ctx, node)
        ):
            ctx.replaced_matmuls += 1
            continue
        if (
            node.op_type == "MatMul"
            and not args.no_matmul
            and pad_matmul_width(ctx, node)
        ):
            ctx.padded_matmuls += 1
            continue
        if node.op_type == "Conv" and not args.no_conv and rewrite_conv(ctx, node):
            ctx.replaced_convs += 1
            print(f"conv {node.name}: rewritten")
            continue
        ctx.new_nodes.append(node)

    pinned = pin_io_dims(graph, ctx.shapes)
    new_graph = helper.make_graph(
        ctx.new_nodes,
        graph.name + "_adreno",
        list(graph.input),
        list(graph.output),
        initializer=ctx.new_inits,
    )
    new_model = helper.make_model(
        new_graph, opset_imports=model.opset_import, producer_name=__name__
    )
    new_model.ir_version = model.ir_version
    onnx.checker.check_model(new_model, full_check=False)
    onnx.save(new_model, args.out)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(
        f"saved {args.out}: transposes {ctx.replaced_transposes}, "
        f"convs {ctx.replaced_convs}, matmuls {ctx.replaced_matmuls}, "
        f"padded {ctx.padded_matmuls}, vec4 {ctx.vec4_matmuls}, "
        f"{size_mb:.0f} MB, pinned {pinned}"
    )


if __name__ == "__main__":
    main()
