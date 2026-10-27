# ruff: noqa: T201
"""Static rewrite of beat_this that avoids the Adreno 6xx broken WebGPU kernels.

- Transpose families that select the broken shared-tile kernel
  ((0,2,1), (0,1,3,2), (0,3,1,2), (0,3,2,1), (2,0,1)) become
  Reshape(1D) -> Gather(const int32 idx) -> Reshape(target).
  Rank-5 (2,0,3,1,4) head-split transposes and rank-4 (0,2,1,3) are kept.
- Conv becomes Pad + Reshape + Gather(im2col idx) + MatMul(reshaped weights,
  bias added back).
- Inference BatchNormalization becomes Mul + Add (folded running stats).

Ship the rewrite pinned at the author's 1500 frames, then split the attention
with split_attention_webgpu.py. The rewrite itself is the same at any frame
count; only the index tables grow with it.

Tensor shapes come from `capture_shapes.py` because ONNX shape inference
cannot resolve runtime-computed Reshape shapes.

Usage:
    uv run python capture_shapes.py --model beat_this.onnx --frames 1500 \
        --out shapes1500.json
    uv run python rewrite_static_adreno.py --model beat_this.onnx \
        --out beat_this.onnx --shapes shapes1500.json \
        [--no-conv] [--no-transpose] [--no-bn]
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
}
CONV_BIAS_INPUT = 2
BATCHNORM_MIN_INPUTS = 5
CHANNEL_AXIS = 1


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
    replaced_batchnorms: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="dynamic source .onnx")
    parser.add_argument("--out", required=True, help="rewritten .onnx output")
    parser.add_argument(
        "--shapes", required=True, help="shapes .json from capture_shapes"
    )
    parser.add_argument("--no-conv", action="store_true", help="keep Conv nodes")
    parser.add_argument(
        "--no-transpose", action="store_true", help="keep Transpose nodes"
    )
    parser.add_argument(
        "--no-bn", action="store_true", help="keep BatchNormalization nodes"
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


def rewrite_conv(ctx: RewriteContext, node: onnx.NodeProto) -> bool:
    weight_name = node.input[1]
    if weight_name not in ctx.initializers:
        return False
    weight = numpy_helper.to_array(ctx.initializers[weight_name])
    shape = ctx.shapes.get(node.input[0], [])
    if any(d is None for d in shape):
        return False
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


def rewrite_batchnorm(ctx: RewriteContext, node: onnx.NodeProto) -> bool:
    if len(node.input) < BATCHNORM_MIN_INPUTS:
        return False
    x_name, scale_name, bias_name, mean_name, var_name = node.input[
        :BATCHNORM_MIN_INPUTS
    ]
    needed = (scale_name, bias_name, mean_name, var_name)
    if any(name not in ctx.initializers for name in needed):
        return False
    x_shape = ctx.shapes.get(x_name)
    if x_shape is None or len(x_shape) <= CHANNEL_AXIS:
        return False
    epsilon = 1e-5
    for attribute in node.attribute:
        if attribute.name == "epsilon":
            epsilon = float(helper.get_attribute_value(attribute))
    scale = numpy_helper.to_array(ctx.initializers[scale_name]).astype(np.float32)
    bias = numpy_helper.to_array(ctx.initializers[bias_name]).astype(np.float32)
    mean = numpy_helper.to_array(ctx.initializers[mean_name]).astype(np.float32)
    var = numpy_helper.to_array(ctx.initializers[var_name]).astype(np.float32)
    inv = scale / np.sqrt(var + epsilon)
    shift = bias - mean * inv
    broadcast = [1] * len(x_shape)
    broadcast[CHANNEL_AXIS] = int(inv.size)
    ctx.new_inits.append(
        numpy_helper.from_array(
            np.ascontiguousarray(inv.reshape(broadcast)), f"{node.name}/bn_scale"
        )
    )
    ctx.new_inits.append(
        numpy_helper.from_array(
            np.ascontiguousarray(shift.reshape(broadcast)), f"{node.name}/bn_shift"
        )
    )
    ctx.new_nodes.extend(
        [
            helper.make_node(
                "Mul",
                [x_name, f"{node.name}/bn_scale"],
                [f"{node.name}/bn_mul"],
                name=node.name + "_mul",
            ),
            helper.make_node(
                "Add",
                [f"{node.name}/bn_mul", f"{node.name}/bn_shift"],
                [node.output[0]],
                name=node.name + "_add",
            ),
        ]
    )
    return True


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
        if node.op_type == "Conv" and not args.no_conv and rewrite_conv(ctx, node):
            ctx.replaced_convs += 1
            print(f"conv {node.name}: rewritten")
            continue
        if (
            node.op_type == "BatchNormalization"
            and not args.no_bn
            and rewrite_batchnorm(ctx, node)
        ):
            ctx.replaced_batchnorms += 1
            print(f"batchnorm {node.name}: folded")
            continue
        ctx.new_nodes.append(node)

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
        f"convs {ctx.replaced_convs}, batchnorms {ctx.replaced_batchnorms}, "
        f"{size_mb:.0f} MB"
    )


if __name__ == "__main__":
    main()
