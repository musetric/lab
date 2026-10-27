# ruff: noqa: T201

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

TIME_AXIS = 2
CHECK_TOLERANCE = 1e-4
PASS_THROUGH = {"BatchNormalization", "Relu", "Add", "Mul", "MatMul"}
MIN_BANDS = 3


@dataclass
class FoldContext:
    bands: int
    shapes: dict[str, list[int]]
    initializers: dict[str, np.ndarray]
    nodes: list[onnx.NodeProto] = field(default_factory=list)
    constants: list[onnx.TensorProto] = field(default_factory=list)
    zero_rows: dict[tuple[int, int], str] = field(default_factory=dict)
    count: int = 0

    def name(self, hint: str) -> str:
        self.count += 1
        return f"fold/{self.count}/{hint}"

    def constant(self, values: np.ndarray, hint: str) -> str:
        name = self.name(hint)
        self.constants.append(numpy_helper.from_array(values, name))
        return name

    def indices(self, values: list[int]) -> str:
        return self.constant(np.array(values, dtype=np.int64), "indices")

    def node(
        self, op: str, inputs: list[str], hint: str, out: str = "", **attrs: object
    ) -> str:
        out = out or self.name(hint)
        self.nodes.append(helper.make_node(op, inputs, [out], name=out, **attrs))
        return out

    def slice(self, value: str, axis: int, start: int, end: int) -> str:
        inputs = [value, *(self.indices([v]) for v in (start, end, axis))]
        return self.node("Slice", inputs, "slice")

    def zero_row(self, channels: int, width: int) -> str:
        key = (channels, width)
        if key not in self.zero_rows:
            shape = self.indices([1, channels, 1, width])
            zero = numpy_helper.from_array(np.zeros(1, dtype=np.float32))
            self.zero_rows[key] = self.node(
                "ConstantOfShape", [shape], "zero_row", value=zero
            )
        return self.zero_rows[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="original KARA2 .onnx")
    parser.add_argument("--out", required=True, help="folded .onnx output")
    parser.add_argument(
        "--bands",
        type=int,
        default=4,
        help="bands of time frames in the batch axis; at least 3, so that every "
        "slice of the halo exchange is itself a batch, and a divisor of the "
        "frame count of every U-Net level",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare source and output on the CPU provider",
    )
    return parser.parse_args()


def attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def copy(
    ctx: FoldContext, node: onnx.NodeProto, inputs: list[str], **override: object
) -> str:
    attrs = attributes(node)
    attrs.update(override)
    return ctx.node(node.op_type, inputs, node.op_type, **attrs)


def is_conv3(ctx: FoldContext, node: onnx.NodeProto) -> bool:
    if node.op_type != "Conv":
        return False
    attrs = attributes(node)
    kernel = list(ctx.initializers[node.input[1]].shape[2:])
    padding = list(attrs.get("pads", [0, 0, 0, 0]))
    return kernel == [3, 3] and padding == [1, 1, 1, 1]


def pairs_frames(ctx: FoldContext, node: onnx.NodeProto) -> bool:
    attrs = attributes(node)
    kernel = list(ctx.initializers[node.input[1]].shape[2:])
    padding = list(attrs.get("pads", [0, 0, 0, 0]))
    return (
        kernel == [2, 2]
        and list(attrs.get("strides", [])) == [2, 2]
        and not any(padding)
    )


def band_rows(ctx: FoldContext, name: str, multiple: int) -> int:
    frames = ctx.shapes[name][TIME_AXIS]
    if frames % (ctx.bands * multiple):
        message = f"{name}: {frames} frames do not split into {ctx.bands} bands"
        raise ValueError(message)
    return frames // ctx.bands


def conv3(ctx: FoldContext, node: onnx.NodeProto, value: str) -> str:
    _, channels, _, width = ctx.shapes[node.input[0]]
    rows = band_rows(ctx, node.input[0], 1)
    last = ctx.slice(value, TIME_AXIS, rows - 1, rows)
    first = ctx.slice(value, TIME_AXIS, 0, 1)
    zero = ctx.zero_row(channels, width)
    above = ctx.node(
        "Concat", [zero, ctx.slice(last, 0, 0, ctx.bands - 1)], "above", axis=0
    )
    below = ctx.node(
        "Concat", [ctx.slice(first, 0, 1, ctx.bands), zero], "below", axis=0
    )
    padded = ctx.node("Concat", [above, value, below], "padded", axis=TIME_AXIS)
    return copy(ctx, node, [padded, *node.input[1:]], pads=[0, 1, 0, 1])


def fold_head(ctx: FoldContext, head: list[onnx.NodeProto], source: str) -> str:
    _, _, _, frames = ctx.shapes[source]
    rows = frames // ctx.bands
    plane = ctx.node("Transpose", [source], "plane", perm=[0, 1, 3, 2])
    bands = [
        ctx.slice(plane, TIME_AXIS, i * rows, (i + 1) * rows) for i in range(ctx.bands)
    ]
    value = ctx.node("Concat", bands, "bands", axis=0)
    for node in head:
        value = copy(ctx, node, [value, *node.input[1:]])
    return value


def fold_tail(
    ctx: FoldContext, final: onnx.NodeProto, value: str, target: str, output: str
) -> None:
    _, out_channels, frequencies, frames = ctx.shapes[output]
    rows = frames // ctx.bands
    weight = ctx.initializers[final.input[1]]
    channels = weight.shape[1]
    flat_shape = ctx.indices([ctx.bands, channels, rows * frequencies])
    flat = ctx.node("Reshape", [value, flat_shape], "flat")
    kernel = ctx.constant(weight.reshape(out_channels, channels), "output_weight")
    bias = ctx.initializers[final.input[2]].reshape(1, out_channels, 1)
    product = ctx.node("MatMul", [kernel, flat], "output_mm")
    biased = ctx.node("Add", [product, ctx.constant(bias, "output_bias")], "biased")
    bands = [ctx.slice(biased, 0, i, i + 1) for i in range(ctx.bands)]
    joined = ctx.node("Concat", bands, "joined", axis=TIME_AXIS)
    plane_shape = ctx.indices([1, out_channels, frames, frequencies])
    plane = ctx.node("Reshape", [joined, plane_shape], "plane")
    ctx.node("Transpose", [plane], "output", out=target, perm=[0, 1, 3, 2])


def fold(model: onnx.ModelProto, bands: int) -> onnx.ModelProto:
    for value in [*model.graph.input, *model.graph.output]:
        value.type.tensor_type.shape.dim[0].dim_value = 1
    inferred = shape_inference.infer_shapes(model).graph
    ctx = FoldContext(
        bands=bands,
        shapes={
            value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
            for value in [*inferred.input, *inferred.value_info, *inferred.output]
        },
        initializers={
            init.name: numpy_helper.to_array(init) for init in model.graph.initializer
        },
    )
    nodes = list(model.graph.node)
    transposes = [i for i, node in enumerate(nodes) if node.op_type == "Transpose"]
    first, last = transposes[0], transposes[-1]
    (final,) = nodes[last + 1 :]
    source = model.graph.input[0].name
    folded = {nodes[first].output[0]: fold_head(ctx, nodes[:first], source)}
    for node in nodes[first + 1 : last]:
        inputs = [folded.get(name, name) for name in node.input]
        if is_conv3(ctx, node):
            folded[node.output[0]] = conv3(ctx, node, inputs[0])
        elif node.op_type in ("Conv", "ConvTranspose") and pairs_frames(ctx, node):
            band_rows(ctx, node.input[0], 2 if node.op_type == "Conv" else 1)
            folded[node.output[0]] = copy(ctx, node, inputs)
        elif node.op_type in PASS_THROUGH:
            folded[node.output[0]] = copy(ctx, node, inputs)
        else:
            message = f"{node.op_type} {node.name}: not foldable into time bands"
            raise ValueError(message)
    output = model.graph.output[0].name
    fold_tail(ctx, final, folded[nodes[last].input[0]], output, output)

    used = {name for node in ctx.nodes for name in node.input}
    kept = [init for init in model.graph.initializer if init.name in used]
    graph = helper.make_graph(
        ctx.nodes,
        model.graph.name,
        list(model.graph.input),
        list(model.graph.output),
        initializer=[*kept, *ctx.constants],
    )
    result = helper.make_model(
        graph, opset_imports=model.opset_import, producer_name=__name__
    )
    result.ir_version = model.ir_version
    onnx.checker.check_model(result, full_check=True)
    return result


def check(source_path: Path, folded_path: Path, shape: list[int]) -> None:
    import onnxruntime as ort  # noqa: PLC0415

    count = int(np.prod(shape))
    values = ((np.arange(count) * 37 + 11) % 211) / 211 - 0.5
    feed = values.astype(np.float32).reshape(shape)
    outputs = []
    for path in (source_path, folded_path):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        outputs.append(session.run(None, {name: feed})[0])
    difference = float(np.max(np.abs(outputs[0] - outputs[1])))
    print(f"cpu max abs difference {difference:.3e}")
    if difference > CHECK_TOLERANCE:
        message = f"folded graph differs from the source by {difference}"
        raise SystemExit(message)


def main() -> None:
    args = parse_args()
    if args.bands < MIN_BANDS:
        message = f"--bands must be at least {MIN_BANDS}"
        raise SystemExit(message)
    model = onnx.load(args.model)
    shape = [
        1,
        *[dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim[1:]],
    ]
    result = fold(model, args.bands)
    onnx.save(result, args.out)
    size = Path(args.out).stat().st_size / 1e6
    print(f"saved {args.out}: {len(result.graph.node)} nodes, {size:.1f} MB")
    if args.check:
        check(Path(args.model), Path(args.out), shape)


if __name__ == "__main__":
    main()
