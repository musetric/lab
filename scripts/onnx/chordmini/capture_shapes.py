# ruff: noqa: T201
"""Capture real runtime shapes of a dynamic ONNX graph's interesting tensors.

Runs the model once in ORT CPU with selected intermediates exposed as outputs
and records the observed tensor shapes. ONNX shape inference cannot resolve
the shapes of graphs whose Reshape nodes read `Shape` outputs at runtime, so
this forward pass is the reliable source for static rewrites.

By default every intermediate is exposed; graphs that fail to load that way
can restrict the probe to the inputs of specific op types with --ops.

The shapes are captured for one batch, --windows, and a rewrite built from them
only runs at that batch.

Usage:
    uv run python capture_shapes.py --model beat_this.onnx --frames 1500 \
        --out shapes1500.json [--probe probe_all.onnx] [--ops Transpose,Conv] \
        [--windows 16]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="dynamic source .onnx")
    parser.add_argument("--frames", required=True, type=int, help="frames dim to pin")
    parser.add_argument(
        "--windows",
        default=1,
        type=int,
        help="windows (batch) dim to pin. The rewrite bakes every index for this "
        "batch, so it is the batch the runtime has to feed",
    )
    parser.add_argument("--out", required=True, help="output shapes .json")
    parser.add_argument(
        "--probe", default="probe_all.onnx", help="temp probe model path"
    )
    parser.add_argument("--input", default="spect", help="graph input name")
    parser.add_argument(
        "--shape",
        default="",
        help="comma-separated feed shape, e.g. 1,108,144 "
        "(default: 1,<frames>,128 for the beat_this mel layout)",
    )
    parser.add_argument(
        "--ops",
        default="",
        help="comma-separated op types; expose only the inputs of these nodes "
        "(empty = every intermediate, which some graphs cannot load)",
    )
    return parser.parse_args()


def pin_windows_dim(model: onnx.ModelProto, windows: int, frames: int) -> None:
    for dim in model.graph.input[0].type.tensor_type.shape.dim:
        if dim.HasField("dim_param"):
            value = windows if dim.dim_param == "windows" else frames
            dim.dim_param = ""
            dim.dim_value = value


def expose_probe_outputs(model: onnx.ModelProto, watch_ops: set[str]) -> None:
    """Infer shapes and declare the watched node inputs as graph outputs.

    Outputs are declared with an unknown rank: dims taken from inference can
    disagree with ORT's own plan and abort the load.
    """
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        model.CopyFrom(inferred)
    except Exception as error:
        print(f"shape inference partial failure: {error}")
    value_types = {
        value.name: value
        for value in list(model.graph.value_info) + list(model.graph.input)
    }
    existing = {output.name for output in model.graph.output}
    skipped = 0
    for node in model.graph.node:
        if watch_ops and node.op_type not in watch_ops:
            continue
        for name in node.input:
            if not name or name in existing:
                continue
            value = value_types.get(name)
            if value is None or not value.HasField("type"):
                skipped += 1
                continue
            tensor_type = value.type.tensor_type
            if not tensor_type.HasField("elem_type"):
                skipped += 1
                continue
            output = onnx.ValueInfoProto()
            output.name = name
            output.type.tensor_type.elem_type = tensor_type.elem_type
            model.graph.output.append(output)
            existing.add(name)
    if skipped:
        print(f"skipped {skipped} tensors without inferred type")


def capture_shapes(
    session: ort.InferenceSession, feed: np.ndarray, input_name: str
) -> dict[str, list[int]]:
    names = [output.name for output in session.get_outputs()]
    values = session.run(None, {input_name: feed})
    shapes = {
        name: list(np.asarray(value).shape)
        for name, value in zip(names, values, strict=True)
    }
    shapes[input_name] = list(feed.shape)
    return shapes


def main() -> None:
    args = parse_args()
    model = onnx.load(args.model)
    feed_shape = (
        [int(d) for d in args.shape.split(",")]
        if args.shape
        else [args.windows, args.frames, 128]
    )
    if feed_shape[0] != args.windows:
        raise SystemExit(
            f"--shape batch {feed_shape[0]} does not match --windows {args.windows}"
        )
    pin_windows_dim(model, args.windows, args.frames)
    watch_ops = {op for op in args.ops.split(",") if op}
    expose_probe_outputs(model, watch_ops)
    onnx.save(model, args.probe)

    session = ort.InferenceSession(args.probe, providers=["CPUExecutionProvider"])
    feed = np.random.default_rng(7).standard_normal(feed_shape, dtype=np.float32) * 0.5
    shapes = capture_shapes(session, feed, args.input)
    Path(args.out).write_text(json.dumps(shapes), encoding="utf-8")
    print(f"captured {len(shapes)} tensor shapes -> {args.out}")


if __name__ == "__main__":
    main()
