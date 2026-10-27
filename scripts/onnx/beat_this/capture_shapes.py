# ruff: noqa: T201

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="dynamic source .onnx")
    parser.add_argument("--frames", required=True, type=int, help="frames dim to pin")
    parser.add_argument("--out", required=True, help="output shapes .json")
    parser.add_argument(
        "--probe", default="probe_all.onnx", help="temp probe model path"
    )
    parser.add_argument("--input", default="spect", help="graph input name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(args.model)
    for dim in model.graph.input[0].type.tensor_type.shape.dim:
        if dim.HasField("dim_param"):
            value = 1 if dim.dim_param == "windows" else args.frames
            dim.dim_param = ""
            dim.dim_value = value
    graph = model.graph
    existing = {output.name for output in graph.output}
    for node in graph.node:
        for out in node.output:
            if out and out not in existing:
                graph.output.append(helper.make_empty_tensor_value_info(out))
                existing.add(out)
    onnx.save(model, args.probe)

    session = ort.InferenceSession(args.probe, providers=["CPUExecutionProvider"])
    frames = args.frames
    feed = (
        np.random.default_rng(7).standard_normal((1, frames, 128), dtype=np.float32)
        * 0.5
    )
    names = [output.name for output in session.get_outputs()]
    values = session.run(None, {args.input: feed})
    shapes = {
        name: list(np.asarray(value).shape)
        for name, value in zip(names, values, strict=True)
    }
    shapes[args.input] = [1, frames, 128]
    Path(args.out).write_text(json.dumps(shapes), encoding="utf-8")
    print(f"captured {len(shapes)} tensor shapes -> {args.out}")


if __name__ == "__main__":
    main()
