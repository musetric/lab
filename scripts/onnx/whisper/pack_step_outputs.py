# ruff: noqa: T201

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

PACKED = "step_outputs"
PREFIX = "pack_step_outputs"
LOGITS = "logits"
CROSS_ATTENTIONS = "cross_attentions."


def packed_sources(model, layers):
    names = {value.name for value in model.graph.output}
    sources = [LOGITS] + [f"{CROSS_ATTENTIONS}{layer}" for layer in sorted(layers)]
    missing = [name for name in sources if name not in names]
    if missing:
        message = f"the decoder has no output named {', '.join(missing)}"
        raise ValueError(message)
    return sources


def apply(model, layers):
    graph = model.graph
    if any(value.name == PACKED for value in graph.output):
        print(f"{PACKED} is already there; nothing to do")
        return []
    sources = packed_sources(model, layers)
    flat = f"{PREFIX}/flat"
    graph.initializer.append(
        numpy_helper.from_array(np.array([-1], dtype=np.int64), flat)
    )
    flattened = []
    for name in sources:
        target = f"{PREFIX}/{name}"
        graph.node.append(
            helper.make_node("Reshape", [name, flat], [target], name=target)
        )
        flattened.append(target)
    graph.node.append(
        helper.make_node("Concat", flattened, [PACKED], name=PACKED, axis=0)
    )
    graph.output.append(
        helper.make_tensor_value_info(PACKED, TensorProto.FLOAT, ["packed_length"])
    )
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.generation_config.read_text(encoding="utf-8"))
    layers = {layer for layer, _ in config["alignment_heads"]}
    model = onnx.load(str(args.input))
    sources = apply(model, layers)
    if sources:
        print(f"{PACKED} = concat({', '.join(sources)})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
