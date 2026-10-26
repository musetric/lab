"""Make the merged Whisper decoder compute correctly on mobile GPUs.

On Adreno 600-series adapters the WebGPU `MatMul` kernel that packs its operands
into vec4 returns wrong values when the left operand has few rows. A decoder
multiplies exactly such operands in every attention score - one query row per
cached step, the prompt on the first step - and the key length reaches a
multiple of four on every fourth self-attention step and on every
cross-attention step. `pad_attention_scores` keeps those products off the packed
kernel. The defect is musetric/musetric#886; musetric/musetric#975 lists it
with the GPU defects the models work around.

The rewrite applies to both merged decoders, float16 and q4, and is idempotent.
`stage_whisper.py` runs the same check before it stages a decoder.

    uv run python scripts/onnx/whisper/mobile_decoder.py \
      --input  decoder_model_merged_fp16.onnx \
      --output decoder_model_merged_fp16.onnx
"""

# ruff: noqa: T201

import argparse
import contextlib
import sys
from pathlib import Path

import onnx
import pad_attention_scores

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")


def verify(model):
    """Report what would still compute wrong on a mobile GPU, as a list of reasons."""
    scores = pad_attention_scores.count_affected(model)
    if scores:
        return [
            f"{scores} attention score products still reach the packed MatMul "
            f"kernel: run pad_attention_scores"
        ]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    print("== pad_attention_scores")
    pad_attention_scores.apply(model)

    reasons = verify(model)
    if reasons:
        listing = "\n  ".join(reasons)
        raise SystemExit(f"the decoder is still not mobile-safe:\n  {listing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
