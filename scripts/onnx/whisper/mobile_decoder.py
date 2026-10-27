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
