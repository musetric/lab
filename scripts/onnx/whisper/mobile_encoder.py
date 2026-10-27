# ruff: noqa: T201

import argparse
import contextlib
import sys
from pathlib import Path

import block_attention
import conv_to_matmul
import onnx
import transpose_to_gather

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")


def verify(model, query_block=block_attention.QUERY_BLOCK):
    reasons = []

    score_bytes = block_attention.largest_score_bytes(model)
    if score_bytes > block_attention.BINDING_LIMIT:
        reasons.append(
            f"largest attention score tensor is {score_bytes / 1048576:.1f} MiB, "
            f"over the {block_attention.BINDING_LIMIT / 1048576:.0f} MiB WebGPU "
            f"binding floor; run block_attention with --query-block {query_block}"
        )

    convolutions = sum(1 for node in model.graph.node if node.op_type == "Conv")
    if convolutions:
        reasons.append(
            f"Conv is still present on {convolutions} of the graph's nodes: "
            f"the provider transposes NCHW to NHWC inside it, so run "
            f"conv_to_matmul"
        )

    transposes = transpose_to_gather.count_affected(model)
    if transposes:
        reasons.append(
            f"Transpose still hits the broken shape on {transposes} of the "
            f"graph's nodes: run transpose_to_gather"
        )

    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--query-block",
        type=int,
        default=block_attention.QUERY_BLOCK,
        help="query rows per attention block",
    )
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    nodes = len(model.graph.node)

    print("== block_attention")
    block_attention.apply(model, args.query_block)
    print("== conv_to_matmul")
    conv_to_matmul.apply(model)
    print("== transpose_to_gather")
    transpose_to_gather.apply(model)

    reasons = verify(model, args.query_block)
    if reasons:
        listing = "\n  ".join(reasons)
        raise SystemExit(f"the encoder is still not mobile-safe:\n  {listing}")

    print(f"== nodes: {nodes} -> {len(model.graph.node)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
