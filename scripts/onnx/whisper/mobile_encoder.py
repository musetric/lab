"""Make the exported Whisper encoder run, and run correctly, on mobile GPUs.

Two independent defects stand between the stock export and a phone, and neither
is in the weights. This applies the three graph rewrites that clear them, in
order, on a single load of the graph, then checks that the result actually is
clear. `stage_whisper.py` runs the same check before it stages anything, so an
encoder that skipped this step cannot reach the model repo.

**The graph is too big to bind.** Each layer computes attention in one shot, so
its score tensor is `(20, 1500, 1500)` fp32 = 171.7 MiB, while WebGPU only
guarantees 128 MiB per storage binding and several mobile adapters offer exactly
that minimum. The encoder does not run there at all. `block_attention` splits
the queries into blocks and concatenates the per-block outputs, which is exact
because softmax normalizes each query row over the full key axis on its own.

**The graph computes the wrong answer.** On Adreno 600-series adapters the
WebGPU `Transpose` returns a quarter of its output as zeros - silently, with no
NaN and no error. The cause is in the execution provider's shader: it stages the
tile in `array<array<T, tile_size + 1>, tile_size>`, 16 x 17 f32 = 1088 bytes,
and on that hardware a workgroup array whose size is not a multiple of 512 bytes
loses the stores of one warp out of four across `workgroupBarrier`. The stores
land - a thread reads its own slot back correctly - but the other threads do not
see them. `conv_to_matmul` and `transpose_to_gather` route the encoder around
every shape that reaches that shader. The defect is kept in
musetric/musetric#975.

    uv run python scripts/onnx/whisper/mobile_encoder.py \
      --input  tmp/whisper-export/.../onnx/encoder_model_q4.onnx \
      --output tmp/whisper-export/.../onnx/encoder_model_q4.onnx
"""

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
    """Report what would still break on a mobile GPU, as a list of reasons."""
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
