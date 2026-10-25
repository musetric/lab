"""Stage the published Whisper file set from an export dir into the deps repo.

`convert.py` writes many graphs (fp32 + q4, all decoder variants); the publish
repo needs only the q4 encoder, the fp16 step decoder with its `cross_kv`
projection, the merged q4 decoder and the tokenizer/config JSON.
This copies exactly that set into `deps/whisper-large-v3-turbo-onnx/` (the folder
`publish_whisper.py` uploads).

The encoder is checked against `mobile_encoder.verify` and both decoders against
`mobile_decoder.verify` on the way through: a stock export cannot run on several
mobile GPUs and returns wrong answers on others, and the faults are silent at
runtime. Staging refuses rather than let such a graph reach the publish repo.

    uv run python scripts/onnx/whisper/stage_whisper.py \
      --export tmp/whisper-export/openai/whisper-large-v3-turbo
"""

# ruff: noqa: T201

import argparse
import contextlib
import shutil
import sys
from pathlib import Path

import mobile_decoder
import mobile_encoder
import onnx

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

ENCODER = "encoder_model_q4.onnx"
DECODERS = ["decoder_model_merged_fp16.onnx", "decoder_model_merged_q4.onnx"]

# Workspace layout: <workspace>/lab/scripts/onnx/whisper/stage_whisper.py
# and <workspace>/models/whisper-large-v3-turbo-onnx — so parents[4] is <workspace>.
DEFAULT_DEST = (
    Path(__file__).resolve().parents[4] / "models" / "whisper-large-v3-turbo-onnx"
)
DEFAULT_EXPORT = Path("tmp/whisper-export/openai/whisper-large-v3-turbo")

PUBLISH_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "encoder_model_q4.onnx",
    "decoder_model_merged_fp16.onnx",
    "cross_kv_fp16.onnx",
    "decoder_model_merged_q4.onnx",
]


def source_path(export: Path, rel: str) -> Path:
    # convert.py leaves the graphs in `onnx/` and the JSON at the export root,
    # while the publish repo is flat.
    nested = export / "onnx" / rel
    return nested if nested.is_file() else export / rel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    export: Path = args.export
    dest: Path = args.dest
    missing = [rel for rel in PUBLISH_FILES if not source_path(export, rel).is_file()]
    if missing:
        listing = "\n  ".join(missing)
        raise SystemExit(f"missing exported files under {export}:\n  {listing}")

    encoder = source_path(export, ENCODER)
    reasons = mobile_encoder.verify(onnx.load(str(encoder)))
    if reasons:
        listing = "\n  ".join(reasons)
        raise SystemExit(
            f"{encoder} would not run correctly on a mobile GPU:\n  {listing}\n"
            f"run mobile_encoder.py over it before staging"
        )
    print(f"checked {ENCODER}: mobile-safe")

    for name in DECODERS:
        decoder = source_path(export, name)
        reasons = mobile_decoder.verify(onnx.load(str(decoder)))
        if reasons:
            listing = "\n  ".join(reasons)
            raise SystemExit(
                f"{decoder} would compute wrong on a mobile GPU:\n  {listing}\n"
                f"run mobile_decoder.py over it before staging"
            )
        print(f"checked {name}: mobile-safe")

    for rel in PUBLISH_FILES:
        source = source_path(export, rel)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".onnx"):
            shutil.copy2(source, target)
        else:
            # Normalize text to LF so the local files match what Hugging Face
            # serves (the hub normalizes text blobs to LF on commit); keeps the
            # sha256 manifest and the @musetric/ai descriptor consistent.
            target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
        print(f"staged {rel}")
    print(f"staged {len(PUBLISH_FILES)} files into {dest}")


if __name__ == "__main__":
    main()
