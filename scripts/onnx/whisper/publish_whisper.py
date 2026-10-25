"""Publish the Whisper large-v3-turbo (word-timestamped) ONNX repo to Hugging Face.

Uploads the staged `deps/whisper-large-v3-turbo-onnx/` folder (see stage_whisper.py):
the q4 encoder + merged decoder, the tokenizer/config JSON, and the README model
card. The `@musetric/ai` runtime loads it with `@huggingface/transformers` on the
onnxruntime-web WebGPU EP, so the repo is the FULL transformers.js layout.

Usage (requires `hf auth login` with write access to the musetric org):

    uv run python scripts/onnx/whisper/publish_whisper.py

    # dry run: verify + print the sha256 manifest, upload nothing
    uv run python scripts/onnx/whisper/publish_whisper.py --dry-run
"""

# ruff: noqa: T201

import argparse
import contextlib
import hashlib
import sys
from pathlib import Path

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

# Relative paths the @musetric/ai whisper runtime fetches. Order is display-only.
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
    "decoder_model_merged_q4.onnx",
]

# Workspace layout: <workspace>/lab/scripts/onnx/whisper/publish_whisper.py
# and <workspace>/models/whisper-large-v3-turbo-onnx — so parents[4] is <workspace>.
DEFAULT_SRC = (
    Path(__file__).resolve().parents[4] / "models" / "whisper-large-v3-turbo-onnx"
)
DEFAULT_REPO = "musetric/whisper-large-v3-turbo-onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(src: Path) -> list[tuple[str, int, str]]:
    missing = [rel for rel in PUBLISH_FILES if not (src / rel).is_file()]
    if missing:
        listing = "\n  ".join(missing)
        raise SystemExit(f"missing expected files under {src}:\n  {listing}")
    manifest: list[tuple[str, int, str]] = []
    for rel in PUBLISH_FILES:
        path = src / rel
        manifest.append((rel, path.stat().st_size, sha256(path)))
    return manifest


def print_manifest(manifest: list[tuple[str, int, str]]) -> None:
    total = 0
    print(
        "sha256                                                            "
        "     size  file"
    )
    for rel, size, digest in manifest:
        total += size
        print(f"{digest}  {size:>11}  {rel}")
    print(f"total: {total / 1e6:.0f} MB across {len(manifest)} files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify files + print the sha256 manifest, upload nothing",
    )
    args = parser.parse_args()

    src: Path = args.src
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")

    manifest = collect(src)
    print(f"source: {src}")
    print_manifest(manifest)

    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    print(f"uploading {len(manifest)} files (+ README.md) to {args.repo} ...")
    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(src),
        allow_patterns=[*PUBLISH_FILES, "README.md"],
        commit_message="Add Whisper large-v3 word-timestamped ONNX (q4)",
    )
    print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
