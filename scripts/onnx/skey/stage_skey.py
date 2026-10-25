"""Stage the exported S-KEY file set for publishing.

Copies the ONNX graph + config.json from the export dir into the clone of the
model repository. The ONNX is copied verbatim; JSON text is normalized to LF so
the staged files match what Hugging Face serves, keeping them consistent with
the sizes and hashes the server pins.

    uv run python scripts/onnx/skey/stage_skey.py \
      --export <export-dir> --dest ../models/skey-onnx
"""

# ruff: noqa: T201

import argparse
import contextlib
import shutil
import sys
from pathlib import Path

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

PUBLISH_FILES = [
    "config.json",
    "skey.onnx",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    export: Path = args.export
    dest: Path = args.dest
    missing = [rel for rel in PUBLISH_FILES if not (export / rel).is_file()]
    if missing:
        listing = "\n  ".join(missing)
        raise SystemExit(f"missing exported files under {export}:\n  {listing}")

    for rel in PUBLISH_FILES:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".onnx"):
            shutil.copy2(export / rel, target)
        else:
            target.write_bytes((export / rel).read_bytes().replace(b"\r\n", b"\n"))
        print(f"staged {rel}")
    print(f"staged {len(PUBLISH_FILES)} files into {dest}")


if __name__ == "__main__":
    main()
