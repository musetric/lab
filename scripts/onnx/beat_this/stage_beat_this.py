# ruff: noqa: T201

import argparse
import contextlib
import shutil
import sys
from pathlib import Path

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

PUBLISH_FILES = [
    "beat_this.onnx",
    "config.json",
    "mel-filterbank.bin",
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
        if rel.endswith(".json"):
            target.write_bytes((export / rel).read_bytes().replace(b"\r\n", b"\n"))
        else:
            shutil.copy2(export / rel, target)
        print(f"staged {rel}")
    print(f"staged {len(PUBLISH_FILES)} files into {dest}")


if __name__ == "__main__":
    main()
