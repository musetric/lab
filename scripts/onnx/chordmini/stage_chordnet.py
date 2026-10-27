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
    "chordnet.onnx",
    "cqt-plan.bin",
    "cqt-plan.manifest.json",
]


def _require_directory(path: Path, option: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{option} is not a directory: {path}")


def _require_file(path: Path, option: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{option} is not a file: {path}")


def _require_export_files(export: Path, files: list[str]) -> None:
    missing = [rel for rel in files if not (export / rel).is_file()]
    if missing:
        listing = "\n  ".join(missing)
        raise SystemExit(f"missing exported files under {export}:\n  {listing}")


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".json":
        target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    else:
        shutil.copy2(source, target)
    print(f"staged {target.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chordnet-export",
        type=Path,
        required=True,
        help="directory containing chordnet.onnx and config.json",
    )
    parser.add_argument(
        "--cqt-plan",
        type=Path,
        required=True,
        help="binary CQT plan generated for the ChordNet artifact",
    )
    parser.add_argument(
        "--cqt-plan-manifest",
        type=Path,
        required=True,
        help="JSON manifest corresponding to --cqt-plan",
    )
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    chordnet_export: Path = args.chordnet_export
    _require_directory(chordnet_export, "--chordnet-export")
    _require_export_files(chordnet_export, ["config.json", "chordnet.onnx"])
    _require_file(args.cqt_plan, "--cqt-plan")
    _require_file(args.cqt_plan_manifest, "--cqt-plan-manifest")

    sources = [
        (chordnet_export / "config.json", "config.json"),
        (chordnet_export / "chordnet.onnx", "chordnet.onnx"),
        (args.cqt_plan, "cqt-plan.bin"),
        (args.cqt_plan_manifest, "cqt-plan.manifest.json"),
    ]
    for source, rel in sources:
        _copy(source, args.dest / rel)

    print(f"staged {len(PUBLISH_FILES)} files into {args.dest}")


if __name__ == "__main__":
    main()
