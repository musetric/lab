import argparse
from pathlib import Path

CROPS = 24


def add_data(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--teacher",
        required=True,
        type=Path,
        help="Directory with the teacher's config.yaml and model.ckpt",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Directory of the teacher's labels and labels.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="dataset.csv of the corpus: decision, role, content and genre per track",
    )
    parser.add_argument(
        "--crops",
        type=int,
        default=CROPS,
        help="Training crops the activations are gathered on",
    )


def add_student(
    parser: argparse.ArgumentParser, help_text: str, default: str | None = None
) -> None:
    parser.add_argument(
        "--student", required=default is None, default=default, help=help_text
    )
    parser.add_argument(
        "--fits",
        type=Path,
        help="JSON of the fit files the student's modifiers read",
    )


def add_out(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument("--out", required=True, type=Path, help=help_text)
