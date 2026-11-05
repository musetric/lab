import argparse
import csv
from pathlib import Path

COLUMNS = (
    "id",
    "fma_track_id",
    "artist",
    "title",
    "license",
    "license_url",
    "page",
    "reason",
    "date",
)
REQUIRED = ("id", "artist", "title", "reason", "date")


def read_excluded(path: Path) -> frozenset[str]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = tuple(reader.fieldnames or ())
        rows = list(reader)
    if columns != COLUMNS:
        message = f"{path} must have the columns {COLUMNS}, not {columns}"
        raise ValueError(message)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} lists a track twice")
    for row in rows:
        empty = [column for column in REQUIRED if not row[column]]
        if empty:
            raise ValueError(f"{path}: {row['id']} has no {empty}")
    return frozenset(ids)


def add_excluded(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--excluded",
        required=True,
        type=Path,
        help="CSV of the tracks a listener took out of the data for good",
    )
