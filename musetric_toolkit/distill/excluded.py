import csv
from pathlib import Path

LIST = Path(__file__).with_name("excluded_tracks.csv")
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


def excluded_rows(path: Path = LIST) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if tuple(reader.fieldnames) != COLUMNS:
        message = f"{path} must have the columns {COLUMNS}, not {reader.fieldnames}"
        raise ValueError(message)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} lists a track twice")
    for row in rows:
        empty = [column for column in REQUIRED if not row[column]]
        if empty:
            raise ValueError(f"{path}: {row['id']} has no {empty}")
    return rows


def excluded_ids() -> set[str]:
    return {row["id"] for row in excluded_rows()}


def without_excluded(tracks: list[dict]) -> list[dict]:
    excluded = excluded_ids()
    return [track for track in tracks if track["id"] not in excluded]


if __name__ == "__main__":
    excluded_rows()
