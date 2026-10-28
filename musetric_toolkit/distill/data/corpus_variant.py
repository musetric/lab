import csv
import logging
import re
from collections import Counter
from pathlib import Path

LOG = logging.getLogger(__name__)

SHAREALIKE = re.compile("share|-sa", re.IGNORECASE)
PELL_SOURCE = "ccMixter a cappella"
LEFT_OUT = {
    "old-sa": ("added", "pell"),
    "old": ("added", "pell", "sharealike"),
    "fma": ("pell", "sharealike"),
    "fma-pells": ("sharealike",),
}


def kinds(row: dict, added: set[str]) -> set[str]:
    out = set()
    if row["id"] in added:
        out.add("added")
    if row["source"] == PELL_SOURCE:
        out.add("pell")
    if SHAREALIKE.search(row["license"]):
        out.add("sharealike")
    return out


def variant_row(row: dict, added: set[str], name: str) -> dict:
    training = row["role"] == "train" and row["decision"] == "use"
    if training and kinds(row, added) & set(LEFT_OUT[name]):
        return {**row, "decision": "excluded", "reason": f"not in variant {name}"}
    return row


def write_variant(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def variants(args) -> None:
    with open(args.dataset, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    added = set(args.added.read_text(encoding="utf-8").split())
    args.out.mkdir(parents=True, exist_ok=True)
    for name in LEFT_OUT:
        variant = [variant_row(row, added, name) for row in rows]
        write_variant(args.out / f"dataset-{name}.csv", variant, columns)
        kept = [r for r in variant if r["role"] == "train" and r["decision"] == "use"]
        hours = sum(float(r["seconds"]) for r in kept) / 3600
        content = dict(Counter(r["content"] for r in kept).most_common())
        LOG.info(f"{name}: {len(kept)} training tracks, {hours:.1f} h; {content}")


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "data-variant",
        parents=[common],
        help="write the corpus variants: training rows a variant leaves out excluded",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="dataset.csv of the corpus",
    )
    parser.add_argument(
        "--added",
        required=True,
        type=Path,
        help="Ids of the clips an expansion added (fma-expand-<tag>.txt)",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory of the variants, dataset-<name>.csv",
    )
    parser.set_defaults(run=variants)
