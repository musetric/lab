import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from musetric_toolkit.distill.excluded import excluded_ids

LOG = logging.getLogger(__name__)

NO_VOICE = "no_voice"
TRAIN, TEST, VOICED, SKIP = "train", "test", "voiced", "skip"
COLUMNS = (
    "id",
    "start_s",
    "seconds",
    "round",
    "clip",
    "group",
    "answer",
    "use",
    "artist",
)
OLD_CONTENT = {"none": NO_VOICE, "one": "lead", "lead_backing": "lead_backing"}
UNSURE = "unsure"


def read_verdicts(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["start_s"] = float(row["start_s"])
        row["seconds"] = float(row["seconds"])
    return rows


def answer_of(label: dict) -> str:
    if label.get("unsure"):
        return UNSURE
    present = label.get("present")
    if present is None:
        return OLD_CONTENT.get(label.get("content"), UNSURE)
    voices = [part for part in ("lead", "backing") if part in present]
    if not voices:
        return NO_VOICE if present else UNSURE
    return "_".join(voices)


def early_answers(folder: Path, seconds: float) -> set[str]:
    log = folder / "labels.log.jsonl"
    if not log.exists():
        return set()
    events = [json.loads(line) for line in log.read_text("utf-8").splitlines()]
    early, closed, last = set(), set(), None
    for event in events:
        if not event["data"].get("done") or event["clip"] in closed:
            continue
        at = datetime.fromisoformat(event["at"])
        if last is not None and (at - last).total_seconds() < seconds:
            early.add(event["clip"])
        closed.add(event["clip"])
        last = at
    return early


def round_rows(
    folder: Path, doubtful: set[str], artists: dict[str, str], seconds: float
) -> list[dict]:
    labels = json.loads((folder / "labels.json").read_text("utf-8"))
    key = json.loads((folder / "key.json").read_text("utf-8"))
    excluded = excluded_ids()
    early = early_answers(folder, seconds)
    rows = []
    for clip, entry in key.items():
        if not entry["train"] or entry.get("repeat_of"):
            continue
        answer = answer_of(labels[clip])
        if answer == UNSURE or entry["source"] in excluded:
            continue
        known = f"{folder.name}:{clip}"
        rows.append(
            dict(
                id=entry["source"],
                start_s=entry["start_s"],
                seconds=seconds,
                round=folder.name,
                clip=clip,
                group=entry["sign"],
                answer=answer,
                use=SKIP if known in doubtful or clip in early else "",
                artist=artists[entry["source"]],
            )
        )
    return rows


def assign_uses(rows: list[dict], test_share: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    strata = {}
    for row in rows:
        if row["answer"] != NO_VOICE:
            row["use"] = VOICED
        elif row["use"] != SKIP:
            stratum = row["round"] if row["round"] == "round1" else row["group"]
            strata.setdefault(stratum, []).append(row)
    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda row: row["id"])
        order = rng.permutation(len(members))
        count = round(len(members) * test_share)
        for place, index in enumerate(order):
            members[index]["use"] = TEST if place < count else TRAIN


def withheld_ids(rows: list[dict], corpus: list[dict]) -> list[str]:
    artists = {row["artist"] for row in rows if row["use"] in (TEST, VOICED)}
    trained = {row["artist"] for row in rows if row["use"] == TRAIN}
    shared = artists & trained
    if shared:
        raise ValueError(f"artists on both sides of the split: {sorted(shared)}")
    return sorted(clip["id"] for clip in corpus if clip["artist"] in artists)


def build_verdicts(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(args.corpus.read_text("utf-8"))
    artists = {clip["id"]: clip["artist"] for clip in corpus}
    doubtful = set(args.doubtful)
    rows = []
    for folder in args.round:
        rows += round_rows(folder, doubtful, artists, args.clip_seconds)
    assign_uses(rows, args.test_share, args.seed)
    with open(args.out / "verdicts.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    withheld = withheld_ids(rows, corpus)
    (args.out / "withheld.txt").write_text("\n".join(withheld) + "\n", "utf-8")
    uses = Counter((row["use"], row["group"]) for row in rows)
    LOG.info(f"{len(rows)} verdicts, {len(withheld)} withheld tracks, {dict(uses)}")


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "verdicts",
        parents=[common],
        help="turn the answers of listening rounds into a split of clips",
    )
    parser.add_argument(
        "--round",
        required=True,
        type=Path,
        action="append",
        help="Folder of a closed round: key.json, labels.json, manifest.json",
    )
    parser.add_argument(
        "--corpus", required=True, type=Path, help="fma.json with id and artist"
    )
    parser.add_argument(
        "--doubtful",
        nargs="*",
        default=[],
        help="Answers kept out of the targets, as <round folder name>:<clip>",
    )
    parser.add_argument("--clip-seconds", type=float, default=8.0)
    parser.add_argument("--test-share", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument(
        "--out", required=True, type=Path, help="Output: verdicts.csv, withheld.txt"
    )
    parser.set_defaults(run=build_verdicts)
