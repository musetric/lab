import csv
import json
import logging
from collections import Counter
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

from musetric_toolkit.distill.release import (
    CONFIG,
    CREDIT_COLUMNS,
    TRAINING_DATA,
    WEIGHTS,
    credit_order,
)
from musetric_toolkit.distill.teacher import CHUNK, SAMPLE_RATE
from musetric_toolkit.distill.voices import INDEX, load_lead_backing, load_split
from musetric_toolkit.distill.voices_sampler import PELL_SHARE

LOG = logging.getLogger(__name__)

CORPUS_FILES = ("fma.json", "ccmixter-pells.json")
FMA_SOURCE = "FMA (fma_large, 30 s clip)"
PELL_SOURCE = "ccMixter a cappella"
USED_AS_MIX = "mix: lead and backing labelled by KARA 2 on the student's vocals"
USED_AS_PELL = "a cappella: lead and backing labelled by KARA 2"
JOINED = "; "


def trained_tracks(labels: Path) -> list[dict]:
    index = json.loads((labels / INDEX).read_text())
    return [
        t
        for t in index
        if t["role"] == "train"
        and t["seconds"] * SAMPLE_RATE > CHUNK + SAMPLE_RATE
        and (not t["pell"] or t["vocal_share"] >= PELL_SHARE)
    ]


def corpus_tracks(corpus: Path) -> dict[str, dict]:
    tracks = {}
    for name in CORPUS_FILES:
        for track in json.loads((corpus / name).read_text(encoding="utf-8")):
            tracks[track["id"]] = track
    return tracks


def voices_credit(entry: dict, track: dict) -> dict:
    return dict(
        source=PELL_SOURCE if entry["pell"] else FMA_SOURCE,
        fma_track_id=str(track["fmaId"]) if "fmaId" in track else "",
        artist=track["artist"],
        title=track["title"],
        license=track["license"],
        license_url=track["licenseUrl"],
        page=track.get("page") or "",
        used_as=USED_AS_PELL if entry["pell"] else USED_AS_MIX,
    )


def credit_key(row: dict) -> tuple:
    return (row["source"], row["fma_track_id"] or row["page"] or row["title"])


def merged_credits(student: Path, labels: Path, corpus: Path) -> list[dict]:
    with open(student / TRAINING_DATA, encoding="utf-8", newline="") as f:
        rows = {credit_key(row): row for row in csv.DictReader(f)}
    tracks = corpus_tracks(corpus)
    for entry in trained_tracks(labels):
        row = voices_credit(entry, tracks[entry["id"]])
        key = credit_key(row)
        if key in rows:
            rows[key]["used_as"] += JOINED + row["used_as"]
        else:
            rows[key] = row
    return sorted(rows.values(), key=credit_order)


def release_voices(args) -> None:
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = load_lead_backing(args.student, state["split_hidden"])
    load_split(model, state)
    args.out.mkdir(parents=True, exist_ok=True)
    weights = {k: v.float().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(weights, args.out / WEIGHTS)
    config = yaml.safe_load((args.student / CONFIG).read_text(encoding="utf-8"))
    config["model"]["split_hidden"] = state["split_hidden"]
    config["training"]["instruments"] = ["vocals", "backing", "other"]
    text = yaml.safe_dump(config, sort_keys=False, default_flow_style=None)
    (args.out / CONFIG).write_text(text, encoding="utf-8", newline="\n")
    rows = merged_credits(args.student, args.labels, args.corpus)
    with open(args.out / TRAINING_DATA, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CREDIT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    millions = sum(v.numel() for v in weights.values()) / 1e6
    sources = dict(Counter(row["source"] for row in rows).most_common())
    LOG.info(f"{millions:.2f} M parameters, {len(rows)} tracks {sources} -> {args.out}")


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "release-voices",
        parents=[common],
        help=f"the lead/backing model as {WEIGHTS}, {CONFIG} and {TRAINING_DATA}",
    )
    parser.add_argument(
        "--student",
        required=True,
        type=Path,
        help=f"Directory of the released student: {WEIGHTS}, {CONFIG}, {TRAINING_DATA}",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--labels", required=True, type=Path, help=f"Directory of {INDEX}"
    )
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help=f"Directory with {' and '.join(CORPUS_FILES)}",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.set_defaults(run=release_voices)
