import json
import logging
from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.distill.precision import PRECISION
from musetric_toolkit.distill.teacher import SAMPLE_RATE
from musetric_toolkit.distill.verdicts import (
    NO_VOICE,
    TEST,
    TRAIN,
    VOICED,
    read_verdicts,
)
from musetric_toolkit.distill.voices import load_lead_backing
from musetric_toolkit.distill.voices_scores import (
    STEMS,
    level_db,
    load_label,
    separate_stems,
)

LOG = logging.getLogger(__name__)

FALSE_VOICE_DB = (-15.0, -20.0, -25.0)
BOOT = 10000
WORST = 5


def clip_levels(model, labels: Path, row: dict) -> dict:
    mix = load_label(labels, row["id"], "mix")
    stems = separate_stems(model, mix)
    start = int(row["start_s"] * SAMPLE_RATE)
    end = start + int(row["seconds"] * SAMPLE_RATE)
    whole = mix[:, start:end]
    return {stem: level_db(stems[stem][:, start:end], whole) for stem in STEMS}


def score_verdicts(args) -> None:
    PRECISION.dtype = torch.float16
    model = load_lead_backing(args.model).cuda().eval()
    report = {}
    for row in read_verdicts(args.verdicts):
        report[f"{row['round']}:{row['clip']}"] = clip_levels(model, args.labels, row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), "utf-8")
    LOG.info(f"{len(report)} clips -> {args.out}")


def interval(values: np.ndarray, rng) -> str:
    medians = np.median(rng.choice(values, (BOOT, len(values))), axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return f"{np.median(values):+.2f} [{low:+.2f}, {high:+.2f}]"


def compare_silent(rows: list[dict], name: str, stem: str, rng) -> None:
    before = np.array([row["before"][stem] for row in rows])
    after = np.array([row["after"][stem] for row in rows])
    change = after - before
    LOG.info(
        f"{name}: n={len(rows)} {stem} median {np.median(before):.1f} -> "
        f"{np.median(after):.1f}, change {interval(change, rng)}, "
        f"down {int((change < 0).sum())} up {int((change > 0).sum())}"
    )
    for limit in FALSE_VOICE_DB:
        LOG.info(
            f"  over {limit} dB: {int((before >= limit).sum())} -> "
            f"{int((after >= limit).sum())}"
        )
    order = np.argsort(-change)[:WORST]
    worst = [(rows[i]["key"], round(float(change[i]), 2)) for i in order]
    LOG.info(f"  least improved: {worst}")


def compare_verdicts(args) -> None:
    before = json.loads(args.baseline.read_text("utf-8"))
    after = json.loads(args.candidate.read_text("utf-8"))
    rows = []
    for row in read_verdicts(args.verdicts):
        key = f"{row['round']}:{row['clip']}"
        rows.append(dict(row, key=key, before=before[key], after=after[key]))
    rng = np.random.default_rng(args.seed)
    silent = [row for row in rows if row["answer"] == NO_VOICE]
    for use in (TEST, TRAIN):
        chosen = [row for row in silent if row["use"] == use]
        compare_silent(chosen, f"no voice, {use}", "vocals", rng)
        for group in sorted({row["group"] for row in chosen}):
            compare_silent(
                [row for row in chosen if row["group"] == group],
                f"  {use}, {group}",
                "vocals",
                rng,
            )
    voiced = [row for row in rows if row["use"] == VOICED]
    for stem in ("vocals", "lead", "backing"):
        compare_silent(voiced, "voiced", stem, rng)
    changes = {
        row["key"]: round(row["after"]["vocals"] - row["before"]["vocals"], 2)
        for row in voiced
    }
    lost = {key: value for key, value in changes.items() if value < args.lost_db}
    LOG.info(f"voiced, under {args.lost_db} dB: {lost}")


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "score-verdicts",
        parents=[common],
        help="level of each stem of a model on the clips of a verdicts.csv",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Directory of a release: config.yaml, model.safetensors",
    )
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--verdicts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.set_defaults(run=score_verdicts)

    compare = commands.add_parser(
        "compare-verdicts",
        parents=[common],
        help="paired change of the levels of two models on a verdicts.csv",
    )
    compare.add_argument("--verdicts", required=True, type=Path)
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--lost-db", type=float, default=-3.0)
    compare.add_argument("--seed", type=int, default=20261105)
    compare.set_defaults(run=compare_verdicts)
