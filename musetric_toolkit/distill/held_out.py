import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from musetric_toolkit.distill.corpus import dataset_rows, held_tracks, load, usable
from musetric_toolkit.distill.inference import infer_overlap
from musetric_toolkit.distill.scores import rms, sdr
from musetric_toolkit.distill.student import load_student
from musetric_toolkit.distill.teacher import SAMPLE_RATE

LOG = logging.getLogger(__name__)

CLIPS = {
    "jw-dust",
    "ccm-winter-sunlight",
    "jamalt-en-rxbyn-bad-side",
    "argonov-2032-10-the-angel-of-the-new-life-milinevskys-aria",
}
QUICK_SECONDS = 30
CLIP_START = 60
ACTIVITY_BINS = [(0.0, 0.4), (0.4, 0.8), (0.8, 1.01)]
FMA = "fma:"
ARIA_CAP = 3


def loudest_span(mix: np.ndarray, voc: np.ndarray, seconds: int):
    span = seconds * SAMPLE_RATE
    if mix.shape[1] <= span:
        return mix, voc
    whole = mix.shape[1] // SAMPLE_RATE
    energy = (voc[:, : whole * SAMPLE_RATE] ** 2).sum(0)
    energy = energy.reshape(whole, SAMPLE_RATE).sum(1)
    start = int(np.argmax(np.convolve(energy, np.ones(seconds), "valid"))) * SAMPLE_RATE
    return mix[:, start : start + span], voc[:, start : start + span]


def quick_set(labels: Path, held: list[dict]):
    chosen, per_group = [], {}
    for t in held:
        cap = 1 if t["group"].startswith(FMA) else ARIA_CAP
        if per_group.get(t["group"], 0) < cap:
            per_group[t["group"]] = per_group.get(t["group"], 0) + 1
            chosen.append(t)
    items = []
    for t in chosen:
        mix = load(labels, t["id"], "mix")
        voc = load(labels, t["id"], "vocals")
        mix, voc = loudest_span(mix, voc, QUICK_SECONDS)
        items.append((t["id"], t["group"], mix, voc))
    return items


def quick_eval(model, items) -> dict:
    sdrs, levels = [], []
    for _, _, mix, teacher in items:
        student = infer_overlap(model, mix)
        sdrs.append(sdr(teacher, student))
        levels.append(rms(student) / max(rms(teacher), 1e-8))
    return dict(
        quick_sdr_median=float(np.median(sdrs)),
        quick_sdr_min=float(np.min(sdrs)),
        quick_level_median=float(np.median(levels)),
    )


def clip_ids(held: list[dict]) -> set[str]:
    first = {}
    for t in held:
        if t["group"].startswith(FMA):
            first.setdefault(t["group"], t["id"])
    return CLIPS | set(first.values())


def clip_start(mix: np.ndarray) -> int:
    return int(
        min(
            CLIP_START * SAMPLE_RATE, max(0, mix.shape[1] - QUICK_SECONDS * SAMPLE_RATE)
        )
    )


def evaluate(model, labels: Path, held: list[dict], clips: Path | None) -> dict:
    listen = clip_ids(held)
    result = {}
    for t in held:
        mix = load(labels, t["id"], "mix")
        teacher = load(labels, t["id"], "vocals")
        student = infer_overlap(model, mix)
        row = dict(group=t["group"], sdr_teacher=sdr(teacher, student))
        if "teacher_sdr_truth" in t:
            truth = load(labels, t["id"], "truth")
            row["sdr_truth"] = sdr(truth, student)
            row["teacher_sdr_truth"] = t["teacher_sdr_truth"]
        result[t["id"]] = row
        if clips is not None and t["id"] in listen:
            a = clip_start(mix)
            part = student[:, a : a + QUICK_SECONDS * SAMPLE_RATE].T
            sf.write(clips / f"{t['id']}.student.flac", part, SAMPLE_RATE)
    return result


def medians(groups: dict[str, list[float]]) -> dict:
    return {
        g: dict(n=len(v), median=float(np.median(v))) for g, v in sorted(groups.items())
    }


def summary(evals: dict, rows: dict[str, dict]) -> dict:
    arias = [r for r in evals.values() if "sdr_truth" in r]
    groups, artists, activity = {}, {}, {}
    for name, r in evals.items():
        groups.setdefault(r["group"], []).append(r["sdr_teacher"])
        row = rows.get(name, {})
        artists.setdefault(row.get("artist", name), []).append(r["sdr_teacher"])
        active = float(row.get("vocal_active_share") or "nan")
        for lo, hi in ACTIVITY_BINS:
            if lo <= active < hi:
                key = f"{lo:.1f}-{min(hi, 1.0):.1f}"
                activity.setdefault(key, []).append(r["sdr_teacher"])
    fma = [r["sdr_teacher"] for r in evals.values() if r["group"].startswith(FMA)]
    truth = [r["sdr_truth"] for r in arias]
    teacher_truth = [r["teacher_sdr_truth"] for r in arias]
    return dict(
        sdr_teacher_median=float(np.median([r["sdr_teacher"] for r in evals.values()])),
        fma_sdr_teacher_median=float(np.median(fma)) if fma else None,
        artist_balanced_median=float(
            np.median([np.median(v) for v in artists.values()])
        ),
        artists=len(artists),
        sdr_truth_median=float(np.median(truth)) if arias else None,
        teacher_sdr_truth_median=float(np.median(teacher_truth)) if arias else None,
        by_activity=medians(activity),
        by_group=medians(groups),
    )


def write_reference_clips(labels: Path, out: Path, held: list[dict]) -> None:
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    listen = clip_ids(held)
    span = QUICK_SECONDS * SAMPLE_RATE
    for t in held:
        if t["id"] not in listen:
            continue
        mix = load(labels, t["id"], "mix")
        a = clip_start(mix)
        sf.write(clips / f"{t['id']}.mix.flac", mix[:, a : a + span].T, SAMPLE_RATE)
        teacher = load(labels, t["id"], "vocals")[:, a : a + span].T
        sf.write(clips / f"{t['id']}.teacher.flac", teacher, SAMPLE_RATE)


def held_or_train(index: list[dict], rows: dict, split: str, limit: int):
    if split != "train":
        return held_tracks(index)
    voiced = [
        dict(t, role="held")
        for t in index
        if t["role"] == "train" and rows.get(t["id"], {}).get("content") == "vocal"
    ]
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(voiced), min(limit, len(voiced)), replace=False)
    return [voiced[i] for i in sorted(chosen)]


def score_checkpoint(args) -> None:
    model, state = load_student(args.checkpoint, args.teacher, args.fits, "cuda")
    rows = dataset_rows(args.dataset)
    index = usable(json.loads((args.labels / "labels.json").read_text()), rows)
    held = held_or_train(index, rows, args.split, args.limit)
    args.out.mkdir(parents=True, exist_ok=True)
    write_reference_clips(args.labels, args.out, held)
    evals = evaluate(model, args.labels, held, args.out / "clips")
    row = dict(
        checkpoint=str(args.checkpoint),
        step=state["step"],
        minutes=state["elapsed"] / 60,
        **summary(evals, rows),
        tracks=evals,
    )
    (args.out / "eval.json").write_text(json.dumps(row, indent=1))
    LOG.info(json.dumps({k: v for k, v in row.items() if k != "tracks"}))
