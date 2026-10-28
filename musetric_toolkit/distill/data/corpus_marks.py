import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from musetric_toolkit.distill.corpus import FULL_SCALE, INT16_MAX, dataset_rows, load
from musetric_toolkit.distill.data.corpus_disagreement import (
    VOICED_DB,
    WINDOW,
    WINDOW_SECONDS,
)
from musetric_toolkit.distill.inference import infer_overlap
from musetric_toolkit.distill.student import load_student
from musetric_toolkit.distill.teacher import SAMPLE_RATE

LOG = logging.getLogger(__name__)

SHARE = 0.10
LEAK_DB = -30.0
BASS_DB = -6.0
LISTEN_TOP = 20
QUIET_POWER = 1e-5
FULL_SCALE_STEP = FULL_SCALE / INT16_MAX
LISTEN_BEFORE, LISTEN_AFTER = 2, 4
MARK_KINDS = ("suspect", "bass", "leak")
TOP_GENRES = 8
HEAVY_TRACKS = 10


def loud_windows(labels: Path, track_id: str, count: int) -> np.ndarray:
    mix = np.load(labels / f"{track_id}.mix.npy", mmap_mode="r")
    return np.array(
        [
            np.mean(
                (
                    mix[:, j * WINDOW : (j + 1) * WINDOW].astype(np.float32)
                    * FULL_SCALE_STEP
                )
                ** 2
            )
            > QUIET_POWER
            for j in range(count)
        ]
    )


def track_marks(args, windows: np.ndarray, loud: np.ndarray, instrumental: bool, cut):
    agree, level, bass = windows
    if instrumental:
        leak = np.where((level > args.leak_db) & loud)[0]
        return {"leak": [j * WINDOW_SECONDS for j in leak]}, []
    suspect = np.where((level > VOICED_DB) & (agree < cut) & loud)[0]
    marks = {
        "suspect": [j * WINDOW_SECONDS for j in suspect],
        "bass": [j * WINDOW_SECONDS for j in suspect if bass[j] > args.bass_db],
    }
    rows = [
        (float(agree[j]), j * WINDOW_SECONDS, float(level[j]), float(bass[j]))
        for j in suspect
    ]
    return marks, rows


def suspect_cut(store: dict[str, np.ndarray], meta: dict, share: float) -> tuple:
    voiced = np.concatenate(
        [
            v[0][v[1] > VOICED_DB]
            for k, v in store.items()
            if meta[k]["content"] != "instrumental"
        ]
    )
    return float(np.percentile(voiced, share * 100)), voiced


def all_marks(args, store: dict[str, np.ndarray], meta: dict, cut: float):
    marks, rows, quiet = {}, [], 0
    for k, windows in store.items():
        loud = loud_windows(args.labels, k, windows.shape[1])
        quiet += int((~loud).sum())
        instrumental = meta[k]["content"] == "instrumental"
        found, suspects = track_marks(args, windows, loud, instrumental, cut)
        rows += [
            (agree, k, start, level, bass) for agree, start, level, bass in suspects
        ]
        if any(found.values()):
            marks[k] = {key: [int(x) for x in val] for key, val in found.items() if val}
    return marks, rows, quiet


def log_marks(store: dict, meta: dict, marks: dict, summary: tuple) -> None:
    voiced, cut, quiet = summary
    count, tracks = Counter(), Counter()
    for m in marks.values():
        for key, val in m.items():
            count[key] += len(val)
            tracks[key] += 1
    total = sum(v.shape[1] for v in store.values())
    LOG.info(
        f"{len(store)} tracks, {total} windows of {WINDOW_SECONDS} s "
        f"({total * WINDOW_SECONDS / 3600:.1f} h); voiced windows {len(voiced)}, "
        f"agreement median {np.median(voiced):.1f} dB, suspect cut {cut:.1f} dB; "
        f"{quiet} windows under -50 dBFS left out"
    )
    for key in MARK_KINDS:
        minutes = count[key] * WINDOW_SECONDS / 60
        LOG.info(
            f"{key}: {count[key]} windows ({minutes:.0f} min) in {tracks[key]} tracks"
        )
    genres = Counter(
        meta[k]["genre"] for k, m in marks.items() for _ in m.get("suspect", [])
    )
    top = ", ".join(f"{g or '?'} {n}" for g, n in genres.most_common(TOP_GENRES))
    LOG.info(f"suspect windows by genre: {top}")
    heavy = sorted(
        (
            (len(m["suspect"]) / store[k].shape[1], k)
            for k, m in marks.items()
            if "suspect" in m
        ),
        reverse=True,
    )[:HEAVY_TRACKS]
    shares = ", ".join(f"{k} {s * 100:.0f}%" for s, k in heavy)
    LOG.info(f"tracks with the largest share of suspect windows: {shares}")


def worst_per_track(rows: list[tuple], count: int) -> list[tuple]:
    seen, picked = set(), []
    for row in sorted(rows):
        if row[1] in seen:
            continue
        seen.add(row[1])
        picked.append(row)
        if len(picked) == count:
            break
    return picked


def write_listening(args, meta: dict, rows: list[tuple]) -> None:
    model, _ = load_student(args.checkpoint, args.teacher, args.fits)
    args.listen.mkdir(parents=True, exist_ok=True)
    lines = [
        "| # | Track | Genre | Start | Agreement | Vocals to mix | Bass share |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, (agree, k, start, level, bass) in enumerate(
        worst_per_track(rows, args.listen_top), 1
    ):
        mix, teacher = load(args.labels, k, "mix"), load(args.labels, k, "vocals")
        a = max(0, (start - LISTEN_BEFORE) * SAMPLE_RATE)
        b = min(mix.shape[1], (start + LISTEN_AFTER) * SAMPLE_RATE)
        student = infer_overlap(model, mix)
        tag = f"{i:02d} {k} {start}s"
        for name, x in (("1 mix", mix), ("2 teacher", teacher), ("3 student", student)):
            path = args.listen / f"{tag} - {name}.flac"
            sf.write(path, x[:, a:b].T, SAMPLE_RATE, subtype="PCM_24")
        lines.append(
            f"| {i} | {k} | {meta[k]['genre']} | {start} s | {agree:.1f} dB "
            f"| {level:.1f} dB | {bass:.1f} dB |"
        )
        LOG.info(f"listen {tag}")
    text = "\n".join(lines) + "\n"
    (args.listen / "readme.md").write_text(text, encoding="utf-8")


def mark(args) -> None:
    store = dict(np.load(args.run / "windows.npz"))
    meta = dataset_rows(args.dataset)
    cut, voiced = suspect_cut(store, meta, args.share)
    marks, rows, quiet = all_marks(args, store, meta, cut)
    text = json.dumps(marks, indent=1)
    (args.run / "marks.json").write_text(text, encoding="utf-8")
    log_marks(store, meta, marks, (voiced, cut, quiet))
    if args.listen:
        write_listening(args, meta, rows)


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "data-marks",
        parents=[common],
        help="mark suspect, leaking and bass windows of the corpus for --marks",
    )
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Output directory of data-disagreement: windows.npz in, marks.json out",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Directory of the teacher's labels",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="dataset.csv of the corpus: content and genre per track",
    )
    parser.add_argument(
        "--share",
        type=float,
        default=SHARE,
        help="Share of voiced windows marked suspect",
    )
    parser.add_argument(
        "--leak-db",
        type=float,
        default=LEAK_DB,
        help="Teacher vocals over the mix that mark a window of an instrumental",
    )
    parser.add_argument(
        "--bass-db",
        type=float,
        default=BASS_DB,
        help="120-250 Hz over 500-2000 Hz of the teacher vocals that marks bass",
    )
    parser.add_argument(
        "--listen",
        type=Path,
        help="Write the worst suspect windows (6 s) of the student here",
    )
    parser.add_argument("--listen-top", type=int, default=LISTEN_TOP)
    parser.add_argument(
        "--checkpoint", type=Path, help="The student for --listen (student.pt)"
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        help="Directory with the teacher's config.yaml and model.ckpt (--listen)",
    )
    parser.add_argument(
        "--fits",
        type=Path,
        help="JSON of the fit files the student's modifiers read (--listen)",
    )
    parser.set_defaults(run=mark)
