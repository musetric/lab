import json
import logging
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from musetric_toolkit.distill.inference import infer_overlap
from musetric_toolkit.distill.labelling import PEAK
from musetric_toolkit.distill.precision import PRECISION
from musetric_toolkit.distill.release import load_release
from musetric_toolkit.distill.scores import sdr
from musetric_toolkit.distill.student import Fits, build_student
from musetric_toolkit.distill.teacher import (
    SAMPLE_RATE,
    STEP,
    load_teacher,
    read_config,
    warm_rotary,
)
from musetric_toolkit.separate_audio.roformer_utils import AudioProcessor

LOG = logging.getLogger(__name__)

TRACK_LIST = "tracks.json"
SEPARATION = ".student.npy"
WHOLE = "whole.json"
PER_TRACK = "per_track.json"
PEAK_FLOOR = 1e-8
NORM_FLOOR = 1e-10
NOISE_FLOOR = 1e-12
VOICED_SECOND = 1e-4
BOOTSTRAP_SEED = 0
BOOTSTRAP_DRAWS = 2000
INTERVAL = [2.5, 97.5]
SECONDS_PER_MINUTE = 60


def stem(musdb: Path, name: str, part: str) -> Path:
    wav = musdb / name / f"{part}.wav"
    return wav if wav.exists() else wav.with_suffix(".flac")


def track_names(musdb: Path) -> list[str]:
    listing = musdb / TRACK_LIST
    if listing.exists():
        return json.loads(listing.read_text(encoding="utf-8"))
    return sorted(p.name for p in musdb.iterdir() if p.is_dir())


def read_stem(musdb: Path, name: str, part: str) -> np.ndarray:
    data, rate = sf.read(stem(musdb, name, part), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        message = f"{name} {part}: {rate} Hz, not {SAMPLE_RATE} Hz"
        raise ValueError(message)
    return data.T


def iter_tracks(musdb: Path) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    for name in track_names(musdb):
        mix = read_stem(musdb, name, "mixture")
        voc = read_stem(musdb, name, "vocals")
        gain = PEAK / max(float(np.abs(mix).max()), PEAK_FLOOR)
        yield name, (mix * gain).copy(), (voc * gain).copy()


def teacher_vocals(cache: Path, name: str) -> np.ndarray:
    return np.load(cache / f"{name}.npy").astype(np.float32)


def separation(run: Path, name: str) -> np.ndarray:
    return np.load(run / f"{name}{SEPARATION}").astype(np.float32)


def per_second(ref: np.ndarray, est: np.ndarray) -> float:
    n = ref.shape[1] // SAMPLE_RATE
    r = ref[:, : n * SAMPLE_RATE].reshape(2, n, SAMPLE_RATE)
    e = est[:, : n * SAMPLE_RATE].reshape(2, n, SAMPLE_RATE)
    energy = (r**2).sum((0, 2))
    voiced = energy > energy.max() * VOICED_SECOND
    err = ((r - e) ** 2).sum((0, 2))
    values = 10 * np.log10(energy[voiced] / np.maximum(err[voiced], NOISE_FLOOR))
    return float(np.median(values)) if len(values) else float("nan")


def teacher_cache(args) -> None:
    config = read_config(args.teacher)
    model = load_teacher(args.teacher).cuda().eval()
    warm_rotary(model)
    processor = AudioProcessor(torch.device("cuda"), config)
    window = config.audio.hop_length * (config.inference.dim_t - 1)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rows = {}
    for track, mix, truth in iter_tracks(args.musdb):
        padded = np.pad(mix, ((0, 0), (0, max(0, window - mix.shape[1]))))
        with torch.autocast("cuda", dtype=torch.float16):
            est = processor.demix(padded, model)["vocals"][:, : mix.shape[1]]
        np.save(args.out / f"{track}.npy", est.astype(np.float16))
        rows[track] = dict(sdr=sdr(truth, est), sdr_seconds=per_second(truth, est))
    summary = {
        k: float(np.median([r[k] for r in rows.values()]))
        for k in ("sdr", "sdr_seconds")
    }
    minutes = (time.perf_counter() - start) / SECONDS_PER_MINUTE
    result = dict(model=str(args.teacher), minutes=minutes, **summary, tracks=rows)
    (args.out / WHOLE).write_text(json.dumps(result, indent=1))
    shown = {k: v for k, v in result.items() if k != "tracks"}
    LOG.info(f"result {json.dumps(shown)}")


def separate_tracks(args, separate: Callable[[np.ndarray], np.ndarray]) -> dict:
    args.out.mkdir(parents=True, exist_ok=True)
    to_truth, to_teacher = [], []
    for track, mix, truth in iter_tracks(args.musdb):
        est = separate(mix)
        np.save(args.out / f"{track}{SEPARATION}", est.astype(np.float16))
        teacher = teacher_vocals(args.teacher_vocals, track)
        to_truth.append(sdr(truth, est))
        to_teacher.append(sdr(teacher, est))
    return dict(
        sdr_truth=float(np.median(to_truth)), sdr_teacher=float(np.median(to_teacher))
    )


def write_whole(out: Path, row: dict) -> None:
    (out / WHOLE).write_text(json.dumps(row, indent=1))
    LOG.info(f"whole tracks {json.dumps(row)}")


def separate_checkpoint(args) -> None:
    if args.fp16:
        PRECISION.dtype = torch.float16
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_student(state["student"], args.teacher, Fits.read(args.fits))
    model.load_state_dict(state["model"], strict=not state.get("partial", False))
    scores = separate_tracks(args, partial(infer_overlap, model))
    row = dict(model=str(args.checkpoint), student=state["student"], **scores)
    write_whole(args.out, row)


def whole_track(model, mix: np.ndarray, chunk: int) -> np.ndarray:
    x = torch.from_numpy(mix).cuda()
    total = x.shape[1]
    window = torch.hamming_window(chunk, device="cuda")
    acc, norm = torch.zeros_like(x), torch.zeros(total, device="cuda")
    starts = sorted({max(0, min(i, total - chunk)) for i in range(0, total, STEP)})
    with torch.no_grad():
        for start in starts:
            part = x[:, start : start + chunk]
            n = part.shape[1]
            y = model(part[None])[0]
            acc[:, start : start + n] += y * window[:n]
            norm[start : start + n] += window[:n]
    return (acc / norm.clamp(min=NORM_FLOOR)).cpu().numpy()


def score_model(args) -> None:
    if args.release is not None:
        directory, model = args.release, load_release(args.release)
    else:
        directory, model = args.teacher, load_teacher(args.teacher).cuda().eval()
    config = read_config(directory)
    chunk = config.audio.hop_length * (config.inference.dim_t - 1)
    scores = separate_tracks(args, partial(whole_track, model, chunk=chunk))
    label = "teacher" if args.release is None else str(args.release)
    write_whole(args.out, dict(model=label, **scores))


def db(x):
    return 10 * np.log10(x)


def per_track(run: Path, musdb: Path, cache: Path) -> dict:
    stored = run / PER_TRACK
    if stored.exists():
        return json.loads(stored.read_text(encoding="utf-8"))
    rows = {}
    for name, _, truth in iter_tracks(musdb):
        s = truth.astype(np.float64)
        t = teacher_vocals(cache, name).astype(np.float64)
        x = separation(run, name).astype(np.float64)
        n = min(s.shape[1], t.shape[1], x.shape[1])
        s, t, x = s[:, :n], t[:, :n], x[:, :n]
        rows[name] = dict(
            truth=db((s**2).sum() / ((x - s) ** 2).sum()),
            teacher=db((t**2).sum() / ((x - t) ** 2).sum()),
        )
    stored.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def paired(values: np.ndarray, base: np.ndarray, rng: np.random.Generator) -> str:
    d = values - base
    draws = rng.choice(d, (BOOTSTRAP_DRAWS, len(d))).mean(1)
    lo, hi = np.percentile(draws, INTERVAL)
    return f"({d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}], better on {np.mean(d > 0):.0%})"


def compare(args) -> None:
    runs = args.runs
    data = {run: per_track(run, args.musdb, args.teacher_vocals) for run in runs}
    base = runs[0]
    names = sorted(data[base])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for run in runs:
        line = [f"{run.name:34s}"]
        for key in ("truth", "teacher"):
            v = np.array([data[run][n][key] for n in names])
            line.append(f"{key} {np.median(v):6.2f}")
            if run != base:
                line.append(
                    paired(v, np.array([data[base][n][key] for n in names]), rng)
                )
        LOG.info("  ".join(line))


def add_musdb(parser) -> None:
    parser.add_argument(
        "--musdb",
        required=True,
        type=Path,
        help="MUSDB18-HQ test: a folder per track with mixture and vocals "
        "(wav or flac); tracks.json lists them, sorted folders without it",
    )


def add_teacher_vocals(parser) -> None:
    parser.add_argument(
        "--teacher-vocals",
        required=True,
        type=Path,
        help="The teacher's vocals of every track, <track>.npy, as musdb-teacher "
        "writes them",
    )


def add_teacher(parser, required: bool) -> None:
    parser.add_argument(
        "--teacher",
        required=required,
        type=Path,
        help="Directory with the teacher's config.yaml and model.ckpt",
    )


def add_separate(commands, common) -> None:
    parser = commands.add_parser(
        "musdb-separate",
        parents=[common],
        help="a training checkpoint on the whole MUSDB18-HQ test tracks",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    add_teacher(parser, required=True)
    parser.add_argument(
        "--fits",
        type=Path,
        help="JSON of the fit files the student's modifiers read",
    )
    add_musdb(parser)
    add_teacher_vocals(parser)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--fp16", action="store_true", help="autocast in fp16 instead of bf16"
    )
    parser.set_defaults(run=separate_checkpoint)


def add_score(commands, common) -> None:
    parser = commands.add_parser(
        "musdb-score",
        parents=[common],
        help="the released class or the teacher, fp32, on the whole MUSDB18-HQ "
        "test tracks",
    )
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument(
        "--release",
        type=Path,
        help="Directory of the release: model.safetensors and config.yaml",
    )
    add_teacher(model, required=False)
    add_musdb(parser)
    add_teacher_vocals(parser)
    parser.add_argument("--out", required=True, type=Path)
    parser.set_defaults(run=score_model)


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "musdb-teacher",
        parents=[common],
        help="the teacher's vocals of MUSDB18-HQ test on the labelling path",
    )
    add_teacher(parser, required=True)
    add_musdb(parser)
    parser.add_argument(
        "--out", required=True, type=Path, help="Directory of the teacher's vocals"
    )
    parser.set_defaults(run=teacher_cache)
    add_separate(commands, common)
    add_score(commands, common)
    parser = commands.add_parser(
        "musdb-compare",
        parents=[common],
        help="paired comparison of saved MUSDB18-HQ separations against the first",
    )
    add_musdb(parser)
    add_teacher_vocals(parser)
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Directories of <track>.student.npy, the first the base",
    )
    parser.set_defaults(run=compare)
