import json
import logging
from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.distill.corpus import dataset_rows, load
from musetric_toolkit.distill.inference import infer_overlap
from musetric_toolkit.distill.student import load_student
from musetric_toolkit.distill.teacher import SAMPLE_RATE

LOG = logging.getLogger(__name__)

WINDOW_SECONDS = 2
WINDOW = WINDOW_SECONDS * SAMPLE_RATE
NFFT = 4096
HOP = 2205
BASS_HZ = (120, 250)
VOICE_HZ = (500, 2000)
VOICED_DB = -25
RANKED = 500
SAVE_EVERY = 50
LOW_PERCENTILE = 10


def db(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(num, 1e-12) / np.maximum(den, 1e-12))


def bass_share(vocals: np.ndarray, n: int) -> np.ndarray:
    mono = torch.from_numpy(np.ascontiguousarray(vocals.mean(0)[: n * WINDOW]))
    spec = torch.stft(
        mono,
        NFFT,
        HOP,
        window=torch.hann_window(NFFT),
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    power = spec.abs().pow(2).numpy()
    freqs = np.fft.rfftfreq(NFFT, 1 / SAMPLE_RATE)
    frames = power.shape[1] // n * n
    power = power[:, :frames].reshape(len(freqs), n, -1).sum(-1)
    bass = (freqs >= BASS_HZ[0]) & (freqs < BASS_HZ[1])
    voice = (freqs >= VOICE_HZ[0]) & (freqs < VOICE_HZ[1])
    return db(power[bass].sum(0), power[voice].sum(0))


def window_scores(model, labels: Path, track_id: str) -> np.ndarray | None:
    mix, teacher = load(labels, track_id, "mix"), load(labels, track_id, "vocals")
    n = mix.shape[1] // WINDOW
    if n == 0:
        return None
    student = infer_overlap(model, mix)
    m, t, s = (
        x[:, : n * WINDOW].reshape(2, n, WINDOW) for x in (mix, teacher, student)
    )
    energy = (t**2).sum((0, 2))
    scores = [
        db(energy, ((s - t) ** 2).sum((0, 2))),
        db(energy, (m**2).sum((0, 2))),
        bass_share(teacher, n),
    ]
    return np.stack(scores).astype(np.float32)


def write_ranking(out: Path, store: dict[str, np.ndarray], rows: list[dict]) -> None:
    content = {r["id"]: r["content"] for r in rows}
    flat = [(k, j, *v[:, j]) for k, v in store.items() for j in range(v.shape[1])]
    voiced = sorted((w for w in flat if w[3] > VOICED_DB), key=lambda w: w[2])
    ranked = [
        dict(
            track=k,
            start=j * WINDOW_SECONDS,
            content=content.get(k),
            agreement=round(float(agreement), 1),
            vocals_to_mix=round(float(level), 1),
            bass_share=round(float(bass), 1),
        )
        for k, j, agreement, level, bass in voiced[:RANKED]
    ]
    text = json.dumps(ranked, indent=1, ensure_ascii=False)
    (out / "least_agreement.json").write_text(text, encoding="utf-8")
    agree = np.array([w[2] for w in voiced])
    LOG.info(
        f"{len(store)} tracks, {len(voiced)} voiced windows; agreement median "
        f"{np.median(agree):.1f} dB, {LOW_PERCENTILE}th percentile "
        f"{np.percentile(agree, LOW_PERCENTILE):.1f} dB"
    )


def disagreement(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [
        r
        for r in dataset_rows(args.dataset).values()
        if r["decision"] == "use" and r["role"] == "train" and r["content"] != "pell"
    ]
    model, _ = load_student(args.checkpoint, args.teacher, args.fits)
    done = args.out / "windows.npz"
    store = dict(np.load(done)) if done.exists() else {}
    for i, row in enumerate(rows):
        if row["id"] in store or not (args.labels / f"{row['id']}.mix.npy").exists():
            continue
        scores = window_scores(model, args.labels, row["id"])
        if scores is None:
            continue
        store[row["id"]] = scores
        if i % SAVE_EVERY == 0:
            np.savez(done, **store)
            LOG.info(f"{i}/{len(rows)} {row['id']}")
    np.savez(done, **store)
    write_ranking(args.out, store, rows)


def add_student_arguments(parser) -> None:
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="A saved student (student.pt of a training run)",
    )
    parser.add_argument(
        "--teacher",
        required=True,
        type=Path,
        help="Directory with the teacher's config.yaml and model.ckpt",
    )
    parser.add_argument(
        "--fits",
        type=Path,
        help="JSON of the fit files the student's modifiers read",
    )


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "data-disagreement",
        parents=[common],
        help="score 2 s windows of the training tracks by student-teacher agreement",
    )
    add_student_arguments(parser)
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
        help="dataset.csv (or a variant): its training tracks are scored",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory of windows.npz (resumable) and least_agreement.json",
    )
    parser.set_defaults(run=disagreement)
