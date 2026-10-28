import csv
import itertools
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.distill.corpus import load
from musetric_toolkit.distill.teacher import SAMPLE_RATE

LOG = logging.getLogger(__name__)

NFFT, HOP = 2048, 512
BANDWIDTH_REFERENCE_HZ = (1000, 4000)
BANDWIDTH_BELOW_DB = 50
SMOOTHING_BINS = 5
STEREO_FLOOR = 1e-12
CLIP_FRACTION = 0.995
SILENCE_DB = -60
VOICE_DB = -45
VOICE_BELOW_MIX_DB = -25
AMBIGUOUS_LOW, AMBIGUOUS_HIGH = 0.2, 0.8
LOUD_BIN = 1e-7
VOCAL_LOW_HZ = 120
VOCAL_HIGH_HZ = 12000
PRINT_BANDS = 24
PRINT_LOW_HZ, PRINT_HIGH_HZ = 60, 16000
PRINT_STEP_SECONDS = 0.5
PRINT_STEPS = 60
PROGRESS_EVERY = 200


def db(x: float) -> float:
    return 20 * math.log10(x) if x > 0 else -math.inf


def amplitude(level_db: float) -> float:
    return 10 ** (level_db / 20)


def bandwidth_khz(power: torch.Tensor, freqs: torch.Tensor) -> float:
    mean_db = 10 * torch.log10(power.mean(1) + 1e-20)
    pad = SMOOTHING_BINS // 2
    mean_db = torch.nn.functional.avg_pool1d(
        mean_db[None, None], SMOOTHING_BINS, 1, pad
    )
    mean_db = mean_db[0, 0]
    low, high = BANDWIDTH_REFERENCE_HZ
    ref = mean_db[(freqs >= low) & (freqs <= high)].median()
    above = torch.nonzero(mean_db > ref - BANDWIDTH_BELOW_DB).flatten()
    return float(freqs[above.max()]) / 1000 if len(above) else 0.0


def ambiguous_share(m: torch.Tensor, v: torch.Tensor, power: torch.Tensor) -> float:
    ratio = v.abs() / (v.abs() + (m - v).abs() + 1e-9)
    loud = power > power.max() * LOUD_BIN
    split = (ratio > AMBIGUOUS_LOW) & (ratio < AMBIGUOUS_HIGH)
    return float((power * split * loud).sum() / (power * loud).sum())


def fingerprint(power: torch.Tensor, freqs: torch.Tensor) -> np.ndarray:
    device = power.device
    edges = torch.logspace(
        math.log10(PRINT_LOW_HZ),
        math.log10(PRINT_HIGH_HZ),
        PRINT_BANDS + 1,
        device=device,
    )
    bands = torch.stack(
        [
            power[(freqs >= lo) & (freqs < hi)].sum(0)
            for lo, hi in itertools.pairwise(edges)
        ]
    )
    step = int(PRINT_STEP_SECONDS * SAMPLE_RATE / HOP)
    count = min(PRINT_STEPS, bands.shape[1] // step)
    env = bands[:, : count * step].reshape(PRINT_BANDS, count, step).mean(2)
    env = torch.log10(env + 1e-10)
    env = torch.nn.functional.pad(env, (0, PRINT_STEPS - count))
    env = (env - env.mean()) / (env.std() + 1e-9)
    return env.flatten().cpu().numpy()


def frame_rms(x: torch.Tensor) -> torch.Tensor:
    return x.unfold(0, NFFT, HOP).pow(2).mean(1).sqrt()


def measure(name: str, mix: torch.Tensor, voc: torch.Tensor) -> tuple[dict, np.ndarray]:
    device = mix.device
    mono_mix = mix.mean(0)
    window = torch.hann_window(NFFT, device=device)
    m = torch.stft(mono_mix, NFFT, HOP, window=window, return_complex=True)
    v = torch.stft(voc.mean(0), NFFT, HOP, window=window, return_complex=True)
    freqs = torch.linspace(0, SAMPLE_RATE / 2, m.shape[0], device=device)
    power = m.abs().pow(2)
    side = (mix[0] - mix[1]).pow(2).mean().sqrt()
    mid = (mix[0] + mix[1]).pow(2).mean().sqrt()
    mix_rms = mix.pow(2).mean().sqrt()
    frames = frame_rms(mono_mix)
    vframes = frame_rms(voc.mean(0))
    voiced = (vframes > amplitude(VOICE_DB)) & (
        vframes > frames * amplitude(VOICE_BELOW_MIX_DB)
    )
    vpower = v.abs().pow(2)
    vtotal = vpower.sum() + 1e-20
    row = dict(
        id=name,
        seconds=mix.shape[1] / SAMPLE_RATE,
        bandwidth_khz=round(bandwidth_khz(power, freqs), 2),
        side_db=round(db(float(side / (mid + STEREO_FLOOR))), 1),
        rms_db=round(db(float(mix_rms)), 1),
        clip_share=float((mix.abs() >= mix.abs().max() * CLIP_FRACTION).float().mean()),
        silence_share=round(float((frames < amplitude(SILENCE_DB)).float().mean()), 3),
        vocal_share_db=round(
            db(float(voc.pow(2).mean().sqrt() / (mix_rms + STEREO_FLOOR))), 1
        ),
        vocal_active_share=round(float(voiced.float().mean()), 3),
        ambiguous_share=round(ambiguous_share(m, v, power), 3),
        vocal_low_share=round(float(vpower[freqs < VOCAL_LOW_HZ].sum() / vtotal), 4),
        vocal_high_share=round(float(vpower[freqs > VOCAL_HIGH_HZ].sum() / vtotal), 4),
    )
    return row, fingerprint(power, freqs)


def features(labels: Path, entry: dict, device: str) -> tuple[dict, np.ndarray]:
    mix = torch.from_numpy(load(labels, entry["id"], "mix")).to(device)
    if entry.get("pell", False):
        return measure(entry["id"], mix, mix)
    voc = torch.from_numpy(load(labels, entry["id"], "vocals")).to(device)
    return measure(entry["id"], mix, voc)


def analyze(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.labels / "labels.json").read_text())
    rows, prints = [], []
    for i, entry in enumerate(index):
        row, print_row = features(args.labels, entry, args.device)
        rows.append(row)
        prints.append(print_row)
        if (i + 1) % PROGRESS_EVERY == 0:
            LOG.info(f"{i + 1}/{len(index)}")
    with open(args.out / "features.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.save(args.out / "fingerprints.npy", np.stack(prints))
    ids = json.dumps([r["id"] for r in rows])
    (args.out / "fingerprint_ids.json").write_text(ids, encoding="utf-8")
    LOG.info(f"{len(rows)} tracks -> {args.out / 'features.csv'}")


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "data-analyze",
        parents=[common],
        help="audio quality, teacher behaviour and a fingerprint per labelled track",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Directory of the teacher's labels and labels.json",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Analysis directory: features.csv, fingerprints.npy, fingerprint_ids.json",
    )
    parser.add_argument("--device", default="cuda", help="Torch device of the STFTs")
    parser.set_defaults(run=analyze)
