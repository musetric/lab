from itertools import pairwise
from pathlib import Path

import librosa
import numpy as np
import soundfile
from PIL import Image

from musetric_toolkit.listening.levels import RATE

FMIN = 55.0
OCTAVES = 8
BINS_PER_OCTAVE = 96
WIDTH = 1600
HOP = 128
Q_TIME = 0.09
Q_PITCH = 0.5
DB_FLOOR = -90.0
POWER_FLOOR = 1e-14
PEAK = 0.9
STEMS = (
    "mix",
    "lead_a",
    "lead_b",
    "backing_a",
    "backing_b",
    "instrumental_a",
    "instrumental_b",
)


def cqt_at(channel: np.ndarray, scale: float) -> np.ndarray:
    z = librosa.cqt(
        channel,
        sr=RATE,
        hop_length=HOP,
        fmin=FMIN,
        n_bins=OCTAVES * BINS_PER_OCTAVE,
        bins_per_octave=BINS_PER_OCTAVE,
        filter_scale=scale,
        sparsity=0.01,
    )
    return np.abs(z) ** 2


def cqt_power(x: np.ndarray) -> np.ndarray:
    total = 0.0
    for channel in x:
        sharp_in_time = cqt_at(channel, Q_TIME)
        sharp_in_pitch = cqt_at(channel, Q_PITCH)
        total = total + np.sqrt(sharp_in_time * sharp_in_pitch)
    return total / x.shape[0]


def picture(power: np.ndarray, reference: float, path: Path, duration: float) -> None:
    edges = np.linspace(0.0, duration, WIDTH + 1) * RATE / HOP
    edges = np.floor(edges + 0.5).astype(int)
    columns = [power[:, a : max(b, a + 1)].mean(1) for a, b in pairwise(edges)]
    pooled = np.stack(columns, axis=1)
    level = 10 * np.log10(pooled / reference + POWER_FLOOR)
    grey = np.clip((level - DB_FLOOR) / -DB_FLOOR, 0, 1)
    image = np.round(255 * grey[::-1]).astype(np.uint8)
    Image.fromarray(image).save(path, optimize=False, compress_level=3)


def write_clip(job: tuple[str, float]) -> str:
    folder, duration = Path(job[0]), job[1]
    data = {name: np.load(folder / f"{name}.raw.npy") for name in STEMS}
    peak = max(float(np.abs(value).max()) for value in data.values())
    gain = PEAK / peak if peak > PEAK else 1.0
    powers = {}
    for name in STEMS:
        wave = (data[name] * gain).T
        soundfile.write(folder / f"{name}.wav", wave, RATE, subtype="FLOAT")
        powers[name] = cqt_power(data[name] * gain)
    reference = float(powers["mix"].max())
    for name in STEMS:
        picture(powers[name], reference, folder / f"{name}.png", duration)
        (folder / f"{name}.raw.npy").unlink()
    return folder.name
