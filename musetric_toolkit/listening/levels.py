from pathlib import Path

import numpy as np

from musetric_toolkit.distill.corpus import FULL_SCALE, INT16_MAX

RATE = 44100
SCALE = FULL_SCALE / INT16_MAX
SILENCE = 1e-12
QUIET_POWER = 1e-20


def energy(audio: np.ndarray) -> float:
    return float((audio.astype(np.float64) ** 2).sum())


def db(numerator: float, denominator: float) -> float:
    return 10 * np.log10((numerator + SILENCE) / (denominator + SILENCE))


def level_db(audio: np.ndarray, reference: np.ndarray) -> float:
    return float(db(energy(audio), energy(reference)))


def corpus_part(labels: Path, track: str, part: str) -> np.ndarray:
    return np.load(labels / f"{track}.{part}.npy", mmap_mode="r")


def read_mix(labels: Path, track: str) -> np.ndarray:
    return np.load(labels / f"{track}.mix.npy").astype(np.float32) * SCALE


def window_level_db(labels: Path, track: str, start_s: float, seconds: float) -> float:
    start, stop = int(start_s * RATE), int((start_s + seconds) * RATE)
    mix = corpus_part(labels, track, "mix")[:, start:stop]
    power = float(((np.asarray(mix, dtype=np.float64) * SCALE) ** 2).mean())
    return 10 * np.log10(power + QUIET_POWER)


def letter_words(text: str) -> int:
    return sum(any(c.isalpha() for c in token) for token in text.split())
