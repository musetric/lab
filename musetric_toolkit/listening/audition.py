import io
import math

import numpy as np
import soundfile
from scipy.ndimage import gaussian_filter
from scipy.signal import istft, stft

NFFT = 2048
HOP = 256
ONLY_PAD_S = 0.15
WITHOUT_PAD_S = 1.5
ONLY_PEAK = 0.5
MAX_BOOST_DB = 36.0
SILENT_PEAK = 1e-7
FADE_S = 0.01
MASK_SIGMA = (1.0, 2.0)
ONLY = "only"
WITHOUT = "without"


def log_pitch(hertz: np.ndarray) -> np.ndarray:
    return np.log2(np.maximum(hertz, 1.0))


def polygon_mask(
    times: np.ndarray, pitches: np.ndarray, polygon: np.ndarray
) -> np.ndarray:
    crossings = np.zeros((len(pitches), len(times) + 1), dtype=np.int32)
    rows = np.arange(len(pitches))
    following = np.roll(polygon, -1, axis=0)
    for (t0, p0), (t1, p1) in zip(polygon, following, strict=True):
        spanned = rows[(p0 > pitches) != (p1 > pitches)]
        at = t0 + (pitches[spanned] - p0) * (t1 - t0) / (p1 - p0)
        np.add.at(crossings, (spanned, np.searchsorted(times, at)), 1)
    to_the_right = np.cumsum(crossings[:, ::-1], axis=1)[:, ::-1][:, 1:]
    return to_the_right % 2 == 1


def boost_to_peak(part: np.ndarray) -> tuple[np.ndarray, float]:
    peak = float(np.abs(part).max()) if part.size else 0.0
    if peak <= SILENT_PEAK:
        return part, 0.0
    boost_db = min(MAX_BOOST_DB, 20 * math.log10(ONLY_PEAK / peak))
    return part * 10 ** (boost_db / 20), boost_db


def fade_ends(part: np.ndarray, rate: int) -> None:
    fade = min(int(FADE_S * rate), part.shape[1] // 2)
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        part[:, :fade] *= ramp
        part[:, -fade:] *= ramp[::-1]


def audition(
    audio: np.ndarray, rate: int, mode: str, polygon: list
) -> tuple[bytes, float, float]:
    freqs, times, spectrum = stft(
        audio, rate, nperseg=NFFT, noverlap=NFFT - HOP, boundary="zeros", padded=True
    )
    points = np.array(polygon, dtype=np.float64)
    shape = np.column_stack([points[:, 0], log_pitch(points[:, 1])])
    inside = polygon_mask(times, log_pitch(freqs), shape)
    mask = gaussian_filter(inside.astype(np.float32), sigma=MASK_SIGMA)
    kept = spectrum * (mask if mode == ONLY else 1 - mask)
    _, wave = istft(kept, rate, nperseg=NFFT, noverlap=NFFT - HOP, boundary=True)
    wave = wave[:, : audio.shape[1]]
    pad = ONLY_PAD_S if mode == ONLY else WITHOUT_PAD_S
    start = max(0.0, float(points[:, 0].min()) - pad)
    stop = min(audio.shape[1] / rate, float(points[:, 0].max()) + pad)
    part = wave[:, int(start * rate) : int(stop * rate)].astype(np.float32)
    boost_db = 0.0
    if mode == ONLY:
        part, boost_db = boost_to_peak(part)
    fade_ends(part, rate)
    buffer = io.BytesIO()
    soundfile.write(buffer, part.T, rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue(), boost_db, start
