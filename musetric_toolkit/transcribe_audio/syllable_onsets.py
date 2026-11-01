import numpy as np

from musetric_toolkit.transcribe_audio.rounding import js_round
from musetric_toolkit.transcribe_audio.spectral_chunker import SAMPLE_RATE

_BAND_HZ = (100, 4000)
_MIN_FRAMES = 5
_WIN = 1024
_HOP = 256


def _spectral_flux(audio: np.ndarray) -> np.ndarray:
    frame_count = (len(audio) - _WIN) // _HOP + 1
    offsets = np.arange(_WIN)[None, :] + _HOP * np.arange(frame_count)[:, None]
    frames = audio[offsets].astype(np.float64) * np.hanning(_WIN)
    magnitude = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(_WIN, 1 / SAMPLE_RATE)
    magnitude = magnitude[:, (freqs >= _BAND_HZ[0]) & (freqs <= _BAND_HZ[1])]
    flux = np.maximum(0.0, magnitude[1:] - magnitude[:-1]).sum(axis=1)
    return flux.astype(np.float32).astype(np.float64)


def _count_flux_peaks(flux: np.ndarray) -> int:
    largest = float(flux.max())
    if largest <= 0:
        return 0
    flux = (flux / largest).astype(np.float32).astype(np.float64)
    floor = max(0.12, float(flux.mean() + 0.5 * flux.std()))
    gap = js_round(0.11 * SAMPLE_RATE / _HOP)
    onsets = 0
    last = -gap
    for index in range(1, len(flux) - 1):
        if (
            flux[index] >= floor
            and flux[index] >= flux[index - 1]
            and flux[index] > flux[index + 1]
            and index - last >= gap
        ):
            onsets += 1
            last = index
    return onsets


def syllable_onsets(audio: np.ndarray) -> int:
    if len(audio) < _WIN * 2:
        return 0
    flux = _spectral_flux(audio)
    if len(flux) < _MIN_FRAMES:
        return 0
    return _count_flux_peaks(flux)
