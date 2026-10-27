import numpy as np


def hann_periodic(n_fft: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)


def analyze(chunk: np.ndarray, n_fft: int, hop: int, frames: int) -> np.ndarray:
    pad = n_fft // 2
    padded = np.pad(chunk.astype(np.float64), ((0, 0), (pad, pad)), mode="reflect")
    index = np.arange(frames)[:, None] * hop + np.arange(n_fft)[None, :]
    return np.fft.rfft(padded[:, index] * hann_periodic(n_fft), axis=-1)


def synthesize(spectrum: np.ndarray, n_fft: int, hop: int, samples: int) -> np.ndarray:
    pad = n_fft // 2
    channels, frames, _ = spectrum.shape
    window = hann_periodic(n_fft)
    frames_time = np.fft.irfft(spectrum, n=n_fft, axis=-1) * window
    length = hop * (frames - 1) + n_fft
    summed = np.zeros((channels, length))
    envelope = np.zeros(length)
    for frame in range(frames):
        start = frame * hop
        summed[:, start : start + n_fft] += frames_time[:, frame]
        envelope[start : start + n_fft] += window * window
    audio = summed[:, pad : pad + samples]
    return audio / np.maximum(envelope[pad : pad + samples], 1e-8)
