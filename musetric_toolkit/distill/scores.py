import numpy as np


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    noise = max(np.sum((ref - est) ** 2), 1e-12)
    return float(10 * np.log10(np.sum(ref**2) / noise))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
