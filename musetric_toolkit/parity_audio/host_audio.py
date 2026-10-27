import numpy as np

from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file

MEAN = 0.5
POWER = 1.0 / np.sqrt(2.0)


def read_mono(path: str, sample_rate: int, weight: float) -> np.ndarray:
    stereo = read_audio_file(path, sample_rate, 2).astype(np.float64)
    return (stereo[0] + stereo[1]) * weight
