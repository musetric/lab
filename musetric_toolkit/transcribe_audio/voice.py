import numpy as np

from musetric_toolkit.parity_audio.host_audio import POWER, read_mono
from musetric_toolkit.transcribe_audio.spectral_chunker import SAMPLE_RATE

BACKING_WEIGHT = 0.5


def read_voice(audio_path: str, backing_path: str | None) -> np.ndarray:
    samples = read_mono(audio_path, SAMPLE_RATE, POWER).astype(np.float32)
    if backing_path:
        backing = read_mono(backing_path, SAMPLE_RATE, POWER).astype(np.float32)
        length = min(samples.shape[0], backing.shape[0])
        samples[:length] += BACKING_WEIGHT * backing[:length]
    return samples
