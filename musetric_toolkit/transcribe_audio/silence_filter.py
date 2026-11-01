import numpy as np

_SILENCE_RMS_RATIO = 0.2


def _segment_rms(segment: dict, audio: np.ndarray, sample_rate: int) -> float:
    start = int(max(0.0, segment["start"]) * sample_rate)
    end = int(max(0.0, segment["end"]) * sample_rate)
    if end <= start:
        return 0.0
    clip = audio[start:end].astype(np.float64)
    return float(np.sqrt(np.mean(np.square(clip)))) if clip.size else 0.0


def filter_silent_segments(
    segments: list[dict],
    audio: np.ndarray,
    sample_rate: int,
) -> list[dict]:
    if not segments:
        return segments
    rms_values = [_segment_rms(segment, audio, sample_rate) for segment in segments]
    threshold = float(np.median(rms_values)) * _SILENCE_RMS_RATIO
    return [
        segment
        for segment, rms in zip(segments, rms_values, strict=True)
        if rms >= threshold
    ]
