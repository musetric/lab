import numpy as np

# A segment is dropped when its RMS is below this share of the median segment
# RMS.
SILENCE_RMS_RATIO = 0.2


def filter_silent_segments(
    segments: list[dict],
    audio,
    sample_rate: int = 16000,
) -> list[dict]:
    if not segments:
        return segments
    rms_values = []
    for segment in segments:
        start = int(max(0.0, float(segment["start"])) * sample_rate)
        end = int(max(0.0, float(segment["end"])) * sample_rate)
        if end <= start:
            rms_values.append(0.0)
            continue
        clip = audio[start:end]
        if len(clip) == 0:
            rms_values.append(0.0)
            continue
        rms_values.append(float(np.sqrt(np.mean(np.square(clip)))))
    median_rms = float(np.median(rms_values)) if rms_values else 0.0
    threshold = median_rms * SILENCE_RMS_RATIO
    filtered = [
        segment
        for segment, rms in zip(segments, rms_values, strict=False)
        if rms >= threshold
    ]
    return filtered
