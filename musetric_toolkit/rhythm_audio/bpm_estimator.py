import numpy as np

_MIN_BEATS = 2
_STEP_TOLERANCE = 0.125
_MIN_PREFERRED_BPM = 60.0
_MAX_PREFERRED_BPM = 160.0
_PROBE_COUNT = 5
_PROBE_SECONDS = 60.0
_MIN_PROBE_SECONDS = 20.0
_CLUSTER_RATIO = 1.08
_DOUBLE_RATIO = 1.5


def estimate_bpm(beats: np.ndarray) -> float:
    if len(beats) < _MIN_BEATS:
        return 0.0
    intervals = np.diff(beats)
    step = float(np.median(intervals))
    if step <= 0:
        return 0.0
    near_step = intervals[np.abs(intervals / step - 1) < _STEP_TOLERANCE]
    return 60.0 / float(np.mean(near_step))


def estimate_meter(
    beats: np.ndarray,
    downbeats: np.ndarray,
    default: int = 4,
) -> int:
    if len(downbeats) < _MIN_BEATS or len(beats) < _MIN_BEATS:
        return default
    counts: list[int] = []
    epsilon = 1e-3
    for i in range(len(downbeats) - 1):
        start = downbeats[i]
        end = downbeats[i + 1]
        in_bar = int(np.sum((beats >= start - epsilon) & (beats < end - epsilon)))
        if in_bar > 0:
            counts.append(in_bar)
    if not counts:
        return default
    return round(float(np.median(counts)))


def map_probe_bpm(bpm: float) -> float:
    if _MIN_PREFERRED_BPM <= bpm <= _MAX_PREFERRED_BPM:
        return bpm
    if bpm > _MAX_PREFERRED_BPM:
        half = bpm / 2.0
        if _MIN_PREFERRED_BPM <= half <= _MAX_PREFERRED_BPM:
            return half
    return 0.0


def consecutive_probe_bpms(beats: np.ndarray, duration: float) -> list[float]:
    if duration < _MIN_PROBE_SECONDS:
        return [estimate_bpm(beats)]
    length = min(_PROBE_SECONDS, duration)
    bpms: list[float] = []
    start = 0.0
    while start + _MIN_PROBE_SECONDS <= duration and len(bpms) < _PROBE_COUNT:
        window = beats[(beats >= start) & (beats < start + length)] - start
        bpms.append(estimate_bpm(window))
        start += length
    return bpms


def consensus_bpm(bpms: list[float]) -> float:
    mapped: list[float] = []
    for bpm in bpms:
        value = map_probe_bpm(bpm)
        if value > 0:
            mapped.append(value)
    if not mapped:
        return 0.0
    values = np.sort(np.asarray(mapped, dtype=np.float64))
    best_start = 0
    best_size = 1
    start = 0
    for end, value in enumerate(values):
        while value / values[start] > _CLUSTER_RATIO:
            start += 1
        size = end - start + 1
        if size > best_size:
            best_size = size
            best_start = start
    cluster = values[best_start : best_start + best_size]
    return float(np.median(cluster))


def keep_from_latest_downbeat(
    counts: np.ndarray,
    downbeat_counts: np.ndarray,
) -> np.ndarray:
    latest = np.maximum(np.searchsorted(downbeat_counts, counts, side="right") - 1, 0)
    return (counts - downbeat_counts[latest]) % 2 == 0


def keep_on_downbeat_parity(
    counts: np.ndarray,
    downbeat_counts: np.ndarray,
) -> np.ndarray:
    even = int(np.sum(downbeat_counts % 2 == 0))
    odd = len(downbeat_counts) - even
    parity = int(downbeat_counts[0] % 2) if even == odd else int(odd > even)
    return counts % 2 == parity


def every_other_beat(beats: np.ndarray, downbeats: np.ndarray) -> np.ndarray:
    if len(beats) < _MIN_BEATS:
        return beats
    intervals = np.diff(beats)
    step = float(np.median(intervals))
    if step <= 0:
        return beats
    steps = np.floor(intervals / step + 0.5).astype(np.int64)
    counts = np.concatenate(([0], np.cumsum(steps)))
    downbeat_counts = (
        counts[np.searchsorted(beats, downbeats)]
        if len(downbeats) > 0
        else np.zeros(1, dtype=np.int64)
    )
    bars = np.diff(downbeat_counts)
    if len(bars) == 0 or round(float(np.median(bars))) % 2 == 0:
        keep = keep_from_latest_downbeat(counts, downbeat_counts)
    else:
        keep = keep_on_downbeat_parity(counts, downbeat_counts)
    kept = beats[keep]
    return kept if len(kept) >= _MIN_BEATS else beats


def summarize_rhythm(
    beats: np.ndarray,
    downbeats: np.ndarray,
    duration: float,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    probes = consecutive_probe_bpms(beats, duration)
    bpm = consensus_bpm(probes if probes else [estimate_bpm(beats)])
    raw = estimate_bpm(beats)
    if bpm <= 0 or raw <= 0 or raw / bpm < _DOUBLE_RATIO:
        return bpm, beats, downbeats, estimate_meter(beats, downbeats)
    folded_beats = every_other_beat(beats, downbeats)
    kept = set(folded_beats.tolist())
    folded_downbeats = np.asarray(
        [time for time in downbeats if time in kept],
        dtype=beats.dtype,
    )
    return (
        bpm,
        folded_beats,
        folded_downbeats,
        estimate_meter(folded_beats, folded_downbeats),
    )
