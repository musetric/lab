import numpy as np

SAMPLE_RATE = 16000

_HOP = 256
_WIN = 1024
_BAND_HZ = (100, 8000)

_FLATNESS_BREATH = 0.12
_FLATNESS_TONAL = 0.08
_SILENCE_ENERGY_RATIO = 0.15
_MIN_PAUSE_SECONDS = 0.10
_MIN_VOICE_SECONDS = 0.40
_REGION_BREAK_SECONDS = 1.2
_SEAM_PAD_SECONDS = 2.0
_DEFAULT_MAX_MIN_RATIO = 0.5
_SILENCE_MAJORITY = 0.5


def compute_features(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(_WIN).astype(np.float32)
    frame_count = max(0, (len(audio) - _WIN) // _HOP + 1)
    if frame_count == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)

    offsets = np.arange(_WIN)[None, :] + _HOP * np.arange(frame_count)[:, None]
    frames = audio[offsets]
    energy = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))

    spectrum = np.fft.rfft(frames * window, axis=1)
    power = spectrum.real**2 + spectrum.imag**2 + 1e-12
    freqs = np.fft.rfftfreq(_WIN, 1 / SAMPLE_RATE)
    band = (freqs >= _BAND_HZ[0]) & (freqs <= _BAND_HZ[1])
    band_power = power[:, band]
    flatness = np.exp(np.mean(np.log(band_power), axis=1)) / np.mean(band_power, axis=1)
    return energy.astype(np.float32), flatness.astype(np.float32)


def _frame_to_seconds(frame: int) -> float:
    return frame * _HOP / SAMPLE_RATE


def _seconds_to_frame(seconds: float) -> int:
    return round(seconds * SAMPLE_RATE / _HOP)


def _is_anchor(
    energy: np.ndarray, flatness: np.ndarray, frame: int, energy_median: float
) -> bool:
    return (
        flatness[frame] < _FLATNESS_TONAL
        or energy[frame] >= _SILENCE_ENERGY_RATIO * energy_median
    )


def _find_pauses(
    energy: np.ndarray, flatness: np.ndarray, energy_median: float
) -> list[dict]:
    silence = energy < _SILENCE_ENERGY_RATIO * energy_median
    cuttable = (flatness >= _FLATNESS_BREATH) | silence
    pauses: list[dict] = []
    index, total = 0, len(flatness)
    while index < total:
        if not cuttable[index]:
            index += 1
            continue
        end = index
        while end < total and cuttable[end]:
            end += 1
        duration = _frame_to_seconds(end - index)
        if duration >= _MIN_PAUSE_SECONDS:
            run_flatness = flatness[index:end]
            run_silence = silence[index:end]
            silence_fraction = float(np.mean(run_silence))
            if silence_fraction > _SILENCE_MAJORITY:
                center = index + int(energy[index:end].argmin())
                is_region_break = duration >= _REGION_BREAK_SECONDS
            else:
                center = index + int(run_flatness.argmax())
                is_region_break = False
            pauses.append(
                {
                    "center": _frame_to_seconds(center),
                    "score": float(run_flatness.mean()) * min(duration, 0.5)
                    + silence_fraction,
                    "is_region_break": is_region_break,
                }
            )
        index = end
    return pauses


def _choose_cut(
    pauses: list[dict],
    flatness: np.ndarray,
    start: float,
    low: float,
    high: float,
) -> float:
    region_breaks = [
        p for p in pauses if start < p["center"] <= high and p["is_region_break"]
    ]
    if region_breaks:
        return min(region_breaks, key=lambda p: p["center"])["center"]
    in_window = [p for p in pauses if low <= p["center"] <= high]
    if in_window:
        return max(in_window, key=lambda p: p["score"])["center"]
    lo_frame, hi_frame = _seconds_to_frame(low), _seconds_to_frame(high)
    if hi_frame > lo_frame:
        return _frame_to_seconds(lo_frame + int(flatness[lo_frame:hi_frame].argmax()))
    return high


def compute_chunks(
    audio: np.ndarray,
    chunk_size: float,
    min_chunk: float | None = None,
) -> list[tuple[float, float]]:
    energy, flatness = compute_features(audio)
    if len(flatness) == 0:
        return []
    energy_median = float(np.median(energy[energy > 0])) or 1e-6
    voiced_end = len(audio) / SAMPLE_RATE
    if min_chunk is None:
        min_chunk = min(15.0, chunk_size * _DEFAULT_MAX_MIN_RATIO)

    pauses = _find_pauses(energy, flatness, energy_median)

    min_voice = max(1, _seconds_to_frame(_MIN_VOICE_SECONDS))

    def next_anchor(from_frame: int) -> float:
        total = len(flatness)
        frame = from_frame
        while frame < total:
            if not _is_anchor(energy, flatness, frame, energy_median):
                frame += 1
                continue
            run = frame
            while run < total and _is_anchor(energy, flatness, run, energy_median):
                run += 1
            if run - frame >= min_voice:
                return _frame_to_seconds(frame)
            frame = run
        return _frame_to_seconds(total - 1)

    chunks: list[tuple[float, float]] = []
    start = next_anchor(0)
    while start < voiced_end - 0.5:
        low = start + min_chunk
        high = min(start + chunk_size, voiced_end)
        cut = _choose_cut(pauses, flatness, start, low, high)
        chunks.append((round(start, 3), round(cut, 3)))
        start = max(next_anchor(_seconds_to_frame(cut)), cut + 0.1)
    return chunks


def compute_packed_chunks(
    audio: np.ndarray,
    chunk_size: float,
    min_chunk: float | None = None,
) -> list[list[tuple[float, float]]]:
    spans = compute_chunks(audio, chunk_size, min_chunk)
    packed: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    current_total = 0.0
    for start, end in spans:
        duration = end - start
        seam = _SEAM_PAD_SECONDS if current else 0.0
        if current and current_total + seam + duration > chunk_size:
            packed.append(current)
            current = []
            current_total = 0.0
            seam = 0.0
        current.append((start, end))
        current_total += seam + duration
    if current:
        packed.append(current)
    return packed


def build_compaction(
    audio: np.ndarray,
    packed_chunks: list[list[tuple[float, float]]],
) -> tuple[np.ndarray, list[dict], list[tuple[float, float, float]]]:
    pieces: list[np.ndarray] = []
    chunks: list[dict] = []
    mapping: list[tuple[float, float, float]] = []
    pad_samples = round(_SEAM_PAD_SECONDS * SAMPLE_RATE)
    pad = np.zeros(pad_samples, dtype=audio.dtype)
    cursor = 0.0
    for chunk in packed_chunks:
        chunk_start = cursor
        prev_end: float | None = None
        for start, end in chunk:
            lo = max(0, round(start * SAMPLE_RATE))
            hi = min(len(audio), round(end * SAMPLE_RATE))
            if hi <= lo:
                continue
            if prev_end is not None and pad_samples:
                pieces.append(pad)
                mapping.append((cursor, cursor + _SEAM_PAD_SECONDS, prev_end))
                cursor += _SEAM_PAD_SECONDS
            piece = audio[lo:hi]
            duration = len(piece) / SAMPLE_RATE
            pieces.append(piece)
            mapping.append((cursor, cursor + duration, lo / SAMPLE_RATE))
            cursor += duration
            prev_end = end
        if cursor > chunk_start:
            chunks.append(
                {
                    "start": chunk_start,
                    "end": cursor,
                    "segments": [(chunk_start, cursor)],
                }
            )
    compacted = np.concatenate(pieces) if pieces else np.zeros(0, dtype=audio.dtype)
    return compacted, chunks, mapping


def map_time(
    compacted_seconds: float,
    mapping: list[tuple[float, float, float]],
) -> float:
    if not mapping:
        return compacted_seconds
    for comp_start, comp_end, original_start in mapping:
        if compacted_seconds < comp_end:
            offset = max(0.0, compacted_seconds - comp_start)
            return original_start + offset
    comp_start, comp_end, original_start = mapping[-1]
    return original_start + (comp_end - comp_start)
