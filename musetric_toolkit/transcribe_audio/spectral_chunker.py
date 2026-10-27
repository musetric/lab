"""Spectral chunking of vocal audio for WhisperX.

This module chooses cut points from the audio spectrum instead of WhisperX's
pyannote VAD boundaries. Two per-frame features are derived from a short-time
Fourier transform:

* ``energy``   - frame RMS amplitude.
* ``flatness`` - spectral flatness (geometric mean / arithmetic mean of the
  power spectrum in a vocal band). It is near 0 for tonal, harmonic content (a
  sung note) and near 1 for noise-like content (a breath, fricative or silence).

A frame is a good place to cut when it is real silence (low energy) or
noise-like (high flatness, i.e. a breath); a tonal frame is protected and never
used as a cut point, even when it is quiet. Within each chunk-size budget the
cut snaps to the best such point; when a chunk must end inside continuous
singing (no breath/silence in the budget window) the cut falls on the most
breath-like instant (maximum flatness) rather than the quietest one, which on
songs is usually a clean but soft note.

Flatness is a scale-invariant ratio, so the spectrum normalisation that would
corrupt a Whisper log-mel is irrelevant here; a plain ``numpy`` rFFT is used.
"""

import numpy as np

SAMPLE_RATE = 16000

_HOP = 256  # 16 ms frame step
_WIN = 1024  # 64 ms analysis window
_BAND_HZ = (100, 8000)  # flatness is measured inside this vocal band

# A frame counts as a cut candidate when it is noise-like (breath/fricative/
# silence) or quiet; a clearly tonal frame is protected.
_FLATNESS_BREATH = 0.12  # >= this -> noise-like, may cut
_FLATNESS_TONAL = 0.08  # < this  -> tonal note, protect
_SILENCE_ENERGY_RATIO = 0.15  # energy < ratio * median -> silence
_MIN_PAUSE_SECONDS = 0.10  # shorter cuttable runs are consonants, not pauses
_MIN_VOICE_SECONDS = 0.40  # sustained voicing needed to open a chunk (not a blip)
_REGION_BREAK_SECONDS = 1.2  # silence at least this long forces a chunk split
_SEAM_PAD_SECONDS = 2.0  # natural pause inserted between bridged spans
_DEFAULT_MAX_MIN_RATIO = 0.5  # default minimum chunk = half the budget (<= 15s)
_SILENCE_MAJORITY = 0.5  # a pause that is mostly silence cuts at its quietest


def compute_features(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame ``(energy, flatness)`` arrays for 16 kHz mono audio."""
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
    """A frame that may start/continue a chunk: tonal, or audible (not silence)."""
    return (
        flatness[frame] < _FLATNESS_TONAL
        or energy[frame] >= _SILENCE_ENERGY_RATIO * energy_median
    )


def _find_pauses(
    energy: np.ndarray, flatness: np.ndarray, energy_median: float
) -> list[dict]:
    """Cuttable runs (breath or silence) at least ``_MIN_PAUSE_SECONDS`` long."""
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
    """Pick where to end a chunk that opened at ``start``, within ``[low, high]``.

    A long enough silence forces an early split (region break); otherwise the
    best breath/silence in the budget window is used; failing that the cut falls
    on the most breath-like instant so it never lands on a sustained note.
    """
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
    """Cut ``audio`` into ``(start, end)`` second spans no longer than
    ``chunk_size``, snapping boundaries to breaths/silence and protecting notes.

    Returns contiguous voiced spans; long silences are skipped (region breaks),
    mirroring how WhisperX drops non-speech gaps.
    """
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
        """First frame of a voiced run lasting at least ``_MIN_VOICE_SECONDS``.

        A single tonal-looking or briefly audible frame is not enough: deep
        silence can momentarily read as tonal (low flatness) and a stray
        transient can clear the energy floor for a few frames. Requiring a
        sustained run keeps a chunk from opening on such blips and dragging the
        following silence in with it; the run terminates the scan at the first
        real onset.
        """
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
    """Group the voiced spans from :func:`compute_chunks` into packed chunks.

    Each packed chunk is a list of ``(start, end)`` voiced spans whose total
    *voiced* duration stays within ``chunk_size``. Consecutive short spans that
    are separated only by a silence gap (a region break) are packed together so
    one inference call carries a full ``chunk_size`` budget of real singing
    instead of a half-empty window; the silence between them is dropped from the
    decoded payload (see :func:`build_compaction`). Span boundaries are unchanged
    -- they still fall on breaths/silence, never on a tonal note -- so packing
    only changes how spans are batched, never where the audio is cut.
    """
    spans = compute_chunks(audio, chunk_size, min_chunk)
    packed: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    current_total = 0.0  # voiced + seam pads; must stay within the 30s window
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
    """Concatenate the voiced spans of ``packed_chunks`` into a silence-free
    audio buffer and return the data needed to transcribe it and map back.

    Returns ``(compacted_audio, chunks, mapping)`` where:

    * ``compacted_audio`` is every span concatenated in order (inter-span
      silence excised), the array fed to WhisperX.
    * ``chunks`` is one ``{"start", "end", "segments"}`` dict per packed chunk in
      *compacted* time, the boundaries WhisperX should decode (one inference call
      per packed chunk).
    * ``mapping`` is ``(compacted_start, compacted_end, original_start)`` per span,
      a monotonic piecewise-linear map from compacted time back to the real
      timeline (see :func:`map_time`).
    """
    pieces: list[np.ndarray] = []
    chunks: list[dict] = []
    mapping: list[tuple[float, float, float]] = []
    # Between two bridged spans, hard concatenation splices distant phrases
    # with no boundary, so Whisper bleeds/drops at the seam. Insert a short
    # silence instead -- a natural pause the decoder can segment on -- while still
    # dropping the bulk of the original inter-span silence.
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
                # pad lives in the excised gap; map it back just after prev span
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
    """Map a compacted-timeline instant back to the original timeline."""
    if not mapping:
        return compacted_seconds
    for comp_start, comp_end, original_start in mapping:
        if compacted_seconds < comp_end:
            offset = max(0.0, compacted_seconds - comp_start)
            return original_start + offset
    comp_start, comp_end, original_start = mapping[-1]
    return original_start + (comp_end - comp_start)
