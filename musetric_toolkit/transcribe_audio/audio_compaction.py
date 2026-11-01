from typing import NamedTuple

import numpy as np

from musetric_toolkit.transcribe_audio.rounding import js_round, round_to
from musetric_toolkit.transcribe_audio.spectral_chunker import (
    SAMPLE_RATE,
    compute_chunks,
)


class Compaction(NamedTuple):
    compacted: np.ndarray
    chunks: list[dict]
    mapping: list[tuple[float, float, float]]


def compute_packed_chunks(
    audio: np.ndarray,
    chunk_size: float,
    seam_seconds: float,
    min_chunk: float | None = None,
) -> list[list[tuple[float, float]]]:
    spans = compute_chunks(audio, chunk_size, min_chunk)
    packed: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    current_total = 0.0
    for start, end in spans:
        duration = end - start
        seam = seam_seconds if current else 0.0
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
    seam_seconds: float,
) -> Compaction:
    pieces: list[np.ndarray] = []
    chunks: list[dict] = []
    mapping: list[tuple[float, float, float]] = []
    pad = np.zeros(js_round(seam_seconds * SAMPLE_RATE), dtype=audio.dtype)
    cursor = 0.0
    for chunk in packed_chunks:
        chunk_start = cursor
        prev_end: float | None = None
        for start, end in chunk:
            lo = js_round(start * SAMPLE_RATE)
            hi = js_round(end * SAMPLE_RATE)
            if hi <= lo:
                continue
            if prev_end is not None and seam_seconds > 0:
                pieces.append(pad)
                mapping.append((cursor, cursor + seam_seconds, prev_end))
                cursor += seam_seconds
            pieces.append(audio[lo:hi])
            duration = (hi - lo) / SAMPLE_RATE
            mapping.append((cursor, cursor + duration, lo / SAMPLE_RATE))
            cursor += duration
            prev_end = end
        if cursor > chunk_start:
            chunks.append({"start": chunk_start, "end": cursor})
    compacted = np.concatenate(pieces) if pieces else np.zeros(0, dtype=audio.dtype)
    return Compaction(compacted, chunks, mapping)


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


def _map_rounded(seconds: float, mapping: list[tuple[float, float, float]]) -> float:
    return round_to(map_time(seconds, mapping), 3)


def remap_segments_to_original(
    segments: list[dict], mapping: list[tuple[float, float, float]]
) -> list[dict]:
    for segment in segments:
        segment["start"] = _map_rounded(segment["start"], mapping)
        segment["end"] = _map_rounded(segment["end"], mapping)
        for word in segment.get("words", []):
            word["start"] = _map_rounded(word["start"], mapping)
            word["end"] = _map_rounded(word["end"], mapping)
    return segments
