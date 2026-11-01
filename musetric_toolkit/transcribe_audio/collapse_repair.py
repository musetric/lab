import collections
import itertools
import logging
import re
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from musetric_toolkit.transcribe_audio.hallucination_filter import is_hallucination
from musetric_toolkit.transcribe_audio.rounding import js_round, round_to
from musetric_toolkit.transcribe_audio.spectral_chunker import SAMPLE_RATE
from musetric_toolkit.transcribe_audio.syllable_onsets import syllable_onsets

_MIN_WINDOW_SECONDS = 12.0
_MIN_ONSET_RATE = 2.0
_MAX_COVERED_FRACTION = 0.3
_MIN_UNCOVERED_SECONDS = 8.0
_MIN_REPAIR_WORD_GAIN = 3
_MIN_UNIQUE_RATIO = 0.35

_WINDOW_CAP_SECONDS = 30.0
_RUNWAY_SECONDS = 15.0
_RUNWAY_PAD_SECONDS = 1.5
_RUNWAY_SNAP_SECONDS = 2.5
_MIN_HALF_SECONDS = 6.0
_MIN_CLEAN_PAYLOAD_SECONDS = 6.0
_MIN_RUNWAY_BUDGET_SECONDS = 4.0

_LOOP_MIN_TOKENS = 4
_LOOP_RUN_LENGTH = 6
_LOOP_DOMINANCE_MIN_TOKENS = 8
_LOOP_DOMINANCE_RATIO = 0.6

_NOT_TOKEN = re.compile(r"[^\w\s]")


def _words_text(words: list[dict]) -> str:
    return " ".join(word["text"].strip() for word in words if word["text"].strip())


def _normalize_tokens(text: str) -> list[str]:
    return _NOT_TOKEN.sub(" ", text.lower()).split()


def unique_ratio(text: str) -> float:
    tokens = _normalize_tokens(text)
    return len(set(tokens)) / len(tokens) if tokens else 1.0


def _looks_looped(text: str) -> bool:
    tokens = _normalize_tokens(text)
    if len(tokens) < _LOOP_MIN_TOKENS:
        return False
    longest_run = run = 1
    for previous, token in itertools.pairwise(tokens):
        run = run + 1 if token == previous else 1
        longest_run = max(longest_run, run)
    if longest_run >= _LOOP_RUN_LENGTH:
        return True
    top = collections.Counter(tokens).most_common(1)[0][1]
    return (
        len(tokens) >= _LOOP_DOMINANCE_MIN_TOKENS
        and top / len(tokens) >= _LOOP_DOMINANCE_RATIO
    )


def _word_coverage(words: list[dict], start: float, end: float) -> float:
    covered = 0.0
    for word in words:
        low, high = max(start, word["start"]), min(end, word["end"])
        if high > low:
            covered += high - low
    return covered


def _slice_seconds(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    return audio[
        max(0, js_round(start * SAMPLE_RATE)) : min(
            len(audio), js_round(end * SAMPLE_RATE)
        )
    ]


def _concat_runway(head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    pad = np.zeros(js_round(_RUNWAY_PAD_SECONDS * SAMPLE_RATE), dtype=head.dtype)
    cap = js_round(_WINDOW_CAP_SECONDS * SAMPLE_RATE)
    return np.concatenate([head, pad, tail])[:cap]


class CollapseRepair(NamedTuple):
    compacted: np.ndarray
    chunks: list[dict]
    packed: list[list[tuple[float, float]]]
    words_per_chunk: list[list[dict]]
    mapping: list[tuple[float, float, float]]
    transcribe_slice: Callable[[np.ndarray], list[dict]]


class _HalfBuild(NamedTuple):
    payload_start: float
    payload_end: float
    runway: np.ndarray
    seam: float


def _rebuild_half(repair: CollapseRepair, half: _HalfBuild) -> list[dict] | None:
    decoded = repair.transcribe_slice(
        _concat_runway(
            half.runway,
            _slice_seconds(repair.compacted, half.payload_start, half.payload_end),
        )
    )
    payload = [word for word in decoded if word["start"] >= half.seam]
    text = _words_text(payload)
    sane = (
        len(text.split()) >= _MIN_REPAIR_WORD_GAIN
        and unique_ratio(text) >= _MIN_UNIQUE_RATIO
        and not _looks_looped(text)
        and not is_hallucination(text)
    )
    if not sane:
        return None
    return [
        {
            "text": word["text"],
            "start": half.payload_start + (word["start"] - half.seam),
            "end": half.payload_start + (word["end"] - half.seam),
        }
        for word in payload
    ]


def _build_snap(
    mapping: list[tuple[float, float, float]],
) -> Callable[[float], float]:
    cuts = sorted(
        {
            round_to(edge, 3)
            for comp_start, comp_end, _ in mapping
            for edge in (comp_start, comp_end)
        }
    )

    def snap(target: float) -> float:
        best = target
        best_distance = _RUNWAY_SNAP_SECONDS
        for cut in cuts:
            distance = abs(cut - target)
            if distance <= best_distance:
                best_distance = distance
                best = cut
        return best

    return snap


def _flag_collapsed_windows(
    repair: CollapseRepair, payloads: list[float]
) -> list[bool]:
    flagged = []
    for chunk, words, payload in zip(
        repair.chunks, repair.words_per_chunk, payloads, strict=True
    ):
        start, end = chunk["start"], chunk["end"]
        onsets = syllable_onsets(_slice_seconds(repair.compacted, start, end))
        cover = _word_coverage(words, start, end)
        collapse = (
            payload >= _MIN_WINDOW_SECONDS
            and onsets / payload >= _MIN_ONSET_RATE
            and cover / payload < _MAX_COVERED_FRACTION
            and payload - cover >= _MIN_UNCOVERED_SECONDS
        )
        flagged.append(collapse or is_hallucination(_words_text(words)))
    return flagged


def _runway_anchor(
    repair: CollapseRepair, flagged: list[bool], payloads: list[float], index: int
) -> float:
    for earlier in range(index - 1, -1, -1):
        if not flagged[earlier] and payloads[earlier] >= _MIN_CLEAN_PAYLOAD_SECONDS:
            return repair.chunks[earlier]["end"]
    return repair.chunks[index]["start"]


def _repair_window(
    repair: CollapseRepair,
    flagged: list[bool],
    payloads: list[float],
    snap: Callable[[float], float],
    index: int,
) -> list[dict] | None:
    start, end = repair.chunks[index]["start"], repair.chunks[index]["end"]
    anchor = _runway_anchor(repair, flagged, payloads, index)
    if anchor < _RUNWAY_SECONDS + 0.5:
        return None
    middle = min(
        max(snap((start + end) / 2), start + _MIN_HALF_SECONDS),
        end - _MIN_HALF_SECONDS,
    )
    if middle <= start or middle >= end:
        return None
    budget = min(
        _RUNWAY_SECONDS,
        _WINDOW_CAP_SECONDS - _RUNWAY_PAD_SECONDS - (middle - start),
        _WINDOW_CAP_SECONDS - _RUNWAY_PAD_SECONDS - (end - middle),
    )
    if budget < _MIN_RUNWAY_BUDGET_SECONDS:
        return None
    runway_start = max(0.0, anchor - budget)
    runway = _slice_seconds(repair.compacted, runway_start, anchor)
    seam = anchor - runway_start + _RUNWAY_PAD_SECONDS
    halves = [
        _rebuild_half(repair, _HalfBuild(start, middle, runway, seam)),
        _rebuild_half(repair, _HalfBuild(middle, end, runway, seam)),
    ]
    kept = [half for half in halves if half is not None]
    original = len(_words_text(repair.words_per_chunk[index]).split())
    total = sum(len(_words_text(half).split()) for half in kept)
    if not kept or total < original + _MIN_REPAIR_WORD_GAIN:
        return None
    logging.info(
        "Collapse repair at window %d (%.1f s): %d -> %d words",
        index,
        start,
        original,
        total,
    )
    return [word for half in kept for word in half]


def repair_collapsed_windows(repair: CollapseRepair) -> list[list[dict]]:
    result = list(repair.words_per_chunk)
    payloads = [sum(end - start for start, end in spans) for spans in repair.packed]
    snap = _build_snap(repair.mapping)
    flagged = _flag_collapsed_windows(repair, payloads)
    for index, is_flagged in enumerate(flagged):
        if not is_flagged:
            continue
        repaired = _repair_window(repair, flagged, payloads, snap, index)
        if repaired is not None:
            result[index] = repaired
    return result
