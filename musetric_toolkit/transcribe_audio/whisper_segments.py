import gzip
import itertools
import re
from typing import NamedTuple

from musetric_toolkit.transcribe_audio.collapse_repair import unique_ratio

_DEGENERATE_REPEAT = re.compile(r"(.{1,3}?)\1{3,}")
_LOOP_MIN_BYTES = 48
_LOOP_COMPRESSION_RATIO = 3.5
_LOOP_UNIQUE_RATIO = 0.35
_DEFAULT_COMPRESSION = 6


def extract_words(chunks: list[dict]) -> list[dict]:
    words = []
    for chunk in chunks:
        text = _DEGENERATE_REPEAT.sub(r"\1\1", chunk["text"].strip())
        if not text:
            continue
        start, end = chunk["timestamp"]
        words.append(
            {"text": text, "start": start, "end": start if end is None else end}
        )
    return words


def span_text(chunks: list[dict]) -> str:
    return "".join(chunk["text"] for chunk in chunks).strip()


def _chunk_end(chunk: dict) -> float:
    start, end = chunk["timestamp"]
    return start if end is None else end


def _widest_gap(chunks: list[dict]) -> float:
    widest = 0.0
    for previous, chunk in itertools.pairwise(chunks):
        widest = max(widest, chunk["timestamp"][0] - _chunk_end(previous))
    return widest


class HalvesRule(NamedTuple):
    min_seconds: float
    coverage: float
    max_gap: float


def needs_halves(chunks: list[dict], duration: float, rule: HalvesRule) -> bool:
    last_end = _chunk_end(chunks[-1]) if chunks else 0.0
    return duration >= rule.min_seconds and (
        last_end < rule.coverage * duration or _widest_gap(chunks) >= rule.max_gap
    )


def join_halves(first: list[dict], second: list[dict], offset: float) -> list[dict]:
    return [
        *first,
        *(
            {
                "text": chunk["text"],
                "timestamp": [
                    chunk["timestamp"][0] + offset,
                    _chunk_end(chunk) + offset,
                ],
            }
            for chunk in second
        ),
    ]


def count_words(text: str) -> int:
    return len(text.split())


def _compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    if len(data) < _LOOP_MIN_BYTES:
        return 0.0
    return len(data) / len(gzip.compress(data, _DEFAULT_COMPRESSION, mtime=0))


def is_looped(text: str) -> bool:
    return (
        _compression_ratio(text) > _LOOP_COMPRESSION_RATIO
        and unique_ratio(text) < _LOOP_UNIQUE_RATIO
    )


def _read_bounds(
    tokens: list[int], timestamp_begin: int, time_precision: float
) -> list[tuple[float, float]]:
    bounds = []
    opened = None
    for token in tokens:
        if token < timestamp_begin:
            continue
        time = (token - timestamp_begin) * time_precision
        if opened is None:
            opened = time
            continue
        bounds.append((opened, time))
        opened = None
    return bounds


def split_by_segments(
    tokens: list[int],
    words: list[dict],
    timestamp_begin: int,
    time_precision: float,
) -> list[list[dict]]:
    bounds = _read_bounds(tokens, timestamp_begin, time_precision)
    if not bounds:
        return [words] if words else []
    segments: list[list[dict]] = [[] for _ in bounds]
    for word in words:
        index = next(
            (
                candidate
                for candidate, (_, end) in enumerate(bounds)
                if word["timestamp"][0] <= end
            ),
            len(bounds) - 1,
        )
        segments[index].append(word)
    return [segment for segment in segments if segment]
