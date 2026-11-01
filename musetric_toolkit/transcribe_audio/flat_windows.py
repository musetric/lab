import re
from collections.abc import Callable

import numpy as np

from musetric_toolkit.transcribe_audio.rounding import js_round
from musetric_toolkit.transcribe_audio.spectral_chunker import SAMPLE_RATE

MIN_STYLED_WORDS = 6
PROMPT_WORDS = 25
MIN_WORD_MATCH = 0.7

_STOP = re.compile(r"[.!?]")
_MATCHABLE = re.compile(r"[^\W_]+")


def _text_of(words: list[dict]) -> str:
    return " ".join(word["text"].strip() for word in words if word["text"].strip())


def is_flat(text: str) -> bool:
    return (
        len(text.split()) >= MIN_STYLED_WORDS
        and not any(ch.isupper() for ch in text[1:])
        and not _STOP.search(text)
    )


def _is_punctuated(text: str) -> bool:
    return (
        len(text.split()) >= MIN_STYLED_WORDS
        and not is_flat(text)
        and bool(_STOP.search(text))
    )


def word_match(first: str, second: str) -> float:
    a = _MATCHABLE.findall(first.lower())
    b = _MATCHABLE.findall(second.lower())
    if not a and not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for word in a:
        row = [0] * (len(b) + 1)
        for index, other in enumerate(b):
            row[index + 1] = (
                previous[index] + 1
                if word == other
                else max(previous[index + 1], row[index])
            )
        previous = row
    return 2 * previous[len(b)] / (len(a) + len(b))


def repair_flat_windows(
    compacted: np.ndarray,
    chunks: list[dict],
    words_per_chunk: list[list[dict]],
    transcribe_slice: Callable[[np.ndarray, str], list[dict]],
) -> list[list[dict]]:
    texts = [_text_of(words) for words in words_per_chunk]
    punctuated = [index for index, text in enumerate(texts) if _is_punctuated(text)]
    if not punctuated:
        return words_per_chunk
    result = list(words_per_chunk)
    for index, text in enumerate(texts):
        if not is_flat(text):
            continue
        nearest = min(punctuated, key=lambda candidate: abs(candidate - index))
        start, end = chunks[index]["start"], chunks[index]["end"]
        words = transcribe_slice(
            compacted[js_round(start * SAMPLE_RATE) : js_round(end * SAMPLE_RATE)],
            _text_of(words_per_chunk[nearest][-PROMPT_WORDS:]),
        )
        repaired = _text_of(words)
        if not is_flat(repaired) and word_match(text, repaired) >= MIN_WORD_MATCH:
            result[index] = [
                {
                    "text": word["text"],
                    "start": word["start"] + start,
                    "end": word["end"] + start,
                }
                for word in words
            ]
    return result
