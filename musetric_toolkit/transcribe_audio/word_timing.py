_WORD_END_LEAD = 0.15
_WORD_START_LEAD = 0.1
_WORD_START_LEAD_PER_SECOND = 0.2
_WORD_START_LEAD_MAX_SECONDS = 1.0


def _earlier(time: float, lead: float) -> float:
    return max(0.0, time - lead)


def _lead_word(word: dict) -> dict:
    length = min(word["end"] - word["start"], _WORD_START_LEAD_MAX_SECONDS)
    start = _earlier(
        word["start"], _WORD_START_LEAD + _WORD_START_LEAD_PER_SECOND * length
    )
    return {
        **word,
        "start": start,
        "end": max(start, _earlier(word["end"], _WORD_END_LEAD)),
    }


def _lead_segment(segment: dict) -> dict:
    words = [_lead_word(word) for word in segment.get("words", [])]
    if not words:
        return {
            **segment,
            "start": _earlier(segment["start"], _WORD_START_LEAD),
            "end": _earlier(segment["end"], _WORD_END_LEAD),
        }
    return {
        **segment,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "words": words,
    }


def lead_word_times(segments: list[dict]) -> list[dict]:
    return [_lead_segment(segment) for segment in segments]
