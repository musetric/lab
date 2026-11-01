UPPERCASE_SPLIT_MIN_LINE_DURATION = 1.2
SOFT_PUNCT_SPLIT_MIN_WORDS = 6


def _starts_with_upper(text: str) -> bool:
    for ch in text:
        if ch.isalpha():
            return ch.isupper()
    return False


def _ends_with_strong_punct(text: str) -> bool:
    return text.endswith((".", "!", "?"))


def _ends_with_soft_punct(text: str) -> bool:
    return text.endswith((",", ";", ":"))


def _join_words(words: list[dict]) -> str:
    return " ".join(word["text"] for word in words if word["text"])


def _normalize_words(words: list[dict]) -> list[dict]:
    return [
        {"text": word["text"].strip(), "start": word["start"], "end": word["end"]}
        for word in words
        if word["text"].strip()
    ]


def _splits_before(word: dict, current: list[dict], line_start: float) -> bool:
    previous = current[-1]
    line_duration = max(0.0, previous["end"] - line_start)
    return (
        _ends_with_strong_punct(previous["text"])
        or (
            _ends_with_soft_punct(previous["text"])
            and len(current) >= SOFT_PUNCT_SPLIT_MIN_WORDS
        )
        or (
            _starts_with_upper(word["text"])
            and not _starts_with_upper(previous["text"])
            and line_duration >= UPPERCASE_SPLIT_MIN_LINE_DURATION
        )
    )


def _split_words_on_capitalization(words: list[dict]) -> list[list[dict]]:
    lines = []
    current: list[dict] = []
    line_start = 0.0
    for word in words:
        if current and not _splits_before(word, current, line_start):
            current.append(word)
            continue
        if current:
            lines.append(current)
        current = [word]
        line_start = word["start"]
    if current:
        lines.append(current)
    return lines


def _split_segment(segment: dict) -> list[dict]:
    text = segment["text"].strip()
    if not text:
        return []
    normalized_words = _normalize_words(segment.get("words", []))
    if normalized_words:
        lines = _split_words_on_capitalization(normalized_words)
        if len(lines) > 1:
            return [
                {
                    "start": line[0]["start"],
                    "end": line[-1]["end"],
                    "text": _join_words(line),
                    "words": line,
                }
                for line in lines
            ]
    return [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": text,
            **({"words": normalized_words} if normalized_words else {}),
        }
    ]


def split_segments_by_lyrics(segments: list[dict]) -> list[dict]:
    return [line for segment in segments for line in _split_segment(segment)]
