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


def _join_words(words: list) -> str:
    return " ".join(word["word"] for word in words if word.get("word"))


def _normalize_words(words: list) -> list:
    normalized = []
    for word in words or []:
        text = (word.get("word") or "").strip()
        start = word.get("start")
        end = word.get("end")
        if not text or start is None or end is None:
            continue
        normalized.append(
            {
                "word": text,
                "start": float(start),
                "end": float(end),
            }
        )
    return normalized


def _split_words_on_capitalization(words: list) -> list:
    lines = []
    current = []
    line_start = None
    for word in words:
        text = word.get("word") or ""
        if not text:
            continue
        if not current:
            current = [word]
            line_start = word.get("start")
            continue
        prev_word = current[-1]
        prev_text = prev_word.get("word") or ""
        line_duration = 0.0
        if line_start is not None:
            line_duration = max(0.0, float(prev_word["end"]) - float(line_start))
        if (
            _ends_with_strong_punct(prev_text)
            or (
                _ends_with_soft_punct(prev_text)
                and len(current) >= SOFT_PUNCT_SPLIT_MIN_WORDS
            )
            or (
                _starts_with_upper(text)
                and not _starts_with_upper(prev_text)
                and line_duration >= UPPERCASE_SPLIT_MIN_LINE_DURATION
            )
        ):
            lines.append(current)
            current = [word]
            line_start = word.get("start")
            continue
        current.append(word)
    if current:
        lines.append(current)
    return lines


def split_segments_by_lyrics(segments: list) -> list:
    split_segments = []
    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        normalized_words = _normalize_words(segment.get("words", []))
        if normalized_words:
            split_lines = _split_words_on_capitalization(normalized_words)
            if len(split_lines) > 1:
                for line_words in split_lines:
                    split_segments.append(
                        {
                            "start": float(line_words[0]["start"]),
                            "end": float(line_words[-1]["end"]),
                            "text": _join_words(line_words),
                            "words": line_words,
                        }
                    )
                continue
        split_segments.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": text,
                **({"words": normalized_words} if normalized_words else {}),
            }
        )
    return split_segments
