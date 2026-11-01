import re

MIN_STYLED_WORDS = 6
PROMPT_WORDS = 25
MIN_WORD_MATCH = 0.7

_STOP = re.compile(r"[.!?]")
_MATCHABLE = re.compile(r"[^\W_]+")


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
    windows: list[dict], segments: list[dict], decode
) -> list[dict]:
    def inside(window: dict) -> list[dict]:
        start, end = float(window["start"]), float(window["end"])
        return [
            segment
            for segment in segments
            if start - 1e-3 <= float(segment.get("start", 0.0)) < end - 1e-3
        ]

    texts = [
        " ".join(
            (segment.get("text") or "").strip() for segment in inside(window)
        ).strip()
        for window in windows
    ]
    punctuated = [index for index, text in enumerate(texts) if _is_punctuated(text)]
    if not punctuated:
        return segments
    result = list(segments)
    for index, (window, text) in enumerate(zip(windows, texts, strict=True)):
        if not is_flat(text):
            continue
        nearest = min(punctuated, key=lambda candidate: abs(candidate - index))
        prompt = " ".join(texts[nearest].split()[-PROMPT_WORDS:])
        repaired = decode(window, prompt)
        if is_flat(repaired) or word_match(text, repaired) < MIN_WORD_MATCH:
            continue
        replaced = {id(segment) for segment in inside(window)}
        result = [segment for segment in result if id(segment) not in replaced]
        result.append(
            {
                "start": float(window["start"]),
                "end": float(window["end"]),
                "text": repaired,
            }
        )
    result.sort(key=lambda segment: float(segment.get("start", 0.0)))
    return result
