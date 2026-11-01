_BUNCH_MIN_WORDS = 3
_BUNCH_TOLERANCE_SECONDS = 0.02


def _at_point(word: dict, point: float) -> bool:
    return (
        abs(word["start"] - point) <= _BUNCH_TOLERANCE_SECONDS
        and abs(word["end"] - point) <= _BUNCH_TOLERANCE_SECONDS
    )


def drop_bunched_words(words: list[dict]) -> list[dict]:
    kept = []
    start = 0
    while start < len(words):
        point = words[start]["start"]
        end = start
        while end < len(words) and _at_point(words[end], point):
            end += 1
        if end - start >= _BUNCH_MIN_WORDS:
            start = end
            continue
        kept.append(words[start])
        start += 1
    return kept
