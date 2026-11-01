def _build_payload_words(words: list[dict]) -> list[dict]:
    return [
        {"start": word["start"], "end": word["end"], "text": word["text"].strip()}
        for word in words
        if word["text"].strip()
    ]


def build_payload_segments(segments: list[dict]) -> list[dict]:
    payload_segments = []
    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        start, end = segment["start"], segment["end"]
        payload_words = _build_payload_words(segment.get("words", []))
        payload_segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "words": payload_words or [{"start": start, "end": end, "text": text}],
            }
        )
    return payload_segments
