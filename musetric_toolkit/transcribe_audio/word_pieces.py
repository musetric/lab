import re

_PIECE = re.compile("^['’-][^\\W\\d_]")  # noqa: RUF001


def join_word_pieces(words: list[dict]) -> list[dict]:
    joined: list[dict] = []
    for word in words:
        if joined and _PIECE.match(word["text"]):
            previous = joined[-1]
            joined[-1] = {
                **previous,
                "text": previous["text"] + word["text"],
                "end": word["end"],
            }
            continue
        joined.append(word)
    return joined
