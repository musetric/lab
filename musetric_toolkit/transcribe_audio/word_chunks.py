import string
import unicodedata

from musetric_toolkit.transcribe_audio.rounding import round_to

TIME_PRECISION = 0.02

_TIMESTAMP_TOKENS = 1500
_PREPENDED = "\"'\u201c\u00a1\u00bf([{-"
_APPENDED = "\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\u201d)]}\u3001"
_UNSPACED_LANGUAGES = {"zh", "ja", "th", "lo", "my"}
_REPLACEMENT = "\ufffd"


def _is_punctuation(character: str) -> bool:
    return character in string.punctuation or unicodedata.category(
        character
    ).startswith("P")


class _PunctuationOnly:
    def __init__(self) -> None:
        self._matched = False

    def test(self, text: str) -> bool:
        if self._matched:
            self._matched = False
            return False
        self._matched = bool(text) and all(_is_punctuation(ch) for ch in text)
        return self._matched


def _merge_prepended(words: list[str], indices: list[list[int]]) -> None:
    target = len(words) - 1
    for source in range(len(words) - 2, -1, -1):
        if words[source].startswith(" ") and words[source].strip() in _PREPENDED:
            words[target] = words[source] + words[target]
            indices[target] = indices[source] + indices[target]
            words[source] = ""
            indices[source] = []
        else:
            target = source


def _merge_appended(words: list[str], indices: list[list[int]]) -> None:
    target = 0
    for source in range(1, len(words)):
        if not words[target].endswith(" ") and words[source] in _APPENDED:
            words[target] += words[source]
            indices[target] = indices[target] + indices[source]
            words[source] = ""
            indices[source] = []
        else:
            target = source


class WordChunker:
    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self._punctuation = _PunctuationOnly()
        self._special = set(tokenizer.all_special_ids)
        self._end_of_text = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.timestamp_begin = tokenizer.convert_tokens_to_ids("<|notimestamps|>") + 1

    def _split_on_unicode(
        self, tokens: list[int]
    ) -> tuple[list[str], list[list[int]], list[list[int]]]:
        decoded_full = self._tokenizer.decode(tokens)
        words, word_tokens, indices = [], [], []
        current_tokens: list[int] = []
        current_indices: list[int] = []
        offset = 0
        for index, token in enumerate(tokens):
            current_tokens.append(token)
            current_indices.append(index)
            decoded = self._tokenizer.decode(current_tokens)
            if (
                _REPLACEMENT not in decoded
                or decoded_full[offset + decoded.index(_REPLACEMENT) :][:1]
                == _REPLACEMENT
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                indices.append(current_indices)
                current_tokens = []
                current_indices = []
                offset += len(decoded)
        return words, word_tokens, indices

    def _split_on_spaces(self, tokens: list[int]) -> tuple[list[str], list[list[int]]]:
        pieces, piece_tokens, piece_indices = self._split_on_unicode(tokens)
        words: list[str] = []
        indices: list[list[int]] = []
        for piece, own_tokens, own_indices in zip(
            pieces, piece_tokens, piece_indices, strict=True
        ):
            special = own_tokens[0] >= self._end_of_text
            punctuation = self._punctuation.test(piece.strip())
            if special or piece.startswith(" ") or punctuation or not words:
                words.append(piece)
                indices.append(list(own_indices))
            else:
                words[-1] += piece
                indices[-1].extend(own_indices)
        return words, indices

    def _collate(
        self, tokens: list[int], spans: list[list[float | None]], unspaced: bool
    ) -> list[dict]:
        if unspaced:
            words, _, indices = self._split_on_unicode(tokens)
        else:
            words, indices = self._split_on_spaces(tokens)
        _merge_prepended(words, indices)
        _merge_appended(words, indices)
        return [
            {"text": word, "timestamp": [spans[group[0]][0], spans[group[-1]][1]]}
            for word, group in zip(words, indices, strict=True)
            if word
        ]

    def _token_span(
        self, token: int, index: int, times: list[float]
    ) -> list[float | None]:
        start = round_to(times[index], 2)
        if index + 1 >= len(times):
            return [start, None]
        end = round_to(times[index + 1], 2)
        if self._punctuation.test(self._tokenizer.decode([token])):
            end = round_to(min(start + TIME_PRECISION, end), 2)
        return [start, end]

    def _is_unspaced_language(self, token: int) -> bool:
        return self._tokenizer.decode([token])[2:-2] in _UNSPACED_LANGUAGES

    def decode_words(
        self, tokens: list[int], times: list[float]
    ) -> tuple[str, list[dict]]:
        timestamp_end = self.timestamp_begin + _TIMESTAMP_TOKENS
        unspaced = False
        opened: float | None = None
        current: list[int] = []
        spans: list[list[float | None]] = []
        texts: list[str] = []
        words: list[dict] = []
        for index, token in enumerate(tokens):
            if token in self._special:
                unspaced = unspaced or self._is_unspaced_language(token)
            elif self.timestamp_begin <= token <= timestamp_end:
                time = round_to((token - self.timestamp_begin) * TIME_PRECISION, 2)
                if opened is None:
                    opened = time
                elif time != opened:
                    closed = self._collate(current, spans, unspaced)
                    for word in closed:
                        if word["timestamp"][1] > time >= word["timestamp"][0]:
                            word["timestamp"][1] = time
                    texts.append(self._tokenizer.decode(current))
                    words.extend(closed)
                    opened = None
                    current = []
                    spans = []
            else:
                current.append(token)
                spans.append(self._token_span(token, index, times))
        if current:
            texts.append(self._tokenizer.decode(current))
            words.extend(self._collate(current, spans, unspaced))
        return "".join(texts), words
