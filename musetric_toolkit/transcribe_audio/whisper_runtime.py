import numpy as np

from musetric_toolkit.transcribe_audio.hallucination_filter import is_hallucination
from musetric_toolkit.transcribe_audio.rounding import js_round
from musetric_toolkit.transcribe_audio.spectral_chunker import SAMPLE_RATE
from musetric_toolkit.transcribe_audio.whisper_decoder import (
    DecodeResult,
    WhisperDecoder,
)
from musetric_toolkit.transcribe_audio.whisper_segments import (
    HalvesRule,
    count_words,
    extract_words,
    is_looped,
    join_halves,
    needs_halves,
    span_text,
)

_GUARD_LADDER = (
    {"no_repeat_ngram_size": 3},
    {"no_repeat_ngram_size": 3, "repetition_penalty": 1.15},
)

_COLLAPSED_WORDS_PER_SECOND = 0.25
_COLLAPSED_MIN_SECONDS = 12
_ALIGNED_WORDS_PER_SECOND = 0.8
_HALVES_RULE = HalvesRule(min_seconds=12, coverage=0.6, max_gap=6)
_SPLIT_MIN_GAIN = 3


def _drop_caption_segments(result: DecodeResult) -> DecodeResult:
    if not result.segments:
        return DecodeResult() if is_hallucination(result.text) else result
    clean = [
        segment
        for segment in result.segments
        if not is_hallucination(span_text(segment))
    ]
    if len(clean) == len(result.segments):
        return result
    chunks = [chunk for segment in clean for chunk in segment]
    return DecodeResult(span_text(chunks), chunks, clean)


def _has_looped_segment(result: DecodeResult) -> bool:
    if not result.segments:
        return is_looped(result.text)
    return any(is_looped(span_text(segment)) for segment in result.segments)


class WhisperRuntime:
    def __init__(self, decoder: WhisperDecoder) -> None:
        self._decoder = decoder

    def detect_language(self, audio: np.ndarray) -> str:
        return self._decoder.detect_language(audio)

    def _run_ladder(
        self, result: DecodeResult, audio: np.ndarray, language: str
    ) -> DecodeResult:
        if not _has_looped_segment(result):
            return result
        for guard in _GUARD_LADDER:
            retried = self._decoder.decode_timestamped(audio, language, guard)
            if not _has_looped_segment(retried):
                return retried
        return result

    def _prefer_aligned(
        self, result: DecodeResult, audio: np.ndarray, language: str
    ) -> DecodeResult:
        duration = audio.shape[0] / SAMPLE_RATE
        words = count_words(result.text)
        collapsed = (
            duration >= _COLLAPSED_MIN_SECONDS
            and words / duration < _COLLAPSED_WORDS_PER_SECOND
        )
        if not collapsed and not is_hallucination(result.text):
            return result
        aligned = self._decoder.decode_aligned(audio, language)
        aligned_words = count_words(aligned.text)
        better = (
            aligned_words > words
            and aligned_words / duration >= _ALIGNED_WORDS_PER_SECOND
            and not is_hallucination(aligned.text)
            and not is_looped(aligned.text)
        )
        return aligned if better else result

    def _split_in_halves(
        self, result: DecodeResult, audio: np.ndarray, language: str
    ) -> DecodeResult:
        duration = audio.shape[0] / SAMPLE_RATE
        if not needs_halves(result.chunks, duration, _HALVES_RULE):
            return result
        middle = js_round(audio.shape[0] / 2)
        kept = []
        for half in (audio[:middle], audio[middle:]):
            clean = _drop_caption_segments(
                self._decoder.decode_timestamped(half, language)
            )
            kept.append([] if _has_looped_segment(clean) else clean.chunks)
        joined = join_halves(kept[0], kept[1], middle / SAMPLE_RATE)
        if count_words(span_text(joined)) < count_words(result.text) + _SPLIT_MIN_GAIN:
            return result
        return DecodeResult(span_text(joined), joined, [joined])

    def transcribe_window(self, audio: np.ndarray, language: str) -> list[dict]:
        if audio.shape[0] == 0:
            return []
        result = self._decoder.decode_timestamped(audio, language)
        result = self._run_ladder(result, audio, language)
        result = _drop_caption_segments(self._prefer_aligned(result, audio, language))
        result = self._split_in_halves(result, audio, language)
        return extract_words(result.chunks)

    def transcribe_aligned(self, audio: np.ndarray, language: str) -> list[dict]:
        return extract_words(self._decoder.decode_aligned(audio, language).chunks)

    def transcribe_prompted(
        self, audio: np.ndarray, language: str, prompt: str
    ) -> list[dict]:
        return extract_words(
            self._decoder.decode_prompted(audio, language, prompt).chunks
        )
