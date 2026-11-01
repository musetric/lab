import collections

import numpy as np

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.transcribe_audio.audio_compaction import (
    Compaction,
    build_compaction,
    compute_packed_chunks,
    remap_segments_to_original,
)
from musetric_toolkit.transcribe_audio.bunched_words import drop_bunched_words
from musetric_toolkit.transcribe_audio.collapse_repair import (
    CollapseRepair,
    repair_collapsed_windows,
)
from musetric_toolkit.transcribe_audio.flat_windows import repair_flat_windows
from musetric_toolkit.transcribe_audio.hallucination_filter import (
    drop_credit_words,
    filter_hallucinated_segments,
    filter_sign_off_segments,
)
from musetric_toolkit.transcribe_audio.lyric_splitter import split_segments_by_lyrics
from musetric_toolkit.transcribe_audio.response_builder import build_payload_segments
from musetric_toolkit.transcribe_audio.rounding import js_round
from musetric_toolkit.transcribe_audio.silence_filter import filter_silent_segments
from musetric_toolkit.transcribe_audio.spectral_chunker import SAMPLE_RATE
from musetric_toolkit.transcribe_audio.whisper_runtime import WhisperRuntime
from musetric_toolkit.transcribe_audio.word_pieces import join_word_pieces
from musetric_toolkit.transcribe_audio.word_timing import lead_word_times

CHUNK_SIZE_SECONDS = 30.0
SEAM_SECONDS = 2.0
SEGMENT_GAP_SECONDS = 1.0
LANGUAGE_WINDOWS = 3
DEFAULT_LANGUAGE = "en"
DECODE_SHARE = 0.8
REPAIR_SHARE = 0.95


def group_words_into_segments(words: list[dict], gap: float) -> list[dict]:
    segments = []
    current: list[dict] = []

    def flush() -> None:
        if not current:
            return
        segments.append(
            {
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(word["text"] for word in current if word["text"]),
                "words": list(current),
            }
        )
        current.clear()

    for word in words:
        if current and word["start"] - current[-1]["end"] > gap:
            flush()
        current.append({**word, "text": word["text"].strip()})
    flush()
    return segments


def _clean_words(words: list[dict]) -> list[dict]:
    return drop_credit_words(join_word_pieces(drop_bunched_words(words)))


def _shift_words(words: list[dict], offset: float) -> list[dict]:
    return [
        {
            "text": word["text"],
            "start": word["start"] + offset,
            "end": word["end"] + offset,
        }
        for word in words
    ]


def _window(compacted: np.ndarray, chunk: dict) -> np.ndarray:
    return compacted[
        js_round(chunk["start"] * SAMPLE_RATE) : js_round(chunk["end"] * SAMPLE_RATE)
    ]


def _rms(clip: np.ndarray) -> float:
    return (
        float(np.sqrt(np.mean(np.square(clip.astype(np.float64)))))
        if clip.size
        else 0.0
    )


def resolve_language(compaction: Compaction, runtime: WhisperRuntime) -> str:
    clips = [_window(compaction.compacted, chunk) for chunk in compaction.chunks]
    loudest = sorted(clips, key=_rms, reverse=True)[:LANGUAGE_WINDOWS]
    votes: collections.Counter[str] = collections.Counter()
    for clip in loudest:
        language = runtime.detect_language(clip)
        votes[language] += 1
        if votes[language] > len(loudest) / 2:
            return language
    return max(votes, key=lambda language: votes[language], default=DEFAULT_LANGUAGE)


def decode_chunks_pass(
    compaction: Compaction, language: str, runtime: WhisperRuntime
) -> list[list[dict]]:
    words = []
    for index, chunk in enumerate(compaction.chunks):
        words.append(
            runtime.transcribe_window(_window(compaction.compacted, chunk), language)
        )
        send_message(
            {
                "type": "progress",
                "progress": DECODE_SHARE * (index + 1) / len(compaction.chunks),
            }
        )
    return words


def repair_pass(
    compaction: Compaction,
    packed: list[list[tuple[float, float]]],
    words: list[list[dict]],
    language: str,
    runtime: WhisperRuntime,
) -> list[list[dict]]:
    shifted = [
        _shift_words(window, chunk["start"])
        for window, chunk in zip(words, compaction.chunks, strict=True)
    ]
    repaired = repair_collapsed_windows(
        CollapseRepair(
            compaction.compacted,
            compaction.chunks,
            packed,
            shifted,
            compaction.mapping,
            lambda clip: runtime.transcribe_aligned(clip, language),
        )
    )
    return repair_flat_windows(
        compaction.compacted,
        compaction.chunks,
        repaired,
        lambda clip, prompt: runtime.transcribe_prompted(clip, language, prompt),
    )


def finalize_pass(compaction: Compaction, words: list[list[dict]]) -> list[dict]:
    cleaned = sorted(
        (word for window in words for word in _clean_words(window)),
        key=lambda word: word["start"],
    )
    segments = group_words_into_segments(cleaned, SEGMENT_GAP_SECONDS)
    segments = filter_silent_segments(segments, compaction.compacted, SAMPLE_RATE)
    segments = filter_hallucinated_segments(segments)
    segments = remap_segments_to_original(segments, compaction.mapping)
    segments = split_segments_by_lyrics(segments)
    segments = filter_sign_off_segments(segments)
    segments = lead_word_times(segments)
    return build_payload_segments(segments)


def transcribe_voice(voice: np.ndarray, runtime: WhisperRuntime) -> list[dict]:
    packed = compute_packed_chunks(voice, CHUNK_SIZE_SECONDS, SEAM_SECONDS)
    if not packed:
        return []
    compaction = build_compaction(voice, packed, SEAM_SECONDS)
    language = resolve_language(compaction, runtime)
    words = decode_chunks_pass(compaction, language, runtime)
    repaired = repair_pass(compaction, packed, words, language, runtime)
    send_message({"type": "progress", "progress": REPAIR_SHARE})
    return finalize_pass(compaction, repaired)
