import collections
import logging

import numpy as np

# The language is the majority of the three loudest decode windows, as in the
# browser runtime (packages/ai/src/transcription/transcribePipeline.ts): a quiet
# window, an intro without singing, makes Whisper write a caption in English
# and must not outvote the sung ones. Detection stops once one language holds
# a majority; without one, the most voted language wins and the louder window
# breaks a tie.
LANGUAGE_WINDOWS = 3


def _rms(clip) -> float:
    return float(np.sqrt(np.mean(np.square(clip)))) if len(clip) else 0.0


def _detect_language_for_window(model, clip) -> str | None:
    try:
        return model.detect_language(clip)
    except Exception as error:
        logging.debug("Language detection failed for a window: %s", error)
        return None


def detect_language_loudest(
    model,
    compacted_audio,
    chunks: list[dict],
    sample_rate: int,
) -> str | None:
    clips = [
        compacted_audio[
            round(float(chunk["start"]) * sample_rate) : round(
                float(chunk["end"]) * sample_rate
            )
        ]
        for chunk in chunks
    ]
    loudest = sorted(clips, key=_rms, reverse=True)[:LANGUAGE_WINDOWS]
    votes: collections.Counter[str] = collections.Counter()
    for clip in loudest:
        language = _detect_language_for_window(model, clip)
        if language is None:
            continue
        votes[language] += 1
        if votes[language] > len(loudest) / 2:
            return language
    if not votes:
        return None
    return max(votes, key=lambda language: votes[language])
