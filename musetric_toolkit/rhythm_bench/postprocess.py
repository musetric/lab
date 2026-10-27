from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.rhythm_audio.bpm_estimator import summarize_rhythm
from musetric_toolkit.rhythm_bench.variant import logits_path, write_variant

FPS = 50
POSTPROCESSINGS = ("minimal", "dbn")
HEADS = ("beat", "downbeat")

# The logits of a track, the beat head and the downbeat head, one value per frame.
Logits = tuple[np.ndarray, np.ndarray]


def save_logits(
    results_dir: Path,
    track: str,
    logits: str,
    heads: Logits,
) -> None:
    for head, values in zip(HEADS, heads, strict=True):
        path = logits_path(results_dir, track, logits, head)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values.astype(np.float32))


def load_logits(
    results_dir: Path,
    track: str,
    logits: str,
) -> Logits:
    beat, downbeat = (
        np.load(logits_path(results_dir, track, logits, head)) for head in HEADS
    )
    return beat, downbeat


def write_postprocessed(
    results_dir: Path,
    track: str,
    logits: str,
    heads: Logits,
    duration: float,
) -> None:
    """Write <logits>-minimal and <logits>-dbn: the beats as the app clicks them.

    Each postprocessing of Beat This! picks beats from the logits, and the
    rhythm summary of the app then folds a double-time grid, so the variant
    holds what the metronome would play.
    """
    from beat_this.model.postprocessor import Postprocessor  # noqa: PLC0415

    beat, downbeat = heads
    for postprocessing in POSTPROCESSINGS:
        beats, downbeats = Postprocessor(type=postprocessing, fps=FPS)(
            torch.from_numpy(beat), torch.from_numpy(downbeat)
        )
        bpm, beats, downbeats, meter = summarize_rhythm(
            np.asarray(beats, dtype=np.float64),
            np.asarray(downbeats, dtype=np.float64),
            duration,
        )
        write_variant(
            results_dir,
            track,
            f"{logits}-{postprocessing}",
            {
                "duration": duration,
                "bpm": float(bpm),
                "meter": int(meter),
                "beats": [float(time) for time in beats],
                "downbeats": [float(time) for time in downbeats],
            },
        )
