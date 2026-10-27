import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from musetric_toolkit.rhythm_bench.track_files import track_files
from musetric_toolkit.rhythm_bench.variant import read_variant

# Each window of a `windows` result, with 3 s before and after, rendered once
# per variant: the audio at 0.7 with the variant's clicks on top, a short
# decaying tone like the metronome of the player, higher on a downbeat.

MARGIN_SECONDS = 3.0
MIX_GAIN = 0.7
CLICK_SECONDS = 0.05
CLICK_DECAY_SECONDS = 0.015
CLICK_GAIN = 0.3
BEAT_HZ = 880.0
DOWNBEAT_HZ = 1320.0


def click(rate: int, frequency: float) -> np.ndarray:
    n = np.arange(int(rate * CLICK_SECONDS))
    return (
        np.sin(2 * np.pi * frequency * n / rate)
        * np.exp(-n / (rate * CLICK_DECAY_SECONDS))
        * CLICK_GAIN
    )


def render(
    audio: np.ndarray,
    rate: int,
    start: float,
    beats: np.ndarray,
    downbeats: np.ndarray,
) -> np.ndarray:
    mixed = audio * MIX_GAIN
    end = start + len(audio) / rate
    downbeat_set = {round(float(time), 6) for time in downbeats}
    for beat in beats:
        if not start <= beat < end:
            continue
        offset = round((beat - start) * rate)
        frequency = DOWNBEAT_HZ if round(float(beat), 6) in downbeat_set else BEAT_HZ
        tick = click(rate, frequency)
        stop = min(len(mixed), offset + len(tick))
        mixed[offset:stop] += tick[: stop - offset, None]
    return np.clip(mixed, -1, 1)


def main(args) -> None:
    results_dir = Path(args.results_dir)
    windows = json.loads(Path(args.windows).read_text())
    files = track_files(args.audio_path, args.file_name)
    out_dir = results_dir / "listen"
    out_dir.mkdir(parents=True, exist_ok=True)
    for region in windows["regions"]:
        track = region["track"]
        source = files[track]
        rate = sf.info(str(source)).samplerate
        start = max(0.0, region["start"] - MARGIN_SECONDS)
        end = region["end"] + MARGIN_SECONDS
        audio, _ = sf.read(
            str(source),
            start=int(start * rate),
            stop=int(end * rate),
            dtype="float32",
            always_2d=True,
        )
        at = int(region["start"])
        for variant in windows["variants"]:
            clicks = read_variant(results_dir, track, variant)
            name = f"{track}-{at // 60}m{at % 60:02d}s-{variant}.wav"
            sf.write(
                str(out_dir / name),
                render(audio, rate, start, clicks.beats, clicks.downbeats),
                rate,
                subtype="PCM_16",
            )
            logging.info("%s", name)
