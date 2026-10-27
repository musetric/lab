# ruff: noqa: T201

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

SAMPLE_RATE = 22050
STEREO_NDIM = 2


def load_mono(path: Path) -> np.ndarray:
    import soxr  # noqa: PLC0415
    from beat_this.preprocessing import load_audio  # noqa: PLC0415

    signal, sample_rate = load_audio(str(path))
    if signal.ndim == STEREO_NDIM:
        signal = signal.mean(1)
    if sample_rate != SAMPLE_RATE:
        signal = soxr.resample(signal, in_rate=sample_rate, out_rate=SAMPLE_RATE)
    return np.asarray(signal, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--models-path",
        required=True,
        help="cache dir for the Beat This! checkpoint download (sets TORCH_HOME)",
    )
    args = parser.parse_args()

    os.environ["TORCH_HOME"] = args.models_path
    from beat_this.preprocessing import LogMelSpect  # noqa: PLC0415

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_mel = LogMelSpect(device="cpu")

    for track in sorted(args.audio.glob("*.flac")):
        audio = load_mono(track)
        with torch.inference_mode():
            spect = log_mel(torch.from_numpy(audio)).numpy().astype(np.float32)
        (out / f"{track.stem}.pcm").write_bytes(audio.tobytes())
        (out / f"{track.stem}.mel").write_bytes(spect.tobytes())
        (out / f"{track.stem}.json").write_text(
            json.dumps({"frames": int(spect.shape[0]), "mels": int(spect.shape[1])}),
            encoding="utf-8",
        )
        print(f"{track.stem}: samples {len(audio)} spect {spect.shape}")


if __name__ == "__main__":
    main()
