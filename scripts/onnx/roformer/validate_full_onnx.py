"""Reference generator for the full-graph validation.

Takes the loudest window of --frames frames from a track, runs the torch
FullSeparator on it, and writes the input chunk + reference vocals as raw
little-endian f32, planar stereo. This repository carries no comparator that
scores an execution provider against them.
"""

# ruff: noqa: T201, N806

import argparse
from pathlib import Path

import build_full_onnx as bfo  # type: ignore[import-not-found]
import numpy as np
import torch

from musetric_toolkit.separate_audio import utils
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.roformer.attend import Attend


def loudest_window(mix: np.ndarray, length: int) -> int:
    """Offset of the highest-energy window of `length` samples."""
    power = (mix**2).mean(axis=0)
    total = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    return int(np.argmax(total[length:] - total[:-length]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--frames",
        type=int,
        default=bfo.T,
        help="chunk frame count T (must match the model under test).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="cpu (default) or cuda. Large T (e.g. 1100) needs cuda — "
        "the T² attention sim overflows CPU; on cuda flash SDPA avoids "
        "materializing it.",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # the DSP functions read these module globals at call time
    bfo.T = args.frames
    bfo.TSAMP = bfo.HOP * (args.frames - 1)
    PACKED, T, TSAMP = bfo.PACKED, bfo.T, bfo.TSAMP
    dev = torch.device(args.device)

    mix = utils.normalize(
        read_audio_file(str(args.source), 44100, 2), max_peak=0.9, min_peak=0.0
    )  # [2, samples]
    start = loudest_window(mix, TSAMP)
    chunk = np.ascontiguousarray(mix[:, start : start + TSAMP], dtype=np.float32)
    x = torch.from_numpy(chunk).unsqueeze(0).to(dev)  # [1,2,TSAMP]
    print(f"window at {start / 44100:.1f}s ({start} samples)")

    model = bfo.load_model(args.checkpoint, args.config)
    if dev.type == "cuda":
        # use flash SDPA so the T² sim is never materialized (fits 6 GB)
        for m in model.modules():
            if isinstance(m, Attend):
                m.flash = True
    full = bfo.FullSeparator(model).to(dev).eval()
    with torch.no_grad():
        full.model.net_forward(torch.randn(1, PACKED, T, 2, device=dev))  # warm rotary
        vocals = full(x)[0].float().cpu().numpy().astype(np.float32)  # [2, TSAMP]

    chunk.tofile(args.out_dir / "full_input.f32")
    vocals.tofile(args.out_dir / "full_ref_vocals.f32")
    print(f"wrote full_input.f32 + full_ref_vocals.f32 ({TSAMP} samples, stereo)")
    vmax = float(np.abs(vocals).max())
    vrms = float(np.sqrt((vocals**2).mean()))
    print(f"ref vocals: max={vmax:.4f} rms={vrms:.4f}")


if __name__ == "__main__":
    main()
