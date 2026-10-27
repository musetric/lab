# ruff: noqa: T201

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch import nn

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_FILENAME = "beat_this.onnx"
FILTERBANK_FILENAME = "mel-filterbank.bin"
CONFIG_FILENAME = "config.json"
CHECKPOINT_NAME = "final0"
OPSET = 17
CHUNK_SIZE = 1500
BORDER_SIZE = 6
MEL_BINS = 128
PEAK_KERNEL = 7
PEAK_THRESHOLD = 0.0
DEDUPLICATE_WIDTH = 1
WINDOW_TOLERANCE = 1e-6


class DynamicPartialFTTransformer(nn.Module):

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.attnF = source.attnF
        self.ffF = source.ffF
        self.attnT = source.attnT
        self.ffT = source.ffT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _batch, channels, freqs, times = x.shape
        across_freqs = x.permute(0, 3, 2, 1).reshape(-1, freqs, channels)
        across_freqs = across_freqs + self.attnF(across_freqs)
        across_freqs = across_freqs + self.ffF(across_freqs)
        across_times = across_freqs.reshape(-1, times, freqs, channels)
        across_times = across_times.permute(0, 2, 1, 3).reshape(-1, times, channels)
        across_times = across_times + self.attnT(across_times)
        across_times = across_times + self.ffT(across_times)
        restored = across_times.reshape(-1, freqs, times, channels)
        return restored.permute(0, 3, 1, 2)


class BeatThisFrames(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, spect: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.model(spect)
        return prediction["beat"], prediction["downbeat"]


def patch_for_export(model: nn.Module) -> None:
    for block in model.frontend.blocks:
        block.partial = DynamicPartialFTTransformer(block.partial)


def check_analytic_window(window: torch.Tensor) -> None:
    size = window.shape[0]
    offsets = torch.arange(size, dtype=torch.float64)
    analytic = 0.5 - 0.5 * torch.cos(2.0 * math.pi * offsets / size)
    deviation = float((window.to(torch.float64) - analytic).abs().max())
    if deviation > WINDOW_TOLERANCE:
        raise ValueError(
            "reference window is not a periodic Hann window "
            f"(max deviation {deviation:.3e}); the WebGPU frame shader "
            "generates it analytically"
        )
    print(f"window: periodic Hann confirmed (max deviation {deviation:.3e})")


class MelContract:

    def __init__(self, reference: nn.Module) -> None:
        spectrogram = reference.spect_class.spectrogram
        self.n_fft = int(spectrogram.n_fft)
        self.hop_length = int(spectrogram.hop_length)
        self.sample_rate = int(reference.spect_class.sample_rate)
        self.log_multiplier = reference.log_multiplier
        self.window = spectrogram.window.detach().clone()
        self.filterbank = reference.spect_class.mel_scale.fb.detach().clone()


def build_modules(models_path: str) -> tuple[BeatThisFrames, MelContract]:
    os.environ["TORCH_HOME"] = models_path
    from beat_this.inference import load_model  # noqa: PLC0415
    from beat_this.preprocessing import LogMelSpect  # noqa: PLC0415

    reference = LogMelSpect(device="cpu")
    model = load_model(CHECKPOINT_NAME, "cpu")
    patch_for_export(model)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"loaded Beat This! {CHECKPOINT_NAME}: {parameters / 1e6:.2f}M parameters")
    return BeatThisFrames(model).eval(), MelContract(reference)


def export_model(model: BeatThisFrames, path: Path) -> None:
    dummy = torch.zeros(1, CHUNK_SIZE, MEL_BINS, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["spect"],
        output_names=["beat", "downbeat"],
        dynamic_axes={
            "spect": {0: "windows", 1: "frames"},
            "beat": {0: "windows", 1: "frames"},
            "downbeat": {0: "windows", 1: "frames"},
        },
        opset_version=OPSET,
    )
    print(f"exported {path} ({path.stat().st_size / 1e6:.1f} MB)")


def write_filterbank(path: Path, mel: MelContract) -> None:
    filterbank = mel.filterbank.to(torch.float32).contiguous()
    path.write_bytes(filterbank.numpy().tobytes())
    rows, columns = filterbank.shape
    print(f"wrote {path} ({rows}x{columns} float32, {path.stat().st_size} B)")


def write_config(path: Path, mel: MelContract) -> None:
    bins, mels = mel.filterbank.shape
    payload = {
        "modelType": "beat-this-final0",
        "sampleRate": mel.sample_rate,
        "mono": True,
        "downmix": "mean",
        "nFft": mel.n_fft,
        "hopLength": mel.hop_length,
        "melBins": mels,
        "window": "hann",
        "logMultiplier": mel.log_multiplier,
        "fps": mel.sample_rate / mel.hop_length,
        "frameDuration": mel.hop_length / mel.sample_rate,
        "chunkSize": CHUNK_SIZE,
        "borderSize": BORDER_SIZE,
        "overlapMode": "keep_first",
        "peakKernel": PEAK_KERNEL,
        "peakThreshold": PEAK_THRESHOLD,
        "deduplicateWidth": DEDUPLICATE_WIDTH,
        "melFilterbank": {
            "file": FILTERBANK_FILENAME,
            "shape": [int(bins), int(mels)],
            "dtype": "float32",
            "layout": "row-major",
        },
        "model": {
            "file": MODEL_FILENAME,
            "input": {
                "name": "spect",
                "shape": ["windows", "frames", int(mels)],
                "dtype": "float32",
            },
            "outputs": [
                {"name": "beat", "shape": ["windows", "frames"], "dtype": "float32"},
                {
                    "name": "downbeat",
                    "shape": ["windows", "frames"],
                    "dtype": "float32",
                },
            ],
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="directory to write beat_this.onnx, mel-filterbank.bin and config.json",
    )
    parser.add_argument(
        "--models-path",
        required=True,
        help="cache dir for the Beat This! checkpoint download (sets TORCH_HOME)",
    )
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    model, mel = build_modules(args.models_path)
    check_analytic_window(mel.window)

    export_model(model, out / MODEL_FILENAME)
    write_filterbank(out / FILTERBANK_FILENAME, mel)
    write_config(out / CONFIG_FILENAME, mel)


if __name__ == "__main__":
    main()
