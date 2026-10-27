# ruff: noqa: T201

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

from musetric_toolkit.key_audio.skey.convnext import (
    ConvNeXtBlock,
    TimeDownsamplingBlock,
)
from musetric_toolkit.key_audio.skey.key_detection import (
    key_map,
    load_checkpoint,
    load_model_components,
)
from musetric_toolkit.key_audio.skey_checkpoint import ensure_checkpoint

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

CROP_HEIGHT = 84
CHROMA_BINS = 12
OPSET = 17
TRACE_SECONDS = 10


class DynLayerNorm(nn.Module):
    def forward(self, x, normalized_shape):
        dims = tuple(range(x.dim() - len(normalized_shape), x.dim()))
        mean = x.mean(dim=dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5)


class GlobalTimePool(nn.Module):
    def forward(self, x):
        return x.mean(dim=3, keepdim=True)


def patch_for_export(chromanet: nn.Module) -> None:
    for module in chromanet.modules():
        if isinstance(module, (TimeDownsamplingBlock, ConvNeXtBlock)):
            module.norm = DynLayerNorm()
    chromanet.global_average_pool = GlobalTimePool()


class KeyPipeline(nn.Module):
    def __init__(self, hcqt: nn.Module, chromanet: nn.Module):
        super().__init__()
        self.hcqt = hcqt
        self.chromanet = chromanet

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        hc = self.hcqt(audio.unsqueeze(0))
        cropped = hc[:, :, 0:CROP_HEIGHT, :]
        chroma_probs = self.chromanet(cropped)
        mean_probs = torch.mean(chroma_probs, dim=0)
        return torch.softmax(mean_probs, dim=-1)


def build_pipeline(models_path: str) -> tuple[KeyPipeline, int]:
    ckpt_path = ensure_checkpoint(models_path)
    ckpt = load_checkpoint(ckpt_path)
    sample_rate = int(ckpt["audio"]["sr"])
    hcqt, chromanet, _ = load_model_components(ckpt, torch.device("cpu"))
    patch_for_export(chromanet)
    print(f"loaded S-KEY: sample_rate={sample_rate}")
    return KeyPipeline(hcqt, chromanet).eval(), sample_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="directory to write skey.onnx and config.json into",
    )
    parser.add_argument(
        "--models-path",
        required=True,
        help="cache dir for the S-KEY checkpoint download",
    )
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    pipe, sample_rate = build_pipeline(args.models_path)

    dummy = torch.zeros(1, sample_rate * TRACE_SECONDS, dtype=torch.float32)
    onnx_path = out / "skey.onnx"
    torch.onnx.export(
        pipe,
        (dummy,),
        str(onnx_path),
        input_names=["audio"],
        output_names=["probs"],
        dynamic_axes={"audio": {1: "samples"}},
        opset_version=OPSET,
    )
    size_kb = onnx_path.stat().st_size / 1024
    print(f"exported {onnx_path} ({size_kb:.0f} KB)")

    config_payload = {
        "modelType": "skey-vqt-chromanet",
        "sampleRate": sample_rate,
        "mono": True,
        "peakNormalize": True,
        "numKeys": len(key_map),
        "input": {"name": "audio", "shape": [1, "samples"], "dtype": "float32"},
        "output": {"name": "probs", "shape": [24], "dtype": "float32"},
        "keyMap": [key_map[i] for i in range(len(key_map))],
    }
    (out / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out / 'config.json'}")


if __name__ == "__main__":
    main()
