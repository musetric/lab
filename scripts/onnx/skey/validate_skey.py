# ruff: noqa: T201

import argparse
import contextlib
import sys
from pathlib import Path

import onnxruntime as ort
import torch

from musetric_toolkit.key_audio.skey.key_detection import (
    key_map,
    load_audio,
    load_checkpoint,
    load_model_components,
)
from musetric_toolkit.key_audio.skey_checkpoint import ensure_checkpoint

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")


def torch_key(hcqt, chromanet, crop_fn, audio, device):
    with torch.no_grad():
        cropped = crop_fn(hcqt(audio.unsqueeze(0)), torch.zeros(1).to(device))
        probs = torch.softmax(torch.mean(chromanet(cropped), dim=0), dim=-1)
    idx = int(probs.argmax())
    return idx, float(probs[idx])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="directory of .flac files to validate on",
    )
    parser.add_argument(
        "--models-path",
        required=True,
        help="cache dir for the S-KEY checkpoint download",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = load_checkpoint(ensure_checkpoint(args.models_path))
    sample_rate = int(ckpt["audio"]["sr"])
    hcqt, chromanet, crop_fn = load_model_components(ckpt, device)

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    ok = 0
    tracks = sorted(args.audio.glob("*.flac"))
    for track in tracks:
        audio = load_audio(str(track), sample_rate).to(device)
        ref_idx, ref_conf = torch_key(hcqt, chromanet, crop_fn, audio, device)

        onnx_probs = session.run(None, {"audio": audio.numpy()})[0]
        onnx_idx = int(onnx_probs.argmax())
        match = onnx_idx == ref_idx
        ok += int(match)
        print(
            f"{track.stem:18s} {key_map[ref_idx]:9s} "
            f"ref={ref_idx:2d} onnx={onnx_idx:2d} "
            f"conf {ref_conf:.4f}/{float(onnx_probs[onnx_idx]):.4f} "
            f"{'OK' if match else 'MISMATCH'}"
        )

    print(f"\nkey argmax agreement: {ok}/{len(tracks)}")


if __name__ == "__main__":
    main()
