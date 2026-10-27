from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from musetric_toolkit.key_audio.skey_checkpoint import ensure_checkpoint
from musetric_toolkit.key_audio.skey_runner import run_skey
from musetric_toolkit.parity_audio.case_writer import CaseWriter
from musetric_toolkit.parity_audio.host_audio import POWER, read_mono

SAMPLE_RATE = 22050


def torch_probs(models_path: str, audio: np.ndarray) -> np.ndarray:
    from musetric_toolkit.key_audio.skey.key_detection import (  # noqa: PLC0415
        load_checkpoint,
        load_model_components,
    )

    checkpoint = load_checkpoint(ensure_checkpoint(models_path))
    hcqt, chromanet, crop = load_model_components(checkpoint, torch.device("cpu"))
    with torch.no_grad():
        batch = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        logits = chromanet(crop(hcqt(batch), torch.zeros(1)))
        probs = torch.softmax(torch.mean(logits, dim=0), dim=-1)
    return probs.double().numpy()


def run(args, writer: CaseWriter) -> None:
    audio = read_mono(args.audio_path, SAMPLE_RATE, POWER)
    peak = float(np.abs(audio).max())
    normalized = audio / peak if peak > 0 else audio
    writer.meta.update({"sampleRate": SAMPLE_RATE, "samples": audio.shape[0]})
    writer.tensor("unit.input", "reference", audio)
    writer.tensor("model.input", "reference", normalized[np.newaxis])

    probs = torch_probs(args.models_path, normalized)
    writer.tensor("model.output", "original", probs)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        args.onnx, options, providers=["CPUExecutionProvider"]
    )
    (onnx_probs,) = session.run(
        ["probs"], {"audio": normalized[np.newaxis].astype(np.float32)}
    )
    writer.tensor("model.output", "onnx-cpu", onnx_probs)
    writer.tensor("result.key", "reference", np.asarray([probs.argmax()]), "int32")

    from musetric_toolkit.key_audio.skey.key_detection import key_map  # noqa: PLC0415

    root, mode, _ = run_skey(args.audio_path, str(Path(args.models_path)))
    author = next(
        index
        for index, name in key_map.items()
        if name.lower() == f"{root} {mode}".lower()
    )
    writer.tensor("result.key", "author", np.asarray([author]), "int32")
