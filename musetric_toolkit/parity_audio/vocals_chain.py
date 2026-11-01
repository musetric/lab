from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.distill.inference import overlap_add
from musetric_toolkit.parity_audio.case_writer import CaseWriter
from musetric_toolkit.parity_audio.unit_plans import normalize_peak, vocals_plan
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.roformer.lead_backing_roformer import (
    LeadBackingRoformer,
)
from musetric_toolkit.separate_audio.student_separator import (
    CHANNELS,
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    load_student,
    separate_mixture,
)

AUTHOR_STEMS = ("vocals", "backing")


def onnx_masks(onnx_path: Path, stft_repr: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path), options, providers=["CPUExecutionProvider"]
    )
    (masks,) = session.run(["masks"], {"stft_repr": stft_repr.astype(np.float32)})
    return masks.astype(np.float64)


def author_separate(model: LeadBackingRoformer, mixture: np.ndarray) -> np.ndarray:
    def forward(windows: torch.Tensor) -> torch.Tensor:
        out = model.separate(windows)
        return torch.stack([out[stem] for stem in AUTHOR_STEMS], dim=1)

    return overlap_add(forward, mixture.astype(np.float32)).astype(np.float64)


def write_steps(
    writer: CaseWriter, source: str, mixture: np.ndarray, vocals, backing
) -> None:
    writer.tensor("step.vocals", source, normalize_peak(vocals))
    writer.tensor("step.instrumental", source, normalize_peak(mixture - vocals))
    writer.tensor("step.lead", source, normalize_peak(vocals - backing))
    writer.tensor("step.backing", source, normalize_peak(backing))


def run(args, writer: CaseWriter) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_student(Path(args.models_path), device)

    source = read_audio_file(args.audio_path, SAMPLE_RATE, CHANNELS).astype(np.float64)
    mixture = normalize_peak(source)
    samples = mixture.shape[1]
    plan = vocals_plan(SAMPLE_RATE, samples, CHUNK_SAMPLES)
    writer.meta.update(
        {
            "sampleRate": SAMPLE_RATE,
            "channels": CHANNELS,
            "samples": samples,
            "units": [[unit.start, unit.length] for unit in plan.units],
        }
    )

    def first_unit(chunk, stft_repr, masks, output) -> None:
        writer.tensor("unit.input", "reference", chunk)
        writer.tensor("model.input", "reference", stft_repr)
        writer.tensor("model.output", "original", masks)
        writer.tensor(
            "model.output", "onnx-cpu", onnx_masks(Path(args.onnx), stft_repr)
        )
        writer.tensor("unit.output", "reference", output)

    folded = separate_mixture(model, mixture, 0.5, first_unit)
    write_steps(writer, "reference", mixture, folded[:CHANNELS], folded[CHANNELS:])

    send_message({"type": "progress", "progress": 0.75})
    author = author_separate(model, source)
    write_steps(writer, "author", source, author[0], author[1])
