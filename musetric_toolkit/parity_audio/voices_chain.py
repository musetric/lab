import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

from musetric_toolkit.common import envs
from musetric_toolkit.common.logger import send_message
from musetric_toolkit.common.model_files import ensure_model_file
from musetric_toolkit.parity_audio.case_writer import CaseWriter
from musetric_toolkit.parity_audio.product_stft import analyze, synthesize
from musetric_toolkit.parity_audio.unit_plans import (
    LEAD_BACKING_COMPENSATE,
    fold,
    lead_backing_plan,
    normalize_peak,
    unit_window,
)
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.mdx_net_separator import MDXNetSeparator

# The voices step: the vocals stem, scaled to a unit peak, is padded and cut into
# 256-frame chunks with a quarter overlap (packages/server/src/unit_plan.rs
# lead_backing); the model returns the backing voices of each chunk, and the
# lead is what remains. The original is UVR-MDXNet KARA2, published as ONNX, so
# the export row compares our rewrite with that graph.

SAMPLE_RATE = 44100
CHANNELS = 2
N_FFT = 5120
HOP = 1024
FRAMES = 256
DIM_F = 2048
CHUNK_SAMPLES = HOP * (FRAMES - 1)


def pack(spectrum: np.ndarray) -> np.ndarray:
    """[channels, frames, bins] -> model input [1, 2 * channels, DIM_F, frames]."""
    channels = spectrum.shape[0]
    packed = np.zeros((1, 2 * channels, DIM_F, spectrum.shape[1]))
    for channel in range(channels):
        packed[0, 2 * channel] = spectrum[channel, :, :DIM_F].real.T
        packed[0, 2 * channel + 1] = spectrum[channel, :, :DIM_F].imag.T
    packed[:, :, :3, :] = 0.0
    return packed


def unpack(output: np.ndarray) -> np.ndarray:
    channels = output.shape[1] // 2
    spectrum = np.zeros(
        (channels, output.shape[3], N_FFT // 2 + 1), dtype=np.complex128
    )
    for channel in range(channels):
        spectrum[channel, :, :DIM_F] = (
            output[0, 2 * channel].T + 1j * output[0, 2 * channel + 1].T
        )
    return spectrum


def cpu_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run_session(session: ort.InferenceSession, model_input: np.ndarray) -> np.ndarray:
    (output,) = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: model_input.astype(np.float32)},
    )
    return output.astype(np.float64)


def author_voices(models_path: Path, vocals_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Lead and backing by UVR-MDXNet inference as the lab runs it."""
    separator = MDXNetSeparator(
        model_path=models_path / envs.karaoke_mdx_model_rel_path,
        model_data_path=models_path / envs.mdx_model_data_rel_path,
        sample_rate=SAMPLE_RATE,
    )
    with tempfile.TemporaryDirectory() as directory:
        lead_path = str(Path(directory) / "lead.wav")
        backing_path = str(Path(directory) / "backing.wav")
        separator.separate_audio(vocals_path, lead_path, backing_path)
        lead = read_audio_file(lead_path, SAMPLE_RATE, CHANNELS).astype(np.float64)
        backing = read_audio_file(backing_path, SAMPLE_RATE, CHANNELS).astype(
            np.float64
        )
    return lead, backing


def run(args, writer: CaseWriter) -> None:
    models_path = Path(args.models_path)
    original_path = models_path / envs.karaoke_mdx_model_rel_path
    ensure_model_file(envs.karaoke_mdx_model_url, original_path, "MDX karaoke model")
    ensure_model_file(
        envs.mdx_model_data_url,
        models_path / envs.mdx_model_data_rel_path,
        "MDX model data",
    )
    original = cpu_session(original_path)
    ours = cpu_session(Path(args.onnx))

    vocals = read_audio_file(args.audio_path, SAMPLE_RATE, CHANNELS).astype(np.float64)
    samples = vocals.shape[1]
    peak = float(np.abs(vocals).max())
    scaled = vocals / peak
    layout = lead_backing_plan(samples, CHUNK_SAMPLES, N_FFT)
    padded = np.zeros((CHANNELS, layout.mixture_samples))
    padded[:, layout.trim : layout.trim + samples] = scaled
    writer.meta.update(
        {
            "sampleRate": SAMPLE_RATE,
            "channels": CHANNELS,
            "samples": samples,
            "units": [[unit.start, unit.length] for unit in layout.plan.units],
        }
    )

    chunks = []
    for index in range(len(layout.plan.units)):
        send_message(
            {"type": "progress", "progress": index / len(layout.plan.units) / 2}
        )
        chunk = unit_window(padded, layout.plan, index)
        spectrum = analyze(chunk, N_FFT, HOP, FRAMES)
        model_input = pack(spectrum)
        output = run_session(original, model_input)
        unit_output = synthesize(unpack(output), N_FFT, HOP, CHUNK_SAMPLES)
        chunks.append(unit_output)
        if index == 0:
            writer.tensor("unit.input", "reference", chunk)
            writer.tensor("model.input", "reference", model_input)
            writer.tensor("model.output", "original", output)
            writer.tensor("model.output", "onnx-cpu", run_session(ours, model_input))
            writer.tensor("unit.output", "reference", unit_output)

    folded = fold(layout.plan, chunks, layout.mixture_samples)
    backing_norm = folded[:, layout.trim : layout.trim + samples]
    writer.tensor("step.backing", "reference", normalize_peak(backing_norm * peak))
    writer.tensor(
        "step.lead",
        "reference",
        normalize_peak((scaled - backing_norm * LEAD_BACKING_COMPENSATE) * peak),
    )

    send_message({"type": "progress", "progress": 0.75})
    lead, backing = author_voices(models_path, args.audio_path)
    writer.tensor("step.lead", "author", lead[:, :samples])
    writer.tensor("step.backing", "author", backing[:, :samples])
