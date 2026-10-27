from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml
from safetensors.torch import load_file

from musetric_toolkit.common import envs
from musetric_toolkit.common.logger import send_message
from musetric_toolkit.common.model_files import ensure_model_file
from musetric_toolkit.parity_audio.case_writer import CaseWriter
from musetric_toolkit.parity_audio.product_stft import analyze, synthesize
from musetric_toolkit.parity_audio.unit_plans import (
    fold,
    normalize_peak,
    unit_window,
    vocals_plan,
)
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.roformer.pooled_mel_band_roformer import (
    PooledMelBandRoformer,
)
from musetric_toolkit.separate_audio.roformer_utils import (
    AudioProcessor,
    dict_to_namespace,
)

# The vocals half of the separation step: the product feeds the core the STFT of
# an 1100-frame chunk (packages/server/src/analysis/models.rs VOCALS_GEOMETRY),
# multiplies the spectrum by the masks it returns and folds the chunks with the
# host's Hamming weights. The original is the student distilled from Aname
# Duality, its published weights in torch.

SAMPLE_RATE = 44100
CHANNELS = 2
N_FFT = 2048
HOP = 441
FRAMES = 1100
CHUNK_SAMPLES = HOP * (FRAMES - 1)


def load_student(models_path: Path, device: torch.device):
    weights_path = models_path / envs.student_weights_rel_path
    config_path = models_path / envs.student_config_rel_path
    ensure_model_file(envs.student_weights_url, weights_path, "Model weights")
    ensure_model_file(envs.student_config_url, config_path, "Model configuration")
    with config_path.open(encoding="utf-8") as handle:
        config = dict_to_namespace(yaml.safe_load(handle))
    model = PooledMelBandRoformer(**vars(config.model))
    model.load_state_dict(load_file(weights_path))
    return model.to(device).eval(), config


def pack(spectrum: np.ndarray) -> np.ndarray:
    """[channels, frames, bins] -> stft_repr [1, bins * channels, frames, 2]."""
    channels, frames, bins = spectrum.shape
    packed = np.zeros((1, bins * channels, frames, 2))
    for channel in range(channels):
        packed[0, channel::channels, :, 0] = spectrum[channel].T.real
        packed[0, channel::channels, :, 1] = spectrum[channel].T.imag
    return packed


def apply_masks(spectrum: np.ndarray, masks: np.ndarray) -> np.ndarray:
    channels = spectrum.shape[0]
    masked = np.empty_like(spectrum)
    for channel in range(channels):
        mask = (
            masks[0, channel::channels, :, 0] + 1j * masks[0, channel::channels, :, 1]
        )
        masked[channel] = spectrum[channel] * mask.T
    return masked


def torch_masks(model: PooledMelBandRoformer, stft_repr: np.ndarray) -> np.ndarray:
    """The per-bin masks of the original forward: band masks scattered and averaged."""
    device = next(model.parameters()).device
    with torch.inference_mode():
        tensor = torch.from_numpy(stft_repr.astype(np.float32)).to(device)
        masks = model.net_forward(tensor)
        index = model.freq_indices.to(device)
        summed = torch.zeros(
            (1, masks.shape[1], tensor.shape[1], *masks.shape[3:]), device=device
        ).index_add_(2, index, masks)
        denom = model.num_bands_per_freq.to(device).repeat_interleave(CHANNELS)
        averaged = summed / denom.clamp(min=1e-8).view(1, 1, -1, 1, 1)
        return averaged[:, 0].double().cpu().numpy()


def onnx_masks(onnx_path: Path, stft_repr: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path), options, providers=["CPUExecutionProvider"]
    )
    (masks,) = session.run(["masks"], {"stft_repr": stft_repr.astype(np.float32)})
    return masks.astype(np.float64)


def author_demix(
    model: PooledMelBandRoformer, config, mixture: np.ndarray
) -> np.ndarray:
    """Vocals by the model's published inference: the lab's separator, 11 s
    windows every 8 s with Hamming weights, in fp32."""
    device = next(model.parameters()).device
    sources = AudioProcessor(device, config).demix(mixture.astype(np.float32), model)
    return sources[config.training.target_instrument].astype(np.float64)


def run(args, writer: CaseWriter) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, config = load_student(Path(args.models_path), device)

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

    chunks = []
    for index in range(len(plan.units)):
        send_message({"type": "progress", "progress": index / len(plan.units) / 2})
        chunk = unit_window(mixture, plan, index)
        spectrum = analyze(chunk, N_FFT, HOP, FRAMES)
        stft_repr = pack(spectrum)
        masks = torch_masks(model, stft_repr)
        output = synthesize(apply_masks(spectrum, masks), N_FFT, HOP, CHUNK_SAMPLES)
        chunks.append(output)
        if index == 0:
            writer.tensor("unit.input", "reference", chunk)
            writer.tensor("model.input", "reference", stft_repr)
            writer.tensor("model.output", "original", masks)
            writer.tensor(
                "model.output", "onnx-cpu", onnx_masks(Path(args.onnx), stft_repr)
            )
            writer.tensor("unit.output", "reference", output)

    raw_vocals = fold(plan, chunks, samples)
    writer.tensor("step.vocals", "reference", normalize_peak(raw_vocals))
    writer.tensor(
        "step.instrumental", "reference", normalize_peak(mixture - raw_vocals)
    )

    send_message({"type": "progress", "progress": 0.75})
    author = author_demix(model, config, source)
    writer.tensor("step.vocals", "author", normalize_peak(author))
    writer.tensor("step.instrumental", "author", normalize_peak(source - author))
