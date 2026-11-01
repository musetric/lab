from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from musetric_toolkit.common import envs
from musetric_toolkit.common.logger import send_message
from musetric_toolkit.common.model_files import ensure_model_file
from musetric_toolkit.parity_audio.product_stft import analyze, synthesize
from musetric_toolkit.parity_audio.unit_plans import fold, unit_window, vocals_plan
from musetric_toolkit.separate_audio.roformer.lead_backing_roformer import (
    LeadBackingRoformer,
)

SAMPLE_RATE = 44100
CHANNELS = 2
N_FFT = 2048
HOP = 441
FRAMES = 1100
CHUNK_SAMPLES = HOP * (FRAMES - 1)


def load_student(models_path: Path, device: torch.device) -> LeadBackingRoformer:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    weights_path = models_path / envs.student_weights_rel_path
    config_path = models_path / envs.student_config_rel_path
    ensure_model_file(envs.student_weights_url, weights_path, "Model weights")
    ensure_model_file(envs.student_config_url, config_path, "Model configuration")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = LeadBackingRoformer(**config["model"])
    model.load_state_dict(load_file(weights_path))
    return model.to(device).eval()


def pack(spectrum: np.ndarray) -> np.ndarray:
    channels, frames, bins = spectrum.shape
    packed = np.zeros((1, bins * channels, frames, 2))
    for channel in range(channels):
        packed[0, channel::channels, :, 0] = spectrum[channel].T.real
        packed[0, channel::channels, :, 1] = spectrum[channel].T.imag
    return packed


def apply_masks(spectrum: np.ndarray, masks: np.ndarray, chained: int) -> np.ndarray:
    channels = spectrum.shape[0]
    masked = spectrum.copy()
    for plane in masks[:chained]:
        for channel in range(channels):
            mask = plane[channel::channels, :, 0] + 1j * plane[channel::channels, :, 1]
            masked[channel] = masked[channel] * mask.T
    return masked


def torch_masks(model: LeadBackingRoformer, stft_repr: np.ndarray) -> np.ndarray:
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
        return averaged[0].double().cpu().numpy()


def separate_mixture(
    model: LeadBackingRoformer,
    mixture: np.ndarray,
    progress_share: float = 1.0,
    first_unit: (
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None] | None
    ) = None,
) -> np.ndarray:
    samples = mixture.shape[1]
    plan = vocals_plan(SAMPLE_RATE, samples, CHUNK_SAMPLES)
    chunks = []
    for index in range(len(plan.units)):
        send_message(
            {"type": "progress", "progress": progress_share * index / len(plan.units)}
        )
        chunk = unit_window(mixture, plan, index)
        spectrum = analyze(chunk, N_FFT, HOP, FRAMES)
        stft_repr = pack(spectrum)
        masks = torch_masks(model, stft_repr)
        output = np.concatenate(
            [
                synthesize(apply_masks(spectrum, masks, s), N_FFT, HOP, CHUNK_SAMPLES)
                for s in range(1, len(masks) + 1)
            ]
        )
        chunks.append(output)
        if index == 0 and first_unit is not None:
            first_unit(chunk, stft_repr, masks, output)
    return fold(plan, chunks, samples)
