from argparse import Namespace
from typing import Any

import numpy as np
import torch

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.separate_audio import utils


def dict_to_namespace(data: Any) -> Any:
    if isinstance(data, dict):
        return Namespace(**{k: dict_to_namespace(v) for k, v in data.items()})
    if isinstance(data, list):
        return [dict_to_namespace(item) for item in data]
    return data


class AudioProcessor:
    def __init__(self, device: torch.device, config):
        self.device = device
        self.config = config

    def demix(self, mix: np.ndarray, model) -> dict:
        original_mix = mix
        mix_tensor = torch.from_numpy(mix).to(dtype=torch.float32, device=self.device)

        segment_size = self.config.inference.dim_t
        chunk_size = self.config.audio.hop_length * (segment_size - 1)
        step_size = int(8 * self.config.audio.sample_rate)
        window = torch.hamming_window(
            chunk_size, dtype=torch.float32, device=self.device
        )

        instruments_count = len(self.config.training.instruments)
        result_shape = (instruments_count, *mix_tensor.shape)
        result = torch.zeros(result_shape, dtype=torch.float32, device=self.device)
        counter = torch.zeros(result_shape, dtype=torch.float32, device=self.device)

        total_steps = (mix_tensor.shape[1] + step_size - 1) // step_size
        progress_interval = max(1, total_steps // 100)

        with torch.no_grad():
            for step_idx, i in enumerate(range(0, mix_tensor.shape[1], step_size)):
                if step_idx % progress_interval == 0:
                    progress = step_idx / total_steps
                    send_message({"type": "progress", "progress": progress / 2})

                total_len = mix_tensor.shape[1]
                if i + chunk_size > total_len:
                    if total_len >= chunk_size:
                        part = mix_tensor[:, -chunk_size:]
                        start_pos = total_len - chunk_size
                        length = chunk_size
                    else:
                        part = mix_tensor
                        start_pos = 0
                        length = total_len
                else:
                    part = mix_tensor[:, i : i + chunk_size]
                    start_pos = i
                    length = part.shape[-1]

                x = model(part.unsqueeze(0), target=None, return_loss_breakdown=False)[
                    0
                ]
                window_slice = window[:length]
                result[..., start_pos : start_pos + length] += (
                    x[..., :length] * window_slice
                )
                counter[..., start_pos : start_pos + length] += window_slice

        outputs = (result / counter.clamp(min=1e-10)).cpu().numpy()
        instruments = list(self.config.training.instruments)
        sources = dict(zip(instruments, outputs, strict=False))

        primary_stem = self.config.training.target_instrument
        secondary_stem = next(
            (name for name in instruments if name != primary_stem), None
        )

        if primary_stem in sources:
            sources[primary_stem] = utils.match_array_shapes(
                sources[primary_stem], original_mix
            )
            if secondary_stem is not None:
                sources[secondary_stem] = original_mix - sources[primary_stem]

        return sources
