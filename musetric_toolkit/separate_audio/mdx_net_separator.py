import hashlib
import io
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.separate_audio import utils
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.ffmpeg.write import write_audio_file


class STFT:
    def __init__(self, n_fft: int, hop_length: int, dim_f: int, device: torch.device):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dim_f = dim_f
        self.device = device
        self.hann_window = torch.hann_window(window_length=self.n_fft, periodic=True)

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        stft_window = self.hann_window.to(input_tensor.device)
        batch_dimensions = input_tensor.shape[:-2]
        channel_dim, time_dim = input_tensor.shape[-2:]
        reshaped_tensor = input_tensor.reshape([-1, time_dim])
        stft_output = torch.stft(
            reshaped_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=stft_window,
            center=True,
            return_complex=True,
        )
        stft_output = torch.view_as_real(stft_output)
        permuted_output = stft_output.permute([0, 3, 1, 2])
        final_output = permuted_output.reshape(
            [*batch_dimensions, channel_dim, 2, -1, permuted_output.shape[-1]]
        ).reshape([*batch_dimensions, channel_dim * 2, -1, permuted_output.shape[-1]])
        return final_output[..., : self.dim_f, :]

    def inverse(self, input_tensor: torch.Tensor) -> torch.Tensor:
        stft_window = self.hann_window.to(input_tensor.device)
        batch_dimensions = input_tensor.shape[:-3]
        channel_dim, freq_dim, time_dim = input_tensor.shape[-3:]
        num_freq_bins = self.n_fft // 2 + 1
        freq_padding = torch.zeros(
            [*batch_dimensions, channel_dim, num_freq_bins - freq_dim, time_dim],
            device=input_tensor.device,
        )
        padded_tensor = torch.cat([input_tensor, freq_padding], -2)
        reshaped = padded_tensor.reshape(
            [*batch_dimensions, channel_dim // 2, 2, num_freq_bins, time_dim]
        )
        flattened = reshaped.reshape([-1, 2, num_freq_bins, time_dim])
        permuted = flattened.permute([0, 2, 3, 1])
        complex_tensor = permuted[..., 0] + permuted[..., 1] * 1.0j
        istft_result = torch.istft(
            complex_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=stft_window,
            center=True,
        )
        return istft_result.reshape([*batch_dimensions, 2, -1])


class MDXNetSeparator:
    def __init__(
        self,
        model_path: Path,
        model_data_path: Path,
        sample_rate: int,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.model_data_path = model_data_path
        self.sample_rate = sample_rate
        self.device = self._get_device()

        self.hop_length = 1024
        self.segment_size = None
        self.overlap = 0.25
        self.batch_size = 1
        self.enable_denoise = False

        self.model_run = None
        self.session = None
        self.compensate = 1.0
        self.dim_f = None
        self.dim_t = None
        self.n_fft = None
        self.primary_stem = "Vocals"
        self.stft = None

        self.n_bins = 0
        self.trim = 0
        self.chunk_size = 0
        self.gen_size = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _get_onnx_providers(self) -> list[str]:
        providers = ort.get_available_providers()
        if torch.cuda.is_available() and "CUDAExecutionProvider" in providers:
            return ["CUDAExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _get_model_hash(self) -> str:
        bytes_to_hash = 10000 * 1024
        file_size = self.model_path.stat().st_size
        with self.model_path.open("rb") as model_file:
            if file_size < bytes_to_hash:
                data = model_file.read()
            else:
                model_file.seek(file_size - bytes_to_hash, io.SEEK_SET)
                data = model_file.read()
        return hashlib.md5(data, usedforsecurity=False).hexdigest()

    def _load_model_data(self) -> dict:
        with self.model_data_path.open("r", encoding="utf-8") as handle:
            model_data = json.load(handle)
        model_hash = self._get_model_hash()
        if model_hash not in model_data:
            raise ValueError(
                f"MDX model parameters not found for hash {model_hash} in "
                f"{self.model_data_path}."
            )
        return model_data[model_hash]

    def _load_model(self) -> None:
        if self.model_run is not None:
            return

        model_data = self._load_model_data()
        self.compensate = model_data.get("compensate", 1.0)
        self.dim_f = model_data["mdx_dim_f_set"]
        self.dim_t = 2 ** model_data["mdx_dim_t_set"]
        self.n_fft = model_data["mdx_n_fft_scale_set"]
        self.primary_stem = model_data.get("primary_stem", "Vocals")
        self.segment_size = self.dim_t

        self.stft = STFT(self.n_fft, self.hop_length, self.dim_f, self.device)

        options = ort.SessionOptions()
        if logging.getLogger().level > logging.DEBUG:
            options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self._get_onnx_providers(),
            sess_options=options,
        )
        self.model_run = self.session.run

    def _initialize_model_settings(self) -> None:
        self.n_bins = self.n_fft // 2 + 1
        self.trim = self.n_fft // 2
        self.chunk_size = self.hop_length * (self.segment_size - 1)
        self.gen_size = self.chunk_size - 2 * self.trim

    def _run_model(self, mix: torch.Tensor, is_match_mix: bool) -> np.ndarray:
        spek = self.stft(mix.to(self.device))
        spek[:, :, :3, :] *= 0

        if is_match_mix:
            spec_pred = spek.detach().cpu().numpy()
        elif self.enable_denoise:
            spec_pred_neg = self.model_run(
                None, {"input": (-spek).detach().cpu().numpy()}
            )[0]
            spec_pred_pos = self.model_run(
                None, {"input": spek.detach().cpu().numpy()}
            )[0]
            spec_pred = (spec_pred_neg * -0.5) + (spec_pred_pos * 0.5)
        else:
            spec_pred = self.model_run(None, {"input": spek.detach().cpu().numpy()})[0]

        result = (
            self.stft.inverse(torch.from_numpy(spec_pred).to(self.device)).cpu().numpy()
        )
        return result

    def _demix(self, mix: np.ndarray, is_match_mix: bool = False) -> np.ndarray:
        self._initialize_model_settings()

        if is_match_mix:
            chunk_size = self.hop_length * (self.segment_size - 1)
            overlap = 0.02
        else:
            chunk_size = self.chunk_size
            overlap = self.overlap

        gen_size = chunk_size - 2 * self.trim
        pad = gen_size + self.trim - (mix.shape[-1] % gen_size)
        mixture = np.concatenate(
            (
                np.zeros((2, self.trim), dtype=np.float32),
                mix,
                np.zeros((2, pad), dtype=np.float32),
            ),
            axis=1,
        )

        step = int((1 - overlap) * chunk_size)
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)

        total_chunks = (mixture.shape[-1] + step - 1) // step
        progress_interval = max(1, total_chunks // 100)

        for chunk_idx, i in enumerate(range(0, mixture.shape[-1], step)):
            if chunk_idx % progress_interval == 0:
                progress = chunk_idx / total_chunks
                send_message({"type": "progress", "progress": progress / 2 + 0.5})

            start = i
            end = min(i + chunk_size, mixture.shape[-1])
            chunk_size_actual = end - start
            window = None
            if overlap != 0:
                window = np.hanning(chunk_size_actual)
                window = np.tile(window[None, None, :], (1, 2, 1))

            mix_part = mixture[:, start:end]
            if end != i + chunk_size:
                pad_size = (i + chunk_size) - end
                mix_part = np.concatenate(
                    (mix_part, np.zeros((2, pad_size), dtype=np.float32)), axis=-1
                )

            mix_part = torch.from_numpy(mix_part).unsqueeze(0).to(self.device)
            for mix_wave in mix_part.split(self.batch_size):
                tar_waves = self._run_model(mix_wave, is_match_mix=is_match_mix)
                if window is not None:
                    tar_waves[..., :chunk_size_actual] *= window
                    divider[..., start:end] += window
                else:
                    divider[..., start:end] += 1
                result[..., start:end] += tar_waves[..., : end - start]

        divider = np.maximum(divider, 1e-10)
        tar_waves = result / divider
        tar_waves = tar_waves[:, :, self.trim : -self.trim]
        tar_waves = tar_waves[:, :, : mix.shape[-1]]
        send_message({"type": "progress", "progress": 1.0})
        return tar_waves[0]

    def separate_audio(
        self, source_path: str, vocal_path: str, instrumental_path: str
    ) -> None:
        self._load_model()

        mixture = read_audio_file(source_path, self.sample_rate, 2)
        peak = np.abs(mixture).max()
        if peak == 0:
            raise RuntimeError("Input audio appears to be silent.")
        mixture = mixture / peak

        primary = utils.match_array_shapes(self._demix(mixture), mixture)
        secondary = utils.match_array_shapes(
            mixture - (primary * self.compensate), mixture
        )
        primary_source = primary * peak
        secondary_source = secondary * peak

        primary_time = utils.normalize(primary_source, max_peak=0.9, min_peak=0.0).T
        secondary_time = utils.normalize(secondary_source, max_peak=0.9, min_peak=0.0).T

        if self.primary_stem.lower() == "instrumental":
            instrumental_audio = primary_time
            vocal_audio = secondary_time
        else:
            vocal_audio = primary_time
            instrumental_audio = secondary_time

        if vocal_path:
            write_audio_file(
                vocal_path,
                vocal_audio.astype(np.float32),
                self.sample_rate,
            )
        if instrumental_path:
            write_audio_file(
                instrumental_path,
                instrumental_audio.astype(np.float32),
                self.sample_rate,
            )
