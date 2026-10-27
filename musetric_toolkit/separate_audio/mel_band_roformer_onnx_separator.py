import gc
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.separate_audio import utils
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.ffmpeg.write import write_audio_file
from musetric_toolkit.separate_audio.roformer.mel_band_roformer import MelBandRoformer
from musetric_toolkit.separate_audio.roformer_utils import dict_to_namespace


class MelBandRoformerOnnxSeparator:

    def __init__(
        self,
        model_onnx_path: Path,
        model_config_path: Path,
        sample_rate: int,
    ):
        self.model_onnx_path = Path(model_onnx_path)
        self.model_config_path = Path(model_config_path)
        self.sample_rate = sample_rate
        self.device = self._get_device()
        self.config = None
        self.host = None
        self.session = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_config(self):
        with open(self.model_config_path) as f:
            return dict_to_namespace(yaml.load(f, Loader=yaml.FullLoader))  # noqa: S506

    def _build_host(self):
        host = MelBandRoformer(**vars(self.config.model))
        for name in ("layers", "band_split", "mask_estimators"):
            if hasattr(host, name):
                delattr(host, name)
        gc.collect()
        return host.to(self.device).eval()

    def _make_session(self):
        try:
            import onnxruntime as ort  # noqa: PLC0415  (lazy import)
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for MelBandRoformerOnnxSeparator. "
                "It ships with the default install (`uv sync`) as onnxruntime-gpu."
            ) from exc

        available = ort.get_available_providers()
        gpu = [
            p
            for p in (
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CoreMLExecutionProvider",
            )
            if p in available
        ]
        providers = [*gpu, "CPUExecutionProvider"]

        def _options() -> "ort.SessionOptions":
            options = ort.SessionOptions()
            if logging.getLogger().level > logging.DEBUG:
                options.log_severity_level = 3
            return options

        try:
            session = ort.InferenceSession(
                str(self.model_onnx_path),
                sess_options=_options(),
                providers=providers,
            )
        except Exception:
            options = _options()
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            session = ort.InferenceSession(
                str(self.model_onnx_path), sess_options=options, providers=providers
            )
        logging.info(
            "MelBand ONNX core EP: %s (host=%s)",
            session.get_providers(),
            self.device,
        )
        return session

    def _sync_config_to_onnx_shape(self) -> None:
        shape = self.session.get_inputs()[0].shape
        frames = shape[2]
        if not isinstance(frames, int):
            raise ValueError(f"Expected static ONNX time dimension, got {shape}")
        config_frames = self.config.inference.dim_t
        if frames != config_frames:
            logging.info(
                "Using ONNX static T=%s instead of config inference.dim_t=%s",
                frames,
                config_frames,
            )
            self.config.inference.dim_t = frames

    def _load_model(self):
        if self.session is not None:
            return
        self.config = self._load_config()
        self.host = self._build_host()
        self.session = self._make_session()
        self._sync_config_to_onnx_shape()

    def _chunk_size(self) -> int:
        return self.config.audio.hop_length * (self.config.inference.dim_t - 1)

    def _step_size(self) -> int:
        return min(int(8 * self.config.audio.sample_rate), self._chunk_size())

    def _core(self, part: torch.Tensor) -> torch.Tensor:
        chunk_size = self._chunk_size()
        length = part.shape[-1]
        if length < chunk_size:
            part = torch.nn.functional.pad(part, (0, chunk_size - length))
        enc = self.host.encode_stft(part.unsqueeze(0))
        masks = self.session.run(
            ["masks"], {"stft_repr": enc.cpu().numpy().astype(np.float32)}
        )[0]
        recon = self.host.decode_istft(enc, torch.from_numpy(masks).to(self.device))[0]
        return recon[..., :length]

    def _demix(self, mix: np.ndarray) -> dict:
        mix_tensor = torch.from_numpy(mix).to(dtype=torch.float32, device=self.device)
        chunk_size = self._chunk_size()
        step_size = self._step_size()
        window = torch.hamming_window(
            chunk_size, dtype=torch.float32, device=self.device
        )
        total_len = mix_tensor.shape[1]
        result = torch.zeros_like(mix_tensor)
        counter = torch.zeros_like(mix_tensor)

        total_steps = (total_len + step_size - 1) // step_size
        progress_interval = max(1, total_steps // 100)

        with torch.no_grad():
            for step_idx, i in enumerate(range(0, total_len, step_size)):
                if step_idx % progress_interval == 0:
                    progress = step_idx / total_steps
                    send_message({"type": "progress", "progress": progress / 2})

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

                x = self._core(part)
                window_slice = window[:length]
                result[..., start_pos : start_pos + length] += (
                    x[..., :length] * window_slice
                )
                counter[..., start_pos : start_pos + length] += window_slice

        target_audio = (result / counter.clamp(min=1e-10)).cpu().numpy()

        instruments = list(self.config.training.instruments)
        primary_stem = self.config.training.target_instrument
        secondary_stem = next(
            (name for name in instruments if name != primary_stem), None
        )

        target_audio = utils.match_array_shapes(target_audio, mix)
        sources = {primary_stem: target_audio}
        if secondary_stem is not None:
            sources[secondary_stem] = mix - target_audio
        return sources

    def separate_audio(
        self, source_path: str, target_path: str, residual_path: str
    ) -> None:
        self._load_model()

        mixture = utils.normalize(
            read_audio_file(source_path, self.sample_rate, 2),
            max_peak=0.9,
            min_peak=0.0,
        )

        separated_sources = self._demix(mixture)

        target_stem = self.config.training.target_instrument

        for stem_name, source_audio in separated_sources.items():
            normalized_source = utils.normalize(
                source_audio, max_peak=0.9, min_peak=0.0
            ).T
            output_path = target_path if stem_name == target_stem else residual_path
            if output_path:
                write_audio_file(
                    output_path,
                    normalized_source.astype(np.float32),
                    self.sample_rate,
                )
