import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from musetric_toolkit.separate_audio import utils
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.ffmpeg.write import write_audio_file
from musetric_toolkit.separate_audio.roformer.mel_band_roformer import MelBandRoformer
from musetric_toolkit.separate_audio.roformer.pooled_mel_band_roformer import (
    PooledMelBandRoformer,
)
from musetric_toolkit.separate_audio.roformer_utils import (
    AudioProcessor,
    dict_to_namespace,
)


class MelBandRoformerSeparator:
    def __init__(
        self,
        model_checkpoint_path: Path,
        model_config_path: Path,
        sample_rate: int,
    ):
        self.model_checkpoint_path = model_checkpoint_path
        self.model_config_path = model_config_path
        self.sample_rate = sample_rate
        self.device = self._get_device()
        self.model = None
        self.config = None
        self.audio_processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_config(self):
        with open(self.model_config_path) as f:
            return dict_to_namespace(yaml.load(f, Loader=yaml.FullLoader))  # noqa: S506

    def _load_model(self):
        if self.model is not None:
            return

        self.config = self._load_config()
        model_class = (
            PooledMelBandRoformer
            if hasattr(self.config.model, "pool")
            else MelBandRoformer
        )
        model = model_class(**vars(self.config.model))
        if Path(self.model_checkpoint_path).suffix == ".safetensors":
            checkpoint = load_file(self.model_checkpoint_path)
        else:
            checkpoint = torch.load(
                self.model_checkpoint_path, map_location="cpu", weights_only=True
            )
        model.load_state_dict(checkpoint)
        self.model = model.to(self.device)
        self.model.eval()

        self.audio_processor = AudioProcessor(self.device, self.config)

    def _demix(self, mix: np.ndarray) -> dict:
        return self.audio_processor.demix(mix, self.model)

    def separate_audio(
        self, source_path: str, vocal_path: str, instrumental_path: str
    ) -> None:
        with tempfile.TemporaryDirectory():
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
                output_path = (
                    vocal_path if stem_name == target_stem else instrumental_path
                )
                if output_path:
                    write_audio_file(
                        output_path,
                        normalized_source.astype(np.float32),
                        self.sample_rate,
                    )
