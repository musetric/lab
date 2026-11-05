from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.distill.precision import PRECISION
from musetric_toolkit.distill.teacher import load_teacher, read_config
from musetric_toolkit.distill.voices import (
    kara_backing,
    kara_separator,
    load_lead_backing,
)
from musetric_toolkit.distill.voices_scores import separate_stems
from musetric_toolkit.separate_audio.roformer_utils import AudioProcessor

RELEASE_FOLDERS = 1
CASCADE_FOLDERS = 2


class VocalsTeacher:
    def __init__(self, folder: Path) -> None:
        PRECISION.dtype = torch.float16
        self.model = load_teacher(folder).cuda().eval()
        self.processor = AudioProcessor(torch.device("cuda"), read_config(folder))

    def vocals(self, mix: np.ndarray) -> np.ndarray:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            stems = self.processor.demix(mix, self.model)
        return stems["vocals"][:, : mix.shape[1]].astype(np.float32)


class Release:
    def __init__(self, folder: Path) -> None:
        PRECISION.dtype = torch.float16
        self.model = load_lead_backing(folder).cuda().eval()

    def separate(self, mix: np.ndarray) -> dict[str, np.ndarray]:
        with torch.no_grad():
            stems = separate_stems(self.model, mix)
        return {name: stem.astype(np.float32) for name, stem in stems.items()}


class Cascade:
    def __init__(self, vocals_teacher: Path, kara: Path) -> None:
        self.teacher = VocalsTeacher(vocals_teacher)
        self.kara = kara_separator(kara)

    def separate(self, mix: np.ndarray) -> dict[str, np.ndarray]:
        vocals = self.teacher.vocals(mix)
        backing = kara_backing(self.kara, vocals) * self.kara.compensation()
        return dict(
            vocals=vocals,
            backing=backing,
            lead=vocals - backing,
            instrumental=mix - vocals,
        )


def load_system(name: str, folders: list[Path]) -> Release | Cascade:
    if len(folders) == RELEASE_FOLDERS:
        return Release(folders[0])
    if len(folders) == CASCADE_FOLDERS:
        return Cascade(folders[0], folders[1])
    message = (
        f"system {name}: name a release folder, or the folders of a vocals teacher "
        f"and of KARA 2, not {folders}"
    )
    raise ValueError(message)
