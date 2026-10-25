from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from musetric_toolkit.pitch_zoo.estimate import ModelContext, PitchModel


@dataclass(frozen=True)
class ZooModel:
    name: str
    package: str
    license: str
    source: str
    create: Callable[[ModelContext], PitchModel]


def _create_rmvpe(context: ModelContext) -> PitchModel:
    from musetric_toolkit.pitch_zoo.rmvpe_model import RmvpeModel  # noqa: PLC0415

    return RmvpeModel(context)


def _create_crepe(context: ModelContext) -> PitchModel:
    from musetric_toolkit.pitch_zoo.crepe_model import CrepeModel  # noqa: PLC0415

    return CrepeModel(context)


def _create_swiftf0(context: ModelContext) -> PitchModel:
    from musetric_toolkit.pitch_zoo.swiftf0_model import (  # noqa: PLC0415
        SwiftF0Model,
    )

    return SwiftF0Model(context)


def _create_fcpe(context: ModelContext) -> PitchModel:
    from musetric_toolkit.pitch_zoo.fcpe_model import FcpeModel  # noqa: PLC0415

    return FcpeModel(context)


def _create_penn(context: ModelContext) -> PitchModel:
    from musetric_toolkit.pitch_zoo.penn_model import PennModel  # noqa: PLC0415

    return PennModel(context)


def _create_pesto(context: ModelContext) -> PitchModel:
    from musetric_toolkit.pitch_zoo.pesto_model import PestoModel  # noqa: PLC0415

    return PestoModel(context)


def _create_pyin(context: ModelContext) -> PitchModel:
    from musetric_toolkit.pitch_zoo.pyin_model import PyinModel  # noqa: PLC0415

    return PyinModel(context)


ZOO_MODELS = {
    model.name: model
    for model in (
        ZooModel(
            "rmvpe",
            "lab",
            "Apache-2.0 (RMVPE), MIT (RVC code and checkpoint)",
            "https://github.com/Dream-High/RMVPE",
            _create_rmvpe,
        ),
        ZooModel(
            "crepe",
            "torchcrepe",
            "MIT",
            "https://github.com/maxrmorrison/torchcrepe",
            _create_crepe,
        ),
        ZooModel(
            "swiftf0",
            "swift-f0",
            "MIT",
            "https://github.com/lars76/swift-f0",
            _create_swiftf0,
        ),
        ZooModel(
            "fcpe",
            "torchfcpe",
            "MIT",
            "https://github.com/CNChTu/FCPE",
            _create_fcpe,
        ),
        ZooModel(
            "penn",
            "penn",
            "MIT",
            "https://github.com/interactiveaudiolab/penn",
            _create_penn,
        ),
        ZooModel(
            "pesto",
            "pesto-pitch",
            "LGPL-3.0",
            "https://github.com/SonyCSLParis/pesto",
            _create_pesto,
        ),
        ZooModel(
            "pyin",
            "librosa",
            "ISC",
            "https://github.com/librosa/librosa",
            _create_pyin,
        ),
    )
}
