from pathlib import Path

import torch
import yaml
from einops import pack, rearrange, unpack
from torch.utils.checkpoint import checkpoint

from musetric_toolkit.distill.precision import PRECISION
from musetric_toolkit.separate_audio.roformer.mel_band_roformer import MelBandRoformer
from musetric_toolkit.separate_audio.roformer_utils import dict_to_namespace

SAMPLE_RATE = 44100
HOP = 441
WINDOW_FRAMES = 1100
CHUNK = HOP * WINDOW_FRAMES
STEP = 8 * SAMPLE_RATE


def read_config(teacher: Path):
    with open(teacher / "config.yaml") as f:
        return dict_to_namespace(yaml.load(f, Loader=yaml.FullLoader))  # noqa: S506


def load_teacher(teacher: Path) -> MelBandRoformer:
    model = MelBandRoformer(**vars(read_config(teacher).model))
    state = torch.load(teacher / "model.ckpt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def frozen_teacher(teacher: Path) -> MelBandRoformer:
    model = load_teacher(teacher)
    for module in (model.band_split, model.mask_estimators):
        for param in module.parameters():
            param.requires_grad_(False)
    return model


def teacher_layers(teacher: Path) -> torch.nn.ModuleList:
    model = load_teacher(teacher).cuda().eval()
    warm_rotary(model)
    layers = model.layers
    for param in layers.parameters():
        param.requires_grad_(False)
    return layers


def warm_rotary(model: MelBandRoformer) -> None:
    with torch.no_grad():
        model.net_forward(model.encode_stft(torch.zeros(1, 2, CHUNK, device="cuda")))


def prepare(model: MelBandRoformer) -> MelBandRoformer:
    model = model.cuda().train()
    for pair in model.layers:
        for transformer in pair:
            inner = transformer.forward

            def forward(x, inner=inner):
                if torch.is_grad_enabled() and PRECISION.recompute:
                    return checkpoint(inner, x, use_reentrant=False)
                return inner(x)

            transformer.forward = forward
    warm_rotary(model)
    return model


def band_features(model: MelBandRoformer, stft: torch.Tensor) -> torch.Tensor:
    batch = stft.shape[0]
    x = stft[torch.arange(batch, device=stft.device)[..., None], model.freq_indices]
    return model.band_split(rearrange(x, "b f t c -> b t (f c)"))


def run_layers(x: torch.Tensor, layers) -> tuple[torch.Tensor, list[torch.Tensor]]:
    outs = []
    for time_transformer, freq_transformer in layers:
        x = rearrange(x, "b t f d -> b f t d")
        x, shape = pack([x], "* t d")
        x = time_transformer(x)
        (x,) = unpack(x, shape, "* t d")
        x = rearrange(x, "b f t d -> b t f d")
        x, shape = pack([x], "* f d")
        x = freq_transformer(x)
        (x,) = unpack(x, shape, "* f d")
        outs.append(x)
    return x, outs


def masks_from(model: MelBandRoformer, x: torch.Tensor, heads=None) -> torch.Tensor:
    estimators = heads if heads is not None else model.mask_estimators
    masks = torch.stack([fn(x) for fn in estimators], dim=1)
    return rearrange(masks, "b n t (f c) -> b n f t c", c=2)
