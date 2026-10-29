import sys

import build_full_onnx as bfo  # type: ignore[import-not-found]
import build_student_onnx as bso  # type: ignore[import-not-found]
import torch
import yaml
from safetensors.torch import load_file

from musetric_toolkit.distill.voices import load_split
from musetric_toolkit.separate_audio.roformer import lead_backing_roformer
from musetric_toolkit.separate_audio.roformer.lead_backing_roformer import (
    LeadBackingRoformer,
)

RANDOM_STD = 0.02
SPLIT = {}


def option(name: str, default: str | None = None) -> str | None:
    if name not in sys.argv:
        return default
    at = sys.argv.index(name)
    value = sys.argv[at + 1]
    del sys.argv[at : at + 2]
    return value


def load_model(checkpoint, config) -> LeadBackingRoformer:
    with open(config) as f:
        cfg = yaml.safe_load(f)
    state = None
    if SPLIT["state"]:
        state = torch.load(SPLIT["state"], map_location="cpu", weights_only=False)
    released = cfg["model"].pop("split_hidden", None)
    hidden = state["split_hidden"] if state else released or SPLIT["hidden"]
    model = LeadBackingRoformer(**cfg["model"], split_hidden=hidden).eval()
    missing, unexpected = model.load_state_dict(load_file(checkpoint), strict=False)
    if unexpected or any(not k.startswith("split_estimator.") for k in missing):
        raise ValueError(f"unexpected weights: {unexpected}, missing: {missing}")
    if state:
        load_split(model, state)
    elif missing:
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for param in model.split_estimator.parameters():
                param.copy_(torch.randn(param.shape, generator=generator) * RANDOM_STD)
    return bso.exportable(model)


def lead_backing_net_forward(
    model: LeadBackingRoformer, stft_repr: torch.Tensor, slabs: int
) -> torch.Tensor:
    return model.heads(bso.hourglass_trunk(model, stft_repr, slabs))


class TwoMaskCore(bfo.Core):

    def forward(self, stft_repr: torch.Tensor) -> torch.Tensor:
        masks = self.model.net_forward(stft_repr)
        masks = masks.reshape(masks.shape[1], -1, masks.shape[-2], 2)
        g0 = masks.index_select(1, self.pair0)
        g1 = masks.index_select(1, self.pair1)
        summed = g0 * self.valid0 + g1 * self.valid1
        return (summed / self.denom).float()


def main() -> None:
    SPLIT["state"] = option("--split")
    SPLIT["hidden"] = int(option("--split-hidden", "768"))
    config = sys.argv[sys.argv.index("--config") + 1]
    with open(config) as f:
        bso.SHAPE.update(yaml.safe_load(f)["model"])
    lead_backing_roformer.down, lead_backing_roformer.up = bso.down, bso.up
    bfo.load_model = load_model
    bfo._stream_slab_net_forward = lead_backing_net_forward
    bfo.keep_time_attention_fp32 = bso.keep_time_attention_fp32
    bfo.Core = TwoMaskCore
    bfo.main()


if __name__ == "__main__":
    main()
