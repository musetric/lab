from pathlib import Path

import torch

LAYERS = "layers."
PATHS = "detail_paths."
ROTARY = "rotary_embed"


def without_rotary(state: dict) -> dict:
    return {k: v for k, v in state.items() if ROTARY not in k}


def load_layers(model, path: Path) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    layers = {
        k[len(LAYERS) :]: v
        for k, v in state.items()
        if k.startswith(LAYERS) and ROTARY not in k
    }
    own = model.layers.state_dict()
    for k, v in layers.items():
        if k in own and own[k].shape != v.shape:
            kept = model.layers.get_submodule(k.rsplit(".", 1)[0]).kept.cpu()
            layers[k] = v[kept] if v.shape[0] != own[k].shape[0] else v[:, kept]
    model.layers.load_state_dict(layers, strict=False)
    paths = {k[len(PATHS) :]: v for k, v in state.items() if k.startswith(PATHS)}
    if paths and hasattr(model, "detail_paths"):
        with torch.no_grad():
            for k, v in paths.items():
                model.detail_paths[k].copy_(v)
    return len(layers), len(paths)


def save_partial(model, path: Path, meta: dict, half: bool) -> None:
    weights = {
        n: (p.detach().half() if half else p.detach()).cpu()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    payload = dict(model=weights, partial=True, **meta)
    torch.save(payload, path)
