import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from musetric_toolkit.distill.ffn import narrow_ffn
from musetric_toolkit.distill.heads import prune_heads
from musetric_toolkit.distill.mask_head import low_rank_head, narrow_head, pooled_head
from musetric_toolkit.distill.pooled import local_attention, merge_bands, pooled_teacher
from musetric_toolkit.distill.sublayers import install
from musetric_toolkit.distill.teacher import prepare

FFN = re.compile(r"f(\d+)([tb]?)([mr]?)(?:@(\d+))?")
NARROW_HEAD = re.compile(r"n\d+t?")
LOW_RANK_HEAD = re.compile(r"h[rs]\d+t?")
POOLED_HEAD = re.compile(r"hp\d")
PRUNED = re.compile(r"a\d[tb]?@\d+")
FITTED_SPECS = (":lin", ":mix")


@dataclass
class Fits:
    ffn_keep: Path | None = None
    ffn_means: Path | None = None
    ffn_refit: Path | None = None
    head_prune: Path | None = None
    head_refit: Path | None = None
    head_lowrank: Path | None = None
    detail: Path | None = None
    sublayer: Path | None = None

    @classmethod
    def read(cls, path: Path | None) -> "Fits":
        if path is None:
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{key: Path(value) for key, value in data.items()})


def load_fit(path: Path | None):
    if path is None:
        message = "the student needs a fit that --fits does not name"
        raise ValueError(message)
    return torch.load(path, weights_only=False)


def ffn_fits(fits: Fits, kind: str):
    chosen = None
    if fits.ffn_keep:
        chosen = json.loads(fits.ffn_keep.read_text(encoding="utf-8"))
    means = None
    if kind == "m":
        means = json.loads(load_path(fits.ffn_means).read_text(encoding="utf-8"))
    stored = load_fit(fits.ffn_refit) if kind == "r" else {}
    return chosen, means, stored


def load_path(path: Path | None) -> Path:
    if path is None:
        message = "the student needs a fit that --fits does not name"
        raise ValueError(message)
    return path


def narrow(model, mod: str, fits: Fits) -> None:
    match = FFN.fullmatch(mod)
    pairs = match[4] and [int(c) for c in match[4]]
    which = match[2] or "tb"
    narrow_ffn(model, int(match[1]) / 100, pairs, which, ffn_fits(fits, match[3]))


def replace_sublayers(model, mod: str, fits: Fits) -> None:
    specs = mod[1:].split("/")
    fitted = {}
    if any(spec.endswith(FITTED_SPECS) for spec in specs):
        fitted = load_fit(fits.sublayer)["fits"]
    for spec in specs:
        install(model, spec, fitted.get(spec))


def add_detail_paths(model, fits: Fits) -> None:
    model.detail_paths = torch.nn.ParameterDict()
    for i, (w, b, kinds) in load_fit(fits.detail)["fits"].items():
        model.detail_paths[f"w{i}"] = torch.nn.Parameter(w.cuda())
        model.detail_paths[f"b{i}"] = torch.nn.Parameter(b.cuda())
        model.detail[i] = (
            model.detail_paths[f"w{i}"],
            model.detail_paths[f"b{i}"],
            kinds,
        )


def apply_modifier(model, mod: str, fits: Fits) -> None:
    if FFN.fullmatch(mod):
        narrow(model, mod, fits)
    elif mod.startswith("w"):
        local_attention(model, mod[1:])
    elif mod.startswith("m"):
        merge_bands(model, mod[1:])
    elif NARROW_HEAD.fullmatch(mod):
        narrow_head(model, load_fit(fits.head_refit)["fits"], mod.endswith("t"))
    elif LOW_RANK_HEAD.fullmatch(mod):
        bands = load_fit(fits.head_lowrank)["bands"]
        low_rank_head(model, bands, mod[1] == "r", mod.endswith("t"))
    elif mod.startswith("x"):
        replace_sublayers(model, mod, fits)
    elif POOLED_HEAD.fullmatch(mod):
        pooled_head(model, int(mod[2:]))
    elif PRUNED.fullmatch(mod):
        prune_heads(model, load_fit(fits.head_prune)["fits"])
    elif mod == "d":
        add_detail_paths(model, fits)
    else:
        message = f"unknown modifier {mod}"
        raise ValueError(message)


def build_student(name: str, teacher: Path, fits: Fits):
    base, *mods = name.split("+")
    model = prepare(pooled_teacher(teacher, base))
    for mod in mods:
        apply_modifier(model, mod, fits)
    return model


def load_student(checkpoint: Path, teacher: Path, fits: Path | None, device="cpu"):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_student(state["student"], teacher, Fits.read(fits))
    model.load_state_dict(state["model"], strict=not state.get("partial", False))
    return model, state
