import re
from pathlib import Path

import torch
from einops import pack, rearrange, unpack
from rotary_embedding_torch import apply_rotary_emb

from musetric_toolkit.distill.teacher import band_features, frozen_teacher, masks_from

SCHEDULE = re.compile(r"pool-s(?P<ks>[1248]{6})")
MERGE = re.compile(r"(\d+)@(\d+)")
WINDOW = re.compile(r"(\d+)(?:c(\d+))?(?:@(\d+))?")
BANDS = 60
DIM = 384


class Stretched(torch.nn.Module):
    def __init__(self, base, scale=1):
        super().__init__()
        self.base, self.scale = base, scale

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, scale=None):
        if self.scale == 1:
            return self.base.rotate_queries_or_keys(t)
        with torch.autocast("cuda", enabled=False):
            pos = torch.arange(t.shape[-2], device=t.device, dtype=torch.float32)
            freqs = torch.einsum("n,f->nf", pos * self.scale, self.base.freqs.float())
            freqs = freqs.repeat_interleave(2, dim=-1)
        return apply_rotary_emb(freqs, t, seq_dim=-2)


class Placed(torch.nn.Module):
    def __init__(self, base, positions):
        super().__init__()
        self.base = base
        positions = torch.tensor(positions, dtype=torch.float32)
        self.register_buffer("positions", positions, persistent=False)

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, scale=None):
        with torch.autocast("cuda", enabled=False):
            pos = self.positions.to(t.device)
            freqs = torch.einsum("n,f->nf", pos, self.base.freqs.float())
            freqs = freqs.repeat_interleave(2, dim=-1)
        return apply_rotary_emb(freqs, t, seq_dim=-2)


def unwrap(rotary):
    return getattr(rotary, "base", rotary)


def install_rotary_schedule(model, ks) -> None:
    time_base = unwrap(model.layers[0][0].layers[0][0].rotary_embed)
    freq_base = unwrap(model.layers[0][1].layers[0][0].rotary_embed)
    for (time_t, freq_t), k in zip(model.layers, ks, strict=True):
        for attn, _ in time_t.layers:
            attn.rotary_embed = Stretched(time_base, k) if k > 1 else time_base
        for attn, _ in freq_t.layers:
            attn.rotary_embed = freq_base


def run_pairs(x, layers):
    for time_t, freq_t in layers:
        x = rearrange(x, "b t f d -> b f t d")
        x, shape = pack([x], "* t d")
        x = time_t(x)
        (x,) = unpack(x, shape, "* t d")
        x = rearrange(x, "b f t d -> b t f d")
        x, shape = pack([x], "* f d")
        x = freq_t(x)
        (x,) = unpack(x, shape, "* f d")
    return x


def down(x, p, axis):
    if p == 1:
        return x
    n = x.shape[axis]
    pad = (-n) % p
    if pad:
        tail = x.narrow(axis, n - 1, 1)
        x = torch.cat([x] + [tail] * pad, axis)
    shape = list(x.shape)
    shape[axis : axis + 1] = [shape[axis] // p, p]
    return x.reshape(shape).mean(axis + 1)


def up(y, p, axis, n):
    return y if p == 1 else y.repeat_interleave(p, dim=axis).narrow(axis, 0, n)


def merge_high(x, keep):
    hi = x[:, :, keep:]
    hi = hi.reshape(*hi.shape[:2], hi.shape[2] // 2, 2, hi.shape[3]).mean(3)
    return torch.cat([x[:, :, :keep], hi], 2)


def unmerge_high(y, keep):
    return torch.cat([y[:, :, :keep], y[:, :, keep:].repeat_interleave(2, dim=2)], 2)


def detail_term(fit, x, coarse=None):
    w, b, kinds = fit
    parts = {"x": x, "c": coarse, "d": None if coarse is None else x - coarse}
    if coarse is not None:
        detail = parts["d"]
        parts["p"] = torch.cat([detail[:, :1] * 0, detail[:, :-1]], 1)
        parts["n"] = torch.cat([detail[:, 1:], detail[:, :1] * 0], 1)
    features = torch.cat([parts[kind] for kind in kinds], -1)
    if w.dim() == 3:  # noqa: PLR2004
        if w.shape[0] != features.shape[2]:
            groups = features.shape[2] // w.shape[0]
            w, b = (t.repeat_interleave(groups, 0) for t in (w, b))
        mapped = torch.einsum("btfi,fio->btfo", features, w.to(features.dtype))
        return mapped + b.to(features.dtype)
    return features @ w.to(features.dtype) + b.to(features.dtype)


def pooled_pair(x, pair, k, keep, fit):
    frames = x.shape[1]
    lo = down(x, k, 1)
    if keep:
        lo = merge_high(lo, keep)
    change = run_pairs(lo, [pair]) - lo
    coarse = None
    if fit:
        coarse = up(unmerge_high(lo, keep) if keep else lo, k, 1, frames)
    if keep:
        change = unmerge_high(change, keep)
    y = x + up(change, k, 1, frames)
    return y + detail_term(fit, x, coarse) if fit else y


def schedule_stack(model, x, ks, outs=None):
    detail = model.detail
    pairs = zip(model.layers, ks, model.band_merges, strict=True)
    for i, (pair, k, keep) in enumerate(pairs):
        fit = detail.get(i)
        if k == 1 and not keep:
            y = run_pairs(x, [pair])
            x = y + detail_term(fit, x) if fit else y
        else:
            x = pooled_pair(x, pair, k, keep, fit)
        if outs is not None:
            outs.append(x)
    return x


def merge_bands(model, spec: str) -> None:
    match = MERGE.fullmatch(spec)
    keep = int(match[1])
    if (BANDS - keep) % 2 or not hasattr(model, "band_merges"):
        message = f"+m{spec}: needs an even number of merged bands and a pool-s student"
        raise ValueError(message)
    base = unwrap(model.layers[0][1].layers[0][0].rotary_embed)
    merged = [keep + 2 * j + 0.5 for j in range((BANDS - keep) // 2)]
    positions = list(range(keep)) + merged
    for i in [int(c) for c in match[2]]:
        model.band_merges[i] = keep
        for attn, _ in model.layers[i][1].layers:
            attn.rotary_embed = Placed(base, positions)


class LocalAttend(torch.nn.Module):
    def __init__(self, window, far=0):
        super().__init__()
        self.window, self.far = window, far

    def forward(self, q, k, v):
        n = q.shape[-2]
        i = torch.arange(n, device=q.device)
        if not self.far:
            band = (i[:, None] - i[None, :]).abs() <= self.window
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=band
            )
        axis = k.dim() - 2
        first = torch.arange(0, n, self.far, device=q.device)
        last = (first + self.far - 1).clamp(max=n - 1)
        outside = (last[None, :] < i[:, None] - self.window) | (
            first[None, :] > i[:, None] + self.window
        )
        near = ~outside[:, i // self.far]
        size = torch.log((last - first + 1).float())
        mask = torch.cat(
            [
                torch.where(near, 0.0, -torch.inf),
                torch.where(outside, size, -torch.inf),
            ],
            1,
        )
        keys = torch.cat([k, down(k, self.far, axis)], axis)
        values = torch.cat([v, down(v, self.far, axis)], axis)
        return torch.nn.functional.scaled_dot_product_attention(
            q, keys, values, attn_mask=mask.to(q.dtype)
        )


def local_attention(model, spec: str) -> None:
    match = WINDOW.fullmatch(spec)
    window, far, pairs = int(match[1]), int(match[2] or 0), match[3]
    rates = getattr(model, "attention_rates", [1] * len(model.layers))
    chosen = [int(c) for c in pairs] if pairs else range(len(model.layers))
    for i in chosen:
        for attn, _ in model.layers[i][0].layers:
            attn.attend = LocalAttend(
                max(1, window // rates[i]), far and max(1, far // rates[i])
            )


def pooled_teacher(teacher: Path, name: str):
    match = SCHEDULE.fullmatch(name)
    if not match:
        message = f"not a pool-s student name: {name}"
        raise ValueError(message)
    ks = [int(c) for c in match["ks"]]
    model = frozen_teacher(teacher)
    model.attention_rates = ks
    install_rotary_schedule(model, ks)
    model.band_merges = [0] * len(ks)
    model.detail = {}

    def run_stack(x):
        outs = [] if getattr(model, "pair_targets", False) else None
        out = schedule_stack(model, x, ks, outs)
        return out, outs or [out]

    model.run_stack = run_stack
    model.net_forward = lambda stft: masks_from(
        model, run_stack(band_features(model, stft))[0]
    )
    return model
