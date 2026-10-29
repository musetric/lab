import torch
from einops import pack, rearrange, unpack
from rotary_embedding_torch import apply_rotary_emb
from torch import nn

from musetric_toolkit.separate_audio.roformer.mel_band_roformer import (
    Attention,
    MelBandRoformer,
)


def down(x: torch.Tensor, p: int, axis: int) -> torch.Tensor:
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


def up(y: torch.Tensor, p: int, axis: int, n: int) -> torch.Tensor:
    return y if p == 1 else y.repeat_interleave(p, dim=axis).narrow(axis, 0, n)


def merge_bands(x: torch.Tensor, first: int) -> torch.Tensor:
    high = x[:, :, first:]
    high = high.reshape(*high.shape[:2], high.shape[2] // 2, 2, high.shape[3]).mean(3)
    return torch.cat([x[:, :, :first], high], 2)


def unmerge_bands(y: torch.Tensor, first: int) -> torch.Tensor:
    return torch.cat([y[:, :, :first], y[:, :, first:].repeat_interleave(2, dim=2)], 2)


class Rotary(nn.Module):

    def __init__(
        self,
        dim_head: int,
        scale: int = 1,
        positions: list[float] | None = None,
        theta: float = 10000,
    ):
        super().__init__()
        self.scale = scale
        freqs = 1.0 / (
            theta ** (torch.arange(0, dim_head, 2)[: dim_head // 2].float() / dim_head)
        )
        self.register_buffer("freqs", freqs)
        placed = (
            None if positions is None else torch.tensor(positions, dtype=torch.float32)
        )
        self.register_buffer("positions", placed, persistent=False)

    def rotate_queries_or_keys(self, t: torch.Tensor) -> torch.Tensor:
        with torch.autocast(t.device.type, enabled=False):
            if self.positions is None:
                pos = (
                    torch.arange(t.shape[-2], device=t.device, dtype=torch.float32)
                    * self.scale
                )
            else:
                pos = self.positions.to(t.device)
            freqs = torch.einsum(
                "n,f->nf", pos, self.freqs.to(t.device)
            ).repeat_interleave(2, dim=-1)
        return apply_rotary_emb(freqs, t, seq_dim=-2)


class LocalAttend(nn.Module):

    def __init__(self, window: int):
        super().__init__()
        self.window = window

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        i = torch.arange(q.shape[-2], device=q.device)
        band = (i[:, None] - i[None, :]).abs() <= self.window
        return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=band)


class BandMap(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(x)


class PooledMelBandRoformer(MelBandRoformer):

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        *,
        pool: list[int],
        merged_bands_from: list[int | None],
        time_window: list[int | None],
        time_heads: list[int],
        ff_mult: list[float],
        band_map: list[bool],
        detail: list[list[str] | None],
        mask_rank: int,
        mask_pool: int,
        **kwargs,
    ):
        kwargs.pop("mask_estimator_depth", None)
        super().__init__(dim, mask_estimator_depth=1, **kwargs)
        self.pool, self.merged_bands_from = pool, merged_bands_from
        self.mask_pool, self.detail_kinds = mask_pool, detail
        heads, dim_head = kwargs.get("heads", 8), kwargs.get("dim_head", 64)
        flash = kwargs.get("flash_attn", True)

        for i, (time_t, band_t) in enumerate(self.layers):
            for layer in time_t.layers:
                if time_heads[i] != heads:
                    layer[0] = Attention(
                        dim, heads=time_heads[i], dim_head=dim_head, flash=flash
                    )
                    layer[0].to_out[0] = nn.Linear(time_heads[i] * dim_head, dim)
                layer[0].rotary_embed = Rotary(dim_head, scale=pool[i])
                if time_window[i]:
                    layer[0].attend = LocalAttend(max(1, time_window[i] // pool[i]))
                self._narrow(layer[1], dim, ff_mult[i])
            if band_map[i]:
                self.layers[i][1] = BandMap(dim)
            else:
                positions = self._band_positions(
                    merged_bands_from[i], kwargs.get("num_bands", 60)
                )
                for layer in band_t.layers:
                    layer[0].rotary_embed = Rotary(dim_head, positions=positions)
                    self._narrow(layer[1], dim, ff_mult[i])

        self.detail_paths = nn.ParameterDict()
        for i, kinds in enumerate(detail):
            if kinds:
                width = len(kinds) * dim
                self.detail_paths[f"w{i}"] = nn.Parameter(torch.zeros(width, dim))
                self.detail_paths[f"b{i}"] = nn.Parameter(torch.zeros(dim))

        self._low_rank_middle(dim * 4, mask_rank)

    def _low_rank_middle(self, hidden: int, rank: int) -> None:
        for estimator in self.mask_estimators:
            for band in estimator.to_freqs:
                first_layer, tanh, last_layer = band[0]
                middle = nn.Sequential(
                    nn.Linear(hidden, rank), nn.Linear(rank, hidden, bias=False)
                )
                band[0] = nn.Sequential(
                    first_layer, tanh, middle, nn.Tanh(), last_layer
                )

    @staticmethod
    def _band_positions(first: int | None, num_bands: int) -> list[float] | None:
        if first is None:
            return None
        merged = [first + 2 * j + 0.5 for j in range((num_bands - first) // 2)]
        return list(range(first)) + merged

    @staticmethod
    def _narrow(ff: nn.Module, dim: int, mult: float) -> None:
        inner = int(dim * mult)
        if inner != ff.net[1].out_features:
            ff.net[1], ff.net[4] = nn.Linear(dim, inner), nn.Linear(inner, dim)

    def _pair(self, i: int, x: torch.Tensor) -> torch.Tensor:
        time_t, band_t = self.layers[i]
        x = rearrange(x, "b t f d -> b f t d")
        x, ps = pack([x], "* t d")
        x = time_t(x)
        (x,) = unpack(x, ps, "* t d")
        x = rearrange(x, "b f t d -> b t f d")
        x, ps = pack([x], "* f d")
        x = band_t(x)
        (x,) = unpack(x, ps, "* f d")
        return x

    def _detail(
        self, i: int, x: torch.Tensor, coarse: torch.Tensor | None
    ) -> torch.Tensor:
        parts = {"x": x, "c": coarse, "d": None if coarse is None else x - coarse}
        features = torch.cat([parts[kind] for kind in self.detail_kinds[i]], -1)
        w, b = self.detail_paths[f"w{i}"], self.detail_paths[f"b{i}"]
        return features @ w.to(features.dtype) + b.to(features.dtype)

    def net_forward(self, stft_repr: torch.Tensor) -> torch.Tensor:
        x = self.trunk(stft_repr)
        frames = x.shape[1]
        pooled = down(x, self.mask_pool, 1)
        masks = torch.stack(
            [up(fn(pooled), self.mask_pool, 1, frames) for fn in self.mask_estimators],
            dim=1,
        )
        return rearrange(masks, "b n t (f c) -> b n f t c", c=2)

    def trunk(self, stft_repr: torch.Tensor) -> torch.Tensor:
        batch = stft_repr.shape[0]
        batch_arange = torch.arange(batch, device=stft_repr.device)[..., None]
        x = stft_repr[batch_arange, self.freq_indices.to(stft_repr.device)]
        x = self.band_split(rearrange(x, "b f t c -> b t (f c)"))

        for i, (k, first) in enumerate(
            zip(self.pool, self.merged_bands_from, strict=True)
        ):
            with_detail = bool(self.detail_kinds[i])
            if k == 1 and first is None:
                y = self._pair(i, x)
                x = y + self._detail(i, x, None) if with_detail else y
                continue
            frames = x.shape[1]
            lo = down(x, k, 1)
            if first is not None:
                lo = merge_bands(lo, first)
            change = self._pair(i, lo) - lo
            coarse = None
            if with_detail:
                coarse = up(
                    unmerge_bands(lo, first) if first is not None else lo, k, 1, frames
                )
            if first is not None:
                change = unmerge_bands(change, first)
            y = x + up(change, k, 1, frames)
            x = y + self._detail(i, x, coarse) if with_detail else y
        return x
