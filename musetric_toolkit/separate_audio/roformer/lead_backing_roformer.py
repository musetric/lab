import torch
from einops import rearrange
from torch import nn

from musetric_toolkit.separate_audio.roformer.pooled_mel_band_roformer import (
    PooledMelBandRoformer,
    down,
    up,
)

VOCALS, BACKING = 0, 1


class LeadBackingRoformer(PooledMelBandRoformer):

    def __init__(self, dim: int, *, split_hidden: int = 0, **kwargs):
        super().__init__(dim, **kwargs)
        self.split_hidden = split_hidden
        self.split_estimator = nn.ModuleList(
            self._split_band(dim, dim_in)
            for dim_in in self.mask_estimators[0].dim_inputs
        )

    def _split_band(self, dim: int, dim_in: int) -> nn.Module:
        out = nn.Linear(self.split_hidden or dim * 4, dim_in * 2)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        if not self.split_hidden:
            return out
        return nn.Sequential(nn.Linear(dim, self.split_hidden), nn.Tanh(), out)

    def scale_backing(self, factor: float) -> None:
        with torch.no_grad():
            for split in self.split_estimator:
                out = split if not self.split_hidden else split[-1]
                values = out.out_features // 2
                out.weight[:values] *= factor
                out.bias[:values] *= factor

    def heads(self, x: torch.Tensor) -> torch.Tensor:
        frames = x.shape[1]
        bands = down(x, self.mask_pool, 1).unbind(dim=-2)
        estimator = self.mask_estimators[0]
        vocals, backing = [], []
        for band, mlp, split in zip(
            bands, estimator.to_freqs, self.split_estimator, strict=True
        ):
            body, glu = mlp[0], mlp[1]
            hidden = body[:-1](band)
            vocals.append(glu(body[-1](hidden)))
            backing.append(glu(split(band if self.split_hidden else hidden)))
        masks = torch.stack([torch.cat(vocals, -1), torch.cat(backing, -1)], dim=1)
        masks = up(masks, self.mask_pool, 2, frames)
        return rearrange(masks, "b n t (f c) -> b n f t c", c=2)

    def net_forward(self, stft_repr: torch.Tensor) -> torch.Tensor:
        return self.heads(self.trunk(stft_repr))

    def stem_spectra(self, stft_repr: torch.Tensor, masks: torch.Tensor):
        mix = torch.view_as_complex(rearrange(stft_repr, "b f t c -> b 1 f t c"))
        averaged = self.average_masks(masks.float(), mix)
        vocals = mix * averaged[:, VOCALS : VOCALS + 1]
        backing = vocals * averaged[:, BACKING : BACKING + 1]
        return vocals, backing

    def separate(self, raw_audio: torch.Tensor, masks_dtype=None) -> dict:
        stft_repr = self.encode_stft(raw_audio)
        if masks_dtype is None:
            masks = self.net_forward(stft_repr)
        else:
            with torch.autocast("cuda", dtype=masks_dtype):
                masks = self.net_forward(stft_repr)
        vocals, backing = self.stem_spectra(stft_repr, masks)
        waves = self.istft_stems(
            torch.cat([vocals, backing], dim=1), raw_audio.shape[-1]
        )
        vocals_wave, backing_wave = waves[:, VOCALS], waves[:, BACKING]
        return dict(
            vocals=vocals_wave,
            backing=backing_wave,
            lead=vocals_wave - backing_wave,
            instrumental=raw_audio - vocals_wave,
        )
