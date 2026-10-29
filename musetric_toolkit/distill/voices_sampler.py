import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.distill.corpus import FULL_SCALE, INT16_MAX
from musetric_toolkit.distill.teacher import CHUNK
from musetric_toolkit.distill.voices import backing_scale

SCALE = FULL_SCALE / INT16_MAX
VOICED_SHARE = 0.1
PELL_SHARE = 0.8
CROP_TRIES = 12
BACKING_DB = -25.0
PROBE_STEP = 64
CHANNEL_SWAP = 0.5
VOICE_FLOOR = 1e-3
MAX_REMIX_GAIN = 10 ** (30 / 20)


@dataclass
class VoicesOptions:
    remix: float = 0.4
    backing: float = 0.5
    pell: float = 0.0
    remix_vocal_db: tuple[float, float] = (-6.0, 3.0)


@dataclass
class VoicesExample:
    x: torch.Tensor
    vocals: torch.Tensor
    backing: torch.Tensor


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12)


class VoicesSampler:
    def __init__(self, labels: Path, seed: int, options: VoicesOptions):
        self.rng = np.random.default_rng(seed)
        self.options = options
        self.scales = dict(
            mix=SCALE, vocals=SCALE, backing=SCALE * backing_scale(labels)
        )
        index = json.loads((labels / "voices.json").read_text())
        train = [
            t
            for t in index
            if t["role"] == "train" and t["seconds"] * 44100 > CHUNK + 44100
        ]
        self.songs = [t for t in train if not t["pell"]]
        self.voiced = [t for t in self.songs if t["vocal_share"] >= VOICED_SHARE]
        self.pells = [t for t in train if t["pell"] and t["vocal_share"] >= PELL_SHARE]
        self.parts = {
            t["id"]: {
                kind: np.load(labels / f"{t['id']}.{kind}.npy", mmap_mode="r")
                for kind in ("mix", "vocals", "backing")
            }
            for t in train
        }

    def crop(self, track: dict, want_backing: bool, need_voice: bool = True):
        parts = self.parts[track["id"]]
        n = parts["mix"].shape[1]
        best = None
        for _ in range(CROP_TRIES):
            s = int(self.rng.integers(0, n - CHUNK))
            voice = rms(parts["vocals"][:, s : s + CHUNK : PROBE_STEP])
            voice *= self.scales["vocals"]
            if need_voice and voice < VOICE_FLOOR:
                continue
            backing = rms(parts["backing"][:, s : s + CHUNK : PROBE_STEP])
            backing *= self.scales["backing"]
            ratio = 20 * np.log10(backing / voice)
            if best is None or ratio > best[0]:
                best = (ratio, s)
            if not want_backing or ratio > BACKING_DB:
                break
        s = best[1] if best else int(self.rng.integers(0, n - CHUNK))
        return {
            k: v[:, s : s + CHUNK].astype(np.float32) * self.scales[k]
            for k, v in parts.items()
        }

    def pick(self, pool: list[dict]) -> dict:
        weights = np.array([t["seconds"] for t in pool], dtype=np.float64)
        return pool[int(self.rng.choice(len(pool), p=weights / weights.sum()))]

    def pick_voiced(self, pool: list[dict], want_backing: bool) -> dict:
        if not want_backing:
            return self.pick(pool)
        rich = [t for t in pool if t["backing_share"] > 10 ** (BACKING_DB / 20)]
        return self.pick(rich or pool)

    def example(self) -> VoicesExample:
        want_backing = self.rng.random() < self.options.backing
        use_pell = bool(self.pells) and self.rng.random() < self.options.pell
        pool = self.pells if use_pell else self.voiced
        voice = self.crop(self.pick_voiced(pool, want_backing), want_backing)
        if use_pell or self.rng.random() < self.options.remix:
            other = self.crop(self.pick(self.songs), False, need_voice=False)
            accompaniment = other["mix"] - other["vocals"]
            low, high = self.options.remix_vocal_db
            target = 10 ** (self.rng.uniform(low, high) / 20) * rms(accompaniment)
            gain = min(target / rms(voice["vocals"]), MAX_REMIX_GAIN)
            vocals, backing = voice["vocals"] * gain, voice["backing"] * gain
            mix = accompaniment + vocals
        else:
            mix, vocals, backing = voice["mix"], voice["vocals"], voice["backing"]
        if self.rng.random() < CHANNEL_SWAP:
            mix, vocals, backing = mix[::-1], vocals[::-1], backing[::-1]
        peak = max(float(np.abs(np.stack([mix, vocals, backing])).max()), 1e-8)
        gain = self.rng.uniform(0.3, 1.0) / peak
        return VoicesExample(
            *(
                torch.from_numpy(np.ascontiguousarray(a * gain))
                for a in (mix, vocals, backing)
            )
        )

    def batch(self, size: int) -> VoicesExample:
        examples = [self.example() for _ in range(size)]
        return VoicesExample(
            *(
                torch.stack([getattr(e, f) for e in examples]).cuda()
                for f in ("x", "vocals", "backing")
            )
        )
