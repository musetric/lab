from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.distill.teacher import CHUNK, SAMPLE_RATE
from musetric_toolkit.distill.verdicts import NO_VOICE, TRAIN
from musetric_toolkit.distill.voices_sampler import (
    CHANNEL_SWAP,
    VoicesOptions,
    VoicesSampler,
)

FADE = SAMPLE_RATE // 20


@dataclass
class VerdictOptions(VoicesOptions):
    share: float = 0.2


@dataclass
class VerdictExample:
    x: torch.Tensor
    gate: torch.Tensor
    backing: torch.Tensor


def silence_gate(start: int, end: int) -> np.ndarray:
    gate = np.ones(CHUNK, dtype=np.float32)
    gate[start:end] = 0.0
    fade = min(FADE, (end - start) // 2)
    ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32)
    gate[start : start + fade] = ramp
    gate[end - fade : end] = ramp[::-1]
    return gate


class VerdictSampler(VoicesSampler):
    def __init__(
        self,
        labels: Path,
        seed: int,
        options: VerdictOptions,
        verdicts: list[dict],
        withheld: frozenset[str],
    ):
        super().__init__(labels, seed, options, withheld)
        self.silent = [
            row for row in verdicts if row["use"] == TRAIN and row["answer"] == NO_VOICE
        ]
        if not self.silent:
            raise ValueError("no clips to train on")
        self.silent_parts = {
            row["id"]: {
                kind: np.load(labels / f"{row['id']}.{kind}.npy", mmap_mode="r")
                for kind in ("mix", "backing")
            }
            for row in self.silent
        }

    def silent_example(self) -> VerdictExample:
        row = self.silent[int(self.rng.integers(len(self.silent)))]
        parts = self.silent_parts[row["id"]]
        length = parts["mix"].shape[1]
        start = int(row["start_s"] * SAMPLE_RATE)
        end = min(start + int(row["seconds"] * SAMPLE_RATE), length)
        low = max(0, end - CHUNK)
        high = min(start, length - CHUNK)
        crop = int(self.rng.integers(low, high + 1))
        gate = silence_gate(max(0, start - crop), min(CHUNK, end - crop))
        mix = parts["mix"][:, crop : crop + CHUNK].astype(np.float32)
        mix = mix * self.scales["mix"]
        backing = parts["backing"][:, crop : crop + CHUNK].astype(np.float32)
        backing = backing * self.scales["backing"]
        if self.rng.random() < CHANNEL_SWAP:
            mix, backing = mix[::-1], backing[::-1]
        peak = max(float(np.abs(mix).max()), 1e-8)
        scale = self.rng.uniform(0.3, 1.0) / peak
        return VerdictExample(
            torch.from_numpy(np.ascontiguousarray(mix * scale)),
            torch.from_numpy(gate),
            torch.from_numpy(np.ascontiguousarray(backing * scale)),
        )

    def example(self) -> VerdictExample:
        if self.rng.random() < self.options.share:
            return self.silent_example()
        plain = super().example()
        return VerdictExample(plain.x, torch.ones(CHUNK), plain.backing)
