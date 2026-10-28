import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from musetric_toolkit.distill.corpus import dataset_rows, usable
from musetric_toolkit.distill.inference import student_forward
from musetric_toolkit.distill.sampler import Sampler, SamplerOptions

SEED = 7


@dataclass
class TrainingCrops:
    labels: Path
    dataset: Path | None
    count: int
    seed: int = SEED

    @classmethod
    def from_args(cls, args: argparse.Namespace, seed: int = SEED) -> "TrainingCrops":
        return cls(args.labels, args.dataset, args.crops, seed)

    def sampler(self) -> Sampler:
        rows = dataset_rows(self.dataset)
        labels = json.loads((self.labels / "labels.json").read_text())
        content = {i: row["content"] for i, row in rows.items()}
        index = usable(labels, rows)
        return Sampler(self.labels, index, content, self.seed, SamplerOptions())

    def mixes(self) -> list[torch.Tensor]:
        sampler = self.sampler()
        return [sampler.batch(1)[0] for _ in range(self.count)]

    def run(self, model) -> None:
        sampler = self.sampler()
        with torch.no_grad():
            for _ in range(self.count):
                x, _, _ = sampler.batch(1)
                student_forward(model, x)
