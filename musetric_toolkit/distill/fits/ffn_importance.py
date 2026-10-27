import argparse
import json
import logging
from pathlib import Path

import torch

from musetric_toolkit.distill.ffn import ffn_layers
from musetric_toolkit.distill.fits.arguments import add_data, add_out
from musetric_toolkit.distill.fits.crops import TrainingCrops
from musetric_toolkit.distill.student import Fits, build_student

LOG = logging.getLogger(__name__)

TEACHER_STUDENT = "pool-s111111"


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "fit-ffn-importance",
        parents=[common],
        help="rank the hidden units of every FFN of the teacher (fits ffn_keep)",
    )
    add_data(parser)
    add_out(parser, "JSON of the unit indices per FFN, most important first")
    parser.add_argument(
        "--means",
        type=Path,
        help="Also write each unit's mean GELU output (fits ffn_means)",
    )
    parser.set_defaults(run=rank_units)


class UnitStatistics:
    def __init__(self, ffns: list):
        self.squares = [
            torch.zeros(ff.net[1].out_features, device="cuda") for ff in ffns
        ]
        self.sums = [torch.zeros(ff.net[1].out_features, device="cuda") for ff in ffns]
        self.counts = [0] * len(ffns)

    def hook(self, i: int):
        def record(_module, _inputs, output):
            flat = output.detach().float().reshape(-1, output.shape[-1])
            self.squares[i] += flat.pow(2).sum(0)
            self.sums[i] += flat.sum(0)
            self.counts[i] += flat.shape[0]

        return record

    def ranking(self, ffns: list) -> list[list[int]]:
        ranking = []
        for i, ff in enumerate(ffns):
            rms = (self.squares[i] / self.counts[i]).sqrt()
            score = rms * ff.net[4].weight.detach().float().norm(dim=0)
            ranking.append(torch.argsort(score, descending=True).tolist())
        return ranking

    def means(self) -> list[list[float]]:
        return [(s / n).tolist() for s, n in zip(self.sums, self.counts, strict=True)]


def rank_units(args: argparse.Namespace) -> None:
    model = build_student(TEACHER_STUDENT, args.teacher, Fits()).eval()
    ffns = ffn_layers(model)
    statistics = UnitStatistics(ffns)
    handles = [
        ff.net[2].register_forward_hook(statistics.hook(i)) for i, ff in enumerate(ffns)
    ]
    TrainingCrops.from_args(args).run(model)
    for handle in handles:
        handle.remove()
    ranking = statistics.ranking(ffns)
    args.out.write_text(json.dumps(ranking), encoding="utf-8")
    LOG.info(f"{len(ranking)} FFNs ranked on {args.crops} crops -> {args.out}")
    if args.means:
        args.means.write_text(json.dumps(statistics.means()), encoding="utf-8")
        LOG.info(f"unit means -> {args.means}")
