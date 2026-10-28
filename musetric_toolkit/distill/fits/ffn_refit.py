import argparse
import json
import logging
from pathlib import Path

import torch

from musetric_toolkit.distill.ffn import ffn_layers
from musetric_toolkit.distill.fits.arguments import add_data, add_out
from musetric_toolkit.distill.fits.crops import TrainingCrops
from musetric_toolkit.distill.fits.ffn_importance import TEACHER_STUDENT
from musetric_toolkit.distill.fits.least_squares import (
    add_gram,
    output_error,
    ridge_refit,
)
from musetric_toolkit.distill.student import Fits, build_student

LOG = logging.getLogger(__name__)


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "fit-ffn-refit",
        parents=[common],
        help="refit the down projection of narrowed FFNs by least squares "
        "(fits ffn_refit, modifier +fNNr)",
    )
    add_data(parser)
    add_out(parser, "File of the kept units and the refit down projections")
    parser.add_argument(
        "--keep", type=int, required=True, help="Percent of hidden units kept"
    )
    parser.add_argument("--pairs", default="01", help="Pairs whose FFNs are refit")
    parser.add_argument(
        "--ranking",
        required=True,
        type=Path,
        help="Unit ranking of fit-ffn-importance",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="Relative to the mean diagonal of G[K, K]",
    )
    parser.add_argument(
        "--select",
        choices=["importance", "obs"],
        default="importance",
        help="obs: drop units one by one by the error each adds after the refit",
    )
    parser.set_defaults(run=refit_ffns)


def obs_select(down, gram: torch.Tensor, n: int, ridge: float) -> list[int]:
    units = down.in_features
    eye = torch.eye(units + 1, dtype=gram.dtype, device=gram.device)
    g = gram + ridge * gram.diagonal()[:units].mean() * eye
    inv = torch.linalg.inv(g)
    w = torch.cat([down.weight.double(), down.bias.double()[:, None]], 1)
    w = w.to(gram.device)
    alive = torch.ones(units + 1, dtype=torch.bool, device=gram.device)
    for _ in range(units - n):
        d = inv.diagonal()
        score = (w**2).sum(0) / d
        score[~alive] = torch.inf
        score[units] = torch.inf
        j = int(score.argmin())
        w -= w[:, j : j + 1] @ inv[j : j + 1, :] / inv[j, j]
        inv -= inv[:, j : j + 1] @ inv[j : j + 1, :] / inv[j, j]
        inv[j, :] = 0
        inv[:, j] = 0
        inv[j, j] = 1
        w[:, j] = 0
        alive[j] = False
    return torch.nonzero(alive[:units]).flatten().tolist()


def unit_grams(model, targets: list[int], crops: TrainingCrops) -> dict:
    ffns = ffn_layers(model)
    grams = {
        i: torch.zeros(
            ffns[i].net[1].out_features + 1,
            ffns[i].net[1].out_features + 1,
            dtype=torch.float64,
            device="cuda",
        )
        for i in targets
    }

    def hook(i):
        def record(_module, _inputs, output):
            add_gram(grams[i], output)

        return record

    handles = [ffns[i].net[2].register_forward_hook(hook(i)) for i in targets]
    crops.run(model)
    for handle in handles:
        handle.remove()
    return grams


def refit_down(down, gram: torch.Tensor, kept: list[int], ridge: float):
    keep = torch.tensor([*kept, down.in_features], device="cuda")
    full = torch.cat([down.weight.double(), down.bias.double()[:, None]], 1)
    a = ridge_refit(full, gram, keep, ridge)
    plain = torch.zeros_like(full)
    plain[:, keep] = full[:, keep]
    fitted = torch.zeros_like(full)
    fitted[:, keep] = a
    errors = output_error(plain, full, gram), output_error(fitted, full, gram)
    return (a[:, :-1].float().cpu(), a[:, -1].float().cpu()), errors


def refit_ffns(args: argparse.Namespace) -> None:
    model = build_student(TEACHER_STUDENT, args.teacher, Fits()).eval()
    ffns = ffn_layers(model)
    chosen = json.loads(args.ranking.read_text())
    pairs = [int(c) for c in args.pairs]
    targets = [i for i in range(len(ffns)) if i // 2 in pairs]
    grams = unit_grams(model, targets, TrainingCrops.from_args(args))
    refit, kept_units = {}, {}
    for i in targets:
        down = ffns[i].net[4]
        n = round(down.in_features * args.keep / 100)
        if args.select == "obs":
            kept = obs_select(down, grams[i], n, args.ridge)
        else:
            kept = sorted(chosen[i][:n])
        kept_units[i] = kept
        refit[i], (plain, fitted) = refit_down(down, grams[i], kept, args.ridge)
        LOG.info(
            f"FFN {i}: kept {n} of {down.in_features}; "
            f"output error plain {plain:.3f}, refit {fitted:.3f}"
        )
    torch.save(
        dict(
            keep=args.keep,
            units=args.ranking.name,
            select=args.select,
            kept=kept_units,
            refit=refit,
        ),
        args.out,
    )
    LOG.info(f"-> {args.out}")
