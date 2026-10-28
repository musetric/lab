import argparse
import logging

import torch

from musetric_toolkit.distill.fits.arguments import add_data, add_out, add_student
from musetric_toolkit.distill.fits.crops import TrainingCrops
from musetric_toolkit.distill.fits.least_squares import add_gram, output_error
from musetric_toolkit.distill.heads import HEAD, attentions
from musetric_toolkit.distill.student import Fits, build_student

LOG = logging.getLogger(__name__)

SEED = 13


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "fit-head-prune",
        parents=[common],
        help="keep the strongest attention heads and refit the output projection "
        "(fits head_prune, modifier +aK[t|b]@pairs)",
    )
    add_data(parser)
    add_student(parser, "The student whose attentions are pruned")
    add_out(parser, "File of the kept heads and the refit output projections")
    parser.add_argument("--keep", type=int, required=True, help="Heads kept of 8")
    parser.add_argument("--pairs", default="01", help="Pairs whose heads are pruned")
    parser.add_argument("--which", default="tb", help="t time, b band attentions")
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="Relative to the mean diagonal of the kept heads' Gram matrix",
    )
    parser.set_defaults(run=prune_attention_heads)


def head_grams(attns: list, targets: list[int], model, crops: TrainingCrops) -> dict:
    grams = {
        i: torch.zeros(
            attns[i].heads * HEAD + 1,
            attns[i].heads * HEAD + 1,
            dtype=torch.float64,
            device="cuda",
        )
        for i in targets
    }

    def hook(i):
        def record(_module, inputs):
            add_gram(grams[i], inputs[0])

        return record

    handles = [attns[i].to_out[0].register_forward_pre_hook(hook(i)) for i in targets]
    crops.run(model)
    for handle in handles:
        handle.remove()
    return grams


def head_shares(w: torch.Tensor, g: torch.Tensor, heads: int) -> list[float]:
    shares = []
    for h in range(heads):
        cols = slice(h * HEAD, (h + 1) * HEAD)
        energy = torch.einsum("oi,ij,oj->", w[:, cols], g[cols, cols], w[:, cols])
        shares.append(float(energy))
    return shares


def strongest_heads(shares: list[float], keep: int) -> list[int]:
    return sorted(sorted(range(len(shares)), key=lambda h: -shares[h])[:keep])


def refit_output(attn, g: torch.Tensor, keep: int, ridge: float):
    heads = attn.heads
    weight = attn.to_out[0].weight.double()
    constant = torch.zeros(weight.shape[0], 1, dtype=torch.float64, device="cuda")
    w = torch.cat([weight, constant], 1)
    shares = head_shares(w, g, heads)
    kept = strongest_heads(shares, keep)
    columns = [h * HEAD + d for h in kept for d in range(HEAD)]
    cols = torch.tensor([*columns, heads * HEAD], device="cuda")
    gkk = g[cols][:, cols]
    lam = ridge * gkk.diagonal()[:-1].mean()
    eye = torch.eye(len(cols), dtype=torch.float64, device="cuda")
    eye[-1, -1] = 0
    a = w @ g[:, cols] @ torch.linalg.inv(gkk + lam * eye)
    plain = torch.zeros_like(w)
    plain[:, cols] = w[:, cols]
    fitted = torch.zeros_like(w)
    fitted[:, cols] = a
    errors = output_error(plain, w, g), output_error(fitted, w, g)
    fit = (kept, a[:, :-1].float().cpu(), a[:, -1].float().cpu())
    return fit, [round(s / sum(shares), 3) for s in shares], errors


def prune_attention_heads(args: argparse.Namespace) -> None:
    model = build_student(args.student, args.teacher, Fits.read(args.fits)).eval()
    attns = attentions(model)
    pairs = [int(c) for c in args.pairs]
    targets = [
        i for i in range(len(attns)) if i // 2 in pairs and "tb"[i % 2] in args.which
    ]
    crops = TrainingCrops.from_args(args, SEED)
    grams = head_grams(attns, targets, model, crops)
    fits = {}
    for i in targets:
        fit, shares, (plain, fitted) = refit_output(
            attns[i], grams[i], args.keep, args.ridge
        )
        LOG.info(
            f"attention {i} (pair {i // 2}, {'tb'[i % 2]}): kept {fit[0]}, shares "
            f"{shares}; output error plain {plain:.3f}, refit {fitted:.3f}"
        )
        fits[i] = fit
    torch.save(dict(student=args.student, keep=args.keep, fits=fits), args.out)
    LOG.info(f"-> {args.out}")
