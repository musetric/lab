import argparse
import logging
from dataclasses import dataclass

import torch

from musetric_toolkit.distill.fits.arguments import add_data, add_out, add_student
from musetric_toolkit.distill.fits.crops import TrainingCrops
from musetric_toolkit.distill.fits.least_squares import add_gram
from musetric_toolkit.distill.student import Fits, build_student

LOG = logging.getLogger(__name__)


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "fit-head-lowrank",
        parents=[common],
        help="factorise the middle layer of the narrowed mask estimator by a "
        "whitened SVD (fits head_lowrank, modifiers +hrR and +hsR)",
    )
    add_data(parser)
    add_student(parser, "A student with +nNN, whose head is factorised")
    add_out(parser, "File of the low-rank factors per band")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--kind", choices=["band", "shared"], required=True)
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-6,
        help="Eigenvalue floor relative to the largest, per Gram",
    )
    parser.add_argument(
        "--group",
        type=int,
        default=15,
        help="Bands per pass over the crops (GPU memory)",
    )
    parser.set_defaults(run=factorise_head)


@dataclass
class Whitened:
    weights: list[torch.Tensor]
    whitened: list[torch.Tensor]
    chols: list[torch.Tensor]
    inverses: list[torch.Tensor]


def input_grams(model, mlps: list, group: int, crops: TrainingCrops) -> list:
    hidden = mlps[0][2].in_features
    grams = [None] * len(mlps)
    for start in range(0, len(mlps), group):
        part = range(start, min(start + group, len(mlps)))
        acc = {
            b: torch.zeros(hidden + 1, hidden + 1, dtype=torch.float64, device="cuda")
            for b in part
        }

        def hook(b, acc=acc):
            def record(_module, _inputs, output):
                add_gram(acc[b], output)

            return record

        handles = [mlps[b][1].register_forward_hook(hook(b)) for b in part]
        crops.run(model)
        for handle in handles:
            handle.remove()
        for b in part:
            grams[b] = acc[b].cpu()
        del acc
        torch.cuda.empty_cache()
    return grams


def whiten(mlps: list, grams: list, ridge: float) -> Whitened:
    out = Whitened([], [], [], [])
    for b, mlp in enumerate(mlps):
        e, q = torch.linalg.eigh(grams[b].cuda())
        e = e.clamp(min=float(e.max()) * ridge)
        chol, inverse = q * e.sqrt(), (q / e.sqrt()).T
        layer = mlp[2]
        w = torch.cat(
            [layer.weight.detach().double(), layer.bias.detach().double()[:, None]], 1
        ).to(q.device)
        out.whitened.append(w @ chol)
        out.chols.append(chol)
        out.inverses.append(inverse)
        out.weights.append(w)
    return out


def shared_basis(parts: Whitened, rank: int) -> tuple[dict, list[float]]:
    bands, errors = {}, []
    u, _, _ = torch.linalg.svd(torch.cat(parts.whitened, 1), full_matrices=False)
    basis = u[:, :rank]
    for b, w in enumerate(parts.weights):
        coeff = basis.T @ w
        missed = ((w - basis @ coeff) @ parts.chols[b]).norm()
        errors.append(float(missed / parts.whitened[b].norm()))
        bands[b] = (coeff.float().cpu(), basis.float().cpu())
    return bands, errors


def band_factors(parts: Whitened, rank: int) -> tuple[dict, list[float]]:
    bands, errors = {}, []
    for b, m in enumerate(parts.whitened):
        u, s, vh = torch.linalg.svd(m, full_matrices=False)
        a = u[:, :rank] * s[:rank]
        coeff = vh[:rank] @ parts.inverses[b]
        errors.append(float((m - a @ vh[:rank]).norm() / m.norm()))
        bands[b] = (coeff.float().cpu(), a.float().cpu())
    return bands, errors


def factorise_head(args: argparse.Namespace) -> None:
    model = build_student(args.student, args.teacher, Fits.read(args.fits)).eval()
    mlps = [seq[0] for seq in model.mask_estimators[0].to_freqs]
    crops = TrainingCrops.from_args(args)
    grams = input_grams(model, mlps, args.group, crops)
    parts = whiten(mlps, grams, args.ridge)
    factorise = shared_basis if args.kind == "shared" else band_factors
    bands, errors = factorise(parts, args.rank)
    errors = torch.tensor(errors)
    LOG.info(
        f"{args.kind} rank {args.rank}: relative error on activations median "
        f"{errors.median():.3f}, max {errors.max():.3f}"
    )
    torch.save(dict(kind=args.kind, rank=args.rank, bands=bands), args.out)
    LOG.info(f"-> {args.out}")
