import argparse
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from musetric_toolkit.distill.fits.arguments import add_data, add_out, add_student
from musetric_toolkit.distill.fits.crops import TrainingCrops
from musetric_toolkit.distill.pooled import (
    BANDS,
    DIM,
    down,
    merge_high,
    run_pairs,
    schedule_stack,
    unmerge_high,
    up,
)
from musetric_toolkit.distill.precision import PRECISION
from musetric_toolkit.distill.student import Fits, build_student
from musetric_toolkit.distill.teacher import band_features, teacher_layers

LOG = logging.getLogger(__name__)

DIAGONAL_FLOOR = 1e-3


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "fit-detail",
        parents=[common],
        help="fit the linear paths of a pooled student's pairs by least squares "
        "(fits detail, modifier +d)",
    )
    add_data(parser)
    add_student(parser, "The pooled student whose pairs get the paths")
    add_out(parser, "File of the linear path of every fitted pair")
    parser.add_argument("--pairs", default="2345", help="Pairs that get a path")
    parser.add_argument(
        "--features",
        default="d",
        help="Pooled pairs: letters of d, c, x, p, n; full-rate pairs use x",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="Relative to the mean diagonal of the Gram matrix",
    )
    parser.add_argument(
        "--per-band",
        action="store_true",
        help="One map per band instead of one per pair",
    )
    parser.add_argument(
        "--target",
        choices=["pair", "global"],
        default="pair",
        help="pair: the teacher's pair on the student's input; global: the "
        "teacher's own stream at that depth, so a path also takes up the error "
        "that reached the pair",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        help="One map per group of neighbouring bands (60 = --per-band)",
    )
    parser.set_defaults(run=fit_detail_paths)


@dataclass
class PairPath:
    index: int
    rate: int
    keep: int
    kinds: tuple[str, ...]
    groups: int
    energy: dict[str, float] = field(
        default_factory=lambda: dict(update=0.0, within=0.0, error=0.0)
    )

    @property
    def pooled(self) -> bool:
        return bool(self.rate > 1 or self.keep)

    @property
    def dim(self) -> int:
        return DIM * len(self.kinds) + 1

    def grouped(self, t: torch.Tensor) -> torch.Tensor:
        n = t.shape[1] * t.shape[0]
        per_group = BANDS // self.groups
        t = t.reshape(n, self.groups, per_group, DIM).transpose(1, 2)
        return t.reshape(-1, self.groups, DIM)

    def block_view(self, t: torch.Tensor) -> torch.Tensor:
        view = down(t, self.rate, 1)
        if self.keep:
            view = merge_high(view, self.keep)
        view = unmerge_high(view, self.keep) if self.keep else view
        return up(view, self.rate, 1, t.shape[1])


def stack_before(model, i: int) -> SimpleNamespace:
    return SimpleNamespace(
        layers=model.layers[:i],
        band_merges=model.band_merges[:i],
        detail=model.detail,
    )


def pair_outputs(model, teacher, stft: torch.Tensor, path: PairPath, target: str):
    i, k, keep = path.index, path.rate, path.keep
    x = band_features(model, stft)
    x = schedule_stack(stack_before(model, i), x, model.attention_rates[:i])
    frames = x.shape[1]
    coarse = None
    if path.pooled:
        lo = down(x, k, 1)
        if keep:
            lo = merge_high(lo, keep)
        change = run_pairs(lo, [model.layers[i]]) - lo
        coarse = up(unmerge_high(lo, keep) if keep else lo, k, 1, frames)
        change = unmerge_high(change, keep) if keep else change
        student = x + up(change, k, 1, frames)
    else:
        student = run_pairs(x, [model.layers[i]])
    if target == "global":
        expected = run_pairs(band_features(model, stft), teacher[: i + 1])
    else:
        expected = run_pairs(x, [teacher[i]])
    return x, expected, student, coarse


def features(x: torch.Tensor, coarse: torch.Tensor | None, kinds) -> dict:
    parts = dict(
        x=x,
        c=None if coarse is None else coarse.float(),
        d=None if coarse is None else x - coarse.float(),
    )
    if coarse is not None:
        detail = parts["d"]
        parts["p"] = torch.cat([detail[:, :1] * 0, detail[:, :-1]], 1)
        parts["n"] = torch.cat([detail[:, 1:], detail[:, :1] * 0], 1)
    return {kind: parts[kind] for kind in kinds}


def gather(model, teacher, mixes: list, path: PairPath, target: str):
    gram = torch.zeros(
        path.groups, path.dim, path.dim, dtype=torch.float64, device="cuda"
    )
    cross = torch.zeros(path.groups, path.dim, DIM, dtype=torch.float64, device="cuda")
    energy = path.energy
    for mix in mixes:
        with torch.no_grad():
            stft = model.encode_stft(mix)
            with torch.autocast("cuda", dtype=PRECISION.dtype):
                x, expected, student, coarse = pair_outputs(
                    model, teacher, stft, path, target
                )
            x, expected, student = x.float(), expected.float(), student.float()
            residual = path.grouped(expected - student)
            parts = features(x, coarse, path.kinds)
            ones = torch.ones(*residual.shape[:2], 1, device="cuda")
            feats = torch.cat(
                [path.grouped(parts[kind]) for kind in path.kinds] + [ones], 2
            )
            gram += torch.einsum("tgi,tgj->gij", feats, feats).double()
            cross += torch.einsum("tgi,tgj->gij", feats, residual).double()
            update = expected - x
            energy["update"] += float((update**2).sum())
            energy["error"] += float((residual**2).sum())
            if path.pooled:
                energy["within"] += float(
                    ((update - path.block_view(update)) ** 2).sum()
                )
    return gram, cross


def solve(gram: torch.Tensor, cross: torch.Tensor, path: PairPath, ridge: float):
    diag = gram.diagonal(dim1=1, dim2=2)[:, :-1].mean(1)
    lam = ridge * diag.clamp(min=DIAGONAL_FLOOR * float(diag.mean()))
    eye = torch.eye(path.dim, dtype=torch.float64, device="cuda")
    eye[-1, -1] = 0
    w = torch.linalg.solve(gram + lam[:, None, None] * eye, cross)
    energy = path.energy
    explained = 2 * float((w * cross).sum())
    kept = float(torch.einsum("gij,gik,gkj->", w, gram, w))
    after = energy["error"] - explained + kept
    u = energy["update"]
    LOG.info(
        f"pair {path.index} (x{path.rate}, merge {path.keep}, "
        f"{''.join(path.kinds)}, {path.groups} groups): error / update"
        f" {energy['error'] / u:.3f} (within-block {energy['within'] / u:.3f}),"
        f" after the linear path {after / u:.3f}"
    )
    if path.groups == 1:
        return (w[0, :-1].float().cpu(), w[0, -1].float().cpu(), path.kinds)
    return (w[:, :-1].float().cpu(), w[:, -1].float().cpu(), path.kinds)


def fit_detail_paths(args: argparse.Namespace) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = build_student(args.student, args.teacher, Fits.read(args.fits)).eval()
    teacher = teacher_layers(args.teacher)
    mixes = TrainingCrops.from_args(args).mixes()
    fits = {}
    for i in [int(c) for c in args.pairs]:
        k, keep = model.attention_rates[i], model.band_merges[i]
        pooled = k > 1 or keep
        path = PairPath(
            index=i,
            rate=k,
            keep=keep,
            kinds=tuple(args.features) if pooled else ("x",),
            groups=BANDS if args.per_band else args.groups,
        )
        gram, cross = gather(model, teacher, mixes, path, args.target)
        fits[i] = solve(gram, cross, path, args.ridge)
        model.detail[i] = (fits[i][0].cuda(), fits[i][1].cuda(), path.kinds)
    torch.save(
        dict(
            student=args.student,
            features=args.features,
            ridge=args.ridge,
            fits=fits,
        ),
        args.out,
    )
    LOG.info(f"-> {args.out}")
