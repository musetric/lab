import argparse
import logging
from pathlib import Path

import torch

from musetric_toolkit.distill.checkpoints import LAYERS, without_rotary
from musetric_toolkit.distill.fits.arguments import add_data, add_out, add_student
from musetric_toolkit.distill.fits.crops import TrainingCrops
from musetric_toolkit.distill.fits.ffn_importance import TEACHER_STUDENT
from musetric_toolkit.distill.fits.least_squares import add_gram, ridge_refit
from musetric_toolkit.distill.student import Fits, build_student

LOG = logging.getLogger(__name__)

TANH_LAYERS = (1, 3)


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "fit-head-refit",
        parents=[common],
        help="narrow the mask estimator's hidden layers with a least-squares refit "
        "(fits head_refit, modifier +nNN)",
    )
    add_data(parser)
    add_student(
        parser, "Whose activations the head is fitted on", default=TEACHER_STUDENT
    )
    add_out(parser, "File of the kept units and the refit layers per band")
    parser.add_argument(
        "--keep",
        type=int,
        required=True,
        help="Percent of each hidden layer kept",
    )
    parser.add_argument(
        "--group", type=int, default=15, help="Bands per pass (GPU memory)"
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--init", type=Path, help="That student's trained checkpoint")
    parser.set_defaults(run=refit_head)


def load_stack(model, path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    layers = {
        k[len(LAYERS) :]: v
        for k, v in without_rotary(state).items()
        if k.startswith(LAYERS)
    }
    model.layers.load_state_dict(layers, strict=False)
    return len(layers)


def refit(weight, bias, gram: torch.Tensor, kept: list[int], ridge: float):
    units = weight.shape[1]
    keep = torch.tensor([*kept, units], device=gram.device)
    w = torch.cat([weight.double(), bias.double()[:, None]], 1).to(gram.device)
    a = ridge_refit(w, gram, keep, ridge)
    return a[:, :-1].float().cpu(), a[:, -1].float().cpu()


def tanh_grams(model, mlps: list, bands: list[int], crops: TrainingCrops) -> dict:
    hidden = mlps[0][0].out_features
    grams = {
        (b, layer): torch.zeros(
            hidden + 1, hidden + 1, dtype=torch.float64, device="cuda"
        )
        for b in bands
        for layer in TANH_LAYERS
    }

    def hook(b, layer):
        def record(_module, _inputs, output):
            add_gram(grams[(b, layer)], output)

        return record

    handles = [
        mlps[b][layer].register_forward_hook(hook(b, layer))
        for b in bands
        for layer in TANH_LAYERS
    ]
    crops.run(model)
    for handle in handles:
        handle.remove()
    return grams


def strongest_units(gram: torch.Tensor, following, hidden: int, n: int) -> list[int]:
    rms = gram.diagonal()[:hidden].sqrt().float().cpu()
    score = rms * following.weight.detach().float().cpu().norm(dim=0)
    return sorted(torch.argsort(score, descending=True)[:n].tolist())


def fit_band(mlp, g1: torch.Tensor, g3: torch.Tensor, n: int, ridge: float):
    hidden = mlp[0].out_features
    k1 = strongest_units(g1, mlp[2], hidden, n)
    k2 = strongest_units(g3, mlp[4], hidden, n)
    w2, b2 = refit(mlp[2].weight.detach(), mlp[2].bias.detach(), g1, k1, ridge)
    w3, b3 = refit(mlp[4].weight.detach(), mlp[4].bias.detach(), g3, k2, ridge)
    w1, b1 = mlp[0].weight.detach()[k1].cpu(), mlp[0].bias.detach()[k1].cpu()
    return (k1, w1, b1, w2[k2], b2[k2], k2, w3, b3)


def refit_head(args: argparse.Namespace) -> None:
    model = build_student(args.student, args.teacher, Fits.read(args.fits)).eval()
    if args.init:
        count = load_stack(model, args.init)
        LOG.info(f"init layers: {args.init} ({count} tensors)")
    mlps = [seq[0] for seq in model.mask_estimators[0].to_freqs]
    crops = TrainingCrops.from_args(args)
    n = round(mlps[0][0].out_features * args.keep / 100)
    fits = {}
    for start in range(0, len(mlps), args.group):
        bands = list(range(start, min(start + args.group, len(mlps))))
        grams = tanh_grams(model, mlps, bands, crops)
        for b in bands:
            g1, g3 = grams[(b, TANH_LAYERS[0])], grams[(b, TANH_LAYERS[1])]
            fits[b] = fit_band(mlps[b], g1, g3, n, args.ridge)
        LOG.info(f"bands {bands[0]}-{bands[-1]} fitted")
        del grams
        torch.cuda.empty_cache()
    torch.save(dict(keep=args.keep, fits=fits), args.out)
    LOG.info(f"-> {args.out}")
