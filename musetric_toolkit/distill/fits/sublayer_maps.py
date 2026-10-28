import argparse
import logging
from pathlib import Path

import torch

from musetric_toolkit.distill.checkpoints import load_layers
from musetric_toolkit.distill.fits.affine import fit_affine
from musetric_toolkit.distill.fits.arguments import add_data, add_out, add_student
from musetric_toolkit.distill.fits.crops import TrainingCrops
from musetric_toolkit.distill.student import Fits, build_student
from musetric_toolkit.distill.sublayers import install, target

LOG = logging.getLogger(__name__)


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "fit-sublayer",
        parents=[common],
        help="fit the sub-layer replacements of a student by least squares "
        "(fits sublayer, modifier +x<specs>)",
    )
    add_data(parser)
    add_student(parser, "The student without the +x modifier")
    add_out(parser, "File of the fitted affine map of every lin or mix spec")
    parser.add_argument(
        "--specs",
        required=True,
        help="Replacements joined by /, such as b0a:drop/b1:lin",
    )
    parser.add_argument(
        "--init-layers",
        type=Path,
        help="Fit on the transformer stack of this checkpoint",
    )
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.set_defaults(run=fit_sublayers)


def fit_sublayers(args: argparse.Namespace) -> None:
    model = build_student(args.student, args.teacher, Fits.read(args.fits)).eval()
    if args.init_layers:
        load_layers(model, args.init_layers)
    crops = TrainingCrops.from_args(args)
    fits = {}
    for spec in args.specs.split("/"):
        where, how = spec.split(":")
        module, _, part = target(model, where)
        if how == "drop":
            install(model, spec, None)
            LOG.info(spec)
            continue
        w, residual = fit_affine(model, module, part == "", crops, args.ridge)
        fits[spec] = (w[:-1].cpu(), w[-1].cpu())
        install(model, spec, fits[spec])
        LOG.info(f"{spec}: unexplained {residual:.3f}")
    torch.save(dict(student=args.student, specs=args.specs, fits=fits), args.out)
    LOG.info(f"-> {args.out}")
