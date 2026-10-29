import argparse
import importlib
import logging
import math
from pathlib import Path

EXTENSIONS = [
    "musetric_toolkit.distill.fits.ffn_importance",
    "musetric_toolkit.distill.fits.ffn_refit",
    "musetric_toolkit.distill.fits.head_prune",
    "musetric_toolkit.distill.fits.head_refit",
    "musetric_toolkit.distill.fits.head_lowrank",
    "musetric_toolkit.distill.fits.detail_paths",
    "musetric_toolkit.distill.fits.sublayer_maps",
    "musetric_toolkit.distill.soup",
    "musetric_toolkit.distill.release",
    "musetric_toolkit.distill.musdb",
    "musetric_toolkit.distill.musdb_bleed",
    "musetric_toolkit.distill.data.open_corpus",
    "musetric_toolkit.distill.data.fma_fetch",
    "musetric_toolkit.distill.data.fma_expand",
    "musetric_toolkit.distill.data.ccmixter_fetch",
    "musetric_toolkit.distill.data.musdb_fetch",
    "musetric_toolkit.distill.data.ccmixter_voices",
    "musetric_toolkit.distill.data.analyze",
    "musetric_toolkit.distill.data.screen",
    "musetric_toolkit.distill.data.screen_dedupe",
    "musetric_toolkit.distill.data.curate",
    "musetric_toolkit.distill.data.corpus_variant",
    "musetric_toolkit.distill.data.corpus_disagreement",
    "musetric_toolkit.distill.data.corpus_marks",
    "musetric_toolkit.distill.data.mine_natural",
    "musetric_toolkit.distill.voices",
    "musetric_toolkit.distill.voices_training",
    "musetric_toolkit.distill.voices_scores",
    "musetric_toolkit.distill.voices_release",
]
NEW_PARTS = (
    r"^(detail_paths|mask_estimators)\.|^layers\.[01]\.[01]\.layers\.0\.0\."
    r"to_out\.|^layers\.[01]\.[01]\.layers\.0\.1\.net\.4\."
)


def lazy(module: str, function: str):
    def run(args) -> None:
        getattr(importlib.import_module(module), function)(args)

    return run


def add_teacher(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--teacher",
        required=True,
        type=Path,
        help="Directory with the teacher's config.yaml and model.ckpt",
    )


def add_labels(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Directory of the teacher's labels and labels.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="dataset.csv of the corpus: decision, role, content and genre per track",
    )


def add_student(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fits",
        type=Path,
        help="JSON of the fit files the student's modifiers read",
    )


def add_label(commands, common) -> None:
    parser = commands.add_parser(
        "label", parents=[common], help="label the corpus with the teacher"
    )
    add_teacher(parser)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Directory with manifest.json, fma.json and ccmixter-pells.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="A separate list of tracks to label instead of the corpus",
    )
    parser.set_defaults(run=lazy("musetric_toolkit.distill.labelling", "label_command"))


def add_schedule(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hours", type=float, default=9.0)
    parser.add_argument("--wall-hours", type=float, default=math.inf)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    parser.add_argument("--decay-share", type=float, default=0.2)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ema", type=float, default=0.0)
    parser.add_argument("--lr-new", type=float, default=1.0)
    parser.add_argument("--lr-new-parts", default=NEW_PARTS)


def add_sampling(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remix", type=float, default=0.5)
    parser.add_argument("--silent", type=float, default=0.1)
    parser.add_argument("--voiced-db", default="-40")
    parser.add_argument("--keep-quiet", action="store_true")
    parser.add_argument("--level", help="share,low_db,high_db")
    parser.add_argument("--pause-share", type=float, default=0.0)
    parser.add_argument("--silence-share", type=float, default=0.0)
    parser.add_argument("--remix-vocal-db", default="-6,3")
    parser.add_argument(
        "--remix-acc",
        default="any",
        help="any, instrumental, or instrumental:Genre+Genre",
    )
    parser.add_argument("--natural", help="events.json,share[,kind+kind]")
    parser.add_argument("--clip-seconds", type=float, default=0.0)
    parser.add_argument("--pell-share", type=float, default=0.0)
    parser.add_argument("--marks", type=Path)
    parser.add_argument("--overfit", type=int, default=0)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--prefetch-workers", type=int, default=1)


def add_losses(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-weight", type=float, default=0.0)
    parser.add_argument("--mask-weight", type=float, default=0.0)
    parser.add_argument("--pair-targets", action="store_true")
    parser.add_argument(
        "--wave-target", choices=["stored", "teacher"], default="stored"
    )
    parser.add_argument("--pause-loss", type=float, default=0.0)
    parser.add_argument("--marks-weight", type=float, default=1.0)
    parser.add_argument("--marks-excess", type=float, default=0.0)
    parser.add_argument("--truth-excess", type=float, default=0.0)
    parser.add_argument("--truth-leak", type=float, default=0.0)
    parser.add_argument("--voice-deficit", type=float, default=0.0)
    parser.add_argument("--gate-quiet", type=float, default=0.0)


def add_checkpoints(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quick-minutes", type=float, default=5.0)
    parser.add_argument("--guard-minutes", type=float, default=10.0)
    parser.add_argument("--guard-sdr", type=float, default=1.0)
    parser.add_argument("--eval-minutes", type=float, default=30.0)
    parser.add_argument("--save-minutes", type=float, default=10.0)
    parser.add_argument("--snapshot-steps", type=int, default=0)
    parser.add_argument("--snapshots", action="store_true")
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-start-eval", action="store_true")
    parser.add_argument("--init", type=Path)
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--init-layers", type=Path)
    parser.add_argument("--freeze-transformer", action="store_true")


def add_train(commands, common) -> None:
    parser = commands.add_parser(
        "train", parents=[common], help="distil the teacher into a student"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--student", required=True, help="a pool-s student name")
    add_teacher(parser)
    add_labels(parser)
    add_student(parser)
    add_schedule(parser)
    add_sampling(parser)
    add_losses(parser)
    add_checkpoints(parser)
    parser.set_defaults(run=lazy("musetric_toolkit.distill.training", "train"))


def add_evaluate(commands, common) -> None:
    parser = commands.add_parser(
        "evaluate",
        parents=[common],
        help="score a saved student on the held-out tracks",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    add_teacher(parser)
    add_labels(parser)
    add_student(parser)
    parser.add_argument("--split", choices=["held", "train"], default="held")
    parser.add_argument("--limit", type=int, default=150)
    parser.set_defaults(
        run=lazy("musetric_toolkit.distill.held_out", "score_checkpoint")
    )


def parse_arguments() -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warn", "error"],
        help="Set the logging level",
    )
    parser = argparse.ArgumentParser(
        description="Distil the vocals separation teacher into a pooled student"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    add_label(commands, common)
    add_train(commands, common)
    add_evaluate(commands, common)
    for name in EXTENSIONS:
        importlib.import_module(name).add_parser(commands, common)
    return parser.parse_args()


def main_cli() -> None:
    args = parse_arguments()
    level = {"warn": "WARNING"}.get(args.log_level, args.log_level.upper())
    logging.basicConfig(level=level)
    args.run(args)
