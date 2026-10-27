import argparse
import importlib
import logging
import sys

from musetric_toolkit.common.logger import redirect_std_streams
from musetric_toolkit.common.paths import default_models_path
from musetric_toolkit.rhythm_audio.main_cli import (
    apply_models_path,
    build_stream_suppression_patterns,
    setup_cli_logging,
)

COMMAND_MODULES = {
    "reference": "musetric_toolkit.rhythm_bench.reference",
    "mean": "musetric_toolkit.rhythm_bench.mean",
    "score": "musetric_toolkit.rhythm_bench.score",
    "windows": "musetric_toolkit.rhythm_bench.windows",
    "listen": "musetric_toolkit.rhythm_bench.listen",
    "stems": "musetric_toolkit.rhythm_bench.stems",
}


def _results_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory of <track>/<variant>.json results",
    )


def _audio(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--audio-path",
        required=True,
        help="Audio file, or a directory whose audio files are each a track",
    )
    parser.add_argument(
        "--file-name",
        help=(
            "In a directory, take <track>/<file-name> from each subdirectory"
            " instead, such as instrumental.flac of a corpus of stems"
        ),
    )


def _out(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        help="Output path without suffix (default: in the results directory)",
    )


def _add_reference(commands, common: argparse.ArgumentParser) -> None:
    reference = commands.add_parser(
        "reference",
        parents=[common],
        help=(
            "run Beat This! in torch and write its logits and the "
            "<label>-w<window>-minimal and -dbn variants"
        ),
    )
    _audio(reference)
    _results_dir(reference)
    reference.add_argument(
        "--label",
        required=True,
        help="Name of the input in the variant names, such as instrumental or mix",
    )
    reference.add_argument(
        "--window",
        type=int,
        default=1500,
        help="Window in frames (default: 1500, the app's and the author's)",
    )
    reference.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu, where the counts of #1108 are taken)",
    )
    reference.add_argument(
        "--overwrite",
        action="store_true",
        help="Run again for tracks whose variants exist",
    )


def _add_mean(commands, common: argparse.ArgumentParser) -> None:
    mean = commands.add_parser(
        "mean",
        parents=[common],
        help="postprocess the mean of the logits of two inputs",
    )
    _results_dir(mean)
    mean.add_argument(
        "--logits",
        nargs=2,
        required=True,
        help="The two logits, such as instrumental-w1500 mix-w1500",
    )
    mean.add_argument(
        "--label",
        required=True,
        help="Name of the result, such as mean-w1500",
    )


def _add_score(commands, common: argparse.ArgumentParser) -> None:
    score = commands.add_parser(
        "score",
        parents=[common],
        help="count spurious beats, gaps and tempo jumps of variants",
    )
    _results_dir(score)
    score.add_argument("--variants", nargs="+", required=True)
    score.add_argument(
        "--reference",
        help="Variant to match beats against for the F-measure",
    )
    _out(score)


def _add_windows(commands, common: argparse.ArgumentParser) -> None:
    windows = commands.add_parser(
        "windows",
        parents=[common],
        help="find the windows of 15 s where two variants click differently",
    )
    _results_dir(windows)
    windows.add_argument("--variants", nargs=2, required=True)
    _out(windows)


def _add_listen(commands, common: argparse.ArgumentParser) -> None:
    listen = commands.add_parser(
        "listen",
        parents=[common],
        help="render the windows of a windows result with each variant's clicks",
    )
    _audio(listen)
    _results_dir(listen)
    listen.add_argument(
        "--windows",
        required=True,
        help="JSON written by windows",
    )


def _add_stems(commands, common: argparse.ArgumentParser) -> None:
    stems = commands.add_parser(
        "stems",
        parents=[common],
        help="print levels and onset periods of one track per window of 4 s",
    )
    stems.add_argument("--instrumental", required=True)
    stems.add_argument("--vocals", required=True)
    stems.add_argument("--mix", required=True)
    stems.add_argument("--start", type=float, required=True, help="Seconds")
    stems.add_argument("--end", type=float, required=True, help="Seconds")


def parse_arguments():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--models-path",
        default=default_models_path(),
        help=(
            "Set HF_HOME, HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE,"
            " TORCH_HOME to this path"
        ),
    )
    common.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warn", "error"],
        help="Set the logging level",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Check the beats the metronome would click: Beat This! references,"
            " the counts of musetric/musetric#1108 and listening renders"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _add_reference(commands, common)
    _add_mean(commands, common)
    _add_score(commands, common)
    _add_windows(commands, common)
    _add_listen(commands, common)
    _add_stems(commands, common)
    return parser.parse_args()


def main_cli() -> None:
    args = parse_arguments()
    apply_models_path(args.models_path)
    setup_cli_logging(args.log_level)
    redirect_std_streams(
        suppress_patterns=build_stream_suppression_patterns(args.log_level),
    )

    try:
        importlib.import_module(COMMAND_MODULES[args.command]).main(args)
    except Exception:
        logging.exception("Rhythm bench %s failed", args.command)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
