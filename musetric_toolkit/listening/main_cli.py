import argparse
import importlib
import logging
from pathlib import Path

SKIP_GENRES = ["Old-Time / Historic", "Spoken"]
SEARCHED_WINDOWS = [0, 4, 8, 11]
OWN_REPEATS = "detector_quiet=2,detector_loud=1,random=1,reverse_words=1"
EARLIER_REPEATS = "no_voice=3,voiced=2"
GROUPS = [
    "random",
    "detector_quiet",
    "detector_loud",
    "reverse_teacher",
    "reverse_words",
]


def lazy(module: str, function: str):
    def run(args) -> None:
        getattr(importlib.import_module(module), function)(args)

    return run


def counts_by_name(text: str) -> dict[str, int]:
    parts = (part.partition("=") for part in text.split(","))
    return {name: int(count) for name, _, count in parts}


def system_spec(text: str) -> tuple[str, list[Path]]:
    name, _, folders = text.partition("=")
    if not name:
        raise argparse.ArgumentTypeError(f"{text} does not start with a name")
    return name, [Path(folder) for folder in folders.split(",") if folder]


def add_labels(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Directory of the <id>.mix.npy and <id>.vocals.npy of the corpus",
    )


def add_corpus(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--corpus",
        required=required,
        type=Path,
        help="Directory of the corpus with fma.json",
    )


def add_voices(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--voices",
        required=required,
        nargs="+",
        default=[],
        type=Path,
        help="Manifests of the split: their role says which tracks may be used",
    )


def add_earlier(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--earlier-round",
        nargs="*",
        default=[],
        type=Path,
        help="Folders of earlier rounds: what the round takes from them or avoids",
    )


def add_windows(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--window-s", type=int, default=8, help="Seconds of a searched window"
    )
    parser.add_argument(
        "--hop-s", type=int, default=2, help="Seconds between searched windows"
    )


def add_searched(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fixed-windows",
        nargs="+",
        type=int,
        default=SEARCHED_WINDOWS,
        help="Indices of the windows searched in every track",
    )
    parser.add_argument(
        "--quiet-db",
        type=float,
        default=-20.0,
        help="Vocals under this level count as quiet",
    )
    parser.add_argument(
        "--min-mix-rms",
        type=float,
        default=800.0,
        help="Tracks with a quieter mix are left out, in int16 steps",
    )
    parser.add_argument(
        "--limit", type=int, help="Handle at most this many items in this run"
    )


def add_whisper(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=16, help="Whisper batch size")


def add_scan(commands, common) -> None:
    parser = commands.add_parser(
        "scan", parents=[common], help="levels of the vocals per window of each track"
    )
    add_labels(parser)
    add_corpus(parser)
    add_voices(parser)
    add_earlier(parser)
    add_windows(parser)
    parser.add_argument("--out", required=True, type=Path, help="JSONL of the levels")
    parser.add_argument(
        "--skip-genres",
        nargs="*",
        default=SKIP_GENRES,
        help="Genres left out of the search",
    )
    parser.add_argument(
        "--stride", type=int, default=16, help="Keep every n-th sample for the levels"
    )
    parser.add_argument("--workers", type=int, default=6, help="Reading processes")
    parser.add_argument(
        "--limit", type=int, help="Handle at most this many tracks in this run"
    )
    parser.set_defaults(
        run=lazy("musetric_toolkit.listening.levels_scan", "scan_command")
    )


def add_words(commands, common) -> None:
    parser = commands.add_parser(
        "words", parents=[common], help="what Whisper hears in the searched windows"
    )
    add_labels(parser)
    add_windows(parser)
    add_searched(parser)
    add_whisper(parser)
    parser.add_argument(
        "--levels", required=True, type=Path, help="JSONL written by scan"
    )
    parser.add_argument("--out", required=True, type=Path, help="JSONL of the words")
    parser.add_argument(
        "--gate-db",
        type=float,
        default=-14.0,
        help="Vocals at or over this level are heard as vocals",
    )
    parser.add_argument(
        "--kind",
        choices=["vocals", "instrumental"],
        help="Only windows of this kind",
    )
    parser.add_argument("--seed", type=int, default=20261106, help="Order of the work")
    parser.set_defaults(
        run=lazy("musetric_toolkit.listening.words_scan", "words_command")
    )


def add_teacher(commands, common) -> None:
    parser = commands.add_parser(
        "teacher", parents=[common], help="the vocals teacher on the candidate tracks"
    )
    add_labels(parser)
    add_windows(parser)
    add_searched(parser)
    parser.add_argument(
        "--teacher",
        required=True,
        type=Path,
        help="Directory with the teacher's config.yaml and model.ckpt",
    )
    parser.add_argument(
        "--levels", required=True, type=Path, help="JSONL written by scan"
    )
    parser.add_argument(
        "--words", required=True, type=Path, help="JSONL written by words"
    )
    parser.add_argument(
        "--signs",
        type=Path,
        help="JSONL with vocals_to_mix and vocals_gap per window of mined tracks",
    )
    parser.add_argument("--out", required=True, type=Path, help="JSONL of the windows")
    parser.add_argument(
        "--max-words",
        type=int,
        default=2,
        help="A detector window has at most this many words",
    )
    parser.add_argument(
        "--artist-pool",
        type=int,
        default=12,
        help="Detector tracks kept per artist",
    )
    parser.add_argument(
        "--reverse-pool", type=int, default=120, help="Reverse tracks kept in all"
    )
    parser.add_argument(
        "--reverse-artist-pool",
        type=int,
        default=3,
        help="Reverse tracks kept per artist",
    )
    parser.add_argument(
        "--reverse-min-words",
        type=int,
        default=3,
        help="A reverse window has at least this many words of letters",
    )
    parser.add_argument(
        "--quiet-extra",
        type=int,
        default=1000,
        help="Further tracks with a quiet window",
    )
    parser.add_argument(
        "--teacher-voice-db",
        type=float,
        default=-14.0,
        help="The teacher hears voice at or over this level",
    )
    parser.add_argument("--seed", type=int, default=20261107, help="Choice of tracks")
    parser.set_defaults(
        run=lazy("musetric_toolkit.listening.teacher_scan", "teacher_command")
    )


def add_select_quotas(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--random-clips", type=int, default=20, help="Clips of the random group"
    )
    parser.add_argument(
        "--detector-quiet-clips",
        type=int,
        default=30,
        help="Clips where the detector fires and the teacher is quiet",
    )
    parser.add_argument(
        "--detector-loud-clips",
        type=int,
        default=30,
        help="Clips where the detector fires and the teacher is loud",
    )
    parser.add_argument(
        "--reverse-clips",
        type=int,
        default=20,
        help="Clips of both reverse groups together",
    )
    parser.add_argument(
        "--reverse-teacher-clips",
        type=int,
        default=10,
        help="At most this many reverse clips found by the teacher",
    )
    parser.add_argument(
        "--own-repeats",
        type=counts_by_name,
        default=counts_by_name(OWN_REPEATS),
        help="Blind copies per group as group=count,group=count",
    )
    parser.add_argument(
        "--earlier-repeats",
        type=counts_by_name,
        default=counts_by_name(EARLIER_REPEATS),
        help="Clips repeated from --repeats-from as no_voice=count,voiced=count",
    )
    parser.add_argument(
        "--repeat-distance",
        type=int,
        default=30,
        help="A blind copy stands at least this many clips from its original",
    )


def add_select_limits(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gate-db",
        type=float,
        default=-14.0,
        help="A random window has vocals at or over this level",
    )
    parser.add_argument(
        "--teacher-quiet-db",
        type=float,
        default=-20.0,
        help="The teacher is quiet under this level",
    )
    parser.add_argument(
        "--teacher-voice-db",
        type=float,
        default=-14.0,
        help="The teacher hears voice at or over this level",
    )
    parser.add_argument(
        "--min-mix-rms",
        type=float,
        default=800.0,
        help="Tracks with a quieter mix are left out, in int16 steps",
    )
    parser.add_argument(
        "--min-window-db",
        type=float,
        default=-45.0,
        help="A window of the mix under this level is silence",
    )
    parser.add_argument(
        "--genre-share",
        type=float,
        default=0.4,
        help="At most this share of a group from one genre",
    )


def add_select(commands, common) -> None:
    parser = commands.add_parser(
        "select", parents=[common], help="choose the clips of a round"
    )
    add_labels(parser)
    add_windows(parser)
    add_select_quotas(parser)
    add_select_limits(parser)
    parser.add_argument(
        "--levels", required=True, type=Path, help="JSONL written by scan"
    )
    parser.add_argument("--windows", type=Path, help="JSONL written by teacher")
    parser.add_argument(
        "--rule-features",
        type=Path,
        help="features.json of the round the detector rule is frozen on",
    )
    parser.add_argument(
        "--repeats-from",
        type=Path,
        help="Folder of the earlier round to repeat clips from",
    )
    parser.add_argument("--out", required=True, type=Path, help="selection.json")
    parser.add_argument("--seed", type=int, default=20261108, help="Choice of clips")
    parser.set_defaults(
        run=lazy("musetric_toolkit.listening.selection", "select_command")
    )


def add_build(commands, common) -> None:
    parser = commands.add_parser(
        "build", parents=[common], help="clips, stems, pictures, manifest and key"
    )
    add_labels(parser)
    add_corpus(parser)
    parser.add_argument(
        "--selection", required=True, type=Path, help="selection.json of the round"
    )
    parser.add_argument(
        "--system",
        required=True,
        action="append",
        type=system_spec,
        help=(
            "A system compared in the round, given twice: name=release folder of a "
            "LeadBackingRoformer, or name=vocals teacher folder,KARA 2 folder. "
            "The names go to key.json"
        ),
    )
    parser.add_argument(
        "--training-data",
        type=Path,
        help="training-data.csv of a release, with the FMA page of a track",
    )
    parser.add_argument("--out", required=True, type=Path, help="Folder of the round")
    parser.add_argument(
        "--ask",
        choices=["present"],
        help="present makes a round of ticks about the song, without comparisons",
    )
    parser.add_argument(
        "--default-stem", default="lead_a", help="The stem a clip opens on"
    )
    parser.add_argument(
        "--id-width", type=int, default=2, help="Digits of a clip's number"
    )
    parser.add_argument(
        "--clip-seconds", type=float, default=8.0, help="Length of a clip"
    )
    parser.add_argument("--seed", type=int, default=20261104, help="Sides A and B")
    parser.add_argument("--workers", type=int, default=6, help="Picture processes")
    parser.set_defaults(
        run=lazy("musetric_toolkit.listening.round_build", "build_command")
    )


def add_check_round(commands, common) -> None:
    parser = commands.add_parser(
        "check-round", parents=[common], help="check a built round before serving it"
    )
    add_corpus(parser, required=False)
    add_voices(parser, required=False)
    add_earlier(parser)
    parser.add_argument("--round", required=True, type=Path, help="Folder of the round")
    parser.add_argument(
        "--min-sum-snr-db",
        type=float,
        default=40.0,
        help="Stems must sum to the mix at least this well",
    )
    parser.add_argument(
        "--min-mix-db", type=float, default=-45.0, help="The mix of a clip is louder"
    )
    parser.add_argument(
        "--min-repeat-distance",
        type=int,
        default=25,
        help="A blind copy stands at least this many clips from its original",
    )
    parser.add_argument(
        "--min-repeat-snr-db",
        type=float,
        default=60.0,
        help="The mix repeated from an earlier round matches at least this well",
    )
    parser.add_argument(
        "--exact-levels",
        nargs="*",
        default=["teacher"],
        help="Systems whose selection level comes from the system itself",
    )
    parser.add_argument(
        "--exact-level-db",
        type=float,
        default=1.0,
        help="The largest gap for --exact-levels",
    )
    parser.add_argument(
        "--drift-db",
        type=float,
        default=3.0,
        help="Show selection levels of stand-ins further than this from the clip",
    )
    parser.add_argument(
        "--gate-db", type=float, default=-14.0, help="Counted as loud, at or over"
    )
    parser.add_argument(
        "--quiet-db", type=float, default=-20.0, help="Counted as quiet, under"
    )
    parser.set_defaults(
        run=lazy("musetric_toolkit.listening.round_check", "check_round_command")
    )


def add_copy_round(commands, common) -> None:
    parser = commands.add_parser(
        "copy-round",
        parents=[common],
        help="a small copy of a round without its key or labels, for browser checks",
    )
    parser.add_argument("--round", required=True, type=Path, help="Folder of the round")
    parser.add_argument(
        "--out", required=True, type=Path, help="New folder of the check round"
    )
    parser.add_argument("--clips", type=int, default=2, help="Clips to copy")
    parser.set_defaults(
        run=lazy("musetric_toolkit.listening.round_copy", "copy_round_command")
    )


def add_serve(commands, common) -> None:
    parser = commands.add_parser(
        "serve", parents=[common], help="the labelling stand, on this machine only"
    )
    parser.add_argument("--round", required=True, type=Path, help="Folder of the round")
    parser.add_argument("--port", type=int, default=8765, help="Port of the stand")
    parser.add_argument("--open", action="store_true", help="Open the page")
    parser.set_defaults(
        run=lazy("musetric_toolkit.listening.stand_server", "serve_command")
    )


def add_features(commands, common) -> None:
    parser = commands.add_parser(
        "features", parents=[common], help="what Whisper and DSP find in a closed round"
    )
    add_corpus(parser)
    add_whisper(parser)
    parser.add_argument("--round", required=True, type=Path, help="Folder of the round")
    parser.add_argument(
        "--echonest", type=Path, help="echonest.csv of the FMA metadata"
    )
    parser.set_defaults(
        run=lazy("musetric_toolkit.listening.round_features", "features_command")
    )


def add_report_lists(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--groups",
        nargs="*",
        default=GROUPS,
        help="Selection groups in the order to show; others follow by name",
    )
    parser.add_argument(
        "--random-group", default="random", help="The group drawn at random"
    )
    parser.add_argument(
        "--detector-groups",
        nargs="*",
        default=["detector_quiet", "detector_loud"],
        help="The groups chosen by the frozen rule",
    )
    parser.add_argument(
        "--reverse-groups",
        nargs="*",
        default=["reverse_teacher", "reverse_words"],
        help="The groups searched for lost voice",
    )
    parser.add_argument(
        "--one-silent-groups",
        nargs="*",
        default=["detector_quiet", "reverse_teacher"],
        help="The groups where exactly one system is silent",
    )
    parser.add_argument(
        "--hedged",
        nargs="*",
        default=[],
        help="Clips whose no voice answer the listener doubted",
    )
    parser.add_argument(
        "--envelope",
        nargs="*",
        default=[],
        help="Clips to show the vocals level per stretch",
    )
    parser.add_argument(
        "--show-clips",
        nargs="*",
        default=[],
        help="Clips to unblind: systems, source and comment",
    )


def add_report(commands, common) -> None:
    parser = commands.add_parser(
        "report", parents=[common], help="the report of a closed round"
    )
    add_earlier(parser)
    add_report_lists(parser)
    parser.add_argument("--round", required=True, type=Path, help="Folder of the round")
    parser.add_argument("--out", type=Path, help="File for the report")
    parser.add_argument(
        "--primary",
        help="The system named first in the report; by default the first by name",
    )
    parser.add_argument(
        "--rule-features",
        type=Path,
        help="features.json of the round the rule is frozen on; a ticks round shows "
        "the rule only with it",
    )
    parser.add_argument(
        "--windows", type=Path, help="JSONL written by teacher, for the supply"
    )
    parser.add_argument(
        "--limits",
        nargs="+",
        type=float,
        default=[-15.0, -20.0, -25.0],
        help="A stem at or over the level is present, in dB to the mix",
    )
    parser.add_argument(
        "--quiet-db", type=float, default=-20.0, help="Counted as quiet, under"
    )
    parser.add_argument(
        "--voice-db", type=float, default=-14.0, help="Counted as voice, at or over"
    )
    parser.add_argument(
        "--envelope-seconds", type=float, default=1.0, help="Stretch of --envelope"
    )
    parser.add_argument(
        "--score-thresholds",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 2.0],
        help="Precision of the rule from these scores",
    )
    parser.add_argument(
        "--supply-thresholds",
        nargs="+",
        type=float,
        default=[2.0, 1.0, 0.49, 0.0],
        help="Count the candidates from these scores",
    )
    parser.add_argument(
        "--bootstraps", type=int, default=10000, help="Resamples of an interval"
    )
    parser.add_argument("--seed", type=int, default=20261109, help="Resampling seed")
    parser.set_defaults(run=lazy("musetric_toolkit.listening.report", "report_command"))


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warn", "error"],
        help="Set the logging level",
    )
    parser = argparse.ArgumentParser(
        description="Listening rounds: search, build, serve and report"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    add_scan(commands, common)
    add_words(commands, common)
    add_teacher(commands, common)
    add_select(commands, common)
    add_build(commands, common)
    add_check_round(commands, common)
    add_copy_round(commands, common)
    add_serve(commands, common)
    add_features(commands, common)
    add_report(commands, common)
    return parser


def main_cli() -> None:
    args = build_parser().parse_args()
    level = {"warn": "WARNING"}.get(args.log_level, args.log_level.upper())
    logging.basicConfig(level=level)
    args.run(args)
