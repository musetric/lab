import argparse
import logging
import os
import sys
import warnings

from musetric_toolkit.common.logger import redirect_std_streams, setup_logging
from musetric_toolkit.common.paths import default_models_path


def configure_warning_filters(log_level: str) -> None:
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    if log_level == "debug":
        return
    warnings.filterwarnings(
        "ignore",
        message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
        category=UserWarning,
    )


def configure_third_party_logging(log_level: str) -> None:
    if log_level == "debug":
        return
    target_level = logging.ERROR if log_level == "error" else logging.WARNING
    for logger_name in ("huggingface_hub", "transformers"):
        logging.getLogger(logger_name).setLevel(target_level)


def apply_models_path(models_path: str) -> None:
    os.environ["HF_HOME"] = models_path
    os.environ["HF_HUB_CACHE"] = models_path
    os.environ["HUGGINGFACE_HUB_CACHE"] = models_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Transcribe vocals with Whisper as the app does"
    )
    parser.add_argument("--audio-path", required=True, help="Path to vocal audio file")
    parser.add_argument(
        "--backing-path",
        help="Path to the backing vocals, added to the audio at half level",
    )
    parser.add_argument(
        "--result-path",
        required=True,
        help="Path to write transcription JSON result",
    )
    parser.add_argument(
        "--models-path",
        default=default_models_path(),
        help="Set HF_HOME, HF_HUB_CACHE, and HUGGINGFACE_HUB_CACHE to this path",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warn", "error"],
        help="Set the logging level",
    )
    return parser.parse_args()


def main_cli() -> None:
    args = parse_arguments()
    apply_models_path(args.models_path)
    configure_warning_filters(args.log_level)
    setup_logging(args.log_level)
    configure_third_party_logging(args.log_level)
    redirect_std_streams()

    try:
        from musetric_toolkit.transcribe_audio.main import main  # noqa: PLC0415

        main(args)
    except Exception:
        logging.exception("Audio transcription failed")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
