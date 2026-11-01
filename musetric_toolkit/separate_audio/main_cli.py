import argparse
import logging
import os
import sys
import warnings

from musetric_toolkit.common.logger import redirect_std_streams, setup_logging
from musetric_toolkit.common.paths import default_models_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Separate audio into lead vocals, backing vocals, and instrumental parts"
        )
    )
    parser.add_argument(
        "--source-path", required=True, help="Path to source audio file"
    )
    parser.add_argument(
        "--lead-path", required=True, help="Path for lead vocal output file"
    )
    parser.add_argument(
        "--backing-path", required=True, help="Path for backing vocal output file"
    )
    parser.add_argument(
        "--instrumental-path",
        required=True,
        help="Path for instrumental output file",
    )
    parser.add_argument(
        "--sample-rate",
        required=True,
        type=int,
        help="Sample rate of the written stems",
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


def apply_models_path(models_path: str) -> None:
    os.environ["HF_HOME"] = models_path
    os.environ["HF_HUB_CACHE"] = models_path
    os.environ["HUGGINGFACE_HUB_CACHE"] = models_path


def configure_warning_filters(log_level: str) -> None:
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    if log_level == "debug":
        return
    warnings.filterwarnings(
        "ignore",
        message=r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Module 'speechbrain\.pretrained' was deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
        category=UserWarning,
    )


def _tune_logger(logger: logging.Logger, target_level: int) -> None:
    logger.setLevel(target_level)
    logger.propagate = True
    if logger.handlers:
        logger.handlers.clear()


def _matches_prefix(logger_name: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        logger_name == prefix or logger_name.startswith(f"{prefix}.")
        for prefix in prefixes
    )


def configure_third_party_logging(log_level: str) -> None:
    if log_level == "debug":
        return
    target_level = logging.ERROR if log_level == "error" else logging.WARNING
    prefixes = (
        "huggingface_hub",
        "torchaudio",
    )

    for logger_name in prefixes:
        _tune_logger(logging.getLogger(logger_name), target_level)

    for logger_name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and _matches_prefix(
            logger_name, prefixes
        ):
            _tune_logger(logger, target_level)


def setup_cli_logging(log_level: str) -> None:
    configure_warning_filters(log_level)
    setup_logging(log_level)
    configure_third_party_logging(log_level)


def main_cli() -> None:
    args = parse_arguments()
    apply_models_path(args.models_path)
    setup_cli_logging(args.log_level)
    redirect_std_streams()

    try:
        from musetric_toolkit.separate_audio.main import main  # noqa: PLC0415

        main(args)
    except Exception:
        logging.exception("Audio separation failed")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
