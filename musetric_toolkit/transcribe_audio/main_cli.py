import argparse
import logging
import os
import re
import sys
import warnings

from musetric_toolkit.common.logger import redirect_std_streams, setup_logging
from musetric_toolkit.common.paths import default_models_path


def build_stream_suppression_patterns(
    log_level: str,
) -> list[re.Pattern[str]]:
    if log_level == "debug":
        return []
    return [
        re.compile(r"\bwhisperx\.[\w\.]+ - INFO - "),
        re.compile(r"^Model was trained with "),
    ]


_AUDIO_SHORT_WARNING = re.compile(r"Audio is shorter than 30s", re.IGNORECASE)
_ALIGNMENT_BACKTRACK_WARNING = re.compile(
    r"Failed to align segment .*backtrack failed, resorting to original",
    re.IGNORECASE,
)


class SuppressMessageFilter(logging.Filter):
    def __init__(self, patterns: list[re.Pattern[str]]) -> None:
        super().__init__()
        self._patterns = patterns

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(pattern.search(message) for pattern in self._patterns)


def add_whisperx_filters(log_level: str) -> None:
    if log_level == "debug":
        return
    suppress_filter = SuppressMessageFilter(
        [_AUDIO_SHORT_WARNING, _ALIGNMENT_BACKTRACK_WARNING]
    )
    for logger_name in ("whisperx", "whisperx.asr", "whisperx.alignment"):
        logger = logging.getLogger(logger_name)
        has_filter = any(
            isinstance(existing, SuppressMessageFilter) for existing in logger.filters
        )
        if not has_filter:
            logger.addFilter(suppress_filter)


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
    warnings.filterwarnings(
        "ignore",
        message=r"\s*torchcodec is not installed correctly.*",
        category=UserWarning,
    )


def configure_third_party_logging(log_level: str) -> None:
    if log_level == "debug":
        return
    target_level = logging.ERROR if log_level == "error" else logging.WARNING
    prefixes = (
        "huggingface_hub",
        "lightning",
        "pytorch_lightning",
        "pyannote",
        "pyannote.audio",
        "speechbrain",
        "torchaudio",
        "whisperx",
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


def apply_models_path(models_path: str) -> None:
    os.environ["HF_HOME"] = models_path
    os.environ["HF_HUB_CACHE"] = models_path
    os.environ["HUGGINGFACE_HUB_CACHE"] = models_path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Transcribe vocals with WhisperX")
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
    setup_cli_logging(args.log_level)
    redirect_std_streams(
        suppress_patterns=build_stream_suppression_patterns(args.log_level),
    )
    add_whisperx_filters(args.log_level)

    try:
        from musetric_toolkit.transcribe_audio.main import main  # noqa: PLC0415

        main(args)
    except Exception:
        logging.exception("Audio transcription failed")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
