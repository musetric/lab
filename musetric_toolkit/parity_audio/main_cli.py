import argparse
import logging
import os
import sys
import warnings

from musetric_toolkit.common.logger import redirect_std_streams, setup_logging
from musetric_toolkit.common.paths import default_models_path

STEPS = ["vocals", "rhythm", "key", "chords", "transcribe"]


def configure_warning_filters(log_level: str) -> None:
    if log_level == "debug":
        return
    warnings.filterwarnings("ignore", category=UserWarning)


def apply_models_path(models_path: str) -> None:
    os.environ["HF_HOME"] = models_path
    os.environ["HF_HUB_CACHE"] = models_path
    os.environ["HUGGINGFACE_HUB_CACHE"] = models_path
    os.environ["TORCH_HOME"] = models_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Write the reference of one parity case: the step's stage boundaries "
            "from the original model through the product's processing, and our "
            "ONNX export on the CPU"
        ),
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=STEPS,
        help="Product step to follow",
    )
    parser.add_argument(
        "--audio-path",
        required=True,
        help="Audio of the case",
    )
    parser.add_argument(
        "--onnx",
        required=True,
        help="The product's pinned ONNX model of the step",
    )
    parser.add_argument(
        "--case-path",
        required=True,
        help="Directory to write the tensors and manifest.json into",
    )
    parser.add_argument(
        "--models-path",
        default=default_models_path(),
        help="Directory for the downloaded original checkpoints",
    )
    parser.add_argument(
        "--language",
        help="Language code of the case, such as en; the transcribe step needs it",
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
    redirect_std_streams()

    try:
        from musetric_toolkit.parity_audio.main import main  # noqa: PLC0415

        main(args)
    except Exception:
        logging.exception("Parity reference failed")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
