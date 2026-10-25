"""Run an exported separator ONNX core with Python host stages.

This is a manual validation entry point. The lab's regular CLI still uses
the torch model; this script is for checking that an exported ONNX core can run
through onnxruntime before the artifact is consumed elsewhere.
"""

import argparse
from pathlib import Path

from musetric_toolkit.separate_audio.mel_band_roformer_onnx_separator import (
    MelBandRoformerOnnxSeparator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an exported MelBand ONNX core through Python host stages."
    )
    parser.add_argument("--model", required=True, type=Path, help="ONNX core path.")
    parser.add_argument(
        "--config", required=True, type=Path, help="Model YAML config path."
    )
    parser.add_argument("--source", required=True, type=Path, help="Input audio path.")
    parser.add_argument(
        "--target-output",
        required=True,
        type=Path,
        help="Output path for the model target source.",
    )
    parser.add_argument(
        "--residual-output",
        required=True,
        type=Path,
        help="Output path for the residual source.",
    )
    parser.add_argument("--sample-rate", default=44100, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.target_output.parent.mkdir(parents=True, exist_ok=True)
    args.residual_output.parent.mkdir(parents=True, exist_ok=True)

    separator = MelBandRoformerOnnxSeparator(
        model_onnx_path=args.model,
        model_config_path=args.config,
        sample_rate=args.sample_rate,
    )
    separator.separate_audio(
        source_path=str(args.source),
        target_path=str(args.target_output),
        residual_path=str(args.residual_output),
    )


if __name__ == "__main__":
    main()
