# ruff: noqa: T201

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from export_chordnet import (
    EXPECTED_N_BINS,
    EXPECTED_NUM_CHORDS,
    EXPECTED_SEQ_LEN,
    INPUT_NAME,
    OUTPUT_NAME,
    build_pipeline,
)

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

WINDOW_COUNT = 3
RANDOM_SEED = 20260715
MAX_ABSOLUTE_ERROR_TOLERANCE = 1e-4


def _validate_onnx_contract(session: ort.InferenceSession) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        message = (
            "ChordNet ONNX must expose exactly one input and output, got "
            f"{len(inputs)} input(s) and {len(outputs)} output(s)"
        )
        raise ValueError(message)

    input_metadata = inputs[0]
    output_metadata = outputs[0]
    if input_metadata.name != INPUT_NAME or output_metadata.name != OUTPUT_NAME:
        message = (
            "ChordNet ONNX I/O names do not match the export contract: "
            f"{input_metadata.name!r} -> {output_metadata.name!r}"
        )
        raise ValueError(message)


def _random_features() -> np.ndarray:
    generator = np.random.default_rng(RANDOM_SEED)
    return generator.standard_normal(
        (WINDOW_COUNT, EXPECTED_SEQ_LEN, EXPECTED_N_BINS),
        dtype=np.float32,
    )


def _validate_logits(
    reference_logits: np.ndarray,
    onnx_logits: np.ndarray,
) -> tuple[float, float]:
    expected_shape = (WINDOW_COUNT, EXPECTED_SEQ_LEN, EXPECTED_NUM_CHORDS)
    if reference_logits.shape != expected_shape:
        raise ValueError(
            "Torch ChordNet output shape mismatch: "
            f"{reference_logits.shape} (expected {expected_shape})"
        )
    if onnx_logits.shape != expected_shape:
        raise ValueError(
            "ONNX ChordNet output shape mismatch: "
            f"{onnx_logits.shape} (expected {expected_shape})"
        )
    if not np.isfinite(onnx_logits).all():
        raise ValueError("ONNX ChordNet output contains non-finite values")

    absolute_error = np.abs(reference_logits - onnx_logits)
    max_absolute_error = float(absolute_error.max())
    mean_absolute_error = float(absolute_error.mean())
    if max_absolute_error > MAX_ABSOLUTE_ERROR_TOLERANCE:
        message = (
            "ChordNet ONNX differs from Torch by "
            f"{max_absolute_error:.8g}; tolerance is "
            f"{MAX_ABSOLUTE_ERROR_TOLERANCE:.8g}"
        )
        raise ValueError(message)
    return max_absolute_error, mean_absolute_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="path to the exported chordnet.onnx classifier",
    )
    parser.add_argument(
        "--models-path",
        required=True,
        help="cache dir containing (or receiving) the ChordMini checkpoint",
    )
    args = parser.parse_args()

    onnx_path: Path = args.onnx
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ChordNet ONNX file does not exist: {onnx_path}")

    pipeline, seq_len, n_bins, num_chords = build_pipeline(args.models_path)
    expected_contract = (EXPECTED_SEQ_LEN, EXPECTED_N_BINS, EXPECTED_NUM_CHORDS)
    actual_contract = (seq_len, n_bins, num_chords)
    if actual_contract != expected_contract:
        raise ValueError(
            "Torch ChordNet contract mismatch: "
            f"{actual_contract} (expected {expected_contract})"
        )

    features = _random_features()
    with torch.inference_mode():
        reference_logits = pipeline(torch.from_numpy(features)).numpy()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    _validate_onnx_contract(session)
    onnx_logits = session.run([OUTPUT_NAME], {INPUT_NAME: features})[0]
    max_absolute_error, mean_absolute_error = _validate_logits(
        reference_logits,
        onnx_logits,
    )

    print(f"onnx={onnx_path}")
    print(f"seed={RANDOM_SEED} windows={WINDOW_COUNT}")
    print(f"shape={tuple(onnx_logits.shape)}")
    print(f"max_abs_error={max_absolute_error:.8g}")
    print(f"mean_abs_error={mean_absolute_error:.8g}")
    print(f"tolerance={MAX_ABSOLUTE_ERROR_TOLERANCE:.8g}")


if __name__ == "__main__":
    main()
