"""Export the ChordMini ChordNet classifier as a feature-to-logits ONNX graph.

The graph intentionally covers only the neural classifier boundary required by
the WebGPU CQT pipeline:

    log-CQT features [W, 108, 144]
      -> (features - mean) / (std + 1e-8)
      -> ChordNet
      -> logits [W, 108, 170]

It does not contain CQT extraction, frame padding/windowing, temporal smoothing,
or argmax; those stages stay in the WebGPU runtime.

    uv run --group export python scripts/onnx/chordmini/export_chordnet.py \
      --out <export-dir> --models-path <checkpoint-cache-dir>
"""

# ruff: noqa: T201

import argparse
import contextlib
import json
import sys
from pathlib import Path

import onnx
import torch
from torch import nn

from musetric_toolkit.chords_audio.chordmini.models import ChordNet, load_model
from musetric_toolkit.chords_audio.chordmini.utils import (
    HParams,
    get_config_value,
    idx2voca_chord,
)
from musetric_toolkit.chords_audio.chordmini_checkpoint import ensure_checkpoint
from musetric_toolkit.chords_audio.chordmini_runner import CONFIG_PATH

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

INPUT_NAME = "features"
OUTPUT_NAME = "logits"
ONNX_FILENAME = "chordnet.onnx"
CONFIG_FILENAME = "config.json"
OPSET = 17
NORMALIZATION_EPSILON = 1e-8
EXPECTED_SEQ_LEN = 108
EXPECTED_N_BINS = 144
EXPECTED_NUM_CHORDS = 170
SAMPLE_RATE = 22050
HOP_LENGTH = 2048
FMIN = 32.70319566257483
BINS_PER_OCTAVE = 24
SMOOTHING_KERNEL = 9


class ChordNetPipeline(nn.Module):
    """Normalize CQT features and return unsmoothed ChordNet logits."""

    def __init__(self, model: ChordNet, mean: float, std: float) -> None:
        super().__init__()
        self.model = model
        self.mean = float(mean)
        self.std = float(std)
        self.register_buffer(
            "normalization_mean", torch.tensor(self.mean, dtype=torch.float32)
        )
        self.register_buffer(
            "normalization_std", torch.tensor(self.std, dtype=torch.float32)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits for unnormalized log-CQT feature windows."""
        normalized_features = (features - self.normalization_mean) / (
            self.normalization_std + NORMALIZATION_EPSILON
        )
        logits, _ = self.model(normalized_features)
        return logits


def _contract_value(
    config: HParams,
    section: str,
    key: str,
    default: int,
) -> int:
    """Read an integer config value and enforce the fixed ONNX contract."""
    return int(get_config_value(config, section, key, default))


def _validate_contract(seq_len: int, n_bins: int, num_chords: int) -> None:
    """Reject checkpoints/configurations that do not match the WebGPU contract."""
    expected_values = {
        "seq_len": (seq_len, EXPECTED_SEQ_LEN),
        "n_bins": (n_bins, EXPECTED_N_BINS),
        "num_chords": (num_chords, EXPECTED_NUM_CHORDS),
    }
    mismatches = [
        f"{name}={actual} (expected {expected})"
        for name, (actual, expected) in expected_values.items()
        if actual != expected
    ]
    if mismatches:
        message = ", ".join(mismatches)
        raise ValueError(f"ChordNet ONNX contract mismatch: {message}")


def build_pipeline(models_path: str) -> tuple[ChordNetPipeline, int, int, int]:
    """Load the checkpoint and return its classifier plus contract dimensions."""
    config = HParams.load(str(CONFIG_PATH))
    checkpoint = ensure_checkpoint(models_path)
    model, mean, std = load_model(
        str(checkpoint), "ChordNet", config, torch.device("cpu")
    )
    model.eval()

    seq_len = _contract_value(config, "model", "seq_len", EXPECTED_SEQ_LEN)
    n_bins = _contract_value(config, "feature", "n_bins", EXPECTED_N_BINS)
    num_chords = int(model.n_classes)
    _validate_contract(seq_len, n_bins, num_chords)

    print(f"loaded ChordNet: mean={mean:.6f} std={std:.6f}")
    return ChordNetPipeline(model, mean, std).eval(), seq_len, n_bins, num_chords


def _set_shape_dimensions(
    value_info: onnx.ValueInfoProto,
    dimensions: tuple[str | int, ...],
) -> None:
    """Set ONNX tensor shape metadata without changing graph operations."""
    tensor_shape = value_info.type.tensor_type.shape
    if len(tensor_shape.dim) != len(dimensions):
        message = (
            f"{value_info.name} has {len(tensor_shape.dim)} dimensions, "
            f"expected {len(dimensions)}"
        )
        raise ValueError(message)

    for dimension, size in zip(tensor_shape.dim, dimensions, strict=True):
        dimension.ClearField("dim_value")
        dimension.ClearField("dim_param")
        if isinstance(size, int):
            dimension.dim_value = size
        else:
            dimension.dim_param = size


def _write_contract_shape_metadata(
    onnx_path: Path,
    seq_len: int,
    n_bins: int,
    num_chords: int,
) -> None:
    """Make the dynamic-window/static-frame ONNX interface explicit."""
    model = onnx.load(onnx_path)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("ChordNet ONNX graph must have exactly one input and output")

    input_value = model.graph.input[0]
    output_value = model.graph.output[0]
    if input_value.name != INPUT_NAME or output_value.name != OUTPUT_NAME:
        message = (
            "ChordNet ONNX graph did not preserve the requested I/O names: "
            f"{input_value.name!r} -> {output_value.name!r}"
        )
        raise ValueError(message)

    _set_shape_dimensions(input_value, ("windows", seq_len, n_bins))
    _set_shape_dimensions(output_value, ("windows", seq_len, num_chords))
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)


def _write_config(
    path: Path,
    pipeline: ChordNetPipeline,
    seq_len: int,
    n_bins: int,
    num_chords: int,
) -> None:
    """Write the feature-to-logits artifact contract next to the ONNX graph."""
    config_payload = {
        "modelType": "chordmini-chordnet-2e1d",
        "input": {
            "name": INPUT_NAME,
            "shape": ["windows", seq_len, n_bins],
            "dtype": "float32",
        },
        "output": {
            "name": OUTPUT_NAME,
            "shape": ["windows", seq_len, num_chords],
            "dtype": "float32",
        },
        "seqLen": seq_len,
        "sampleRate": SAMPLE_RATE,
        "hopLength": HOP_LENGTH,
        "frameDuration": HOP_LENGTH / SAMPLE_RATE,
        "fmin": FMIN,
        "nBins": n_bins,
        "binsPerOctave": BINS_PER_OCTAVE,
        "smoothingKernel": SMOOTHING_KERNEL,
        "numChords": num_chords,
        "mean": pipeline.mean,
        "std": pipeline.std,
        "normalizationEpsilon": NORMALIZATION_EPSILON,
        "normalizationInGraph": True,
        "chordVocab": [idx2voca_chord()[index] for index in range(num_chords)],
    }
    path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Export ChordNet and its machine-readable input/output contract."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="directory to write chordnet.onnx and config.json into",
    )
    parser.add_argument(
        "--models-path",
        required=True,
        help="cache dir for the ChordMini checkpoint download",
    )
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    pipeline, seq_len, n_bins, num_chords = build_pipeline(args.models_path)

    dummy_features = torch.zeros(1, seq_len, n_bins, dtype=torch.float32)
    onnx_path = out / ONNX_FILENAME
    torch.onnx.export(
        pipeline,
        (dummy_features,),
        str(onnx_path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "windows"}, OUTPUT_NAME: {0: "windows"}},
        opset_version=OPSET,
    )
    _write_contract_shape_metadata(onnx_path, seq_len, n_bins, num_chords)
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"exported {onnx_path} ({size_mb:.1f} MB)")

    config_path = out / CONFIG_FILENAME
    _write_config(
        config_path,
        pipeline,
        seq_len=seq_len,
        n_bins=n_bins,
        num_chords=num_chords,
    )
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
