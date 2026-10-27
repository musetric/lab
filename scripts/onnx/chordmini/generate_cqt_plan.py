# ruff: noqa: T201

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import textwrap
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import scipy
from librosa import filters
from librosa.core import constantq
from scipy.signal import firwin

FORMAT_VERSION = 1
HEADER_BYTE_LENGTH = 128
MAGIC = 0x5451434D
OUTPUT_LOG_MAGNITUDE = 1
SAMPLE_RATE = 22050.0
HOP_LENGTH = 2048
FMIN = 32.70319566257483
N_BINS = 144
BINS_PER_OCTAVE = 24
FILTER_SCALE = 1
SPARSITY = 0.01
FIR_TAP_COUNT = 255
FIR_CUTOFF = 0.48
FIR_KAISER_BETA = 12


@dataclass(frozen=True)
class Octave:
    index: int
    sample_rate: float
    hop_length: int
    fft_size: int
    bin_start: int
    bin_count: int


@dataclass(frozen=True)
class PlanData:
    early_downsample_count: int
    octaves: list[Octave]
    row_offsets: np.ndarray
    fft_bins: np.ndarray
    coefficients: np.ndarray
    bin_lengths: np.ndarray
    half_coefficients: np.ndarray


def _get_plan_data() -> PlanData:
    frequencies = librosa.cqt_frequencies(
        n_bins=N_BINS,
        fmin=FMIN,
        bins_per_octave=BINS_PER_OCTAVE,
        tuning=0,
    )
    alpha = filters._relative_bandwidth(freqs=frequencies)
    _lengths, filter_cutoff = filters.wavelet_lengths(
        freqs=frequencies,
        sr=SAMPLE_RATE,
        window="hann",
        filter_scale=FILTER_SCALE,
        gamma=0,
        alpha=alpha,
    )
    octave_count = int(np.ceil(N_BINS / BINS_PER_OCTAVE))
    early_downsample_count = constantq.__early_downsample_count(
        SAMPLE_RATE / 2,
        filter_cutoff,
        HOP_LENGTH,
        octave_count,
    )
    effective_sample_rate = SAMPLE_RATE / 2**early_downsample_count
    effective_hop = HOP_LENGTH // 2**early_downsample_count
    scaled_lengths, _ = filters.wavelet_lengths(
        freqs=frequencies,
        sr=effective_sample_rate,
        window="hann",
        filter_scale=FILTER_SCALE,
        gamma=0,
        alpha=alpha,
    )

    rows: list[tuple[np.ndarray, np.ndarray]] = [
        (
            np.empty(0, dtype="<u4"),
            np.empty(0, dtype="<c8"),
        )
        for _ in range(N_BINS)
    ]
    octaves: list[Octave] = []
    for octave_index in range(octave_count):
        if octave_index == 0:
            frequency_slice = slice(-BINS_PER_OCTAVE, None)
        else:
            frequency_slice = slice(
                -BINS_PER_OCTAVE * (octave_index + 1),
                -BINS_PER_OCTAVE * octave_index,
            )
        octave_frequencies = frequencies[frequency_slice]
        octave_alpha = alpha[frequency_slice]
        octave_sample_rate = effective_sample_rate / 2**octave_index
        octave_hop = effective_hop // 2**octave_index
        fft_basis, fft_size, _ = constantq.__vqt_filter_fft(
            octave_sample_rate,
            octave_frequencies,
            FILTER_SCALE,
            1,
            SPARSITY,
            window="hann",
            gamma=0,
            dtype=np.complex64,
            alpha=octave_alpha,
        )
        fft_basis = fft_basis.tocsr()
        fft_basis.data *= np.sqrt(effective_sample_rate / octave_sample_rate)
        bin_start = N_BINS - BINS_PER_OCTAVE * (octave_index + 1)
        for local_bin in range(BINS_PER_OCTAVE):
            global_bin = bin_start + local_bin
            row = fft_basis.getrow(local_bin)
            values = np.asarray(row.data, dtype=np.complex64)
            values /= np.sqrt(scaled_lengths[global_bin])
            rows[global_bin] = (
                np.asarray(row.indices, dtype="<u4"),
                np.asarray(values, dtype="<c8"),
            )
        octaves.append(
            Octave(
                index=octave_index,
                sample_rate=octave_sample_rate,
                hop_length=octave_hop,
                fft_size=int(fft_size),
                bin_start=bin_start,
                bin_count=BINS_PER_OCTAVE,
            )
        )

    row_offsets = np.zeros(N_BINS + 1, dtype="<u4")
    fft_bin_parts: list[np.ndarray] = []
    coefficient_parts: list[np.ndarray] = []
    for global_bin, (fft_bins, values) in enumerate(rows):
        row_offsets[global_bin + 1] = row_offsets[global_bin] + len(fft_bins)
        fft_bin_parts.append(fft_bins)
        complex_values = np.empty(len(values) * 2, dtype="<f4")
        complex_values[::2] = values.real
        complex_values[1::2] = values.imag
        coefficient_parts.append(complex_values)

    half_band = firwin(
        FIR_TAP_COUNT,
        FIR_CUTOFF,
        window=("kaiser", FIR_KAISER_BETA),
        scale=True,
        fs=2,
    )
    delay = (FIR_TAP_COUNT - 1) // 2
    return PlanData(
        early_downsample_count=early_downsample_count,
        octaves=octaves,
        row_offsets=row_offsets,
        fft_bins=np.concatenate(fft_bin_parts).astype("<u4", copy=False),
        coefficients=np.concatenate(coefficient_parts).astype("<f4", copy=False),
        bin_lengths=np.asarray(scaled_lengths, dtype="<f4"),
        half_coefficients=np.asarray(half_band[delay:], dtype="<f4"),
    )


def _align(payload: bytearray) -> None:
    while len(payload) % 4:
        payload.append(0)


def _append(payload: bytearray, value: bytes, *, align: bool = True) -> int:
    if align:
        _align(payload)
    offset = HEADER_BYTE_LENGTH + len(payload)
    payload.extend(value)
    return offset


def _write_octaves(octaves: list[Octave]) -> bytes:
    return b"".join(
        struct.pack(
            "<IdIIIII",
            octave.index,
            octave.sample_rate,
            octave.hop_length,
            octave.fft_size,
            octave.bin_start,
            octave.bin_count,
            0,
        )
        for octave in octaves
    )


def _create_artifact(plan: PlanData) -> tuple[bytes, str, str]:
    generator = (
        f"librosa=={librosa.__version__};numpy=={np.__version__};"
        f"scipy=={scipy.__version__};resampler=kaiser-lowpass-"
        f"{FIR_TAP_COUNT}-cutoff-{FIR_CUTOFF}-beta-{FIR_KAISER_BETA}"
    ).encode("ascii")
    payload = bytearray()
    generator_offset = _append(payload, generator, align=False)
    _align(payload)
    octaves_offset = _append(payload, _write_octaves(plan.octaves), align=False)
    row_offsets_offset = _append(payload, plan.row_offsets.tobytes(), align=True)
    fft_bins_offset = _append(payload, plan.fft_bins.tobytes(), align=True)
    coefficients_offset = _append(payload, plan.coefficients.tobytes(), align=True)
    bin_lengths_offset = _append(payload, plan.bin_lengths.tobytes(), align=True)
    downsample_offset = _append(payload, plan.half_coefficients.tobytes(), align=True)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    header = bytearray(HEADER_BYTE_LENGTH)
    struct.pack_into("<I", header, 0, MAGIC)
    struct.pack_into("<I", header, 4, FORMAT_VERSION)
    struct.pack_into("<I", header, 8, HEADER_BYTE_LENGTH)
    struct.pack_into("<I", header, 12, OUTPUT_LOG_MAGNITUDE)
    struct.pack_into("<d", header, 16, SAMPLE_RATE)
    struct.pack_into("<I", header, 24, HOP_LENGTH)
    struct.pack_into("<I", header, 28, N_BINS)
    struct.pack_into("<I", header, 32, BINS_PER_OCTAVE)
    struct.pack_into("<I", header, 36, plan.early_downsample_count)
    struct.pack_into("<d", header, 40, FMIN)
    struct.pack_into("<I", header, 48, len(plan.octaves))
    struct.pack_into("<I", header, 52, FIR_TAP_COUNT)
    struct.pack_into("<I", header, 56, len(plan.fft_bins))
    struct.pack_into("<I", header, 60, len(payload))
    struct.pack_into("<I", header, 64, len(generator))
    struct.pack_into("<I", header, 68, generator_offset)
    struct.pack_into("<I", header, 72, octaves_offset)
    struct.pack_into("<I", header, 76, row_offsets_offset)
    struct.pack_into("<I", header, 80, fft_bins_offset)
    struct.pack_into("<I", header, 84, coefficients_offset)
    struct.pack_into("<I", header, 88, bin_lengths_offset)
    struct.pack_into("<I", header, 92, downsample_offset)
    header[96:128] = bytes.fromhex(payload_sha256)
    artifact = bytes(header + payload)
    return artifact, generator.decode("ascii"), payload_sha256


def _get_manifest(
    plan: PlanData,
    artifact: bytes,
    generator: str,
    payload_sha256: str,
) -> dict[str, object]:
    return {
        "formatVersion": FORMAT_VERSION,
        "generator": generator,
        "payloadSha256": payload_sha256,
        "artifactSha256": hashlib.sha256(artifact).hexdigest(),
        "config": {
            "sampleRate": SAMPLE_RATE,
            "hopLength": HOP_LENGTH,
            "fmin": FMIN,
            "nBins": N_BINS,
            "binsPerOctave": BINS_PER_OCTAVE,
            "output": "logMagnitude",
        },
        "earlyDownsampleCount": plan.early_downsample_count,
        "octaveCount": len(plan.octaves),
        "coefficientCount": len(plan.fft_bins),
        "downsampleTapCount": FIR_TAP_COUNT,
        "downsample": {
            "algorithm": "kaiser-lowpass-fir",
            "cutoff": FIR_CUTOFF,
            "kaiserBeta": FIR_KAISER_BETA,
            "gain": float(np.sqrt(2)),
            "delay": (FIR_TAP_COUNT - 1) // 2,
            "boundary": "constant",
            "outputLength": "ceil(input/2)",
        },
        "octaves": [
            {
                "index": octave.index,
                "sampleRate": octave.sample_rate,
                "hopLength": octave.hop_length,
                "fftSize": octave.fft_size,
                "binStart": octave.bin_start,
                "binCount": octave.bin_count,
            }
            for octave in plan.octaves
        ],
    }


def _write_typescript_payload(path: Path, artifact: bytes) -> None:
    encoded = base64.b64encode(artifact).decode("ascii")
    chunks = textwrap.wrap(encoded, 100)
    text = "export const referencePlanBase64 = [\n"
    text += "".join(f"  '{chunk}',\n" for chunk in chunks)
    text += "].join('');\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--typescript-payload", type=Path, required=True)
    args = parser.parse_args()

    plan = _get_plan_data()
    artifact, generator, payload_sha256 = _create_artifact(plan)
    manifest = _get_manifest(plan, artifact, generator, payload_sha256)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(artifact)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_typescript_payload(args.typescript_payload, artifact)
    print(
        f"wrote {args.out} ({len(artifact)} bytes, " f"payload sha256={payload_sha256})"
    )


if __name__ == "__main__":
    main()
