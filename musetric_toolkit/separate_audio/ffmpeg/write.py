import os

import numpy as np

from musetric_toolkit.separate_audio.ffmpeg.runner import run_ffmpeg


def write_audio_file(
    output_path: str,
    audio_time_channels: np.ndarray,
    sample_rate: int,
    output_sample_rate: int | None = None,
) -> None:
    if audio_time_channels.dtype != np.float32:
        audio_time_channels = audio_time_channels.astype(np.float32, order="C")

    ffmpeg_command = [
        "ffmpeg",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(audio_time_channels.shape[1]),
        "-i",
        "-",
    ]

    ffmpeg_command += [
        "-ar",
        str(output_sample_rate or sample_rate),
        "-c:a",
        "flac",
        "-sample_fmt",
        "s32",
        "-f",
        "flac",
        "-y",
        output_path,
    ]

    pcm_interleaved_bytes = audio_time_channels.tobytes(order="C")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    run_ffmpeg(
        ffmpeg_command,
        input_bytes=pcm_interleaved_bytes,
        capture_stdout=False,
        context="ffmpeg failed to write audio",
    )
