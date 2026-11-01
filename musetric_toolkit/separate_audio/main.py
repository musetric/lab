from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.parity_audio.unit_plans import normalize_peak
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.ffmpeg.write import write_audio_file
from musetric_toolkit.separate_audio.student_separator import (
    CHANNELS,
    SAMPLE_RATE,
    load_student,
    separate_mixture,
)
from musetric_toolkit.separate_audio.system_info import (
    ensure_ffmpeg,
    print_acceleration_info,
    setup_torch_optimization,
)


def main(args) -> None:
    ensure_ffmpeg()
    print_acceleration_info()
    setup_torch_optimization()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_student(Path(args.models_path), device)
    source = read_audio_file(args.source_path, SAMPLE_RATE, CHANNELS)
    mixture = normalize_peak(source.astype(np.float64))
    folded = separate_mixture(model, mixture)
    vocals, backing = folded[:CHANNELS], folded[CHANNELS:]
    stems = (
        (args.lead_path, vocals - backing),
        (args.backing_path, backing),
        (args.instrumental_path, mixture - vocals),
    )
    for path, stem in stems:
        write_audio_file(path, normalize_peak(stem).T, SAMPLE_RATE, args.sample_rate)
