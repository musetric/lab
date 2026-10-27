from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.chords_audio.chordmini.models import load_model
from musetric_toolkit.chords_audio.chordmini.utils import (
    HParams,
    get_config_value,
    get_device,
    idx2voca_chord,
)
from musetric_toolkit.common.logger import send_message

CONFIG_PATH = Path(__file__).parent / "chordmini" / "ChordMini.yaml"
BATCH_SIZE = 16


def _extract_features(audio_path: str, config) -> tuple[np.ndarray, float]:
    import librosa  # noqa: PLC0415

    sample_rate = get_config_value(config, "mp3", "song_hz", 22050)
    hop_length = get_config_value(config, "feature", "hop_length", 2048)
    n_bins = get_config_value(config, "feature", "n_bins", 144)
    bins_per_octave = get_config_value(config, "feature", "bins_per_octave", 24)

    audio, sr = librosa.load(audio_path, sr=sample_rate)
    cqt = librosa.cqt(
        audio,
        sr=sr,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        fmin=librosa.note_to_hz("C1"),
    )
    frame_duration = float(hop_length) / float(sr)
    features = np.log(np.abs(cqt) + 1e-6).T.astype(np.float32)
    return features, frame_duration


def _predict_frames(
    model,
    features: np.ndarray,
    mean,
    std,
    seq_len: int,
) -> np.ndarray:
    num_frames = int(features.shape[0])
    if num_frames == 0:
        return np.array([], dtype=np.int64)

    device = next(model.parameters()).device
    remainder = num_frames % seq_len
    pad = 0 if remainder == 0 else seq_len - remainder
    padded = np.pad(features, ((0, pad), (0, 0))) if pad else features

    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device)

    starts = list(range(0, padded.shape[0], seq_len))
    predictions = np.zeros(num_frames, dtype=np.int64)

    with torch.no_grad():
        for batch_start in range(0, len(starts), BATCH_SIZE):
            batch_starts = starts[batch_start : batch_start + BATCH_SIZE]
            windows = np.stack([padded[s : s + seq_len] for s in batch_starts])
            tensor = torch.from_numpy(windows).float().to(device)
            tensor = (tensor - mean_tensor) / (std_tensor + 1e-8)
            output = model.predict(tensor, per_frame=True).cpu().numpy()
            for local_idx, start_frame in enumerate(batch_starts):
                valid = min(seq_len, num_frames - start_frame)
                predictions[start_frame : start_frame + valid] = output[
                    local_idx, :valid
                ]
            done = min(batch_start + BATCH_SIZE, len(starts))
            send_message(
                {"type": "progress", "progress": 0.5 + 0.4 * done / len(starts)}
            )

    return predictions


def run_chordmini(
    audio_path: str,
    checkpoint_path: Path,
) -> tuple[np.ndarray, float, dict]:
    config = HParams.load(str(CONFIG_PATH))
    device = get_device()

    model, mean, std = load_model(str(checkpoint_path), "ChordNet", config, device)
    idx_to_chord = idx2voca_chord()
    model.idx_to_chord = idx_to_chord
    model.eval()
    send_message({"type": "progress", "progress": 0.4})

    seq_len = int(get_config_value(config, "model", "seq_len", 108))
    features, frame_duration = _extract_features(audio_path, config)
    send_message({"type": "progress", "progress": 0.5})

    predictions = _predict_frames(model, features, mean, std, seq_len)
    return predictions, frame_duration, idx_to_chord
