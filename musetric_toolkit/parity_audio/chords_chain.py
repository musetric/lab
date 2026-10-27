import numpy as np
import onnxruntime as ort
import torch

from musetric_toolkit.chords_audio.chordmini_checkpoint import ensure_checkpoint
from musetric_toolkit.chords_audio.chordmini_runner import CONFIG_PATH, run_chordmini
from musetric_toolkit.parity_audio.case_writer import CaseWriter
from musetric_toolkit.parity_audio.host_audio import MEAN, read_mono

SAMPLE_RATE = 22050
HOP_LENGTH = 2048
N_BINS = 144
BINS_PER_OCTAVE = 24
SEQ_LEN = 108
WINDOWS_PER_RUN = 16
SMOOTHING_RADIUS = 4


def log_cqt(audio: np.ndarray) -> np.ndarray:
    import librosa  # noqa: PLC0415

    cqt = librosa.cqt(
        audio,
        sr=SAMPLE_RATE,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz("C1"),
    )
    return np.log(np.abs(cqt) + 1e-6).T.astype(np.float64)


def model_windows(features: np.ndarray) -> np.ndarray:
    frames = features.shape[0]
    windows = -(-frames // SEQ_LEN)
    padded_windows = -(-windows // WINDOWS_PER_RUN) * WINDOWS_PER_RUN
    padded = np.zeros((padded_windows * SEQ_LEN, N_BINS))
    padded[:frames] = features
    return padded.reshape(padded_windows, SEQ_LEN, N_BINS)


def smooth_argmax(logits: np.ndarray, frames: int) -> np.ndarray:
    indices = np.zeros(frames, dtype=np.int64)
    for frame in range(frames):
        window, local = divmod(frame, SEQ_LEN)
        first = max(0, local - SMOOTHING_RADIUS)
        last = min(SEQ_LEN, local + SMOOTHING_RADIUS + 1)
        indices[frame] = int(np.argmax(logits[window, first:last].sum(axis=0)))
    return indices


def torch_logits(models_path: str, windows: np.ndarray) -> np.ndarray:
    from musetric_toolkit.chords_audio.chordmini.models import (  # noqa: PLC0415
        load_model,
    )
    from musetric_toolkit.chords_audio.chordmini.utils import HParams  # noqa: PLC0415

    config = HParams.load(str(CONFIG_PATH))
    model, mean, std = load_model(
        str(ensure_checkpoint(models_path)), "ChordNet", config, torch.device("cpu")
    )
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(windows.astype(np.float32))
        normalized = (tensor - torch.as_tensor(mean, dtype=torch.float32)) / (
            torch.as_tensor(std, dtype=torch.float32) + 1e-8
        )
        logits, _ = model(normalized)
    return logits.double().numpy()


def run(args, writer: CaseWriter) -> None:
    audio = read_mono(args.audio_path, SAMPLE_RATE, MEAN)
    features = log_cqt(audio)
    frames = features.shape[0]
    windows = model_windows(features)
    writer.meta.update(
        {"sampleRate": SAMPLE_RATE, "samples": audio.shape[0], "frames": frames}
    )
    writer.tensor("unit.input", "reference", audio)
    writer.tensor("features", "reference", features)

    logits = torch_logits(args.models_path, windows)
    writer.tensor("model.input", "reference", windows[:WINDOWS_PER_RUN])
    writer.tensor("model.output", "original", logits[:WINDOWS_PER_RUN])
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    session = ort.InferenceSession(
        args.onnx, options, providers=["CPUExecutionProvider"]
    )
    (onnx_logits,) = session.run(
        ["logits"], {"features": windows[:WINDOWS_PER_RUN].astype(np.float32)}
    )
    writer.tensor("model.output", "onnx-cpu", onnx_logits)
    writer.tensor("result.indices", "reference", smooth_argmax(logits, frames), "int32")

    predictions, _, _ = run_chordmini(
        args.audio_path, ensure_checkpoint(args.models_path)
    )
    writer.tensor("result.indices", "author", predictions[:frames], "int32")
