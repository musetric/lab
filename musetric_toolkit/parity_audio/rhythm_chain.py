import numpy as np
import onnxruntime as ort
import torch

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.parity_audio.case_writer import CaseWriter
from musetric_toolkit.parity_audio.host_audio import MEAN, read_mono
from musetric_toolkit.rhythm_audio.bpm_estimator import summarize_rhythm

SAMPLE_RATE = 22050
FPS = 50
CHUNK_SIZE = 1500
BORDER_SIZE = 6
EMPTY_LOGIT = -1000.0


def window_starts(frames: int) -> list[int]:
    stride = CHUNK_SIZE - 2 * BORDER_SIZE
    starts = list(range(-BORDER_SIZE, frames - BORDER_SIZE, stride))
    if frames > stride and starts:
        starts[-1] = frames - (CHUNK_SIZE - BORDER_SIZE)
    return starts


def cut_windows(spect: np.ndarray, starts: list[int]) -> np.ndarray:
    windows = np.zeros((len(starts), CHUNK_SIZE, spect.shape[1]))
    for index, start in enumerate(starts):
        first = max(start, 0)
        last = min(start + CHUNK_SIZE, spect.shape[0])
        windows[index, first - start : last - start] = spect[first:last]
    return windows


def stitch(logits: np.ndarray, starts: list[int], frames: int) -> np.ndarray:
    stitched = np.full(frames, EMPTY_LOGIT)
    for index in reversed(range(len(starts))):
        at = starts[index] + BORDER_SIZE
        count = min(CHUNK_SIZE - 2 * BORDER_SIZE, frames - at)
        stitched[at : at + count] = logits[index, BORDER_SIZE : BORDER_SIZE + count]
    return stitched


def cpu_session(path: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def run(args, writer: CaseWriter) -> None:
    from beat_this.inference import Audio2Beats, load_model  # noqa: PLC0415
    from beat_this.model.postprocessor import Postprocessor  # noqa: PLC0415
    from beat_this.preprocessing import LogMelSpect  # noqa: PLC0415

    audio = read_mono(args.audio_path, SAMPLE_RATE, MEAN)
    samples = audio.shape[0]
    with torch.inference_mode():
        spect = (
            LogMelSpect()(torch.from_numpy(audio.astype(np.float32))).double().numpy()
        )
    frames = spect.shape[0]
    starts = window_starts(frames)
    windows = cut_windows(spect, starts)
    writer.meta.update(
        {
            "sampleRate": SAMPLE_RATE,
            "samples": samples,
            "frames": frames,
            "starts": starts,
        }
    )
    writer.tensor("unit.input", "reference", audio)
    writer.tensor("features", "reference", windows)

    send_message({"type": "progress", "progress": 0.2})
    model = load_model("final0", "cpu")
    beat = np.zeros((len(starts), CHUNK_SIZE))
    downbeat = np.zeros((len(starts), CHUNK_SIZE))
    with torch.inference_mode():
        for index in range(len(starts)):
            output = model(
                torch.from_numpy(windows[index : index + 1].astype(np.float32))
            )
            beat[index] = output["beat"][0].double().numpy()
            downbeat[index] = output["downbeat"][0].double().numpy()

    writer.tensor("model.input", "reference", windows[:1])
    writer.tensor("model.output.beat", "original", beat[:1])
    writer.tensor("model.output.downbeat", "original", downbeat[:1])
    session = cpu_session(args.onnx)
    onnx_beat, onnx_downbeat = session.run(
        ["beat", "downbeat"], {"spect": windows[:1].astype(np.float32)}
    )
    writer.tensor("model.output.beat", "onnx-cpu", onnx_beat)
    writer.tensor("model.output.downbeat", "onnx-cpu", onnx_downbeat)

    send_message({"type": "progress", "progress": 0.6})
    stitched_beat = stitch(beat, starts, frames)
    stitched_downbeat = stitch(downbeat, starts, frames)
    writer.tensor("logits.beat", "reference", stitched_beat)
    writer.tensor("logits.downbeat", "reference", stitched_downbeat)
    beats, downbeats = Postprocessor(type="dbn", fps=FPS)(
        torch.from_numpy(stitched_beat), torch.from_numpy(stitched_downbeat)
    )
    bpm, beats, downbeats, meter = summarize_rhythm(
        np.asarray(beats, dtype=np.float64),
        np.asarray(downbeats, dtype=np.float64),
        samples / SAMPLE_RATE,
    )
    writer.tensor("result.beats", "reference", beats)
    writer.tensor("result.downbeats", "reference", downbeats)
    writer.tensor("result.bpm", "reference", np.asarray([bpm]))
    writer.tensor("result.meter", "reference", np.asarray([meter]))

    send_message({"type": "progress", "progress": 0.8})
    author_beats, author_downbeats = Audio2Beats("final0", "cpu", dbn=True)(
        audio, SAMPLE_RATE
    )
    writer.tensor("result.beats", "author", np.asarray(author_beats))
    writer.tensor("result.downbeats", "author", np.asarray(author_downbeats))
