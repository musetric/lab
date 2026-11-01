import numpy as np
import torch

from musetric_toolkit.parity_audio.case_writer import CaseWriter
from musetric_toolkit.transcribe_audio.spectral_chunker import SAMPLE_RATE
from musetric_toolkit.transcribe_audio.transcribe_pipeline import transcribe_voice
from musetric_toolkit.transcribe_audio.voice import read_voice
from musetric_toolkit.transcribe_audio.whisper_decoder import WhisperDecoder
from musetric_toolkit.transcribe_audio.whisper_runtime import WhisperRuntime

FNV_OFFSET = 0x811C9DC5
FNV_PRIME = 0x01000193
WORD_BITS = 32


def word_hash(text: str) -> int:
    letters = "".join(ch for ch in text.lower() if ch.isalnum())
    value = FNV_OFFSET
    for byte in letters.encode("utf-8"):
        value = ((value ^ byte) * FNV_PRIME) % 2**WORD_BITS
    return value - 2**WORD_BITS if value >= 2 ** (WORD_BITS - 1) else value


def word_rows(words: list[dict], time_key: str) -> np.ndarray:
    rows = [(word_hash(word["text"]), round(word[time_key] * 1000)) for word in words]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def run(args, writer: CaseWriter) -> None:
    voice = read_voice(args.audio_path, args.backing_path)
    writer.meta.update({"sampleRate": SAMPLE_RATE, "samples": voice.shape[0]})
    writer.tensor("voice", "reference", voice)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lines = transcribe_voice(voice, WhisperRuntime(WhisperDecoder(device)))
    words = [word for line in lines for word in line["words"]]
    line_ends = [line["words"][-1] for line in lines]
    writer.tensor("lines.words", "reference", word_rows(words, "start"), "int32")
    writer.tensor("lines.breaks", "reference", word_rows(line_ends, "end"), "int32")
    writer.result("lines", "reference", lines)
