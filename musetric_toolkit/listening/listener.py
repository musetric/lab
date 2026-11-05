import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from musetric_toolkit.listening.levels import RATE
from musetric_toolkit.transcribe_audio.hallucination_filter import is_hallucination
from musetric_toolkit.transcribe_audio.whisper_decoder import (
    ORIGINAL_ID,
    ORIGINAL_REVISION,
)

WHISPER_RATE = 16000
MAX_TOKENS = 64


class Listener:
    def __init__(self, batch: int) -> None:
        self.batch = batch
        self.processor = WhisperProcessor.from_pretrained(
            ORIGINAL_ID, revision=ORIGINAL_REVISION
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            ORIGINAL_ID, revision=ORIGINAL_REVISION, torch_dtype=torch.float16
        )
        self.model.to("cuda").eval()

    @torch.inference_mode()
    def run_batch(self, batch: list[np.ndarray]) -> list[dict]:
        inputs = self.processor(
            batch,
            sampling_rate=WHISPER_RATE,
            return_tensors="pt",
            padding="max_length",
        )
        features = inputs.input_features.to("cuda", torch.float16)
        output = self.model.generate(
            features, task="transcribe", max_new_tokens=MAX_TOKENS
        )
        rows = []
        for text in self.processor.batch_decode(output, skip_special_tokens=True):
            clean = "" if is_hallucination(text) else text.strip()
            rows.append(dict(words=len(clean.split()), text=clean))
        return rows

    def __call__(self, audio: list[np.ndarray]) -> list[dict]:
        small = [
            librosa.resample(
                x.mean(0).astype(np.float32), orig_sr=RATE, target_sr=WHISPER_RATE
            )
            for x in audio
        ]
        rows = []
        for start in range(0, len(small), self.batch):
            rows += self.run_batch(small[start : start + self.batch])
        return rows
