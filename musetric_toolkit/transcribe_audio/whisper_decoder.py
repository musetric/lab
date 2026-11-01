from dataclasses import dataclass, field

import numpy as np
import torch
from numba import njit
from transformers import (
    AutoTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput

from musetric_toolkit.transcribe_audio.rounding import js_round, round_to
from musetric_toolkit.transcribe_audio.spectral_chunker import SAMPLE_RATE
from musetric_toolkit.transcribe_audio.whisper_segments import (
    span_text,
    split_by_segments,
)
from musetric_toolkit.transcribe_audio.word_chunks import TIME_PRECISION, WordChunker

ENCODER_POSITION_SAMPLES = 320
ENCODER_POSITIONS = 1500
MAX_DECODE_TOKENS = 400
MIN_DECODE_TOKENS = 32
TOKENS_PER_SECOND = 12
MAX_DECODER_POSITIONS = 448
ORIGINAL_ID = "openai/whisper-large-v3-turbo"
ORIGINAL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
MEDIAN_FILTER_WIDTH = 7


def decode_budget(samples: int) -> tuple[int, int]:
    positions = min(ENCODER_POSITIONS, js_round(samples / ENCODER_POSITION_SAMPLES))
    seconds = samples / SAMPLE_RATE
    budget = min(
        MAX_DECODE_TOKENS,
        max(
            MIN_DECODE_TOKENS, js_round(seconds * TOKENS_PER_SECOND) + MIN_DECODE_TOKENS
        ),
    )
    return positions, budget


def median_filter(weights: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    length = weights.shape[-1]
    index = np.abs(np.arange(length)[:, None] + np.arange(-half, half + 1)[None, :])
    index = np.where(index >= length, 2 * (length - 1) - index, index)
    return np.sort(weights[..., index], axis=-1)[..., half]


@njit(cache=True)
def dynamic_time_warping(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = matrix.shape
    cost = np.full((rows + 1, columns + 1), np.inf)
    cost[0, 0] = 0
    trace = np.full((rows + 1, columns + 1), -1)
    for column in range(1, columns + 1):
        for row in range(1, rows + 1):
            diagonal = cost[row - 1, column - 1]
            above = cost[row - 1, column]
            beside = cost[row, column - 1]
            if diagonal < above and diagonal < beside:
                move, best = 0, diagonal
            elif above < diagonal and above < beside:
                move, best = 1, above
            else:
                move, best = 2, beside
            cost[row, column] = matrix[row - 1, column - 1] + best
            trace[row, column] = move
    trace[0, :] = 2
    trace[:, 0] = 1
    text = np.zeros(rows + columns, dtype=np.int64)
    time = np.zeros(rows + columns, dtype=np.int64)
    length = 0
    row, column = rows, columns
    while row > 0 or column > 0:
        text[length] = row - 1
        time[length] = column - 1
        length += 1
        move = trace[row, column]
        if move in (0, 1):
            row -= 1
        if move in (0, 2):
            column -= 1
    return text[:length][::-1], time[:length][::-1]


def token_times(
    cross_attentions, heads: list[list[int]], positions: int, prompt_length: int
) -> np.ndarray:
    weights = np.stack(
        [
            cross_attentions[layer][0, head, :, :positions].float().cpu().numpy()
            for layer, head in heads
        ]
    )
    mean = weights.mean(axis=1, keepdims=True)
    std = weights.std(axis=1, keepdims=True)
    weights = median_filter((weights - mean) / std, MEDIAN_FILTER_WIDTH)
    matrix = weights[:, prompt_length:].mean(axis=0)
    text, time = dynamic_time_warping(-matrix.astype(np.float64))
    jumps = np.concatenate([[True], np.diff(text) != 0])
    return time[jumps] * TIME_PRECISION


def load_original(device: torch.device) -> WhisperForConditionalGeneration:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = WhisperForConditionalGeneration.from_pretrained(
        ORIGINAL_ID,
        revision=ORIGINAL_REVISION,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    return model.to(device).eval()


def generate_tokens(
    model: WhisperForConditionalGeneration,
    config,
    features: torch.Tensor,
    encoder_outputs: BaseModelOutput,
    language: str,
    **options,
) -> list[int]:
    sequence = model.generate(
        input_features=features,
        encoder_outputs=encoder_outputs,
        generation_config=config,
        language=language,
        task="transcribe",
        force_unique_generate_call=True,
        **options,
    )
    return [int(token) for token in sequence[0].tolist()]


@dataclass
class DecodeResult:
    text: str = ""
    chunks: list[dict] = field(default_factory=list)
    segments: list[list[dict]] = field(default_factory=list)


class WhisperDecoder:
    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._model = load_original(device)
        self._config = self._model.generation_config
        self._extractor = WhisperFeatureExtractor.from_pretrained(
            ORIGINAL_ID, revision=ORIGINAL_REVISION
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            ORIGINAL_ID, revision=ORIGINAL_REVISION
        )
        self._chunker = WordChunker(self._tokenizer)

    def _encode(self, audio: np.ndarray) -> tuple[torch.Tensor, BaseModelOutput]:
        features = self._extractor(
            audio.astype(np.float32), sampling_rate=SAMPLE_RATE, return_tensors="np"
        ).input_features
        tensor = torch.from_numpy(features).to(self._device)
        with torch.inference_mode():
            encoded = self._model.model.encoder(tensor).last_hidden_state
        return tensor, BaseModelOutput(last_hidden_state=encoded)

    def _timed_tokens(
        self,
        encoder_outputs: BaseModelOutput,
        tokens: list[int],
        prompt_length: int,
        positions: int,
    ) -> list[float]:
        with torch.inference_mode():
            forward = self._model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=torch.tensor([tokens[:-1]], device=self._device),
                output_attentions=True,
            )
        jumps = token_times(
            forward.cross_attentions,
            self._config.alignment_heads,
            positions,
            prompt_length,
        ).tolist()
        return [0.0] * prompt_length + jumps + ([jumps[-1]] if jumps else [])

    def _decode(
        self,
        audio: np.ndarray,
        language: str,
        prompt_length: int,
        return_timestamps: bool,
        **options,
    ) -> tuple[list[int], list[float]]:
        positions, budget = decode_budget(audio.shape[0])
        features, encoder_outputs = self._encode(audio)
        tokens = generate_tokens(
            self._model,
            self._config,
            features,
            encoder_outputs,
            language,
            return_timestamps=return_timestamps,
            **{"max_new_tokens": budget, **options},
        )
        return tokens, self._timed_tokens(
            encoder_outputs, tokens, prompt_length, positions
        )

    def _timestamped_result(
        self, tokens: list[int], times: list[float]
    ) -> DecodeResult:
        begin = self._chunker.timestamp_begin
        prefix = next(
            (index for index, token in enumerate(tokens) if token >= begin), 0
        )
        kept = tokens[prefix:]
        text, chunks = self._chunker.decode_words(
            kept, [round_to(time, 2) for time in times[prefix:]]
        )
        return DecodeResult(
            text,
            chunks,
            split_by_segments(kept, chunks, begin, TIME_PRECISION),
        )

    def decode_timestamped(
        self, audio: np.ndarray, language: str, guard: dict | None = None
    ) -> DecodeResult:
        tokens, times = self._decode(audio, language, 3, True, **(guard or {}))
        return self._timestamped_result(tokens, times)

    def decode_prompted(
        self, audio: np.ndarray, language: str, prompt: str
    ) -> DecodeResult:
        prompt_ids = self._tokenizer.get_prompt_ids(prompt)
        prompt_length = len(prompt_ids) + 3
        _, budget = decode_budget(audio.shape[0])
        tokens, times = self._decode(
            audio,
            language,
            prompt_length,
            True,
            prompt_ids=torch.from_numpy(prompt_ids).to(self._device),
            max_new_tokens=min(budget, MAX_DECODER_POSITIONS - prompt_length),
        )
        return self._timestamped_result(tokens, times)

    def decode_aligned(self, audio: np.ndarray, language: str) -> DecodeResult:
        tokens, times = self._decode(audio, language, 4, False)
        chunks: list[dict] = []
        for token, time in zip(tokens, times, strict=True):
            piece = self._tokenizer.decode([token], skip_special_tokens=True)
            if not piece:
                continue
            if piece.startswith(" ") or not chunks:
                chunks.append({"text": piece, "timestamp": [time, time]})
                continue
            chunks[-1]["text"] += piece
            chunks[-1]["timestamp"][1] = time
        return DecodeResult(span_text(chunks), chunks, [chunks])

    def detect_language(self, audio: np.ndarray) -> str:
        _, encoder_outputs = self._encode(audio)
        start = [[self._config.decoder_start_token_id]]
        with torch.inference_mode():
            logits = self._model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=torch.tensor(start, device=self._device),
            ).logits[0, 0]
        best = max(
            self._config.lang_to_id.items(),
            key=lambda entry: float(logits[entry[1]]),
        )
        return best[0][2:-2]
