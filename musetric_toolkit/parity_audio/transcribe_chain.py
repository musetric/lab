from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import GenerationConfig, WhisperFeatureExtractor
from transformers.modeling_outputs import BaseModelOutput

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.parity_audio.case_writer import CaseWriter
from musetric_toolkit.parity_audio.host_audio import POWER, read_mono
from musetric_toolkit.transcribe_audio.rounding import js_round
from musetric_toolkit.transcribe_audio.whisper_decoder import (
    decode_budget,
    generate_tokens,
    load_original,
    token_times,
)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 30 * SAMPLE_RATE
END_OF_TEXT = 50257
DECODER_FILE = "decoder_model_merged_fp16.onnx"
CROSS_FILE = "cross_kv_fp16.onnx"
BRANCH_INPUT = "use_cache_branch"


def cpu_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def generated_tokens(tokens: list[int], timestamp_begin: int) -> list[int]:
    start = next(
        (index for index, token in enumerate(tokens) if token >= timestamp_begin),
        len(tokens),
    )
    kept = tokens[start:]
    return kept[:-1] if kept and kept[-1] == END_OF_TEXT else kept


def text_rows(tokens: list[int], times: list[float]) -> np.ndarray:
    rows = [
        (token, js_round(time * 100) / 100)
        for token, time in zip(tokens, times, strict=True)
        if token < END_OF_TEXT
    ]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2)


def empty_past(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    past = {}
    for model_input in session.get_inputs():
        if model_input.name.startswith("past_key_values."):
            _, heads, _, width = model_input.shape
            past[model_input.name] = np.zeros((1, heads, 0, width), dtype=np.float16)
    return past


def cross_past(
    cross: ort.InferenceSession, states: np.ndarray
) -> dict[str, np.ndarray]:
    names = [output.name for output in cross.get_outputs()]
    values = cross.run(names, {"encoder_hidden_states": states})
    return {
        name.replace("present.", "past_key_values."): value
        for name, value in zip(names, values, strict=True)
    }


def teacher_forced_onnx(
    session: ort.InferenceSession,
    encoder_states: np.ndarray,
    forced: list[int],
    prompt_length: int,
    cross: ort.InferenceSession | None = None,
) -> np.ndarray:
    past = empty_past(session)
    names = [output.name for output in session.get_outputs()]
    states = encoder_states.astype(np.float32)
    merged = any(
        model_input.name == BRANCH_INPUT for model_input in session.get_inputs()
    )
    if merged:
        prompt_steps = [forced[:prompt_length]]
    else:
        if cross is None:
            raise ValueError("a flat decoder step needs its cross_kv projection")
        past.update(cross_past(cross, states))
        prompt_steps = [[token] for token in forced[:prompt_length]]

    def step(ids: list[int], use_cache: bool) -> dict[str, np.ndarray]:
        feeds = {
            **past,
            "input_ids": np.asarray([ids], dtype=np.int64),
            "encoder_hidden_states": states,
        }
        if merged:
            feeds[BRANCH_INPUT] = np.asarray([use_cache])
        values = session.run(names, feeds)
        outputs = dict(zip(names, values, strict=True))
        for name in past:
            if ".decoder." in name or not use_cache:
                past[name] = outputs[name.replace("past_key_values.", "present.")]
        return outputs

    logits = [step(ids, use_cache=not merged)["logits"][0] for ids in prompt_steps]
    for position in range(prompt_length, len(forced) - 1):
        logits.append(step([forced[position]], use_cache=True)["logits"][0])
    return np.concatenate(logits).astype(np.float64)


def run(args, writer: CaseWriter) -> None:
    bundle = Path(args.onnx).parent
    language = args.language
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio = read_mono(args.audio_path, SAMPLE_RATE, POWER)[:CHUNK_SAMPLES]
    samples = audio.shape[0]
    positions, budget = decode_budget(samples)
    extractor = WhisperFeatureExtractor.from_pretrained(bundle)
    features = extractor(
        audio.astype(np.float32), sampling_rate=SAMPLE_RATE, return_tensors="np"
    ).input_features
    config = GenerationConfig.from_pretrained(bundle)
    timestamp_begin = config.no_timestamps_token_id + 1
    prompt = [
        config.decoder_start_token_id,
        config.lang_to_id[f"<|{language}|>"],
        config.task_to_id["transcribe"],
    ]
    writer.tensor("unit.input", "reference", audio)
    writer.tensor("model.input", "reference", features)

    send_message({"type": "progress", "progress": 0.1})
    model = load_original(device)
    config.alignment_heads = model.generation_config.alignment_heads
    features_t = torch.from_numpy(features).to(device)
    with torch.inference_mode():
        encoded = model.model.encoder(features_t).last_hidden_state
    writer.tensor("encoder.output", "original", encoded.double().cpu().numpy())
    encoder_outputs = BaseModelOutput(last_hidden_state=encoded)
    encoder = cpu_session(Path(args.onnx))
    (onnx_encoded,) = encoder.run(["last_hidden_state"], {"input_features": features})
    writer.tensor("encoder.output", "onnx-cpu", onnx_encoded)

    send_message({"type": "progress", "progress": 0.4})
    tokens = generate_tokens(
        model,
        config,
        features_t,
        encoder_outputs,
        language,
        return_timestamps=True,
        max_new_tokens=budget,
    )
    generated = generated_tokens(tokens, timestamp_begin)
    forced = prompt + generated + [END_OF_TEXT]
    writer.tensor("decoder.tokens", "reference", np.asarray(forced), "int32")
    writer.meta.update(
        {
            "sampleRate": SAMPLE_RATE,
            "samples": samples,
            "language": language,
            "positions": positions,
            "maxNewTokens": budget,
            "promptLength": len(prompt),
            "timestampBegin": timestamp_begin,
        }
    )

    send_message({"type": "progress", "progress": 0.6})
    with torch.inference_mode():
        forward = model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=torch.tensor([forced[:-1]], device=device),
            output_attentions=True,
        )
    writer.tensor(
        "decoder.logits", "original", forward.logits[0].double().cpu().numpy()
    )
    times = token_times(
        forward.cross_attentions, config.alignment_heads, positions, len(prompt)
    )
    writer.tensor("result.tokens", "reference", text_rows(generated, times.tolist()))
    decoder = cpu_session(bundle / DECODER_FILE)
    cross = (
        cpu_session(bundle / CROSS_FILE) if (bundle / CROSS_FILE).is_file() else None
    )
    writer.tensor(
        "decoder.logits",
        "onnx-cpu",
        teacher_forced_onnx(decoder, onnx_encoded, forced, len(prompt), cross),
    )

    send_message({"type": "progress", "progress": 0.8})
    author = model.generate(
        input_features=features_t,
        encoder_outputs=encoder_outputs,
        language=language,
        task="transcribe",
        return_timestamps=True,
        return_token_timestamps=True,
    )
    writer.tensor(
        "result.tokens",
        "author",
        text_rows(
            [int(token) for token in author["sequences"][0].tolist()],
            [float(time) for time in author["token_timestamps"][0].tolist()],
        ),
    )
