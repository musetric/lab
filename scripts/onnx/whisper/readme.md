# Whisper large-v3-turbo → word-timestamped ONNX

Exports **openai/whisper-large-v3-turbo** to the [transformers.js](https://github.com/huggingface/transformers.js)
ONNX layout, with the cross-attention alignment heads in `generation_config.json`
for word timestamps, and prepares the graphs of [`musetric/whisper-large-v3-turbo-onnx`](https://huggingface.co/musetric/whisper-large-v3-turbo-onnx)
for the `@musetric/ai` WebGPU runtime. `convert.py`, `quantize.py` and `extra/whisper.py`
are vendored from [transformers.js](https://github.com/huggingface/transformers.js/tree/main/scripts)
(Apache-2.0); only their whisper path runs.

## Export

The export runs in an isolated uv environment of `requirements.txt` on Python
3.11, never synced into the project:

- `transformers` 4.42.4: 4.43 adds a `cache_position` decoder input that the runtime does not feed.
- `torch` 2.4.1: its `torch.onnx.export` is the TorchScript exporter that `optimum` 1.21.3 takes.

```bash
PYTHONPATH=scripts/onnx uv run --no-project --python 3.11 --with-requirements scripts/onnx/whisper/requirements.txt python -m whisper.convert --model_id openai/whisper-large-v3-turbo --quantize --modes q4 --output_attentions --skip_validation --output_parent_dir tmp/whisper-export
```

`--output_attentions` selects the config that outputs cross-attentions and the
task stays `auto`; the graphs land in `tmp/whisper-export/openai/whisper-large-v3-turbo/onnx/`.

## Passes, in order

```bash
uv run python scripts/onnx/whisper/mobile_encoder.py --input X --output Y  # q4 encoder
uv run python scripts/onnx/whisper/fp16_decoder.py --input X --output Y  # fp32 merged decoder to float16
uv run python scripts/onnx/whisper/mobile_decoder.py --input X --output Y  # both merged decoders
uv run python scripts/onnx/whisper/trim_cross_attentions.py --input X --output Y --generation-config G  # both
uv run python scripts/onnx/whisper/pack_step_outputs.py --input X --output Y --generation-config G  # both
uv run python scripts/onnx/whisper/flatten_decoder_step.py --input X --step Y --cross Z  # fp16 decoder
uv run python scripts/onnx/whisper/scale_residual_stream.py --input X --output Y  # fp16 step
```

- `mobile_encoder.py` runs `block_attention.py`, `conv_to_matmul.py` and `transpose_to_gather.py`, `mobile_decoder.py` runs `pad_attention_scores.py`; both are idempotent, and each inner pass also runs alone.
- `trim_cross_attentions.py` rewrites the generation config in place: run it once per repository, and give the q4 decoder a copy of the original.
- `flatten_decoder_step.py` takes the fp16 decoder only; the q4 decoder stays merged.
- A pass keys the nodes it replaces by their position in `graph.node`, never by `id(node)`.

`stage_whisper.py` copies the published set flat into the clone of the model
repository, refusing a graph that skipped its mobile pass; the model card holds
each file's contract, and the [practice of models](https://github.com/musetric/headquarters/blob/main/practices/models.md) the publication.

To find the graph a provider gets wrong, transformers.js takes `device` as a map
keyed by file name, such as `{ encoder_model: 'webgpu', decoder_model_merged: 'wasm' }`,
while `session_options.executionProviders` is unset; the graph whose move changes the output is the wrong one.

Details: [musetric/lab#59](https://github.com/musetric/lab/pull/59), [#66](https://github.com/musetric/lab/pull/66), [#67](https://github.com/musetric/lab/pull/67), [#71](https://github.com/musetric/lab/pull/71), [#83](https://github.com/musetric/lab/pull/83), [#99](https://github.com/musetric/lab/pull/99), [#100](https://github.com/musetric/lab/pull/100), [#118](https://github.com/musetric/lab/pull/118); the GPU defects the graphs work around: [musetric/musetric#975](https://github.com/musetric/musetric/issues/975), [#886](https://github.com/musetric/musetric/issues/886), [#881](https://github.com/musetric/musetric/issues/881).
