# Whisper large-v3-turbo → word-timestamped ONNX (q4)

Exports **openai/whisper-large-v3-turbo** to the [transformers.js][tjs] ONNX layout
with the cross-attention **alignment heads** baked into `generation_config.json`,
so the decoder emits word-level timestamps. Quantized to **q4** and published to
Hugging Face for the `@musetric/ai` (`packages/ai`) WebGPU runtime.

This is a different toolchain from `../roformer/` (which exports the separation
core from the lab's own torch model). Here `convert.py`, `quantize.py` and
`extra/whisper.py` are **vendored from [huggingface/transformers.js][tjsconv]**
(Apache-2.0); only the whisper code path is exercised.

## Environment

Pinned deps conflict with the lab's own `transformers`, so the export runs
in an **isolated** uv environment (never `uv sync` these into the project):

- `requirements.txt` — the pinned export stack (Python 3.11).

## Export

Run from the lab root. `--output_attentions` selects the custom ONNX config
that outputs cross-attentions; the task is left as `auto` (whisper resolves to
`automatic-speech-recognition-with-past`, which the alignment-head config needs).

```bash
PYTHONPATH=scripts/onnx uv run --no-project --python 3.11 \
  --with-requirements scripts/onnx/whisper/requirements.txt \
  python -m whisper.convert \
  --model_id openai/whisper-large-v3-turbo \
  --quantize --modes q4 \
  --output_attentions \
  --skip_validation \
  --output_parent_dir tmp/whisper-export
```

Output lands in `tmp/whisper-export/openai/whisper-large-v3-turbo/` with fp32 and q4
graphs under `onnx/` plus the tokenizer/config JSON. The runtime uses only:

```
config.json  generation_config.json  preprocessor_config.json
tokenizer.json  tokenizer_config.json  vocab.json  merges.txt
added_tokens.json  special_tokens_map.json  normalizer.json
onnx/encoder_model_q4.onnx  onnx/decoder_model_merged_q4.onnx
```

`stage_whisper.py` copies exactly those into the clone of the model repository,
`../models/whisper-large-v3-turbo-onnx/` of the workspace.

## Make the encoder run on mobile GPUs (required)

Two independent defects stand between the stock export and a phone, and neither
is in the weights. `mobile_encoder.py` applies all three rewrites in order on a
single load of the graph, then checks that the result is actually clear:

```bash
uv run python scripts/onnx/whisper/mobile_encoder.py   --input  tmp/whisper-export/openai/whisper-large-v3-turbo/onnx/encoder_model_q4.onnx   --output tmp/whisper-export/openai/whisper-large-v3-turbo/onnx/encoder_model_q4.onnx
```

It is idempotent: a second run over an already-prepared graph reports that each
pass has nothing to do and writes the same bytes. `stage_whisper.py` runs the
same check before it copies anything, so an encoder that skipped this step
cannot reach the clone of the model repository.

### The graph is too big to bind

The exported encoder computes each layer's attention in one shot, so its score
tensor is `(20, 1500, 1500)` fp32 - 171.7 MiB. WebGPU guarantees only 128 MiB
per storage binding and several mobile adapters offer exactly that minimum, so
the encoder cannot run there at all; ONNX Runtime fails the `Softmax` dispatch
with `binding index 1 not present in the bind group layout`.

`block_attention.py` splits the queries into blocks and concatenates the
per-block outputs. Softmax normalizes each query row over the full key axis on
its own, so this is exact, not an approximation. The default 250-row block caps
the score tensor at 28.6 MiB and grows the graph from 1751 to 2423 nodes. No
weight is touched - the q4 `MatMulNBits` nodes are copied through - and CPU
output is bit-identical to the unblocked graph. The pass refuses to write a
graph whose blocks would still exceed the 128 MiB floor.

### The graph computes the wrong answer

The second defect is not a limit but a silent wrong answer. On Adreno 600-series
adapters the WebGPU `Transpose` returns a quarter of its output as zeros. The
cause is in the execution provider's shader: it stages the tile in
`array<array<T, tile_size + 1>, tile_size>` - 16 x 17 f32, 1088 bytes - and on
that hardware a workgroup array whose size is not a multiple of 512 bytes loses
the stores of one warp out of four across `workgroupBarrier`. The stores land, a
thread reads its own slot back correctly, but the other threads do not see them.

Only the shapes that reach that shader are affected: after leading extents of
one are squeezed away, a rank-2 permutation swapping both axes. In this graph
that is one node out of 161 - plus both `Conv` nodes, because the provider
transposes NCHW to NHWC inside them. The defect and every graph it affects are
kept in [musetric/musetric#975](https://github.com/musetric/musetric/issues/975).

`conv_to_matmul.py` writes the convolution as `Pad -> Slice x3 -> Concat ->
Reshape -> MatMul -> Add -> Reshape` with the weight on the left, so no
transpose appears at all. `transpose_to_gather.py` replaces the remaining
affected node with a `Gather` against a precomputed index table - 7.3 MiB of
int32 for this graph - and leaves the other 160 transposes alone.

Together: 2423 -> 2441 nodes and CPU output unchanged to 9e-5, while the
affected adapter goes from visibly wrong output to what a healthy one reports.

The index table is static, so the rewrite pins the encoder to batch 1 - which is
what the runtime feeds anyway, one 30-second window at a time.

## Make the decoder compute correctly on mobile GPUs (required)

The WebGPU `MatMul` kernel that packs its operands into vec4 is taken whenever
the inner and the last extent are both multiples of four. On Adreno 600-series
adapters it returns garbage when the left operand has few rows and the last
extent is not a multiple of 32 - in float16 and float32 alike. Every decoder
attention score is such a product: `Q [heads, 1 or prompt, 64]` against
`K^T [heads, 64, keys]`, so self-attention breaks on every fourth step and
cross-attention (1500 keys) on every step, and the cache carries the error on.

`mobile_decoder.py` runs `pad_attention_scores.py` over both merged decoders: it
appends one zero column to the query and one zero row to the transposed key, so
the product runs with an inner extent of 65 and the provider takes its scalar
kernel. The added row and column multiply to zero, the output keeps its shape,
and the CPU output is bit-identical; `stage_whisper.py` refuses a decoder
without it.

The padding is a constant, and that is the point. The earlier form padded the
key length instead, which has to be read at run time (`Shape -> Gather -> Mod ->
Equal`); shape arithmetic runs on the CPU provider, so it split each step's GPU
work and forced a synchronisation around every attention. Measured on 30 s of
audio: 64.1 ms per token on a desktop NVIDIA against 37.0 ms with the constant
form and 33.5 ms with no padding at all, and 36 queue submissions per token
against 23. On the Adreno 660 the same change takes a token from 324.9 to
266.3 ms with the text unchanged.

```bash
uv run python scripts/onnx/whisper/mobile_decoder.py \
  --input  ../models/whisper-large-v3-turbo-onnx/decoder_model_merged_fp16.onnx \
  --output ../models/whisper-large-v3-turbo-onnx/decoder_model_merged_fp16.onnx
```

## Narrow the cross-attention outputs

`generate` asks the decoder for the cross-attention weights of every layer and
gets all twenty heads of all four: `[batch, 20, tokens, 1500]` each, cast to
float32 on the way out. Word times read six of those eighty head slices, the
ones `alignment_heads` names in the generation config; the rest is copied off
the device and thrown away.

`trim_cross_attentions.py` narrows each output to the heads that config names -
here one, one, two and four - and rewrites `alignment_heads` to the new
positions. Layers no head is taken from keep one, because the pipeline
concatenates one tensor per decoder layer before it indexes them. The CPU output
is bit-identical on both branches, and so are the word times on a GPU.

Measured on 30 s of audio: on the Adreno 660 a token goes from 197.2 to
174.0 ms and the window from 34.8 to 32.9 s; on the Adreno 750 and on a desktop
NVIDIA the step is unchanged and only the JavaScript that assembles the weights
gets cheaper, by 185 ms and 55 ms per window.

```bash
uv run python scripts/onnx/whisper/trim_cross_attentions.py   --input  ../models/whisper-large-v3-turbo-onnx/decoder_model_merged_fp16.onnx   --output ../models/whisper-large-v3-turbo-onnx/decoder_model_merged_fp16.onnx   --generation-config ../models/whisper-large-v3-turbo-onnx/generation_config.json
```

The config is rewritten in place, so the pass runs once per repository: a second
run over an already narrowed graph would remap heads that are no longer there.
The q4 decoder takes the same pass with a copy of the original config, which
comes out identical and is dropped.

## Pack what a step reads into one output

On the WebGPU execution provider every output read back from the device is its
own copy, queue submission and wait, about 3 ms on a desktop NVIDIA whether it
holds a hundred values or a hundred thousand. A generation step reads the
logits and the cross-attention heads the word times use.

`pack_step_outputs.py` appends `step_outputs`, those tensors flattened and
concatenated, and leaves the original outputs in place. A caller that fetches
only the packed output and the key/value cache reads one tensor per step: on a
30 s window a step goes from 31.3 ms with five tensors to 18.6 ms with one, the
text and the word times unchanged. On CPU the packed tensor equals its pieces.

```bash
uv run python scripts/onnx/whisper/pack_step_outputs.py   --input  ../models/whisper-large-v3-turbo-onnx/decoder_model_merged_fp16.onnx   --output ../models/whisper-large-v3-turbo-onnx/decoder_model_merged_fp16.onnx   --generation-config ../models/whisper-large-v3-turbo-onnx/generation_config.json
```

## Split the decoder into a flat step and cross K/V

The merged decoder keeps its first step and its cached step as the two branches
of an `If`, and ONNX Runtime fuses nothing through it: on the WebGPU provider a
cached step costs 20.1 ms merged and 10.4 ms flat on a desktop NVIDIA, 49.7 and
30.9 ms on an Adreno 750, outputs kept on the device in both.

`flatten_decoder_step.py` writes the decoder with the `If` replaced by the
cached branch, keeping `encoder_hidden_states` as an unread input so the
pipeline still hands it over, and a `cross_kv` graph with the encoder states'
key and value projections of every layer. The runtime recognizes the step by
the missing `use_cache_branch` input, runs `cross_kv` once per window and
feeds the prompt one token at a time; the parity reference does the same. On
CPU that matches the merged first step to 0.004 at a logit scale of 21 and a
cached step to 0.006, fp16 rounding, with the same argmax.

The pass reads the projection weights of `MatMul` nodes, so it takes the fp16
decoder only; the q4 decoder, whose projections are `MatMulNBits`, stays
merged.

```bash
uv run python scripts/onnx/whisper/flatten_decoder_step.py   --input  ../models/whisper-large-v3-turbo-onnx/decoder_model_merged_fp16.onnx   --step   ../models/whisper-large-v3-turbo-onnx/decoder_model_merged_fp16.onnx   --cross  ../models/whisper-large-v3-turbo-onnx/cross_kv_fp16.onnx
```

## Keep the step's small products out of the float16 subnormal range

Adreno GPUs compute float16 without subnormals: a product below 2^-14 is zero
even when both factors are normal numbers (2^-7 times 2^-8 is zero there), and
a stored value below 2^-14 reads as zero. Three products of a step are sums of
exactly such terms - the attention probabilities times the values, the
attention output times `out_proj`, and the GELU tail times `fc2` - and on an
Adreno 750 each loses a fifth to a half of its result while NVIDIA, which
keeps subnormals, stays within 1-3 % of the CPU. That is musetric#881: logits
up to 2.3 from the CPU at a scale of 23 against 0.35 on the desktop, and word
times moving with them. The cross-attention softmax loses on top of it the
probabilities below 2^-14, 1-3 % of a head's mass and up to a fifth of the
layer 0 context.

`scale_residual_stream.py` scales what those products read by 2^10, exactly,
and folds the scale away where it would be seen: the residual stream carries
2^10 x (positional table, one `Mul` after the token embedding, the `out_proj`
and `fc2` biases and the `fc2` weights), the self-attention values carry 2^10 v
(`v_proj`), each layer norm becomes one `LayerNormalization` with its epsilon
times 2^20 (ONNX Runtime's pattern fusion reads a float16 epsilon back as
1e-5 on the CPU provider, an explicit node keeps it), and the cross-attention
softmax runs in float32, feeds `cross_attentions.*` as float32 and scales its
probabilities by 2^10 before the float16 cast the value product reads. The
final layer norm removes the scale, so the logits and `cross_kv_fp16.onnx`
are untouched. On the recorded inputs of one step, an Adreno 750 goes from
2.30 to 0.34 from the CPU in logits, the desktop's 0.36; on CPU the scaled
step matches the unscaled one to 0.03 at a scale of 29 with the same argmax
on every step of a 30 s window.

```bash
uv run python scripts/onnx/whisper/scale_residual_stream.py   --input  ../models/whisper-large-v3-turbo-onnx/decoder_model_merged_fp16.onnx   --output ../models/whisper-large-v3-turbo-onnx/decoder_model_merged_fp16.onnx
```

The published set is then the q4 encoder, the fp16 step decoder with
`cross_kv_fp16.onnx`, the merged q4 decoder and the JSON; the passes run in
this order: `mobile_decoder.py`, `trim_cross_attentions.py`,
`pack_step_outputs.py`, `flatten_decoder_step.py`,
`scale_residual_stream.py`.

### Running a single pass

Each pass is still its own script with the same `--input`/`--output` interface,
which is what you want when bisecting a graph or trying a different block size:

```bash
uv run python scripts/onnx/whisper/block_attention.py --input X --output Y --query-block 250
uv run python scripts/onnx/whisper/conv_to_matmul.py --input X --output Y
uv run python scripts/onnx/whisper/transpose_to_gather.py --input X --output Y
uv run python scripts/onnx/whisper/pad_attention_scores.py --input X --output Y
uv run python scripts/onnx/whisper/trim_cross_attentions.py --input X --output Y --generation-config G
uv run python scripts/onnx/whisper/pack_step_outputs.py --input X --output Y --generation-config G
uv run python scripts/onnx/whisper/flatten_decoder_step.py --input X --step Y --cross Z
uv run python scripts/onnx/whisper/scale_residual_stream.py --input X --output Y
```

### Finding the graph a provider gets wrong

The parity run of Musetric compares each graph on wasm and on WebGPU from the
same inputs ([`measure:parity`](https://github.com/musetric/musetric/blob/main/packages/ai/readme.md#parity-run)).
In a page that drives the whole pipeline, transformers.js also takes `device`
as a map keyed by the graph's file name, so one graph moves to another
provider while the other stays:

```js
await pipeline('automatic-speech-recognition', modelId, {
  device: { encoder_model: 'webgpu', decoder_model_merged: 'wasm' },
});
```

The map decides only while `session_options.executionProviders` is unset:
transformers.js fills the providers from `device` only when they are missing,
and the runtime of Musetric sets them for every graph. The graph whose move
changes the output is the one the provider gets wrong, and its passes are then
bisected one at a time as above.

### Writing a pass

A pass keys the nodes it replaces by their position in `graph.node`, as
`conv_to_matmul.py` and `transpose_to_gather.py` do, not by `id(node)`. The
`id()` of a protobuf node's Python wrapper is recycled between loops over
`graph.node`, so an `id()` recorded in one loop can name an unrelated node in
the next, and the rewrite replaces that node without an error.
`block_attention.py` keys by `id()` and holds every node it matched in its
list of chains until the rewrite is done.

## Publish

The staged files are published from the clone through git, as the
[practice of models](https://github.com/musetric/headquarters/blob/main/practices/models.md)
describes; the server pins the size and sha256 of each file in
`packages/server/src/analysis/models.rs`. The repo is a **flat** layout (the q4
graphs sit next to the config, not under `onnx/`); the runtime loads them with
`subfolder: ''`.

[tjs]: https://github.com/huggingface/transformers.js
[tjsconv]: https://github.com/huggingface/transformers.js/tree/main/scripts
