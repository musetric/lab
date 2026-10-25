# Converting a PyTorch Separator to ONNX

The lab keeps the torch implementation as the reference path and CLI
default. The files in this directory are offline tooling for exporting the
neural network core to ONNX and manually validating that core through Python
`onnxruntime`.

One graph is published, and the recipe below is the only way it is built: the
blocked, all-fp16 core at T = 1100 is what `musetric` loads on desktop and on a
phone alike.

## Export Boundary

Only the pure neural network goes into the ONNX graph:

| method | role | runtime |
|---|---|---|
| `encode_stft(raw_audio) -> stft_repr` | host front-end | torch/Python |
| `net_forward(stft_repr) -> masks` | neural network core | ONNX |
| `decode_istft(stft_repr, masks) -> audio` | host back-end | torch/Python |

STFT, iSTFT, complex tensor handling, chunking, normalization, overlap-add, and
mask scatter stay outside the ONNX graph. The regular `forward` still calls the
same stages and remains the torch reference.

Attention uses the matmul path during export because it is portable across
onnxruntime execution providers. The normal torch path remains unchanged.

## Prebuilt Artifacts (published)

You do not have to rebuild anything. The core is published (Apache-2.0, from
[`Aname-Tommy/Mel-Band-Roformer_Duality`](https://huggingface.co/Aname-Tommy/Mel-Band-Roformer_Duality))
at <https://huggingface.co/musetric/aname-mel-band-roformer-duality-onnx>:

| File | SHA256 |
|---|---|
| `duality_core_t1100.onnx` | `1ae7a97c87d854cc27259a5be5115adfe8e44d046390819a6b2fe259465f5a77` |
| `duality_core_t1100.onnx.data` | `ba2a1daacde1608a57564c7bb24a3efe2f50388b2168143044019a6cbe3f21c6` |

Download both files into `tmp/models` (the `.data` file must sit next to its graph):

```bash
uv run hf download musetric/aname-mel-band-roformer-duality-onnx \
  duality_core_t1100.onnx duality_core_t1100.onnx.data \
  --local-dir tmp/models
```

T = 1100 is the published window.

## Install Tooling

The ONNX build dependencies live in the `export` dependency group, not in the
default install. `onnxruntime-gpu` (CUDA / DirectML / CPU execution providers)
ships in the default install, so syncing the group is enough to run or validate
the exported graph:

```bash
uv sync --group export
```

## Build a Core

Five steps: export, re-tree the wide `Concat`/`Split` nodes, audit the epsilon,
audit the dispatch rows, and fuse the node chains onnxruntime-web runs as one
kernel each (see below). The checkpoint and config are the ones
`musetric_toolkit/common/envs.py` downloads (`duality_v1.ckpt` and
`config_v1.yaml` at the pinned revision).

```bash
uv run --group export python scripts/onnx/roformer/build_full_onnx.py \
  --checkpoint tmp/models/mel_band_roformer_duality/model.ckpt \
  --config tmp/models/mel_band_roformer_duality/config.yaml \
  --output tmp/models/core_t1100.onnx \
  --core-only --fuse-rmsnorm --attn-block 128 --all-fp16 --frames 1100 --skip-gate \
  --split-rows 2 --stream-slabs 4

uv run --group export python scripts/onnx/roformer/split_concat_webgpu.py \
  --input tmp/models/core_t1100.onnx \
  --output tmp/models/duality_core_t1100.onnx

uv run --group export python scripts/onnx/roformer/fp16_epsilon_audit.py \
  tmp/models/duality_core_t1100.onnx

uv run --group export python scripts/onnx/roformer/dispatch_rows_audit.py \
  tmp/models/duality_core_t1100.onnx

uv run python scripts/onnx/roformer/fuse_webgpu_kernels.py \
  --model tmp/models/duality_core_t1100.onnx \
  --out tmp/models/fused/duality_core_t1100.onnx
```

The fused graph reads the same `.onnx.data`; put it next to the output.

What each flag is for:

- `--core-only` exports `stft_repr -> masks` rather than the full
  `raw_audio -> vocals` graph, because the host owns the STFT/iSTFT.
- `--attn-block 128` caps the attention score tensor of a slab at
  `[15, 8, 128, T]` instead of `[15, 8, T, T]`. Exact — softmax normalizes each
  query row over the full key axis on its own — and it is what keeps the graph
  inside a mobile storage-buffer binding, which stops working past 256 MiB
  whatever limit the adapter declares. The time attentions hold that block in
  fp32 (below), 64.5 MiB, the size the fp16 block of 256 rows had.
- `--fuse-rmsnorm` gives the normalization a single `ai.onnx::RMSNormalization`
  kernel that accumulates the sum of squares in `f32` inside the shader.
- `--all-fp16` drops the fp32 pins, which the fused RMSNorm makes safe. Without
  it the `[T, 60, 1536]` activations become 387 MiB fp32 tensors with a cast copy
  each at T = 1100.
- `--split-rows 2` shortens the longest dispatches; see below.
- `--stream-slabs 4` runs every transformer over four slabs of the stream, so
  onnxruntime shares buffers between the time and band transformers and a run
  holds less than half the GPU memory; see below.
- `split_concat_webgpu.py` re-trees wide `Concat`/`Split` to <=8-wide so every
  shader stays at <=9 storage buffers, under the strictest shipping cap
  (Dawn/Metal on macOS reports `maxStorageBuffersPerShaderStage = 10`).

## Keep Row-Dispatched Kernels Under 65535 Rows

The exporter applies this on every build; there is no flag. Several onnxruntime
WebGPU kernels run one workgroup per row and have no bounds check, because their
shared-memory reduction needs `workgroupBarrier()` in uniform control flow: the
normalizations, `Softmax`, `LpNormalization`, `TopK`, `InstanceNormalization`
and the `Reduce*` family. Past `maxComputeWorkgroupsPerDimension` rows (65535,
the WebGPU default and what Dawn/Metal reports) the EP turns the dispatch into a
`ceil(sqrt(rows))^2` square, and the spare workgroups write past the end of the
output. Dawn on Metal clamps that index to the last element:

- with the default bucketed storage cache the buffer is rounded up and the stray
  writes land in the slack, so nothing shows;
- with `storageBufferCacheMode: 'simple'` the buffer has the exact size and the
  last row is overwritten. `RMSNormalization` returns `+inf` there and attention
  spreads it as NaN over every mask; `Softmax` returns silently wrong values.

At T = 1100 the transformer norms see `[60, 1100, 384]` and `[1100, 60, 384]`,
66000 rows each, and the band-attention softmax sees `[1100, 8, 60, 60]`, 528000
rows. The exporter wraps every RMSNorm so it chunks its batch axis once the rows
pass the cap (two chunks here) and sets `Attend.softmax_rows`, which chunks only
the score tensor ahead of the softmax (nine chunks). Both are exact: on
Chrome/Metal the rebuilt core returns masks bit-identical to the previous one
with the default cache, now with `'simple'` too, and on D3D12 and on Adreno,
where the stray writes leave the output intact, the two cores agree bit for bit.
The weights file does not change. The cost is mostly memory: under `'simple'`
the buffer cache keeps a buffer of every new chunk size for the whole session,
about 270 MB of peak GPU memory, while wall clock moves by 2-5 %. Chunking q/k/v
instead of the scores measured 9 %.

`export()` refuses to write a graph where such a node sees more rows or has no
static input shape, and `dispatch_rows_audit.py` re-checks any artifact. The
`Reduce*` count is an upper bound: onnxruntime runs some reductions through a
naive kernel that has the guard.

## Keep the Rotary Tables in fp32

The exporter applies this on every build; there is no flag. The rotary
embedding exports its angle cache, position times frequency, as an initializer
that the graph slices and feeds to `Cos` and `Sin`. The fp16 conversion would
store those angles in fp16, whose step is 0.5 rad near the end of an 1100-frame
window, and every rotation there would come out wrong. `fold_rotary_tables`
computes the cosines and sines in fp32 before the conversion and stores them as
constants; values in [-1, 1] keep 5e-4 in fp16.

On the loudest 1100-frame window of every CC BY and CC BY-SA track of
[JamendoLyrics](https://github.com/f90/jamendolyrics), thirteen songs, an
export of `Aname-Tommy/Mel-Band-Roformer_Duality` scores 40.4-53.1 dB against
the torch model with the angles in fp16 and 63.7-71.0 dB with the tables
folded. Cores exported before this change carry the angles in fp16.

## Keep the Band Split in fp32

`--all-fp16` applies this; there is no flag. The fp16 conversion casts
`stft_repr` to fp16 before the band gather. Adreno GPUs read every fp16
subnormal, below 6.1e-5, as zero, and round the cast from fp32 toward zero,
while 12-18 % of the values of a music STFT lie in that range. Each band-split
`RMSNormalization` then scales its band row to unit rms, so the zeroed bins of a
quiet band become errors as large as its features: the masks of the Adreno 660
and 750 stayed 20-25 dB from onnxruntime's CPU provider against 38-42 dB on the
desktop, and further the quieter the input (musetric/musetric#971).
`keep_band_split_fp32` removes that cast, so the band gather, its reshapes and
the 60 band norms run in fp32, and casts each norm's unit-scale output to fp16
for the band `Linear`.

Those norms also take `RMSNORM_EPS` (1e-12) instead of the fp16 epsilon. At
1e-9 a band row whose rms is below sqrt(1e-9) = 3.2e-5 is damped against the
model's `F.normalize`, which alone kept the masks of the CPU provider 26-36 dB
from torch; at 1e-12 they are 55-62 dB from it.

Masks of four 1100-frame units of Rxbyn, "Bad Side"
(<https://www.jamendo.com/track/1556580/bad-side>, CC BY 3.0, from
JamendoLyrics): 60-80 s, the same at half level, 20-26 s, and the first unit of
the whole track. WebGPU with the app's provider and session options, against
the CPU provider of the same graph / against torch:

| Core | Desktop NVIDIA | Adreno 750 and 660 |
|---|---|---|
| band split in fp16 | 37.6-41.6 dB / 25.7-33.5 dB | 20.0-25.5 dB / 19.8-26.3 dB |
| band split in fp32 | 37.6-42.0 dB / 37.5-42.1 dB | 34.9-39.8 dB / 34.8-39.7 dB |

The Adreno 750 and the Adreno 660 return the same bits, and the time per unit
does not change.

## Keep the Time Attention's Probabilities in fp32

`--all-fp16` applies this; there is no flag. A time attention normalizes its
scores over the T keys of the window, so most of its probabilities lie far below
1 / T, and in fp16 every one below 6.1e-5 is subnormal. Adreno GPUs read those as
zero and drop every product below 2^-14 inside the matmul with the values, so
each time attention loses the tail of its probabilities, the same way in every
layer. The desktop computes the same fp16 graph with subnormals; on the Adreno
660 and 750 the masks stayed 1.8-3.3 dB further from onnxruntime's CPU provider
(musetric/musetric#988). A softmax in fp32 alone does not help while its
probabilities are stored in fp16, and scaling them by 2^10 before the fp16 cast,
as the Whisper decoder does, closes about half of the gap: the matmul with the
values needs them in fp32.

`keep_time_attention_fp32` casts the scores of every attention whose softmax
runs over T keys to fp32, takes the softmax and the matmul with the values in
fp32, and casts the result back to fp16. The attention's scale, 2^-3, moves from
the scores onto the queries, which is exact and leaves no fp16 copy of the
scaled scores. The band attentions normalize over 60 bands and stay in fp16.
`split_concat_webgpu.py` unwraps only a softmax with a Cast on both sides, so it
leaves these alone.

The same four units against the CPU provider of the same graph:

| Core | Desktop NVIDIA | Adreno 750 and 660 | Apple M3 (iPad Air) |
|---|---|---|---|
| time attention in fp16 | 37.6-42.0 dB | 34.9-39.8 dB | 37.3-41.9 dB |
| time attention in fp32 | 43.2-45.1 dB | 43.5-44.6 dB | 42.7-45.3 dB |

Every device moves closer to the CPU provider, and the phones now stay within
0.7 dB of the desktop on every unit. Against torch on the model card's 13
JamendoLyrics windows, both on CUDA, the core scores 61.2 / 66.7 / 71.0 dB
(worst / median / best), against 61.5 / 67.0 / 71.0 dB before.

The cost: onnxruntime holds 1913 MiB of GPU buffers at the peak against 1767 MiB,
and the whole GPU of the Adreno 750 and 660 holds 2484 and 2434 MiB against
2282 MiB. In the app, paced and with graph capture, a chunk takes about 10 %
longer on the desktop, 8 % on the Adreno 750 and 3-4 % on the Adreno 660. The
first chunk of a load, which compiles the shaders, holds the page somewhat
longer: at worst 2.8, 3.1 and 5.3 s on those devices against 2.5, 2.6 and 5.0 s,
and 5.0-7.1 s on the iPad against 2.1-3.0 s. The iPad tab still runs a 23-chunk
track, at most 4927 MiB.

## Shorten the Longest Dispatches

`--split-rows N` and `--split-projections N` are aimed at mobile GPUs. The
published core used `--split-rows 8 --split-projections 4` until it moved to
stream slabs (below), whose slabs already cut every matmul of a layer to a
quarter of its rows, so it now needs `--split-rows 2` alone. A run of
this core is
dominated by a handful of very long matmuls: the feed-forward projections and
the fused qkv projection of each layer are each a single dispatch tens of times
longer than the median one. Nothing at the runtime level can break those up -- a
flush threshold only decides how many dispatches ride in one submit, never how
long one of them runs -- so the compositor is locked out for as long as the
longest one takes, and that is what a freeze is.

The flag chunks every feed-forward over `N` row groups and replaces the fused
qkv projection with the three projections the following rearrange splits it into
anyway:

```bash
uv run --group export python scripts/onnx/roformer/build_full_onnx.py \
  --checkpoint tmp/models/mel_band_roformer_duality/model.ckpt \
  --config tmp/models/mel_band_roformer_duality/config.yaml \
  --output tmp/models/core_split4_t1100.onnx \
  --core-only --fuse-rmsnorm --attn-block 64 --all-fp16 --frames 1100 \
  --skip-gate --split-rows 4
```

`--split-rows` leaves the four projections of every attention -- q, k, v and
the output -- at full height: `[T * 60, 384] x [384, 512]` and its reverse, 66000
rows at T = 1100. Once the feed-forwards are chunked, those are the longest
dispatches of a run, about 200 ms each on Adreno 660. `--split-projections N`
chunks each of them into `N` row groups the same way. With
`--split-rows 8 --split-projections 4` the worst wait of another GPU client
during a paced run on Adreno 660 falls from about 650 ms to about 440 ms for
2-4 % of run time; finer splits cost more on the desktop without a steady gain
on the phones. The measurements are in musetric/musetric#894.

Both rewrites are exact in fp16, because rows of a matmul are independent and
the qkv split only skips a fusion. Splitting the *reduction* axis instead --
which is the obvious way to cut the same matmul -- is not exact in fp16: it
reorders the sums and costs roughly 66 dB a layer. Doing it on the exported
graph with `Split`/`Concat` nodes is exact but materializes the wide activation
twice, which costs several hundred MB of peak memory; doing it here costs none,
because the wide intermediate is never built at full height.

Run the same validation as any other core afterwards. Peak GPU memory falls
rather than rises, and total time moves by about a percent.

## Run the Transformers over Stream Slabs

`--stream-slabs N` runs the time transformer of every layer over `N` slabs of
the bands and the band transformer over `N` slabs of the frames. Each slab is cut
straight from the `[1, T, 60, 384]` stream, runs through attention, feed-forward
and the output norm as `[1, rows, 384]`, and goes back by one `Concat`. Nothing
in a transformer layer reduces over its batch axis, so the slabs change no
arithmetic: on WebGPU, on the desktop NVIDIA and on the Adreno 750 and 660, the
masks are bit for bit those of the core without them. onnxruntime's CPU
provider picks its matmul kernels by shape, and there the two cores differ by
fp16 rounding, 60 dB.

The point is onnxruntime's allocation planner. It hands a freed buffer only to a
later tensor of the same static shape and keeps it for that shape until the run
ends, so every distinct large shape holds a pool of its own for the whole run.
Without slabs the two transformers of a layer see `[60, T, 384]` and
`[T, 60, 384]`, the same size in different shapes, plus the transposed and
packed copies between them, and those pools sit idle in turn. With `N` dividing
both 60 and `T` a band slab and a frame slab hold the same rows, 16500 at
`N = 4` and T = 1100, so the norms, projections, feed-forwards and residual adds
of both transformers produce the same shapes and share their buffers; only the
`[b, h, n, d]` view inside attention differs. The stream stays 3-D because a
`Linear` on a 2-D input exports as `Gemm`, whose fused bias changes the bits.

`--split-rows 2` inside a slab leaves feed-forward row groups of 8250 rows, as
before.

A run of the published core at T = 1100 with onnxruntime-web 1.30 on WebGPU,
`storageBufferCacheMode: 'simple'` as the app sets it, on a synthetic input:

| | without slabs | `--stream-slabs 4` |
|---|---|---|
| GPU buffers onnxruntime holds at the peak, RTX 3060 Laptop | 3747 MiB | 1767 MiB |
| Whole-GPU memory during a run (`dumpsys gpu`), Adreno 750 | 4280 MiB | 2282 MiB |
| Whole-GPU memory during a run (`dumpsys gpu`), Adreno 660 | 4358 MiB | 2282 MiB |
| Footprint of the Safari tab on iPad Air (M3), killed at 5120 MiB | killed in the first run | a 23-chunk track, 4898 MiB at most |

The weights account for 436 MiB of each count. The iPad figure is the app's
run with onnxruntime's graph capture. The tab holds 2.4 GiB through most of the
track; the peak comes in its first one or two minutes, when WebKit keeps about
2 GiB more and then gives it back on its own, and where that memory comes from
is still open. The time per
chunk in the app stays within the spread of alternating loads on the desktop,
the Adreno 750 and the Adreno 660; building the session takes about a second
longer, since the graph is larger.

Four is the count that keeps the time. Five slabs (`--attn-block 320`) hold
1641 MiB, but a chunk of the app takes 26 s on the Adreno 750 against 16 s
without slabs. Two slabs (`--attn-block 128 --split-rows 4`) keep the time but
hold 2261 MiB, and the iPad tab peaks at 4953 MiB, too close to its limit.
The figures in this section are from before the time attentions moved to fp32
(above).

## Keep the Published Weights File

A re-export of weights that are already published lays them out in another
order, and the row-chunked projections keep their weight transposes as
`Transpose` nodes that the exporter otherwise folds into constants. The weights
are the same, but the `.onnx.data` is not, and an app that pins the new revision
downloads the weights again. Such a rebuild ends with one more step; the first
export of new weights does not need it:

```bash
uv run python scripts/onnx/roformer/reuse_external_data.py \
  --model tmp/models/duality_core_t1100.onnx \
  --reference published/duality_core_t1100.onnx \
  --out tmp/models/release/duality_core_t1100.onnx
```

`reuse_external_data.py` folds each such transpose into the matrix the
published core stores and points every tensor of the new graph at the offset of
the same bytes in the published weights file. It refuses when any tensor is not
found there. Only the `.onnx` then changes between revisions.

**The epsilon is dtype-dependent and the audit is not optional.** In fp16 the
`RMSNormalization` reciprocal `1/sqrt(mean(x^2) + eps)` is cast back to fp16, so
any row under `1/65504^2 = 2.33e-10` becomes `+inf` and then `0 * inf = NaN`,
which attention spreads over the whole time axis — a silent output, not a
warning. `--all-fp16` therefore emits `RMSNORM_EPS_FP16` (1e-9) instead of
`RMSNORM_EPS` (1e-12); the exporter refuses to write a graph that breaks the
rule, and `fp16_epsilon_audit.py` re-checks any artifact, including ones built
elsewhere. This shipped broken once — `epsilon=1e-12` on all 96 fp16 nodes — and
`packages/mobile/docs/iosWebgpu.md` in the `musetric` repository carries the
device measurements.

## Fuse Node Chains for onnxruntime-web

onnxruntime-web runs every node as one dispatch over its whole tensor, and the
exported core spells four things as chains of nodes:

| what | exported as | rewritten as | count |
|---|---|---|---|
| GELU of the feed-forwards | `Div`, `Erf`, `Add`, `Mul`, `Mul` | `Gelu` | 96 |
| rotary embedding of the queries and keys | `Mul`, two `Slice`s, `Neg`, `Concat`, `Mul`, `Add` | `RotaryEmbedding`, interleaved | 96 |
| projection to heads | `MatMul`, then a `Transpose` that puts the heads ahead of the sequence | one `MatMul` batched over the heads | 144 |
| query scale of the time attentions | a `Slice` and a `Mul` per block of query rows | one `Mul`, then a `Split` | 24 |

onnxruntime runs the GELU chain of this graph node by node. The last step of
**Build a Core** rewrites the chains into those kernels; on a rebuild that keeps
the published weights file, run it on the output of `reuse_external_data.py`.

The weights and their file stay as they are: the batched projection reshapes
the stored weight and the rotary kernel takes every other column of the stored
tables, and onnxruntime folds both at load. A run goes from 4939 dispatches to
3475, and every new node is a standard `ai.onnx` operator (`Gelu` from opset
20, `RotaryEmbedding` from 23). On WebGPU the masks are the same bits as
those of the core without the rewrite on a desktop NVIDIA GPU and on the
Adreno 750 and 660; on Apple's M3 they differ by rounding. onnxruntime's CPU
and CUDA providers round these kernels differently from the chains: 66 dB from
the core without the rewrite on the CPU provider.

Against the core without the rewrite, on 2026-10-04. The GPU time sums the
timestamps of every dispatch of one run; the chunk is musetric's, paced one
submission at a time with graph capture, three loads of each core in A-B-B-A-A-B
order:

| runtime | GPU time of a chunk | chunk p50 | peak GPU memory |
|---|---|---|---|
| RTX 3060 Laptop, Chrome | 1798 → 1611 ms | 2190 → 1900 ms | 2.59 → 2.25 GiB |
| Galaxy S24 Ultra, Adreno 750, Chrome | 13.4 → 12.7 s | 29.9 → 16.5 s | 2.44 → 2.17 GiB |
| OnePlus 9RT, Adreno 660, Chrome | 39.6 → 37.2 s | 70.5 → 38.4 s | 2.47 → 2.25 GiB |
| iPad Air (M3), Safari 26.4 | 14.0 → 12.7 s | 15.0 → 14.2 s | 4.74 → 4.43 GiB |

Under the pacer the Adreno 750's clock governor holds a higher clock for the
fused core, 629 MHz for half of a chunk against 422-500 MHz for two thirds of
one without the rewrite. The Adreno 660 runs a chunk near 40 s or near 70 s as
it heats, so its medians carry the heat; its best chunks fall 41.3 → 38.4 s.
The iPad's chunks spread from 11 to 19 s with heat for either core.

The matmuls are what is left: 60 % of a chunk's GPU time on the desktop, 72 %
on the Adreno 750, 81 % on the Adreno 660 and 66 % on the M3, in
onnxruntime-web's own kernel, which a graph cannot replace.

## Validate

Gates: conversion SNR vs torch >= 40 dB, NaN = 0. The reference I/O comes from
Python:

```bash
# --frames must match the model, and --out-dir is per-T. Large T (>=~901)
# needs --device cuda: the CPU torch forward segfaults on the 2.3 GB T²
# attention sim, while flash SDPA on cuda never materializes it.
uv run python scripts/onnx/roformer/validate_full_onnx.py \
  --checkpoint tmp/models/mel_band_roformer_duality/model.ckpt \
  --config tmp/models/mel_band_roformer_duality/config.yaml \
  --source tmp/sample.flac --out-dir tmp/bench_out_t1100 --frames 1100 --device cuda
```

It writes `full_input.f32` and `full_ref_vocals.f32` into `--out-dir`, planar
little-endian stereo, and prints the offset of the window it chose. Scoring an
execution provider against them takes a comparator that runs the ONNX and
measures SNR; this repository does not carry one.

## Inspect Ops

```bash
uv run --group export python scripts/onnx/roformer/op_audit.py tmp/models/duality_core_t1100.onnx
```

## Run Python ONNX Inference

```bash
uv run --group export python scripts/onnx/roformer/infer_separator.py \
  --model tmp/models/duality_core_t1100.onnx \
  --config tmp/models/mel_band_roformer_duality/config.yaml \
  --source path/to/input.wav \
  --target-output tmp/out/target.flac \
  --residual-output tmp/out/residual.flac
```

The script runs the exported ONNX core through Python `onnxruntime` with the
same host-side torch STFT/iSTFT stages used by the reference model. It is a
manual validation tool and is not wired into the default lab CLI.

## Notes

- Static shapes are intentional: a dynamic time axis is fragile for this
  architecture, because rotary and reshape logic bakes sequence lengths into the
  exported graph, so the window is fixed at build time.
- Repack external-data models before path-loading them with onnxruntime.
- Warm the model once at the export shape before `torch.onnx.export`.
- Delete stale `.onnx.data` files before saving external-data models; the scripts
  do this.
- Keep generated model files out of git unless explicitly publishing artifacts.
