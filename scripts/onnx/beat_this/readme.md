# Beat This! → ONNX export

Exports the **Beat This!** beat/downbeat tracker (the `beat-this` package the
`musetric-rhythm` CLI runs) to ONNX for the
[`musetric`](https://github.com/musetric/musetric) `@musetric/ai` runtime on
**onnxruntime-web**.

Like the chordmini export, the graph covers only the neural boundary; feature
extraction is the host's job:

```
beat_this.onnx  spect [windows, frames, 128] → beat, downbeat [windows, frames]
```

The runtime computes the log-mel on **WebGPU** (`@musetric/fft` STFT + the mel
filterbank exported here) and runs the graph on the WebGPU execution provider,
the same way chords runs its CQT on WebGPU around ChordNet. Chunking and
aggregation also stay in the runtime: they are index arithmetic, not DSP. So
does beat tracking: the runtime ports the Beat This! `dbn` postprocessing,
madmom's `DBNDownBeatTrackingProcessor`, and the parity reference runs the
original through the `madmom` dependency.

One window per session call is not a detail — it is what makes the tracker
runnable. Batching every window into a single call materializes an attention
tensor of `windows × 32 × 1500 × 1500` floats, which is ~2.9 GB on a
five-minute track.

## Why the filterbank ships as a file

`mel-filterbank.bin` is torchaudio's own `MelScale.fb`, written verbatim as
row-major float32 `[n_fft // 2 + 1, n_mels]`. Shipping it means the runtime
reuses the reference filterbank rather than reimplementing the slaney mel scale,
so the front end carries no approximation of its own. This mirrors how the
chordmini export ships `cqt-plan.bin`.

The analysis window is *not* shipped: the frame shader generates a periodic Hann
analytically, and the export fails if the reference window ever stops matching
that (`check_analytic_window`).

## Export-time swap

`PartialFTTransformer.forward` reads its batch size with `b = len(x)`, which
bypasses the tracer and freezes the window count into the graph. The export
swaps in the same forward written with `x.shape`, which traces to `Shape`/
`Gather`. The installed `beat_this` package is left untouched, and the patched
module is bit-identical to the vendored one in torch.

This one is worth remembering: `len()` does not raise, it *succeeds* and yields
a graph that returns wrong-length output for any track whose window count
differs from the traced one. Run the exported graph at several input lengths and
diff the shapes.

## Split does not compile on Adreno

onnxruntime's WebGPU `Split` kernel fails to build a compute pipeline on
Adreno. On a Galaxy S24 the graph loses its very first `Split` with
`CreateComputePipelines failed with VK_ERROR_UNKNOWN`, onnxruntime drops the
session to the CPU provider, and the tracker still returns beats — just far
slower, with nothing in the log naming the cause. This is a shader that will
not compile, not a binding too large, so nothing about tensor sizes helps.

`split_to_slice_webgpu.py` lowers every fixed-size `Split` into one static
`Slice` per output: it reads the sizes off the `Constant` feeding the node and
writes them as `starts`/`ends`/`axes` initializers. Same partition of the same
axis, no weight read, three initializers added per output. Run it on the export
before validating, so every later step sees the graph that ships.

## Adreno WebGPU rewrite

On Adreno 6xx the stock graph returns silently wrong logits: shared-tile
`Transpose`/`Conv` and inference `BatchNormalization`. The rewrite pins the
author's **1500 frames**, replaces the broken Transpose/Conv families with
index tables, and folds BatchNorm to Mul+Add.

The attention needs one more step. Each of the nine attention cores keeps its
scores in one `[b, h, 1500, 1500]` tensor: 137 MiB for the 16 heads of the main
transformer, 275 MiB for the 32 head slices of a frontend block. That does not
fit one storage binding on the phones: the Adreno 750 (Galaxy S24) reports
`maxStorageBufferBindingSize` 128 MiB and returns garbage above it, the Adreno
660 (OnePlus 9RT) reports 512 MiB yet breaks between 137 and 275 MiB; both are
exact up to 120 MiB. `split_attention_webgpu.py` cuts every core along its head
axis into groups of 68.7 MiB and concatenates the results, which changes no
arithmetic. With the split, the whole graph matches the CPU provider within
7.5e-5 on both phones, deterministically, and the two GPUs agree with each
other.

Until 2026-10 the shipped graph used 513-frame windows instead, which kept the
scores small but gave the model a third of the context the author trained it
with; without drums it lost the beat far more often.

```sh
uv run --group export python scripts/onnx/beat_this/capture_shapes.py \
  --model <export-dir>/beat_this.onnx --frames 1500 --out shapes1500.json
uv run --group export python scripts/onnx/beat_this/rewrite_static_adreno.py \
  --model <export-dir>/beat_this.onnx --shapes shapes1500.json \
  --out <export-dir>/beat_this_static.onnx
uv run --group export python scripts/onnx/beat_this/split_attention_webgpu.py \
  --model <export-dir>/beat_this_static.onnx --shapes shapes1500.json \
  --out <export-dir>/beat_this.onnx
```

The host must always feed `[1, 1500, 128]` (pad short clips; long tracks use
overlapping 1500-frame windows).

## Files

| script | role |
|---|---|
| `export_beat_this.py` | build the graph, write `beat_this.onnx`, `mel-filterbank.bin` + `config.json` |
| `split_to_slice_webgpu.py` | lower fixed-size `Split` into static `Slice` so the graph compiles on Adreno |
| `validate_beat_this.py` | beat/downbeat time agreement of the ONNX vs the torch `File2Beats` reference |
| `dump_reference_mel.py` | dump reference waveforms + log-mels, the parity gate for the runtime's WebGPU front end |
| `capture_shapes.py` | pin every intermediate shape at a chosen frame count |
| `rewrite_static_adreno.py` | replace broken Adreno Transpose/Conv/BatchNorm; ship at 1500 frames |
| `split_attention_webgpu.py` | split each attention core into head groups that fit one WebGPU storage binding |
| `stage_beat_this.py` | copy the file set into the clone of `musetric/beat-this-onnx` |

All paths are explicit arguments; the scripts carry no implicit defaults.

## Run

The export reuses the main lab env (it imports `beat_this`) plus the
`export` dependency group (`onnx`):

```sh
uv run --group export python scripts/onnx/beat_this/export_beat_this.py \
  --out <export-dir> --models-path <checkpoint-cache-dir>

uv run --group export python scripts/onnx/beat_this/split_to_slice_webgpu.py \
  --input <export-dir>/beat_this.onnx --output <export-dir>/beat_this.onnx

uv run --group export python scripts/onnx/beat_this/validate_beat_this.py \
  --onnx <export-dir>/beat_this.onnx --audio <audio-dir> \
  --models-path <checkpoint-cache-dir>

uv run python scripts/onnx/beat_this/dump_reference_mel.py \
  --audio <audio-dir> --out <dump-dir> --models-path <checkpoint-cache-dir>

uv run python scripts/onnx/beat_this/stage_beat_this.py \
  --export <export-dir> --dest ../models/beat-this-onnx
```

The staged files are published from that clone through git, as the
[practice of models](https://github.com/musetric/headquarters/blob/main/practices/models.md)
describes.

Validate on the material the model is fed in production — the **instrumental
stem**.

`--models-path` sets `TORCH_HOME`, matching the CLI: the checkpoint
(`beat_this-final0.ckpt`, ~81 MB) is fetched there on first run. The exported
graph is ~83 MB fp32; the filterbank is 257 KB.

The reference downmixes stereo with `signal.mean(1)`, so the runtime must decode
with a mean downmix rather than `ffmpeg -ac 1`, which is √2 louder. `log1p`
features make that gain a non-constant offset; `config.json` records the
`downmix` contract.

Beat This! is used through the `beat-this` package; see `thirdPartyNotices.md`
for source and license.
