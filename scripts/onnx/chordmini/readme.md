# ChordMini → ONNX/CQT export

This directory exports the vendored **ChordMini** chord recognizer (ChordNet
"2E1D", 170-class large vocabulary) as a **ChordNet/CQT artifact**: a small
`features → logits` ONNX classifier plus the versioned CQT plan that defines its
input features. Feature extraction runs on WebGPU in `@musetric/cqt`, not in the
graph.

All source and destination paths are explicit arguments. The scripts do not
guess folders or reuse a machine-specific cache, so a model cannot silently be
combined with a stale CQT plan.

## The artifact

`export_chordnet.py` produces the classifier boundary used after the WebGPU CQT
pipeline:

```text
features [W, 108, 144]                  # unnormalized log-CQT windows
  → (x − mean) / (std + 1e-8)           # normalization is inside ONNX
  → ChordNet
  → logits [W, 108, 170]
```

`W` is dynamic, while the 108-frame window and 144 CQT bins are fixed model
contract dimensions. CQT extraction, tail padding/windowing, temporal smoothing,
and argmax intentionally remain outside ONNX so the WebGPU runtime can keep the
feature and logits buffers on the GPU. `config.json` records the input/output
names and shapes, checkpoint normalization, and the 170-label vocabulary.

The staged and published file set is:

| file | role |
|---|---|
| `chordnet.onnx` | `features [W, 108, 144] float32 → logits [W, 108, 170] float32` |
| `config.json` | ChordNet input/output and normalization contract |
| `cqt-plan.bin` | little-endian sparse CQT-plan binary |
| `cqt-plan.manifest.json` | plan format, generator provenance, and hashes |

The binary plan and its manifest are published with the classifier so each
release has a hashable model/feature-extraction contract. The generated
TypeScript plan payload is for the `@musetric/cqt` tests; it is intentionally not
copied into the HF artifact.

## Scripts

| script | role |
|---|---|
| `export_chordnet.py` | write `chordnet.onnx` + `config.json` |
| `validate_chordnet.py` | compare deterministic feature-window logits from the exported classifier against the same Torch ChordNet pipeline |
| `generate_cqt_plan.py` | generate the binary CQT plan, its JSON provenance manifest, and the TypeScript payload the `@musetric/cqt` tests use |
| `generate_cqt_reference.py` | generate the librosa peak-magnitude table the `@musetric/cqt` tests measure against |
| `stage_chordnet.py` | stage the ChordNet/CQT bundle under its published names in the clone of `musetric/chordmini-onnx` |

## Build and stage

The ONNX export uses the main lab environment plus the `export` dependency
group (`onnx`). The plan generator uses the pinned lab `librosa`/`numpy`/
`scipy` environment.

```sh
uv run --group export python scripts/onnx/chordmini/export_chordnet.py \
  --out <chordnet-export-dir> --models-path <checkpoint-cache-dir>

uv run --group export python scripts/onnx/chordmini/validate_chordnet.py \
  --onnx <chordnet-export-dir>/chordnet.onnx \
  --models-path <checkpoint-cache-dir>

uv run python scripts/onnx/chordmini/generate_cqt_plan.py \
  --out <cqt-plan.bin> --manifest <cqt-plan.manifest.json> \
  --typescript-payload <packages/cqt/src/cqt/__test__/planPayload.ts>

uv run python scripts/onnx/chordmini/generate_cqt_reference.py \
  --out <packages/cqt/src/cqt/__test__/reference.ts>

uv run python scripts/onnx/chordmini/stage_chordnet.py \
  --chordnet-export <chordnet-export-dir> \
  --cqt-plan <cqt-plan.bin> \
  --cqt-plan-manifest <cqt-plan.manifest.json> \
  --dest ../models/chordmini-onnx
```

`stage_chordnet.py` always renames the two explicit plan inputs to the stable
published names `cqt-plan.bin` and `cqt-plan.manifest.json`. The staged files are
published from that clone through git, as the
[practice of models](https://github.com/musetric/headquarters/blob/main/practices/models.md)
describes.

`validate_chordnet.py` runs three fixed-seed random unnormalized feature windows
through the checkpoint-backed Torch classifier and CPU ONNX Runtime. It checks
the `[3, 108, 170]` contract and fails when the maximum absolute logit error
exceeds `1e-4`.

## Static WebGPU rewrite

`chordnet.onnx` as exported routes 79 of its Transposes to onnxruntime's
shared-tile WebGPU kernel, which computes wrong logits on Adreno 6xx, and its
MatMul shapes hit the scalar MatMul kernel, which the Adreno 750 driver
miscompiles. `capture_shapes.py` records the runtime shapes of the exported
graph and `rewrite_static_adreno.py` replaces those Transposes with
constant-index Gathers, keeps the MatMuls off that kernel, and widens the
classifier weight to a multiple of four columns. The published classifier is
this rewrite:

```sh
uv run --group export python scripts/onnx/chordmini/capture_shapes.py \
  --model <chordnet-export-dir>/chordnet.onnx --input features \
  --frames 108 --windows 16 --shape 16,108,144 --ops Transpose,MatMul \
  --out <shapes.json> --probe <probe.onnx>

uv run --group export python scripts/onnx/chordmini/rewrite_static_adreno.py \
  --model <chordnet-export-dir>/chordnet.onnx --shapes <shapes.json> \
  --out <rewrite-dir>/chordnet.onnx --vec4-attention

uv run --group export python scripts/onnx/roformer/dispatch_rows_audit.py \
  <rewrite-dir>/chordnet.onnx
```

The build is deterministic: the batch-16 rewrite of the `fbd620e6` export comes
out the same on Windows and on macOS alike, and `--windows 1` rebuilds the
earlier batch-1 graph node for node.

### What the MatMul rewrites are for

onnxruntime's WebGPU MatMul runs a vec4 kernel when K and N are multiples of
four and a scalar one otherwise. The Adreno 750 driver miscompiles the scalar
kernel that onnxruntime generates for two- and four-dimensional products, while
the three-dimensional one comes out right (musetric/musetric#891). The exported
graph uses it for the attention, `[16, 8, 108, 108] x [16, 8, 108, 18]` and
`[W * 108, 8, 2, 2] x [W * 108, 8, 2, 9]`, and for the classifier
`[16, 108, 144] x [144, 170]`, deterministically wrong by units there while
wasm on the same device is right. The rewrite keeps them off it:

- every four-dimensional MatMul folds into three dimensions;
- the classifier weight widens to 172 columns, which is the vec4 kernel, and
  the result is sliced back.

Together they are what makes the chord logits match wasm on that GPU: the
published rewrite differs from wasm by 3.13 there, with 386 of 1728 frames
choosing another chord, and this one by 5.7e-6 with no frame changed. On
Adreno 660 and on desktop the rewrites change no result and cost no time.

`--vec4-attention` takes the time attention, the MatMuls of more than eight
rows, to the vec4 kernel instead: K and N are padded with zeros to multiples of
four and the extra output columns are sliced off, so the scalar kernel is left
only for the two-row frequency attention. That one keeps the fold, because
onnxruntime runs a product of eight rows or fewer on the vec4 kernel with one
row per thread, which is wrong on Adreno 660. On 16 windows of real CQT
features against ORT CPU, both graphs differ by 4.8e-6 with no frame changed on
the desktop, the Adreno 750 and the Adreno 660, and the vec4 attention takes
20.7 / 97.8 / 283.4 ms per run against 24.0 / 101.2 / 286.5 ms for the fold;
without the MatMul rewrites the Adreno 750 differs by 2.07 with 94 frames
changed. The published revision `8ab85fc0` is built with the flag; without it
the build reproduces the previous revision `08616241` byte for byte.

The rewrite is exact against the exported graph (max |delta logit| ~5e-6 on ORT
CPU and on WebGPU) but static: every Gather index and Reshape target is baked
for the captured batch, so it runs at `[--windows, 108, 144]` and nothing else.
The rewrite pins the graph input and output to that shape, which also lets
onnxruntime fold the source graph's shape arithmetic instead of evaluating it on
the CPU at every run (411 -> 333 WebGPU dispatches per run).

The runtime feeds a track in groups of `--windows` windows and pads the last
group with zero windows; windows never interact inside the model, so padding
changes no real logit. The batch decides the speed. A WebGPU run pays a fixed
~4 ms of dispatch encoding for the ~333 kernels on top of ~1.5 ms of GPU work
per window (Chrome/Metal, M2 Pro), so batch 1 spends most of its time
re-dispatching:

| batch | 20 windows (~3 min) | 60 windows (~10 min) |
|---|---|---|
| 1 | 82 ms | 242 ms |
| 16 | 52 ms | 102 ms |
| 32 | 50 ms | 98 ms |

Keep the batch at or below 37. The frequency encoder runs Softmax over
`[windows * 108, 8, 2, 2]`, 1728 rows per window, and onnxruntime's WebGPU
Softmax and LayerNormalization kernels have no bounds check once a dispatch
passes 65535 workgroups per dimension: the spare workgroups of the square it
falls back to write past the end of the output. 16 keeps it at 27648 rows with
room to spare, and pads at most 15 windows.

The batch-16 graph was checked on devices against the wasm EP of the same file,
one input of 26 and 60 windows:

- OnePlus 9RT (Adreno 660): max |delta logit| 5.7e-6, no mismatched logit or argmax,
  in both buffer cache modes; the exported graph deviates by 0.75 there;
- desktop NVIDIA (D3D12): max |delta logit| 6.7e-6;
- Galaxy S24 Ultra (Adreno 750): every graph deviated, the exported one as
  much as the rewrite (max |delta logit| 1.99, one argmax of 2808 frames). The cause
  is the provider's `MatMul` on that GPU (musetric/musetric#891), which the
  MatMul rewrites below avoid: the published graph is within 5.7e-6 of
  onnxruntime's CPU provider there, with no argmax changed (the parity run of
  musetric/musetric#974, onnxruntime-web 1.30.0).

Re-check any other batch on the device before publishing it.

## Validating end to end

The classifier check above covers the export, not the features. Only the
TypeScript side can measure the whole pipeline, because the CQT runs on WebGPU;
compare its chord output against the `musetric-chords` CLI on the same audio.

Validate on the material fed in production — the instrumental stem. Agreement
measured on audio where the reference emits a near-constant label (for example,
an isolated vocal where "no chord" is correct) carries no useful information: a
stub returning that label scores just as well.

The checkpoint (`2e1d_model_best.pth`, ~28 MB) is fetched on first run by
`ensure_checkpoint`. ChordMini is vendored under
`musetric_toolkit/chords_audio/chordmini`; see `thirdPartyNotices.md` for source
and license.
