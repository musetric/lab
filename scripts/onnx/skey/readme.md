# S-KEY → ONNX export

Exports the vendored **S-KEY** key detector (harmonic VQT + ChromaNet) to a
single self-contained ONNX graph consumed by the
[`musetric`](https://github.com/musetric/musetric) `@musetric/ai` runtime on
**onnxruntime-web**.

Like the chordmini export (and unlike whisper's transformers.js layout), the
graph bakes the whole `run_skey` pipeline, so the runtime feeds raw mono audio
and reads back the 24 key probabilities with no DSP of its own:

```
audio [1, N] @ 22050 Hz mono, peak-normalized
  → nnAudio harmonic VQT (log-amplitude HCQT)   [1, 1, 99, T]
  → crop 84 bins (CropCQT, transpose 0)         [1, 1, 84, T]
  → ChromaNet (ConvNeXt stack → chroma → 24)    [1, 24] (softmax'd)
  → mean over batch → softmax                   [24]
```

The whole track is fed at once (the runner does the same); ChromaNet's time
pooling collapses it to one key.

## Dynamic-axis swaps

Two vendored ops don't export with a dynamic time axis and are replaced by
numerically identical equivalents at export time (the vendored source is left
untouched):

| vendored op | replacement | why |
|---|---|---|
| `nn.functional.layer_norm(x, x.shape[1:])` (affine-free) | explicit mean/var reduction | `normalized_shape` would carry the dynamic time dim |
| `AdaptiveAvgPool2d((12, 1))` | mean over the time axis | input already has H=12 chroma bins; not ONNX-exportable with dynamic width |

Both replacements are numerically identical, so the graph carries **no
approximation** and its parity does not depend on the input material (unlike the
chordmini export, which substitutes librosa's CQT). `validate_skey.py` confirms
the key argmax and confidence against the torch runner.

## Files

| script | role |
|---|---|
| `export_skey.py` | build the `audio → probs` graph, write `skey.onnx` + `config.json` |
| `validate_skey.py` | key-index agreement of the ONNX vs the torch run_skey pipeline |
| `stage_skey.py` | copy the file set into the publish folder |
| `publish_skey.py` | upload the staged folder to `musetric/skey-onnx` on HF |

All paths are explicit arguments; the scripts carry no implicit defaults.

## Run

The export reuses the main lab env (it imports the vendored S-KEY code) plus
the `export` dependency group (`onnx`):

```sh
uv run --group export python scripts/onnx/skey/export_skey.py \
  --out <export-dir> --models-path <checkpoint-cache-dir>

uv run --group export python scripts/onnx/skey/validate_skey.py \
  --onnx <export-dir>/skey.onnx --audio <audio-dir> \
  --models-path <checkpoint-cache-dir>

uv run python scripts/onnx/skey/stage_skey.py \
  --export <export-dir> --dest <publish-dir>

uv run python scripts/onnx/skey/publish_skey.py --src <publish-dir> --dry-run
```

Validate on the material the model is fed in production — the **instrumental
stem**.

The checkpoint (`skey.pt`, ~0.75 MB) is fetched on first run. The exported graph
is ~0.33 MB fp32 — tiny, so no quantization.

S-KEY is vendored under `musetric_toolkit/key_audio/skey`; see
`thirdPartyNotices.md` for source and license.
