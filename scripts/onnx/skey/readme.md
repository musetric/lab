# S-KEY → ONNX export

Exports the vendored **S-KEY** key detector (harmonic VQT + ChromaNet) to a
single self-contained ONNX graph for the `@musetric/ai` runtime on
**onnxruntime-web**, published as [`musetric/skey-onnx`](https://huggingface.co/musetric/skey-onnx).
The graph bakes the whole `run_skey` pipeline, which `export_skey.py` lays out:
the runtime feeds a whole track of raw mono audio at 22050 Hz, peak-normalized,
and reads back the 24 key probabilities, with no DSP of its own.

- The export swaps the affine-free `layer_norm` and `AdaptiveAvgPool2d((12, 1))` of the vendored code for numerically identical forms with a dynamic time axis; the vendored source stays untouched.
- The graph stays in fp32.
- The audio decoder is part of the contract: `ffmpeg` and `torchaudio` move `confidence` by about 1e-4.
- Validate on the material the model is fed in production, the instrumental stem.
- Every path is an explicit argument; the scripts have no implicit defaults.

## Run

The export takes the main lab environment, for the vendored S-KEY code, plus
the `export` group:

```sh
uv run --group export python scripts/onnx/skey/export_skey.py --out <export-dir> --models-path <checkpoint-cache-dir>
uv run --group export python scripts/onnx/skey/validate_skey.py --onnx <export-dir>/skey.onnx --audio <audio-dir> --models-path <checkpoint-cache-dir>
uv run python scripts/onnx/skey/stage_skey.py --export <export-dir> --dest ../models/skey-onnx
```

`export_skey.py` writes `skey.onnx` and `config.json`, `validate_skey.py`
compares the key index of the ONNX with the torch pipeline's, and
`stage_skey.py` copies the file set into the clone of the model repository,
published through git by the [practice of models](https://github.com/musetric/headquarters/blob/main/practices/models.md).
The checkpoint `skey.pt` is fetched on first run.

S-KEY is vendored under `musetric_toolkit/key_audio/skey`; its source and
license are in `thirdPartyNotices.md`. The export came with
[musetric/lab#62](https://github.com/musetric/lab/pull/62).
