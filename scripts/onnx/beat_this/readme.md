# Beat This! → ONNX export

Exports the **Beat This!** beat/downbeat tracker (the `beat-this` package the
`musetric-rhythm` CLI runs) to ONNX for the `@musetric/ai` runtime on
**onnxruntime-web**, published as [`musetric/beat-this-onnx`](https://huggingface.co/musetric/beat-this-onnx),
whose model card holds the feature contract. The graph covers only the network:

```
beat_this.onnx  spect [windows, frames, 128] → beat, downbeat [windows, frames]
```

- The host computes the log-mel on WebGPU with `mel-filterbank.bin`, torchaudio's `MelScale.fb` as row-major float32 `[n_fft // 2 + 1, n_mels]`, and a periodic Hann window, which `check_analytic_window` checks.
- The host downmixes stereo as the mean of the channels, not with `ffmpeg -ac 1`; `config.json` records the `downmix` contract.
- The host feeds one 1500-frame window per call, `[1, 1500, 128]`.
- The host picks beats with the `dbn` postprocessing, madmom's `DBNDownBeatTrackingProcessor`, which the parity reference runs through the `madmom` dependency.
- Validate on the input the model gets in production, the instrumental stem.
- Every path is an explicit argument; the scripts have no implicit defaults.

## Run

`split_to_slice_webgpu.py` runs on the export before every later step; the
published graph is the static rewrite at the author's 1500 frames, with its
attention split into head groups:

```sh
uv run --group export python scripts/onnx/beat_this/export_beat_this.py --out <export-dir> --models-path <checkpoint-cache-dir>
uv run --group export python scripts/onnx/beat_this/split_to_slice_webgpu.py --input <export-dir>/beat_this.onnx --output <export-dir>/beat_this.onnx
uv run --group export python scripts/onnx/beat_this/validate_beat_this.py --onnx <export-dir>/beat_this.onnx --audio <audio-dir> --models-path <checkpoint-cache-dir>
uv run python scripts/onnx/beat_this/dump_reference_mel.py --audio <audio-dir> --out <dump-dir> --models-path <checkpoint-cache-dir>
uv run --group export python scripts/onnx/beat_this/capture_shapes.py --model <export-dir>/beat_this.onnx --frames 1500 --out shapes1500.json
uv run --group export python scripts/onnx/beat_this/rewrite_static_adreno.py --model <export-dir>/beat_this.onnx --shapes shapes1500.json --out <export-dir>/beat_this_static.onnx
uv run --group export python scripts/onnx/beat_this/split_attention_webgpu.py --model <export-dir>/beat_this_static.onnx --shapes shapes1500.json --out <export-dir>/beat_this.onnx
uv run python scripts/onnx/beat_this/stage_beat_this.py --export <export-dir> --dest ../models/beat-this-onnx
```

- `validate_beat_this.py` compares the beat and downbeat times of the ONNX with the torch `File2Beats` reference, and `dump_reference_mel.py` writes the reference waveforms and log-mels that gate the runtime's WebGPU front end.
- Run the export at several input lengths and compare the output shapes.
- `--models-path` sets `TORCH_HOME`, as the CLI does; the checkpoint `beat_this-final0.ckpt` is fetched there on first run.
- The staged files are published from the clone through git by the [practice of models](https://github.com/musetric/headquarters/blob/main/practices/models.md).

Beat This! comes from the `beat-this` package; its source and license are in
`thirdPartyNotices.md`.

Details: [musetric/lab#63](https://github.com/musetric/lab/pull/63), [#69](https://github.com/musetric/lab/pull/69), [#78](https://github.com/musetric/lab/pull/78), [#105](https://github.com/musetric/lab/pull/105), [#106](https://github.com/musetric/lab/pull/106); the GPU defects the graph works around: [musetric/musetric#975](https://github.com/musetric/musetric/issues/975).
