# Musetric Lab

The lab of [Musetric](https://github.com/musetric/musetric): the reference of
every processing step in torch, the export of the models to ONNX, and their
training. Defects and ideas go to [musetric/musetric](https://github.com/musetric/musetric/issues);
the lab has no issues of its own.

## Installation

- An NVIDIA GPU on Linux or Windows on x86_64; torch is the CUDA 12.9 build of
  the PyTorch index, held at 2.8.0: whisperx 3.8.6 requires `torch~=2.8.0`,
  `torchaudio~=2.8.0` and `torchvision~=0.23.0`.
- `onnxruntime-gpu` 1.30.0, which runs the lead and backing model of
  `musetric-separate`, is a CUDA 13.0 build and needs CUDA 13 and cuDNN 9
  ([requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)).
- The `export` group pins onnx 1.22.0 and onnxscript 0.7.1 ([musetric/lab#72](https://github.com/musetric/lab/pull/72)).
- The commands are installed from this clone, editable, so that the scripts of
  Musetric that call them, such as `measure:parity`, run the clone's code:

```bash
uv tool install --editable "."
```

## Development

`scripts/check.py` runs every check CI runs. Besides uv it needs
editorconfig-checker on `PATH` and `npm ci` on the Node of `.node-version`:

```bash
npm ci
uv run --only-group check python scripts/check.py
```

## Commands

A command's `--help` lists its flags; every command takes `--models-path`, the
directory of the downloaded models, and `--log-level`.

`musetric-separate` splits audio into lead, backing vocals and instrumental:

```bash
musetric-separate --source-path song.wav --lead-path lead.flac \
  --backing-path backing.flac --instrumental-path instrumental.flac --sample-rate 48000
```

`musetric-transcribe` writes the lyrics of the lead vocal with WhisperX
(`--backing-path` adds the backing vocals at half level), `musetric-rhythm` the
BPM, beats and downbeats of Beat This!, `musetric-key` the key of S-KEY and
`musetric-chords` the timed chords of ChordMini, each as JSON:

```bash
musetric-transcribe --audio-path lead.flac --result-path transcription.json
musetric-rhythm --audio-path instrumental.flac --result-path rhythm.json
musetric-key --audio-path instrumental.flac --result-path key.json
musetric-chords --audio-path instrumental.flac --result-path chords.json
```

`musetric-pitch` writes the reference pitch track of Musetric's pitch bench,
`time_s,f0_hz,confidence,trusted` per frame, from RMVPE, CREPE, SwiftF0 and
FCPE combined. Two benches measure rather than process: `musetric-pitch-zoo`
checks that reference ([pitch_zoo](musetric_toolkit/pitch_zoo/readme.md)) and
`musetric-rhythm-bench` the beats the metronome would click ([rhythm_bench](musetric_toolkit/rhythm_bench/readme.md)).

```bash
musetric-pitch --audio-path lead.flac --result-path f0.csv
```

`musetric-parity` writes the reference side of a case of Musetric's
`measure:parity`, the step with the original model in torch, our ONNX export
on the CPU and the author's inference: every stage boundary a raw little-endian
`<boundary>.<source>.bin`, listed with dtype and shape in `manifest.json`:

```bash
musetric-parity --step vocals --audio-path song.wav --onnx model.onnx --case-path case
```

## License

The lab is [MIT licensed](https://github.com/musetric/musetric/blob/main/license.md).
Third-party notices are listed in [thirdPartyNotices.md](thirdPartyNotices.md).
