# Musetric Lab

The lab of [Musetric](https://github.com/musetric/musetric): the reference of every processing step in torch, the export of the models to ONNX, and their training.

## Installation

Requires an NVIDIA GPU on Linux or Windows on x86_64. torch is the CUDA 12.9
build from the PyTorch index. `onnxruntime-gpu` 1.30.0, which runs the lead
and backing model of `musetric-separate`, is a CUDA 13.0 build: its CUDA
provider needs CUDA 13 and cuDNN 9 on the machine
([requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)).

torch stays at 2.8.0 because whisperx 3.8.6 requires `torch~=2.8.0`,
`torchaudio~=2.8.0` and `torchvision~=0.23.0`. The `export` group pins onnx
1.22.0 and onnxscript 0.7.1: with onnx 1.21 and onnxscript 0.7.0 a
restructured separation graph lost accuracy in its fp16 conversion, while the
same change was bit-exact in torch and clean in an fp32 export
(musetric/lab#72).

The lab is not published to a package index. Its commands are installed from
this clone, editable, so the scripts of Musetric that call them, such as
`measure:parity` and `measure:pitch`, run the code of the clone:

```bash
uv tool install --editable "."
```

## Development

`scripts/check.py` runs every check CI runs. Besides uv, it needs
editorconfig-checker on `PATH` and the npm tools of `package.json`, installed
with `npm ci` on the Node that `.node-version` pins:

```bash
npm ci
uv run --only-group check python scripts/check.py
```

## CLI Usage

Separate audio into lead vocals, backing vocals, and instrumental parts.

```bash
musetric-separate \
  --source-path /path/to/input.wav \  # input audio file
  --lead-path /path/to/output-lead.flac \  # output path for the lead vocal track
  --backing-path /path/to/output-backing.flac \  # output path for the backing vocal track
  --instrumental-path /path/to/output-instrumental.flac \  # output path for the instrumental track
  --sample-rate 48000 \  # target sample rate
  --models-path /path/to/models \  # base directory for downloaded models
  --log-level info  # debug|info|warn|error (default: info)
```

Transcribe vocals with WhisperX.

```bash
musetric-transcribe \
  --audio-path /path/to/vocals.wav \  # input vocal audio file (the lead)
  --backing-path /path/to/backing.wav \  # optional: backing vocals, added at half level
  --result-path /path/to/transcription.json \  # output JSON file
  --models-path /path/to/models \  # base directory for downloaded models
  --log-level info  # debug|info|warn|error (default: info)
```

Analyze rhythm (BPM, beats, downbeats) with beat_this.

```bash
musetric-rhythm \
  --audio-path /path/to/input.wav \  # input audio file
  --result-path /path/to/rhythm.json \  # output JSON file (BPM, beats, downbeats)
  --models-path /path/to/models \  # base directory for downloaded models
  --log-level info  # debug|info|warn|error (default: info)
```

Detect the musical key (root + mode) with S-KEY.

```bash
musetric-key \
  --audio-path /path/to/input.wav \  # input audio file
  --result-path /path/to/key.json \  # output JSON file (root, mode, confidence)
  --models-path /path/to/models \  # base directory for downloaded models
  --log-level info  # debug|info|warn|error (default: info)
```

Detect chords with timings using ChordMini.

```bash
musetric-chords \
  --audio-path /path/to/input.wav \  # input audio file
  --result-path /path/to/chords.json \  # output JSON file (chord segments with timings)
  --models-path /path/to/models \  # base directory for downloaded models
  --log-level info  # debug|info|warn|error (default: info)
```

Extract a reference pitch track with RMVPE and a global Viterbi decode. It is a
measurement reference for the pitch tracker in Musetric, not a processing step.

```bash
musetric-pitch \
  --audio-path /path/to/vocals.wav \  # input audio file, any format ffmpeg reads
  --result-path /path/to/f0.csv \  # output CSV: time_s,f0_hz,confidence,trusted
  --from-seconds 60 \  # optional start of the analysed range
  --to-seconds 90 \  # optional end of the analysed range
  --hop-ms 5 \  # frame step (default: 5)
  --models-path /path/to/models \  # base directory for downloaded models
  --log-level info  # debug|info|warn|error (default: info)
```

`f0_hz` is 0 on unvoiced frames, `confidence` is the salience peak of the
frame and `trusted` marks frames that a comparison should score: confident
voiced frames grown through pitch-continuous neighbours, after an energy gate
against near-silent tails.

Write the reference of one parity case: a step of Musetric with the original
model in torch inside the app's own processing, our ONNX export on the CPU and
the author's own inference. It is the reference side of the app's
`measure:parity` script, not a processing step.

```bash
musetric-parity \
  --step vocals \  # vocals|voices|rhythm|key|chords|transcribe
  --audio-path /path/to/input.wav \  # input audio file
  --onnx /path/to/model.onnx \  # the app's pinned ONNX model of the step; for transcribe its encoder, next to the rest of the bundle
  --case-path /path/to/case \  # output directory: raw tensors and manifest.json
  --models-path /path/to/models \  # base directory for downloaded models
  --language en \  # transcribe only: the language of the case
  --log-level info  # debug|info|warn|error (default: info)
```

Every stage boundary of the step is a raw little-endian file
`<boundary>.<source>.bin`; `manifest.json` lists them with their dtype and shape.

## License

The lab is [MIT licensed](https://github.com/musetric/musetric/blob/main/license.md).
Third-party notices are listed in [thirdPartyNotices.md](thirdPartyNotices.md).
