# ChordMini → ONNX/CQT export

This directory exports the vendored **ChordMini** chord recognizer (ChordNet
"2E1D", 170-class large vocabulary) as a **ChordNet/CQT artifact**: a small
`features → logits` ONNX classifier plus the versioned CQT plan that defines its
input features, published as [`musetric/chordmini-onnx`](https://huggingface.co/musetric/chordmini-onnx),
whose model card holds the file set and its contract. Feature extraction runs
on WebGPU in `@musetric/cqt`, not in the graph.

- The graph maps unnormalized log-CQT windows `features [W, 108, 144]` to `logits [W, 108, 170]`, with the normalization inside it.
- CQT extraction, tail padding and windowing, temporal smoothing and argmax stay in the runtime.
- `export_chordnet.py` writes a dynamic `W`; the published `chordnet.onnx` is the static rewrite at `W` = 16.
- The TypeScript plan payload is for the tests of `@musetric/cqt` and is not published.
- Every path is an explicit argument; the scripts guess no folder and reuse no cache of the machine.

## Build and stage

```sh
uv run --group export python scripts/onnx/chordmini/export_chordnet.py --out <chordnet-export-dir> --models-path <checkpoint-cache-dir>
uv run --group export python scripts/onnx/chordmini/validate_chordnet.py --onnx <chordnet-export-dir>/chordnet.onnx --models-path <checkpoint-cache-dir>
uv run python scripts/onnx/chordmini/generate_cqt_plan.py --out <cqt-plan.bin> --manifest <cqt-plan.manifest.json> --typescript-payload <packages/cqt/src/cqt/__test__/planPayload.ts>
uv run python scripts/onnx/chordmini/generate_cqt_reference.py --out <packages/cqt/src/cqt/__test__/reference.ts>
uv run python scripts/onnx/chordmini/stage_chordnet.py --chordnet-export <chordnet-export-dir> --cqt-plan <cqt-plan.bin> --cqt-plan-manifest <cqt-plan.manifest.json> --dest ../models/chordmini-onnx

uv run --group export python scripts/onnx/chordmini/capture_shapes.py --model <chordnet-export-dir>/chordnet.onnx --input features --frames 108 --windows 16 --shape 16,108,144 --ops Transpose,MatMul --out <shapes.json> --probe <probe.onnx>
uv run --group export python scripts/onnx/chordmini/rewrite_static_adreno.py --model <chordnet-export-dir>/chordnet.onnx --shapes <shapes.json> --out <rewrite-dir>/chordnet.onnx --vec4-attention
uv run --group export python scripts/onnx/roformer/dispatch_rows_audit.py <rewrite-dir>/chordnet.onnx
```

- `validate_chordnet.py` fails when a logit of three fixed-seed random windows is more than `1e-4` off the torch classifier.
- The rewrite bakes every Gather index and Reshape target for `--windows` and pins the graph's input and output to that batch.
- The batch stays at or below 37, the 65535-row cap that `dispatch_rows_audit.py` checks; any other batch is re-checked on the device before it is published.
- The published graph is built with `--vec4-attention`.
- `stage_chordnet.py` renames the plan to `cqt-plan.bin` and `cqt-plan.manifest.json`; the clone is published through git by the [practice of models](https://github.com/musetric/headquarters/blob/main/practices/models.md).
- The end-to-end check runs on the TypeScript side, its chords against `musetric-chords` on the same instrumental stem, never on audio where the reference emits a near-constant label, such as an isolated vocal.

The checkpoint is fetched on first run by `ensure_checkpoint`; the vendored
source and its license are in `thirdPartyNotices.md`.

Details: [musetric/lab#60](https://github.com/musetric/lab/pull/60), [#77](https://github.com/musetric/lab/pull/77), [#81](https://github.com/musetric/lab/pull/81), [#87](https://github.com/musetric/lab/pull/87), [#97](https://github.com/musetric/lab/pull/97); the GPU defects the graph works around: [musetric/musetric#975](https://github.com/musetric/musetric/issues/975), [#891](https://github.com/musetric/musetric/issues/891).
