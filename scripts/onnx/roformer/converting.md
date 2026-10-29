# Converting a PyTorch Separator to ONNX

The lab keeps the torch implementation as the reference path and CLI default.
This directory exports its neural network core to ONNX and checks that core
through Python `onnxruntime`. One recipe builds two cores: the teacher's,
[Aname's Mel-Band RoFormer Duality](https://huggingface.co/Aname-Tommy/Mel-Band-Roformer_Duality),
and the student distilled from it
([musetric/musetric#1081](https://github.com/musetric/musetric/issues/1081)),
which `musetric` runs on desktop and on phones alike.

A core (`--core-only`) maps `stft_repr [1, 2050, 1100, 2]` to
`masks [1, 2050, 1100, 2]`, float32 in and out, at a window of T = 1100 frames
fixed at build time; the STFT, iSTFT, chunking and overlap-add stay in the
host. The published files are on the model cards of [the teacher's core](https://huggingface.co/musetric/aname-mel-band-roformer-duality-onnx)
and [the student's](https://huggingface.co/musetric/mel-band-roformer-duality-student-onnx).

## Build a Core

The tools are in the `export` group (`uv sync --group export`); the weights and
configs of both models are the ones `musetric_toolkit/common/envs.py` pins.

```bash
uv run --group export python scripts/onnx/roformer/build_full_onnx.py --checkpoint tmp/models/mel_band_roformer_duality/model.ckpt --config tmp/models/mel_band_roformer_duality/config.yaml --output tmp/models/core_t1100.onnx --core-only --fuse-rmsnorm --attn-block 128 --all-fp16 --frames 1100 --skip-gate --split-rows 2 --stream-slabs 4
uv run --group export python scripts/onnx/roformer/split_concat_webgpu.py --input tmp/models/core_t1100.onnx --output tmp/models/duality_core_t1100.onnx
uv run --group export python scripts/onnx/roformer/fp16_epsilon_audit.py tmp/models/duality_core_t1100.onnx
uv run --group export python scripts/onnx/roformer/dispatch_rows_audit.py tmp/models/duality_core_t1100.onnx
uv run python scripts/onnx/roformer/fuse_webgpu_kernels.py --model tmp/models/duality_core_t1100.onnx --out tmp/models/fused/duality_core_t1100.onnx
```

The student's core takes `build_student_onnx.py` with the same flags on the
`model.safetensors` and `config.yaml` of [`musetric/mel-band-roformer-duality-student`](https://huggingface.co/musetric/mel-band-roformer-duality-student),
then the same four steps under the name `duality_student_core_t1100`. A fused
graph reads the `.onnx.data` of its input, which goes next to it.

The core that also splits the vocals into lead and backing
([musetric/musetric#1084](https://github.com/musetric/musetric/issues/1084))
takes `build_lead_backing_onnx.py` with the same flags and files, plus
`--split` with a `split.pt` of `musetric-distill train-voices`, or
`--split-hidden N` for random split weights of that width, to time a shape
before training it. Its `masks` are `[2, 2050, 1100, 2]`: the vocals mask and
the share of the vocals that is backing.

- `--attn-block 128` splits the time attentions into blocks of 128 query rows.
- `--fuse-rmsnorm` gives every RMSNorm one `ai.onnx::RMSNormalization` kernel, and `--all-fp16` drops the fp32 pins.
- Under `--all-fp16` the transformer norms take `RMSNORM_EPS_FP16` (1e-9); the band split, with `RMSNORM_EPS` (1e-12), and every attention over T keys from its scores to its product with the values stay fp32.
- The rotary cosines and sines become fp32 constants before the fp16 conversion.
- No row-dispatched WebGPU kernel (the normalizations, `Softmax`, `LpNormalization`, `TopK`, `InstanceNormalization`, `Reduce*`) sees more than 65535 rows; the exporter refuses a graph that breaks this or the epsilon, and the two audits re-check any artifact.
- `--split-rows N` and `--split-projections N` chunk the feed-forwards and the projections over `N` row groups, exactly.
- `--stream-slabs N` runs the transformers over `N` slabs of a 3-D stream; `N` divides both 60 and T.
- `split_concat_webgpu.py` re-trees every `Concat` and `Split` to at most 8 wide, so no shader binds more than 9 storage buffers.
- `fuse_webgpu_kernels.py` turns GELU, the rotary embedding, the projection to heads and the query scale of the time attentions into one `ai.onnx` kernel each, and keeps the weights file.
- A rebuild of published weights runs `reuse_external_data.py` against the published core before `fuse_webgpu_kernels.py`, so that only the `.onnx` changes.

## Validate and Inspect

```bash
uv run python scripts/onnx/roformer/validate_full_onnx.py --checkpoint <ckpt> --config <config> --source <audio> --out-dir <dir> --frames 1100 --device cuda
uv run --group export python scripts/onnx/roformer/op_audit.py <core>
uv run --group export python scripts/onnx/roformer/infer_separator.py --model <core> --config <config> --source <audio> --target-output <vocals> --residual-output <rest>
```

`validate_full_onnx.py` writes the torch reference of one window as planar
little-endian stereo `full_input.f32` and `full_ref_vocals.f32`; a core passes
at 40 dB SNR or more with no NaN, by a comparator this repository lacks.

Details: [musetric/lab#70](https://github.com/musetric/lab/pull/70), [#74](https://github.com/musetric/lab/pull/74), [#80](https://github.com/musetric/lab/pull/80), [#86](https://github.com/musetric/lab/pull/86), [#88](https://github.com/musetric/lab/pull/88), [#89](https://github.com/musetric/lab/pull/89), [#98](https://github.com/musetric/lab/pull/98), [#101](https://github.com/musetric/lab/pull/101), [#102](https://github.com/musetric/lab/pull/102), [#104](https://github.com/musetric/lab/pull/104), [#124](https://github.com/musetric/lab/pull/124); the GPU defects it works around: [musetric/musetric#975](https://github.com/musetric/musetric/issues/975).
