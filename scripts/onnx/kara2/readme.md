# UVR-MDX-NET KARA2 → WebGPU graph with time bands in the batch

The lead/backing separation step runs `kara2.onnx`, which `fold_time_bands.py`
builds from the UVR release file `UVR_MDXNET_KARA_2.onnx`, a convolutional
U-Net over a `[1, 4, 2048, 256]` complex spectrogram. The fold keeps every
operator of the source and holds each tensor of the U-Net body, `[1, C, T, F]`
in the source, as `[B, C, T / B, F]`: `B` consecutive bands of time frames, one
per batch item.

```sh
uv run python scripts/onnx/kara2/fold_time_bands.py \
  --model UVR_MDXNET_KARA_2.onnx --out kara2.onnx --bands 4 --check
```

- `--bands 4` builds the published graph.
- `B` is at least 3 and divides the frame count of every U-Net level (256 down to 8).
- `--check` fails unless the CPU provider's output stays within `1e-4` of the source's.
- The output `1x1` convolution stays a `MatMul` followed by a rank-3 `Reshape`.

The graph is published in [`musetric/uvr-mdxnet-kara2-onnx`](https://huggingface.co/musetric/uvr-mdxnet-kara2-onnx),
whose model card pins the source file and the published bytes by sha256.

Details: [musetric/lab#82](https://github.com/musetric/lab/pull/82) and [#85](https://github.com/musetric/lab/pull/85) (the earlier static rewrite), [#103](https://github.com/musetric/lab/pull/103) (the fold); [musetric/musetric#954](https://github.com/musetric/musetric/issues/954) and [#975](https://github.com/musetric/musetric/issues/975) (the speed and the GPU defect behind it).
