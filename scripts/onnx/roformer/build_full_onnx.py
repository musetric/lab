# ruff: noqa: T201

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from einops import rearrange
from torch import nn

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

from musetric_toolkit.separate_audio.roformer.attend import Attend, row_chunks
from musetric_toolkit.separate_audio.roformer.mel_band_roformer import (
    Attention,
    FeedForward,
    MelBandRoformer,
    RMSNorm,
    Transformer,
)
from musetric_toolkit.separate_audio.roformer_utils import dict_to_namespace

N_FFT = 2048
HOP = 441
WIN = 2048
HEADS = 8
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM
T = 1100
TSAMP = HOP * (T - 1)
PAD = N_FFT // 2
FREQS = N_FFT // 2 + 1
PACKED = FREQS * 2
RMSNORM_EPS = 1e-12
RMSNORM_EPS_FP16 = 1e-9
FP16_RSQRT_FLOOR = 1.0 / 65504.0**2
WEBGPU_DISPATCH_ROWS = 65535
LEADING_AXES_OPS = frozenset(
    {"LayerNormalization", "RMSNormalization", "SimplifiedLayerNormalization"}
)
LAST_AXIS_OPS = frozenset(
    {"SkipLayerNormalization", "SkipSimplifiedLayerNormalization"}
)
ONE_AXIS_OPS = frozenset({"LpNormalization", "Softmax", "TopK"})
REDUCE_OPS = frozenset(
    {
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
    }
)
ROW_DISPATCH_OPS = (
    LEADING_AXES_OPS
    | LAST_AXIS_OPS
    | ONE_AXIS_OPS
    | REDUCE_OPS
    | {"InstanceNormalization"}
)


def hann_periodic() -> np.ndarray:
    return np.hanning(WIN + 1)[:-1].astype(np.float64)


def stft_conv_weight() -> torch.Tensor:
    window = hann_periodic()
    n = np.arange(N_FFT)
    f = np.arange(FREQS)
    angle = 2 * np.pi * np.outer(f, n) / N_FFT
    w_real = np.cos(angle) * window
    w_imag = -np.sin(angle) * window
    w = np.concatenate([w_real, w_imag], axis=0)[:, None, :]
    return torch.from_numpy(w.astype(np.float32))


def istft_conv_weight() -> torch.Tensor:
    window = hann_periodic()
    n = np.arange(N_FFT)
    k = np.arange(FREQS)
    a = np.full(k.shape, 2.0)
    a[0] = 1.0
    a[-1] = 1.0
    angle = 2 * np.pi * np.outer(k, n) / N_FFT
    coeff_real = (a[:, None] / N_FFT) * np.cos(angle)
    coeff_imag = -(a[:, None] / N_FFT) * np.sin(angle)
    w = np.concatenate([coeff_real, coeff_imag], axis=0) * window[None, :]
    return torch.from_numpy(w[:, None, :].astype(np.float32))


def window_envelope() -> torch.Tensor:
    window = hann_periodic()
    ones = torch.ones(1, 1, T, dtype=torch.float64)
    wsq = torch.from_numpy(window**2).view(1, 1, N_FFT)
    env = F.conv_transpose1d(ones, wsq, stride=HOP)[0, 0, PAD : PAD + TSAMP]
    return env.clamp(min=1e-8).float()


class FullSeparator(nn.Module):

    def __init__(self, model: MelBandRoformer):
        super().__init__()
        self.model = model
        self.register_buffer("stft_w", stft_conv_weight())
        self.register_buffer("istft_w", istft_conv_weight())
        self.register_buffer("env", window_envelope())
        nbpf = model.num_bands_per_freq.float()
        denom = nbpf.repeat_interleave(2).clamp(min=1e-8)
        self.register_buffer("denom", denom)
        self.register_buffer("freq_indices", model.freq_indices.long())

    def stft(self, raw_audio: torch.Tensor) -> torch.Tensor:
        x = raw_audio.reshape(2, 1, TSAMP)
        x = F.pad(x, (PAD, PAD), mode="reflect")
        y = F.conv1d(x, self.stft_w, stride=HOP)
        y = y.reshape(1, 2, 2, FREQS, T)
        y = y.permute(0, 3, 1, 4, 2)
        return y.reshape(1, PACKED, T, 2)

    def apply_mask_istft(
        self, stft_repr: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        masks = masks.reshape(1, self.freq_indices.shape[0], T, 2)
        index = self.freq_indices.view(1, -1, 1, 1).expand_as(masks)
        summed = torch.zeros(
            1, PACKED, T, 2, dtype=masks.dtype, device=masks.device
        ).scatter_add(1, index, masks)
        averaged = summed / self.denom.view(1, PACKED, 1, 1)
        sr, si = stft_repr[..., 0], stft_repr[..., 1]
        mr, mi = averaged[..., 0], averaged[..., 1]
        out_r = sr * mr - si * mi
        out_i = sr * mi + si * mr
        rr = out_r.reshape(1, FREQS, 2, T)
        ii = out_i.reshape(1, FREQS, 2, T)
        re = rr.permute(0, 2, 1, 3).reshape(2, FREQS, T)
        im = ii.permute(0, 2, 1, 3).reshape(2, FREQS, T)
        stacked = torch.cat([re, im], dim=1)
        y = F.conv_transpose1d(stacked, self.istft_w, stride=HOP)
        y = y[:, 0, PAD : PAD + TSAMP] / self.env
        return y.reshape(1, 2, TSAMP)

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        stft_repr = self.stft(raw_audio)
        masks = self.model.net_forward(stft_repr)
        return self.apply_mask_istft(stft_repr, masks)


def set_attention_block(model: MelBandRoformer, q_block: int) -> int:
    count = 0
    for m in model.modules():
        if isinstance(m, Attend):
            m.q_block = q_block
            count += 1
    return count


class RowChunkedFeedForward(nn.Module):

    def __init__(self, net: nn.Sequential, parts: int, dim: int = 0) -> None:
        super().__init__()
        self.net = net
        self.parts = parts
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(self.parts, dim=self.dim)
        return torch.cat([self.net(c) for c in chunks], dim=self.dim)


class RowChunkedLinear(nn.Module):

    def __init__(self, linear: nn.Module, parts: int) -> None:
        super().__init__()
        self.linear = linear
        self.parts = parts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.linear(c) for c in x.chunk(self.parts, dim=0)], dim=0)


def _attention_forward_unfused(self: Attention, x: torch.Tensor) -> torch.Tensor:
    x = self.norm(x)
    q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)
    k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=self.heads)
    v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=self.heads)
    if self.rotary_embed is not None:
        q = self.rotary_embed.rotate_queries_or_keys(q)
        k = self.rotary_embed.rotate_queries_or_keys(k)
    out = self.attend(q, k, v)
    gates = self.to_gates(x)
    out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()
    out = rearrange(out, "b h n d -> b n (h d)")
    return self.to_out(out)


def split_heavy_projections(
    model: MelBandRoformer, parts: int, projection_parts: int = 1
) -> tuple[int, int]:
    feeds = 0
    for module in list(model.modules()):
        if isinstance(module, FeedForward):
            module.net = RowChunkedFeedForward(module.net, parts)
            feeds += 1
    attentions = 0
    for module in list(model.modules()):
        if not isinstance(module, Attention):
            continue
        weight = module.to_qkv.weight.data
        inner = weight.shape[0] // 3
        for name, index in (("to_q", 0), ("to_k", 1), ("to_v", 2)):
            linear = nn.Linear(module.to_qkv.in_features, inner, bias=False)
            linear.weight.data.copy_(weight[index * inner : (index + 1) * inner])
            setattr(
                module,
                name,
                (
                    RowChunkedLinear(linear, projection_parts)
                    if projection_parts > 1
                    else linear
                ),
            )
        if projection_parts > 1:
            module.to_out[0] = RowChunkedLinear(module.to_out[0], projection_parts)
        del module.to_qkv
        module.forward = _attention_forward_unfused.__get__(module, Attention)
        attentions += 1
    return feeds, attentions


def set_stream_slabs(model: MelBandRoformer, slabs: int) -> None:
    for module in model.modules():
        if isinstance(module, RowChunkedFeedForward):
            module.dim = 1
    model.net_forward = lambda stft_repr: _stream_slab_net_forward(
        model, stft_repr, slabs
    )


def _stream_slab_net_forward(
    model: MelBandRoformer, stft_repr: torch.Tensor, slabs: int
) -> torch.Tensor:
    batch = stft_repr.shape[0]
    freq_indices = model.freq_indices.to(stft_repr.device)
    batch_arange = torch.arange(batch, device=stft_repr.device)[..., None]
    x = stft_repr[batch_arange, freq_indices]
    x = rearrange(x, "b f t c -> b t (f c)")
    x = model.band_split(x)
    for time_transformer, freq_transformer in model.layers:
        outs = []
        for slab in x.chunk(slabs, dim=2):
            bands = slab.shape[2]
            rows = rearrange(slab, "b t f d -> b (f t) d")
            rows = _transformer_over_rows(time_transformer, rows, bands)
            outs.append(rearrange(rows, "b (f t) d -> b t f d", f=bands))
        x = torch.cat(outs, dim=2)
        outs = []
        for slab in x.chunk(slabs, dim=1):
            frames = slab.shape[1]
            rows = rearrange(slab, "b t f d -> b (t f) d")
            rows = _transformer_over_rows(freq_transformer, rows, frames)
            outs.append(rearrange(rows, "b (t f) d -> b t f d", t=frames))
        x = torch.cat(outs, dim=1)
    masks = torch.stack([fn(x) for fn in model.mask_estimators], dim=1)
    return rearrange(masks, "b n t (f c) -> b n f t c", c=2)


def _transformer_over_rows(
    transformer: Transformer, x: torch.Tensor, batch: int
) -> torch.Tensor:
    for attn, ff in transformer.layers:
        x = _attention_over_rows(attn, x, batch) + x
        x = ff(x) + x
    return transformer.norm(x)


def _attention_over_rows(attn: Attention, x: torch.Tensor, batch: int) -> torch.Tensor:
    x = attn.norm(x)
    q = rearrange(attn.to_q(x), "o (b n) (h d) -> (o b) h n d", b=batch, h=attn.heads)
    k = rearrange(attn.to_k(x), "o (b n) (h d) -> (o b) h n d", b=batch, h=attn.heads)
    v = rearrange(attn.to_v(x), "o (b n) (h d) -> (o b) h n d", b=batch, h=attn.heads)
    if attn.rotary_embed is not None:
        q = attn.rotary_embed.rotate_queries_or_keys(q)
        k = attn.rotary_embed.rotate_queries_or_keys(k)
    out = attn.attend(q, k, v)
    gates = rearrange(attn.to_gates(x), "o (b n) h -> (o b) h n 1", b=batch)
    out = out * gates.sigmoid()
    return attn.to_out(rearrange(out, "b h n d -> 1 (b n) (h d)"))


class RowChunkedRMSNorm(nn.Module):

    def __init__(self, norm: RMSNorm) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows_per_batch = int(np.prod(x.shape[1:-1]))
        parts = row_chunks(x.shape[0], rows_per_batch, WEBGPU_DISPATCH_ROWS)
        if parts == 1:
            return self.norm(x)
        return torch.cat([self.norm(c) for c in x.chunk(parts, dim=0)], dim=0)


def cap_dispatch_rows(model: MelBandRoformer) -> tuple[int, int]:
    norms = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, RMSNorm):
                setattr(module, name, RowChunkedRMSNorm(child))
                norms += 1
    attends = 0
    for module in model.modules():
        if isinstance(module, Attend):
            module.softmax_rows = WEBGPU_DISPATCH_ROWS
            attends += 1
    return norms, attends


def load_model(checkpoint: Path, config: Path) -> MelBandRoformer:
    with open(config) as f:
        cfg = dict_to_namespace(yaml.load(f, Loader=yaml.FullLoader))  # noqa: S506
    model = MelBandRoformer(**vars(cfg.model)).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    for m in model.modules():
        if isinstance(m, Attend):
            m.flash = False
    return model


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    err = ((ref - got) ** 2).sum().clamp(min=1e-20)
    return (10 * torch.log10((ref**2).sum() / err)).item()


def gate1(model: MelBandRoformer, full: FullSeparator) -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 2, TSAMP) * 0.1
    with torch.no_grad():
        ref = model(x)
        got = full(x)
    if ref.shape != got.shape:
        print(f"shape mismatch: ref {tuple(ref.shape)} vs got {tuple(got.shape)}")
    snr = snr_db(ref, got)
    print(
        f"gate1 FullSeparator vs model.forward: SNR={snr:.1f} dB  "
        f"max|diff|={float((ref-got).abs().max()):.3e}"
    )
    print("VERDICT:", "PASS" if snr > 60 else "FAIL")  # noqa: PLR2004


def main() -> None:
    global T, TSAMP  # noqa: PLW0603
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--check", action="store_true", help="numerics gate only, no export")
    p.add_argument(
        "--frames",
        type=int,
        default=T,
        help="chunk frame count T (STFT/iSTFT weights are T-independent; only "
        "TSAMP and the static shapes change). The default is the full reference "
        "context the published core uses; a shorter window halves the activation "
        "footprint but loses separation context, measuring ~24 dB at T=501 "
        "against the same track at the default.",
    )
    p.add_argument(
        "--skip-gate",
        action="store_true",
        help="skip the CPU torch-vs-conv numerics gate (T-invariant for the DSP; "
        "validated at T=501). Halves export-time peak RAM at large T.",
    )
    p.add_argument(
        "--fp32",
        action="store_true",
        help="export the full graph in fp32 (skip fp16 conversion) — experiment to "
        "compare quality/perf/VRAM vs the mixed fp16 build.",
    )
    p.add_argument(
        "--fuse-rmsnorm",
        action="store_true",
        help="fuse only the RMSNorm chains to ai.onnx RMSNormalization, leaving "
        "attention alone so --attn-block survives. Unlike the hand-lowered chain, "
        "the WebGPU kernel for this op casts to f32 inside the shader and "
        "accumulates the sum of squares there, so it needs no fp32 island and no "
        "giant cast copies: fp16 in, fp16 out, f32 where it matters.",
    )
    p.add_argument(
        "--all-fp16",
        action="store_true",
        help="convert everything except graph IO to fp16, including the RMSNorm "
        "islands and Softmax that are normally pinned to fp32. Those pins cost "
        "nothing at T=501 but at T=1100 they turn the [T,60,1536] activations "
        "into 387 MiB fp32 tensors plus a cast copy each, which is far past what "
        "an Adreno storage buffer addresses. RMSNormalization then carries "
        f"epsilon={RMSNORM_EPS_FP16:g} instead of {RMSNORM_EPS:g}, which is what "
        "keeps its fp16 reciprocal finite; check parity before trusting the rest, "
        "mean(x^2) can still overflow fp16 in the unfused chain.",
    )
    p.add_argument(
        "--attn-block",
        type=int,
        default=0,
        help="split attention over the query axis into chunks of this many rows "
        "(0 = off). Exact, same FLOPs; it only caps the peak score tensor at "
        "[60,8,block,T] instead of [60,8,T,T], which is what mobile WebGPU "
        "storage buffers cannot bind.",
    )
    p.add_argument(
        "--split-rows",
        type=int,
        default=0,
        help="chunk every feed-forward into this many row groups and unfuse the "
        "qkv projection into three (0 = off). Exact, fp16 included, and the same "
        "FLOPs; it only shortens the two longest matmuls of each layer, which a "
        "mobile GPU otherwise runs as one uninterruptible dispatch each -- the "
        "part of a run that a queue-level flush threshold cannot break up.",
    )
    p.add_argument(
        "--split-projections",
        type=int,
        default=0,
        help="with --split-rows, also chunk the q/k/v and output projections of "
        "every attention into this many row groups (0 = off). Exact like "
        "--split-rows; it shortens the matmuls that --split-rows leaves at full "
        "height, the longest dispatches of a run on Adreno 660.",
    )
    p.add_argument(
        "--stream-slabs",
        type=int,
        default=0,
        help="with --split-rows, run every time and band transformer over this "
        "many slabs of the stream, each held as [1, rows, 384] (0 = off). Exact; "
        "a divisor of both 60 and T makes the time and band slabs the same shape, "
        "so onnxruntime's planner shares their buffers and the peak memory of a "
        "run falls by more than half.",
    )
    p.add_argument(
        "--core-only",
        action="store_true",
        help="export the web core (stft_repr -> per-bin masks), NOT the full "
        "STFT/iSTFT graph. Produces the published core that onnxruntime-web runs "
        "with @musetric/fft host-side DSP. The model bakes the mel-band "
        "gather/average tables into the graph.",
    )
    args = p.parse_args()

    T = args.frames
    TSAMP = HOP * (T - 1)
    print(f"frames T={T}  TSAMP={TSAMP}  (~{TSAMP / 44100:.1f}s @44.1k)")

    model = load_model(args.checkpoint, args.config)
    if args.attn_block:
        n = set_attention_block(model, args.attn_block)
        peak = 60 * 8 * args.attn_block * T * 2
        print(
            f"attention blocked at {args.attn_block} query rows over {n} layers; "
            f"peak time-attention score tensor {peak / 1048576:.1f} MiB "
            f"(was {60 * 8 * T * T * 2 / 1048576:.1f} MiB)"
        )
    if args.split_rows:
        feeds, attentions = split_heavy_projections(
            model, args.split_rows, max(args.split_projections, 1)
        )
        print(
            f"split {feeds} feed-forwards into {args.split_rows} row groups, "
            f"unfused {attentions} qkv projections"
            + (
                f" and split them into {args.split_projections} row groups"
                if args.split_projections > 1
                else ""
            )
        )
    if args.stream_slabs:
        set_stream_slabs(model, args.stream_slabs)
        print(f"every transformer runs over {args.stream_slabs} stream slabs")
    norms, attends = cap_dispatch_rows(model)
    print(
        f"capped dispatch rows at {WEBGPU_DISPATCH_ROWS} over {norms} norms "
        f"and {attends} attention layers"
    )
    with torch.no_grad():
        model.net_forward(torch.randn(1, PACKED, T, 2))

    if args.core_only:
        net = Core(model).eval()
        if args.check:
            return
        if args.output is None:
            raise SystemExit("--output required for export")
        export(
            net,
            args.output,
            fp32=args.fp32,
            all_fp16=args.all_fp16,
            fuse_rmsnorm_only=args.fuse_rmsnorm,
            core_only=True,
        )
        return

    full = FullSeparator(model).eval()
    if not args.skip_gate:
        gate1(model, full)

    if args.check:
        return
    if args.output is None:
        raise SystemExit("--output required for export")
    export(
        full,
        args.output,
        fp32=args.fp32,
        all_fp16=args.all_fp16,
        fuse_rmsnorm_only=args.fuse_rmsnorm,
    )


class Core(nn.Module):

    def __init__(self, model: MelBandRoformer):
        super().__init__()
        self.model = model
        freq_indices = model.freq_indices.long()
        pair0 = torch.zeros(PACKED, dtype=torch.long)
        pair1 = torch.zeros(PACKED, dtype=torch.long)
        valid0 = torch.zeros(PACKED)
        valid1 = torch.zeros(PACKED)
        seen = torch.zeros(PACKED, dtype=torch.long)
        for band in range(freq_indices.shape[0]):
            b = int(freq_indices[band])
            if seen[b] == 0:
                pair0[b], valid0[b] = band, 1.0
            elif seen[b] == 1:
                pair1[b], valid1[b] = band, 1.0
            else:
                raise RuntimeError(f"mask fan-in > 2 at bin {b}")
            seen[b] += 1
        denom = model.num_bands_per_freq.float().repeat_interleave(2).clamp(min=1e-8)
        self.register_buffer("pair0", pair0)
        self.register_buffer("pair1", pair1)
        self.register_buffer("valid0", valid0.view(1, PACKED, 1, 1))
        self.register_buffer("valid1", valid1.view(1, PACKED, 1, 1))
        self.register_buffer("denom", denom.view(1, PACKED, 1, 1))

    def forward(self, stft_repr: torch.Tensor) -> torch.Tensor:
        masks = self.model.net_forward(stft_repr)
        masks = masks.reshape(1, -1, masks.shape[-2], 2)
        g0 = masks.index_select(1, self.pair0)
        g1 = masks.index_select(1, self.pair1)
        summed = g0 * self.valid0 + g1 * self.valid1
        return (summed / self.denom).float()


def export(  # noqa: C901, PLR0912, PLR0913, PLR0915
    full: nn.Module,
    output: Path,
    fp32: bool = False,
    all_fp16: bool = False,
    fuse_rmsnorm_only: bool = False,
    core_only: bool = False,
) -> None:
    import time  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

    rms_eps = RMSNORM_EPS_FP16 if all_fp16 and not fp32 else RMSNORM_EPS
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_name(f"{output.stem}_raw.onnx")
    if core_only:
        dummy = torch.randn(1, PACKED, T, 2, dtype=torch.float32) * 0.1
        in_names, out_names = ["stft_repr"], ["masks"]
        print(f"export core graph at stft_repr (1,{PACKED},{T},2) (dynamo, static) ...")
    else:
        dummy = torch.randn(1, 2, TSAMP, dtype=torch.float32) * 0.1
        in_names, out_names = ["raw_audio"], ["vocals"]
        print(f"export full graph at raw_audio (1,2,{TSAMP}) (dynamo, static) ...")

    t0 = time.time()
    torch.onnx.export(
        full,
        (dummy,),
        str(raw),
        input_names=in_names,
        output_names=out_names,
        dynamo=True,
        opset_version=18,
    )
    print(f"  exported in {time.time()-t0:.1f}s")

    import gc  # noqa: PLC0415

    del full
    gc.collect()

    import onnx  # noqa: PLC0415
    from onnxconverter_common import float16  # noqa: PLC0415

    proto = onnx.load(str(raw))
    counts = Counter(n.op_type for n in proto.graph.node)
    audit = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    print("op audit (fp32 export):", audit)
    if "Einsum" in counts:
        raise RuntimeError("Einsum present - matmul attention rewrite did not take")
    for native in ("STFT", "DFT"):
        if native in counts:
            raise RuntimeError(f"{native} op present - DSP did not lower to conv")

    if fuse_rmsnorm_only:
        n = fuse_rmsnorm(proto.graph, rms_eps)
        for op in proto.opset_import:
            if op.domain in ("", "ai.onnx") and op.version < 23:  # noqa: PLR2004
                op.version = 23
        onnx.checker.check_model(proto, full_check=False)
        print(
            f"  fused {n} RMSNorm chains -> RMSNormalization "
            f"(fp16 IO, f32 inside, epsilon={rms_eps:g})"
        )

    folded = fold_rotary_tables(proto.graph)
    print(f"  folded {folded} rotary Cos/Sin tables into constants")

    if fp32:
        print("keeping FP32 (no fp16 conversion) ...")
        save_fp16(proto, output)
    else:
        extra_block = (
            ["Conv", "ConvTranspose", "Pad"]
            if all_fp16
            else [
                "Softmax",
                "ReduceL2",
                "Clip",
                "Div",
                "Conv",
                "ConvTranspose",
                "Pad",
                "RMSNormalization",
            ]
        )
        op_block = list(dict.fromkeys([*float16.DEFAULT_OP_BLOCK_LIST, *extra_block]))
        print(f"converting to FP16 (keep_io_types, block += {extra_block}) ...")
        model16 = float16.convert_float_to_float16(
            proto,
            keep_io_types=True,
            disable_shape_infer=True,
            op_block_list=op_block,
        )
        del proto
        gc.collect()

        if core_only:
            ensure_float32_outputs(model16)
        if all_fp16:
            moved = keep_band_split_fp32(model16, RMSNORM_EPS)
            print(f"  kept the band gather and {moved} band norms in fp32")
            kept = keep_time_attention_fp32(model16, T)
            print(f"  kept {kept} time-attention softmaxes and their p x V in fp32")
        assert_fp16_epsilon_safe(model16)
        assert_dispatch_rows_safe(model16)
        save_fp16(model16, output)
        reloaded = onnx.load(str(output))
        nfix = sanitize_fp16_initializers(reloaded)
        if nfix:
            save_fp16(reloaded, output)
        print(f"  sanitized {nfix} non-finite fp16 weight value(s)")

    raw.unlink(missing_ok=True)
    raw.with_suffix(".onnx.data").unlink(missing_ok=True)
    total = output.stat().st_size + output.with_suffix(".onnx.data").stat().st_size
    print(f"  wrote {output.name} (+data, {total/1e6:.0f} MB)")

    import onnxruntime as ort  # noqa: PLC0415

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        str(output), sess_options=so, providers=["CPUExecutionProvider"]
    )
    print("ORT path-load OK:")
    print("  inputs :", [(i.name, i.shape) for i in sess.get_inputs()])
    print("  outputs:", [(o.name, o.shape) for o in sess.get_outputs()])


def _io_maps(graph):
    producer = {}
    consumers: dict = {}
    for n in graph.node:
        for o in n.output:
            producer[o] = n
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    return producer, consumers


def _sole_consumer(consumers, name):
    cs = consumers.get(name, [])
    return cs[0] if len(cs) == 1 else None


def _rewrite_nodes(graph, remove: set, replace_at: dict) -> None:
    nodes = list(graph.node)
    del graph.node[:]
    for n in nodes:
        if n.name in replace_at:
            rep = replace_at[n.name]
            for r in rep if isinstance(rep, list) else [rep]:
                graph.node.append(r)
        if n.name in remove:
            continue
        graph.node.append(n)


def fold_rotary_tables(graph) -> int:
    from onnx import numpy_helper  # noqa: PLC0415

    producer, _ = _io_maps(graph)
    inits = {i.name: i for i in graph.initializer}
    folded, remove = 0, set()
    for node in list(graph.node):
        if node.op_type not in ("Cos", "Sin"):
            continue
        sl = producer.get(node.input[0])
        if sl is None or sl.op_type != "Slice" or any(i not in inits for i in sl.input):
            continue
        data, starts, ends, *rest = (numpy_helper.to_array(inits[i]) for i in sl.input)
        axes = rest[0] if rest else np.arange(len(starts))
        steps = rest[1] if len(rest) > 1 else np.ones(len(starts), dtype=np.int64)
        index = [slice(None)] * data.ndim
        for a, b, e, st in zip(axes, starts, ends, steps, strict=True):
            index[int(a)] = slice(int(b), int(e), int(st))
        angles = data[tuple(index)].astype(np.float64)
        value = (np.cos(angles) if node.op_type == "Cos" else np.sin(angles)).astype(
            np.float32
        )
        graph.initializer.append(numpy_helper.from_array(value, node.output[0]))
        remove.add(node.name)
        folded += 1
    _rewrite_nodes(graph, remove, {})
    _, consumers = _io_maps(graph)
    dead = {
        n.name
        for n in graph.node
        if n.op_type == "Slice" and not any(consumers.get(o) for o in n.output)
    }
    _rewrite_nodes(graph, dead, {})
    _, consumers = _io_maps(graph)
    used = set(consumers) | {o.name for o in graph.output}
    keep = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(keep)
    return folded


def _ensure_ms_opset(model) -> None:
    import onnx  # noqa: PLC0415

    if not any(op.domain == "com.microsoft" for op in model.opset_import):
        model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))


def fuse_rmsnorm(graph, epsilon: float) -> int:
    import onnx  # noqa: PLC0415

    _producer, consumers = _io_maps(graph)
    inits = {t.name for t in graph.initializer}
    remove: set = set()
    replace_at: dict = {}
    count = 0
    for red in [n for n in graph.node if n.op_type == "ReduceL2"]:
        x = red.input[0]
        clip = _sole_consumer(consumers, red.output[0])
        if clip is None or clip.op_type != "Clip":
            continue
        expand = _sole_consumer(consumers, clip.output[0])
        if expand is None or expand.op_type != "Expand":
            continue
        div = _sole_consumer(consumers, expand.output[0])
        if div is None or div.op_type != "Div" or div.input[0] != x:
            continue
        mul1 = _sole_consumer(consumers, div.output[0])
        if mul1 is None or mul1.op_type != "Mul":
            continue
        mul2 = _sole_consumer(consumers, mul1.output[0])
        if mul2 is None or mul2.op_type != "Mul":
            continue
        gamma = next((i for i in mul2.input if i != mul1.output[0]), None)
        if gamma is None or gamma not in inits:
            continue
        rms = onnx.helper.make_node(
            "RMSNormalization",
            [x, gamma],
            [mul2.output[0]],
            name=f"RMSNorm_{count}",
            axis=-1,
            epsilon=epsilon,
        )
        replace_at[red.name] = rms
        remove.update(
            {red.name, clip.name, expand.name, div.name, mul1.name, mul2.name}
        )
        count += 1
    if count:
        _rewrite_nodes(graph, remove, replace_at)
    return count


def save_fp16(model, output: Path) -> None:
    import onnx  # noqa: PLC0415

    output.unlink(missing_ok=True)
    output.with_suffix(".onnx.data").unlink(missing_ok=True)
    onnx.save_model(
        model,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output.name + ".data",
        size_threshold=4096,
        convert_attribute=True,
    )


def sanitize_fp16_initializers(model) -> int:
    import onnx  # noqa: PLC0415
    from onnx import numpy_helper  # noqa: PLC0415

    fixed = 0
    for t in model.graph.initializer:
        if t.data_type != onnx.TensorProto.FLOAT16:
            continue
        a = numpy_helper.to_array(t).astype(np.float32)
        if not np.isfinite(a).all():
            fixed += int((~np.isfinite(a)).sum())
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
            t.CopyFrom(numpy_helper.from_array(a, t.name))
    return fixed


def assert_fp16_epsilon_safe(model) -> None:
    import onnx  # noqa: PLC0415

    graph = model.graph
    dtypes = {t.name: t.data_type for t in graph.initializer}
    for value in list(graph.value_info) + list(graph.input) + list(graph.output):
        dtypes[value.name] = value.type.tensor_type.elem_type

    bad: list[str] = []
    for node in graph.node:
        fp16 = any(dtypes.get(n) == onnx.TensorProto.FLOAT16 for n in node.output)
        if not fp16:
            continue
        if node.op_type != "RMSNormalization":
            continue
        eps = next((a.f for a in node.attribute if a.name == "epsilon"), None)
        if eps is not None and eps < FP16_RSQRT_FLOOR:
            bad.append(f"{node.name or node.op_type}={eps:g}")
    if bad:
        raise RuntimeError(
            f"{len(bad)} fp16 normalization node(s) carry an epsilon below the fp16 "
            f"reciprocal floor {FP16_RSQRT_FLOOR:g}, which makes every near-zero row "
            f"NaN on a WebGPU kernel: {', '.join(bad[:4])}"
            + (" ..." if len(bad) > 4 else "")  # noqa: PLR2004
        )


def assert_dispatch_rows_safe(model) -> None:
    graph = model.graph
    shapes = {}
    for value in list(graph.value_info) + list(graph.input) + list(graph.output):
        dims = value.type.tensor_type.shape.dim
        if dims and all(d.HasField("dim_value") for d in dims):
            shapes[value.name] = [d.dim_value for d in dims]
    constants = {t.name: t for t in graph.initializer}

    bad: list[str] = []
    for node in graph.node:
        if node.op_type not in ROW_DISPATCH_OPS:
            continue
        dims = shapes.get(node.input[0])
        rows = None if dims is None else dispatch_rows(node, dims, constants)
        if rows is None:
            bad.append(f"{node.name or node.op_type}=unknown rows")
            continue
        if rows > WEBGPU_DISPATCH_ROWS:
            bad.append(f"{node.name or node.op_type}{dims}={rows}")
    if bad:
        raise RuntimeError(
            f"{len(bad)} row-dispatched node(s) exceed {WEBGPU_DISPATCH_ROWS} rows, "
            f"which overflows their WebGPU output: {', '.join(bad[:4])}"
            + (" ..." if len(bad) > 4 else "")  # noqa: PLR2004
        )


def dispatch_rows(node, dims: list[int], constants: dict) -> int | None:
    import onnx  # noqa: PLC0415

    def attribute(name: str, default):
        return next(
            (
                onnx.helper.get_attribute_value(a)
                for a in node.attribute
                if a.name == name
            ),
            default,
        )

    rank = len(dims)
    size = int(np.prod(dims))
    if node.op_type in LAST_AXIS_OPS:
        return size // dims[-1]
    if node.op_type == "InstanceNormalization":
        return dims[0] * dims[1]
    if node.op_type in REDUCE_OPS:
        axes = attribute("axes", None)
        if axes is None and len(node.input) > 1 and node.input[1]:
            if node.input[1] not in constants:
                return None
            axes = onnx.numpy_helper.to_array(constants[node.input[1]]).tolist()
        axes = range(rank) if axes is None else axes
        return size // int(np.prod([dims[axis] for axis in axes]))
    axis = attribute("axis", -1)
    axis = axis + rank if axis < 0 else axis
    if node.op_type in LEADING_AXES_OPS:
        return int(np.prod(dims[:axis]))
    return size // dims[axis]


def keep_band_split_fp32(model, epsilon: float) -> int:
    import onnx  # noqa: PLC0415

    graph = model.graph
    producer, _ = _io_maps(graph)
    inits = {i.name: i for i in graph.initializer}
    norms = [
        n
        for n in graph.node
        if n.op_type == "RMSNormalization" and "band_split" in n.input[1]
    ]
    front, cast = _band_split_front(producer, norms)
    for node in graph.node:
        if node.name in front:
            node.input[:] = [
                cast.input[0] if name == cast.output[0] else name for name in node.input
            ]

    fp16_gammas = {norm.input[1] for norm in norms}
    replace_at = {
        norm.name: _band_norm_to_fp32(norm, inits, graph, epsilon) for norm in norms
    }
    _rewrite_nodes(graph, {cast.name, *replace_at}, replace_at)
    kept_inits = [i for i in graph.initializer if i.name not in fp16_gammas]
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)

    front_tensors = {o for n in graph.node if n.name in front for o in n.output}
    for value in graph.value_info:
        if value.name in front_tensors:
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    return len(norms)


def keep_time_attention_fp32(model, frames: int) -> int:
    import onnx  # noqa: PLC0415
    from onnx import numpy_helper  # noqa: PLC0415

    graph = model.graph
    producer, consumers = _io_maps(graph)
    inits = {i.name: i for i in graph.initializer}
    dims = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in graph.value_info
    }
    replace_at, remove, values_fp32 = {}, set(), {}
    probabilities, scaled_scores, typed = set(), set(), []
    for softmax in [n for n in graph.node if n.op_type == "Softmax"]:
        if dims.get(softmax.input[0], [0])[-1] != frames:
            continue
        scale, (pv,) = producer[softmax.input[0]], consumers[softmax.output[0]]
        scores = producer[scale.input[0]]
        if (scale.op_type, scores.op_type, pv.op_type) != ("Mul", "MatMul", "MatMul"):
            raise RuntimeError(f"unexpected time attention around {softmax.name}")
        factor = float(numpy_helper.to_array(inits[scale.input[1]]))
        if not np.log2(factor).is_integer():
            raise RuntimeError(f"the scale {factor} of {softmax.name} is not 2^k")
        queries = onnx.helper.make_node(
            "Mul",
            [scores.input[0], scale.input[1]],
            [f"{scores.name}_scaled_queries"],
            name=f"{scores.name}_scale_queries",
        )
        scores.input[0] = queries.output[0]
        scores_fp32 = onnx.helper.make_node(
            "Cast",
            [scores.output[0]],
            [f"{scores.output[0]}_fp32"],
            name=f"{scores.name}_to_float32",
            to=onnx.TensorProto.FLOAT,
        )
        softmax.input[0] = scores_fp32.output[0]
        typed.append(
            onnx.helper.make_tensor_value_info(
                scores_fp32.output[0],
                onnx.TensorProto.FLOAT,
                dims[scale.output[0]],
            )
        )
        probabilities.add(softmax.output[0])
        scaled_scores.add(scale.output[0])
        replace_at[scores.name] = [queries, scores, scores_fp32]
        remove.update({scores.name, scale.name})

        values = pv.input[1]
        cast_values = []
        if values not in values_fp32:
            values_fp32[values] = f"{values}_fp32"
            cast_values.append(
                onnx.helper.make_node(
                    "Cast",
                    [values],
                    [values_fp32[values]],
                    name=f"{pv.name}_values_to_float32",
                    to=onnx.TensorProto.FLOAT,
                )
            )
        pv.input[1] = values_fp32[values]
        fp16_output = pv.output[0]
        pv.output[0] = f"{fp16_output}_fp32"
        replace_at[pv.name] = [
            *cast_values,
            pv,
            onnx.helper.make_node(
                "Cast",
                [pv.output[0]],
                [fp16_output],
                name=f"{pv.name}_to_float16",
                to=onnx.TensorProto.FLOAT16,
            ),
        ]
        remove.add(pv.name)
    _rewrite_nodes(graph, remove, replace_at)

    for value in graph.value_info:
        if value.name in probabilities:
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    kept = [v for v in graph.value_info if v.name not in scaled_scores]
    del graph.value_info[:]
    graph.value_info.extend([*kept, *typed])
    return len(probabilities)


def _band_split_front(producer: dict, norms: list) -> tuple[set, object]:
    front, casts, stack = set(), {}, [n.input[0] for n in norms]
    while stack:
        node = producer[stack.pop()]
        if node.op_type == "Cast":
            casts[node.name] = node
        elif node.op_type in ("GatherND", "Transpose", "Reshape", "Split"):
            if node.name not in front:
                front.add(node.name)
                stack.append(node.input[0])
        else:
            raise RuntimeError(f"unexpected {node.op_type} before the band norms")
    if len(casts) != 1:
        raise RuntimeError(
            f"expected one fp16 Cast before the band norms: {set(casts)}"
        )
    return front, next(iter(casts.values()))


def _band_norm_to_fp32(norm, inits: dict, graph, epsilon: float) -> list:
    import onnx  # noqa: PLC0415
    from onnx import numpy_helper  # noqa: PLC0415

    gamma = numpy_helper.to_array(inits[norm.input[1]]).astype(np.float32)
    graph.initializer.append(numpy_helper.from_array(gamma, f"{norm.input[1]}_fp32"))
    norm.input[1] = f"{norm.input[1]}_fp32"
    for attribute in norm.attribute:
        if attribute.name == "epsilon":
            attribute.f = epsilon
    fp16_output = norm.output[0]
    norm.output[0] = f"{fp16_output}_fp32"
    return [
        norm,
        onnx.helper.make_node(
            "Cast",
            [norm.output[0]],
            [fp16_output],
            name=f"{norm.name}_to_float16",
            to=onnx.TensorProto.FLOAT16,
        ),
    ]


def ensure_float32_outputs(model) -> int:
    import onnx  # noqa: PLC0415

    graph = model.graph
    rewritten = 0
    for output in graph.output:
        tensor_type = output.type.tensor_type
        if tensor_type.elem_type == onnx.TensorProto.FLOAT:
            continue

        original_name = output.name
        cast_input = f"{original_name}_pre_float32_cast"
        producer = next(
            (node for node in graph.node if original_name in node.output),
            None,
        )
        if producer is None:
            raise RuntimeError(f"cannot find producer for graph output {original_name}")
        for i, name in enumerate(producer.output):
            if name == original_name:
                producer.output[i] = cast_input

        graph.node.append(
            onnx.helper.make_node(
                "Cast",
                [cast_input],
                [original_name],
                name=f"{original_name}_to_float32",
                to=onnx.TensorProto.FLOAT,
            )
        )
        tensor_type.elem_type = onnx.TensorProto.FLOAT
        rewritten += 1
    return rewritten


if __name__ == "__main__":
    main()
