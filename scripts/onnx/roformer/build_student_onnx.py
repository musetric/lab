"""Export the core of the distilled vocals student with the teacher's recipe.

The student of musetric/musetric#1081 is published as
`musetric/mel-band-roformer-duality-student`: `model.safetensors` and
`config.yaml` of `PooledMelBandRoformer`. This script builds its core with
build_full_onnx.py, with the same flags as the teacher's, and changes only
what the student's shape needs:

- the pair loop as an hourglass: the high bands merged once, the stream
  averaged once to the deepest frame rate of the pooled pairs and repeated back
  at the end, the detail the pooled pairs cannot see kept at full rate and
  passed through the detail maps folded into one product. The average and the
  repeat commute, so this is the function of the PyTorch model, pair by pair,
  and no full-rate stream is held inside the pooled pairs;
- the windowed time attention as blocks of 2W queries over the 4W keys around
  them with a constant mask, the pooled time attentions in blocks of 32 query
  rows, the band maps as one MatMul and Add;
- the probabilities x values of the pooled time attentions, which see 275 and
  550 frames, padded to a multiple of 4 (pad_values);
- every rotary table a constant, sliced as the library slices its cache, so
  the exporter folds it and fuse_webgpu_kernels.py makes it one kernel;
- means and repeats as Reshape, Split, Add and Concat, which the product graph
  already has: a ReduceMean over the stream would dispatch millions of rows.

    uv run --group export python scripts/onnx/roformer/build_student_onnx.py \\
      --checkpoint tmp/models/student/model.safetensors \\
      --config tmp/models/student/config.yaml \\
      --output tmp/models/core_t1100.onnx \\
      --core-only --fuse-rmsnorm --attn-block 128 --all-fp16 --frames 1100 \\
      --skip-gate --split-rows 2 --stream-slabs 4

then `split_concat_webgpu.py`, the two audits and `fuse_webgpu_kernels.py` as
for the teacher (converting.md).
"""

import build_full_onnx as bfo  # type: ignore[import-not-found]
import numpy as np
import torch
import yaml
from einops import rearrange
from rotary_embedding_torch import apply_rotary_emb
from safetensors.torch import load_file

from musetric_toolkit.separate_audio.roformer.attend import Attend
from musetric_toolkit.separate_audio.roformer.pooled_mel_band_roformer import (
    BandMap,
    LocalAttend,
    PooledMelBandRoformer,
    Rotary,
)

POOLED_Q_BLOCK = 32
MERGED_SLABS = 4


def down(x: torch.Tensor, p: int, axis: int) -> torch.Tensor:
    """The mean of p neighbours as Reshape, Split and adds."""
    if p == 1:
        return x
    shape = list(x.shape)
    shape[axis : axis + 1] = [shape[axis] // p, p]
    parts = x.reshape(shape).split(1, dim=axis + 1)
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return (total * (1.0 / p)).squeeze(axis + 1)


def up(y: torch.Tensor, p: int, axis: int, n: int) -> torch.Tensor:
    """Each value p times as Concat and Reshape."""
    if p == 1:
        return y
    shape = list(y.shape)
    shape[axis : axis + 1] = [shape[axis] * p]
    return (
        torch.cat([y.unsqueeze(axis + 1)] * p, dim=axis + 1)
        .reshape(shape)
        .narrow(axis, 0, n)
    )


def merge_high(x: torch.Tensor, first: int) -> torch.Tensor:
    return torch.cat([x[:, :, :first], down(x[:, :, first:], 2, 2)], dim=2)


def unmerge_high(y: torch.Tensor, first: int) -> torch.Tensor:
    high = up(y[:, :, first:], 2, 2, 2 * (y.shape[2] - first))
    return torch.cat([y[:, :, :first], high], dim=2)


def rotate_from_table(
    self: Rotary, t: torch.Tensor, seq_dim=None, offset=0, scale=None
) -> torch.Tensor:
    """Rotary.rotate_queries_or_keys from a constant table one row longer than
    the sequence, filled by the warm-up forward and sliced as the library
    slices its cached frequencies."""
    n = t.shape[-2]
    if not hasattr(self, "cached_angles"):
        with torch.no_grad():
            if self.positions is None:
                pos = torch.arange(n + 1, dtype=torch.float32) * self.scale
            else:
                last = self.positions[-1:].cpu() + 1
                pos = torch.cat([self.positions.cpu(), last])
            angles = torch.einsum("n,f->nf", pos, self.freqs.float().cpu())
            angles = angles.repeat_interleave(2, dim=-1)
        self.register_buffer("cached_angles", angles.to(t.device), persistent=False)
    return apply_rotary_emb(self.cached_angles[:n], t, seq_dim=-2)


Rotary.rotate_queries_or_keys = rotate_from_table


class BlockedWindow(torch.nn.Module):
    """Attention within +-W positions as blocks: 2W queries attend to their
    own block and half of each neighbour, 4W keys, in one batched MatMul; a
    constant bias keeps exactly |i - j| <= W and drops the padding. The
    softmax rows are chunked by the wrapped Attend."""

    def __init__(self, base: Attend, window: int) -> None:
        super().__init__()
        self.base, self.window = base, window

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        n, d, w = q.shape[-2], q.shape[-1], self.window
        block = 2 * w
        blocks = -(-n // block)
        lead = q.shape[:-2]
        pad_front = torch.zeros(*lead, w, d, dtype=k.dtype, device=k.device)
        pad_back = torch.zeros(
            *lead, blocks * block - n + w, d, dtype=k.dtype, device=k.device
        )

        def windows(t: torch.Tensor) -> torch.Tensor:
            t = torch.cat([pad_front, t, pad_back], dim=-2)
            t = t.reshape(*lead, blocks + 1, block, d)
            return torch.cat([t[..., :blocks, :, :], t[..., 1:, :, :]], dim=-2)

        tail = pad_back[..., : blocks * block - n, :]
        qb = torch.cat([q, tail], dim=-2).reshape(*lead, blocks, block, d)
        sim = torch.matmul(qb * d**-0.5, windows(k).transpose(-1, -2))
        sim = sim + self.bias(n, blocks, block, q)
        out = torch.matmul(self.base.softmax(sim), windows(v))
        return out.reshape(*lead, blocks * block, d)[..., :n, :]

    def bias(self, n: int, blocks: int, block: int, like: torch.Tensor) -> torch.Tensor:
        # a buffer filled by the warm-up forward, so the graph holds a constant
        # rather than Abs, LessOrEqual, And and Where nodes
        name = f"window_bias_{n}"
        if not hasattr(self, name):
            with torch.no_grad():
                i = torch.arange(blocks * block).reshape(blocks, block, 1)
                j = torch.arange(blocks).reshape(blocks, 1, 1) * block
                j = j - self.window + torch.arange(2 * block)
                ok = ((i - j).abs() <= self.window) & (j >= 0) & (j < n)
                bias = torch.where(ok, 0.0, -30000.0)
            self.register_buffer(name, bias.to(like.device), persistent=False)
        return getattr(self, name).to(like.dtype)


def load_model(checkpoint, config) -> PooledMelBandRoformer:
    with open(config) as f:
        cfg = yaml.safe_load(f)
    model = PooledMelBandRoformer(**cfg["model"]).eval()
    model.load_state_dict(load_file(checkpoint))
    for module in model.modules():
        if isinstance(module, Attend):
            module.flash = False
    for module in list(model.modules()):
        if isinstance(getattr(module, "attend", None), LocalAttend):
            window = module.attend.window
            module.attend = BlockedWindow(Attend(dropout=0.0, flash=False), window)
    # the full-rate time attentions share one rotary table, as the teacher's
    # share the library's cache: one constant for the graph, not one each
    shared = {}
    for time_t, _ in model.layers:
        for attn, _ in time_t.layers:
            rotary = attn.rotary_embed
            if rotary.scale == 1 and rotary.positions is None:
                key = tuple(rotary.freqs.tolist())
                attn.rotary_embed = shared.setdefault(key, rotary)
    return model


def grouped_matmul(x, w, b, slabs):
    """x [b, t, f, d] times w [d, o] as one MatMul over rows [1, t * f, d]."""
    t = x.shape[1]
    rows = rearrange(x, "b t f d -> b (t f) d")
    out = rearrange(torch.matmul(rows, w.to(x.device)), "b (t f) d -> b t f d", t=t)
    return out if b is None else out + b.to(x.device)


def run_pair(
    time_t, freq_t, x: torch.Tensor, slabs: int, band_map=None
) -> torch.Tensor:
    """One pair over stream slabs: the time transformer over band slabs, then
    the band transformer over frame slabs, or the band map x (I + W) + b."""
    outs = []
    for slab in x.chunk(slabs, dim=2):
        bands = slab.shape[2]
        rows = rearrange(slab, "b t f d -> b (f t) d")
        rows = bfo._transformer_over_rows(time_t, rows, bands)
        outs.append(rearrange(rows, "b (f t) d -> b t f d", f=bands))
    x = torch.cat(outs, dim=2)
    if band_map is not None:
        return grouped_matmul(x, *band_map, slabs)
    outs = []
    for slab in x.chunk(slabs, dim=1):
        frames = slab.shape[1]
        rows = rearrange(slab, "b t f d -> b (t f) d")
        rows = bfo._transformer_over_rows(freq_t, rows, frames)
        outs.append(rearrange(rows, "b (t f) d -> b t f d", t=frames))
    return torch.cat(outs, dim=1)


def detail_maps(model: PooledMelBandRoformer) -> tuple[dict, torch.Tensor]:
    """A path x Wx + c Wc + b over features ("x" the stream, "c" its pooled
    view, "d" = x - c) as d Wa + c (Wa + Wb) + b: the pooled part joins the
    pair's change, and the detail parts of all path pairs fold into one
    product of (I + Wa) applied to the detail on the way back."""
    pooled, product = {}, None
    for i, kinds in enumerate(model.detail_kinds):
        if not kinds:
            continue
        w, b = (
            model.detail_paths[f"w{i}"].detach(),
            model.detail_paths[f"b{i}"].detach(),
        )
        blocks = dict(zip(kinds, w.split(w.shape[-1], dim=-2), strict=True))
        zero = torch.zeros_like(next(iter(blocks.values())))
        wa = blocks.get("x", zero) + blocks.get("d", zero)
        wb = blocks.get("c", zero) - blocks.get("d", zero)
        pooled[i] = (wa + wb, b)
        step = wa + torch.eye(wa.shape[-1], dtype=wa.dtype)
        product = step if product is None else product @ step
    return pooled, product


def prepare(model: PooledMelBandRoformer) -> None:
    """Once, in the eager warm-up after build_full_onnx blocks attention: the
    pooled time attentions in blocks of 32 query rows, and the maps as
    constants of the graph, computed outside the trace."""
    for i, k in enumerate(model.pool):
        if k > 1:
            for attn, _ in model.layers[i][0].layers:
                attn.attend.q_block = POOLED_Q_BLOCK
    model.hourglass = detail_maps(model)
    model.band_maps = {}
    for i, (_, freq_t) in enumerate(model.layers):
        if isinstance(freq_t, BandMap):
            w, b = freq_t.linear.weight.detach(), freq_t.linear.bias.detach()
            model.band_maps[i] = (w.T + torch.eye(w.shape[0], dtype=w.dtype), b)


class Hourglass:
    """The stream of the pair loop: `m` at the current rate, the full-rate
    `detail` it lacks, the merged bands' `band_detail`, and the folded path
    product still to be applied to both."""

    def __init__(self, x: torch.Tensor, first: int) -> None:
        self.first = first
        self.m = merge_high(x, first)
        self.band_detail = x[:, :, first:] - unmerge_high(self.m, first)[:, :, first:]
        self.frames = self.m.shape[1]
        self.rate, self.detail, self.product = 1, None, None

    def apply_product(self, slabs: int) -> None:
        if self.product is not None:
            self.detail = grouped_matmul(self.detail, self.product, None, slabs)
            self.band_detail = grouped_matmul(
                self.band_detail, self.product, None, slabs
            )
            self.product = None

    def to_rate(self, k: int, slabs: int) -> None:
        if self.rate == 1:
            lo = down(self.m, k, 1)
            self.detail = self.m - up(lo, k, 1, self.frames)
            self.m = lo
        else:
            self.apply_product(slabs)
            level = down(self.detail, k, 1)
            self.detail = self.detail - up(level, k, 1, self.frames)
            half = self.rate // k
            self.m = up(self.m, half, 1, self.m.shape[1] * half) + level
        self.rate = k

    def full_rate(self, slabs: int) -> torch.Tensor:
        if self.rate > 1:
            self.apply_product(slabs)
            self.m = up(self.m, self.rate, 1, self.frames) + self.detail
        merged = unmerge_high(self.m, self.first)
        high = merged[:, :, self.first :] + self.band_detail
        return torch.cat([merged[:, :, : self.first], high], dim=2)


def hourglass_net_forward(
    model: PooledMelBandRoformer, stft_repr: torch.Tensor, slabs: int
) -> torch.Tensor:
    """net_forward of the student: full-rate pairs, then the pairs at the
    deepest rate, then the pairs at half of it, on one merge of the bands."""
    if not hasattr(model, "hourglass"):
        prepare(model)
    pooled_maps, path_product = model.hourglass
    batch = stft_repr.shape[0]
    freq_indices = model.freq_indices.to(stft_repr.device)
    batch_arange = torch.arange(batch, device=stft_repr.device)[..., None]
    x = stft_repr[batch_arange, freq_indices]
    x = model.band_split(rearrange(x, "b f t c -> b t (f c)"))

    stream = Hourglass(x, model.merged_bands_from[0])
    for i, ((time_t, freq_t), k) in enumerate(
        zip(model.layers, model.pool, strict=True)
    ):
        if k != stream.rate:
            stream.to_rate(k, slabs)
        pair_slabs = max(1, MERGED_SLABS // k)
        y = run_pair(time_t, freq_t, stream.m, pair_slabs, model.band_maps.get(i))
        if i in pooled_maps:
            y = y + grouped_matmul(stream.m, *pooled_maps[i], max(1, slabs // k))
            stream.product = path_product
        stream.m = y
    x = stream.full_rate(slabs)

    frames = x.shape[1]
    pooled = down(x, model.mask_pool, 1)
    masks = torch.stack(
        [up(fn(pooled), model.mask_pool, 1, frames) for fn in model.mask_estimators],
        dim=1,
    )
    return rearrange(masks, "b n t (f c) -> b n f t c", c=2)


_keep_time_attention_fp32 = bfo.keep_time_attention_fp32
SHAPE = {}


def pad_values(model, frames: int) -> int:
    """The probabilities x values MatMul of the time attentions over `frames`
    keys with its reduction padded to a multiple of 4: zero columns appended to
    the probabilities after the softmax, zero rows to the values. The zeros add
    nothing, exactly. On the Adreno 750, onnxruntime-web computes that MatMul
    with visibly less precision when its reduction length is not a multiple of
    4, and the pooled time attentions see 275 and 550 frames. The scores and
    the softmax stay as they are: padded keys there cost the Adreno 660 its
    precision."""
    import onnx  # noqa: PLC0415
    from onnx import numpy_helper  # noqa: PLC0415

    graph = model.graph
    _, consumers = bfo._io_maps(graph)
    dims = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in graph.value_info
    }
    pad = (-frames) % 4
    replace_at, remove, padded = {}, set(), {}

    def zeros(shape: list[int], name: str) -> str:
        if name not in padded:
            values = np.zeros(shape, dtype=np.float32)
            graph.initializer.append(numpy_helper.from_array(values, name))
            padded[name] = name
        return name

    def concat(inputs: list[str], axis: int, name: str) -> onnx.NodeProto:
        return onnx.helper.make_node(
            "Concat", inputs, [name], name=f"{name}_concat", axis=axis
        )

    for softmax in [n for n in graph.node if n.op_type == "Softmax"]:
        probabilities = dims.get(softmax.input[0], [0])
        if probabilities[-1] != frames:
            continue
        (pv,) = consumers[softmax.output[0]]
        values = pv.input[1]
        lead = probabilities[:-1]
        nodes = [
            concat(
                [
                    softmax.output[0],
                    zeros([*lead, pad], "zero_columns_" + "x".join(map(str, lead))),
                ],
                -1,
                f"{softmax.output[0]}_padded",
            )
        ]
        if f"{values}_padded" not in padded:
            # the values as the fp32 pass cast them: their shape is the fp16 input's
            shape = dims.get(values) or dims[values.removesuffix("_fp32")]
            rows = [*shape[:-2], pad, shape[-1]]
            nodes.append(
                concat(
                    [values, zeros(rows, "zero_rows_" + "x".join(map(str, rows)))],
                    -2,
                    f"{values}_padded",
                )
            )
            padded[f"{values}_padded"] = f"{values}_padded"
        pv.input[0] = nodes[0].output[0]
        pv.input[1] = f"{values}_padded"
        replace_at[pv.name] = [*nodes, pv]
        remove.add(pv.name)
    bfo._rewrite_nodes(graph, remove, replace_at)
    return len(replace_at)


def keep_time_attention_fp32(model, frames: int) -> int:
    """The full-attention time softmaxes in fp32 at every rate they run at,
    their probabilities x values padded where the rate is not a multiple of
    4; the windowed ones are left as they are."""
    windows = zip(SHAPE["pool"], SHAPE["time_window"], strict=True)
    rates = {frames // k for k, window in windows if not window}
    kept = sum(_keep_time_attention_fp32(model, n) for n in sorted(rates))
    for n in sorted(rates):
        if n % 4:
            pad_values(model, n)
    return kept


def main() -> None:
    import sys  # noqa: PLC0415

    config = sys.argv[sys.argv.index("--config") + 1]
    with open(config) as f:
        SHAPE.update(yaml.safe_load(f)["model"])
    bfo.load_model = load_model
    bfo._stream_slab_net_forward = hourglass_net_forward
    bfo.keep_time_attention_fp32 = keep_time_attention_fp32
    bfo.main()


if __name__ == "__main__":
    main()
