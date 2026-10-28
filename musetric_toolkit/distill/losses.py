from dataclasses import dataclass

import torch

from musetric_toolkit.distill.precision import PRECISION
from musetric_toolkit.distill.teacher import HOP, band_features, masks_from, run_layers

STFT_SIZES = (4096, 2048, 1024, 512, 256)
STFT_HOP = 147
MIN_FFT = 2048
PAUSE_FRAME = 4410
GATE_FRAME = 2205
SPECTRUM_FFT = 2048
PAUSE_RATIO = 1e-3
SOUNDING = 1e-7
SUNG_LEVEL = 0.9


def multi_stft_l1(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.l1_loss(est, ref)
    e = est.reshape(-1, est.shape[-1])
    r = ref.reshape(-1, ref.shape[-1])
    for size in STFT_SIZES:
        options = dict(
            n_fft=max(size, MIN_FFT),
            win_length=size,
            hop_length=STFT_HOP,
            window=torch.hann_window(size, device=est.device),
            return_complex=True,
        )
        spectra = torch.stft(e, **options), torch.stft(r, **options)
        loss = loss + torch.nn.functional.l1_loss(*spectra)
    return loss


def frame_energy(s: torch.Tensor, b: int, n: int, size: int) -> torch.Tensor:
    s = s.reshape(b, -1, s.shape[-1])[..., : n * size].float()
    return (s**2).mean(1).reshape(b, n, size).mean(-1)


def pause_excess(
    x: torch.Tensor, y: torch.Tensor, vocals: torch.Tensor
) -> torch.Tensor:
    b = x.shape[0]
    n = x.shape[-1] // PAUSE_FRAME
    ex = frame_energy(x, b, n, PAUSE_FRAME)
    ey = frame_energy(y, b, n, PAUSE_FRAME)
    es = frame_energy(vocals, b, n, PAUSE_FRAME)
    pause = (ey < PAUSE_RATIO * ex) & (ex > SOUNDING)
    tau = 1e-6 * ex + 1e-12
    excess = torch.relu(10 * torch.log10(es + tau) - 10 * torch.log10(ey + tau))
    total = torch.where(pause, excess, torch.zeros_like(excess)).sum()
    return total / pause.sum().clamp(min=1)


def quiet_gate(x, y, truth, db: float) -> torch.Tensor:
    b = x.shape[0]
    n = x.shape[-1] // GATE_FRAME

    def energy(s):
        return (
            (s[..., : n * GATE_FRAME].float() ** 2)
            .mean(1)
            .reshape(b, n, GATE_FRAME)
            .mean(-1)
        )

    ey, ex = energy(y), energy(x)
    ratio = 10 ** (-db / 10)
    level = torch.quantile(ey, SUNG_LEVEL, dim=1, keepdim=True)
    quiet = (ey < ratio * level) & (ey < ratio * ex)
    if truth is not None:
        quiet &= ~truth[:, None]
    keep = torch.nn.functional.interpolate(
        (~quiet).float()[:, None],
        size=n * GATE_FRAME,
        mode="linear",
        align_corners=False,
    )[:, 0]
    return torch.nn.functional.pad(keep, (0, x.shape[-1] - keep.shape[1]), value=1.0)


def mark_weights(marked: torch.Tensor, weight: float) -> torch.Tensor:
    ramp = torch.nn.functional.avg_pool1d(
        marked[:, None].float(),
        GATE_FRAME,
        stride=1,
        padding=GATE_FRAME // 2,
        count_include_pad=False,
    )[:, 0]
    return 1 - (1 - weight) * ramp


def magnitude(s: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    flat = s.reshape(-1, s.shape[-1]).float()
    return torch.stft(flat, SPECTRUM_FFT, HOP, window=window, return_complex=True).abs()


def marked_excess(x, y, vocals, marked) -> torch.Tensor:
    window = torch.hann_window(SPECTRUM_FFT, device=x.device)
    floor = 1e-3 * magnitude(x, window) + 1e-6
    log_vocals = torch.log(magnitude(vocals, window) + floor)
    excess = torch.relu(log_vocals - torch.log(magnitude(y, window) + floor))
    frames = marked[:, ::HOP][:, : excess.shape[-1]].float()
    frames = frames.repeat_interleave(x.shape[1], 0)
    total = (excess[..., : frames.shape[-1]] * frames[:, None]).sum()
    return total / (frames.sum() * excess.shape[1]).clamp(min=1)


def voice_balance(x, y, vocals) -> tuple[torch.Tensor, torch.Tensor]:
    window = torch.hann_window(SPECTRUM_FFT, device=x.device)
    mx, my = magnitude(x, window), magnitude(y, window)
    floor = 1e-3 * mx + 1e-6
    d = torch.log(magnitude(vocals, window) + floor) - torch.log(my + floor)
    voice = my >= magnitude(x - y, window)
    leak = torch.where(voice, 0.0, torch.relu(d)).flatten(1).mean(1)
    deficit = torch.where(voice, torch.relu(-d), 0.0).flatten(1).mean(1)
    batch = x.shape[0]
    return leak.reshape(batch, -1).mean(1), deficit.reshape(batch, -1).mean(1)


@dataclass
class LossWeights:
    hidden: float = 0.0
    mask: float = 0.0
    wave_target: str = "stored"
    pause: float = 0.0
    marks: float = 1.0
    marks_excess: float = 0.0
    truth_excess: float = 0.0
    truth_leak: float = 0.0
    voice_deficit: float = 0.0


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    truth: torch.Tensor | None = None
    marked: torch.Tensor | None = None
    gate: torch.Tensor | None = None


def frame_weights(weights: torch.Tensor, frames: int) -> torch.Tensor:
    per_frame = weights[:, :frames]
    return torch.nn.functional.pad(
        per_frame, (0, frames - per_frame.shape[1]), value=1.0
    )


def teacher_targets(model, teacher, features, need_masks: bool, gate):
    _, targets = run_layers(features, teacher)
    heads = getattr(model, "teacher_heads", None)
    target_masks = masks_from(model, targets[-1], heads) if need_masks else None
    if gate is not None and target_masks is not None:
        frames = frame_weights(gate, target_masks.shape[3])
        target_masks.mul_(frames[:, None, None, :, None].to(target_masks.dtype))
    return targets, target_masks


def mask_loss(masks, target_masks, w, truth):
    m, t = masks.float(), target_masks.float()
    mean = 1.0
    if w is not None:
        frames = frame_weights(w[:, ::HOP], m.shape[3])
        root = frames.sqrt()[:, None, None, :, None]
        m, t, mean = m * root, t * root, frames.mean()
    ones = torch.ones(m.shape[0], device=m.device)
    keep = ones if truth is None else (~truth).float()
    cosine = torch.nn.functional.cosine_similarity(m.flatten(1), t.flatten(1))
    per = ((m - t) ** 2).flatten(1).mean(1) / mean + 1 - cosine
    return (keep * per).sum() / keep.sum().clamp(min=1)


def extra_terms(loss, batch: Batch, y, vocals, weights: LossWeights):
    x, truth, marked = batch.x, batch.truth, batch.marked
    if weights.pause > 0:
        loss = loss + weights.pause * pause_excess(x, y, vocals)
    if weights.marks_excess > 0 and marked is not None:
        loss = loss + weights.marks_excess * marked_excess(x, y, vocals, marked)
    has_truth = truth is not None and bool(truth.any())
    if weights.truth_excess > 0 and has_truth:
        whole = truth[:, None].float().expand(-1, x.shape[-1])
        loss = loss + weights.truth_excess * marked_excess(x, y, vocals, whole)
    if weights.truth_leak > 0 or weights.voice_deficit > 0:
        leak, deficit = voice_balance(x, y, vocals)
        if weights.truth_leak > 0 and has_truth:
            t = truth.float()
            loss = loss + weights.truth_leak * (leak * t).sum() / t.sum()
        loss = loss + weights.voice_deficit * deficit.mean()
    return loss


def hidden_step(model, teacher, pairing, batch: Batch, weights: LossWeights):
    x, y, truth = batch.x, batch.y, batch.truth
    stft = model.encode_stft(x)
    run = getattr(model, "run_stack", None)
    need_masks = weights.mask > 0 or weights.wave_target == "teacher"
    with torch.autocast("cuda", dtype=PRECISION.dtype):
        with torch.no_grad():
            features = band_features(model, stft)
            targets, target_masks = teacher_targets(
                model, teacher, features, need_masks, batch.gate
            )
        out, outs = run(features) if run else run_layers(features, model.layers)
        masks = masks_from(model, out)
    vocals = model.decode_istft(stft, masks.float(), length=x.shape[-1])
    if weights.wave_target == "teacher":
        answer = model.decode_istft(stft, target_masks.float(), length=x.shape[-1])
        y = answer if truth is None else torch.where(truth[:, None, None], y, answer)
    hidden = torch.stack(
        [
            ((outs[i].float() - targets[j].float()) ** 2).mean()
            / (targets[j].float() ** 2).mean().clamp(min=1e-8)
            for i, j in enumerate(pairing)
        ]
    ).mean()
    w = None
    if batch.marked is not None and weights.marks < 1:
        w = mark_weights(batch.marked, weights.marks)
    if w is None:
        loss = multi_stft_l1(vocals, y) + weights.hidden * hidden
    else:
        loss = (
            multi_stft_l1(vocals * w[:, None], y * w[:, None]) + weights.hidden * hidden
        )
    if weights.mask > 0:
        loss = loss + weights.mask * mask_loss(masks, target_masks, w, truth)
    return extra_terms(loss, batch, y, vocals, weights), hidden.item()
