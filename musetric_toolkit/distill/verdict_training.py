import logging
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from musetric_toolkit.distill.excluded import add_excluded, read_excluded
from musetric_toolkit.distill.losses import multi_stft_l1
from musetric_toolkit.distill.precision import PRECISION
from musetric_toolkit.distill.teacher import load_teacher, warm_rotary
from musetric_toolkit.distill.verdict_sampler import VerdictOptions, VerdictSampler
from musetric_toolkit.distill.verdicts import read_verdicts
from musetric_toolkit.distill.voices import load_lead_backing
from musetric_toolkit.distill.voices_training import (
    Prefetcher,
    lr_scale,
    trained_pairs,
)

LOG = logging.getLogger(__name__)

LOG_EVERY = 25
FIELDS = ("x", "gate", "backing")


def trainable(model, pairs: str) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad_(False)
    params = list(model.split_estimator.parameters())
    params += list(model.mask_estimators.parameters())
    params += list(trained_pairs(model, pairs).values())
    for param in params:
        param.requires_grad_(True)
    return params


def verdict_loss(model, teacher, batch, args) -> tuple[torch.Tensor, dict]:
    x, gate, backing = batch
    gate = gate.unsqueeze(1)
    stft = model.encode_stft(x)
    with torch.no_grad(), torch.autocast("cuda", dtype=PRECISION.dtype):
        teacher_masks = teacher.net_forward(stft)
    with torch.no_grad():
        teacher_vocals = teacher.decode_istft(
            stft, teacher_masks.float(), length=x.shape[-1]
        )
    vocals_ref = teacher_vocals * gate
    backing_ref = backing * gate
    with torch.autocast("cuda", dtype=PRECISION.dtype):
        masks = model.heads(model.trunk(stft))
    vocals_spec, backing_spec = model.stem_spectra(stft, masks)
    waves = model.istft_stems(
        torch.cat([vocals_spec, backing_spec], dim=1), x.shape[-1]
    )
    vocals, backing_est = waves[:, 0], waves[:, 1]
    parts = dict(
        backing=multi_stft_l1(backing_est, backing_ref),
        lead=multi_stft_l1(vocals - backing_est, vocals_ref - backing_ref),
        vocals=args.vocals_weight * multi_stft_l1(vocals, vocals_ref),
    )
    return sum(parts.values()), {k: v.item() for k, v in parts.items()}


def save(out: Path, model, optimizer, step: int, elapsed: float) -> None:
    state = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        step=step,
        elapsed=elapsed,
    )
    torch.save(state, out / "model.tmp")
    (out / "model.tmp").replace(out / "model.pt")


def write_snapshot(folder: Path, model, release: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(release / "config.yaml", folder / "config.yaml")
    weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(weights, folder / "model.safetensors")


def make_sampler(args, step: int) -> VerdictSampler:
    excluded = read_excluded(args.excluded)
    withheld = frozenset(args.withheld.read_text("utf-8").split()) | excluded
    verdicts = read_verdicts(args.verdicts)
    options = VerdictOptions(
        remix=args.remix,
        backing=args.backing_share,
        pell=args.pell_share,
        share=args.share,
    )
    return VerdictSampler(
        args.labels,
        args.seed + step,
        options,
        [row for row in verdicts if row["id"] not in excluded],
        withheld,
    )


def train_verdicts(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    teacher = load_teacher(args.teacher).cuda().eval()
    warm_rotary(teacher)
    for param in teacher.parameters():
        param.requires_grad_(False)
    model = load_lead_backing(args.release).cuda()
    params = trainable(model, args.pairs)
    optimizer = torch.optim.AdamW(
        [dict(params=params, base=args.lr)],
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=0.01,
    )
    step, before = 0, 0.0
    if args.resume and (args.out / "model.pt").exists():
        state = torch.load(
            args.out / "model.pt", map_location="cuda", weights_only=False
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, before = state["step"], state["elapsed"]
    sampler = make_sampler(args, step)
    prefetch = Prefetcher(sampler, args.batch, FIELDS)
    model.train()
    millions = sum(param.numel() for param in params) / 1e6
    LOG.info(f"{millions:.1f} M trainable, {len(sampler.silent)} clips")
    start, logged, first = time.perf_counter(), [], step
    next_save = before + args.save_minutes * 60
    pending = [minutes for minutes in sorted(args.snapshots) if minutes * 60 > before]
    while (elapsed := before + time.perf_counter() - start) < args.minutes * 60:
        for group in optimizer.param_groups:
            group["lr"] = group["base"] * lr_scale(args, elapsed, step)
        loss, parts = verdict_loss(model, teacher, prefetch.next(), args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        logged.append(parts)
        step += 1
        if step % LOG_EVERY == 0:
            mean = {k: round(sum(p[k] for p in logged) / len(logged), 4) for k in parts}
            rate = (elapsed - before) / max(step - first, 1)
            memory = torch.cuda.max_memory_allocated() / 2**30
            LOG.info(
                f"step {step} {mean} {elapsed / 60:.1f} min, "
                f"{rate:.2f} s/step, {memory:.1f} GiB"
            )
            logged = []
        if elapsed >= next_save:
            save(args.out, model, optimizer, step, elapsed)
            next_save = elapsed + args.save_minutes * 60
        if pending and elapsed >= pending[0] * 60:
            folder = args.out / "snapshots" / f"m{pending.pop(0):03d}"
            write_snapshot(folder, model, args.release)
    save(args.out, model, optimizer, step, elapsed)
    write_snapshot(args.out / "final", model, args.release)


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "train-verdicts",
        parents=[common],
        help="refine a released student on a listener's clips without a voice",
    )
    parser.add_argument(
        "--release",
        required=True,
        type=Path,
        help="Directory of the released student: config.yaml, model.safetensors",
    )
    parser.add_argument(
        "--teacher",
        required=True,
        type=Path,
        help="Directory with the vocals teacher's config.yaml and model.ckpt",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Directory of label-voices: the mixes and KARA 2's backing",
    )
    parser.add_argument(
        "--verdicts", required=True, type=Path, help="verdicts.csv of `verdicts`"
    )
    parser.add_argument(
        "--withheld", required=True, type=Path, help="withheld.txt of `verdicts`"
    )
    add_excluded(parser)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--minutes", type=float, default=120.0)
    parser.add_argument(
        "--snapshots",
        type=int,
        nargs="*",
        default=[20, 60],
        help="Minutes of training at which to write a release into snapshots/",
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--pairs", default="45")
    parser.add_argument("--share", type=float, default=0.2)
    parser.add_argument("--vocals-weight", type=float, default=3.0)
    parser.add_argument("--remix", type=float, default=0.4)
    parser.add_argument("--backing-share", type=float, default=0.5)
    parser.add_argument("--pell-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-minutes", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(run=train_verdicts)
