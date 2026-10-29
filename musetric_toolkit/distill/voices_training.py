import json
import logging
import math
import queue
import threading
import time
from pathlib import Path

import torch

from musetric_toolkit.distill.losses import multi_stft_l1
from musetric_toolkit.distill.precision import PRECISION
from musetric_toolkit.distill.voices import load_lead_backing, load_split
from musetric_toolkit.distill.voices_sampler import VoicesOptions, VoicesSampler
from musetric_toolkit.distill.voices_scores import held_out_scores, summary

LOG = logging.getLogger(__name__)

LOG_EVERY = 25


class Prefetcher:
    def __init__(self, sampler: VoicesSampler, size: int, depth: int = 4):
        self.queue: queue.Queue = queue.Queue(depth)
        self.sampler, self.size = sampler, size
        threading.Thread(target=self.fill, daemon=True).start()

    def fill(self) -> None:
        while True:
            examples = [self.sampler.example() for _ in range(self.size)]
            self.queue.put(examples)

    def next(self):
        examples = self.queue.get()
        return [
            torch.stack([getattr(e, f) for e in examples]).cuda(non_blocking=True)
            for f in ("x", "vocals", "backing")
        ]


def trained_pairs(model, pairs: str) -> dict[str, torch.nn.Parameter]:
    chosen = {}
    for name, param in model.named_parameters():
        parts = name.split(".")
        if parts[0] == "layers" and parts[1] in pairs:
            chosen[name] = param
        if parts[0] == "detail_paths" and parts[1][1:] in pairs:
            chosen[name] = param
    return chosen


def split_parameters(model, pairs: str) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad_(False)
    params = list(model.split_estimator.parameters())
    params += list(trained_pairs(model, pairs).values())
    for param in params:
        param.requires_grad_(True)
    return params


def lr_scale(args, elapsed: float, step: int) -> float:
    if step < args.warmup:
        return (step + 1) / args.warmup
    progress = min(1.0, elapsed / (args.minutes * 60))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def param_groups(model, args) -> list[dict]:
    pairs = list(trained_pairs(model, args.train_pairs).values())
    groups = [dict(params=list(model.split_estimator.parameters()), base=args.lr)]
    if pairs:
        groups.append(dict(params=pairs, base=args.lr_pairs))
    return groups


def voices_loss(model, batch, args) -> torch.Tensor:
    x, vocals, backing = batch
    stft = model.encode_stft(x)
    with torch.autocast("cuda", dtype=PRECISION.dtype):
        with torch.set_grad_enabled(bool(args.train_pairs)):
            trunk = model.trunk(stft)
        masks = model.heads(trunk)
    vocals_spec, backing_spec = model.stem_spectra(stft, masks)
    waves = model.istft_stems(
        torch.cat([vocals_spec, backing_spec], dim=1), x.shape[-1]
    )
    vocals_est, backing_est = waves[:, 0], waves[:, 1]
    lead_est = vocals_est - backing_est
    lead_ref = vocals - backing
    loss = multi_stft_l1(backing_est, backing) + multi_stft_l1(lead_est, lead_ref)
    if args.vocals_weight > 0:
        loss = loss + args.vocals_weight * multi_stft_l1(vocals_est, vocals)
    return loss


def save(out: Path, model, optimizer, progress: tuple[int, float], args) -> None:
    step, elapsed = progress
    state = dict(
        split=model.split_estimator.state_dict(),
        pairs={
            k: v.detach() for k, v in trained_pairs(model, args.train_pairs).items()
        },
        optimizer=optimizer.state_dict(),
        step=step,
        elapsed=elapsed,
        split_hidden=args.split_hidden,
        student=str(args.student),
    )
    torch.save(state, out / "split.tmp")
    (out / "split.tmp").replace(out / "split.pt")


def evaluate(out: Path, model, args, step: int, elapsed: float) -> None:
    rows = held_out_scores(model, args.labels, args.eval_tracks)
    row = dict(step=step, minutes=elapsed / 60, **summary(rows))
    with open(out / "evals.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    LOG.info(f"eval {row}")
    model.train()


def restore(model, optimizer, args) -> tuple[int, float]:
    if not args.init:
        return 0, 0.0
    state = torch.load(args.init, map_location="cuda", weights_only=False)
    load_split(model, state)
    if not args.resume:
        return 0, 0.0
    optimizer.load_state_dict(state["optimizer"])
    return state["step"], state["elapsed"]


def update(model, optimizer, params, batch, args) -> float:
    loss = voices_loss(model, batch, args)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    return loss.item()


def train_voices(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_lead_backing(args.student, args.split_hidden).cuda()
    params = split_parameters(model, args.train_pairs)
    optimizer = torch.optim.AdamW(
        param_groups(model, args), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01
    )
    step, elapsed_before = restore(model, optimizer, args)
    first_step = step
    options = VoicesOptions(
        remix=args.remix, backing=args.backing_share, pell=args.pell_share
    )
    prefetch = Prefetcher(VoicesSampler(args.labels, args.seed, options), args.batch)
    model.train()
    if args.eval_tracks and not args.no_start_eval:
        evaluate(args.out, model, args, step, elapsed_before)
    start, paused = time.perf_counter(), 0.0
    next_save, next_eval = 0.0, elapsed_before + args.eval_minutes * 60
    losses = []
    while True:
        elapsed = elapsed_before + time.perf_counter() - start - paused
        if elapsed >= args.minutes * 60:
            break
        for group in optimizer.param_groups:
            group["lr"] = group["base"] * lr_scale(args, elapsed, step)
        losses.append(update(model, optimizer, params, prefetch.next(), args))
        step += 1
        if step % LOG_EVERY == 0:
            mean = sum(losses[-LOG_EVERY:]) / LOG_EVERY
            rate = (elapsed - elapsed_before) / max(step - first_step, 1)
            LOG.info(
                f"step {step} loss {mean:.5f} lr {optimizer.param_groups[0]['lr']:.2e} "
                f"{elapsed / 60:.1f} min, {rate:.2f} s/step"
            )
        if elapsed >= next_save:
            save(args.out, model, optimizer, (step, elapsed), args)
            next_save = elapsed + args.save_minutes * 60
        if args.eval_tracks and elapsed >= next_eval:
            began = time.perf_counter()
            evaluate(args.out, model, args, step, elapsed)
            paused += time.perf_counter() - began
            next_eval = elapsed + args.eval_minutes * 60
    elapsed = elapsed_before + time.perf_counter() - start - paused
    save(args.out, model, optimizer, (step, elapsed), args)
    if args.eval_tracks:
        evaluate(args.out, model, args, step, elapsed)


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "train-voices",
        parents=[common],
        help="train the lead/backing split of the vocals student on frozen layers",
    )
    parser.add_argument("--student", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--init", type=Path)
    parser.add_argument(
        "--train-pairs",
        default="",
        help="Transformer pairs that train with the split, such as 45; none by default",
    )
    parser.add_argument("--vocals-weight", type=float, default=0.0)
    parser.add_argument("--lr-pairs", type=float, default=5e-5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--split-hidden",
        type=int,
        default=0,
        help="0: the split reads the vocals head's hidden layer; N: its own layer of N",
    )
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--remix", type=float, default=0.4)
    parser.add_argument("--backing-share", type=float, default=0.5)
    parser.add_argument("--pell-share", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-minutes", type=float, default=5.0)
    parser.add_argument("--eval-minutes", type=float, default=10.0)
    parser.add_argument("--eval-tracks", type=int, default=60)
    parser.add_argument("--no-start-eval", action="store_true")
    parser.set_defaults(run=train_voices)
