import datetime
import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from musetric_toolkit.distill.checkpoints import (
    load_layers,
    save_partial,
    without_rotary,
)
from musetric_toolkit.distill.corpus import dataset_rows, held_tracks, usable
from musetric_toolkit.distill.held_out import (
    evaluate,
    quick_eval,
    quick_set,
    summary,
    write_reference_clips,
)
from musetric_toolkit.distill.inference import student_forward
from musetric_toolkit.distill.losses import (
    Batch,
    LossWeights,
    hidden_step,
    multi_stft_l1,
    pause_excess,
)
from musetric_toolkit.distill.precision import PRECISION
from musetric_toolkit.distill.sampler import Prefetcher, Sampler, SamplerOptions
from musetric_toolkit.distill.student import Fits, build_student
from musetric_toolkit.distill.teacher import teacher_layers
from musetric_toolkit.distill.telemetry import Telemetry, gpu_state, log_gpu

LOG = logging.getLogger(__name__)

BETAS = (0.9, 0.99)
WEIGHT_DECAY = 0.01
CLIP_NORM = 5.0
TAIL = 200
FLOOR = 0.1
SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60
PREFETCH_SEED_STRIDE = 1000


def floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split(","))


def sampler_options(args, marks, genres) -> SamplerOptions:
    return SamplerOptions(
        remix=args.remix,
        silent=args.silent,
        pell=args.pell_share,
        clip=args.clip_seconds,
        voiced_db=None if args.voiced_db == "none" else float(args.voiced_db),
        keep_quiet=args.keep_quiet,
        level=floats(args.level) if args.level else None,
        pause=args.pause_share,
        silence=args.silence_share,
        remix_vocal_db=floats(args.remix_vocal_db),
        remix_acc=args.remix_acc,
        natural=args.natural,
        marks=marks,
        genres=genres,
    )


def loss_weights(args) -> LossWeights:
    return LossWeights(
        hidden=args.hidden_weight,
        mask=args.mask_weight,
        wave_target=args.wave_target,
        pause=args.pause_loss,
        marks=args.marks_weight,
        marks_excess=args.marks_excess,
        truth_excess=args.truth_excess,
        truth_leak=args.truth_leak,
        voice_deficit=args.voice_deficit,
    )


def mean_tail(values: list[float]) -> float | None:
    return float(np.mean(values[-TAIL:])) if values else None


class Clock:
    def __init__(self, elapsed_before: float):
        self.start = time.perf_counter() - elapsed_before
        self.wall_start = time.perf_counter()

    def trained(self) -> float:
        return time.perf_counter() - self.start

    def wall(self) -> float:
        return time.perf_counter() - self.wall_start

    def paused(self, work):
        begin = time.perf_counter()
        result = work()
        self.start += time.perf_counter() - begin
        return result


class Training:
    def __init__(self, args):
        self.args = args
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        if self.world > 1:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            timeout = datetime.timedelta(minutes=60)
            dist.init_process_group("nccl", timeout=timeout)
        self.lead = self.rank == 0
        if args.precision == "fp16":
            PRECISION.dtype = torch.float16
        PRECISION.recompute = not args.no_checkpoint
        self.out, self.labels = args.out, args.labels
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / "telemetry.jsonl" if args.telemetry and self.lead else None
        self.telemetry = Telemetry(path)
        self.telemetry.event("start")
        self.prepare_data()
        self.prepare_model()
        self.prepare_optimizer()
        self.losses, self.hidden_losses = [], []

    def say(self, text: str) -> None:
        if self.lead:
            LOG.info(text)

    def prepare_data(self) -> None:
        args = self.args
        self.rows = dataset_rows(args.dataset)
        labels = json.loads((self.labels / "labels.json").read_text())
        self.index = usable(labels, self.rows)
        self.held = held_tracks(self.index)
        content = {i: row["content"] for i, row in self.rows.items()}
        genres = {i: row.get("genre", "") for i, row in self.rows.items()}
        if args.overfit:
            self.overfit(content)
        marks = json.loads(args.marks.read_text()) if args.marks else None
        self.content = content
        self.options = sampler_options(args, marks, genres)
        self.with_marks = marks is not None
        self.sampler = self.sampler_for(args.seed + self.rank)
        if self.lead:
            write_reference_clips(self.labels, self.out, self.held)
        self.quick = []
        if self.lead and self.world == 1 and args.quick_minutes > 0:
            self.quick = quick_set(self.labels, self.held)
        if self.quick:
            LOG.info(f"quick set: {len(self.quick)} excerpts of 30 s")
        self.telemetry.event("sampler")

    def overfit(self, content) -> None:
        voiced = [
            t
            for t in self.index
            if t["role"] == "train"
            and t["id"].startswith("fma-")
            and content.get(t["id"]) == "vocal"
        ]
        rng = np.random.default_rng(self.args.seed)
        pick = rng.choice(len(voiced), self.args.overfit, replace=False)
        self.index = [voiced[i] for i in sorted(pick)]
        self.held = [dict(t, role="held") for t in self.index]
        minutes = sum(t["seconds"] for t in self.index) / 60
        LOG.info(f"overfit: {len(self.index)} clips, {minutes:.1f} min of audio")

    def sampler_for(self, seed: int) -> Sampler:
        return Sampler(self.labels, self.index, self.content, seed, self.options)

    def prepare_model(self) -> None:
        args = self.args
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        fits = Fits.read(args.fits)
        self.model = build_student(args.student, args.teacher, fits)
        self.telemetry.event("model")
        self.teacher, self.pairing = None, []
        if args.hidden_weight > 0:
            self.teacher = teacher_layers(args.teacher)
            self.telemetry.event("teacher")
            self.pairing = [len(self.teacher) - 1]
            if args.pair_targets:
                self.model.pair_targets = True
                self.pairing = list(range(len(self.teacher)))
            self.say(
                f"hidden-state targets: student pairs -> teacher pairs {self.pairing}, "
                f"weight {args.hidden_weight}"
            )
        if args.init_layers:
            layers, paths = load_layers(self.model, args.init_layers)
            counts = f"{layers} tensors, {paths} path tensors"
            self.say(f"init layers: {args.init_layers} ({counts})")
        if args.init_weights:
            self.load_init_weights(args.init_weights)

    def load_init_weights(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)["model"]
        missing, _ = self.model.load_state_dict(without_rotary(state), strict=False)
        built = sorted({k.split(".")[0] for k in missing if "rotary_embed" not in k})
        self.say(
            f"init weights: {path} ({len(state)} tensors; kept as built: "
            f"{', '.join(built) or 'none'})"
        )

    def prepare_optimizer(self) -> None:
        args, model = self.args, self.model
        learnt = [p for p in model.parameters() if p.requires_grad]
        if args.freeze_transformer:
            model.layers.requires_grad_(False)
        self.trainable = [p for p in model.parameters() if p.requires_grad]
        total = sum(p.numel() for p in model.parameters()) / 1e6
        count = sum(p.numel() for p in self.trainable) / 1e6
        self.say(
            f"trainable: {count:.1f} M of {total:.1f} M parameters; "
            f"{self.world} GPU, {args.precision}"
        )
        self.opt = torch.optim.AdamW(
            self.trainable, lr=args.lr, betas=BETAS, weight_decay=WEIGHT_DECAY
        )
        self.ema = None
        if args.ema > 0:
            self.ema = [p.detach().float().clone() for p in self.trainable]
        if args.init:
            self.load_init(args.init, learnt)
        if args.lr_new != 1.0:
            self.group_new_parts()
        self.progress = 0.0
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self.lr_at)
        self.scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
        self.step, self.elapsed_before = 0, 0.0
        self.checkpoint = self.out / "student.pt"
        if args.resume and self.checkpoint.exists():
            self.resume()

    def load_init(self, path: Path, learnt) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(without_rotary(state["model"]), strict=False)
        if self.args.freeze_transformer:
            saved = torch.optim.AdamW(learnt)
            saved.load_state_dict(state["opt"])
            for p in self.trainable:
                self.opt.state[p] = saved.state[p]
            del saved
        else:
            self.opt.load_state_dict(state["opt"])
        for group in self.opt.param_groups:
            group["lr"] = group["initial_lr"] = self.args.lr
        minutes = state["elapsed"] / 60
        self.say(f"init: {path} (step {state['step']}, {minutes:.0f} min)")
        del state
        self.telemetry.event("init")

    def group_new_parts(self) -> None:
        args = self.args
        names = {id(p): n for n, p in self.model.named_parameters()}
        fresh = re.compile(args.lr_new_parts)
        new = [p for p in self.trainable if fresh.search(names[id(p)])]
        old = [p for p in self.trainable if not fresh.search(names[id(p)])]
        groups = [
            dict(params=old, lr=args.lr),
            dict(params=new, lr=args.lr * args.lr_new),
        ]
        grouped = torch.optim.AdamW(groups, betas=BETAS, weight_decay=WEIGHT_DECAY)
        for p in self.trainable:
            if p in self.opt.state:
                grouped.state[p] = self.opt.state[p]
        for group in grouped.param_groups:
            group["initial_lr"] = group["lr"]
        self.opt = grouped
        new_m = sum(p.numel() for p in new) / 1e6
        old_m = sum(p.numel() for p in old) / 1e6
        self.say(
            f"lr groups: {new_m:.1f} M parameters at {args.lr_new}x "
            f"({args.lr * args.lr_new:g}), {old_m:.1f} M at {args.lr:g}"
        )

    def lr_at(self, step: int) -> float:
        args = self.args
        if step < args.warmup:
            return (step + 1) / args.warmup
        if args.schedule == "wsd":
            start = 1 - args.decay_share
            decayed = max(0.0, self.progress - start) / args.decay_share
            return 1.0 - (1 - FLOOR) * min(1.0, decayed)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, self.progress)))
        return FLOOR + (1 - FLOOR) * cosine

    def resume(self) -> None:
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.sched.load_state_dict(state["sched"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        self.step, self.elapsed_before = state["step"], state["elapsed"]
        del state

    def shared(self, *values: float) -> list[float]:
        if self.world == 1:
            return list(values)
        t = torch.tensor(values, dtype=torch.float64, device="cuda")
        dist.broadcast(t, 0)
        return t.tolist()

    def meta(self, elapsed: float) -> dict:
        return dict(step=self.step, elapsed=elapsed, student=self.args.student)

    def scale(self):
        return self.scaler.get_scale() if self.scaler.is_enabled() else None

    def save(self, elapsed: float) -> None:
        if not self.lead:
            return
        state = dict(
            model=self.model.state_dict(),
            opt=self.opt.state_dict(),
            sched=self.sched.state_dict(),
            scaler=self.scaler.state_dict(),
            step=self.step,
            elapsed=elapsed,
            student=self.args.student,
        )
        torch.save(state, self.checkpoint)
        if self.ema is not None:
            named = [n for n, p in self.model.named_parameters() if p.requires_grad]
            weights = self.model.state_dict()
            for n, e in zip(named, self.ema, strict=True):
                weights[n] = e.to(weights[n].dtype)
            shadow = dict(
                model=weights,
                step=self.step,
                elapsed=elapsed,
                student=self.args.student,
            )
            torch.save(shadow, self.out / "student-ema.pt")

    def evaluate_all(self, elapsed: float) -> None:
        args = self.args
        if args.no_eval:
            self.save(elapsed)
            return
        if self.world > 1:
            for p in self.trainable:
                dist.broadcast(p.data, 0)
        minutes = elapsed / SECONDS_PER_MINUTE
        clips = self.out / "clips" / f"{int(minutes):04d}min"
        clips.mkdir(parents=True, exist_ok=True)
        part = self.held[self.rank :: self.world]
        evals = evaluate(self.model, self.labels, part, clips)
        if self.world > 1:
            parts = [None] * self.world
            dist.all_gather_object(parts, evals)
            evals = {k: v for chunk in parts for k, v in chunk.items()}
        if not self.lead:
            return
        examples = self.step * self.world * args.batch * args.accum
        row = dict(
            step=self.step,
            examples=examples,
            minutes=minutes,
            loss=mean_tail(self.losses),
            hidden=mean_tail(self.hidden_losses),
            lr=self.sched.get_last_lr()[0],
            **summary(evals, self.rows),
            gpu=gpu_state(),
            tracks=evals,
        )
        self.log.write(json.dumps(row) + "\n")
        self.log.flush()
        shown = {k: v for k, v in row.items() if k not in ("tracks", "by_group")}
        LOG.info("eval %s", json.dumps(shown))
        self.save(elapsed)
        if args.snapshots:
            path = self.out / f"snapshot-{int(minutes):04d}min.pt"
            save_partial(self.model, path, self.meta(elapsed), half=False)

    def write_quick(self, row: dict) -> None:
        self.quick_log.write(json.dumps(row) + "\n")
        self.quick_log.flush()
        LOG.info("quick %s", json.dumps(row))

    def next_at(self, minutes: float) -> float:
        period = minutes * SECONDS_PER_MINUTE
        return (self.elapsed_before // period + 1) * period

    def open_logs(self) -> None:
        self.stop = threading.Event()
        self.log = self.quick_log = None
        if self.lead:
            gpu_log = threading.Thread(
                target=log_gpu, args=(self.out / "gpu.log", self.stop), daemon=True
            )
            gpu_log.start()
            self.log = open(self.out / "train.jsonl", "a")  # noqa: SIM115
            self.quick_log = open(self.out / "quick.jsonl", "a")  # noqa: SIM115

    def first_evaluation(self) -> None:
        args = self.args
        if self.step == 0 and not args.no_eval and not args.no_start_eval:
            self.clock.paused(lambda: self.evaluate_all(0.0))
        elif self.step == 0 and self.quick:
            scores = self.clock.paused(lambda: quick_eval(self.model, self.quick))
            self.write_quick(dict(step=0, minutes=0.0, **scores))

    def prefetcher(self) -> Prefetcher:
        args = self.args
        seeds = [
            args.seed + self.rank + PREFETCH_SEED_STRIDE * k
            for k in range(1, args.prefetch_workers)
        ]
        samplers = [self.sampler] + [self.sampler_for(seed) for seed in seeds]
        return Prefetcher(
            samplers,
            args.batch,
            depth=args.prefetch,
            with_marks=self.with_marks,
            gate_db=args.gate_quiet,
        )

    def micro_step(self) -> torch.Tensor:
        args = self.args
        x, y, truth, *rest = self.feed.batch()
        gate = rest.pop() if args.gate_quiet > 0 else None
        marked = rest[0] if rest else None
        if self.teacher is None:
            vocals = student_forward(self.model, x)
            loss = multi_stft_l1(vocals, y)
            if args.pause_loss > 0:
                loss = loss + args.pause_loss * pause_excess(x, y, vocals)
            return loss
        batch = Batch(x=x, y=y, truth=truth, marked=marked, gate=gate)
        loss, hidden = hidden_step(
            self.model, self.teacher, self.pairing, batch, self.weights
        )
        self.hidden_losses.append(hidden)
        return loss

    def average_gradients(self) -> None:
        flat = torch.cat([p.grad.reshape(-1) for p in self.trainable])
        dist.all_reduce(flat)
        flat /= self.world
        offset = 0
        for p in self.trainable:
            n = p.grad.numel()
            p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
            offset += n

    def update(self) -> torch.Tensor:
        args = self.args
        self.opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            loss = self.micro_step()
            self.scaler.scale(loss / args.accum).backward()
        if self.world > 1:
            self.average_gradients()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.trainable, CLIP_NORM)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        if self.ema is not None:
            with torch.no_grad():
                for e, p in zip(self.ema, self.trainable, strict=True):
                    if self.step < args.warmup:
                        e.copy_(p.detach().float())
                    else:
                        e.lerp_(p.detach().float(), 1 - args.ema)
        return loss

    def snapshot(self, elapsed: float) -> None:
        path = self.out / f"snapshot-step{self.step:05d}.pt"
        save_partial(self.model, path, self.meta(elapsed), half=True)
        self.telemetry.event("snapshot", step=self.step)

    def progress_row(self, elapsed: float) -> dict:
        args = self.args
        steps = max(1, self.step - self.first_step)
        return dict(
            minutes=round(elapsed / SECONDS_PER_MINUTE, 1),
            step=self.step,
            examples=self.step * self.world * args.batch * args.accum,
            seconds_per_step=round((elapsed - self.first_time) / steps, 3),
            loss=mean_tail(self.losses),
            hidden=mean_tail(self.hidden_losses),
            lr=self.sched.get_last_lr()[0],
            scale=self.scale(),
            gpu=gpu_state(),
        )

    def quick_check(self, elapsed: float) -> bool:
        args = self.args
        scores = self.clock.paused(lambda: quick_eval(self.model, self.quick))
        row = dict(
            step=self.step,
            minutes=round(elapsed / SECONDS_PER_MINUTE, 1),
            loss=mean_tail(self.losses),
            hidden=mean_tail(self.hidden_losses),
            **scores,
            gpu=gpu_state(),
        )
        self.write_quick(row)
        guarded = elapsed >= args.guard_minutes * SECONDS_PER_MINUTE
        if guarded and row["quick_sdr_median"] < args.guard_sdr:
            LOG.info(
                f"stopped: quick SDR median {row['quick_sdr_median']:.2f} dB < "
                f"{args.guard_sdr} dB after {elapsed / 60:.0f} min, "
                "the student is not learning"
            )
            self.clock.paused(lambda: self.evaluate_all(elapsed))
            return True
        return False

    def after_step(self, elapsed: float) -> bool:
        args = self.args
        if args.snapshot_steps and self.lead and self.step % args.snapshot_steps == 0:
            self.clock.paused(lambda: self.snapshot(elapsed))
        if elapsed >= self.next_save:
            self.next_save += args.save_minutes * SECONDS_PER_MINUTE
            self.clock.paused(lambda: self.save(elapsed))
            if self.lead:
                LOG.info("progress %s", json.dumps(self.progress_row(elapsed)))
        if elapsed >= self.next_quick:
            self.next_quick += args.quick_minutes * SECONDS_PER_MINUTE
            if self.quick_check(elapsed):
                self.evaluated_at = self.step
                return True
        if elapsed >= self.next_eval and not args.no_eval:
            self.next_eval += args.eval_minutes * SECONDS_PER_MINUTE
            self.clock.paused(lambda: self.evaluate_all(elapsed))
            self.evaluated_at = self.step
        return False

    def run(self) -> None:
        args = self.args
        self.weights = loss_weights(args)
        self.open_logs()
        self.clock = Clock(self.elapsed_before)
        self.first_evaluation()
        self.evaluated_at = self.step
        self.next_eval = self.next_at(args.eval_minutes)
        self.next_quick = self.next_at(args.quick_minutes) if self.quick else math.inf
        self.next_save = self.next_at(args.save_minutes)
        self.first_step, self.first_time = self.step, self.clock.trained()
        self.feed = self.prefetcher()
        self.telemetry.event("loop", step=self.step)
        self.telemetry.window = (time.perf_counter(), 0.0, self.step)
        budget = args.hours * SECONDS_PER_HOUR
        while True:
            elapsed, wall = self.shared(self.clock.trained(), self.clock.wall())
            if elapsed >= budget or wall >= args.wall_hours * SECONDS_PER_HOUR:
                break
            self.progress = elapsed / budget
            loss = self.update()
            self.step += 1
            self.losses.append(loss.item())
            if self.step == self.first_step + 1:
                self.telemetry.event("first step", step=self.step)
            lr = self.sched.get_last_lr()[0]
            self.telemetry.step(self.step, self.feed, self.losses, lr, self.scale())
            if self.after_step(elapsed):
                break
        if self.step > self.evaluated_at:
            self.evaluate_all(self.shared(self.clock.trained())[0])
        self.stop.set()
        self.telemetry.event("end", step=self.step)
        if self.world > 1:
            dist.destroy_process_group()


def train(args) -> None:
    Training(args).run()
