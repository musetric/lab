import json
import subprocess
import threading
import time
from pathlib import Path

import torch

PROCESS_START = time.perf_counter()
GPU_QUERY = (
    "clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active,memory.used"
)
GPU_LOG_SECONDS = 30


def gpu_state() -> str:
    command = ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader"]
    out = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip()


def log_gpu(path: Path, stop: threading.Event) -> None:
    with open(path, "a") as f:
        while not stop.wait(GPU_LOG_SECONDS):
            f.write(f"{time.strftime('%H:%M:%S')}, {gpu_state()}\n")
            f.flush()


def gib(value: int) -> float:
    return round(value / 2**30, 2)


class Telemetry:
    def __init__(self, path: Path | None, every: int = 25):
        self.file = open(path, "a") if path else None  # noqa: SIM115
        self.every = every
        self.window = (time.perf_counter(), 0.0, 0)

    def event(self, name: str, **extra) -> None:
        if self.file:
            seconds = round(time.perf_counter() - PROCESS_START, 3)
            row = dict(event=name, t=seconds, wall=round(time.time(), 3), **extra)
            self.file.write(json.dumps(row) + "\n")
            self.file.flush()

    def step(self, step: int, feed, losses: list[float], lr: float, scale) -> None:
        if not self.file or step % self.every:
            return
        now = time.perf_counter()
        begin, waited, first = self.window
        self.window = (now, feed.waited, step)
        n = step - first
        self.event(
            "window",
            step=step,
            steps=n,
            seconds=round(now - begin, 3),
            step_seconds=round((now - begin) / n, 4),
            data_wait=round(feed.waited - waited, 3),
            queue=feed.queue.qsize(),
            loss=round(sum(losses[-n:]) / n, 5),
            lr=lr,
            scale=scale,
            mem_alloc_gb=gib(torch.cuda.max_memory_allocated()),
            mem_reserved_gb=gib(torch.cuda.max_memory_reserved()),
        )
