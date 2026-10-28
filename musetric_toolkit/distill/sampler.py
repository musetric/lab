import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.distill.corpus import (
    FULL_SCALE,
    INT16_MAX,
    FlacTrack,
    label_files,
)
from musetric_toolkit.distill.losses import quiet_gate
from musetric_toolkit.distill.teacher import CHUNK, HOP, SAMPLE_RATE

LOG = logging.getLogger(__name__)

VOICE_CONTENT = {"vocal", "vocal-only", "speech", "stems"}
NEAR_SILENT = 1e-5
CROP_TRIES = 20
SCALE = FULL_SCALE / INT16_MAX
NATURAL_KINDS = ["fade", "onset", "novoice", "soft"]
MARK_SECONDS = 2


@dataclass
class SamplerOptions:
    remix: float = 0.5
    silent: float = 0.1
    pell: float = 0.0
    clip: float = 0.0
    voiced_db: float | None = -40.0
    keep_quiet: bool = False
    level: tuple | None = None
    pause: float = 0.0
    silence: float = 0.0
    remix_vocal_db: tuple = (-6.0, 3.0)
    remix_acc: str = "any"
    natural: str | None = None
    marks: dict | None = None
    genres: dict[str, str] = field(default_factory=dict)


def as_float(part) -> torch.Tensor:
    return torch.from_numpy(part.astype(np.float32)) * SCALE


class Sampler:
    def __init__(
        self, labels: Path, index, content, seed: int, options: SamplerOptions
    ):
        self.rng = np.random.default_rng(seed)
        self.options = options
        self.remix, self.silent, self.pell = options.remix, options.silent, options.pell
        voiced_db = options.voiced_db
        self.voiced_rms = None if voiced_db is None else 10 ** (voiced_db / 20)
        self.natural_share, self.natural_events = 0.0, {}
        train = [
            t
            for t in index
            if t["role"] == "train" and t["seconds"] * SAMPLE_RATE > CHUNK + SAMPLE_RATE
        ]
        self.load_tracks(labels, train)
        self.kinds = [
            "pell" if t.get("pell") else content.get(t["id"], "vocal") for t in train
        ]
        self.set_spans(options.clip)
        self.set_pools(train)
        self.marks = [
            np.array(
                (options.marks or {}).get(t["id"], {}).get("suspect", []),
                dtype=np.int64,
            )
            for t in train
        ]
        self.last_crop = (0, 0)
        if options.marks:
            marked = sum(len(m) > 0 for m in self.marks)
            windows = sum(len(m) for m in self.marks)
            LOG.info(f"marks: {windows} suspect windows in {marked} tracks")
        if options.natural:
            self.set_natural(options.natural, train)

    def load_tracks(self, labels: Path, train) -> None:
        stored = (labels / f"{train[0]['id']}.mix.npy").exists()
        flac = {} if stored else label_files(labels)

        def track(t: dict, part: str):
            if flac:
                return FlacTrack(flac[f"{t['id']}.{part}"], t["frames"])
            return np.load(labels / f"{t['id']}.{part}.npy", mmap_mode="r")

        self.mix = [track(t, "mix") for t in train]
        self.voc = [
            m if t.get("pell") else track(t, "vocals")
            for t, m in zip(train, self.mix, strict=True)
        ]

    def set_spans(self, clip: float) -> None:
        span_rng = np.random.default_rng(1234)
        self.spans = []
        for m in self.mix:
            n = m.shape[1]
            length = n if clip <= 0 else max(min(int(clip * SAMPLE_RATE), n), CHUNK + 1)
            start = int(span_rng.integers(0, n - length + 1))
            self.spans.append((start, start + length))
        self.lengths = np.array([hi - lo for lo, hi in self.spans], dtype=np.float64)

    def pool(self, mask: list[bool]) -> tuple[np.ndarray, np.ndarray]:
        ids = np.flatnonzero(mask)
        return ids, self.lengths[ids] / self.lengths[ids].sum()

    def set_pools(self, train) -> None:
        kinds = self.kinds
        self.voiced = self.pool([k in VOICE_CONTENT for k in kinds])
        self.instrumental = self.pool([k == "instrumental" for k in kinds])
        self.any = self.pool([k != "pell" for k in kinds])
        self.pells = self.pool([k == "pell" for k in kinds])
        remix_acc = self.options.remix_acc
        if remix_acc.startswith("instrumental:"):
            wanted = set(remix_acc.split(":", 1)[1].split("+"))
            genres = self.options.genres
            self.accompaniment = self.pool(
                [
                    k == "instrumental" and genres.get(t["id"]) in wanted
                    for k, t in zip(kinds, train, strict=True)
                ]
            )
            count = len(self.accompaniment[0])
            LOG.info(f"remix accompaniment: {count} instrumentals of {sorted(wanted)}")
        elif remix_acc == "instrumental":
            self.accompaniment = self.instrumental
        else:
            self.accompaniment = self.any
        stems = sum(k == "stems" for k in kinds)
        LOG.info(
            f"sampler: {len(self.voiced[0])} voiced, "
            f"{len(self.instrumental[0])} instrumental, "
            f"{len(self.pells[0])} a cappellas, {stems} true-stem songs, "
            f"{len(train)} in all"
        )

    def set_natural(self, natural: str, train) -> None:
        path, share, *wanted = natural.split(",")
        self.natural_share = float(share)
        kinds = wanted[0].split("+") if wanted else NATURAL_KINDS
        found = json.loads(Path(path).read_text(encoding="utf-8"))
        ids = {t["id"]: i for i, t in enumerate(train)}
        for kind in kinds:
            events = [
                (ids[tid], pos)
                for tid, places in found.items()
                if tid in ids
                for pos in places.get(kind, [])
            ]
            if events:
                self.natural_events[kind] = events
        summary = ", ".join(f"{k} {len(e)}" for k, e in self.natural_events.items())
        LOG.info(f"natural places: {summary}")

    def draw(self, pool: tuple[np.ndarray, np.ndarray]) -> int:
        return int(self.rng.choice(pool[0], p=pool[1]))

    def crop(self, i: int, voiced: bool) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(CROP_TRIES):
            lo, hi = self.spans[i]
            o = int(self.rng.integers(lo, hi - CHUNK))
            m = as_float(self.mix[i][:, o : o + CHUNK])
            v = as_float(self.voc[i][:, o : o + CHUNK])
            if m.pow(2).mean() <= NEAR_SILENT and not self.options.keep_quiet:
                continue
            loud = self.voiced_rms is None or v.pow(2).mean().sqrt() >= self.voiced_rms
            if not voiced or loud:
                break
        self.last_crop = (i, o)
        return m, v

    def mark_mask(self, i: int, o: int) -> torch.Tensor:
        mask = torch.zeros(CHUNK)
        for start in self.marks[i]:
            a = int(start) * SAMPLE_RATE - o
            b = (int(start) + MARK_SECONDS) * SAMPLE_RATE - o
            if b > 0 and a < CHUNK:
                mask[max(a, 0) : min(b, CHUNK)] = 1
        return mask

    def spans_gain(self, n: int, count: int, lo: float, hi: float) -> torch.Tensor:
        gain = np.ones(n, dtype=np.float32)
        ramp = int(0.05 * SAMPLE_RATE)
        for _ in range(count):
            length = int(self.rng.uniform(lo, hi) * SAMPLE_RATE)
            start = int(self.rng.integers(0, max(1, n - length)))
            gain[start : start + length] = 0.0
        kernel = np.ones(ramp, dtype=np.float32) / ramp
        gain = np.convolve(gain, kernel, "same").astype(np.float32)
        return torch.from_numpy(gain)[None]

    def natural_example(self):
        kinds = list(self.natural_events)
        events = self.natural_events[kinds[int(self.rng.integers(len(kinds)))]]
        i, pos = events[int(self.rng.integers(len(events)))]
        n = self.mix[i].shape[1]
        offset = pos - int(self.rng.uniform(SAMPLE_RATE, CHUNK - SAMPLE_RATE))
        o = int(np.clip(offset, 0, n - CHUNK))
        x = as_float(self.mix[i][:, o : o + CHUNK])
        y = as_float(self.voc[i][:, o : o + CHUNK])
        if self.rng.random() < 0.5:  # noqa: PLR2004
            x, y = x.flip(0), y.flip(0)
        peak = x.abs().max().clamp(min=1e-4)
        g = float(self.rng.uniform(0.3, 1.0)) / peak
        return x * g, y * g, False, self.mark_mask(i, o)

    def remix_example(self):
        use_pells = self.pell > 0 and self.rng.random() < self.pell
        source = self.pells if use_pells else self.voiced
        true_stem = source is self.pells
        mark = torch.zeros(CHUNK)
        _, voc_a = self.crop(self.draw(source), voiced=True)
        true_stem = true_stem or self.kinds[self.last_crop[0]] == "stems"
        if not true_stem:
            mark = self.mark_mask(*self.last_crop)
        mix_b, voc_b = self.crop(self.draw(self.accompaniment), voiced=False)
        gv = 10 ** (self.rng.uniform(*self.options.remix_vocal_db) / 20)
        gr = 10 ** (self.rng.uniform(-6, 3) / 20)
        if self.options.pause > 0 and self.rng.random() < self.options.pause:
            count = int(self.rng.integers(1, 4))
            voc_a = voc_a * self.spans_gain(voc_a.shape[1], count, 1.0, 4.0)
        return voc_a * gv + (mix_b - voc_b) * gr, voc_a * gv, true_stem, mark

    def drawn_example(self):
        kind = self.rng.random()
        if kind < self.silent:
            x, y = self.crop(self.draw(self.instrumental), voiced=False)
            return x, y, False, torch.zeros(CHUNK)
        if kind < self.silent + self.remix:
            return self.remix_example()
        x, y = self.crop(self.draw(self.voiced), voiced=True)
        mark = self.mark_mask(*self.last_crop)
        return x, y, self.kinds[self.last_crop[0]] == "stems", mark

    def finish(self, x, y):
        if self.rng.random() < 0.5:  # noqa: PLR2004
            x, y = x.flip(0), y.flip(0)
        if self.options.silence > 0 and self.rng.random() < self.options.silence:
            cut = self.spans_gain(x.shape[1], int(self.rng.integers(1, 3)), 0.5, 3.0)
            x, y = x * cut, y * cut
        peak = x.abs().max().clamp(min=1e-4)
        level = self.options.level
        if level is not None and self.rng.random() < level[0]:
            g = 10 ** (float(self.rng.uniform(level[1], level[2])) / 20) / peak
        else:
            g = float(self.rng.uniform(0.3, 1.0)) / peak
        return x * g, y * g

    def example(self):
        natural = self.natural_share > 0 and self.rng.random() < self.natural_share
        if natural:
            return self.natural_example()
        x, y, true_stem, mark = self.drawn_example()
        x, y = self.finish(x, y)
        return x, y, true_stem, mark

    def cpu_batch(
        self, size: int, with_marks: bool = False
    ) -> tuple[torch.Tensor, ...]:
        examples = [self.example() for _ in range(size)]
        xs, ys, truth, marks = zip(*examples, strict=True)
        out = (
            torch.stack(xs).pin_memory(),
            torch.stack(ys).pin_memory(),
            torch.tensor(truth),
        )
        return (*out, torch.stack(marks).pin_memory()) if with_marks else out

    def batch(self, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, truth = self.cpu_batch(size)
        return (
            x.cuda(non_blocking=True),
            y.cuda(non_blocking=True),
            truth.cuda(non_blocking=True),
        )


class Prefetcher:
    def __init__(
        self,
        samplers,
        size: int,
        depth: int = 4,
        with_marks: bool = False,
        gate_db: float = 0.0,
    ):
        self.queue = queue.Queue(maxsize=depth)
        self.stopped = threading.Event()
        self.waited = 0.0
        self.size, self.with_marks, self.gate_db = size, with_marks, gate_db
        for sampler in samplers:
            threading.Thread(target=self.fill, args=(sampler,), daemon=True).start()

    def gated(self, batch):
        if self.gate_db <= 0:
            return batch
        x, y, truth = batch[:3]
        gate = quiet_gate(x, y, truth, self.gate_db)
        return (x, y * gate[:, None], truth, *batch[3:], gate[:, ::HOP].contiguous())

    def fill(self, sampler) -> None:
        while not self.stopped.is_set():
            batch = self.gated(sampler.cpu_batch(self.size, self.with_marks))
            while not self.stopped.is_set():
                try:
                    self.queue.put(batch, timeout=0.5)
                    break
                except queue.Full:
                    pass

    def close(self) -> None:
        self.stopped.set()

    def batch(self) -> tuple[torch.Tensor, ...]:
        begin = time.perf_counter()
        items = self.queue.get()
        self.waited += time.perf_counter() - begin
        return tuple(t.cuda(non_blocking=True) for t in items)
