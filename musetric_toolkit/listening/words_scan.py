import json
import logging
import time

import numpy as np

from musetric_toolkit.listening.levels import RATE, SCALE, corpus_part
from musetric_toolkit.listening.listener import Listener
from musetric_toolkit.listening.round_files import read_jsonl

LOG = logging.getLogger(__name__)
VOCALS = "vocals"
INSTRUMENTAL = "instrumental"
PROGRESS_BATCHES = 20


def kind_of_level(level: float, args) -> str | None:
    if level >= args.gate_db:
        return VOCALS
    return INSTRUMENTAL if level < args.quiet_db else None


def windows(args) -> list[dict]:
    jobs = []
    for row in read_jsonl(args.levels):
        levels = row["vocals_to_mix"]
        if row["mix_rms"] < args.min_mix_rms:
            continue
        for index in args.fixed_windows:
            if index >= len(levels):
                continue
            level = levels[index]
            kind = kind_of_level(level, args)
            if kind:
                start_s = index * args.hop_s
                jobs.append(dict(id=row["id"], start_s=start_s, kind=kind, level=level))
    return jobs


def read_window(job: dict, args) -> np.ndarray:
    start, stop = job["start_s"] * RATE, (job["start_s"] + args.window_s) * RATE
    mix = corpus_part(args.labels, job["id"], "mix")[:, start:stop]
    vocals = corpus_part(args.labels, job["id"], VOCALS)[:, start:stop]
    if job["kind"] == VOCALS:
        return np.asarray(vocals, dtype=np.float32) * SCALE
    return (
        np.asarray(mix, dtype=np.float32) - np.asarray(vocals, dtype=np.float32)
    ) * SCALE


def jobs_to_do(args) -> list[dict]:
    jobs = windows(args)
    LOG.info(
        "windows %d, %s",
        len(jobs),
        {k: sum(j["kind"] == k for j in jobs) for k in (VOCALS, INSTRUMENTAL)},
    )
    done = set()
    if args.out.exists():
        done = {(row["id"], row["start_s"]) for row in read_jsonl(args.out)}
    rng = np.random.default_rng(args.seed)
    jobs = [jobs[i] for i in rng.permutation(len(jobs))]
    jobs = [
        job
        for job in jobs
        if (job["id"], job["start_s"]) not in done
        and (args.kind is None or job["kind"] == args.kind)
    ]
    jobs.sort(key=lambda job: job["kind"] != VOCALS)
    return jobs if args.limit is None else jobs[: args.limit]


def words_command(args) -> None:
    jobs = jobs_to_do(args)
    listener = Listener(args.batch)
    LOG.info("whisper ready, to do %d", len(jobs))
    begin = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as out:
        for at in range(0, len(jobs), args.batch):
            part = jobs[at : at + args.batch]
            heard = listener([read_window(job, args) for job in part])
            for job, words in zip(part, heard, strict=True):
                out.write(json.dumps({**job, **words}, ensure_ascii=False) + "\n")
            out.flush()
            if at % (args.batch * PROGRESS_BATCHES) == 0:
                LOG.info("%d, %.0f s", at, time.perf_counter() - begin)
    LOG.info("done, %.0f s", time.perf_counter() - begin)
