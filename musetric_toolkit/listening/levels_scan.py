import json
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from musetric_toolkit.listening.levels import RATE, corpus_part
from musetric_toolkit.listening.round_files import (
    artists_of,
    corpus_tracks,
    earlier_sources,
    read_json,
    read_jsonl,
)

LOG = logging.getLogger(__name__)
LEVEL_FLOOR = 1e-3
PROGRESS_EVERY = 500
CHUNK = 8
STEREO = 2


def roles_of(voices: list[Path]) -> tuple[set[str], set[str]]:
    train, held = set(), set()
    for manifest in voices:
        for entry in read_json(manifest):
            (train if entry["role"] == "train" else held).add(entry["id"])
    return train, held


def universe(args) -> list[dict]:
    tracks = corpus_tracks(args.corpus)
    train, held = roles_of(args.voices)
    used = earlier_sources(args.earlier_round)
    used_artists = artists_of(used, tracks)
    rows = []
    for name in sorted(train - held - used):
        item = tracks.get(name)
        if item is None or item["genre"] in args.skip_genres:
            continue
        if item["artist"] not in used_artists:
            rows.append(dict(id=name, artist=item["artist"], genre=item["genre"]))
    LOG.info(
        "train %d, held %d, earlier sources %d, earlier artists %d, universe %d",
        len(train),
        len(held),
        len(used),
        len(used_artists),
        len(rows),
    )
    return rows


def window_powers(audio, step: int, size: int, stride: int) -> np.ndarray:
    power = (np.asarray(audio[:, ::stride], dtype=np.float64) ** 2).sum(0)
    total = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, max(1, len(power) - size + 1), step)
    return total[starts + size] - total[starts]


def track_levels(
    labels: Path, step: int, size: int, stride: int, row: dict
) -> dict | None:
    try:
        mix = corpus_part(labels, row["id"], "mix")
        vocals = corpus_part(labels, row["id"], "vocals")
    except FileNotFoundError:
        return None
    a = window_powers(vocals, step, size, stride)
    b = window_powers(mix, step, size, stride)
    ratio = 10 * np.log10((a + LEVEL_FLOOR) / (b + LEVEL_FLOOR))
    return dict(
        **row,
        vocals_to_mix=[round(float(v), 2) for v in ratio],
        mix_rms=round(float(np.sqrt(b.mean() / (size * STEREO))), 1),
    )


def scan_command(args) -> None:
    rows = universe(args)
    done = set()
    if args.out.exists():
        done = {row["id"] for row in read_jsonl(args.out)}
    todo = [row for row in rows if row["id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    step = RATE * args.hop_s // args.stride
    size = RATE * args.window_s // args.stride
    worker = partial(track_levels, args.labels, step, size, args.stride)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.out.open("a", encoding="utf-8") as out,
        ProcessPoolExecutor(args.workers) as pool,
    ):
        for index, found in enumerate(pool.map(worker, todo, chunksize=CHUNK)):
            if found:
                out.write(json.dumps(found, ensure_ascii=False) + "\n")
            if index % PROGRESS_EVERY == 0:
                LOG.info("%d of %d", index, len(todo))
    LOG.info("done")
