import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from musetric_toolkit.listening.levels import RATE, letter_words, level_db, read_mix
from musetric_toolkit.listening.round_files import read_jsonl
from musetric_toolkit.listening.systems import VocalsTeacher
from musetric_toolkit.listening.voice_features import dominance

LOG = logging.getLogger(__name__)
PROGRESS_EVERY = 20


@dataclass
class Levels:
    artist: dict[str, str]
    levels: dict[str, list[float]]
    loud_enough: set[str]


def read_levels(args) -> Levels:
    found = Levels(artist={}, levels={}, loud_enough=set())
    for row in read_jsonl(args.levels):
        found.artist[row["id"]] = row["artist"]
        found.levels[row["id"]] = row["vocals_to_mix"]
        if row["mix_rms"] >= args.min_mix_rms:
            found.loud_enough.add(row["id"])
    return found


def quiet_windows(levels: list[float], args) -> list[dict]:
    return [
        dict(
            start_s=index * args.hop_s,
            kind="quiet",
            words=None,
            text="",
            level=levels[index],
        )
        for index in args.fixed_windows
        if index < len(levels) and levels[index] < args.quiet_db
    ]


def detector_windows(rows, found: Levels, rng, args) -> dict[str, list[dict]]:
    windows = defaultdict(list)
    few_words = [
        row
        for row in rows
        if row["kind"] == "vocals" and row["words"] <= args.max_words
    ]
    for row in few_words:
        windows[row["id"]].append(row)
    per_artist = defaultdict(list)
    for name in sorted(windows):
        per_artist[found.artist[name]].append(name)
    keep = set()
    for names in per_artist.values():
        keep.update(rng.permutation(names)[: args.artist_pool].tolist())
    LOG.info(
        "detector windows %d, clips %d, artists %d, kept clips %d",
        len(few_words),
        len(windows),
        len(per_artist),
        len(keep),
    )
    return {name: windows[name] for name in keep}


def add_reverse_windows(picked: dict, rows, found: Levels, args) -> None:
    heard = sorted(
        (
            row
            for row in rows
            if row["kind"] == "instrumental"
            and letter_words(row["text"]) >= args.reverse_min_words
        ),
        key=lambda row: -letter_words(row["text"]),
    )
    counts, taken = defaultdict(int), set()
    for row in heard:
        if row["id"] in taken:
            continue
        artist = found.artist[row["id"]]
        if counts[artist] >= args.reverse_artist_pool:
            continue
        if len(taken) == args.reverse_pool:
            continue
        counts[artist] += 1
        taken.add(row["id"])
        picked.setdefault(row["id"], []).append(row)
    LOG.info("reverse windows %d, kept clips %d", len(heard), len(taken))


def add_signed_windows(picked: dict, found: Levels, args) -> None:
    extra = 0
    for row in read_jsonl(args.signs):
        if row["id"] not in found.levels:
            continue
        student = np.array(row["vocals_to_mix"])
        teacher = student + np.array(row["vocals_gap"])
        hits = np.flatnonzero(
            (student < args.quiet_db) & (teacher >= args.teacher_voice_db)
        )
        for index in hits:
            window = dict(
                start_s=int(index) * args.hop_s,
                kind="quiet",
                words=None,
                text="",
                level=float(student[index]),
            )
            picked.setdefault(row["id"], []).append(window)
            extra += 1
    LOG.info("quiet windows where the mined teacher hears voice: %d", extra)


def add_quiet_tracks(picked: dict, found: Levels, rng, args) -> None:
    others = sorted(
        name
        for name in found.loud_enough - set(picked)
        if quiet_windows(found.levels[name], args)
    )
    for index in rng.permutation(len(others))[: args.quiet_extra]:
        picked[others[index]] = []
    LOG.info(
        "clips with quiet windows %d, added %d, all clips %d",
        len(others),
        min(args.quiet_extra, len(others)),
        len(picked),
    )


def candidates(args) -> tuple[dict[str, list[dict]], dict[str, list[float]]]:
    found = read_levels(args)
    rows = read_jsonl(args.words)
    rng = np.random.default_rng(args.seed)
    picked = detector_windows(rows, found, rng, args)
    add_reverse_windows(picked, rows, found, args)
    if args.signs:
        add_signed_windows(picked, found, args)
    add_quiet_tracks(picked, found, rng, args)
    return picked, found.levels


def window_row(window: dict, vocals: np.ndarray, mix: np.ndarray, args) -> dict:
    start = window["start_s"] * RATE
    stop = (window["start_s"] + args.window_s) * RATE
    part, whole = vocals[:, start:stop], mix[:, start:stop]
    return dict(
        start_s=window["start_s"],
        kind=window["kind"],
        words=window["words"],
        text=window["text"],
        student=window["level"],
        teacher=round(level_db(part, whole), 2),
        dominant=round(dominance(part, whole), 4),
    )


def teacher_command(args) -> None:
    picked, levels = candidates(args)
    done = set()
    if args.out.exists():
        done = {row["id"] for row in read_jsonl(args.out)}
    names = [name for name in sorted(picked) if name not in done]
    if args.limit is not None:
        names = names[: args.limit]
    LOG.info("clips to separate %d", len(names))
    teacher = VocalsTeacher(args.teacher)
    begin = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as out:
        for at, name in enumerate(names):
            mix = read_mix(args.labels, name)
            vocals = teacher.vocals(mix)
            windows = [*picked[name], *quiet_windows(levels[name], args)]
            rows = [window_row(window, vocals, mix, args) for window in windows]
            out.write(json.dumps(dict(id=name, windows=rows), ensure_ascii=False))
            out.write("\n")
            out.flush()
            if at % PROGRESS_EVERY == 0:
                LOG.info("%d, %.0f s", at, time.perf_counter() - begin)
    LOG.info("done, %.0f s", time.perf_counter() - begin)
