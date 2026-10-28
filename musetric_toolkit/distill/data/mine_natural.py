import json
import logging
from pathlib import Path

import numpy as np

from musetric_toolkit.distill.corpus import INT16_MAX, dataset_rows

LOG = logging.getLogger(__name__)

FRAME = 4410
VOICE = {"vocal", "vocal-only", "weak-vocal", "speech"}
KINDS = ("fade", "onset", "novoice", "soft")
VOICED_BELOW_PEAK_DB = 60
MIN_VOICED_FRAMES = 20
LOUD_PERCENTILE = 95
ACTIVE_DB = 20
ABSENT_DB = 45
PLAYING_DB = 30
SOFT_VOICE_DB = 35
SOFT_UNDER_MIX_DB = (15, 30)
GAP = 20
AHEAD = 30
HOLD = 10
ONSET_FROM = 11
NOVOICE_FRAMES = 20
SOFT_FRAMES = 10


def frames_db(x: np.ndarray) -> np.ndarray:
    n = x.shape[1] // FRAME
    power = (np.asarray(x[:, : n * FRAME], dtype=np.float32) / INT16_MAX) ** 2
    return 10 * np.log10(power.mean(0).reshape(n, FRAME).mean(1) + 1e-12)


def runs(mask: np.ndarray, length: int) -> list[tuple[int, int]]:
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= length:
                out.append((start, i))
            start = None
    return out


def fades(active: np.ndarray, absent: np.ndarray) -> list[int]:
    found, last = [], -GAP
    for t in np.flatnonzero(active):
        if t - last < GAP:
            continue
        ahead = absent[t + 1 : t + AHEAD + 1]
        if not ahead.any():
            continue
        u = t + 1 + int(np.argmax(ahead))
        held = u + HOLD <= len(absent) and absent[u : u + HOLD].all()
        if held and not active[t + 1 : u].any():
            found.append(int(t * FRAME))
            last = t
    return found


def onsets(active: np.ndarray, absent: np.ndarray) -> list[int]:
    found, last = [], -GAP
    for t in np.flatnonzero(active):
        if t - last < GAP or t < ONSET_FROM:
            continue
        behind = absent[max(0, t - AHEAD - 1) : t]
        if not behind.any():
            continue
        w = max(0, t - AHEAD - 1) + int(np.flatnonzero(behind)[-1])
        held = w >= HOLD - 1 and absent[w - HOLD + 1 : w + 1].all()
        if held and not active[w + 1 : t].any():
            found.append(int((w + 1) * FRAME))
            last = t
    return found


def places(v: np.ndarray, m: np.ndarray) -> dict[str, list[int]] | None:
    voiced = v > v.max() - VOICED_BELOW_PEAK_DB
    if voiced.sum() < MIN_VOICED_FRAMES:
        return None
    loud = np.percentile(v[voiced], LOUD_PERCENTILE)
    active = v > loud - ACTIVE_DB
    absent = v < loud - ABSENT_DB
    playing = m > np.percentile(m, LOUD_PERCENTILE) - PLAYING_DB
    low, high = SOFT_UNDER_MIX_DB
    soft_mask = (v > loud - SOFT_VOICE_DB) & (v < m - low) & (v > m - high)
    novoice = runs(absent & playing, NOVOICE_FRAMES)
    soft = runs(soft_mask, SOFT_FRAMES)
    found = dict(
        fade=fades(active, absent),
        onset=onsets(active, absent),
        novoice=[int((a + b) // 2 * FRAME) for a, b in novoice],
        soft=[int((a + b) // 2 * FRAME) for a, b in soft],
    )
    return {kind: events for kind, events in found.items() if events}


def mine(args) -> None:
    rows = [
        r
        for r in dataset_rows(args.dataset).values()
        if r["decision"] == "use" and r["role"] == "train" and r["content"] in VOICE
    ]
    found, totals = {}, dict.fromkeys(KINDS, 0)
    for r in rows:
        v = frames_db(np.load(args.labels / f"{r['id']}.vocals.npy", mmap_mode="r"))
        m = frames_db(np.load(args.labels / f"{r['id']}.mix.npy", mmap_mode="r"))
        kinds = places(v, m)
        if kinds:
            found[r["id"]] = kinds
            for kind, events in kinds.items():
                totals[kind] += len(events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(found), encoding="utf-8")
    counts = ", ".join(
        f"{kind} {n} (in {sum(kind in e for e in found.values())} tracks)"
        for kind, n in totals.items()
    )
    LOG.info(
        f"{len(rows)} tracks with voice, {len(found)} with natural places; "
        f"{counts} -> {args.out}"
    )


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "data-natural",
        parents=[common],
        help="find natural fades, onsets, voiceless and soft places for --natural",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="dataset.csv (or a variant) whose voiced training tracks are searched",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Directory of the teacher's labels",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="JSON of {track: {kind: [sample, ...]}}, such as natural-<variant>.json",
    )
    parser.set_defaults(run=mine)
