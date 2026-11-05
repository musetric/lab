import argparse
import csv
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile

from musetric_toolkit.listening.levels import RATE, read_mix
from musetric_toolkit.listening.pictures import (
    DB_FLOOR,
    FMIN,
    OCTAVES,
    STEMS,
    WIDTH,
    write_clip,
)
from musetric_toolkit.listening.round_files import (
    CLIPS,
    KEY,
    MANIFEST,
    MESSAGES,
    SELECTION,
    SYSTEMS_IN_ROUND,
    corpus_tracks,
    read_json,
    write_json,
)
from musetric_toolkit.listening.systems import load_system

LOG = logging.getLogger(__name__)
HALF = 0.5
PARTS = ("lead", "backing", "instrumental", "vocals")
CLASS_COLORS = dict(
    backing="#ffb74d",
    echo="#ce93d8",
    instrument="#81c784",
    breath="#80deea",
    split="#f48fb1",
    lead="#fff176",
    artifact="#ef9a9a",
    unknown="#bdbdbd",
)


def stem_rows(messages: dict) -> list[dict]:
    return [dict(id=stem, title=messages["stems"][stem]) for stem in STEMS]


def class_rows(messages: dict) -> list[dict]:
    return [
        dict(
            id=name,
            title=messages["classes"][name]["title"],
            hint=messages["classes"][name]["hint"],
            color=color,
        )
        for name, color in CLASS_COLORS.items()
    ]


def fma_pages(training_data: Path | None) -> dict[str, str]:
    if training_data is None:
        return {}
    pages = {}
    with training_data.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("fma_track_id"):
                pages[f"fma-{int(row['fma_track_id']):06d}"] = row["page"]
    return pages


def credit_of(source: str, tracks: dict, pages: dict) -> dict:
    item = tracks[source]
    return dict(
        artist=item["artist"],
        title=item["title"],
        license=item["license"],
        license_url=item["licenseUrl"],
        page=pages.get(source) or item.get("page"),
        source=item["source"],
    )


def stems_from_folders(
    entry: dict, names: list[str], seconds: float
) -> tuple[dict, np.ndarray]:
    start = int(entry["start_s"] * RATE)
    stop = int((entry["start_s"] + seconds) * RATE)
    stems = {}
    for name in names:
        folder = Path(entry["stems"][name])
        stems[name] = {
            part: soundfile.read(
                folder / f"{part}.wav", dtype="float32", start=start, stop=stop
            )[0].T
            for part in PARTS
        }
    last = stems[names[-1]]
    return stems, last["vocals"] + last["instrumental"]


def stems_from_systems(
    entry: dict, systems: dict, args: argparse.Namespace
) -> tuple[dict, np.ndarray]:
    mix = read_mix(args.labels, entry["source"])
    start = int(entry["start_s"] * RATE)
    stop = int((entry["start_s"] + args.clip_seconds) * RATE)
    stems = {}
    for name, system in systems.items():
        separated = system.separate(mix)
        stems[name] = {part: stem[:, start:stop] for part, stem in separated.items()}
    return stems, mix[:, start:stop]


def draw_sides(rng, names: list[str]) -> tuple[str, str]:
    first, second = names
    return (first, second) if rng.random() < HALF else (second, first)


def save_raw_clip(folder: Path, mix: np.ndarray, a: dict, b: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        mix=mix,
        lead_a=a["lead"],
        lead_b=b["lead"],
        backing_a=a["backing"],
        backing_b=b["backing"],
        instrumental_a=a["instrumental"],
        instrumental_b=b["instrumental"],
    )
    for name, value in arrays.items():
        np.save(folder / f"{name}.raw.npy", value.astype(np.float32))


def key_row(entry: dict, a: str, b: str) -> dict:
    row = dict(
        source=entry["source"],
        sign=entry["sign"],
        a=a,
        b=b,
        start_s=entry["start_s"],
        train=entry.get("train", True),
        scores=entry.get("scores"),
    )
    if entry.get("repeat_of"):
        row["repeat_of"] = entry["repeat_of"]
    return row


def manifest_row(clip_id: str, entry: dict, tracks: dict, pages: dict) -> dict:
    train = entry.get("train", True)
    row = dict(id=clip_id, start_s=entry["start_s"], train=train)
    if train:
        row["credit"] = credit_of(entry["source"], tracks, pages)
    else:
        row["note"] = entry["note"]
    return row


def make_manifest(args: argparse.Namespace, clips: list[dict]) -> dict:
    messages = read_json(MESSAGES)
    return dict(
        **({"ask": args.ask} if args.ask else {}),
        round=args.out.name,
        duration=args.clip_seconds,
        rate=RATE,
        default_stem=args.default_stem,
        spectrum=dict(fmin=FMIN, octaves=OCTAVES, width=WIDTH, db_floor=DB_FLOOR),
        stems=stem_rows(messages),
        classes=class_rows(messages),
        clips=clips,
    )


def system_names(args: argparse.Namespace) -> list[str]:
    names = [name for name, _ in args.system]
    if len(names) != SYSTEMS_IN_ROUND or len(set(names)) != SYSTEMS_IN_ROUND:
        message = f"a round compares two different systems, not {names}"
        raise ValueError(message)
    return names


def separate_clips(args: argparse.Namespace, selection: list[dict]) -> tuple:
    names = system_names(args)
    rng = np.random.default_rng(args.seed)
    tracks = corpus_tracks(args.corpus)
    pages = fma_pages(args.training_data)
    systems = None
    clips, key = [], {}
    for number, entry in enumerate(selection, start=1):
        clip_id = f"c{number:0{args.id_width}d}"
        if entry.get("stems"):
            stems, mix = stems_from_folders(entry, names, args.clip_seconds)
        else:
            if systems is None:
                systems = {name: load_system(name, f) for name, f in args.system}
            stems, mix = stems_from_systems(entry, systems, args)
        a, b = draw_sides(rng, names)
        folder = args.out / CLIPS / clip_id
        save_raw_clip(folder, mix, stems[a], stems[b])
        clips.append(manifest_row(clip_id, entry, tracks, pages))
        key[clip_id] = key_row(entry, a, b)
        LOG.info("separated %s", clip_id)
    return clips, key


def build_command(args: argparse.Namespace) -> None:
    selection = read_json(args.selection)
    clips, key = separate_clips(args, selection)
    jobs = [(str(args.out / CLIPS / clip["id"]), args.clip_seconds) for clip in clips]
    with ProcessPoolExecutor(args.workers) as pool:
        for name in pool.map(write_clip, jobs):
            LOG.info("pictured %s", name)
    write_json(args.out / MANIFEST, make_manifest(args, clips))
    write_json(args.out / KEY, key)
    kept = args.out / SELECTION
    if not (kept.exists() and kept.samefile(args.selection)):
        shutil.copyfile(args.selection, kept)
    LOG.info("done")
