from pathlib import Path

import soundfile

from musetric_toolkit.listening.levels import db, energy
from musetric_toolkit.listening.round_files import (
    CLIPS,
    KEY,
    LABELS,
    SELECTION,
    read_json,
)

GROUPS = ("lead", "backing", "instrumental")
OLD_CONTENT = {
    "none": dict(lead=False, backing=False, instrumental=None),
    "one": dict(lead=True, backing=False, instrumental=None),
    "lead_backing": dict(lead=True, backing=True, instrumental=None),
}
NO_ANSWER = dict(lead=None, backing=None, instrumental=None)
SIDES = ("a", "b")
NO_VOICE = "no voice"
UNSURE = "unsure"
NO_ANSWER_KIND = "no answer"
LEAD_ONLY = "lead only"
BACKING_ONLY = "backing only"
LEAD_AND_BACKING = "lead and backing"
VOICED_KINDS = (LEAD_ONLY, BACKING_ONLY, LEAD_AND_BACKING)


def present_of(label: dict) -> dict:
    present = label.get("present")
    if present is None:
        return OLD_CONTENT.get(label.get("content"), NO_ANSWER)
    if not present:
        return NO_ANSWER
    return {group: group in present for group in GROUPS}


def kind_of(row: dict) -> str:
    lead, backing = row["present"]["lead"], row["present"]["backing"]
    if row["unsure"]:
        return UNSURE
    if lead is None or backing is None:
        return NO_ANSWER_KIND
    if not lead and not backing:
        return NO_VOICE
    if lead and backing:
        return LEAD_AND_BACKING
    return LEAD_ONLY if lead else BACKING_ONLY


def sides_of(entry: dict) -> dict[str, str]:
    return {entry[side]: side for side in SIDES}


def clip_levels(folder: Path, clip: str, entry: dict) -> dict[str, dict]:
    audio = {}
    for path in (folder / CLIPS / clip).glob("*.wav"):
        audio[path.stem], _ = soundfile.read(path, dtype="float32")
    mix = energy(audio["mix"])
    systems = {}
    for system, side in sides_of(entry).items():
        lead, backing, instrumental = (audio[f"{g}_{side}"] for g in GROUPS)
        systems[system] = dict(
            lead=db(energy(lead), mix),
            backing=db(energy(backing), mix),
            instrumental=db(energy(instrumental), mix),
            vocals=db(energy(lead + backing), mix),
        )
    return systems


def label_row(clip: str, entry: dict, label: dict, genre: str | None) -> dict:
    stems = label["stems"]
    answers = {group: stems[group]["better"] for group in GROUPS}
    return dict(
        clip=clip,
        source=entry["source"],
        sign=entry["sign"],
        train=entry["train"],
        genre=genre,
        repeat_of=entry.get("repeat_of"),
        scores=entry.get("scores"),
        unsure=label.get("unsure", False),
        present=present_of(label),
        votes={g: entry[a] if a in SIDES else a for g, a in answers.items()},
        sides=answers,
        foreign={
            group: {entry[s]: stems[group]["foreign"][s] for s in SIDES}
            for group in GROUPS
        },
        comments={group: stems[group]["comment"] for group in GROUPS},
        comment=label.get("comment", ""),
    )


def load_round(folder: Path, excluded: set[str]) -> list[dict]:
    labels = read_json(folder / LABELS)
    key = read_json(folder / KEY)
    selection = {
        entry["source"]: entry
        for entry in read_json(folder / SELECTION)
        if not entry.get("repeat_of")
    }
    rows = []
    for clip, entry in key.items():
        genre = selection.get(entry["source"], {}).get("genre")
        row = label_row(clip, entry, labels[clip], genre)
        row["systems"] = clip_levels(folder, clip, entry)
        row["excluded"] = entry["source"] in excluded
        rows.append(row)
    return rows
