import json
import math
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile

from musetric_toolkit.listening.audition import audition
from musetric_toolkit.listening.round_files import (
    ASK_PRESENT,
    CLIPS,
    LABELS,
    LABELS_LOG,
    MANIFEST,
    read_json,
)
from musetric_toolkit.listening.round_labels import GROUPS, SIDES

ANSWERS = {"a", "b", "same", "unknown"}
MIN_POINTS = 3
MAX_POINTS = 2000
POINT_SIZE = 2
CACHED_STEMS = 28


def valid_point(point) -> bool:
    if not isinstance(point, list) or len(point) != POINT_SIZE:
        return False
    return all(isinstance(v, int | float) and math.isfinite(v) for v in point)


def valid_polygon(polygon) -> bool:
    if not isinstance(polygon, list):
        return False
    if not MIN_POINTS <= len(polygon) <= MAX_POINTS:
        return False
    return all(valid_point(point) for point in polygon)


def valid_stem_answer(group: str, answer) -> bool:
    if not isinstance(answer, dict) or not isinstance(answer.get("comment", ""), str):
        return False
    if answer.get("better") is not None and answer.get("better") not in ANSWERS:
        return False
    foreign = answer.get("foreign")
    if not isinstance(foreign, dict) or set(foreign) != set(SIDES):
        return False
    allowed = set(GROUPS) - {group}
    return all(
        isinstance(heard, list) and set(heard) <= allowed for heard in foreign.values()
    )


class StandRound:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.manifest = read_json(folder / MANIFEST)
        self.clips = {clip["id"]: clip for clip in self.manifest["clips"]}
        self.stems = {stem["id"] for stem in self.manifest["stems"]}
        self.classes = {item["id"] for item in self.manifest["classes"]}
        self.asks_present = self.manifest.get("ask") == ASK_PRESENT
        self.labels_path = folder / LABELS
        self.log_path = folder / LABELS_LOG
        self.lock = threading.Lock()
        self.labels = {}
        if self.labels_path.exists():
            self.labels = read_json(self.labels_path)
        self.read = lru_cache(maxsize=CACHED_STEMS)(self.read_stem)

    def read_stem(self, clip: str, stem: str) -> tuple[np.ndarray, int]:
        path = self.folder / CLIPS / clip / f"{stem}.wav"
        audio, rate = soundfile.read(path, dtype="float32")
        return audio.T, rate

    def valid_region(self, region) -> bool:
        if not isinstance(region, dict):
            return False
        if region.get("cls") not in self.classes:
            return False
        if region.get("stem") not in self.stems:
            return False
        return valid_polygon(region.get("polygon"))

    def valid_stems(self, data: dict) -> bool:
        stems = data.get("stems")
        if not isinstance(stems, dict) or set(stems) != set(GROUPS):
            return False
        if not all(valid_stem_answer(group, stems[group]) for group in GROUPS):
            return False
        if not data.get("done") or self.asks_present:
            return True
        return all(stems[group].get("better") in ANSWERS for group in GROUPS)

    def valid_present(self, data: dict) -> bool:
        present = data.get("present")
        if present is not None and not (
            isinstance(present, list) and all(v in GROUPS for v in present)
        ):
            return False
        if data.get("unsure") not in (None, False, True):
            return False
        if not data.get("done") or not self.asks_present:
            return True
        return bool(present or data.get("unsure"))

    def valid_label(self, clip, data) -> bool:
        if clip not in self.clips or not isinstance(data, dict):
            return False
        if not self.valid_stems(data) or not self.valid_present(data):
            return False
        regions = data.get("regions", [])
        if not isinstance(regions, list):
            return False
        return all(self.valid_region(region) for region in regions)

    def save(self, clip, data) -> bool:
        if not self.valid_label(clip, data):
            return False
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.lock:
            self.labels[clip] = {**data, "updated": stamp}
            text = json.dumps(self.labels, indent=1, ensure_ascii=False)
            temporary = self.labels_path.with_suffix(".tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(self.labels_path)
            event = dict(clip=clip, at=stamp, data=data)
            with self.log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(event, ensure_ascii=False) + "\n")
        return True

    def audition(
        self, clip: str, stem: str, mode: str, polygon: list
    ) -> tuple[bytes, float, float]:
        audio, rate = self.read(clip, stem)
        return audition(audio, rate, mode, polygon)
