import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

HELD_OTHER = {"jw-dust", "ccm-winter-sunlight", "jamalt-en-rxbyn-bad-side"}
VOCAL_SHARE_MIN = 0.1
FULL_SCALE = 2.0
INT16_MAX = 32767


def manifest_tracks(manifest: Path) -> list[dict]:
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    tracks = []
    for e in entries:
        track = dict(id=e["id"], mix=Path(e["audio"]), role=e["role"], group=e["group"])
        if "truth" in e:
            track["truth"] = Path(e["truth"])
        tracks.append(track)
    return tracks


def open_tracks(root: Path) -> list[dict]:
    out = []
    for song in json.loads((root / "manifest.json").read_text(encoding="utf-8")):
        track = dict(
            id=song["id"], mix=root / song["audio"], role="train", group="open"
        )
        if "truth" in song:
            track.update(truth=root / song["truth"], role="held", group="argonov-aria")
        elif song["id"] in HELD_OTHER:
            track.update(role="held", group="open-other")
        out.append(track)
    fma = root / "fma.json"
    if fma.exists():
        for clip in json.loads(fma.read_text(encoding="utf-8")):
            if clip["split"] in ("train", "held"):
                group = f"fma:{clip['genre']}"
                out.append(
                    dict(
                        id=clip["id"],
                        mix=root / clip["audio"],
                        role=clip["split"],
                        group=group,
                    )
                )
    pells = root / "ccmixter-pells.json"
    if pells.exists():
        for song in json.loads(pells.read_text(encoding="utf-8")):
            out.append(
                dict(
                    id=song["id"],
                    mix=root / song["audio"],
                    role="train",
                    group="pell",
                    pell=True,
                )
            )
    return out


def tracks(root: Path, manifest: Path | None = None) -> list[dict]:
    return manifest_tracks(manifest) if manifest else open_tracks(root)


def dataset_rows(dataset: Path | None) -> dict[str, dict]:
    if dataset is None or not dataset.exists():
        return {}
    with open(dataset, encoding="utf-8-sig", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def usable(index: list[dict], rows: dict[str, dict]) -> list[dict]:
    if not rows:
        return index
    return [t for t in index if rows.get(t["id"], {}).get("decision") == "use"]


def held_tracks(index: list[dict]) -> list[dict]:
    return [
        t
        for t in index
        if t["role"] == "held" and t.get("vocal_share", 1.0) >= VOCAL_SHARE_MIN
    ]


def to_int16(x: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.round(x / scale * INT16_MAX)
    return np.clip(scaled, -INT16_MAX - 1, INT16_MAX).astype(np.int16)


def load(labels: Path, name: str, kind: str) -> np.ndarray:
    data = np.load(labels / f"{name}.{kind}.npy").astype(np.float32)
    return data * (FULL_SCALE / INT16_MAX)


def label_files(labels: Path) -> dict[str, Path]:
    return {f.name.removesuffix(".flac"): f for f in labels.rglob("*.flac")}


class FlacTrack:
    def __init__(self, path: Path, frames: int, channels: int = 2):
        self.path, self.shape = path, (channels, frames)

    def __getitem__(self, key):
        _, frames = key
        data = sf.read(
            self.path,
            start=frames.start,
            stop=frames.stop,
            dtype="int16",
            always_2d=True,
        )[0]
        return data.T
