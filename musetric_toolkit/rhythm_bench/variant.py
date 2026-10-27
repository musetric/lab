import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# A variant is one way of getting the clicks of a track: an input, a window and
# a postprocessing of Beat This!, or the app's own result. Each lives in the
# results directory as <track>/<variant>.json with its beats and downbeats in
# seconds; the logits it came from, when there are any, sit next to it as
# <track>/<logits>-beat.npy and <track>/<logits>-downbeat.npy.


@dataclass
class Variant:
    beats: np.ndarray
    downbeats: np.ndarray


def variant_path(results_dir: Path, track: str, variant: str) -> Path:
    return results_dir / track / f"{variant}.json"


def read_variant(results_dir: Path, track: str, variant: str) -> Variant:
    data = json.loads(variant_path(results_dir, track, variant).read_text())
    return Variant(
        beats=np.asarray(data["beats"], dtype=np.float64),
        downbeats=np.asarray(data["downbeats"], dtype=np.float64),
    )


def write_variant(
    results_dir: Path,
    track: str,
    variant: str,
    result: dict,
) -> None:
    path = variant_path(results_dir, track, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result) + "\n")


def logits_path(results_dir: Path, track: str, logits: str, head: str) -> Path:
    return results_dir / track / f"{logits}-{head}.npy"


def tracks_with(results_dir: Path, variants: list[str]) -> list[str]:
    return sorted(
        track.name
        for track in results_dir.iterdir()
        if track.is_dir()
        and all((track / f"{variant}.json").is_file() for variant in variants)
    )
