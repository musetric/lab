import csv
import logging
from pathlib import Path

import numpy as np
import soundfile

from musetric_toolkit.distill.excluded import excluded_ids
from musetric_toolkit.listening.listener import Listener
from musetric_toolkit.listening.round_files import (
    CLIPS,
    FEATURES,
    KEY,
    corpus_tracks,
    read_json,
    write_json,
)
from musetric_toolkit.listening.round_labels import load_round, sides_of
from musetric_toolkit.listening.voice_features import dsp_features

LOG = logging.getLogger(__name__)


def echonest_rows(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle)
        next(rows)
        next(rows)
        names = next(rows)
        next(rows)
        column = {name: index for index, name in enumerate(names)}
        last = max(column["instrumentalness"], column["speechiness"])
        return {
            int(row[0]): dict(
                instrumentalness=float(row[column["instrumentalness"]]),
                speechiness=float(row[column["speechiness"]]),
            )
            for row in rows
            if len(row) > last
        }


def read_stem(folder: Path, name: str) -> np.ndarray:
    return soundfile.read(folder / f"{name}.wav", dtype="float32")[0].T


def add_features(row: dict, sides: dict, folder: Path, listener: Listener) -> None:
    mix = read_stem(folder, "mix")
    vocals = {
        system: read_stem(folder, f"lead_{sides[system]}")
        + read_stem(folder, f"backing_{sides[system]}")
        for system in sorted(sides)
    }
    heard = listener([mix, *vocals.values()])
    row["whisper"] = dict(mix=heard[0], **dict(zip(vocals, heard[1:], strict=True)))
    row["dsp"] = {system: dsp_features(vocals[system], mix) for system in vocals}


def features_command(args) -> None:
    key = read_json(args.round / KEY)
    rows = load_round(args.round, excluded_ids())
    tracks = corpus_tracks(args.corpus)
    echonest = echonest_rows(args.echonest)
    listener = Listener(args.batch)
    LOG.info("whisper ready")
    for row in rows:
        folder = args.round / CLIPS / row["clip"]
        add_features(row, sides_of(key[row["clip"]]), folder, listener)
        info = tracks.get(row["source"], {})
        row["instrumental_flag"] = info.get("instrumental")
        row["echonest"] = echonest.get(info.get("fmaId"))
        LOG.info("%s", row["clip"])
    write_json(args.round / FEATURES, rows)
