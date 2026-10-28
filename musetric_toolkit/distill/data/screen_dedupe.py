import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.distill.data.analyze import measure
from musetric_toolkit.distill.data.duplicates import against
from musetric_toolkit.distill.data.fma_fetch import FMA_PREFIX, clip_file
from musetric_toolkit.distill.teacher import SAMPLE_RATE
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file

LOG = logging.getLogger(__name__)

PEAK = 0.9
PROGRESS_EVERY = 500


def clip_fingerprint(path: Path, clip_id: str, device: str) -> np.ndarray:
    mix = read_audio_file(str(path), SAMPLE_RATE, 2)
    mix = mix * (PEAK / max(float(np.abs(mix).max()), 1e-8))
    m = torch.from_numpy(mix.astype(np.float32)).to(device)
    return measure(clip_id, m, torch.zeros_like(m))[1]


def dedupe(args) -> None:
    with open(args.analysis / "fma-screen.csv", encoding="utf-8", newline="") as f:
        ids = [row["id"] for row in csv.DictReader(f)]
    folder = args.corpus / "fma"
    prints = []
    for n, clip_id in enumerate(ids):
        track_id = int(clip_id.removeprefix(FMA_PREFIX))
        path = folder / clip_file(track_id)
        prints.append(clip_fingerprint(path, clip_id, args.device))
        if n % PROGRESS_EVERY == 0:
            LOG.info(f"{n}/{len(ids)}")
    new = np.stack(prints)
    np.save(args.analysis / "fma-screen-fingerprints.npy", new)
    path = args.analysis / "fma-screen-fingerprint_ids.json"
    path.write_text(json.dumps(ids), encoding="utf-8")
    old = np.load(args.analysis / "fingerprints.npy")
    old_ids = json.loads((args.analysis / "fingerprint_ids.json").read_text())
    found = against(new, ids, old, old_ids)
    path = args.analysis / "fma-screen-duplicates.json"
    path.write_text(json.dumps(found, indent=1), encoding="utf-8")
    LOG.info(f"{len(found)} duplicates: {found}")


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "data-dedupe",
        parents=[common],
        help="find the screened clips that repeat each other or the corpus",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Corpus directory with the clips in fma/",
    )
    parser.add_argument(
        "--analysis",
        required=True,
        type=Path,
        help="Analysis directory with fma-screen.csv and the corpus fingerprints; "
        "fma-screen-fingerprints.npy, -fingerprint_ids.json, -duplicates.json "
        "are written there",
    )
    parser.add_argument("--device", default="cuda", help="Torch device of the STFTs")
    parser.set_defaults(run=dedupe)
