import json
import logging
from pathlib import Path

from musetric_toolkit.distill.data.remote_zip import (
    GIGABYTE,
    fetch_members,
    open_archive,
)

LOG = logging.getLogger(__name__)

URL = "https://zenodo.org/api/records/3338373/files/musdb18hq.zip/content"
SPLIT = "test"
WANTED = ("mixture.wav", "vocals.wav")
MEMBER_PARTS = 3
WORKERS = 6
PROGRESS_EVERY = 10
LICENSE = (
    "# MUSDB18-HQ test set\n\n"
    "Rafii, Liutkus, Stoter, Mimilakis, Bittner: MUSDB18-HQ, "
    "https://zenodo.org/records/3338373 (license: non-commercial research). "
    "Evaluation only: never trained on, never published. "
    "Only mixture.wav and vocals.wav of the test tracks are here.\n"
)


def fetch(args) -> None:
    out = args.out
    archive = open_archive(URL)
    jobs = []
    for info in archive.infolist():
        parts = info.filename.split("/")
        if len(parts) == MEMBER_PARTS and parts[0] == SPLIT and parts[2] in WANTED:
            target = out / parts[1] / parts[2]
            if not target.exists():
                jobs.append((info, target))
    tracks = {info.filename.split("/")[1] for info, _ in jobs}
    if out.exists():
        tracks |= {p.name for p in out.iterdir()}
    size = sum(info.compress_size for info, _ in jobs) / GIGABYTE
    LOG.info(f"{len(tracks)} test tracks, {len(jobs)} files to fetch, {size:.1f} GB")
    fetch_members(URL, jobs, args.workers, PROGRESS_EVERY)
    names = sorted(p.name for p in out.iterdir() if p.is_dir())
    (out / "tracks.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    (out / "license.md").write_text(LICENSE, encoding="utf-8")
    LOG.info("done")


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "data-musdb",
        parents=[common],
        help="fetch the MUSDB18-HQ test set (mixture and vocals) for evaluation",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory of <track>/mixture.wav and vocals.wav, tracks.json, license.md",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.set_defaults(run=fetch)
