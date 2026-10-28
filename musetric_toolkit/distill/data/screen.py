import csv
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.distill.data.analyze import measure
from musetric_toolkit.distill.data.curate import content_of
from musetric_toolkit.distill.data.fma_expand import CANDIDATES, read_candidates
from musetric_toolkit.distill.inference import student_forward
from musetric_toolkit.distill.teacher import CHUNK, SAMPLE_RATE, load_teacher
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file

LOG = logging.getLogger(__name__)

SCREEN = "fma-screen.csv"
BATCH = 2
PEAK = 0.9
PROGRESS_EVERY = 100


def screened(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def middle_window(path: Path, clip_id: str) -> np.ndarray | None:
    try:
        mix = read_audio_file(str(path), SAMPLE_RATE, 2)
    except RuntimeError as error:
        LOG.info(f"skip {clip_id}: {error}")
        return None
    if mix.shape[1] < CHUNK:
        LOG.info(f"skip {clip_id}: {mix.shape[1] / SAMPLE_RATE:.1f} s")
        return None
    start = (mix.shape[1] - CHUNK) // 2
    window = mix[:, start : start + CHUNK]
    return window * (PEAK / max(float(np.abs(window).max()), 1e-8))


def screen_batch(model, folder: Path, batch: list) -> list[dict]:
    names, windows = [], []
    for track in batch:
        window = middle_window(folder / track.file_name, track.clip_id)
        if window is not None:
            windows.append(window)
            names.append(track)
    if not windows:
        return []
    x = torch.from_numpy(np.stack(windows).astype(np.float32)).cuda()
    with torch.no_grad():
        y = student_forward(model, x)
    rows = []
    for track, m, v in zip(names, x, y, strict=True):
        row, _ = measure(track.clip_id, m, v.float())
        row["genre"] = track.genre
        row["content"] = content_of(row)
        rows.append(row)
    return rows


def screen(args) -> None:
    out = args.analysis / SCREEN
    cand = read_candidates(args.corpus / CANDIDATES)
    done = screened(out)
    todo = [t for t in cand if t.clip_id not in done]
    model = load_teacher(args.teacher).cuda().eval()
    folder = args.corpus / "fma"
    start = time.perf_counter()
    with open(out, "a", newline="", encoding="utf-8") as f:
        writer = None
        for group in range(0, len(todo), BATCH):
            rows = screen_batch(model, folder, todo[group : group + BATCH])
            for row in rows:
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    if not done:
                        writer.writeheader()
                writer.writerow(row)
            if (group // BATCH) % PROGRESS_EVERY == 0:
                f.flush()
                minutes = (time.perf_counter() - start) / 60
                LOG.info(f"{group + len(rows)}/{len(todo)}, {minutes:.1f} min")
    with open(out, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    minutes = (time.perf_counter() - start) / 60
    LOG.info(f"{len(rows)} screened in {minutes:.1f} min")
    pairs = Counter((r["genre"], r["content"]) for r in rows)
    for (genre, content), count in sorted(pairs.items()):
        LOG.info(f"{genre}, {content}: {count}")


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "data-screen",
        parents=[common],
        help="screen the candidate clips with the teacher on one 11 s window each",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Corpus directory with fma-candidates.csv and the clips in fma/",
    )
    parser.add_argument(
        "--analysis",
        required=True,
        type=Path,
        help="Analysis directory: fma-screen.csv is appended to there",
    )
    parser.add_argument(
        "--teacher",
        required=True,
        type=Path,
        help="Directory with the teacher's config.yaml and model.ckpt",
    )
    parser.set_defaults(run=screen)
