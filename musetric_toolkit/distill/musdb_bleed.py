import logging
from pathlib import Path

import numpy as np

from musetric_toolkit.distill.musdb import (
    add_musdb,
    add_teacher_vocals,
    iter_tracks,
    separation,
    teacher_vocals,
)
from musetric_toolkit.distill.scores import sdr
from musetric_toolkit.distill.teacher import SAMPLE_RATE

LOG = logging.getLogger(__name__)

FRAMES_PER_SECOND = 10
FRAME = SAMPLE_RATE // FRAMES_PER_SECOND
POWER_FLOOR = 1e-12
VOICED_RANGE_DB = 60
LOUD_PERCENTILE = 95
SILENT_BELOW_LOUD_DB = 50
MIN_SILENT_FRAMES = 20


def frames_db(x: np.ndarray) -> np.ndarray:
    n = x.shape[1] // FRAME
    power = (x[:, : n * FRAME] ** 2).mean(0).reshape(n, FRAME).mean(1)
    return 10 * np.log10(power + POWER_FLOOR)


def pauses(truth: np.ndarray) -> tuple[np.ndarray, float] | None:
    t_db = frames_db(truth)
    voiced = t_db > t_db.max() - VOICED_RANGE_DB
    if not voiced.any():
        return None
    loud = np.percentile(t_db[voiced], LOUD_PERCENTILE)
    silent = t_db < loud - SILENT_BELOW_LOUD_DB
    if silent.sum() < MIN_SILENT_FRAMES:
        return None
    return silent, loud


def leakage(x: np.ndarray, silent: np.ndarray, loud: float) -> tuple[float, float]:
    levels = frames_db(x)[silent]
    return np.median(levels) - loud, np.percentile(levels, LOUD_PERCENTILE) - loud


def leakage_line(who: str, leaks: np.ndarray) -> str:
    return (
        f"  {who}: median {np.median(leaks[:, 0]):.1f} dB, "
        f"p95 {np.median(leaks[:, 1]):.1f} dB; "
        f"worst track p95 {leaks[:, 1].max():.1f} dB"
    )


def bleed(args) -> None:
    leak_student, leak_teacher, base = [], [], []
    for track, _, truth in iter_tracks(args.musdb):
        student = separation(args.separations, track)
        teacher = teacher_vocals(args.teacher_vocals, track)
        found = pauses(truth)
        if found is None:
            continue
        silent, loud = found
        leak_student.append(leakage(student, silent, loud))
        leak_teacher.append(leakage(teacher, silent, loud))
        base.append(sdr(truth, student))
    LOG.info(
        f"{len(base)} tracks with pauses; leakage in pauses relative to loud "
        "vocals, median of tracks:"
    )
    LOG.info(leakage_line("teacher", np.array(leak_teacher)))
    LOG.info(leakage_line("student", np.array(leak_student)))
    LOG.info(f"  student SDR to truth on these tracks: {np.median(base):.2f} dB")


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "musdb-bleed",
        parents=[common],
        help="how loud saved MUSDB18-HQ separations are where the true vocals "
        "are silent",
    )
    add_musdb(parser)
    add_teacher_vocals(parser)
    parser.add_argument(
        "--separations",
        required=True,
        type=Path,
        help="Directory of <track>.student.npy, as musdb-separate writes them",
    )
    parser.set_defaults(run=bleed)
