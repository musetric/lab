import json
import logging
from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.distill.precision import PRECISION
from musetric_toolkit.distill.scores import sdr
from musetric_toolkit.distill.teacher import SAMPLE_RATE
from musetric_toolkit.distill.voices import load_lead_backing
from musetric_toolkit.distill.voices_scores import (
    golden_tracks,
    read_stem,
    score_stems,
    separate_stems,
)
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file

LOG = logging.getLogger(__name__)

SETS = ("mix", "acappella")
METRICS = ("lead_sdr", "backing_sdr", "vocals_sdr")
BOOT = 10000


def read_stereo(path: Path) -> np.ndarray:
    return read_audio_file(str(path), SAMPLE_RATE, 2).astype(np.float32)


def per_track(model, musdb: Path, golden: Path) -> dict:
    rows = dict(musdb={})
    for name in json.loads((musdb / "tracks.json").read_text("utf-8")):
        mix = read_stereo(musdb / name / "mixture.wav")
        truth = read_stereo(musdb / name / "vocals.wav")
        rows["musdb"][name] = sdr(truth, separate_stems(model, mix)["vocals"])
    for kind in SETS:
        rows[kind] = {}
        for folder in golden_tracks(golden / kind):
            mix = read_stem(folder, "mixture")
            refs = {
                stem: audio
                for stem in ("vocals", "lead", "backing")
                if (audio := read_stem(folder, stem)) is not None
            }
            if "vocals" not in refs and {"lead", "backing"} <= refs.keys():
                refs["vocals"] = refs["lead"] + refs["backing"]
            rows[kind][folder.name] = score_stems(refs, separate_stems(model, mix), mix)
    return rows


def paired(a: dict, b: dict, rng) -> str:
    diffs = np.array(
        [a[k] - b[k] for k in a if a[k] is not None and b.get(k) is not None]
    )
    if not len(diffs):
        return "n=0"
    means = rng.choice(diffs, (BOOT, len(diffs))).mean(1)
    low, high = np.quantile(means, [0.025, 0.975])
    return f"{diffs.mean():+.2f} [{low:+.2f}, {high:+.2f}] n={len(diffs)}"


def score_paired(args) -> None:
    path = args.checks / f"{args.name}.json"
    if path.exists():
        rows = json.loads(path.read_text("utf-8"))
    else:
        PRECISION.dtype = torch.float16
        model = load_lead_backing(args.model).cuda().eval()
        with torch.no_grad():
            rows = per_track(model, args.musdb, args.golden)
        args.checks.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1), "utf-8")
    base = json.loads((args.checks / f"{args.baseline}.json").read_text("utf-8"))
    rng = np.random.default_rng(args.seed)
    LOG.info(
        f"{args.name} against {args.baseline}, MUSDB vocals median "
        f"{np.median(list(rows['musdb'].values())):.2f} and "
        f"{np.median(list(base['musdb'].values())):.2f}, "
        f"paired {paired(rows['musdb'], base['musdb'], rng)}"
    )
    for kind in SETS:
        for metric in METRICS:
            a = {track: row.get(metric) for track, row in rows[kind].items()}
            b = {track: row.get(metric) for track, row in base[kind].items()}
            known = [value for value in a.values() if value is not None]
            if known:
                LOG.info(
                    f"{kind} {metric} median {np.median(known):.2f}, "
                    f"paired {paired(a, b, rng)}"
                )


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "score-paired",
        parents=[common],
        help="score a release on MUSDB18-HQ and ccMixter, paired with another",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Directory of a release: config.yaml, model.safetensors",
    )
    parser.add_argument("--name", required=True, help="Its name among the checks")
    parser.add_argument(
        "--baseline", required=True, help="Name of the scores to pair with"
    )
    parser.add_argument(
        "--checks",
        required=True,
        type=Path,
        help="Directory of the scores per model, <name>.json",
    )
    parser.add_argument(
        "--musdb",
        required=True,
        type=Path,
        help="MUSDB18-HQ test set with tracks.json",
    )
    parser.add_argument(
        "--golden",
        required=True,
        type=Path,
        help="Set of data-ccmixter-voices: mix/ and acappella/",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(run=score_paired)
