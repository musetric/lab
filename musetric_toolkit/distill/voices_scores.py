import json
import logging
from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.distill.corpus import FULL_SCALE, INT16_MAX
from musetric_toolkit.distill.inference import overlap_add
from musetric_toolkit.distill.precision import PRECISION
from musetric_toolkit.distill.scores import rms, sdr
from musetric_toolkit.distill.teacher import SAMPLE_RATE
from musetric_toolkit.distill.voices import (
    backing_scale,
    kara_backing,
    kara_separator,
    load_lead_backing,
    load_split,
)
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file

LOG = logging.getLogger(__name__)

STEMS = ("vocals", "backing", "lead", "instrumental")
FRAME = 4410
ACTIVE_DB = -30.0
VOICED_SHARE = 0.1
BACKING_PRESENT_DB = -20.0


def separate_stems(model, mix: np.ndarray) -> dict[str, np.ndarray]:
    model.eval()

    def forward(windows: torch.Tensor) -> torch.Tensor:
        out = model.separate(windows, PRECISION.dtype)
        return torch.stack([out[s] for s in STEMS], dim=1)

    stacked = overlap_add(forward, mix)
    return dict(zip(STEMS, stacked, strict=True))


def active_frames(ref: np.ndarray, db: float = ACTIVE_DB) -> np.ndarray:
    n = ref.shape[1] // FRAME
    energy = (
        (ref[:, : n * FRAME].astype(np.float64) ** 2).mean(0).reshape(n, FRAME).mean(1)
    )
    loud = np.quantile(energy, 0.9) if n else 0.0
    return energy > loud * 10 ** (db / 10)


def active_sdr(ref: np.ndarray, est: np.ndarray) -> float | None:
    frames = active_frames(ref)
    if frames.sum() == 0:
        return None
    keep = np.repeat(frames, FRAME)
    n = keep.shape[0]
    return sdr(ref[:, :n][:, keep], est[:, :n][:, keep])


def silent_level(ref: np.ndarray, est: np.ndarray, mix: np.ndarray) -> float | None:
    frames = ~active_frames(ref)
    if frames.sum() == 0:
        return None
    keep = np.repeat(frames, FRAME)
    n = keep.shape[0]
    est_energy = np.mean(est[:, :n][:, keep].astype(np.float64) ** 2) + 1e-20
    mix_energy = np.mean(mix[:, :n].astype(np.float64) ** 2) + 1e-20
    return float(10 * np.log10(est_energy / mix_energy))


def level_db(part: np.ndarray, whole: np.ndarray) -> float:
    return float(20 * np.log10(rms(part) / max(rms(whole), 1e-12) + 1e-12))


def score_stems(refs: dict, ests: dict, mix: np.ndarray) -> dict:
    row = {}
    if "lead" in refs:
        row["lead_sdr"] = sdr(refs["lead"], ests["lead"])
        row["lead_active_sdr"] = active_sdr(refs["lead"], ests["lead"])
    if "backing" in refs:
        present = (
            level_db(refs["backing"], refs.get("vocals", mix)) > BACKING_PRESENT_DB
        )
        if present:
            row["backing_sdr"] = sdr(refs["backing"], ests["backing"])
            row["backing_active_sdr"] = active_sdr(refs["backing"], ests["backing"])
            row["backing_silent_db"] = silent_level(
                refs["backing"], ests["backing"], mix
            )
        else:
            row["backing_absent_db"] = level_db(ests["backing"], mix)
    if "vocals" in refs:
        row["vocals_sdr"] = sdr(refs["vocals"], ests["vocals"])
    return row


def load_label(labels: Path, track_id: str, kind: str) -> np.ndarray:
    return np.load(labels / f"{track_id}.{kind}.npy").astype(np.float32) * (
        FULL_SCALE / INT16_MAX
    )


def held_out_scores(model, labels: Path, limit: int) -> list[dict]:
    index = json.loads((labels / "voices.json").read_text())
    held = [
        t
        for t in index
        if t["role"] == "held" and not t["pell"] and t["vocal_share"] >= VOICED_SHARE
    ][:limit]
    scale = backing_scale(labels)
    rows = []
    for track in held:
        mix = load_label(labels, track["id"], "mix")
        vocals = load_label(labels, track["id"], "vocals")
        backing = load_label(labels, track["id"], "backing") * scale
        refs = dict(backing=backing, lead=vocals - backing, vocals=vocals)
        ests = separate_stems(model, mix)
        rows.append(dict(id=track["id"], **score_stems(refs, ests, mix)))
    return rows


def summary(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, float)})
    out = {}
    for key in keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            out[key] = float(np.median(values))
    out["tracks"] = len(rows)
    out["backing_tracks"] = sum("backing_sdr" in r for r in rows)
    return out


def read_stem(folder: Path, name: str) -> np.ndarray | None:
    path = folder / f"{name}.wav"
    if not path.exists():
        return None
    return read_audio_file(str(path), SAMPLE_RATE, 2).astype(np.float32)


def golden_tracks(golden: Path) -> list[Path]:
    return sorted(p for p in golden.iterdir() if (p / "mixture.wav").exists())


def cascade_stems(student_stems: dict, separator) -> dict:
    vocals = student_stems["vocals"]
    backing = kara_backing(separator, vocals) * separator.compensation()
    return dict(vocals=vocals, backing=backing, lead=vocals - backing)


def score_golden(args) -> None:
    PRECISION.dtype = torch.float16
    state = {}
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    hidden = state.get("split_hidden", 0)
    model = load_lead_backing(args.student, hidden).cuda().eval()
    if state:
        load_split(model, state)
    separator = kara_separator(args.kara) if args.cascade else None
    student = model
    if separator is not None and state:
        student = load_lead_backing(args.student).cuda().eval()
    rows = []
    for folder in golden_tracks(args.golden)[: args.limit or None]:
        mix = read_stem(folder, "mixture")
        refs = {
            name: stem
            for name in ("vocals", "lead", "backing")
            if (stem := read_stem(folder, name)) is not None
        }
        if "vocals" not in refs and {"lead", "backing"} <= refs.keys():
            refs["vocals"] = refs["lead"] + refs["backing"]
        model_stems = separate_stems(model, mix)
        row = dict(track=folder.name)
        if args.checkpoint:
            row["model"] = score_stems(refs, model_stems, mix)
        if separator is not None:
            student_stems = (
                model_stems if student is model else separate_stems(student, mix)
            )
            cascade = cascade_stems(student_stems, separator)
            row["cascade"] = score_stems(refs, cascade, mix)
        rows.append(row)
        LOG.info(json.dumps(row))
    report = dict(
        golden=str(args.golden),
        tracks=rows,
        median={
            system: summary([r[system] for r in rows if system in r])
            for system in ("model", "cascade")
        },
    )
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    LOG.info(json.dumps(report["median"]))


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "score-voices",
        parents=[common],
        help="score lead and backing against true stems: the split and the cascade",
    )
    parser.add_argument("--student", required=True, type=Path)
    parser.add_argument("--kara", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cascade", action="store_true")
    parser.add_argument(
        "--golden",
        required=True,
        type=Path,
        help="Directory of <track>/mixture.wav with lead.wav, backing.wav, vocals.wav",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    parser.set_defaults(run=score_golden)
