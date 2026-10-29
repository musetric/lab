import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from musetric_toolkit.distill.corpus import FULL_SCALE, to_int16, tracks
from musetric_toolkit.distill.inference import infer_overlap
from musetric_toolkit.distill.labelling import PEAK, read_mix
from musetric_toolkit.distill.precision import PRECISION
from musetric_toolkit.distill.scores import rms
from musetric_toolkit.distill.teacher import SAMPLE_RATE
from musetric_toolkit.separate_audio.mdx_net_separator import MDXNetSeparator
from musetric_toolkit.separate_audio.roformer.lead_backing_roformer import (
    LeadBackingRoformer,
)
from musetric_toolkit.separate_audio.roformer.pooled_mel_band_roformer import (
    PooledMelBandRoformer,
)

LOG = logging.getLogger(__name__)

KARA_MODEL = "UVR_MDXNET_KARA_2.onnx"
KARA_DATA = "model_data.json"
INDEX = "voices.json"
TEACHER = "teacher.json"
SAVE_EVERY = 50
SILENT = 1e-6


def model_config(release: Path) -> dict:
    return yaml.safe_load((release / "config.yaml").read_text(encoding="utf-8"))[
        "model"
    ]


def load_release(release: Path) -> PooledMelBandRoformer:
    model = PooledMelBandRoformer(**model_config(release))
    model.load_state_dict(load_file(release / "model.safetensors"))
    return model


def load_lead_backing(release: Path, split_hidden: int = 0) -> LeadBackingRoformer:
    config = model_config(release)
    split_hidden = config.pop("split_hidden", split_hidden)
    model = LeadBackingRoformer(**config, split_hidden=split_hidden)
    state = load_file(release / "model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("split_estimator.") for k in missing):
        message = f"unexpected weights: {unexpected}, missing: {missing}"
        raise ValueError(message)
    return model


def load_split(model: LeadBackingRoformer, state: dict) -> None:
    model.split_estimator.load_state_dict(state["split"])
    model.load_state_dict(state.get("pairs", {}), strict=False)
    factor = state.get("compensate", 1.0)
    if factor != 1.0:
        model.scale_backing(factor)


def backing_scale(labels: Path) -> float:
    return json.loads((labels / TEACHER).read_text())["backing_scale"]


def write_teacher(labels: Path, separator: MDXNetSeparator) -> None:
    path = labels / TEACHER
    scale = separator.compensation()
    if path.exists() and backing_scale(labels) != scale:
        message = f"{path} has another backing scale than KARA 2's model data"
        raise ValueError(message)
    teacher = dict(backing_scale=scale, source=KARA_DATA)
    path.write_text(json.dumps(teacher, indent=1))


def kara_separator(kara: Path) -> MDXNetSeparator:
    return MDXNetSeparator(kara / KARA_MODEL, kara / KARA_DATA, SAMPLE_RATE)


def kara_backing(separator: MDXNetSeparator, vocals: np.ndarray) -> np.ndarray:
    if np.abs(vocals).max() < SILENT:
        return np.zeros_like(vocals)
    backing, _ = separator.demix_array(vocals.astype(np.float32))
    return backing.astype(np.float32)


def save_parts(labels: Path, track_id: str, parts: dict[str, np.ndarray]) -> None:
    for kind, audio in parts.items():
        np.save(labels / f"{track_id}.{kind}.npy", to_int16(audio, FULL_SCALE))


def entry_of(track: dict, mix, vocals, backing) -> dict:
    return dict(
        id=track["id"],
        role=track["role"],
        group=track["group"],
        pell=bool(track.get("pell")),
        seconds=mix.shape[1] / SAMPLE_RATE,
        vocal_share=rms(vocals) / max(rms(mix), 1e-8),
        backing_share=rms(backing) / max(rms(vocals), 1e-8),
    )


def merged(index: list[dict], before: dict[str, dict]) -> list[dict]:
    done = {e["id"] for e in index}
    return index + [e for key, e in before.items() if key not in done]


def label_voices(args) -> None:
    labels = args.labels
    labels.mkdir(parents=True, exist_ok=True)
    index_path = labels / INDEX
    before = {}
    if index_path.exists():
        before = {e["id"]: e for e in json.loads(index_path.read_text())}
    PRECISION.dtype = torch.float16
    student = load_release(args.student).cuda().eval()
    separator = kara_separator(args.kara)
    write_teacher(labels, separator)
    index, start, new_seconds = [], time.perf_counter(), 0.0
    for track in tracks(args.corpus, args.manifest):
        if track["id"] in before and (labels / f"{track['id']}.backing.npy").exists():
            index.append(before[track["id"]])
            continue
        mix = read_mix(track)
        if mix is None:
            continue
        mix = (mix * PEAK / max(float(np.abs(mix).max()), 1e-8)).astype(np.float32)
        vocals = infer_overlap(student, mix)
        backing = kara_backing(separator, vocals)
        save_parts(labels, track["id"], dict(mix=mix, vocals=vocals, backing=backing))
        entry = entry_of(track, mix, vocals, backing)
        index.append(entry)
        new_seconds += entry["seconds"]
        if len(index) % SAVE_EVERY == 0:
            index_path.write_text(json.dumps(merged(index, before), indent=1))
            minutes = (time.perf_counter() - start) / 60
            LOG.info(
                f"{len(index)} tracks, {new_seconds / 3600:.2f} h new, "
                f"{minutes:.1f} min"
            )
    index_path.write_text(json.dumps(merged(index, before), indent=1))
    minutes = (time.perf_counter() - start) / 60
    LOG.info(
        f"{len(index)} tracks; labelled {new_seconds / 3600:.2f} h new "
        f"in {minutes:.1f} min"
    )


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "label-voices",
        parents=[common],
        help="label the corpus with the vocals student and KARA 2's backing",
    )
    parser.add_argument(
        "--student",
        required=True,
        type=Path,
        help="Directory of the released student: config.yaml, model.safetensors",
    )
    parser.add_argument(
        "--kara",
        required=True,
        type=Path,
        help=f"Directory of {KARA_MODEL} and UVR's {KARA_DATA}",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Directory with manifest.json, fma.json and ccmixter-pells.json",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Output: <id>.mix/vocals/backing.npy and voices.json",
    )
    parser.set_defaults(run=label_voices)
