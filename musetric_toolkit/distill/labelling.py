import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.distill.corpus import FULL_SCALE, load, to_int16, tracks
from musetric_toolkit.distill.inference import student_forward
from musetric_toolkit.distill.scores import rms, sdr
from musetric_toolkit.distill.teacher import (
    CHUNK,
    SAMPLE_RATE,
    load_teacher,
    read_config,
)
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.roformer_utils import AudioProcessor

LOG = logging.getLogger(__name__)

PEAK = 0.9
PELL_WINDOW_SECONDS = 11
PROGRESS_EVERY = 50


def loudest_window(mix: np.ndarray) -> int:
    seconds = mix.shape[1] // SAMPLE_RATE
    energy = (mix[:, : seconds * SAMPLE_RATE] ** 2).sum(0)
    energy = energy.reshape(seconds, SAMPLE_RATE).sum(1)
    ones = np.ones(PELL_WINDOW_SECONDS)
    start = int(np.argmax(np.convolve(energy, ones, "valid"))) * SAMPLE_RATE
    return min(start, mix.shape[1] - CHUNK)


def label_pell(model, labels: Path, track: dict, mix: np.ndarray) -> dict:
    np.save(labels / f"{track['id']}.mix.npy", to_int16(mix, FULL_SCALE))
    s = loudest_window(mix)
    window = torch.from_numpy(mix[None, :, s : s + CHUNK]).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        heard = student_forward(model, window)[0].float().cpu().numpy()
    share = rms(heard) / max(rms(mix[:, s : s + CHUNK]), 1e-8)
    return dict(
        id=track["id"],
        role="train",
        group="pell",
        pell=True,
        seconds=mix.shape[1] / SAMPLE_RATE,
        vocal_share=1.0,
        teacher_share_db=20 * math.log10(max(share, 1e-8)),
    )


def label_song(teacher, labels: Path, track: dict, mix, gain):
    model, processor, config = teacher
    window = config.audio.hop_length * (config.inference.dim_t - 1)
    padded = np.pad(mix, ((0, 0), (0, max(0, window - mix.shape[1]))))
    with torch.autocast("cuda", dtype=torch.float16):
        vocals = processor.demix(padded, model)["vocals"][:, : mix.shape[1]]
    np.save(labels / f"{track['id']}.mix.npy", to_int16(mix, FULL_SCALE))
    np.save(labels / f"{track['id']}.vocals.npy", to_int16(vocals, FULL_SCALE))
    entry = dict(
        id=track["id"],
        role=track["role"],
        group=track["group"],
        seconds=mix.shape[1] / SAMPLE_RATE,
        vocal_share=rms(vocals) / max(rms(mix), 1e-8),
    )
    if "truth" in track:
        truth = read_audio_file(str(track["truth"]), SAMPLE_RATE, 2)
        truth = truth[:, : mix.shape[1]] * gain
        truth = np.pad(truth, ((0, 0), (0, mix.shape[1] - truth.shape[1])))
        np.save(labels / f"{track['id']}.truth.npy", to_int16(truth, FULL_SCALE))
        entry["teacher_sdr_truth"] = sdr(truth, vocals)
    return entry


def kept_entry(labels: Path, track: dict, before: dict) -> dict:
    entry = {**before[track["id"]], "role": track["role"], "group": track["group"]}
    if not track.get("pell") and "vocal_share" not in entry:
        vocals = rms(load(labels, track["id"], "vocals"))
        entry["vocal_share"] = vocals / max(rms(load(labels, track["id"], "mix")), 1e-8)
    return entry


def read_mix(track: dict):
    try:
        mix = read_audio_file(str(track["mix"]), SAMPLE_RATE, 2)
    except RuntimeError as error:
        LOG.info(f"skip {track['id']}: {error}")
        return None
    if mix.shape[1] < CHUNK + SAMPLE_RATE:
        LOG.info(f"skip {track['id']}: {mix.shape[1] / SAMPLE_RATE:.1f} s")
        return None
    return mix


def label(labels: Path, root: Path, manifest: Path | None, teacher: Path) -> None:
    labels.mkdir(parents=True, exist_ok=True)
    index_path = labels / "labels.json"
    before = {}
    if index_path.exists():
        before = {e["id"]: e for e in json.loads(index_path.read_text())}
    config = read_config(teacher)
    model = load_teacher(teacher).cuda().eval()
    processor = AudioProcessor(torch.device("cuda"), config)
    index = []
    start = time.perf_counter()
    new_seconds = 0.0
    for track in tracks(root, manifest):
        kind = "mix" if track.get("pell") else "vocals"
        if track["id"] in before and (labels / f"{track['id']}.{kind}.npy").exists():
            index.append(kept_entry(labels, track, before))
            continue
        mix = read_mix(track)
        if mix is None:
            continue
        gain = PEAK / max(float(np.abs(mix).max()), 1e-8)
        mix = (mix * gain).astype(np.float32)
        if track.get("pell"):
            entry = label_pell(model, labels, track, mix)
        else:
            teacher_parts = (model, processor, config)
            entry = label_song(teacher_parts, labels, track, mix, gain)
        index.append(entry)
        new_seconds += entry["seconds"]
        if not track.get("pell") and len(index) % PROGRESS_EVERY == 0:
            done = {x["id"] for x in index}
            rest = [e for k, e in before.items() if k not in done]
            index_path.write_text(json.dumps(index + rest, indent=1))
            minutes = (time.perf_counter() - start) / 60
            hours = new_seconds / 3600
            LOG.info(f"{len(index)} tracks, {hours:.2f} h new, {minutes:.1f} min")
    index_path.write_text(json.dumps(index, indent=1))
    hours = sum(t["seconds"] for t in index) / 3600
    minutes = (time.perf_counter() - start) / 60
    LOG.info(
        f"{len(index)} tracks, {hours:.2f} h; labelled {new_seconds / 3600:.2f} h new "
        f"in {minutes:.1f} min"
    )


def label_command(args) -> None:
    label(args.labels, args.corpus, args.manifest, args.teacher)
