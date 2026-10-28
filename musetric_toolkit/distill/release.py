import csv
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file

from musetric_toolkit.distill.labelling import PEAK
from musetric_toolkit.distill.student import Fits, build_student
from musetric_toolkit.distill.teacher import CHUNK, SAMPLE_RATE, WINDOW_FRAMES
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.roformer.pooled_mel_band_roformer import (
    PooledMelBandRoformer,
)

LOG = logging.getLogger(__name__)

A_B01 = "pool-s114422+w32@01+m44@012345+f50r@01+a4@01+hr128t+hp2+d+xb0:lin/b1:lin"
SHAPES = {
    A_B01: dict(
        pool=[1, 1, 4, 4, 2, 2],
        merged_bands_from=[44] * 6,
        time_window=[32, 32, None, None, None, None],
        time_heads=[4, 4, 8, 8, 8, 8],
        ff_mult=[2, 2, 4, 4, 4, 4],
        band_map=[True, True, False, False, False, False],
        detail=[None, None, ["d", "c"], ["d", "c"], None, None],
        mask_rank=128,
        mask_pool=2,
    ),
}
WEIGHTS = "model.safetensors"
CONFIG = "config.yaml"
REFERENCE = "release-reference.npz"
TRAINING_DATA = "training-data.csv"
TRAINED_ROTARY = "rotary_embed.base.freqs"
RELEASED_ROTARY = "rotary_embed.freqs"
BAND_MAP_PREFIX = "sublayer_maps.b"
AUDIO_SUFFIXES = {".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}
PARITY_SONGS = 6
PEAK_FLOOR = 1e-8
NOISE_FLOOR = 1e-300
PELL = "pell"
USED_AS = {PELL: "a cappella: the true voice of remixes"}
LABELLED = "mix: labelled by the teacher"
SOURCES = {"FMA fma_large 30 s": "FMA (fma_large, 30 s clip)"}
FMA_PREFIX = "fma-"
CREDIT_COLUMNS = [
    "source",
    "fma_track_id",
    "artist",
    "title",
    "license",
    "license_url",
    "page",
    "used_as",
]
COPIED_COLUMNS = ("artist", "title", "license", "license_url", "page")
CREDIT_ORDER = ("source", "artist", "title")


def released_state(model: dict, band_maps: list[int]) -> dict:
    out = {}
    for key, value in model.items():
        if any(key.startswith(f"layers.{i}.1.") for i in band_maps):
            continue
        name = key.replace(TRAINED_ROTARY, RELEASED_ROTARY)
        if name.startswith(BAND_MAP_PREFIX):
            i, rest = name.removeprefix(BAND_MAP_PREFIX).split(".", 1)
            name = f"layers.{i}.1.linear.{rest}"
        out[name] = value.float().clone().contiguous()
    return out


def released_config(teacher: Path, shape: dict) -> dict:
    text = (teacher / CONFIG).read_text(encoding="utf-8")
    cfg = yaml.load(text, Loader=yaml.FullLoader)  # noqa: S506
    model = {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg["model"].items()
    }
    audio = {**cfg["audio"], "chunk_size": cfg["audio"]["hop_length"] * WINDOW_FRAMES}
    training = dict(
        instruments=cfg["training"]["instruments"],
        target_instrument=cfg["training"]["target_instrument"],
    )
    return dict(
        audio=audio,
        model={**model, **shape},
        training=training,
        inference=dict(dim_t=WINDOW_FRAMES + 1),
    )


def convert(args) -> None:
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    shape = SHAPES.get(state["student"])
    if shape is None:
        message = f"no released shape for the student {state['student']}"
        raise ValueError(message)
    args.out.mkdir(parents=True, exist_ok=True)
    band_maps = [i for i, mapped in enumerate(shape["band_map"]) if mapped]
    weights = released_state(state["model"], band_maps)
    save_file(weights, args.out / WEIGHTS)
    config = released_config(args.teacher, shape)
    text = yaml.safe_dump(config, sort_keys=False, default_flow_style=None)
    (args.out / CONFIG).write_text(text, encoding="utf-8", newline="\n")
    millions = sum(v.numel() for v in weights.values()) / 1e6
    LOG.info(f"{len(weights)} tensors, {millions:.2f} M parameters -> {args.out}")


def load_release(release: Path) -> PooledMelBandRoformer:
    cfg = yaml.safe_load((release / CONFIG).read_text(encoding="utf-8"))
    model = PooledMelBandRoformer(**cfg["model"])
    model.load_state_dict(load_file(release / WEIGHTS))
    return model.cuda().eval()


def reference_path(release: Path) -> Path:
    return release.parent / REFERENCE


def snr(ref: torch.Tensor, est: torch.Tensor) -> float:
    ref, est = ref.double(), est.double()
    noise = ((ref - est) ** 2).sum().clamp(min=NOISE_FLOOR)
    return float(10 * torch.log10((ref**2).sum() / noise))


def audio_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            inside = [p for p in path.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES]
            files.extend(sorted(inside))
        else:
            files.append(path)
    return files


def peak_normalized(path: Path) -> np.ndarray:
    mix = read_audio_file(str(path), SAMPLE_RATE, 2)
    return (mix * (PEAK / max(float(np.abs(mix).max()), PEAK_FLOOR))).astype(np.float32)


def compare_window(trained, released, part: torch.Tensor, label: str):
    stft = trained.encode_stft(part)
    a, b = trained.net_forward(stft), released.net_forward(stft)
    va = trained.decode_istft(stft, a, length=part.shape[-1])
    vb = released.decode_istft(stft, b, length=part.shape[-1])
    LOG.info(
        f"{label} masks {snr(a, b):6.1f} dB, max |diff| "
        f"{(a - b).abs().max().item():.2e}; vocals {snr(va, vb):6.1f} dB"
    )
    return stft, a


def parity(args) -> None:
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trained = build_student(state["student"], args.teacher, Fits.read(args.fits))
    trained.load_state_dict(state["model"])
    trained.eval()
    released = load_release(args.release)
    saved = {}
    with torch.no_grad():
        for path in audio_files(args.audio)[: args.songs]:
            name = path.stem
            x = torch.from_numpy(peak_normalized(path)).cuda()
            for start in (0, x.shape[1] // 2):
                part = x[None, :, start : start + CHUNK]
                label = f"{name[:40]:40s} {start / SAMPLE_RATE:6.1f} s "
                stft, masks = compare_window(trained, released, part, label)
                saved[f"{name}@{start}.stft"] = stft.cpu().numpy()
                saved[f"{name}@{start}.masks"] = masks.cpu().numpy()
    np.savez(reference_path(args.release), **saved)


def check(args) -> None:
    model = load_release(args.release)
    ref = np.load(reference_path(args.release))
    with torch.no_grad():
        for key in [k for k in ref.files if k.endswith(".stft")]:
            window = key.removesuffix(".stft")
            a = torch.from_numpy(ref[f"{window}.masks"]).cuda()
            b = model.net_forward(torch.from_numpy(ref[key]).cuda())
            LOG.info(
                f"{window:50s} masks {snr(a, b):6.1f} dB, "
                f"max |diff| {(a - b).abs().max().item():.2e}"
            )


def credit(row: dict) -> dict:
    track = row["id"]
    fma_id = track.removeprefix(FMA_PREFIX)
    return dict(
        source=SOURCES.get(row["source"], row["source"]),
        fma_track_id=str(int(fma_id)) if track.startswith(FMA_PREFIX) else "",
        **{k: row[k] for k in COPIED_COLUMNS},
        used_as=USED_AS.get(row["content"], LABELLED),
    )


def credit_order(row: dict) -> tuple:
    return tuple((row[k] == "", row[k]) for k in CREDIT_ORDER)


def credits(args) -> None:
    with open(args.dataset, encoding="utf-8-sig", newline="") as f:
        rows = [
            credit(row)
            for row in csv.DictReader(f)
            if row["decision"] == "use" and row["role"] == "train"
        ]
    rows.sort(key=credit_order)
    target = args.release / TRAINING_DATA
    with open(target, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CREDIT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    sources = dict(Counter(row["source"] for row in rows).most_common())
    LOG.info(f"{len(rows)} tracks, {sources}")


def add_teacher(parser) -> None:
    parser.add_argument(
        "--teacher",
        required=True,
        type=Path,
        help="Directory with the teacher's config.yaml and model.ckpt",
    )


def add_release(parser) -> None:
    parser.add_argument(
        "--release",
        required=True,
        type=Path,
        help=f"Directory of the release: {WEIGHTS} and {CONFIG}",
    )


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "release-convert",
        parents=[common],
        help=f"a training checkpoint as {WEIGHTS} and {CONFIG} of the class",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    add_teacher(parser)
    parser.add_argument("--out", required=True, type=Path)
    parser.set_defaults(run=convert)

    parser = commands.add_parser(
        "release-parity",
        parents=[common],
        help=f"the class against the training student; writes ../{REFERENCE}",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    add_teacher(parser)
    parser.add_argument(
        "--fits",
        type=Path,
        help="JSON of the fit files the student's modifiers read",
    )
    add_release(parser)
    parser.add_argument(
        "--audio",
        required=True,
        nargs="+",
        type=Path,
        help="Audio files or directories of them; two windows of each are compared",
    )
    parser.add_argument("--songs", type=int, default=PARITY_SONGS)
    parser.set_defaults(run=parity)

    parser = commands.add_parser(
        "release-check",
        parents=[common],
        help=f"the class on the windows of ../{REFERENCE} that release-parity saved",
    )
    add_release(parser)
    parser.set_defaults(run=check)

    parser = commands.add_parser(
        "release-credits",
        parents=[common],
        help=f"{TRAINING_DATA}: every track the student learned from",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="dataset.csv of the variant the student trained on",
    )
    add_release(parser)
    parser.set_defaults(run=credits)
