import copy
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import soundfile
from PIL import Image

from musetric_toolkit.distill.excluded import read_excluded
from musetric_toolkit.listening.levels import QUIET_POWER
from musetric_toolkit.listening.round_files import (
    ASK_PRESENT,
    CLIPS,
    KEY,
    MANIFEST,
    read_json,
)
from musetric_toolkit.listening.round_labels import GROUPS
from musetric_toolkit.listening.round_origins import check_origins
from musetric_toolkit.listening.round_verdict import Verdict

LOG = logging.getLogger(__name__)
ALLOWED_LICENCE = re.compile(
    r"^(attribution|creative commons attribution|cc0|public domain)", re.IGNORECASE
)
BANNED_LICENCE = re.compile(
    r"noncommercial|non-commercial|noderiv|no deriv|sharealike|share alike",
    re.IGNORECASE,
)
SELECTION_WORDS = (
    "student",
    "teacher",
    "lead_gap",
    "residue",
    "vocals_gap",
    "private:",
    "detector_",
    "repeat_",
    '"sign"',
)
CREDIT_FIELDS = {"artist", "title", "license", "license_url", "page", "source"}
DURATION_TOLERANCE = 0.01
GREY_PICTURE = "L"


def served_text(manifest: dict) -> str:
    served = copy.deepcopy(manifest)
    served["round"] = ""
    for clip in served["clips"]:
        clip.pop("credit", None)
    return json.dumps(served, ensure_ascii=False).lower()


def key_words(key: dict) -> set[str]:
    words = set()
    for entry in key.values():
        words |= {entry["a"], entry["b"], entry["sign"], entry["source"]}
    return {word.lower() for word in words}


def check_blind(manifest: dict, key: dict, verdict: Verdict) -> None:
    served = served_text(manifest)
    for word in SELECTION_WORDS:
        if word in served:
            verdict.fail(f"the manifest gives away {word}")
    for word in sorted(key_words(key)):
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", served):
            verdict.fail(f"the manifest gives away {word}")
    for clip in manifest["clips"]:
        extra = set(clip.get("credit", {})) - CREDIT_FIELDS
        if extra:
            verdict.fail(f"{clip['id']}: the credit carries {sorted(extra)}")


def read_clip_stems(folder: Path, clip: str, manifest: dict, verdict: Verdict) -> dict:
    data = {}
    for stem in (item["id"] for item in manifest["stems"]):
        audio, rate = soundfile.read(folder / f"{stem}.wav", dtype="float64")
        seconds = len(audio) / rate
        if (
            rate != manifest["rate"]
            or abs(seconds - manifest["duration"]) > DURATION_TOLERANCE
        ):
            verdict.fail(f"{clip} {stem}: rate {rate}, {seconds} s")
        if not np.isfinite(audio).all():
            verdict.fail(f"{clip} {stem}: not finite")
        data[stem] = audio
        image = Image.open(folder / f"{stem}.png")
        if image.mode != GREY_PICTURE or image.size[0] != manifest["spectrum"]["width"]:
            verdict.fail(f"{clip} {stem}: picture {image.mode} {image.size}")
    return data


def check_sums(clip: str, data: dict, args, verdict: Verdict) -> list[float]:
    sums = []
    for side in "ab":
        restored = sum(data[f"{group}_{side}"] for group in GROUPS)
        noise = ((restored - data["mix"]) ** 2).sum() + QUIET_POWER
        snr = float(10 * np.log10((data["mix"] ** 2).sum() / noise))
        sums.append(snr)
        if snr < args.min_sum_snr_db:
            verdict.fail(f"{clip} {side}: stems sum to the mix at {snr:.1f} dB")
    return sums


def check_credit(clip: dict, key: dict, verdict: Verdict) -> None:
    name = clip["id"]
    if not clip.get("train", True):
        if clip.get("credit"):
            verdict.fail(f"{name}: a private clip carries a credit")
        return
    credit = clip.get("credit") or {}
    licence = credit.get("license") or ""
    if not ALLOWED_LICENCE.match(licence) or BANNED_LICENCE.search(licence):
        verdict.fail(f"{name}: licence {licence!r}")
    if not credit.get("artist") or not credit.get("title"):
        verdict.fail(f"{name}: no credit")
    if key[name]["source"].startswith("private"):
        verdict.fail(f"{name}: private source marked for training")


def check_clips(manifest: dict, key: dict, args, verdict: Verdict) -> None:
    sums, levels = [], []
    for clip in manifest["clips"]:
        name = clip["id"]
        folder = args.round / CLIPS / name
        data = read_clip_stems(folder, name, manifest, verdict)
        sums += check_sums(name, data, args, verdict)
        mix_db = 10 * np.log10((data["mix"] ** 2).mean() + QUIET_POWER)
        lead_db = 10 * np.log10((data["lead_a"] ** 2).mean() + QUIET_POWER)
        levels.append(float(lead_db - mix_db))
        if mix_db < args.min_mix_db:
            verdict.fail(f"{name}: quiet mix {mix_db:.1f} dB")
        check_credit(clip, key, verdict)
    LOG.info(
        "stems sum to the mix, SNR dB: min %.1f, median %.1f",
        min(sums),
        float(np.median(sums)),
    )
    LOG.info(
        "lead to mix dB: min %.1f, median %.1f, max %.1f",
        min(levels),
        float(np.median(levels)),
        max(levels),
    )


def log_summary(manifest: dict, key: dict) -> None:
    clips = manifest["clips"]
    entries = [key[clip["id"]] for clip in clips]
    signs, sides = {}, {}
    for entry in entries:
        signs[entry["sign"]] = signs.get(entry["sign"], 0) + 1
        sides[entry["a"]] = sides.get(entry["a"], 0) + 1
    untrained = sum(not clip.get("train", True) for clip in clips)
    LOG.info("clips %d, not for training %d", len(clips), untrained)
    LOG.info("signs %s", signs)
    LOG.info("system on side A %s", sides)
    licences = {clip["credit"]["license"] for clip in clips if clip.get("credit")}
    LOG.info("licences %s", sorted(licences))


def check_form(manifest: dict, verdict: Verdict) -> None:
    stems = [item["id"] for item in manifest["stems"]]
    if manifest["default_stem"] not in stems:
        verdict.fail(f"the default stem {manifest['default_stem']} is not a stem")
    if manifest.get("ask") == ASK_PRESENT and manifest["default_stem"] != "mix":
        verdict.fail("a round of ticks must start from the mix")


def check_round_command(args) -> None:
    manifest = read_json(args.round / MANIFEST)
    key = read_json(args.round / KEY)
    verdict = Verdict()
    check_form(manifest, verdict)
    check_blind(manifest, key, verdict)
    check_clips(manifest, key, args, verdict)
    log_summary(manifest, key)
    check_origins(manifest, key, args, read_excluded(args.excluded), verdict)
    LOG.info("%s %d", "FAILED" if verdict.failures else "all passed", verdict.failures)
    sys.exit(1 if verdict.failures else 0)
