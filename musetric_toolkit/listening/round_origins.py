import logging
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile

from musetric_toolkit.listening.levels import QUIET_POWER, energy, level_db
from musetric_toolkit.listening.round_files import (
    CLIPS,
    KEY,
    artists_of,
    corpus_tracks,
    earlier_sources,
    read_json,
    rounds_by_name,
    split_earlier_repeat,
)
from musetric_toolkit.listening.round_labels import sides_of
from musetric_toolkit.listening.round_verdict import Verdict

LOG = logging.getLogger(__name__)
SHOWN_DRIFTS = 12


def read_stem(folder: Path, clip: str, stem: str) -> np.ndarray:
    return soundfile.read(folder / CLIPS / clip / f"{stem}.wav", dtype="float64")[0]


def live_levels(folder: Path, clip: str, entry: dict) -> dict[str, float]:
    mix = read_stem(folder, clip, "mix")
    return {
        system: level_db(
            read_stem(folder, clip, f"lead_{side}")
            + read_stem(folder, clip, f"backing_{side}"),
            mix,
        )
        for system, side in sides_of(entry).items()
    }


def voice_roles(voices: list[Path]) -> dict[str, set[str]]:
    roles = {}
    for manifest in voices:
        for entry in read_json(manifest):
            roles.setdefault(entry["id"], set()).add(entry["role"])
    return roles


def check_track_origins(key: dict, args, verdict: Verdict) -> None:
    tracks = corpus_tracks(args.corpus)
    used = earlier_sources(args.earlier_round)
    used_artists = artists_of(used, tracks)
    roles = voice_roles(args.voices)
    seen_artists = Counter()
    for clip, entry in key.items():
        repeat = entry.get("repeat_of", "")
        if not entry["train"] or split_earlier_repeat(repeat):
            continue
        artist = tracks[entry["source"]]["artist"]
        if not repeat:
            seen_artists[artist] += 1
        if entry["source"] in used or artist in used_artists:
            verdict.fail(f"{clip}: comes from an earlier round")
        if args.voices and roles.get(entry["source"]) != {"train"}:
            verdict.fail(f"{clip}: role {roles.get(entry['source'])}")
    if any(count > 1 for count in seen_artists.values()):
        verdict.fail("an artist appears twice")


def check_own_repeats(key: dict, order: list[str], args, verdict: Verdict) -> None:
    for clip, entry in key.items():
        repeat = entry.get("repeat_of", "")
        if not repeat or split_earlier_repeat(repeat):
            continue
        twins = [
            other
            for other, item in key.items()
            if item["source"] == entry["source"]
            and item["start_s"] == entry["start_s"]
            and other != clip
        ]
        if len(twins) != 1:
            verdict.fail(f"{clip}: {len(twins)} originals")
            continue
        distance = abs(order.index(clip) - order.index(twins[0]))
        if distance < args.min_repeat_distance:
            verdict.fail(f"{clip}: only {distance} away from {twins[0]}")
        same = np.array_equal(
            read_stem(args.round, clip, "mix"), read_stem(args.round, twins[0], "mix")
        )
        if not same:
            verdict.fail(f"{clip}: the mix differs from {twins[0]}")


def check_earlier_repeats(key: dict, args, excluded: set[str], verdict: Verdict):
    rounds = rounds_by_name(args.earlier_round)
    for clip, entry in key.items():
        found = split_earlier_repeat(entry.get("repeat_of", ""))
        if not found:
            continue
        name, old_clip = found
        if name not in rounds:
            verdict.fail(f"{clip}: repeats {name}, which --earlier-round does not name")
            continue
        old_entry = read_json(rounds[name] / KEY)[old_clip]
        if not old_entry["train"] or old_entry["source"] in excluded:
            verdict.fail(f"{clip}: repeats a closed or excluded clip")
        old = read_stem(rounds[name], old_clip, "mix")
        new = read_stem(args.round, clip, "mix")
        limiter_gain = float((old * new).sum()) / (energy(old) + QUIET_POWER)
        noise = energy(new - limiter_gain * old) + QUIET_POWER
        snr = 10 * np.log10(energy(new) / noise)
        if snr < args.min_repeat_snr_db:
            old_name = f"{name}:{old_clip}"
            verdict.fail(f"{clip}: the mix differs from {old_name}, {snr:.1f} dB")


def compare_selection_levels(entry, clip, live, args, verdict: Verdict) -> list:
    scores = entry.get("scores") or {}
    drifts = []
    for system, level in live.items():
        chosen = scores.get(system)
        if chosen is None:
            continue
        if system not in args.exact_levels:
            drifts.append((round(level - chosen, 2), clip, system))
        elif abs(level - chosen) > args.exact_level_db:
            verdict.fail(f"{clip}: {system} {chosen} in selection, {level:.1f} in clip")
    return drifts


def log_drifts(drifts: dict[str, list], args) -> None:
    LOG.info("selection levels from stand-ins against the live clips, dB")
    for sign, rows in drifts.items():
        values = np.array([row[0] for row in rows])
        over = int((np.abs(values) > args.drift_db).sum())
        LOG.info(
            "  %-16s n=%2d median %6.2f min %7.2f max %7.2f, over %s dB: %d",
            sign,
            len(rows),
            np.median(values),
            values.min(),
            values.max(),
            args.drift_db,
            over,
        )
        for row in sorted(rows, key=lambda row: -abs(row[0]))[:SHOWN_DRIFTS]:
            if abs(row[0]) > args.drift_db:
                LOG.info("    %s", row)


def log_live_levels(key: dict, live: dict[str, dict], args) -> None:
    LOG.info("groups against the live levels")
    for sign in sorted({entry["sign"] for entry in key.values()}):
        clips = [
            clip
            for clip, entry in key.items()
            if entry["sign"] == sign and not entry.get("repeat_of")
        ]
        if not clips:
            continue
        for system in sorted(live[clips[0]]):
            levels = np.array([live[clip][system] for clip in clips])
            LOG.info(
                "  %-16s n=%2d %-10s over %s: %2d, under %s: %2d",
                sign,
                len(clips),
                system,
                args.gate_db,
                int((levels >= args.gate_db).sum()),
                args.quiet_db,
                int((levels < args.quiet_db).sum()),
            )


def check_levels(key: dict, args, verdict: Verdict) -> None:
    drifts, live = {}, {}
    for clip, entry in key.items():
        live[clip] = live_levels(args.round, clip, entry)
        found = compare_selection_levels(entry, clip, live[clip], args, verdict)
        if found:
            drifts.setdefault(entry["sign"], []).extend(found)
    log_drifts(drifts, args)
    log_live_levels(key, live, args)


def log_excluded(key: dict, excluded: set[str]) -> None:
    clips = [clip for clip, entry in key.items() if entry["source"] in excluded]
    if clips:
        LOG.warning("clips of tracks excluded since: %s", clips)


def check_origins(
    manifest: dict, key: dict, args, excluded: set[str], verdict: Verdict
) -> None:
    order = [clip["id"] for clip in manifest["clips"]]
    if args.corpus:
        check_track_origins(key, args, verdict)
    else:
        LOG.warning("no --corpus: artists and earlier rounds are not checked")
    check_own_repeats(key, order, args, verdict)
    check_earlier_repeats(key, args, excluded, verdict)
    check_levels(key, args, verdict)
    log_excluded(key, excluded)
