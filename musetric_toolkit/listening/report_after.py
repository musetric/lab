from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile

from musetric_toolkit.listening.levels import SILENCE
from musetric_toolkit.listening.report_numbers import (
    share,
    system_letters,
    ticks,
)
from musetric_toolkit.listening.report_ticks import TicksRound
from musetric_toolkit.listening.round_files import (
    CLIPS,
    FEATURES,
    LABELS_LOG,
    read_json,
    read_jsonl,
)
from musetric_toolkit.listening.round_labels import NO_VOICE, UNSURE, sides_of

REPEATS = "repeats"
SHOWN_WORD_CLIPS = 10
SHOWN_TEXT = 60
SHOWN_COMMENT = 70
SHOWN_GENRES = 10
MANY_WORDS = 2


def seconds_per_clip(folder: Path) -> dict:
    events = read_jsonl(folder / LABELS_LOG)
    last = None
    spent = {}
    for event in events:
        at = datetime.fromisoformat(event["at"])
        if event["data"].get("done") and event["clip"] not in spent:
            if last is not None:
                spent[event["clip"]] = (at - last).total_seconds()
            last = at
    first = datetime.fromisoformat(events[0]["at"])
    final = datetime.fromisoformat(events[-1]["at"])
    return dict(spent=spent, total=(final - first).total_seconds())


def timing_section(tick: TicksRound, timing: dict, duration: float) -> list[str]:
    spent_by_clip = timing["spent"]
    lines = [f"  labelling took {timing['total'] / 60:.1f} min"]
    repeats = [r for r in tick.rows if r["repeat_of"]]
    for group in (*tick.groups(), REPEATS):
        chosen = [
            r
            for r in (repeats if group == REPEATS else tick.in_group(group))
            if r["clip"] in spent_by_clip
        ]
        if not chosen:
            continue
        spent = np.array([spent_by_clip[r["clip"]] for r in chosen])
        lines.append(
            f"  {group:16} n={len(chosen):2} seconds median {np.median(spent):5.1f} "
            f"min {spent.min():5.1f} max {spent.max():6.1f}; "
            f"under one play of {duration:.0f} s: {int((spent < duration).sum())}"
        )
    by_kind = {}
    for r in tick.rows:
        if r["clip"] in spent_by_clip:
            by_kind.setdefault(r["kind"], []).append(spent_by_clip[r["clip"]])
    medians = {k: round(float(np.median(v)), 1) for k, v in by_kind.items()}
    lines.append(f"  seconds by answer: {medians}")
    return lines


def quick_clips(tick: TicksRound, timing: dict, duration: float) -> list[str]:
    return [
        r["clip"]
        for r in tick.new
        if r["kind"] == NO_VOICE
        and r["clip"] in timing["spent"]
        and timing["spent"][r["clip"]] < duration
    ]


def outside_section(tick: TicksRound) -> list[str]:
    new = tick.new
    with_flag = [r for r in new if r["instrumental_flag"] is not None]
    flags = Counter((r["kind"] == NO_VOICE, r["instrumental_flag"]) for r in with_flag)
    lines = [f"  FMA instrumental flag on the track against the answer: {flags}"]
    nest = [r for r in new if r["echonest"]]
    lines.append(f"  echonest coverage {len(nest)} of {len(new)}")
    answers = (
        ("no voice", [r for r in nest if r["kind"] == NO_VOICE]),
        ("voice", [r for r in nest if r["kind"] not in (NO_VOICE, UNSURE)]),
    )
    for answer, chosen in answers:
        if chosen:
            values = np.array([r["echonest"]["instrumentalness"] for r in chosen])
            lines.append(
                f"    {answer}: n={len(chosen)} instrumentalness median "
                f"{np.median(values):.2f} min {values.min():.2f} max {values.max():.2f}"
            )
    return lines


def words_section(tick: TicksRound) -> list[str]:
    silent = [r for r in tick.new if r["kind"] == NO_VOICE]
    lines = []
    for answer, chosen in (("no voice", silent), ("voice", tick.voiced)):
        if chosen:
            words = [r["whisper"]["mix"]["words"] for r in chosen]
            many = sum(count > MANY_WORDS for count in words)
            lines.append(
                f"  whisper words on the mix, {answer}: median {np.median(words)}, "
                f"over {MANY_WORDS} words {many} of {len(chosen)}"
            )
    heard = ("mix", *tick.names)
    lines.append("  no voice with most whisper words on any stem:")
    ranked = sorted(silent, key=lambda r: -max(r["whisper"][k]["words"] for k in heard))
    for r in ranked[:SHOWN_WORD_CLIPS]:
        texts = [r["whisper"][k]["text"][:SHOWN_TEXT] for k in heard]
        lines.append(f"    {r['clip']} {r['sign']} {texts}")
    return lines


def earlier_rounds_section(tick: TicksRound) -> list[str]:
    args = tick.args
    this_round = tick.in_group(args.random_group)
    if not args.earlier_round or not this_round:
        return []
    drawn = []
    for folder in args.earlier_round:
        drawn += [
            r
            for r in read_json(folder / FEATURES)
            if r["sign"] == args.random_group and r["train"] and not r["excluded"]
        ]
    earlier = sum(
        r["present"]["lead"] is False and r["present"]["backing"] is False
        for r in drawn
    )
    now = sum(r["kind"] == NO_VOICE for r in this_round)
    names = " and ".join(folder.name for folder in args.earlier_round)
    together = share(earlier + now, len(drawn) + len(this_round))
    return [f"  {args.random_group}, {names} and this round: no voice {together}"]


def supply_section(tick: TicksRound) -> list[str]:
    args = tick.args
    if not args.windows or tick.rule is None:
        return []
    best = {}
    for row in read_jsonl(args.windows):
        for window in row["windows"]:
            if window["kind"] == "vocals":
                group = "quiet" if window["teacher"] < args.quiet_db else "loud"
                value = tick.rule(
                    window["dominant"], window["teacher"], window["words"]
                )
                found = best.setdefault(group, {})
                found[row["id"]] = max(value, found.get(row["id"], -np.inf))
    lines = ["  supply of candidates in the searched pool"]
    for group, found in best.items():
        values = np.array(list(found.values()))
        counts = ", ".join(
            f">= {limit:g}: {int((values >= limit).sum())}"
            for limit in args.supply_thresholds
        )
        lines.append(f"    {group}: clips {len(values)}; score {counts}")
    return lines


def genres_section(tick: TicksRound) -> list[str]:
    genres = Counter(r["genre"] for r in tick.new if r["kind"] == NO_VOICE)
    drawn = Counter(r["kind"] for r in tick.in_group(tick.args.random_group))
    return [
        f"  no voice clips by genre: {genres.most_common(SHOWN_GENRES)}",
        f"  random ticks: {dict(drawn)}",
    ]


def doubt_section(tick: TicksRound, quick: list[str]) -> list[str]:
    hedged, early = set(tick.args.hedged), set(quick)
    lines = ["", "  sensitivity of the no voice share to doubtful answers"]
    groups = [g for g in tick.groups() if g != tick.args.random_group]
    for group in groups:
        chosen = tick.in_group(group)
        none = [r for r in chosen if r["kind"] == NO_VOICE]
        doubted = sum(r["clip"] in hedged for r in none)
        fast = sum(r["clip"] in early for r in none)
        lines.append(
            f"    {group:16} as answered {share(len(none), len(chosen))}; "
            f"hedged as errors {share(len(none) - doubted, len(chosen))}; "
            "hedged and early left out "
            f"{share(len(none) - doubted - fast, len(chosen) - doubted - fast)}"
        )
    clean = [
        r
        for r in tick.new
        if r["kind"] == NO_VOICE and r["clip"] not in hedged and r["clip"] not in early
    ]
    by_group = dict(Counter(r["sign"] for r in clean))
    lines.append(
        f"    no voice without hedged and early: {len(clean)}; by group {by_group}"
    )
    return lines


def one_silent_section(tick: TicksRound) -> list[str]:
    groups = tick.args.one_silent_groups
    chosen = [r for group in groups for r in tick.in_group(group)]
    if not chosen:
        return []
    wins = sum(r["kind"] == NO_VOICE for r in chosen)
    return [
        f"    exactly one model silent, the silent model is right: "
        f"{share(wins, len(chosen))}"
    ]


def voiced_levels_section(tick: TicksRound) -> list[str]:
    drawn = tick.args.random_group
    first, second = tick.names
    a, b = system_letters(tick.names)
    parts = (
        (drawn, [r for r in tick.voiced if r["sign"] == drawn]),
        ("other groups", [r for r in tick.voiced if r["sign"] != drawn]),
    )
    lines = ["", "  lead and backing on voiced clips, levels of both systems"]
    for name, chosen in parts:
        lines.append(f"  {name}: {len(chosen)} clips")
        for r in chosen:
            s, t = r["systems"][first], r["systems"][second]
            lines.append(
                f"    {r['clip']} {ticks(r['present'])} "
                f"lead {a} {s['lead']:6.1f} {b} {t['lead']:6.1f} "
                f"backing {a} {s['backing']:6.1f} {b} {t['backing']:6.1f}  |  "
                f"{r['comment'][:SHOWN_COMMENT]}"
            )
    return lines


def absent_section(tick: TicksRound) -> list[str]:
    args = tick.args
    first, second = tick.names
    drawn = [r for r in tick.usable if r["sign"] == args.random_group]
    if not drawn:
        return []
    both = "both wrong"
    absent = [r for r in drawn if tick.outcome(r, args.quiet_db) != "nothing absent"]
    split = [r for r in absent if r["kind"] != NO_VOICE]
    wrong_all = sum(tick.outcome(r, args.quiet_db) == both for r in absent)
    wrong_split = sum(tick.outcome(r, args.quiet_db) == both for r in split)
    gaps = np.array(
        [
            abs(r["systems"][first]["vocals"] - r["systems"][second]["vocals"])
            for r in drawn
        ]
    )
    return [
        f"  {args.random_group}, an absent stem exists: {len(absent)}; both wrong at "
        f"{args.quiet_db:g} dB {share(wrong_all, len(absent))}; "
        f"split only {share(wrong_split, len(split))}",
        f"  {args.random_group}, vocals, {first} against {second}: "
        f"median gap {np.median(gaps):.2f} dB, max {gaps.max():.2f} dB",
    ]


def lost_section(tick: TicksRound) -> list[str]:
    args = tick.args
    first, second = tick.names
    lines = []
    for stem in ("lead", "backing"):
        lost = [
            (
                r["clip"],
                round(r["systems"][first][stem], 1),
                round(r["systems"][second][stem], 1),
            )
            for r in tick.voiced
            if r["present"][stem] and r["systems"][first][stem] < args.quiet_db
        ]
        lines.append(
            f"  {stem} ticked, {first} {stem} under {args.quiet_db:g} dB: {lost}"
        )
    return lines


def shown_clips_section(tick: TicksRound) -> list[str]:
    if not tick.args.show_clips:
        return []
    rows = {r["clip"]: r for r in tick.rows}
    lines = ["", "  systems and sources behind the named clips"]
    for clip in tick.args.show_clips:
        entry = tick.key[clip]
        lines.append(
            f"    {clip}: A is {entry['a']}, B is {entry['b']}, source "
            f"{entry['source']}: {rows[clip]['comment']!r}"
        )
    return lines


def envelope_lines(folder: Path, clip: str, entry: dict, seconds: float) -> list[str]:
    path = folder / CLIPS / clip
    mix, rate = soundfile.read(path / "mix.wav", dtype="float64")
    size = int(seconds * rate)
    lines = [f"    {clip} dB to the mix per {seconds:.0f} s"]
    for system, side in sides_of(entry).items():
        vocals = (
            soundfile.read(path / f"lead_{side}.wav", dtype="float64")[0]
            + soundfile.read(path / f"backing_{side}.wav", dtype="float64")[0]
        )
        parts = [
            10
            * np.log10(
                ((vocals[i : i + size] ** 2).sum() + SILENCE)
                / ((mix[i : i + size] ** 2).sum() + SILENCE)
            )
            for i in range(0, len(mix) - size + 1, size)
        ]
        lines.append(f"      {system:8} {' '.join(f'{p:6.1f}' for p in parts)}")
    return lines


def envelopes_section(tick: TicksRound) -> list[str]:
    args = tick.args
    if not args.envelope:
        return []
    lines = ["  envelopes"]
    for clip in args.envelope:
        lines += envelope_lines(args.round, clip, tick.key[clip], args.envelope_seconds)
    return lines


def after_sections(tick: TicksRound, duration: float) -> list[str]:
    timing = seconds_per_clip(tick.args.round)
    quick = quick_clips(tick, timing, duration)
    silent = sum(r["kind"] == NO_VOICE for r in tick.new)
    return [
        "",
        "AFTER THE ANSWERS, not in the plan",
        *timing_section(tick, timing, duration),
        f"  no voice answered in under one play: {len(quick)} of {silent}: {quick}",
        *outside_section(tick),
        *words_section(tick),
        *earlier_rounds_section(tick),
        *supply_section(tick),
        *genres_section(tick),
        *doubt_section(tick, quick),
        *one_silent_section(tick),
        *voiced_levels_section(tick),
        *absent_section(tick),
        *lost_section(tick),
        *shown_clips_section(tick),
        *envelopes_section(tick),
    ]
