import argparse
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from musetric_toolkit.listening.frozen_rule import Rule
from musetric_toolkit.listening.report_numbers import (
    absent_outcome,
    share,
    system_letters,
    ticks,
)
from musetric_toolkit.listening.round_files import (
    LABELS,
    read_json,
    rounds_by_name,
    split_earlier_repeat,
)
from musetric_toolkit.listening.round_labels import (
    NO_VOICE,
    UNSURE,
    VOICED_KINDS,
    kind_of,
    present_of,
)


@dataclass
class TicksRound:
    rows: list[dict]
    key: dict
    names: tuple[str, str]
    rule: Rule | None
    args: argparse.Namespace

    @property
    def new(self) -> list[dict]:
        return [row for row in self.rows if not row["repeat_of"]]

    @property
    def usable(self) -> list[dict]:
        return [row for row in self.new if row["kind"] != UNSURE]

    @property
    def voiced(self) -> list[dict]:
        return [row for row in self.new if row["kind"] in VOICED_KINDS]

    @property
    def primary(self) -> str:
        return self.names[0]

    @property
    def reference(self) -> str:
        return self.names[1]

    def in_group(self, group: str) -> list[dict]:
        return [row for row in self.new if row["sign"] == group]

    def outcome(self, row: dict, limit: float) -> str:
        return absent_outcome(row, limit, self.names)

    def groups(self) -> list[str]:
        signs = {row["sign"] for row in self.new}
        named = [group for group in self.args.groups if group in signs]
        return named + sorted(signs - set(named))


def prepare_rows(tick: TicksRound) -> None:
    for row in tick.rows:
        row["kind"] = kind_of(row)
        row["selected_score"] = (row["scores"] or {}).get("score")
        row["score"] = None
        if tick.rule:
            row["score"] = tick.rule(
                row["dsp"][tick.reference]["dominant"],
                row["systems"][tick.reference]["vocals"],
                row["whisper"][tick.primary]["words"],
            )


def clip_table(tick: TicksRound) -> list[str]:
    p, t = system_letters(tick.names)
    lines = [
        "clip  group            ticks kind             "
        f"{p} voc  {t} voc  {p} lead {t} lead {p} back {t} back  {p} inst {t} inst  "
        f"score sel    words {p}/{t}/mix"
    ]
    for r in sorted(tick.rows, key=lambda r: (r["sign"], r["clip"])):
        s, o = r["systems"][tick.primary], r["systems"][tick.reference]
        w = r["whisper"]
        sel = "" if r["selected_score"] is None else f"{r['selected_score']:5.2f}"
        score = " " * 6 if r["score"] is None else f"{r['score']:6.2f}"
        lines.append(
            f"{r['clip']:5} {r['sign']:16} {ticks(r['present']):5} {r['kind']:16} "
            f"{s['vocals']:6.1f} {o['vocals']:6.1f} {s['lead']:6.1f} {o['lead']:6.1f} "
            f"{s['backing']:6.1f} {o['backing']:6.1f} "
            f"{s['instrumental']:6.1f} {o['instrumental']:6.1f} {score} {sel:5}  "
            f"{w[tick.primary]['words']}/{w[tick.reference]['words']}/"
            f"{w['mix']['words']}"
        )
    return lines


def no_voice_section(tick: TicksRound) -> list[str]:
    lines = [
        "",
        "1. no voice by group, new clips, exact 95 % interval. "
        "Strict: unsure counts against; second: unsure left out",
    ]
    for group in tick.groups():
        chosen = tick.in_group(group)
        counts = Counter(r["kind"] for r in chosen)
        none, unsure = counts[NO_VOICE], counts[UNSURE]
        lines.append(f"  {group:16} n={len(chosen):2} {dict(counts)}")
        lines.append(
            f"  {'':16} no voice {share(none, len(chosen))}; "
            f"without unsure {share(none, len(chosen) - unsure)}"
        )
    lines.append(f"  all new clips: {dict(Counter(r['kind'] for r in tick.new))}")
    return lines


def detector_section(tick: TicksRound) -> list[str]:
    found = [g for g in tick.args.detector_groups if tick.in_group(g)]
    if not found:
        return []
    return [
        "",
        f"2. detector precision: see {' and '.join(found)} in 1; "
        "a lower bound of 0.95 needs at least 72 blind clips with no error",
    ]


def systems_section(tick: TicksRound) -> list[str]:
    args = tick.args
    lines = [
        "",
        "3. systems against the song answer, live levels, new clips without unsure",
    ]
    for limit in args.limits:
        usable = tick.usable
        tally = dict(Counter(tick.outcome(r, limit) for r in usable))
        lines.append(f"  limit {limit} dB, all: {tally}")
        for group in tick.groups():
            part = Counter(tick.outcome(r, limit) for r in usable if r["sign"] == group)
            lines.append(f"    {group:16} {dict(part)}")
    silent = [r for r in tick.new if r["kind"] == NO_VOICE]
    tally = dict(Counter(tick.outcome(r, args.quiet_db) for r in silent))
    lines.append(f"  no voice only, limit {args.quiet_db:g} dB: {tally}")
    no_instrumental = [
        (
            r["clip"],
            round(r["systems"][tick.primary]["instrumental"], 1),
            round(r["systems"][tick.reference]["instrumental"], 1),
        )
        for r in tick.new
        if r["present"]["instrumental"] is False
    ]
    lines.append(f"  no instrumental ticked: {no_instrumental}")
    return lines


def reverse_section(tick: TicksRound) -> list[str]:
    args = tick.args
    first, second = tick.names
    quiet = f"{args.quiet_db:g} dB"
    lines = [
        "",
        f"4. reverse error: voice in the song, live {first} vocals under {quiet}",
    ]
    voiced = tick.voiced
    lost = [r for r in voiced if r["systems"][first]["vocals"] < args.quiet_db]
    lines.append(f"  voiced new clips {len(voiced)}, lost by the {first} {len(lost)}")
    for r in lost:
        lines.append(
            f"    {r['clip']} {r['sign']} {r['kind']} "
            f"{first} {r['systems'][first]['vocals']:.1f} "
            f"{second} {r['systems'][second]['vocals']:.1f} comment {r['comment']!r}"
        )
    under = [r for r in tick.new if r["systems"][first]["vocals"] < args.quiet_db]
    tally = dict(Counter(r["kind"] for r in under))
    lines.append(f"  live {first} under {quiet}, all new clips: {tally}")
    for group in args.reverse_groups:
        chosen = tick.in_group(group)
        if chosen:
            voiced_here = sum(r["kind"] not in (NO_VOICE, UNSURE) for r in chosen)
            lines.append(f"  {group}: voice found {share(voiced_here, len(chosen))}")
    alone = [r for r in under if r["systems"][second]["vocals"] >= args.voice_db]
    tally = dict(Counter(r["kind"] for r in alone))
    lines.append(
        f"  {first} under {quiet}, {second} at or over {args.voice_db} dB: {tally}"
    )
    return lines


def other_answer(tick: TicksRound, row: dict, rows_by_clip: dict) -> tuple:
    earlier = split_earlier_repeat(row["repeat_of"])
    if earlier:
        name, clip = earlier
        rounds = rounds_by_name(tick.args.earlier_round)
        if name not in rounds:
            message = f"{row['clip']} repeats {name}: name it with --earlier-round"
            raise ValueError(message)
        labels = read_json(rounds[name] / LABELS)
        return row["repeat_of"], present_of(labels[clip]), False
    entry = tick.key[row["clip"]]
    twin = next(
        clip
        for clip, item in tick.key.items()
        if item["source"] == entry["source"]
        and item["start_s"] == entry["start_s"]
        and clip != row["clip"]
    )
    other = rows_by_clip[twin]
    return twin, other["present"], other["unsure"]


def repeats_section(tick: TicksRound) -> list[str]:
    repeats = [r for r in tick.rows if r["repeat_of"]]
    if not repeats:
        return []
    lines = ["", f"5. stability on the {len(repeats)} blind repeats"]
    rows_by_clip = {r["clip"]: r for r in tick.rows}
    same_voice, same_ticks = 0, 0
    for r in repeats:
        other_name, other, other_unsure = other_answer(tick, r, rows_by_clip)
        now = r["present"]
        voice_now = bool(now["lead"] or now["backing"])
        voice_then = bool(other["lead"] or other["backing"])
        same_v = voice_now == voice_then and not r["unsure"] and not other_unsure
        same_t = same_v and all(
            bool(now[g]) == bool(other[g]) for g in ("lead", "backing")
        )
        same_voice += same_v
        same_ticks += same_t
        lines.append(
            f"  {r['clip']} {r['sign']:24} now {ticks(now)} {other_name} "
            f"{ticks(other)} voice same {same_v} lead and backing same {same_t}"
        )
    pairs = len(repeats)
    lines.append(
        f"  voice yes or no the same: {share(same_voice, pairs)}; "
        f"lead and backing the same: {share(same_ticks, pairs)}"
    )
    return lines


def auc_with_interval(rows: list[dict], rng, bootstraps: int) -> str:
    y = np.array([r["kind"] == NO_VOICE for r in rows], dtype=int)
    x = np.array([r["score"] for r in rows])
    if y.min() == y.max():
        return f"n={len(rows)} one class only"
    aucs = []
    for _ in range(bootstraps):
        pick = rng.integers(0, len(rows), len(rows))
        if y[pick].min() != y[pick].max():
            aucs.append(roc_auc_score(y[pick], x[pick]))
    low, high = np.quantile(aucs, 0.025), np.quantile(aucs, 0.975)
    return (
        f"n={len(rows)} no voice {int(y.sum())} AUC {roc_auc_score(y, x):.3f} "
        f"[{low:.3f}, {high:.3f}]"
    )


def rule_section(tick: TicksRound) -> list[str]:
    if tick.rule is None:
        return []
    args = tick.args
    rng = np.random.default_rng(args.seed)
    usable = tick.usable
    searched = (args.random_group, *args.detector_groups)
    pool = [r for r in usable if r["sign"] in searched]
    lines = ["", "6. frozen rule on new data, live features"]
    auc = auc_with_interval(pool, rng, args.bootstraps)
    lines.append(f"  {args.random_group} and detectors: {auc}")
    for group in (args.random_group, *reversed(args.detector_groups)):
        chosen = [r for r in usable if r["sign"] == group]
        auc = auc_with_interval(chosen, rng, args.bootstraps)
        lines.append(f"  {group} alone: {auc}")
    for threshold in args.score_thresholds:
        flagged = [r for r in pool if r["score"] >= threshold]
        wins = sum(r["kind"] == NO_VOICE for r in flagged)
        lines.append(f"  score >= {threshold}: {share(wins, len(flagged))}")
    return lines


def plan_sections(tick: TicksRound) -> list[str]:
    return [
        *clip_table(tick),
        *no_voice_section(tick),
        *detector_section(tick),
        *systems_section(tick),
        *reverse_section(tick),
        *repeats_section(tick),
        *rule_section(tick),
    ]
