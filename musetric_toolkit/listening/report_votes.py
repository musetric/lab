import argparse
from collections import Counter

import numpy as np
from scipy.stats import fisher_exact

from musetric_toolkit.listening.report_numbers import (
    absent_outcome,
    exact,
    system_letters,
)
from musetric_toolkit.listening.round_labels import (
    BACKING_ONLY,
    GROUPS,
    LEAD_AND_BACKING,
    LEAD_ONLY,
    NO_VOICE,
    kind_of,
)

ENERGY_ROWS = (
    ("no voice: vocals", (NO_VOICE,), "vocals"),
    ("voiced: vocals", (LEAD_ONLY, BACKING_ONLY, LEAD_AND_BACKING), "vocals"),
    ("no lead, backing: lead", (BACKING_ONLY,), "lead"),
    ("no lead, backing: backing", (BACKING_ONLY,), "backing"),
    ("no backing, lead: backing", (LEAD_ONLY,), "backing"),
    ("no backing, lead: lead", (LEAD_ONLY,), "lead"),
    ("lead and backing: lead", (LEAD_AND_BACKING,), "lead"),
    ("lead and backing: backing", (LEAD_AND_BACKING,), "backing"),
)


def yes_no(value) -> str:
    if value is None:
        return "?"
    return "+" if value else "-"


def clip_table(rows: list[dict], names: tuple[str, str]) -> list[str]:
    a, b = system_letters(names)
    lines = [
        "clip  sign        genre                present(L B I)  votes(L B I)        "
        f"{a} voc  {b} voc  {a} lead {b} lead {a} back {b} back  {a} inst {b} inst"
    ]
    for r in rows:
        s, t = r["systems"][names[0]], r["systems"][names[1]]
        present = " ".join(yes_no(r["present"][g]) for g in GROUPS)
        votes = " ".join((r["votes"][g] or "-")[:3] for g in GROUPS)
        marks = ("  EXCL" if r["excluded"] else "") + (
            "  PRIVATE" if not r["train"] else ""
        )
        lines.append(
            f"{r['clip']:5} {r['sign']:11} {str(r['genre'])[:20]:20} {present:15} "
            f"{votes:19} {s['vocals']:6.1f} {t['vocals']:6.1f} {s['lead']:6.1f} "
            f"{t['lead']:6.1f} {s['backing']:6.1f} {t['backing']:6.1f} "
            f"{s['instrumental']:6.1f} {t['instrumental']:6.1f}{marks}"
        )
    return lines


def song_content(usable: list[dict]) -> list[str]:
    lines = ["", "song content, usable clips"]
    lines.append(str(dict(Counter(r["kind"] for r in usable))))
    for sign in sorted({r["sign"] for r in usable}):
        counts = dict(Counter(r["kind"] for r in usable if r["sign"] == sign))
        lines.append(f"  {sign} {counts}")
    return lines


def clip_bootstrap(usable: list[dict], names: tuple[str, str], args) -> list[str]:
    rng = np.random.default_rng(args.seed)
    shares = []
    for _ in range(args.bootstraps):
        pick = rng.integers(0, len(usable), len(usable))
        sample = [usable[i]["votes"][g] for i in pick for g in GROUPS]
        wins, losses = sample.count(names[0]), sample.count(names[1])
        shares.append(wins / max(1, wins + losses))
    low, high = np.quantile(shares, 0.025), np.quantile(shares, 0.975)
    return [f"  clip bootstrap of {names[0]} share: [{low:.2f}, {high:.2f}]"]


def random_lines(usable: list[dict], names: tuple[str, str], args) -> list[str]:
    drawn = [r for r in usable if r["sign"] == args.random_group]
    if not drawn:
        return []
    chosen = [r for r in usable if r["sign"] != args.random_group]
    first, second = names
    votes = [r["votes"]["lead"] for r in drawn]
    silent_chosen = sum(r["kind"] == NO_VOICE for r in chosen)
    silent_drawn = sum(r["kind"] == NO_VOICE for r in drawn)
    table = [
        [silent_chosen, len(chosen) - silent_chosen],
        [silent_drawn, len(drawn) - silent_drawn],
    ]
    low, high, _ = exact(silent_drawn, len(drawn))
    return [
        f"  lead on random clips only: {first} {votes.count(first)} "
        f"{second} {votes.count(second)}",
        f"  no voice, picked by disagreement {table[0]}, random {table[1]}, "
        f"Fisher p={fisher_exact(table)[1]:.3f}",
        f"  no voice share on random clips [{low:.2f}, {high:.2f}]",
    ]


def blind_votes(usable: list[dict], names: tuple[str, str], args) -> list[str]:
    first, second = names
    lines = ["", f"blind preference, usable clips: {first} wins of decisive"]
    for group in GROUPS:
        counts = Counter(r["votes"][group] for r in usable)
        wins, losses = counts.get(first, 0), counts.get(second, 0)
        low, high, p = exact(wins, wins + losses)
        lines.append(
            f"  {group:13} {first} {wins} {second} {losses} "
            f"same {counts.get('same', 0)} unknown {counts.get('unknown', 0)}  "
            f"share [{low:.2f}, {high:.2f}] p={p:.3f}"
        )
    votes = [r["votes"][g] for r in usable for g in GROUPS]
    lines.append(
        f"  all stems     {first} {votes.count(first)} {second} {votes.count(second)}"
    )
    lines += clip_bootstrap(usable, names, args)
    sides = [r["sides"][g] for r in usable for g in GROUPS]
    a, b = sides.count("a"), sides.count("b")
    low, high, p = exact(a, a + b)
    lines.append(
        f"  control, position: A {a} B {b} share of A [{low:.2f}, {high:.2f}] p={p:.3f}"
    )
    return lines + random_lines(usable, names, args)


def absent_stems(usable: list[dict], names: tuple[str, str], args) -> list[str]:
    lines = [
        "",
        "absent stems against the song answer: a system is wrong when a stem the "
        "owner calls absent is over the limit",
    ]
    for limit in args.limits:
        tally = dict(Counter(absent_outcome(r, limit, names) for r in usable))
        lines.append(f"  limit {limit} dB: {tally}")
    return lines


def foreign_ticks(usable: list[dict], names: tuple[str, str]) -> list[str]:
    lines = ["", "foreign ticks, usable clips, by system"]
    for group in GROUPS:
        for system in names:
            heard = Counter(v for r in usable for v in r["foreign"][group][system])
            lines.append(f"  in {group:13} {system:8} {dict(heard)}")
    return lines


def energy_table(usable: list[dict], names: tuple[str, str]) -> list[str]:
    lines = ["", "energy against the song answer, usable clips, median dB to mix"]
    for name, kinds, column in ENERGY_ROWS:
        chosen = [r for r in usable if r["kind"] in kinds]
        if not chosen:
            lines.append(f"  {name:28} none")
            continue
        parts = []
        for system in names:
            v = np.array([r["systems"][system][column] for r in chosen])
            parts.append(
                f"{system} {np.median(v):6.1f} [{v.min():6.1f},{v.max():6.1f}]"
            )
        lines.append(f"  {name:28} n={len(chosen):2} {parts[0]}  {parts[1]}")
    return lines


def votes_report(
    rows: list[dict], names: tuple[str, str], args: argparse.Namespace
) -> list[str]:
    usable = [r for r in rows if r["train"] and not r["excluded"]]
    for row in usable:
        row["kind"] = kind_of(row)
    lines = clip_table(rows, names)
    lines += ["", f"usable clips {len(usable)}"]
    lines += song_content(usable)
    lines += blind_votes(usable, names, args)
    lines += absent_stems(usable, names, args)
    lines += foreign_ticks(usable, names)
    lines += energy_table(usable, names)
    return lines
