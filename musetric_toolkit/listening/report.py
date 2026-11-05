import argparse
import sys

from musetric_toolkit.distill.excluded import read_excluded
from musetric_toolkit.listening.frozen_rule import frozen_rule
from musetric_toolkit.listening.report_after import after_sections
from musetric_toolkit.listening.report_ticks import (
    TicksRound,
    plan_sections,
    prepare_rows,
)
from musetric_toolkit.listening.report_votes import votes_report
from musetric_toolkit.listening.round_files import (
    ASK_PRESENT,
    FEATURES,
    KEY,
    MANIFEST,
    SYSTEMS_IN_ROUND,
    read_json,
)
from musetric_toolkit.listening.round_labels import load_round


def system_names(rows: list[dict], args: argparse.Namespace) -> tuple[str, str]:
    found = sorted({name for row in rows for name in row["systems"]})
    if len(found) != SYSTEMS_IN_ROUND:
        raise ValueError(f"a round compares two systems, this one has {found}")
    if args.primary is None:
        return found[0], found[1]
    if args.primary not in found:
        raise ValueError(f"{args.primary} is not among the systems {found}")
    return args.primary, next(name for name in found if name != args.primary)


def ticks_report(args: argparse.Namespace, duration: float) -> list[str]:
    rows = read_json(args.round / FEATURES)
    rule = frozen_rule(args.rule_features) if args.rule_features else None
    tick = TicksRound(
        rows=rows,
        key=read_json(args.round / KEY),
        names=system_names(rows, args),
        rule=rule,
        args=args,
    )
    prepare_rows(tick)
    return [*plan_sections(tick), *after_sections(tick, duration)]


def round_report(args: argparse.Namespace, excluded: frozenset[str]) -> str:
    manifest = read_json(args.round / MANIFEST)
    if manifest.get("ask") == ASK_PRESENT:
        lines = ticks_report(args, manifest["duration"])
    else:
        rows = load_round(args.round, excluded)
        lines = votes_report(rows, system_names(rows, args), args)
    return "\n".join(lines) + "\n"


def report_command(args: argparse.Namespace) -> None:
    text = round_report(args, read_excluded(args.excluded))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    sys.__stdout__.buffer.write(text.encode("utf-8"))
    sys.__stdout__.flush()
