import json
import sys
from dataclasses import asdict
from pathlib import Path

from musetric_toolkit.rhythm_bench.beat_counts import beat_f_measure, count_beats
from musetric_toolkit.rhythm_bench.variant import read_variant, tracks_with


def format_report(
    tracks: list[str],
    variants: list[str],
    per_track: dict[str, dict[str, dict]],
    totals: dict[str, dict],
    reference: str | None,
) -> str:
    agreement = f" | F vs {reference}" if reference else ""
    lines = [
        f"# Beat counts on {len(tracks)} track{'' if len(tracks) == 1 else 's'}",
        "",
        f"| Variant | Spurious beats | Gap seconds | Tempo jumps{agreement} |",
        "| --- | ---: | ---: | ---:" + (" | ---:" if reference else "") + " |",
    ]
    for variant in variants:
        total = totals[variant]
        row = (
            f"| {variant} | {total['spurious']} | {total['gap_seconds']:.0f}"
            f" | {total['tempo_jumps']}"
        )
        if reference:
            row += f" | {total['f_measure']:.3f}"
        lines.append(row + " |")
    lines += [
        "",
        "Per track: spurious beats / gap seconds / tempo jumps.",
        "",
        "| Track | " + " | ".join(variants) + " |",
        "| --- |" + " ---: |" * len(variants),
    ]
    for track in tracks:
        cells = [
            f"{counts['spurious']} / {counts['gap_seconds']:.0f}"
            f" / {counts['tempo_jumps']}"
            for counts in (per_track[track][variant] for variant in variants)
        ]
        lines.append(f"| {track} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(args) -> None:
    results_dir = Path(args.results_dir)
    needed = [*args.variants, *([args.reference] if args.reference else [])]
    tracks = tracks_with(results_dir, needed)
    if not tracks:
        raise SystemExit(f"no track in {results_dir} has all of {needed}")
    per_track: dict[str, dict[str, dict]] = {}
    for track in tracks:
        reference = (
            read_variant(results_dir, track, args.reference).beats
            if args.reference
            else None
        )
        per_track[track] = {}
        for variant in args.variants:
            beats = read_variant(results_dir, track, variant).beats
            counts = asdict(count_beats(beats))
            if reference is not None:
                counts["f_measure"] = beat_f_measure(reference, beats)
            per_track[track][variant] = counts
    totals = {}
    for variant in args.variants:
        rows = [per_track[track][variant] for track in tracks]
        totals[variant] = {
            "spurious": sum(row["spurious"] for row in rows),
            "gap_seconds": sum(row["gap_seconds"] for row in rows),
            "tempo_jumps": sum(row["tempo_jumps"] for row in rows),
        }
        if args.reference:
            totals[variant]["f_measure"] = sum(row["f_measure"] for row in rows) / len(
                rows
            )
    report = format_report(tracks, args.variants, per_track, totals, args.reference)
    out = Path(args.out) if args.out else results_dir / "score"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(report, encoding="utf-8")
    result = {"reference": args.reference, "totals": totals, "tracks": per_track}
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    sys.__stdout__.buffer.write(report.encode("utf-8"))
    sys.__stdout__.flush()
