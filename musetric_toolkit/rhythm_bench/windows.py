import json
import sys
from pathlib import Path

from musetric_toolkit.rhythm_bench.beat_counts import off_beat_intervals
from musetric_toolkit.rhythm_bench.variant import read_variant, tracks_with

# Where two variants click differently: the track is cut into windows of 15 s,
# and a window is kept when the seconds of spurious and gap intervals starting
# in it differ by 2 s or more between the variants. The kept windows are the
# places to listen to with `listen`.

WINDOW_SECONDS = 15.0
MIN_DIFFERENCE = 2.0


def off_beat_seconds(intervals: list[tuple[float, float]], start: float) -> float:
    return sum(
        length for time, length in intervals if start <= time < start + WINDOW_SECONDS
    )


def clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def main(args) -> None:
    results_dir = Path(args.results_dir)
    first, second = args.variants
    regions = []
    lines = []
    for track in tracks_with(results_dir, [first, second]):
        a = read_variant(results_dir, track, first).beats
        b = read_variant(results_dir, track, second).beats
        off_a = off_beat_intervals(a)
        off_b = off_beat_intervals(b)
        end = max(a[-1] if len(a) else 0.0, b[-1] if len(b) else 0.0)
        found = []
        start = 0.0
        while start < end:
            seconds_a = off_beat_seconds(off_a, start)
            seconds_b = off_beat_seconds(off_b, start)
            if abs(seconds_a - seconds_b) >= MIN_DIFFERENCE:
                regions.append(
                    {"track": track, "start": start, "end": start + WINDOW_SECONDS}
                )
                found.append(f"{clock(start)} {seconds_a:.1f} -> {seconds_b:.1f}")
            start += WINDOW_SECONDS
        if found:
            lines.append(f"{track}: " + ", ".join(found))
    out = Path(args.out) if args.out else results_dir / f"windows-{first}-{second}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps({"variants": [first, second], "regions": regions}, indent=2) + "\n"
    )
    report = "\n".join(lines) + "\n" if lines else "no window differs\n"
    sys.__stdout__.buffer.write(report.encode("utf-8"))
    sys.__stdout__.flush()
