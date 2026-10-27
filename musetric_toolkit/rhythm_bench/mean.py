import json
import logging
from pathlib import Path

from musetric_toolkit.rhythm_bench.postprocess import (
    load_logits,
    write_postprocessed,
)
from musetric_toolkit.rhythm_bench.variant import tracks_with, variant_path


def main(args) -> None:
    results_dir = Path(args.results_dir)
    first, second = args.logits
    tracks = tracks_with(results_dir, [f"{first}-minimal", f"{second}-minimal"])
    for track in tracks:
        a = load_logits(results_dir, track, first)
        b = load_logits(results_dir, track, second)
        count = min(len(a[0]), len(b[0]))
        beat = (a[0][:count] + b[0][:count]) / 2
        downbeat = (a[1][:count] + b[1][:count]) / 2
        duration = json.loads(
            variant_path(results_dir, track, f"{first}-minimal").read_text()
        )["duration"]
        write_postprocessed(results_dir, track, args.label, (beat, downbeat), duration)
        logging.info("%s %s: %d frames", track, args.label, count)
