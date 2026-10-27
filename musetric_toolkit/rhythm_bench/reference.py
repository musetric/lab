import logging
from pathlib import Path

import torch

from musetric_toolkit.rhythm_bench.postprocess import (
    POSTPROCESSINGS,
    save_logits,
    write_postprocessed,
)
from musetric_toolkit.rhythm_bench.track_files import track_files
from musetric_toolkit.rhythm_bench.variant import variant_path

# Beat This! "final0" in torch, the reference the rhythm step of the app
# matches (musetric/musetric#1031). The track is cut into windows of the given
# number of frames with a border of 6, and where windows overlap the first
# window's frames are kept, as the author's inference does: 1500 frames is the
# author's window and the app's, 513 the window the app ran before #1029.
BORDER_SIZE = 6


def main(args) -> None:
    from beat_this.inference import (  # noqa: PLC0415
        Audio2Frames,
        split_predict_aggregate,
    )
    from beat_this.preprocessing import load_audio  # noqa: PLC0415

    results_dir = Path(args.results_dir)
    logits = f"{args.label}-w{args.window}"
    files = track_files(args.audio_path, args.file_name)
    frames = Audio2Frames(checkpoint_path="final0", device=args.device)
    for track, path in files.items():
        done = all(
            variant_path(results_dir, track, f"{logits}-{postprocessing}").is_file()
            for postprocessing in POSTPROCESSINGS
        )
        if done and not args.overwrite:
            logging.info("%s %s: done before, skipped", track, logits)
            continue
        signal, rate = load_audio(str(path))
        spect = frames.signal2spect(signal, rate)
        with torch.inference_mode():
            prediction = split_predict_aggregate(
                spect=spect,
                chunk_size=args.window,
                border_size=BORDER_SIZE,
                overlap_mode="keep_first",
                model=frames.model,
            )
        beat = prediction["beat"].float().cpu().numpy()
        downbeat = prediction["downbeat"].float().cpu().numpy()
        save_logits(results_dir, track, logits, (beat, downbeat))
        write_postprocessed(
            results_dir, track, logits, (beat, downbeat), len(signal) / rate
        )
        logging.info("%s %s: %d frames", track, logits, len(beat))
