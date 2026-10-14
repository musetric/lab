import json
from pathlib import Path

from musetric_toolkit.transcribe_audio.hallucination_filter import (
    filter_sign_off_segments,
)
from musetric_toolkit.transcribe_audio.lyric_splitter import (
    split_segments_by_lyrics,
)
from musetric_toolkit.transcribe_audio.response_builder import (
    build_payload_segments,
)
from musetric_toolkit.transcribe_audio.whisperx_runner import (
    transcribe_with_whisperx,
)


def main(args) -> None:
    segments, _language = transcribe_with_whisperx(
        args.audio_path,
        args.backing_path,
        args.log_level,
    )
    lyric_segments = filter_sign_off_segments(split_segments_by_lyrics(segments))
    payload_segments = build_payload_segments(lyric_segments)
    output_path = Path(args.result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as result_file:
        json.dump(payload_segments, result_file, ensure_ascii=False, indent=2)
