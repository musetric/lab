import json
from pathlib import Path

import torch

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.separate_audio.system_info import (
    ensure_ffmpeg,
    print_acceleration_info,
    setup_torch_optimization,
)
from musetric_toolkit.transcribe_audio.download_progress import intercept_hf_downloads
from musetric_toolkit.transcribe_audio.transcribe_pipeline import transcribe_voice
from musetric_toolkit.transcribe_audio.voice import read_voice
from musetric_toolkit.transcribe_audio.whisper_decoder import WhisperDecoder
from musetric_toolkit.transcribe_audio.whisper_runtime import WhisperRuntime


def main(args) -> None:
    ensure_ffmpeg()
    print_acceleration_info()
    setup_torch_optimization()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with intercept_hf_downloads("Whisper model"):
        decoder = WhisperDecoder(device)
    voice = read_voice(args.audio_path, args.backing_path)
    payload_segments = transcribe_voice(voice, WhisperRuntime(decoder))
    output_path = Path(args.result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as result_file:
        json.dump(payload_segments, result_file, ensure_ascii=False, indent=2)
    send_message({"type": "progress", "progress": 1.0})
