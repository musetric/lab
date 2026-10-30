from pathlib import Path

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.parity_audio import (
    chords_chain,
    key_chain,
    rhythm_chain,
    transcribe_chain,
    vocals_chain,
)
from musetric_toolkit.parity_audio.case_writer import CaseWriter
from musetric_toolkit.separate_audio.system_info import ensure_ffmpeg

CHAINS = {
    "vocals": vocals_chain.run,
    "rhythm": rhythm_chain.run,
    "key": key_chain.run,
    "chords": chords_chain.run,
    "transcribe": transcribe_chain.run,
}


def main(args) -> None:
    ensure_ffmpeg()
    send_message({"type": "progress", "progress": 0.0})
    writer = CaseWriter(Path(args.case_path), args.step)
    CHAINS[args.step](args, writer)
    writer.finish()
    send_message({"type": "progress", "progress": 1.0})
