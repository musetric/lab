from pathlib import Path

from musetric_toolkit.common.model_files import ensure_model_file

CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/ptnghia-j/ChordMini/main/"
    "checkpoints/2e1d_model_best.pth"
)
CHECKPOINT_FILENAME = "2e1d_model_best.pth"
DOWNLOAD_LABEL = "ChordMini checkpoint"


def ensure_checkpoint(models_path: str) -> Path:
    path = Path(models_path) / "chordmini" / CHECKPOINT_FILENAME
    ensure_model_file(CHECKPOINT_URL, path, DOWNLOAD_LABEL)
    return path
