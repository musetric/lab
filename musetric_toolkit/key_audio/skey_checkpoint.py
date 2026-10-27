from pathlib import Path

from musetric_toolkit.common.model_files import ensure_model_file

# S-KEY checkpoint source. The S-KEY inference subset is vendored under
# musetric_toolkit/key_audio/skey. See thirdPartyNotices.md.

CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/deezer/skey/"
    "918b83d273568d5041569bb8068843d19a335726/skey/models/skey.pt"
)
CHECKPOINT_FILENAME = "skey.pt"
DOWNLOAD_LABEL = "S-KEY checkpoint"


def ensure_checkpoint(models_path: str) -> Path:
    path = Path(models_path) / "skey" / CHECKPOINT_FILENAME
    ensure_model_file(CHECKPOINT_URL, path, DOWNLOAD_LABEL)
    return path
