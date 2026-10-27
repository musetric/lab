from pathlib import Path

import torch

from musetric_toolkit.common.logger import send_message

CHECKPOINT_FILENAME = "beat_this-final0.ckpt"
DOWNLOAD_LABEL = "beat_this checkpoint"


def _checkpoint_cache_path() -> Path:
    hub_dir = Path(torch.hub.get_dir())
    return hub_dir / "checkpoints" / CHECKPOINT_FILENAME


def _emit_download(
    status: str,
    downloaded: int = 0,
    total: int | None = None,
) -> None:
    payload: dict[str, object] = {
        "type": "download",
        "label": DOWNLOAD_LABEL,
        "file": CHECKPOINT_FILENAME,
        "downloaded": downloaded,
        "status": status,
    }
    if total is not None:
        payload["total"] = total
    send_message(payload)


def run_beat_this(audio_path: str):
    from beat_this.inference import File2Beats  # noqa: PLC0415

    cache_path = _checkpoint_cache_path()
    was_cached = cache_path.exists()
    if was_cached:
        size = cache_path.stat().st_size
        _emit_download(status="cached", downloaded=size, total=size)
    else:
        _emit_download(status="processing")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    file2beats = File2Beats(checkpoint_path="final0", device=device, dbn=True)

    if not was_cached and cache_path.exists():
        size = cache_path.stat().st_size
        _emit_download(status="done", downloaded=size, total=size)

    return file2beats(audio_path)
