from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from musetric_toolkit.common.logger import send_message

CHUNK_SIZE = 1024 * 1024


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        scheme = parsed.scheme or "<missing>"
        raise ValueError(f"Unsupported URL scheme: {scheme}")


def _download_file(url: str, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _validate_url(url)

    total: int | None = None
    try:
        with urlopen(url) as response, destination.open("wb") as target:  # noqa: S310
            total_value = int(response.headers.get("Content-Length", "0") or 0)
            total = total_value if total_value > 0 else None
            downloaded = 0
            send_message(
                {
                    "type": "download",
                    "label": label,
                    "file": destination.name,
                    "downloaded": 0,
                    "total": total,
                    "status": "processing",
                }
            )
            for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                target.write(chunk)
                downloaded += len(chunk)
                status = "done" if downloaded >= total else "processing"
                send_message(
                    {
                        "type": "download",
                        "label": label,
                        "file": destination.name,
                        "downloaded": downloaded,
                        "total": total,
                        "status": status,
                    }
                )
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Failed to download {label}: {error}") from error


def ensure_model_file(url: str, path: Path, label: str) -> None:
    if path.exists():
        size = path.stat().st_size
        send_message(
            {
                "type": "download",
                "label": label,
                "file": path.name,
                "downloaded": size,
                "total": size,
                "status": "cached",
            }
        )
        return

    _download_file(url, path, label)
