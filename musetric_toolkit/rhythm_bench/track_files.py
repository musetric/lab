from pathlib import Path

AUDIO_SUFFIXES = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}


def track_files(audio_path: str, file_name: str | None) -> dict[str, Path]:
    path = Path(audio_path)
    if path.is_file():
        return {path.stem: path}
    if file_name:
        return {file.parent.name: file for file in sorted(path.glob(f"*/{file_name}"))}
    return {
        file.stem: file
        for file in sorted(path.iterdir())
        if file.is_file() and file.suffix.lower() in AUDIO_SUFFIXES
    }
