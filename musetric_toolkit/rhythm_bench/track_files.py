from pathlib import Path

AUDIO_SUFFIXES = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}


def track_files(audio_path: str, file_name: str | None) -> dict[str, Path]:
    """Map track names to audio files.

    A file is one track named by its stem. In a directory, every audio file
    directly inside is a track named by its stem; with `file_name`, every
    subdirectory holding a file of that name is a track named by the
    subdirectory, as in a corpus of `<song>/instrumental.flac` stems.
    """
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
