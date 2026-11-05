import json
from pathlib import Path

STAND = Path(__file__).with_name("stand")
MESSAGES = STAND / "messages.ru.json"
CLIPS = "clips"
FEATURES = "features.json"
KEY = "key.json"
LABELS = "labels.json"
LABELS_LOG = "labels.log.jsonl"
MANIFEST = "manifest.json"
SELECTION = "selection.json"
CORPUS_TRACKS = "fma.json"
ASK_PRESENT = "present"
SYSTEMS_IN_ROUND = 2


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=1, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def corpus_tracks(corpus: Path) -> dict[str, dict]:
    return {track["id"]: track for track in read_json(corpus / CORPUS_TRACKS)}


def earlier_sources(rounds: list[Path]) -> set[str]:
    keys = [read_json(folder / KEY) for folder in rounds]
    return {entry["source"] for key in keys for entry in key.values()}


def artists_of(sources: set[str], tracks: dict[str, dict]) -> set[str]:
    return {tracks[source]["artist"] for source in sources if source in tracks}


def rounds_by_name(rounds: list[Path]) -> dict[str, Path]:
    return {folder.name: folder for folder in rounds}


def split_earlier_repeat(repeat: str) -> tuple[str, str] | None:
    name, separator, clip = repeat.partition(":")
    return (name, clip) if separator else None
