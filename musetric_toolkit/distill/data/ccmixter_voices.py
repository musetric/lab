import json
import logging
import re
from pathlib import Path

import numpy as np

from musetric_toolkit.distill.data.ccmixter_fetch import CRAWL_FILES, LICENSES, get
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.ffmpeg.write import write_audio_file

LOG = logging.getLogger(__name__)

SPEC = Path(__file__).with_name("ccmixter_voices.json")
SAMPLE_RATE = 44100
CHECK_RATE = 11025
MAX_LAG_SECONDS = 2.0
MIN_SHIFT_MS = 5.0
MIN_PEAK = 0.2
HEADROOM = 0.95
VOICES = ("lead", "backing")
PARTS = (*VOICES, "accompaniment")
SOURCES = "src"
TRACKS = "tracks.json"
LICENSE_FILE = "license.md"
TITLE = "# Open lead and backing set from ccMixter"
NOTE = (
    "Separately uploaded lead and backing vocal parts of ccMixter a cappellas, "
    "each under its uploader's license (CC BY or CC0). acappella/: lead plus "
    "backing; mix/: the same over the upload's own instrument stems. Evaluation "
    "only: its uploaders are kept out of training."
)


def crawl_entries(crawl: Path) -> dict[int, dict]:
    entries = {}
    for name in CRAWL_FILES:
        for item in json.loads((crawl / name).read_text(encoding="utf-8")):
            entries[int(item["id"])] = item
    return entries


def base_name(entry: dict) -> str:
    shortest = min((f["name"] for f in entry["files"]), key=len)
    return re.sub(r"_\d+$", "", shortest.rsplit(".", 1)[0])


def file_name(entry: dict, base: str, suffix: str) -> str:
    names = {f["name"] for f in entry["files"]}
    for name in (base + suffix, base.rstrip("_") + suffix):
        if name in names:
            return name
    message = f"{entry['id']}: no file {base + suffix}"
    raise KeyError(message)


def fetch(out: Path, entry: dict, name: str) -> Path:
    target = out / SOURCES / str(entry["id"]) / name
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = next(f["url"] for f in entry["files"] if f["name"] == name)
    part = target.with_suffix(".part")
    part.write_bytes(get(url, referer=entry["url"]))
    part.replace(target)
    return target


def summed(paths: list[Path], rate: int, channels: int) -> np.ndarray | None:
    parts = [read_audio_file(str(p), rate, channels).astype(np.float32) for p in paths]
    if not parts:
        return None
    out = np.zeros((channels, max(p.shape[1] for p in parts)), np.float32)
    for part in parts:
        out[:, : part.shape[1]] += part
    return out


def lag_of(ref: np.ndarray, part: np.ndarray) -> tuple[int, float]:
    a, b = ref[0].astype(np.float64), part[0].astype(np.float64)
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    corr = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    max_lag = int(MAX_LAG_SECONDS * CHECK_RATE)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    window = np.concatenate([corr[: max_lag + 1], corr[-max_lag:]])
    norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    best = int(np.argmax(np.abs(window)))
    return int(lags[best]), float(window[best] / norm)


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12)


def shifted(a: np.ndarray, samples: int) -> np.ndarray:
    if samples > 0:
        return np.pad(a, ((0, 0), (samples, 0)))[:, : a.shape[1]]
    if samples < 0:
        return np.pad(a[:, -samples:], ((0, 0), (0, -samples)))
    return a


def write_set(folder: Path, stems: dict[str, np.ndarray]) -> None:
    peak = max(float(np.abs(s).max()) for s in stems.values())
    gain = HEADROOM / peak if peak > HEADROOM else 1.0
    folder.mkdir(parents=True, exist_ok=True)
    for name, audio in stems.items():
        write_audio_file(str(folder / f"{name}.wav"), (audio * gain).T, SAMPLE_RATE)


def alignment(out: Path, item: dict, entry: dict, base: str, paths: dict) -> dict:
    if not item["reference"]:
        return {}
    ref_path = fetch(out, entry, file_name(entry, base, item["reference"]))
    ref = read_audio_file(str(ref_path), CHECK_RATE, 1)
    checked = PARTS if item["kind"] == "mix" else VOICES
    check = {}
    for part in (p for p in checked if paths[p]):
        lag, peak = lag_of(ref, summed(paths[part], CHECK_RATE, 1))
        check[part] = dict(
            lag_ms=round(lag / CHECK_RATE * 1000, 1), peak=round(peak, 3)
        )
    return check


def build_track(out: Path, item: dict, entry: dict) -> dict:
    base = base_name(entry)
    names = {p: [file_name(entry, base, s) for s in item[p]] for p in PARTS}
    paths = {p: [fetch(out, entry, n) for n in ns] for p, ns in names.items()}
    check = alignment(out, item, entry, base, paths)
    groups = {p: summed(ps, SAMPLE_RATE, 2) for p, ps in paths.items()}
    n = max(a.shape[1] for a in groups.values() if a is not None)
    stems = {
        p: (
            np.zeros((2, n), np.float32)
            if a is None
            else np.pad(a, ((0, 0), (0, n - a.shape[1])))
        )
        for p, a in groups.items()
    }
    if set(VOICES) <= check.keys():
        relative = check["lead"]["lag_ms"] - check["backing"]["lag_ms"]
        if abs(relative) >= MIN_SHIFT_MS and check["backing"]["peak"] >= MIN_PEAK:
            samples = round(relative / 1000 * SAMPLE_RATE)
            stems["backing"] = shifted(stems["backing"], samples)
            check["backing_shifted_ms"] = relative
    vocals = stems["lead"] + stems["backing"]
    track = f"ccm-{item['id']}"
    voices = dict(vocals=vocals, lead=stems["lead"], backing=stems["backing"])
    write_set(out / "acappella" / track, dict(mixture=vocals, **voices))
    if item["accompaniment"]:
        mixture = vocals + stems["accompaniment"]
        sets = dict(mixture=mixture, **voices, instrumental=stems["accompaniment"])
        write_set(out / "mix" / track, sets)
    licence = str(entry["licUrl"]).replace("https://", "http://")
    return dict(
        track=track,
        upload=item["id"],
        title=entry["name"],
        uploader=entry["user"],
        artist=entry.get("real") or entry["user"],
        license=LICENSES[licence],
        licenseUrl=licence,
        page=entry["url"],
        files=names,
        seconds=round(n / SAMPLE_RATE, 1),
        backing_db=round(20 * np.log10(rms(stems["backing"]) / rms(vocals)), 1),
        accompaniment=bool(item["accompaniment"]),
        check=check,
    )


def write_license(out: Path, rows: list[dict]) -> None:
    header = "| Artist | Title | License | Source |\n| --- | --- | --- | --- |\n"
    table = "".join(
        f"| {r['artist']} | {r['title']} | [{r['license']}]({r['licenseUrl']}) "
        f"| [page]({r['page']}) |\n"
        for r in rows
    )
    text = f"{TITLE}\n\n{NOTE}\n\n{header}{table}"
    (out / LICENSE_FILE).write_text(text, encoding="utf-8", newline="\n")


def build(args) -> None:
    entries = crawl_entries(args.crawl)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    rows = []
    for item in spec:
        rows.append(build_track(args.out, item, entries[item["id"]]))
        LOG.info(json.dumps({k: rows[-1][k] for k in ("track", "seconds", "check")}))
    text = json.dumps(rows, indent=1, ensure_ascii=False)
    (args.out / TRACKS).write_text(text, encoding="utf-8", newline="\n")
    write_license(args.out, rows)
    mixes = sum(r["accompaniment"] for r in rows)
    LOG.info(f"{len(rows)} a cappellas, {mixes} also as mixes -> {args.out}")


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "data-ccmixter-voices",
        parents=[common],
        help="the open test set of true lead and backing from ccMixter a cappellas",
    )
    parser.add_argument(
        "--crawl",
        required=True,
        type=Path,
        help=f"Directory of a crawl: {', '.join(CRAWL_FILES)}",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=SPEC,
        help="JSON of the uploads and which of their files are lead, backing, "
        "accompaniment and the reference mix",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help=f"Output: acappella/, mix/, {TRACKS}, {LICENSE_FILE}; {SOURCES}/ caches "
        "the downloads",
    )
    parser.set_defaults(run=build)
