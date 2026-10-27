import csv
import json
import logging
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from musetric_toolkit.distill.corpus import load
from musetric_toolkit.distill.data.duplicates import within
from musetric_toolkit.distill.data.fma_fetch import raw_tracks
from musetric_toolkit.distill.teacher import SAMPLE_RATE

LOG = logging.getLogger(__name__)

COLUMNS = [
    "id",
    "decision",
    "reason",
    "owner_verdict",
    "role",
    "content",
    "teacher_risk",
    "teacher_risk_reason",
    "listen_set",
    "source",
    "artist",
    "title",
    "album",
    "genre",
    "language",
    "license",
    "license_url",
    "page",
    "seconds",
    "bitrate_kbps",
    "bandwidth_khz",
    "stereo",
    "side_db",
    "rms_db",
    "clip_share",
    "silence_share",
    "vocal_share_db",
    "vocal_active_share",
    "ambiguous_share",
    "vocal_low_share",
    "vocal_high_share",
    "meta_instrumental",
    "duplicate_of",
    "note",
]
FEATURES = (
    "seconds",
    "bandwidth_khz",
    "side_db",
    "rms_db",
    "clip_share",
    "silence_share",
    "vocal_share_db",
    "vocal_active_share",
    "ambiguous_share",
    "vocal_low_share",
    "vocal_high_share",
)
FMA_SOURCE = "FMA fma_large 30 s"
PELL_SOURCE = "ccMixter a cappella"
OPEN_GENRE = "open corpus"
PELL_GENRE = "a cappella"
SPOKEN = "Spoken"
HISTORIC = "Old-Time / Historic"
REJECTED_NOTE = "instrumental (no lead vocal)"
VOICED = ("vocal", "vocal-only")

INSTRUMENTAL_DB = -40
WHOLE_MIX_DB = -1.5
WEAK_DB = -20
WEAK_ACTIVE = 0.1
UNSURE_SHARE = 0.25
BASS_SHARE = 0.1
SILENT_SHARE = 0.5
MIN_BANDWIDTH_KHZ = 8
FULL_BANDWIDTH_KHZ = 15.5
CLIP_SHARE = 1e-3
HISTORIC_SIDE_DB = -30
MONO_SIDE_DB = -60
PELL_MIN_DB = -3.0
HISTORIC_YEAR = re.compile(r"\((18\d\d|19[0-3]\d)\b")
LATER_RECORDING = re.compile("piano roll|version", re.IGNORECASE)

AUTHOR_INSTRUMENTAL = (
    "marked instrumental by the author, but the teacher hears a voice: "
    "an instrument likely went into the vocals"
)
WHOLE_MIX = (
    "the teacher put almost the whole mix into the vocals: an a cappella (right) "
    "or instruments as voice (a bad target)"
)
UNSURE = "the teacher is unsure"
BASS = "bass in the vocals"
RISK_HEARD = "teacher risk heuristic, the class was heard and is fine"
NOT_LABELLED = "not labelled"
SKIP_LINE = re.compile(r"\bskip (\S+): (.*)")
SKIP_TEXT = 60
VERDICT_DATE = 10

IDEAL_BANDWIDTH_KHZ = 19
IDEAL_CLIP_SHARE = 1e-4
IDEAL_SILENCE = 0.05
IDEAL_VOCAL_DB = (-12, -3)
IDEAL_ACTIVE = 0.4
IDEAL_LOW_SHARE = 0.01
IDEAL_INSTRUMENTALS = 2
WORST_AUTHOR, WORST_UNSURE, WORST_BASS = 3, 3, 2
DOUBTFUL_AUDIO = 4
DOUBTFUL_GENRES = ("(none)", "Experimental", "International", "Classical")
BAND_OR_CLIPPING = re.compile("band|clipping")
EXCERPT_SECONDS = 30
LISTEN_GROUPS = {
    "ideal": "Ideal: clean audio, clear vocals, confident teacher",
    "teacher-worst": (
        "Teacher at its worst: fine audio, the teacher's vocals are likely wrong"
    ),
    "doubtful": "Doubtful: kept out of training until heard",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def content_of(r: dict) -> str:
    v, active = r["vocal_share_db"], r["vocal_active_share"]
    if v < INSTRUMENTAL_DB:
        return "instrumental"
    if v >= WHOLE_MIX_DB:
        return "speech" if r["genre"] == SPOKEN else "vocal-only"
    if v < WEAK_DB or active < WEAK_ACTIVE:
        return "weak-vocal"
    return "vocal"


def is_historic(r: dict) -> bool:
    if r["genre"] != HISTORIC:
        return False
    title = str(r["title"])
    dated = HISTORIC_YEAR.search(title) and not LATER_RECORDING.search(title)
    return bool(
        r["stereo"] == "mono"
        or r["side_db"] < HISTORIC_SIDE_DB
        or r["bandwidth_khz"] < FULL_BANDWIDTH_KHZ
        or dated
    )


def judge_pell(r: dict) -> tuple[str, str]:
    if r["duplicate_of"]:
        return "trash", f"duplicate of {r['duplicate_of']}"
    if r["teacher_share_db"] < PELL_MIN_DB:
        level = r["teacher_share_db"]
        return "trash", (
            f"not a clean voice: the teacher hears {level:.1f} dB of voice "
            "against the mix"
        )
    if r["bandwidth_khz"] < MIN_BANDWIDTH_KHZ:
        return "trash", f"band {r['bandwidth_khz']:.1f} kHz"
    return "use", ""


def teacher_risks(r: dict) -> list[str]:
    risks = []
    if r["meta_instrumental"] is True and r["content"] in VOICED:
        risks.append(AUTHOR_INSTRUMENTAL)
    if r["content"] == "vocal-only" and r["genre"] != SPOKEN:
        risks.append(WHOLE_MIX)
    unsure = r["ambiguous_share"]
    if r["content"] in ("vocal", "weak-vocal") and unsure > UNSURE_SHARE:
        risks.append(f"{UNSURE}: {unsure:.0%} of the energy split 20-80 %")
    if r["content"] == "vocal" and r["vocal_low_share"] > BASS_SHARE:
        low = r["vocal_low_share"]
        risks.append(f"{BASS}: {low:.0%} of the vocal energy below 120 Hz")
    return risks


def trash_reason(r: dict) -> str | None:
    if r["duplicate_of"]:
        return f"duplicate of {r['duplicate_of']}"
    if r["silence_share"] > SILENT_SHARE:
        return f"silence {r['silence_share']:.0%} of the time"
    if is_historic(r):
        return (
            "historic recording (cylinder, 78 rpm) before the 1930s: nearly mono "
            f"(side {r['side_db']:.0f} dB), band {r['bandwidth_khz']:.1f} kHz, "
            "outside the product's domain"
        )
    if r["bandwidth_khz"] < MIN_BANDWIDTH_KHZ:
        return f"band {r['bandwidth_khz']:.1f} kHz"
    return None


def use_notes(r: dict, risk: str) -> list[str]:
    notes = []
    if r["bandwidth_khz"] < FULL_BANDWIDTH_KHZ:
        notes.append(
            f"narrow band {r['bandwidth_khz']:.1f} kHz (low bitrate), the class is fine"
        )
    if r["clip_share"] > CLIP_SHARE:
        notes.append(
            f"clipping {r['clip_share']:.2%} of samples at the peak, the class is fine"
        )
    if risk == "high":
        notes.append(RISK_HEARD)
    return notes


def judge(r: dict) -> tuple[str, str, str, str]:
    risks = teacher_risks(r)
    risk = "high" if risks else ("medium" if r["content"] == "weak-vocal" else "low")
    reason = trash_reason(r)
    if reason is not None:
        return "trash", reason, risk, "; ".join(risks)
    return "use", "; ".join(use_notes(r, risk)), risk, "; ".join(risks)


def skipped_in_labelling(logs: list[Path]) -> dict[str, str]:
    reasons = {}
    for log in logs:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            match = SKIP_LINE.search(line)
            if match:
                text = match.group(2)
                reasons[match.group(1)] = (
                    f"does not decode ({text[:SKIP_TEXT]})"
                    if "ffmpeg" in text
                    else f"too short: {text}"
                )
    return reasons


def open_rows(corpus: Path) -> list[dict]:
    return [
        dict(
            id=t["id"],
            source=t.get("source", ""),
            artist=t["artist"],
            title=t["title"],
            genre=OPEN_GENRE,
            license=t["license"],
            license_url=t.get("licenseUrl", ""),
            page=t.get("page") or "",
            meta_instrumental=t.get("note") == REJECTED_NOTE,
            language="",
            bitrate_kbps="",
            album="",
        )
        for t in read_json(corpus / "manifest.json")
    ]


def fma_rows(corpus: Path, meta: Path) -> list[dict]:
    raw = raw_tracks(meta)
    rows = []
    for t in read_json(corpus / "fma.json"):
        info = raw.get(t["fmaId"], {})
        bit_rate = info.get("track_bit_rate")
        rows.append(
            dict(
                id=t["id"],
                source=FMA_SOURCE,
                artist=t["artist"],
                title=t["title"],
                genre=t["genre"],
                license=t["license"],
                license_url=t["licenseUrl"] or "",
                page=t["page"] or "",
                meta_instrumental=bool(t["instrumental"]),
                album=info.get("album_title", ""),
                language=info.get("track_language_code", ""),
                bitrate_kbps=int(float(bit_rate)) // 1000 if bit_rate else "",
            )
        )
    return rows


def pell_rows(corpus: Path) -> list[dict]:
    path = corpus / "ccmixter-pells.json"
    if not path.exists():
        return []
    return [
        dict(
            id=t["id"],
            source=PELL_SOURCE,
            artist=t["artist"],
            title=t["title"],
            genre=PELL_GENRE,
            license=t["license"],
            license_url=t["licenseUrl"],
            page=t["page"],
            meta_instrumental=False,
            language="",
            bitrate_kbps="",
            album="",
        )
        for t in read_json(path)
    ]


def read_features(analysis: Path) -> dict[str, dict[str, float]]:
    return {
        row["id"]: {key: float(row[key]) for key in FEATURES}
        for row in read_rows(analysis / "features.csv")
    }


def corpus_duplicates(analysis: Path) -> dict[str, str]:
    prints = np.load(analysis / "fingerprints.npy")
    ids = json.loads((analysis / "fingerprint_ids.json").read_text())
    return within(prints, ids)


def measured(rows: list[dict], args) -> None:
    features = read_features(args.analysis)
    labels = {e["id"]: e for e in json.loads((args.labels / "labels.json").read_text())}
    dup = corpus_duplicates(args.analysis)
    missing = dict.fromkeys(FEATURES, math.nan)
    for row in rows:
        row.update(features.get(row["id"], missing))
        label = labels.get(row["id"], {})
        row["role"] = label.get("role")
        row["teacher_share_db"] = label.get("teacher_share_db", math.nan)
        row["duplicate_of"] = dup.get(row["id"], "")
        row["stereo"] = "mono" if row["side_db"] < MONO_SIDE_DB else "stereo"


def decide(row: dict, skipped: dict[str, str], decisions: dict[str, dict]) -> None:
    if math.isnan(row["seconds"]):
        row["decision"] = "trash"
        row["reason"] = skipped.get(row["id"], NOT_LABELLED)
        row["content"] = row["teacher_risk"] = row["teacher_risk_reason"] = ""
    else:
        row["content"] = content_of(row)
        decision, reason, risk, risk_reason = judge(row)
        row.update(
            decision=decision,
            reason=reason,
            teacher_risk=risk,
            teacher_risk_reason=risk_reason,
        )
        if row["source"] == PELL_SOURCE:
            row["content"] = "pell"
            row["decision"], row["reason"] = judge_pell(row)
    if row["id"] in decisions:
        row["decision"] = decisions[row["id"]]["decision"]
        row["reason"] = decisions[row["id"]]["reason"]
    if row["decision"] == "trash":
        row["role"] = ""


def note_artists(rows: list[dict], open_count: int) -> None:
    open_artists = {r["artist"] for r in rows if r["genre"] == OPEN_GENRE}
    fma_training = Counter(
        r["artist"]
        for r in rows
        if r["source"] == FMA_SOURCE and r["role"] == "train" and r["decision"] == "use"
    )
    for row in rows:
        row["note"] = ""
        if row["source"] == FMA_SOURCE and row["artist"] in open_artists:
            row["note"] = (
                f"the artist is also among the {open_count} open-corpus tracks"
            )
        held = row["genre"] == OPEN_GENRE and row["role"] == "held"
        if held and fma_training[row["artist"]]:
            count = fma_training[row["artist"]]
            row["note"] = (
                f"held out, but {count} FMA training clips are by the same artist"
            )


def verdicts(folder: Path | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if folder is None:
        return out
    for path in sorted(folder.glob("*.csv")):
        date = path.stem[:VERDICT_DATE]
        for row in read_rows(path):
            window = row.get("window_s")
            heard = f"{date}, window {window} s" if window else date
            out.setdefault(row["track"], []).append(f"{heard}: {row['owner_text']}")
    return out


def build(args) -> list[dict]:
    rows = open_rows(args.corpus)
    open_count = len(rows)
    rows += fma_rows(args.corpus, args.meta) + pell_rows(args.corpus)
    before = {r["id"]: r for r in read_rows(args.dataset)}
    measured(rows, args)
    skipped = skipped_in_labelling(args.label_logs)
    decisions = {r["id"]: r for r in read_rows(args.decisions)}
    for row in rows:
        decide(row, skipped, decisions)
    note_artists(rows, open_count)
    fresh = {r["id"] for r in rows}
    rows += [
        dict(r)
        for i, r in before.items()
        if r["decision"] == "trash" and i not in fresh
    ]
    heard = verdicts(args.verdicts)
    for row in rows:
        row["listen_set"] = before.get(row["id"], {}).get("listen_set", "")
        row["owner_verdict"] = " | ".join(heard.get(row["id"], []))
    return rows


def top(rows: list[dict], key: str, count: int) -> list[dict]:
    return sorted(rows, key=lambda r: r[key], reverse=True)[:count]


def ideal_set(usable: list[dict]) -> list[dict]:
    low, high = IDEAL_VOCAL_DB
    vocal = [
        r
        for r in usable
        if r["content"] == "vocal"
        and r["bandwidth_khz"] >= IDEAL_BANDWIDTH_KHZ
        and r["stereo"] == "stereo"
        and r["clip_share"] < IDEAL_CLIP_SHARE
        and r["silence_share"] < IDEAL_SILENCE
        and low <= r["vocal_share_db"] <= high
        and r["vocal_active_share"] >= IDEAL_ACTIVE
        and r["teacher_risk"] == "low"
        and r["vocal_low_share"] < IDEAL_LOW_SHARE
    ]
    clearest: dict[str, dict] = {}
    for r in sorted(vocal, key=lambda r: r["ambiguous_share"]):
        clearest.setdefault(r["genre"], r)
    left_out = (OPEN_GENRE, HISTORIC, SPOKEN)
    ideal = [r for genre, r in clearest.items() if genre not in left_out]
    ideal += [clearest[OPEN_GENRE]] if OPEN_GENRE in clearest else []
    instrumental = [
        r
        for r in usable
        if r["content"] == "instrumental"
        and r["bandwidth_khz"] >= IDEAL_BANDWIDTH_KHZ
        and r["decision"] == "use"
    ]
    return ideal + top(instrumental, "rms_db", IDEAL_INSTRUMENTALS)


def worst_set(usable: list[dict], accompanied: list[str]) -> list[dict]:
    risky = [
        r
        for r in usable
        if r["bandwidth_khz"] >= FULL_BANDWIDTH_KHZ
        and r["clip_share"] < CLIP_SHARE
        and r["teacher_risk"] == "high"
    ]
    by_id = {r["id"]: r for r in usable}

    def with_risk(text: str) -> list[dict]:
        return [r for r in risky if text in r["teacher_risk_reason"]]

    return [
        *top(with_risk(AUTHOR_INSTRUMENTAL), "vocal_share_db", WORST_AUTHOR),
        *[by_id[i] for i in accompanied if i in by_id],
        *top(with_risk(UNSURE), "ambiguous_share", WORST_UNSURE),
        *top(with_risk(BASS), "vocal_low_share", WORST_BASS),
    ]


def doubtful_set(usable: list[dict]) -> list[dict]:
    doubt = [r for r in usable if r["decision"] == "listen"]
    audio = [r for r in doubt if BAND_OR_CLIPPING.search(r["reason"])]
    first: dict[str, dict] = {}
    for r in doubt:
        if r["content"] == "vocal-only" and r["genre"] in DOUBTFUL_GENRES:
            first.setdefault(r["genre"], r)
    return audio[:DOUBTFUL_AUDIO] + list(first.values())


def pick_listening(rows: list[dict], accompanied: list[str]) -> None:
    usable = [r for r in rows if r["decision"] != "trash"]
    worst = {r["id"] for r in worst_set(usable, accompanied)}
    doubtful = {r["id"] for r in doubtful_set(usable)} - worst
    ideal = {r["id"] for r in ideal_set(usable)}
    for row in rows:
        row["listen_set"] = ""
        for name, chosen in (("ideal", ideal), ("teacher-worst", worst)):
            if row["id"] in chosen:
                row["listen_set"] = name
        if row["id"] in doubtful:
            row["listen_set"] = "doubtful"


def excerpt(labels: Path, name: str, seconds: int) -> tuple[np.ndarray, np.ndarray]:
    mix, voc = load(labels, name, "mix"), load(labels, name, "vocals")
    if mix.shape[1] <= seconds * SAMPLE_RATE:
        return mix, voc
    whole = mix.shape[1] // SAMPLE_RATE * SAMPLE_RATE
    energy = (voc**2).sum(0)[:whole].reshape(-1, SAMPLE_RATE).sum(1)
    energy = np.convolve(energy, np.ones(seconds), "valid")
    start = int(np.argmax(energy)) * SAMPLE_RATE
    span = slice(start, start + seconds * SAMPLE_RATE)
    return mix[:, span], voc[:, span]


def listening_row(n: int, r: dict) -> str:
    why = r["teacher_risk_reason"] or r["reason"]
    if not why:
        why = (
            f"vocals {r['vocal_share_db']:.0f} dB, "
            f"ambiguity {r['ambiguous_share']:.0%}"
        )
    track = f"{r['artist']} - {r['title']} (`{r['id']}`)"
    return f"| {n:02d} | {track} | {r['genre']} | {r['content']} | {why} |"


def write_listening(rows: list[dict], labels: Path, out: Path) -> None:
    lines = [
        "# Listening: what the students learn from",
        "",
        "Each item: `<n>-mix.flac` (the track, 30 s) and `<n>-teacher.flac` "
        "(the teacher's vocals on it).",
        "",
    ]
    for group, title in LISTEN_GROUPS.items():
        folder = out / group
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        lines += [f"## {title}", "", "| # | Track | Genre | Content | Why |"]
        lines += ["| --- | --- | --- | --- | --- |"]
        chosen = [r for r in rows if r["listen_set"] == group]
        for n, r in enumerate(chosen, 1):
            mix, voc = excerpt(labels, r["id"], EXCERPT_SECONDS)
            sf.write(folder / f"{n:02d}-{r['id']}-mix.flac", mix.T, SAMPLE_RATE)
            sf.write(folder / f"{n:02d}-{r['id']}-teacher.flac", voc.T, SAMPLE_RATE)
            lines.append(listening_row(n, r))
        lines.append("")
    (out / "listen.md").write_text("\n".join(lines), encoding="utf-8")


def without_trash(path: Path, trash: set[str], corpus: Path) -> None:
    if not path.exists():
        return
    entries = read_json(path)
    for e in entries:
        if e["id"] in trash:
            (corpus / e["audio"]).unlink(missing_ok=True)
    kept = [e for e in entries if e["id"] not in trash]
    text = json.dumps(kept, indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def apply_trash(rows: list[dict], corpus: Path, labels: Path) -> None:
    trash = {r["id"] for r in rows if r["decision"] == "trash"}
    if any(e["id"] in trash for e in read_json(corpus / "manifest.json")):
        message = "a track of the open corpus is trash: leave it out with data-open"
        raise ValueError(message)
    without_trash(corpus / "fma.json", trash, corpus)
    without_trash(corpus / "ccmixter-pells.json", trash, corpus)
    index = json.loads((labels / "labels.json").read_text())
    for name in trash:
        for kind in ("mix", "vocals", "truth"):
            (labels / f"{name}.{kind}.npy").unlink(missing_ok=True)
    kept = [e for e in index if e["id"] not in trash]
    (labels / "labels.json").write_text(json.dumps(kept, indent=1))
    LOG.info(f"deleted {len(trash)} trash tracks")


def cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else repr(value)
    return str(value)


def clip_cell(value):
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.2e}"
    return value


def write_dataset(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in rows:
            cells = {**row, "clip_share": clip_cell(row.get("clip_share"))}
            writer.writerow([cell(cells.get(column)) for column in COLUMNS])


def log_counts(rows: list[dict]) -> None:
    LOG.info(f"decisions: {dict(Counter(r['decision'] for r in rows))}")
    for column in ("genre", "content"):
        pairs = Counter((r[column], r["decision"]) for r in rows)
        for (value, decision), count in sorted(pairs.items()):
            LOG.info(f"{column} {value or '-'}, {decision}: {count}")
    LOG.info(f"listen sets: {dict(Counter(r['listen_set'] for r in rows))}")
    reasons = Counter(
        re.sub(r"[:(].*", "", r["reason"]) for r in rows if r["decision"] == "trash"
    )
    for reason, count in reasons.most_common():
        LOG.info(f"trash, {reason}: {count}")


def curate(args) -> None:
    rows = build(args)
    if args.listen:
        accompanied = []
        if args.accompanied:
            accompanied = args.accompanied.read_text(encoding="utf-8").split()
        pick_listening(rows, accompanied)
        write_listening(rows, args.labels, args.listen)
    if args.apply:
        apply_trash(rows, args.corpus, args.labels)
    write_dataset(args.dataset, rows)
    log_counts(rows)


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "data-curate",
        parents=[common],
        help="decide per track what trains, from the rules and hand-made inputs",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Directory with manifest.json, fma.json and ccmixter-pells.json",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Directory of the teacher's labels and labels.json",
    )
    parser.add_argument(
        "--analysis",
        required=True,
        type=Path,
        help="Directory of features.csv, fingerprints.npy and fingerprint_ids.json",
    )
    parser.add_argument(
        "--meta",
        required=True,
        type=Path,
        help="Directory of the FMA metadata (raw_tracks.csv)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="dataset.csv: read as the record of earlier decisions, then written",
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        help="Directory of listening verdicts, <date>-<what>.csv with the columns "
        "track and owner_text, and window_s where a spot was heard",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        help="CSV of hand-made decisions (id, decision, reason) over the rules",
    )
    parser.add_argument(
        "--label-logs",
        type=Path,
        nargs="*",
        default=[],
        help="Logs of the labelling whose 'skip <id>: <why>' lines give the reason "
        "for a track without labels",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the trash: its audio, its labels, its manifest entries",
    )
    parser.add_argument(
        "--listen",
        type=Path,
        help="Pick the listening sets anew and write their 30 s excerpts here",
    )
    parser.add_argument(
        "--accompanied",
        type=Path,
        help="Ids of tracks known to be voice with instruments, one per line, "
        "listed among the teacher's worst with --listen",
    )
    parser.set_defaults(run=curate)
