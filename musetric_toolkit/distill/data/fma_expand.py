import csv
import logging
import random
from collections import Counter
from pathlib import Path

from musetric_toolkit.distill.data.fma_fetch import (
    FMA_PREFIX,
    LARGE_PREFIX,
    LARGE_URL,
    PROGRESS_EVERY,
    WORKERS,
    FmaTrack,
    archive_member,
    catalogue,
    clip_entry,
    clip_file,
    grouped,
    judged_trash,
    read_json,
    write_json,
)
from musetric_toolkit.distill.data.remote_zip import (
    GIGABYTE,
    fetch_members,
    open_archive,
)

LOG = logging.getLogger(__name__)

FULL_URL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
FULL_PREFIX = "fma_full"
CAP = 6
MIN_SECONDS = 60
MIN_BIT_RATE = 128000
HISTORIC = "Old-Time / Historic"
VOICED = ("vocal", "vocal-only")
INSTRUMENTAL = 300
CANDIDATES = "fma-candidates.csv"
COLUMNS = [
    "track_id",
    "artist",
    "artist_id",
    "title",
    "genre",
    "seconds",
    "track_bit_rate",
    "track_instrumental",
    "license_title",
    "license_url",
    "sharealike",
    "track_url",
]


def spread_over_albums(group: list[FmaTrack], rng: random.Random) -> list[FmaTrack]:
    tracks = list(group)
    rng.shuffle(tracks)
    tracks.sort(key=lambda t: t.instrumental)
    picked, albums = [], set()
    for track in tracks:
        if track.album_id is None or track.album_id not in albums:
            picked.append(track)
            albums.add(track.album_id)
    taken = {t.track_id for t in picked}
    return picked + [t for t in tracks if t.track_id not in taken]


def candidates(args) -> list[FmaTrack]:
    tracks = catalogue(args.meta)
    by_id = {t.track_id: t for t in tracks}
    clips = read_json(args.corpus / "fma.json")
    in_corpus = {e["fmaId"] for e in clips}
    held = {by_id[e["fmaId"]].artist_id for e in clips if e["split"] == "held"}
    trash = judged_trash(args.dataset)
    pool = [
        t
        for t in tracks
        if t.track_id not in in_corpus
        and t.track_id not in trash
        and t.artist_id not in held
        and t.genre != HISTORIC
        and t.seconds >= MIN_SECONDS
        and (t.bit_rate or 0) >= MIN_BIT_RATE
    ]
    used = Counter(by_id[i].artist_id for i in in_corpus if i in by_id)
    rng = random.Random(args.seed)  # noqa: S311
    chosen = []
    for artist, group in grouped(pool, lambda t: t.artist_id):
        room = CAP - used[artist]
        if room > 0:
            chosen += spread_over_albums(group, rng)[:room]
    return sorted(chosen, key=lambda t: t.track_id)


def write_candidates(path: Path, tracks: list[FmaTrack]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for t in tracks:
            writer.writerow(
                [
                    t.track_id,
                    t.artist,
                    t.artist_id,
                    t.title,
                    t.genre,
                    t.seconds,
                    "" if t.bit_rate is None else float(t.bit_rate),
                    t.instrumental,
                    t.license_title,
                    t.license_url or "",
                    t.sharealike,
                    t.page or "",
                ]
            )


def read_candidates(path: Path) -> list[FmaTrack]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [
            FmaTrack(
                track_id=int(row["track_id"]),
                title=row["title"],
                artist=row["artist"],
                artist_id=int(row["artist_id"]),
                genre=row["genre"],
                seconds=int(row["seconds"]),
                license_title=row["license_title"],
                license_url=row["license_url"] or None,
                page=row["track_url"] or None,
                instrumental=int(row["track_instrumental"]),
                bit_rate=int(float(row["track_bit_rate"])),
                album_id=None,
            )
            for row in csv.DictReader(f)
        ]


def fetch_tracks(url: str, prefix: str, ids: list[int], folder: Path, workers: int):
    folder.mkdir(parents=True, exist_ok=True)
    archive = open_archive(url)
    jobs = [
        (archive.getinfo(archive_member(prefix, i)), folder / clip_file(i)) for i in ids
    ]
    todo = [
        (info, out)
        for info, out in jobs
        if not out.exists() or out.stat().st_size != info.file_size
    ]
    LOG.info(f"{len(ids)} wanted, {len(todo)} to fetch from {url}")
    size = fetch_members(url, todo, workers, PROGRESS_EVERY)
    LOG.info(f"fetched {len(todo)}, {size / GIGABYTE:.2f} GB")


def screened_ids(
    analysis: Path, by_id: dict[int, FmaTrack], duplicates: dict[str, str]
) -> list[tuple[int, str]]:
    with open(analysis / "fma-screen.csv", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["id"] not in duplicates]
    screened = [(int(r["id"].removeprefix(FMA_PREFIX)), r["content"]) for r in rows]
    return [(i, content) for i, content in screened if not by_id[i].sharealike]


def add_clips(args, cand: list[FmaTrack]) -> None:
    by_id = {t.track_id: t for t in cand}
    duplicates = read_json(args.analysis / "fma-screen-duplicates.json")
    screened = screened_ids(args.analysis, by_id, duplicates)
    voiced = {i for i, content in screened if content in VOICED}
    pool = sorted(i for i, content in screened if content == "instrumental")
    rng = random.Random(args.seed)  # noqa: S311
    sample = rng.sample(pool, min(args.instrumental, len(pool)))
    chosen = sorted(voiced | set(sample))
    manifest_path = args.corpus / "fma.json"
    manifest = read_json(manifest_path)
    present = {e["fmaId"] for e in manifest}
    manifest += [clip_entry(by_id[i], "train") for i in chosen if i not in present]
    write_json(manifest_path, manifest)
    added = "".join(f"{FMA_PREFIX}{i:06d}\n" for i in chosen)
    path = args.corpus / f"fma-expand-{args.add}.txt"
    path.write_text(added, encoding="utf-8")
    LOG.info(
        f"added {len(chosen)} clips: {len(voiced)} voiced, "
        f"{len(chosen) - len(voiced)} instrumental; "
        f"{len(duplicates)} duplicates and CC BY-SA left out"
    )


def log_candidates(cand: list[FmaTrack]) -> None:
    hours = sum(t.seconds for t in cand) / 3600
    artists = len({t.artist_id for t in cand})
    sharealike = sum(t.sharealike for t in cand)
    flagged = sum(t.instrumental == 0 for t in cand)
    LOG.info(
        f"{len(cand)} candidates from {artists} artists, {hours:.0f} h in full; "
        f"{sharealike} CC BY-SA; author's vocal flag on {flagged}"
    )
    by_genre = grouped(cand, lambda t: t.genre)
    for genre, group in sorted(by_genre, key=lambda item: len(item[1])):
        genre_hours = sum(t.seconds for t in group) / 3600
        LOG.info(f"{genre}: {len(group)} tracks, {genre_hours:.1f} h")


def expand(args) -> None:
    path = args.corpus / CANDIDATES
    if path.exists():
        cand = read_candidates(path)
    else:
        cand = candidates(args)
        write_candidates(path, cand)
    log_candidates(cand)
    if args.clips:
        ids = [t.track_id for t in cand]
        fetch_tracks(LARGE_URL, LARGE_PREFIX, ids, args.corpus / "fma", args.workers)
    if args.full:
        ids = [int(i) for i in args.full.read_text(encoding="utf-8").split()]
        folder = args.corpus / "fma-full"
        fetch_tracks(FULL_URL, FULL_PREFIX, ids, folder, args.workers)
    if args.add:
        add_clips(args, cand)


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "data-fma-expand",
        parents=[common],
        help="list more permissive FMA tracks, fetch them and add the screened ones",
    )
    parser.add_argument(
        "--meta",
        required=True,
        type=Path,
        help="Directory of the FMA metadata: tracks.csv, raw_tracks.csv, genres.csv",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Corpus directory with fma.json; fma-candidates.csv is written there",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="dataset.csv of the corpus: tracks judged trash are no candidates",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        help="Directory of fma-screen.csv and fma-screen-duplicates.json (--add)",
    )
    parser.add_argument(
        "--clips",
        action="store_true",
        help="Fetch the candidates' 30 s clips from fma_large.zip into fma/",
    )
    parser.add_argument(
        "--full",
        type=Path,
        help="Fetch the full tracks whose ids this file lists into fma-full/",
    )
    parser.add_argument(
        "--add",
        help="Add the screened clips to fma.json; the ids go to fma-expand-<add>.txt",
    )
    parser.add_argument(
        "--instrumental",
        type=int,
        default=INSTRUMENTAL,
        help="Instrumental clips to add with --add",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.set_defaults(run=expand)
