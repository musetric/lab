import argparse
import logging
from collections import Counter

import numpy as np

from musetric_toolkit.distill.excluded import excluded_ids
from musetric_toolkit.listening.frozen_rule import Rule, frozen_rule
from musetric_toolkit.listening.levels import letter_words, window_level_db
from musetric_toolkit.listening.round_files import (
    KEY,
    LABELS,
    read_json,
    read_jsonl,
    write_json,
)
from musetric_toolkit.listening.round_labels import present_of

LOG = logging.getLogger(__name__)
RANDOM = "random"
DETECTOR_QUIET = "detector_quiet"
DETECTOR_LOUD = "detector_loud"
REVERSE_TEACHER = "reverse_teacher"
REVERSE_WORDS = "reverse_words"
NO_VOICE_GROUP = "no_voice"
VOICED_GROUP = "voiced"


class Picks:
    def __init__(self, levels: dict[str, dict], args: argparse.Namespace) -> None:
        self.levels = levels
        self.args = args
        self.quotas = {
            RANDOM: args.random_clips,
            DETECTOR_QUIET: args.detector_quiet_clips,
            DETECTOR_LOUD: args.detector_loud_clips,
        }
        groups = (*self.quotas, REVERSE_TEACHER, REVERSE_WORDS)
        self.genres = {group: Counter() for group in groups}
        self.artists = set()
        self.picked = []

    def fits(self, meta: dict, group: str) -> bool:
        if meta["artist"] in self.artists:
            return False
        if group not in self.quotas:
            return True
        limit = int(self.quotas[group] * self.args.genre_share)
        return self.genres[group][meta["genre"]] < limit

    def audible(self, name: str, start_s: float) -> bool:
        args = self.args
        level = window_level_db(args.labels, name, start_s, args.window_s)
        return level >= args.min_window_db

    def take(self, meta: dict, group: str, start_s: float, scores: dict) -> None:
        self.artists.add(meta["artist"])
        self.genres[group][meta["genre"]] += 1
        self.picked.append(
            dict(
                source=meta["id"],
                sign=group,
                start_s=float(start_s),
                scores=scores,
                genre=meta["genre"],
            )
        )

    def count(self, group: str) -> int:
        return sum(pick["sign"] == group for pick in self.picked)


def pick_random(picks: Picks, rng) -> None:
    args = picks.args
    names = sorted(picks.levels)
    for index in rng.permutation(len(names)):
        meta = picks.levels[names[index]]
        voiced = np.flatnonzero(np.array(meta["vocals_to_mix"]) >= args.gate_db)
        if meta["mix_rms"] < args.min_mix_rms or not len(voiced):
            continue
        if not picks.fits(meta, RANDOM):
            continue
        window = int(rng.choice(voiced))
        start_s = window * args.hop_s
        if not picks.audible(meta["id"], start_s):
            continue
        scores = dict(student=meta["vocals_to_mix"][window])
        picks.take(meta, RANDOM, start_s, scores)
        if picks.count(RANDOM) == args.random_clips:
            break


def window_group(
    window: dict, scores: dict, rule: Rule | None, args: argparse.Namespace
) -> tuple[str, float] | None:
    if window["kind"] == "vocals":
        if rule is None:
            return None
        score = rule(window["dominant"], window["teacher"], window["words"])
        scores["score"] = round(score, 3)
        quiet = window["teacher"] < args.teacher_quiet_db
        return (DETECTOR_QUIET if quiet else DETECTOR_LOUD), score
    if window["kind"] == "instrumental":
        return REVERSE_WORDS, float(letter_words(window["text"]))
    if window["teacher"] >= args.teacher_voice_db:
        return REVERSE_TEACHER, window["teacher"] - window["student"]
    return None


def best_windows(row: dict, rule: Rule | None, args: argparse.Namespace) -> dict:
    best = {}
    for window in row["windows"]:
        scores = {
            name: window[name]
            for name in ("student", "teacher", "dominant", "words", "text")
        }
        found = window_group(window, scores, rule, args)
        if found is None:
            continue
        group, value = found
        if group not in best or value > best[group][0]:
            best[group] = (value, window["start_s"], scores)
    return best


def rank_windows(
    separated: list[dict], rule: Rule | None, args: argparse.Namespace
) -> dict[str, list]:
    groups = (DETECTOR_QUIET, DETECTOR_LOUD, REVERSE_TEACHER, REVERSE_WORDS)
    ranked = {group: [] for group in groups}
    for row in separated:
        best = best_windows(row, rule, args)
        for group, (value, start_s, scores) in best.items():
            ranked[group].append((value, row["id"], start_s, scores))
    for group, rows in ranked.items():
        rows.sort(key=lambda row: (-row[0], row[1]))
        LOG.info("%s candidates %d", group, len(rows))
    return ranked


def fill(picks: Picks, ranked: list, group: str, quota: int) -> int:
    count, last = 0, None
    for value, name, start_s, scores in ranked:
        if count == quota:
            break
        meta = picks.levels[name]
        if not picks.fits(meta, group) or not picks.audible(name, start_s):
            continue
        picks.take(meta, group, start_s, scores)
        count, last = count + 1, value
    LOG.info("%s taken %d, last value %s", group, count, last)
    return count


def pick_ranked(picks: Picks, ranked: dict[str, list]) -> None:
    args = picks.args
    from_teacher = fill(
        picks, ranked[REVERSE_TEACHER], REVERSE_TEACHER, args.reverse_teacher_clips
    )
    fill(picks, ranked[REVERSE_WORDS], REVERSE_WORDS, args.reverse_clips - from_teacher)
    fill(picks, ranked[DETECTOR_QUIET], DETECTOR_QUIET, args.detector_quiet_clips)
    fill(picks, ranked[DETECTOR_LOUD], DETECTOR_LOUD, args.detector_loud_clips)


def insert_far_from(order: list[dict], copy: dict, original: dict, rng, gap: int):
    place = order.index(original)
    if max(place, len(order) - place) < gap:
        message = f"a round of {len(order)} clips has no place {gap} from its original"
        raise ValueError(message)
    while True:
        at = int(rng.integers(0, len(order) + 1))
        if abs(at - place) >= gap:
            order.insert(at, copy)
            return


def add_own_repeats(order: list[dict], rng, args: argparse.Namespace) -> None:
    own = []
    for group, count in args.own_repeats.items():
        members = [pick for pick in order if pick["sign"] == group]
        own += [members[i] for i in rng.permutation(len(members))[:count]]
    for original in own:
        repeat_of = f"{original['source']}@{original['start_s']}"
        copy = {**original, "repeat_of": repeat_of}
        insert_far_from(order, copy, original, rng, args.repeat_distance)


def earlier_repeats(rng, args: argparse.Namespace, excluded: set[str]) -> list[dict]:
    folder = args.repeats_from
    key = read_json(folder / KEY)
    labels = read_json(folder / LABELS)
    groups = {NO_VOICE_GROUP: [], VOICED_GROUP: []}
    for clip, entry in key.items():
        if not entry["train"] or entry["source"] in excluded:
            continue
        present = present_of(labels[clip])
        if present["lead"] is False and present["backing"] is False:
            groups[NO_VOICE_GROUP].append((clip, entry))
        elif present["lead"] or present["backing"]:
            groups[VOICED_GROUP].append((clip, entry))
    chosen = []
    for group, count in args.earlier_repeats.items():
        for index in rng.permutation(len(groups[group]))[:count]:
            clip, entry = groups[group][index]
            chosen.append(
                dict(
                    source=entry["source"],
                    sign=f"repeat_{folder.name}_{group}",
                    start_s=entry["start_s"],
                    repeat_of=f"{folder.name}:{clip}",
                    genre=None,
                )
            )
    return chosen


def check_inputs(args: argparse.Namespace) -> None:
    detectors = args.detector_quiet_clips + args.detector_loud_clips
    if not args.windows and detectors + args.reverse_clips > 0:
        raise ValueError("the detector and reverse groups need --windows")
    if not args.rule_features and detectors > 0:
        raise ValueError("the detector groups need --rule-features")


def select_clips(args: argparse.Namespace, excluded: set[str]) -> list[dict]:
    check_inputs(args)
    rng = np.random.default_rng(args.seed)
    rule = frozen_rule(args.rule_features) if args.rule_features else None
    rows = read_jsonl(args.levels)
    levels = {row["id"]: row for row in rows if row["id"] not in excluded}
    picks = Picks(levels, args)
    pick_random(picks, rng)
    if args.windows:
        rows = read_jsonl(args.windows)
        separated = [row for row in rows if row["id"] not in excluded]
        pick_ranked(picks, rank_windows(separated, rule, args))
    order = [picks.picked[index] for index in rng.permutation(len(picks.picked))]
    add_own_repeats(order, rng, args)
    if args.repeats_from:
        for copy in earlier_repeats(rng, args, excluded):
            order.insert(int(rng.integers(0, len(order) + 1)), copy)
    LOG.info(
        "clips %d, %s, repeats %d, artists %d",
        len(order),
        dict(Counter(pick["sign"] for pick in order)),
        sum("repeat_of" in pick for pick in order),
        len(picks.artists),
    )
    return order


def select_command(args: argparse.Namespace) -> None:
    write_json(args.out, select_clips(args, excluded_ids()))
