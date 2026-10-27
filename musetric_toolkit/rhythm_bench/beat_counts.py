from dataclasses import dataclass

import numpy as np

NEIGHBOURS = 6
SHORT = 0.8
LONG = 1.25
JUMP_SPAN = 8
JUMP_UP = 1.15
JUMP_DOWN = 0.87
MATCH_SECONDS = 0.07


@dataclass
class BeatCounts:
    spurious: int
    gap_seconds: float
    tempo_jumps: int


def upper_median(values: np.ndarray) -> float:
    return float(np.sort(values)[len(values) // 2])


def local_beats(intervals: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            upper_median(intervals[max(0, index - NEIGHBOURS) : index + NEIGHBOURS + 1])
            for index in range(len(intervals))
        ]
    )


def off_beat_intervals(beats: np.ndarray) -> list[tuple[float, float]]:
    intervals = np.diff(beats)
    ratios = intervals / local_beats(intervals)
    return [
        (float(beats[index]), float(intervals[index]))
        for index in np.flatnonzero((ratios < SHORT) | (ratios > LONG))
    ]


def count_tempo_jumps(local: np.ndarray) -> int:
    jumps = 0
    index = JUMP_SPAN
    while index < len(local):
        change = local[index] / local[index - JUMP_SPAN]
        if change > JUMP_UP or change < JUMP_DOWN:
            jumps += 1
            index += JUMP_SPAN
        index += 1
    return jumps


def count_beats(beats: np.ndarray) -> BeatCounts:
    intervals = np.diff(beats)
    if len(intervals) == 0:
        return BeatCounts(spurious=0, gap_seconds=0.0, tempo_jumps=0)
    local = local_beats(intervals)
    ratios = intervals / local
    return BeatCounts(
        spurious=int(np.sum(ratios < SHORT)),
        gap_seconds=float(np.sum(intervals[ratios > LONG])),
        tempo_jumps=count_tempo_jumps(local),
    )


def beat_f_measure(reference: np.ndarray, beats: np.ndarray) -> float:
    if len(reference) + len(beats) == 0:
        return 1.0
    used: set[int] = set()
    hits = 0
    for time in beats:
        best = -1
        distance = MATCH_SECONDS
        for index, candidate in enumerate(reference):
            gap = abs(candidate - time)
            if index not in used and gap <= distance:
                best = index
                distance = gap
        if best >= 0:
            used.add(best)
            hits += 1
    return 2 * hits / (len(reference) + len(beats))
