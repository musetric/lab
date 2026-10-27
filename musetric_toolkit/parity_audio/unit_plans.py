from dataclasses import dataclass

import numpy as np

VOCALS_STEP_SECONDS = 8
LEAD_BACKING_OVERLAP = 0.25
LEAD_BACKING_COMPENSATE = 1.065
COUNTER_FLOOR = 1e-10
MAX_PEAK = 0.9


@dataclass(frozen=True)
class PlanUnit:
    start: int
    length: int


@dataclass(frozen=True)
class UnitPlan:
    units: list[PlanUnit]
    chunk_samples: int
    weights: dict[int, np.ndarray]


def vocals_plan(sample_rate: int, samples: int, chunk_samples: int) -> UnitPlan:
    step = min(VOCALS_STEP_SECONDS * sample_rate, chunk_samples)
    units = []
    offset = 0
    while offset < samples:
        if offset + chunk_samples <= samples:
            units.append(PlanUnit(offset, chunk_samples))
        elif samples >= chunk_samples:
            units.append(PlanUnit(samples - chunk_samples, chunk_samples))
        else:
            units.append(PlanUnit(0, samples))
        offset += step
    position = np.arange(chunk_samples)
    table = 0.54 - 0.46 * np.cos(2.0 * np.pi * position / chunk_samples)
    weights = {unit.length: table[: unit.length] for unit in units}
    return UnitPlan(units, chunk_samples, weights)


@dataclass(frozen=True)
class LeadBackingLayout:
    plan: UnitPlan
    trim: int
    mixture_samples: int


def lead_backing_plan(
    samples: int, chunk_samples: int, n_fft: int
) -> LeadBackingLayout:
    trim = n_fft // 2
    gen_samples = chunk_samples - 2 * trim
    mixture_samples = 2 * trim + samples + gen_samples - samples % gen_samples
    step = int((1.0 - LEAD_BACKING_OVERLAP) * chunk_samples)
    units = []
    start = 0
    while start < mixture_samples:
        units.append(PlanUnit(start, min(chunk_samples, mixture_samples - start)))
        start += step
    weights = {}
    for unit in units:
        if unit.length == 1:
            weights[1] = np.ones(1)
        else:
            position = np.arange(unit.length)
            weights[unit.length] = 0.5 - 0.5 * np.cos(
                2.0 * np.pi * position / (unit.length - 1)
            )
    return LeadBackingLayout(
        UnitPlan(units, chunk_samples, weights), trim, mixture_samples
    )


def unit_window(planar: np.ndarray, plan: UnitPlan, index: int) -> np.ndarray:
    unit = plan.units[index]
    chunk = np.zeros((planar.shape[0], plan.chunk_samples))
    chunk[:, : unit.length] = planar[:, unit.start : unit.start + unit.length]
    return chunk


def fold(plan: UnitPlan, chunks: list[np.ndarray], frames: int) -> np.ndarray:
    target = np.zeros((chunks[0].shape[0], frames))
    counter = np.zeros((chunks[0].shape[0], frames))
    for unit, chunk in zip(plan.units, chunks, strict=True):
        weight = plan.weights[unit.length]
        target[:, unit.start : unit.start + unit.length] += (
            chunk[:, : unit.length] * weight
        )
        counter[:, unit.start : unit.start + unit.length] += weight
    return target / np.maximum(counter, COUNTER_FLOOR)


def normalize_peak(samples: np.ndarray, max_peak: float = MAX_PEAK) -> np.ndarray:
    peak = float(np.abs(samples).max())
    if peak == 0.0 or peak <= max_peak:
        return samples
    return samples * (max_peak / peak)
