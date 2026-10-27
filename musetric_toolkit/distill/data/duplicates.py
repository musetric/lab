import numpy as np

SAME = 0.99


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def first_of_pairs(pairs, ids: list[str], found: dict[str, str]) -> dict[str, str]:
    for i, j in pairs:
        if ids[i] not in found and ids[j] not in found:
            found[ids[j]] = ids[i]
    return found


def within(prints: np.ndarray, ids: list[str]) -> dict[str, str]:
    sim = unit(prints) @ unit(prints).T
    np.fill_diagonal(sim, 0)
    pairs = zip(*np.where(np.triu(sim) > SAME), strict=True)
    return first_of_pairs(pairs, ids, {})


def against(new: np.ndarray, ids: list[str], old: np.ndarray, old_ids: list[str]):
    cross = unit(new) @ unit(old).T
    found = {ids[i]: old_ids[j] for i, j in zip(*np.where(cross > SAME), strict=True)}
    sim = unit(new) @ unit(new).T
    np.fill_diagonal(sim, 0)
    pairs = zip(*np.where(np.triu(sim) > SAME), strict=True)
    return first_of_pairs(pairs, ids, found)
