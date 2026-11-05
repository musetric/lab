from scipy.stats import binomtest

from musetric_toolkit.listening.round_labels import GROUPS

EVEN_ODDS = 0.5
CONFIDENCE = 0.95


def exact(wins: int, total: int) -> tuple[float, float, float]:
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    test = binomtest(wins, total, EVEN_ODDS)
    interval = test.proportion_ci(CONFIDENCE, method="exact")
    return interval.low, interval.high, test.pvalue


def share(wins: int, total: int) -> str:
    low, high, _ = exact(wins, total)
    return f"{wins}/{total} [{low:.2f}, {high:.2f}]"


def ticks(present: dict) -> str:
    return "".join(g[0].upper() if present[g] else "-" for g in GROUPS)


def system_letters(names: tuple[str, str]) -> tuple[str, str]:
    first, second = names[0][0].upper(), names[1][0].upper()
    return (first, second) if first != second else ("1", "2")


def absent_outcome(row: dict, limit: float, names: tuple[str, str]) -> str:
    present, first, second = row["present"], *names
    if present["lead"] is False and present["backing"] is False:
        stems = ["vocals"]
    else:
        stems = [group for group in ("lead", "backing") if present[group] is False]
    if not stems:
        return "nothing absent"
    wrong_first = any(row["systems"][first][stem] >= limit for stem in stems)
    wrong_second = any(row["systems"][second][stem] >= limit for stem in stems)
    if wrong_first and wrong_second:
        return "both wrong"
    if wrong_first:
        return f"{first} only"
    return f"{second} only" if wrong_second else "both right"
