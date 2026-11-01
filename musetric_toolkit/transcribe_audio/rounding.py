import math


def js_round(value: float) -> int:
    return math.floor(value + 0.5)


def round_to(value: float, digits: int) -> float:
    power = 10**digits
    return js_round(value * power) / power
