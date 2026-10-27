import logging
from pathlib import Path

import torch

LOG = logging.getLogger(__name__)

TRAINING_STATE = ("model", "opt", "sched", "scaler")


def checked_first(states: list[dict], checkpoints: list[Path]) -> dict:
    first = states[0]
    if first.get("partial"):
        message = f"{checkpoints[0]} is a partial snapshot: the first must be whole"
        raise ValueError(message)
    keys = first["model"].keys()
    for path, state in zip(checkpoints, states, strict=True):
        if state["student"] != first["student"]:
            message = f"{path} is {state['student']}, not {first['student']}"
            raise ValueError(message)
        extra = state["model"].keys() - keys
        if extra:
            message = f"{path} holds {len(extra)} tensors the first does not"
            raise ValueError(message)
    return first


def averaged(states: list[dict], weights: list[float]) -> dict:
    model = {}
    for k, t in states[0]["model"].items():
        held = [
            (w, s["model"][k])
            for w, s in zip(weights, states, strict=True)
            if k in s["model"]
        ]
        if t.is_floating_point():
            total = sum(w * v.float() for w, v in held)
            model[k] = (total / sum(w for w, _ in held)).to(t.dtype)
        else:
            model[k] = t
    return model


def soup(args) -> None:
    checkpoints = args.checkpoints
    weights = [1.0] * len(checkpoints)
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
    states = [
        torch.load(c, map_location="cpu", weights_only=False) for c in checkpoints
    ]
    first = checked_first(states, checkpoints)
    model = averaged(states, weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = {k: v for k, v in first.items() if k not in TRAINING_STATE}
    payload = {
        **kept,
        "model": model,
        "soup": [str(c) for c in checkpoints],
        "soup_weights": weights,
    }
    torch.save(payload, args.out)
    LOG.info(f"{args.out}: {len(model)} tensors from {len(states)} checkpoints")


def add_parser(commands, common) -> None:
    parser = commands.add_parser(
        "soup",
        parents=[common],
        help="average the weights of students trained from the same start",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--weights", help="comma-separated weights, equal by default")
    parser.add_argument(
        "checkpoints",
        nargs="+",
        type=Path,
        help="the first a whole checkpoint, the rest whole or partial snapshots",
    )
    parser.set_defaults(run=soup)
