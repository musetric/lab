from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from musetric_toolkit.listening.round_files import read_json

RULE_STUDENT = "student"
RULE_TEACHER = "teacher"

Rule = Callable[[float, float, float], float]


def frozen_rule(features: Path) -> Rule:
    rows = [row for row in read_json(features) if row["train"] and not row["excluded"]]
    x = np.array(
        [
            [
                row["dsp"][RULE_TEACHER]["dominant"],
                row["systems"][RULE_TEACHER]["vocals"],
                row["whisper"][RULE_STUDENT]["words"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.array(
        [
            row["present"]["lead"] is False and row["present"]["backing"] is False
            for row in rows
        ],
        dtype=int,
    )
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(x), y)

    def score(dominant: float, teacher: float, words: float) -> float:
        point = np.array([[dominant, teacher, words]], dtype=np.float64)
        return float(model.decision_function(scaler.transform(point))[0])

    return score
