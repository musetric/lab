import sys

import librosa
import numpy as np

SAMPLE_RATE = 22050
HOP = 512
WINDOW_SECONDS = 4
PULSE_CONTEXT_SECONDS = 2
MIN_PULSE_SECONDS = 0.25
MAX_PULSE_SECONDS = 1.2


def level(signal: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(signal**2)) + 1e-9))


def pulses(signal: np.ndarray) -> list[float]:
    envelope = librosa.onset.onset_strength(y=signal, sr=SAMPLE_RATE, hop_length=HOP)
    correlation = librosa.autocorrelate(envelope - envelope.mean())
    lags = np.arange(len(correlation)) * HOP / SAMPLE_RATE
    keep = (lags > MIN_PULSE_SECONDS) & (lags < MAX_PULSE_SECONDS)
    best = np.argsort(correlation[keep])[::-1][:2]
    return [round(float(lags[keep][index]), 3) for index in best]


def main(args) -> None:
    load = {
        kind: librosa.load(path, sr=SAMPLE_RATE, mono=True)[0]
        for kind, path in (
            ("instrumental", args.instrumental),
            ("vocals", args.vocals),
            ("mix", args.mix),
        )
    }
    harmonic, percussive = librosa.effects.hpss(load["instrumental"])
    lines = [
        "window    instr  vocal   perc   harm | pulses: instr | perc | mix",
    ]
    context = PULSE_CONTEXT_SECONDS * SAMPLE_RATE
    for start in range(int(args.start), int(args.end), WINDOW_SECONDS):
        a = start * SAMPLE_RATE
        b = (start + WINDOW_SECONDS) * SAMPLE_RATE
        levels = [
            level(load["instrumental"][a:b]),
            level(load["vocals"][a:b]),
            level(percussive[a:b]),
            level(harmonic[a:b]),
        ]
        lags = [
            pulses(signal[max(0, a - context) : b + context])
            for signal in (load["instrumental"], percussive, load["mix"])
        ]
        lines.append(
            f"{start:4d}-{start + WINDOW_SECONDS:<4d}"
            + " ".join(f"{value:6.1f}" for value in levels)
            + " | "
            + " | ".join(str(lag) for lag in lags)
        )
    sys.__stdout__.buffer.write(("\n".join(lines) + "\n").encode("utf-8"))
    sys.__stdout__.flush()
