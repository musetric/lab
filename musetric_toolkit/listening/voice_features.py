import librosa
import numpy as np
from scipy.signal import stft

from musetric_toolkit.listening.levels import RATE, SILENCE

ACTIVE_DB = -30.0
NFFT = 2048
HOP = 512
PITCH_RATE = 22050
PITCH_HOP = 220
PITCH_LOW = 75
PITCH_HIGH = 1000
DOMINANT = 0.5
STILL_SEMITONES = 0.03


def powers(vocals: np.ndarray, mix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, _, spectrum = stft(vocals.mean(0), RATE, nperseg=NFFT, noverlap=NFFT - HOP)
    _, _, spectrum_mix = stft(mix.mean(0), RATE, nperseg=NFFT, noverlap=NFFT - HOP)
    return np.abs(spectrum) ** 2, np.abs(spectrum_mix) ** 2


def dominance(vocals: np.ndarray, mix: np.ndarray) -> float:
    power, power_mix = powers(vocals, mix)
    mask = power / (power_mix + SILENCE)
    return float((power * (mask > DOMINANT)).sum() / (power.sum() + SILENCE))


def flatness_of(power: np.ndarray) -> float:
    total = power.sum(0)
    active = total > total.max() * 10 ** (ACTIVE_DB / 10)
    band = (power + SILENCE)[:, active]
    return float(np.median(np.exp(np.log(band).mean(0)) / band.mean(0)))


def side_to_mid(vocals: np.ndarray) -> float:
    mid = (vocals[0] + vocals[1]) / 2
    side = (vocals[0] - vocals[1]) / 2
    return float(
        10 * np.log10(((side**2).sum() + SILENCE) / ((mid**2).sum() + SILENCE))
    )


def percussive_share(mono: np.ndarray) -> float:
    harmonic, percussive = librosa.effects.hpss(np.asarray(mono, dtype=np.float32))
    part = (percussive**2).sum()
    return float(part / ((harmonic**2).sum() + part + SILENCE))


def pitch_features(mono: np.ndarray) -> dict:
    small = librosa.resample(
        np.asarray(mono, dtype=np.float32), orig_sr=RATE, target_sr=PITCH_RATE
    )
    f0, voiced, _ = librosa.pyin(
        small,
        fmin=PITCH_LOW,
        fmax=PITCH_HIGH,
        sr=PITCH_RATE,
        frame_length=NFFT,
        hop_length=PITCH_HOP,
    )
    frame_power = (
        librosa.feature.rms(y=small, frame_length=NFFT, hop_length=PITCH_HOP)[0] ** 2
    )
    frames = min(len(f0), len(frame_power))
    f0, voiced, frame_power = f0[:frames], voiced[:frames], frame_power[:frames]
    loud = frame_power > frame_power.max() * 10 ** (ACTIVE_DB / 10)
    runs = voiced & np.concatenate([[False], voiced[:-1]])
    pitch = 12 * np.log2(np.nan_to_num(f0, nan=1.0) + 1e-9)
    motion = np.abs(np.diff(pitch, prepend=0.0))[runs]
    moved = len(motion) > 0
    return dict(
        voiced=float(voiced[loud].mean()) if loud.any() else 0.0,
        glide=float(np.median(motion)) if moved else 0.0,
        still=float((motion < STILL_SEMITONES).mean()) if moved else 0.0,
    )


def dsp_features(vocals: np.ndarray, mix: np.ndarray) -> dict:
    mono = vocals.mean(0)
    power, _ = powers(vocals, mix)
    return dict(
        dominant=dominance(vocals, mix),
        flatness=flatness_of(power),
        side_mid=side_to_mid(vocals),
        percussive=percussive_share(mono),
        **pitch_features(mono),
    )
