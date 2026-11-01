import soundfile as sf
import torch
import torchaudio


def load_skey_audio(path: str, sample_rate: int) -> torch.Tensor:
    samples, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy())
    if source_rate != sample_rate:
        waveform = torchaudio.transforms.Resample(source_rate, sample_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform
