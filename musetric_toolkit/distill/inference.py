import numpy as np
import torch

from musetric_toolkit.distill.precision import PRECISION
from musetric_toolkit.distill.teacher import CHUNK, STEP


def student_forward(model, mix: torch.Tensor) -> torch.Tensor:
    stft = model.encode_stft(mix)
    with torch.autocast("cuda", dtype=PRECISION.dtype):
        masks = model.net_forward(stft)
    return model.decode_istft(stft, masks.float(), length=mix.shape[-1])


def infer_overlap(model, mix: np.ndarray) -> np.ndarray:
    length = mix.shape[1]
    padded = np.pad(mix, ((0, 0), (0, max(0, CHUNK - length))))
    x = torch.from_numpy(padded).cuda()
    total = x.shape[1]
    starts = sorted({min(i, total - CHUNK) for i in range(0, total, STEP)})
    window = torch.hamming_window(CHUNK, device="cuda")
    acc = torch.zeros_like(x)
    norm = torch.zeros(total, device="cuda")
    model.eval()
    with torch.no_grad():
        for group in range(0, len(starts), 2):
            batch = starts[group : group + 2]
            windows = torch.stack([x[:, s : s + CHUNK] for s in batch])
            for s, part in zip(batch, student_forward(model, windows), strict=True):
                acc[:, s : s + CHUNK] += part * window
                norm[s : s + CHUNK] += window
    model.train()
    return (acc / norm.clamp(min=1e-10))[:, :length].cpu().numpy()
