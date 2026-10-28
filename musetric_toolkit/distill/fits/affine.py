import torch

from musetric_toolkit.distill.fits.crops import TrainingCrops
from musetric_toolkit.distill.fits.least_squares import with_constant
from musetric_toolkit.distill.pooled import DIM


def fit_affine(
    model, module, whole: bool, crops: TrainingCrops, ridge: float
) -> tuple[torch.Tensor, float]:
    gram = torch.zeros(DIM + 1, DIM + 1, dtype=torch.float64, device="cuda")
    cross = torch.zeros(DIM + 1, DIM, dtype=torch.float64, device="cuda")
    energy = [0.0]

    def record(_module, inputs, output):
        x = inputs[0].detach().float().reshape(-1, DIM)
        y = output.detach().float().reshape(-1, DIM)
        if whole:
            y = y - x
        xa = with_constant(x)
        gram.add_((xa.T @ xa).double())
        cross.add_((xa.T @ y).double())
        energy[0] += float((y.double() ** 2).sum())

    handle = module.register_forward_hook(record)
    crops.run(model)
    handle.remove()
    lam = ridge * gram.diagonal()[:DIM].mean()
    reg = torch.eye(DIM + 1, dtype=gram.dtype, device=gram.device) * lam
    reg[DIM, DIM] = 0
    w = torch.linalg.solve(gram + reg, cross)
    residual = float((energy[0] - (cross * w).sum()) / max(energy[0], 1e-12))
    return w.float(), residual
