import copy

import torch

from musetric_toolkit.distill.pooled import down, up
from musetric_toolkit.distill.precision import PRECISION


def keep_teacher_head(model) -> None:
    if not hasattr(model, "teacher_heads"):
        heads = copy.deepcopy(model.mask_estimators).to(PRECISION.dtype)
        object.__setattr__(model, "teacher_heads", heads)


def fitted_linear(weight, bias, device, trainable: bool) -> torch.nn.Linear:
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0]).to(device)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    linear.requires_grad_(trainable)
    return linear


def narrow_head(model, fits, trainable: bool) -> None:
    heads = copy.deepcopy(model.mask_estimators).to(PRECISION.dtype)
    object.__setattr__(model, "teacher_heads", heads)
    for band, seq in enumerate(model.mask_estimators[0].to_freqs):
        mlp = seq[0]
        _, w1, b1, w2, b2, _, w3, b3 = fits[band]
        for at, (weight, bias) in ((0, (w1, b1)), (2, (w2, b2)), (4, (w3, b3))):
            mlp[at] = fitted_linear(weight, bias, mlp[at].weight.device, trainable)


def balanced(coeff, out):
    rank = coeff.shape[0]
    u, s, vh = torch.linalg.svd(out.double() @ coeff.double(), full_matrices=False)
    root = s[:rank].sqrt()
    return (root[:, None] * vh[:rank]).float(), (u[:, :rank] * root).float()


def low_rank_head(model, fits, per_band: bool, trainable: bool) -> None:
    keep_teacher_head(model)
    shared = None
    for band, seq in enumerate(model.mask_estimators[0].to_freqs):
        mlp = seq[0]
        coeff, out = fits[band]
        if per_band:
            coeff, out = balanced(coeff, out)
        device = mlp[2].weight.device
        reduce = torch.nn.Linear(coeff.shape[1] - 1, coeff.shape[0]).to(device)
        if shared is None or per_band:
            expand = torch.nn.Linear(out.shape[1], out.shape[0], bias=False).to(device)
            with torch.no_grad():
                expand.weight.copy_(out)
            shared = expand
        with torch.no_grad():
            reduce.weight.copy_(coeff[:, :-1])
            reduce.bias.copy_(coeff[:, -1])
        mlp[2] = torch.nn.Sequential(reduce, shared)
        mlp.requires_grad_(trainable)


def pooled_head(model, k: int) -> None:
    keep_teacher_head(model)
    estimator = model.mask_estimators[0]

    def forward(x, inner=estimator.forward):
        return up(inner(down(x, k, 1)), k, 1, x.shape[1])

    estimator.forward = forward
