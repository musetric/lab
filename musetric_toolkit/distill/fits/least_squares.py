import torch


def with_constant(h: torch.Tensor) -> torch.Tensor:
    return torch.cat([h, torch.ones(h.shape[0], 1, device=h.device)], 1)


def add_gram(gram: torch.Tensor, values: torch.Tensor) -> None:
    h = with_constant(values.detach().float().reshape(-1, values.shape[-1]))
    gram.add_((h.T @ h).double())


def ridge_refit(
    weight: torch.Tensor, gram: torch.Tensor, keep: torch.Tensor, ridge: float
) -> torch.Tensor:
    gkk = gram[keep][:, keep]
    lam = ridge * gkk.diagonal().mean()
    eye = torch.eye(len(keep), dtype=gram.dtype, device=gram.device)
    return weight @ gram[:, keep] @ torch.linalg.inv(gkk + lam * eye)


def output_error(weight: torch.Tensor, full: torch.Tensor, gram: torch.Tensor) -> float:
    d = weight - full
    missed = torch.einsum("oi,ij,oj->", d, gram, d)
    return float(torch.sqrt(missed / torch.einsum("oi,ij,oj->", full, gram, full)))
