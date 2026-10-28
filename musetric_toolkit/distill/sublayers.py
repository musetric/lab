import torch

from musetric_toolkit.distill.pooled import BANDS


class BandMix(torch.nn.Module):
    def __init__(self, a, b, bands: int, rank: int = 64, heads: int = 4):
        super().__init__()
        dim = a.shape[0]
        self.heads = heads
        self.linear = torch.nn.Linear(dim, dim)
        self.down = torch.nn.Linear(dim, rank, bias=False)
        self.mix = torch.nn.Parameter(torch.zeros(heads, bands, bands))
        self.up = torch.nn.Linear(rank, dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(a.T)
            self.linear.bias.copy_(b)
            torch.nn.init.orthogonal_(self.down.weight)
            torch.nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, x):
        z = self.down(x)
        z = z.reshape(*z.shape[:2], self.heads, -1)
        z = torch.einsum("hgf,nfhc->nghc", self.mix.to(z.dtype), z).flatten(2)
        return self.linear(x) + self.up(z)


def target(model, spec: str):
    kind, pair, part = spec[0], int(spec[1]), spec[2:]
    transformer = model.layers[pair][0 if kind == "t" else 1]
    attn, ff = transformer.layers[0]
    return {"": transformer, "a": attn, "f": ff}[part], transformer, part


def drop(module, transformer, part: str) -> None:
    if part == "":
        transformer.forward = lambda x: x
    else:
        module.forward = torch.zeros_like


def install(model, spec: str, fit) -> None:
    where, how = spec.split(":")
    module, transformer, part = target(model, where)
    if how == "drop":
        drop(module, transformer, part)
        return
    a, b = fit
    device = next(model.parameters()).device
    if how == "mix":
        keep = model.band_merges[int(where[1])]
        bands = keep + (BANDS - keep) // 2 if keep else BANDS
        linear = BandMix(a, b, bands).to(device)
    else:
        linear = torch.nn.Linear(a.shape[0], a.shape[1]).to(device)
        with torch.no_grad():
            linear.weight.copy_(a.T)
            linear.bias.copy_(b)
    if not hasattr(model, "sublayer_maps"):
        model.sublayer_maps = torch.nn.ModuleDict()
    model.sublayer_maps[where] = linear
    if part == "":
        transformer.forward = lambda x, lin=linear: x + lin(x)
    else:
        module.forward = linear.__call__
