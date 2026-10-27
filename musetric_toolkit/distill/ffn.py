import torch


def ffn_layers(model) -> list:
    return [
        ff
        for pair in model.layers
        for transformer in pair
        for _, ff in transformer.layers
    ]


def kept_units(chosen, stored, i, n):
    if stored.get("select") == "obs":
        chosen = [stored["kept"].get(j, rank) for j, rank in enumerate(chosen)]
    return sorted(chosen[i][:n]) if chosen else list(range(n))


def narrow_ffn(model, keep: float, pairs, which, fits) -> None:
    chosen, means, stored = fits
    fitted = stored.get("refit")
    for i, ff in enumerate(ffn_layers(model)):
        if (pairs is not None and i // 2 not in pairs) or "tb"[i % 2] not in which:
            continue
        up, down = ff.net[1], ff.net[4]
        n = round(up.out_features * keep)
        device = up.weight.device
        idx = torch.tensor(kept_units(chosen, stored, i, n), device=device)
        new_up = torch.nn.Linear(up.in_features, n).to(device)
        new_down = torch.nn.Linear(n, down.out_features).to(down.weight.device)
        with torch.no_grad():
            new_up.weight.copy_(up.weight[idx])
            new_up.bias.copy_(up.bias[idx])
            new_down.weight.copy_(down.weight[:, idx])
            new_down.bias.copy_(down.bias)
            if means is not None:
                dropped = torch.ones(up.out_features, dtype=torch.bool, device=device)
                dropped[idx] = False
                mean = torch.tensor(means[i], device=down.weight.device)
                new_down.bias += down.weight[:, dropped] @ mean[dropped]
            if fitted is not None:
                weight, bias = fitted[i]
                new_down.weight.copy_(weight)
                new_down.bias.copy_(bias)
        new_up.kept = new_down.kept = idx
        ff.net[1], ff.net[4] = new_up, new_down
