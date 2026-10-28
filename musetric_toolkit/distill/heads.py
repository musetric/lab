import torch

HEAD = 64


def attentions(model):
    return [
        attn
        for pair in model.layers
        for transformer in pair
        for attn, _ in transformer.layers
    ]


def prune_heads(model, fits: dict) -> None:
    for i, attn in enumerate(attentions(model)):
        if i not in fits:
            continue
        kept, weight, bias = fits[i]
        heads, device = attn.heads, attn.to_qkv.weight.device
        rows = torch.tensor(
            [
                part * heads * HEAD + h * HEAD + d
                for part in range(3)
                for h in kept
                for d in range(HEAD)
            ],
            device=device,
        )
        qkv = torch.nn.Linear(attn.to_qkv.in_features, len(rows), bias=False)
        gates = torch.nn.Linear(attn.to_gates.in_features, len(kept))
        out = torch.nn.Linear(len(kept) * HEAD, weight.shape[0])
        qkv, gates, out = qkv.to(device), gates.to(device), out.to(device)
        with torch.no_grad():
            qkv.weight.copy_(attn.to_qkv.weight[rows])
            gates.weight.copy_(attn.to_gates.weight[kept])
            gates.bias.copy_(attn.to_gates.bias[kept])
            out.weight.copy_(weight)
            out.bias.copy_(bias)
        pairs = ((qkv, attn.to_qkv), (gates, attn.to_gates), (out, attn.to_out[0]))
        for new, old in pairs:
            new.requires_grad_(old.weight.requires_grad)
        attn.to_qkv, attn.to_gates, attn.heads = qkv, gates, len(kept)
        attn.to_out[0] = out
