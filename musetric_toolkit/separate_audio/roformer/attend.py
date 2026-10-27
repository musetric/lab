import logging
import math
import warnings

import torch
from torch import nn
from torch.nn import functional
from torch.nn.attention import SDPBackend

_backends_logged = {"value": False}


def row_chunks(batch: int, rows_per_batch: int, row_limit: int) -> int:
    if rows_per_batch > row_limit:
        raise ValueError(
            f"one batch row already spans {rows_per_batch} rows, past {row_limit}"
        )
    parts = math.ceil(batch * rows_per_batch / row_limit)
    while math.ceil(batch / parts) * rows_per_batch > row_limit:
        parts += 1
    return parts


def log_selected_backend(q, k, v, backends):
    if not _backends_logged["value"]:
        backend_names = [b.name for b in backends]
        logging.debug(f"Available SDPA backends for attention: {backend_names}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for backend in backends:
                try:
                    with torch.nn.attention.sdpa_kernel([backend]):
                        functional.scaled_dot_product_attention(
                            q[:1], k[:1], v[:1], dropout_p=0.0
                        )
                    logging.debug(f"Selected SDPA backend: {backend.name}")
                    break
                except Exception:
                    logging.debug(
                        "Failed to probe SDPA backend %s", backend.name, exc_info=True
                    )
                    continue
            else:
                logging.debug("Selected SDPA backend: MATH (fallback)")

        _backends_logged["value"] = True


class Attend(nn.Module):
    def __init__(self, dropout, flash):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.flash = flash
        self.q_block = 0
        self.softmax_rows = 0

    def flash_attn(self, q, k, v):
        backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        log_selected_backend(q, k, v, backends)

        with torch.nn.attention.sdpa_kernel(backends):
            return functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

    def forward(self, q, k, v):
        if self.flash:
            return self.flash_attn(q, k, v)

        if not _backends_logged["value"]:
            logging.debug("Using manual attention computation (matmul-based)")
            _backends_logged["value"] = True
        scale = q.shape[-1] ** -0.5

        if self.q_block and q.shape[-2] > self.q_block:
            outputs = []
            for start in range(0, q.shape[-2], self.q_block):
                q_part = q[..., start : start + self.q_block, :]
                sim = torch.matmul(q_part, k.transpose(-1, -2)) * scale
                attn = self.attn_dropout(self.softmax(sim))
                outputs.append(torch.matmul(attn, v))
            return torch.cat(outputs, dim=-2)

        sim = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = self.attn_dropout(self.softmax(sim))
        return torch.matmul(attn, v)

    def softmax(self, sim):
        if self.softmax_rows:
            rows_per_batch = math.prod(sim.shape[1:-1])
            parts = row_chunks(sim.shape[0], rows_per_batch, self.softmax_rows)
            if parts > 1:
                return torch.cat([c.softmax(dim=-1) for c in sim.chunk(parts)])
        return sim.softmax(dim=-1)
