from dataclasses import dataclass

import torch


@dataclass
class Precision:
    dtype: torch.dtype = torch.bfloat16
    recompute: bool = True


PRECISION = Precision()
