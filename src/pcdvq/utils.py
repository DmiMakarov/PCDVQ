from typing import Callable
import torch.nn as nn
import torch
from torch import Tensor


def get_linear_layers(model: nn.Module, filter_fn: Callable = None):
    """Return linear layers given the filter function"""
    if filter_fn is None:
        filter_fn = lambda *_: True
    return {
        name: module
        for name, module in model.named_modules()
        if filter_fn(name.rsplit(".", 1)[1] if '.' in name else name) and isinstance(module, nn.Linear)
    }


### Vector functions

def to_unit_vectors(angles: Tensor) -> Tensor:
    '''Converts vector of angles to cartesian vectors on unit sphere'''
    sin_matrix = torch.sin(angles)
    cos_matrix = torch.cos(angles)

    cum_sin = torch.cumprod(sin_matrix, dim=-1)
    sin_prefix = torch.cat([torch.ones_like(sin_matrix[:, :1]), cum_sin[:, :-1]], dim=-1)

    x_except_last = sin_prefix * cos_matrix
    x_last = cum_sin[:, -1:].clone()

    return torch.cat([x_except_last, x_last], dim=-1)

def to_cartesian(phis: Tensor, r: Tensor) -> Tensor:
    """
    Convert tensor from polar to cartesian coordinates
    """
    if r.ndim == 1:
        r = r.unsqueeze(-1)

    unit_dirs = to_unit_vectors(phis)
    return r * unit_dirs

def to_polar(x: Tensor) -> tuple[Tensor, Tensor]:
    """
    Convert tensor of vectors to polar coordinates.
    Returns:
        (phis, r): tuple of tensors with angles and magnitudes
    """
    n, k = x.shape
    x_squares_flipped = torch.flip(x.pow(2), dims=[1])
    suffix_sum_squares = torch.flip(torch.cumsum(x_squares_flipped, dim=1), dims=[1])
    r = suffix_sum_squares[:, 0].sqrt()

    phis = torch.empty(n, k - 1, dtype=x.dtype, device=x.device)
    y_head = suffix_sum_squares[:, 1:-1].sqrt()
    x_head = x[:, :-2]
    phis[:, :-1] = torch.atan2(y_head, x_head)
    phi_last = torch.atan2(x[:, -1], x[:, -2])
    phis[:, -1] = phi_last.remainder(2 * torch.pi)

    return phis, r
