from typing import Callable
import torch.nn as nn


def get_linear_layers(model: nn.Module, filter_fn: Callable = None):
    """Return linear layers given the filter function"""
    if filter_fn is None:
        filter_fn = lambda *_: True
    return {
        name: module
        for name, module in model.named_modules()
        if filter_fn(name.rsplit(".", 1)[1] if '.' in name else name) and isinstance(module, nn.Linear)
    }
