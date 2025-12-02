import torch
from codebook import Codebook

class Quantizer:
    def __init__(self, codebook:Codebook):
        self.codebook = codebook
        pass

    def _normilize_weights(self, weights:torch.Tensor)->torch.Tensor:
        """Normalize the input tensor."""
        return (weights - weights.mean()) / weights.std(), weights.mean(), weights.std()

    def quantize(self, weights:torch.Tensor)->torch.Tensor:
        """Quantize the input tensor.  Return indicies of the codebooks, original mean and std."""
        normalized_weights, mean, std = self._normilize_weights(weights)

        indices = self.codebook.quantize(normalized_weights)

        return indices, mean, std

    def _quantize_column(self, column:torch.Tensor)->torch.Tensor:
        """Quantize a single column of the input tensor."""


    def dequantize(self, quant_weights: torch.Tensor,
                         indices: torch.Tensor,
                         original_mean: torch.Tensor,
                         original_std: torch.Tensor)->torch.Tensor:
        """Dequantize the input tensor."""
        pass

    def save_quantizer(self, path:str):
        """Save the quantizer to a file."""
        pass