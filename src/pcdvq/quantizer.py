import logging
import torch, torch.nn.functional as F
from torch import Tensor

from .nearest import find_nearest
from .codebooks import PCDVQCodebook
from .normalization import RandomizedHadamard
from .utils import *
from collections.abc import Callable
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


class Quantizer:
    def __init__(
        self, codebook: PCDVQCodebook, codebook_chunk_size: int = 1024, phi_chunk_size: int = 1024, device=default_device
    ):
        self.k, self.device = codebook.k, device
        self.batch_size, self.phi_batch_size = codebook_chunk_size, phi_chunk_size
        self.codebook = codebook.to(device)

    def _to_blocks(self, x: Tensor) -> Tensor:
        """Flatten input and reshape into blocks of k. Pads if necessary."""
        k = self.k
        flat = x.view(-1)
        if (rem := flat.numel() % k) != 0:
            flat = F.pad(flat, (0, k - rem))
        return flat.view(-1, k)

    def _from_blocks(self, x_blocks: Tensor, original_shape: torch.Size) -> Tensor:
        """Flatten blocks, crop padding, and reshape to original shape."""
        numel = original_shape.numel()
        return x_blocks.view(-1)[:numel].view(original_shape)

    def _log_stats(self, weights: Tensor, prefix="Original"):
        q_min = weights.min().item()
        q_max = weights.max().item()
        q_mean = weights.mean().item()
        q_std = weights.std(unbiased=False).item()
        logger.info(f"{prefix} weights stats - min: {q_min:.4e}, max: {q_max:.4e}, " f"mean: {q_mean:.4e}, std: {q_std:.4e}")

    def quantize(self, weights: Tensor, chunk_size: int = None, phi_chunk_size: int = None) -> Tensor:
        """Quantize the input tensor. Return indicies of the codebooks, original mean and std."""
        chunk_size = chunk_size or self.batch_size
        phi_chunk_size = phi_chunk_size or self.phi_batch_size

        orig_device, orig_dtype = weights.device, weights.dtype
        w = weights.float().to(self.device)
        _, cols = w.shape

        self._log_stats(w)

        randomized_hadamard = RandomizedHadamard(cols, device=self.device)
        w_norm, scale = randomized_hadamard(w)

        w_blocks = self._to_blocks(w_norm)

        phis, magnitudes = to_polar(w_blocks)

        cb_phis = self.codebook.directions

        ## handle direction
        target_dirs = to_unit_vectors(phis)
        cb_dirs = to_unit_vectors(cb_phis)

        idx_dir = find_nearest(
            target_dirs,
            cb_dirs,
            batch_size=chunk_size,
            codebook_batch_size=phi_chunk_size,
        )

        ## handle magnitude
        cb_mags = self.codebook.magnitudes
        d = (magnitudes.unsqueeze(1) - cb_mags.unsqueeze(0)).abs()
        idx_rad = d.argmin(dim=1)

        ## reconstruction
        w_q_blocks = to_cartesian(cb_phis[idx_dir], cb_mags[idx_rad].unsqueeze(1))

        w_q_norm = self._from_blocks(w_q_blocks, w_norm.shape)
        w_q = randomized_hadamard.reverse(w_q_norm, scale)

        self._log_stats(w_q, "Quantized")

        return w_q.to(dtype=orig_dtype, device=orig_device)


@torch.no_grad()
def quantize_linear_inplace(model, quantizer: Quantizer, filter_fn: Callable = None):
    """Quantize each linear layer in the input model inplace."""
    for nm, m in get_linear_layers(model, filter_fn).items():
        w = m.weight.detach().clone()
        logger.info(f"Quantizing layer {nm} | {w.shape} with device {w.device} and dtype {w.dtype}")
        wq = quantizer.quantize(w)

        if not torch.isfinite(wq).all():
            logger.warning(f"Skipping update for {nm} due to non-finite weights")
        else:
            m.weight.copy_(wq)
