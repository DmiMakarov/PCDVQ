import logging
import torch
from torch import Tensor

from .nearest import find_nearest
from .codebooks import Codebook
from .normalization import RandomizedHadamard
from .utils import *
from collections.abc import Callable
from tqdm.auto import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Quantizer:
    def __init__(self, codebook: Codebook, k: int = 8, codebook_chunk_size: int = 1024, phi_chunk_size: int = 1024):
        self.k = k
        self.cb_chunk_size, self.phi_chunk_size = codebook_chunk_size, phi_chunk_size
        self.codebook = codebook

    def load_codebooks(self, path: str):
        """Load the codebooks from a file."""
        self.codebook.load_codebooks(path)
        logger.info("Loaded codebooks successfully")

    def reshape_weights(self, weights: Tensor, pad_value: float = 0.0) -> Tensor:
        """Reshape the input tensor to the shape of the codebooks."""
        p, q = weights.shape
        n = p * q
        rem = n % self.k

        flat = weights.reshape(-1)
        if rem != 0:
            pad_elems = self.k - rem
            pad = flat.new_full((pad_elems,), pad_value)
            flat = torch.cat([flat, pad], dim=0)

        return flat.view(-1, self.k)

    def unreshape_weights(self, weights: Tensor, p: int, q: int) -> Tensor:
        """Unreshape the input tensor to the shape of the original weights."""
        n = p * q
        flat = weights.reshape(-1)

        return flat[:n].reshape(p, q)

    def _log_stats(self, weights: Tensor, prefix="Original"):
        q_min = weights.min().item()
        q_max = weights.max().item()
        q_mean = weights.mean().item()
        q_std = weights.std(unbiased=False).item()
        logger.info(f"{prefix} weights stats - min: {q_min:.4e}, max: {q_max:.4e}, " f"mean: {q_mean:.4e}, std: {q_std:.4e}")

    def quantize(self, weights: Tensor, chunk_size: int = None, phi_chunk_size: int = None) -> Tensor:
        """Quantize the input tensor. Return indicies of the codebooks, original mean and std."""
        chunk_size = chunk_size or self.cb_chunk_size
        phi_chunk_size = phi_chunk_size or self.phi_chunk_size

        device, orig_dtype = weights.device, weights.dtype
        # Work in float32 for stability, then cast back.
        work_weights = weights.to(dtype=torch.float32)

        reshaped_weights = self.reshape_weights(work_weights)

        self._log_stats(work_weights)
        logger.info(f"Quantizing weights of shape {weights.shape} with device {device} and dtype {orig_dtype}")

        randomized_hadamard = RandomizedHadamard(reshaped_weights.shape[1], device=device, dtype=torch.float32)
        standardized_weights, scale = randomized_hadamard(reshaped_weights)

        phis, magnitudes = to_polar(standardized_weights)

        C_phis = self.codebook.directions.to(device=device)
        C_magnitudes = self.codebook.magnitudes.to(device=device)

        logger.info(
            f"Quantization shapes - phis: {phis.shape}, codebook_phis: {C_phis.shape}, "
            f"magnitudes: {magnitudes.shape}, codebook_magnitudes: {C_magnitudes.shape}, "
            f"chunk_size: {chunk_size}, phi_chunk_size: {phi_chunk_size}"
        )
        # handle direction
        phis_unit_vec = to_unit_vectors(phis)
        C_unit_vec = to_unit_vectors(C_phis)

        idx_dir = find_nearest(
            phis_unit_vec,
            C_unit_vec,
            batch_size=chunk_size,
            codebook_batch_size=phi_chunk_size,
        )
        # handle magnitude
        d = (magnitudes.view(-1, 1) - C_magnitudes.view(1, -1)).abs()
        idx_rad = d.argmin(dim=1)

        ## reconstruction

        weights_q = to_cartesian(C_phis[idx_dir], C_magnitudes[idx_rad].unsqueeze(1))
        unnormalized_weights_q = randomized_hadamard.reverse(weights_q, scale)

        if not torch.isfinite(unnormalized_weights_q).all():
            bad = torch.isfinite(unnormalized_weights_q) == False
            logger.warning(f"Found non-finite values in quantized weights; count={bad.sum().item()}")
        else:
            self._log_stats(unnormalized_weights_q, "Quantized")

        unreshaped_weights_q = self.unreshape_weights(unnormalized_weights_q, weights.shape[0], weights.shape[1])

        return unreshaped_weights_q.to(dtype=orig_dtype)


@torch.no_grad()
def quantize_linear_inplace(
    model, quantizer: Quantizer, filter_fn: Callable = None, chunk_size: int = 512, phi_chunk_size: int = 1024
):
    """Quantize each linear layer in the input model inplace."""
    for nm, m in get_linear_layers(model, filter_fn).items():
        logger.info(f"Quantizing layer {nm} with PCDVQ...")
        w = m.weight.detach().clone()
        wq = quantizer.quantize(w, chunk_size=chunk_size, phi_chunk_size=phi_chunk_size)

        if not torch.isfinite(wq).all():
            logger.warning(f"Skipping update for {nm} due to non-finite weights")
        else:
            m.weight.copy_(wq)
