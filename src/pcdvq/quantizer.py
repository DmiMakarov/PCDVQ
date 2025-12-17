import logging
import torch, torch.nn.functional as F
from torch import Tensor

from .nearest import find_nearest
from .codebooks import PCDVQCodebook
from .normalization import StandardRegularization, RandomizedHadamard
from .utils import *
from collections.abc import Callable
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


class Quantizer:
    def __init__(
        self,
        codebook: PCDVQCodebook,
        regularizer_cls: type[StandardRegularization] = RandomizedHadamard,
        codebook_chunk_size: int = 1024,
        phi_chunk_size: int = 1024,
        svd_rank: int = 0,
        device=default_device,
    ):
        self.svd_rank = svd_rank
        self.k, self.device = codebook.k, device
        self.regularizer_cls = regularizer_cls
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

        randomized_hadamard = self.regularizer_cls(cols, device=self.device)
        w_norm, scale = randomized_hadamard(w)

        w_blocks = self._to_blocks(w_norm)

        magnitudes = w_blocks.norm(dim=1, keepdim=True)  # (N, 1)

        ## Handle direction
        target_dirs = w_blocks / (magnitudes + 1e-8)

        idx_dir = find_nearest(
            target_dirs,
            self.codebook.directions,
            batch_size=chunk_size,
            codebook_batch_size=phi_chunk_size,
        )

        ## Handle magnitude
        d = (magnitudes - self.codebook.magnitudes.unsqueeze(0)).abs()
        idx_rad = d.argmin(dim=1)

        ## Reconstruction
        best_dirs = self.codebook.directions[idx_dir]
        best_mags = self.codebook.magnitudes[idx_rad].unsqueeze(1)
        w_q_blocks = best_dirs * best_mags

        w_q_norm = self._from_blocks(w_q_blocks, w_norm.shape)
        w_q = randomized_hadamard.reverse(w_q_norm, scale)
        
        if self.svd_rank > 0:
                    logger.info(f"Applying SVD correction with rank {self.svd_rank}...")
                    # Calculate Residual (Error matrix)
                    residual = w - w_q
                    
                    # Compute SVD (Topic 6)
                    # Use 'full_matrices=False' for economy SVD
                    try:
                        # torch.linalg.svd is standard, but svd_lowrank might be faster for very small k
                        # stick to standard for robustness on 0.6B model
                        U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
                        
                        # Construct low-rank approximation of the error
                        # C = U_k * Sigma_k * V_k^T
                        U_k = U[:, :self.svd_rank]
                        S_k = torch.diag(S[:self.svd_rank])
                        Vh_k = Vh[:self.svd_rank, :]
                        
                        correction = U_k @ S_k @ Vh_k
                        
                        # Update quantized weights
                        w_q = w_q + correction
                    except Exception as e:
                        logger.error(f"SVD Correction failed: {e}")

        
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
