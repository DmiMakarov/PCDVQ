import logging
import math
import torch
from codebooks import Codebook
from standart_requlazition import RandomizedHadamard
from collections.abc import Callable
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class Quantizer:
    def __init__(self, codebook:Codebook, k: int=8):
        self.k = k
        self.codebook = codebook

    def load_codebooks(self, path:str):
        """Load the codebooks from a file."""
        self.codebook.load_codebooks(path)
        logger.info("Loaded codebooks successfully")

    def reshape_weights(self, weights:torch.Tensor, pad_value:float=0.0)->torch.Tensor:
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


    def unreshape_weights(self, weights:torch.Tensor, p:int, q:int)->torch.Tensor:
        """Unreshape the input tensor to the shape of the original weights."""
        n = p * q
        flat = weights.reshape(-1)

        return flat[:n].reshape(p, q)

    def to_polar(self, weights:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        """Convert the input tensor to polar coordinates."""
        num_vectors, k = weights.shape
        magnitudes = torch.linalg.vector_norm(weights, dim=1, keepdim=True)
        phis = torch.empty(num_vectors, k-1, dtype=weights.dtype, device=weights.device)
        x_squares = weights * weights
        x_squares_flipped = torch.flip(x_squares, dims=[1])
        x_squares_cumsum_flipped = torch.cumsum(x_squares_flipped, dim=1)
        x_squares_cumsum = torch.flip(x_squares_cumsum_flipped, dims=[1])
        y_head = x_squares_cumsum[:, 1:-1].sqrt()
        x_head = weights[:, :-2]
        phis[:, :-1] = torch.atan2(y_head, x_head)
        phi_last = torch.atan2(weights[:, -1], weights[:, -2])
        phi_last = (phi_last + 2 * math.pi) % (2 * math.pi)
        phis[:, -1] = phi_last

        return phis, magnitudes.squeeze(-1)

    def get_unit_directions(self, directions: torch.Tensor) -> torch.Tensor:
        sin_matrix = torch.sin(directions)
        cos_matrix = torch.cos(directions)

        # cum_sin[i, j] = sin(directions[i, 0]) * ... * sin(directions[i, j])
        cum_sin = torch.cumprod(sin_matrix, dim=-1)
        # sin_prefix[i] = [1, sin(directions[i, 0]), sin(directions[i, 0]) * sin(directions[i, 1]), ...,
        # sin(directions[i, 0]) * ... * sin(directions[i, k-2])]
        sin_prefix = torch.cat([torch.ones_like(sin_matrix[:, :1]), cum_sin[:, :-1]], dim=-1)

        x_except_last = sin_prefix * cos_matrix
        x_last = cum_sin[:, -1:].clone()
        unitary_directions = torch.cat([x_except_last, x_last], dim=-1)

        return unitary_directions

    def to_cartesian(self, phis: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if r.ndim == 1:
            r = r.unsqueeze(-1)

        unit_dirs = self.get_unit_directions(phis)
        x = r * unit_dirs

        return x


    def quantize_inplace(self, weights:torch.Tensor, chunk_size: int = 1024, phi_chunk_size: int = 1024)->torch.Tensor:
        """Quantize the input tensor.  Return indicies of the codebooks, original mean and std."""
        device = weights.device
        orig_dtype = weights.dtype
        # Work in float32 for stability, then cast back.
        work_weights = weights.detach().to(dtype=torch.float32)

        q_min = work_weights.min().item()
        q_max = work_weights.max().item()
        q_mean = work_weights.mean().item()
        q_std = work_weights.std(unbiased=False).item()
        logger.info(
            f"Original weights stats - min: {q_min:.4e}, max: {q_max:.4e}, "
            f"mean: {q_mean:.4e}, std: {q_std:.4e}"
        )

        logger.info(f"Quantizing weights of shape {weights.shape} with device {device} and dtype {orig_dtype}")
        reshaped_weights = self.reshape_weights(work_weights)
        randomized_hadamard = RandomizedHadamard(reshaped_weights.shape[1], device=device, dtype=torch.float32)
        standardized_weights = randomized_hadamard.forward(reshaped_weights)

        phis, magnitudes = self.to_polar(standardized_weights)

        C_phis = self.codebook.codebook_direction.to(device=device, dtype=torch.float32)
        C_magnitudes = self.codebook.codebook_magnitude.to(device=device, dtype=torch.float32)

        logger.info(
            f"Quantization shapes - phis: {phis.shape}, C_phis: {C_phis.shape}, "
            f"magnitudes: {magnitudes.shape}, C_magnitudes: {C_magnitudes.shape}, "
            f"chunk_size: {chunk_size}, phi_chunk_size: {phi_chunk_size}"
        )

        idx_dir = find_best_index_chunk(
            phis,
            C_phis,
            chunk_size=chunk_size,
            phi_chunk_size=phi_chunk_size,
        )

        d = (magnitudes.view(-1, 1) - C_magnitudes.view(1, -1)).abs()
        idx_rad = d.argmin(dim=1)

        weights_q = self.to_cartesian(C_phis[idx_dir], C_magnitudes[idx_rad].unsqueeze(1))
        unnormalized_weights_q = randomized_hadamard.reverse(weights_q)

        if not torch.isfinite(unnormalized_weights_q).all():
            bad = torch.isfinite(unnormalized_weights_q) == False
            logger.warning(f"Found non-finite values in quantized weights; count={bad.sum().item()}")
        else:
            # Lightweight stats for debugging stability issues.
            q_min = unnormalized_weights_q.min().item()
            q_max = unnormalized_weights_q.max().item()
            q_mean = unnormalized_weights_q.mean().item()
            q_std = unnormalized_weights_q.std(unbiased=False).item()
            logger.info(
                f"Quantized weights stats - min: {q_min:.4e}, max: {q_max:.4e}, "
                f"mean: {q_mean:.4e}, std: {q_std:.4e}"
            )

        unreshaped_weights_q = self.unreshape_weights(unnormalized_weights_q, weights.shape[0], weights.shape[1])

        return unreshaped_weights_q.to(dtype=orig_dtype)

def find_best_index_chunk(
    phis: torch.Tensor,
    C_phis: torch.Tensor,
    chunk_size: int = 512,
    phi_chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Find best matching indices in C_phis for each row in phis.

    This function chunks over BOTH:
      - rows of phis (size N) with phi_chunk_size, and
      - rows of C_phis (size M) with chunk_size,
    so the largest similarity tensor in memory is only
    [phi_chunk_size, chunk_size].
    """
    N = phis.shape[0]
    M = C_phis.shape[0]

    best_idx = torch.empty(N, dtype=torch.long, device=phis.device)

    for n_start in tqdm(range(0, N, phi_chunk_size), desc="phi chunks"):
        n_end = min(n_start + phi_chunk_size, N)

        best_sim_chunk = None
        best_idx_chunk = None

        for m_start in range(0, M, chunk_size):
            m_end = min(m_start + chunk_size, M)
            Z_chunk = C_phis[m_start:m_end]  # [C, D]

            # [B, C] similarities for this (phis, C_phis) chunk
            sim_chunk = phis[n_start:n_end] @ C_phis[m_start:m_end].T

            # best inside this codebook chunk
            sim_vals, idx_local = sim_chunk.max(dim=1)  # [B], [B]

            if best_sim_chunk is None:
                best_sim_chunk = sim_vals
                best_idx_chunk = m_start + idx_local
            else:
                better = sim_vals > best_sim_chunk
                best_idx_chunk[better] = m_start + idx_local[better]
                best_sim_chunk[better] = sim_vals[better]

        best_idx[n_start:n_end] = best_idx_chunk

    return best_idx


@torch.no_grad()
def quantize_linear_inplace(module,
                            *,
                            quantizer: Quantizer,
                            filter_fn: Callable,
                            prefix: str = "",
                            chunk_size=512,
                            phi_chunk_size: int = 1024):
    """Quantize each linear layer in the input module inplace."""
    for child_name, child in list(module.named_children()):

        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, torch.nn.Linear) and filter_fn(full_name, child):
            logger.info("Quantizing linear layers with PCDVQ...")
            W = child.weight.detach().clone()
            Wq = quantizer.quantize_inplace(W, chunk_size=chunk_size, phi_chunk_size=phi_chunk_size)
            if not torch.isfinite(Wq).all():
                logger.warning(f"Skipping update for layer {child} due to non-finite quantized weights")
            else:
                child.weight.copy_(Wq)
        else:
            quantize_linear_inplace(child, quantizer=quantizer, filter_fn=filter_fn, prefix=full_name, chunk_size=chunk_size, phi_chunk_size=phi_chunk_size)
