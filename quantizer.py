import logging
import math
import torch
from codebooks import Codebook
from standart_requlazition import RandomizedHadamard

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


    def quantize_inplace(self, weights:torch.Tensor, chunk_size=512)->torch.Tensor:
        """Quantize the input tensor.  Return indicies of the codebooks, original mean and std."""
        device = weights.device
        dtype = weights.dtype
        reshaped_weights = self.reshape_weights(weights)
        randomized_hadamard = RandomizedHadamard(reshaped_weights.shape[0], device=device, dtype=dtype)
        standardized_weights = randomized_hadamard.forward(reshaped_weights)

        phis, magnitudes = self.to_polar(standardized_weights)

        C_phis = self.codebook.codebook_direction.to(device=device, dtype=dtype)
        C_magnitudes = self.codebook.codebook_magnitude.to(device=device, dtype=dtype)

        idx_dir = find_best_index_chunk(phis, C_phis)
        d = (magnitudes.view(-1, 1) - C_magnitudes.view(1, -1)).abs()
        idx_rad = d.argmin(dim=1)

        phis_q = C_phis[idx_dir]

        r_q = C_magnitudes[idx_rad].unsqueeze(1)

        weights_q = self.to_cartesian(phis_q, r_q)
        unnormalized_weights_q = randomized_hadamard.reverse(weights_q)

        unreshaped_weights_q = self.unreshape_weights(unnormalized_weights_q, weights.shape[0], weights.shape[1])

        return unreshaped_weights_q

def find_best_index(phis: torch.Tensor, C_phis: torch.Tensor)->torch.Tensor:
    best_idxs = []

    for i in range(phis.shape[0]):
        best_idx = None
        best_sim = None

        for j in range(C_phis.shape[0]):
            sim = torch.dot(phis[i], C_phis[j])

            if best_sim is None or sim > best_sim:
                best_sim = sim
                best_idx = j

        best_idxs.append(best_idx)

    return torch.tensor(best_idxs)

def find_best_index_chunk(phis: torch.Tensor, C_phis: torch.Tensor, chunk_size: int=512)->torch.Tensor:
    N = phis.shape[0]
    M = C_phis.shape[0]

    best_sim = None                  # [N]
    best_idx = torch.empty(N, dtype=torch.long, device=phis.device)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        Z_chunk = C_phis[start:end]       # [C, D]

        # [N, C] similarities for this chunk
        sim_chunk = phis @ Z_chunk.T

        # best inside this chunk
        sim_vals, idx_chunk = sim_chunk.max(dim=1)  # [N], [N]

        if best_sim is None:
            best_sim = sim_vals
            best_idx = start + idx_chunk
        else:
            better = sim_vals > best_sim
            best_idx[better] = start + idx_chunk[better]
            best_sim[better] = sim_vals[better]

    return best_idx

def quantize_linear_inplace(module, *, quantizer: Quantizer):
    """Quantize each linear layer in the input module inplace."""
    logger.info("Quantizing linear layers with PCDVQ...")
    for _, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            with torch.no_grad():
                W = child.weight.data.detach().to("cpu")
                Wq = quantizer.quantize_inplace(W).to(device=child.weight.device, dtype=child.weight.dtype)
                child.weight.data.copy_(Wq)
        else:
            quantize_linear_inplace(child, quantizer=quantizer)
