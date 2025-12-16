import torch, scipy.stats
import numpy as np
from tqdm.auto import tqdm
from ..utils import default_device


def optimize_direction_codebook(candidates: torch.Tensor, n_bits: int = 14, device=default_device) -> torch.Tensor:
    """
    Selects 2^n_bits directions from candidates using a memory-efficient greedy algorithm.
    """
    n_target = 1 << n_bits
    if len(candidates) < n_target:
        raise ValueError(f"Not enough candidates ({len(candidates)}) for {n_bits} bits.")

    if n_target == len(candidates):
        return candidates.to(device)

    pool = candidates.to(device)

    # 1. Start with the first vector (arbitrary choice)
    selected_indices = [0]

    # 2. Initialize max_sims:
    # Stores max(dot(candidate, selected)) for every candidate.
    # We want to pick the candidate where this value is MINIMIZED (furthest away).
    max_sims = pool @ pool[0]
    max_sims[0] = float("inf")  # Mask the selected one

    # 3. Incrementally add vectors
    # We perform one matrix-vector product per iteration: O(N) memory, O(K*N) compute
    for _ in tqdm(range(1, n_target), desc="Greedy Selection"):
        best_idx = torch.argmin(max_sims).item()
        selected_indices.append(best_idx)

        new_sims = pool @ pool[best_idx]

        torch.maximum(max_sims, new_sims, out=max_sims)

        max_sims[best_idx] = float("inf")

    return candidates[selected_indices].cpu()


def optimize_magnitude_codebook(k=8, n_bits=2, m_iters=100):
    """Algorithm 2: Lloyd-Max for Chi-distributed magnitudes."""
    n_centers = 2**n_bits

    probs = np.linspace(0, 1, n_centers + 1)

    # Inverse CDF of Chi distribution (which is sqrt of Chi2)
    # If X ~ Chi2(k), then sqrt(X) ~ Chi(k)
    limit_max, grid_res = 20.0, 50_000
    limits = np.sqrt(scipy.stats.chi2.ppf(probs, df=k))
    limits[-1] = limit_max

    centers = torch.zeros(n_centers)

    # PDF and expectation helper
    # f(r) ~ r^(k-1) e^(-r^2/2)
    # \int r f(r) dr involves Gamma functions.

    # Simplified Lloyd-Max iteration
    # Since we have the analytical formula in the paper, we can implement the integration loop
    # Or use a discrete approximation which is often more stable and sufficient

    # Discrete approximation for stability:
    r_grid = torch.linspace(0, limits[-1], grid_res)
    pdf = r_grid ** (k - 1) * torch.exp(-(r_grid**2) / 2)
    pdf /= pdf.sum()  # Normalize grid probability

    for _ in range(m_iters):
        # 1. Update centers: Center of mass of each region
        for i in range(n_centers):
            mask = (r_grid >= limits[i]) & (r_grid < limits[i + 1])
            if mask.sum() > 0:
                centers[i] = (r_grid[mask] * pdf[mask]).sum() / pdf[mask].sum()

        # 2. Update limits: Midpoint between centers
        limits[1:-1] = (centers[:-1] + centers[1:]).numpy() / 2

    return centers.float()
