import torch


def generate_e8_candidates(max_norm: float=4.0):
    """
    Generates unique unit vectors from E8 lattice shells up to max_norm.
    E8 = D8 U (D8 + 0.5).
    """
    # 1. D8 part: Integer coordinates, sum is even
    # We generate integers in range [-4, 4] (roughly)
    r = torch.arange(-max_norm, max_norm + 1)
    grid = torch.cartesian_prod(*[r] * 8)
    # Filter: sum must be even
    d8 = grid[grid.sum(dim=1) % 2 == 0]

    # 2. D8 + 0.5 part: Half-integers, sum is even
    # effectively (2*int + 1)/2. Sum of (2x+1) is even => sum 2x + 8 is even => sum 2x even.
    # We generate odd integers and divide by 2.
    r_odd = torch.arange(-max_norm * 2 + 1, max_norm * 2, 2)
    grid_odd = torch.cartesian_prod(*[r_odd] * 8) / 2.0
    # Filter: sum of 0.5 coords must be even integer
    d8_half = grid_odd[grid_odd.sum(dim=1) % 2 == 0]

    # Combine and normalize
    all_vecs = torch.cat([d8, d8_half])

    return filter_and_normalize(all_vecs, max_norm)


def filter_and_normalize(all_vecs: torch.Tensor, max_norm = 4.0):
    norms = all_vecs.norm(dim=1)
    # Filter origin and too large vectors
    mask = (norms > 1e-8) & (norms <= max_norm)
    candidates = all_vecs[mask] / norms[mask].unsqueeze(1)

    # Remove duplicates (numerically stable unique)
    # rounding to 5 decimal places to handle float errors
    return torch.unique(torch.round(candidates * 10_000) / 10_000, dim=0)