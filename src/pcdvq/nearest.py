import torch
from tqdm.auto import tqdm

def find_nearest(x: torch.Tensor, codes: torch.Tensor, batch_size:int=1024, codebook_batch_size:int=512):
    """Find best matching indices in codebook for each vector in the input"""
    bs, cs = batch_size, codebook_batch_size
    n,m = x.shape[0], codes.shape[0]
    res = torch.empty(n, dtype=torch.long, device=x.device)

    for i in tqdm(range(0, n, bs), desc='Matching'):
        xb = x[i:i+bs]
        # Init best scores to -inf to avoid 'if None' checks
        max_v = torch.full((xb.shape[0],), -float('inf'), device=x.device)
        max_i = torch.zeros((xb.shape[0],), dtype=torch.long, device=x.device)

        for j in range(0, m, cs):
            # Calculate sim and find max for this codebook chunk
            v, idx = (xb @ codes[j:j+cs].T).max(dim=1)
            
            mask = v > max_v
            max_v[mask] = v[mask]
            max_i[mask] = idx[mask] + j
            
        res[i:i+bs] = max_i
    return res