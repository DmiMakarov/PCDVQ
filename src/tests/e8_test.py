import torch
from itertools import product

def generate_e8_roots():
        # Standard way to generate the 240 E8 roots in 8D
        roots = []

        # Type 1: ±e_i ± e_j (i < j)
        for i in range(8):
            for j in range(i + 1, 8):
                for signs in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    v = torch.zeros(8)
                    v[i] = signs[0]
                    v[j] = signs[1]
                    roots.append(v)

        print(f"Type 1: {len(roots)} roots")

        # Type 2: (±1/2, ±1/2, ..., ±1/2) with even number of + signs
        for signs in product([-0.5, 0.5], repeat=8):
            if sum(s > 0 for s in signs) % 2 == 0:  # even number of +1/2
                v = torch.tensor(signs)
                roots.append(v)

        print(f"Type 2: {len(roots)} roots")

        roots = torch.stack(roots)
        # Normalize to unit length (all have norm sqrt(2) actually)
        roots = roots / roots.norm(dim=1, keepdim=True)
        # Remove duplicates (there are exactly 240 unique up to sign)
        roots = torch.unique(roots, dim=0)
        assert roots.shape[0] == 240

        return roots  # (240, 8)


e8_roots = generate_e8_roots()
print(e8_roots.shape)
print()