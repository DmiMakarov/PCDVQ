from itertools import product

import numpy as np
import scipy.special
import torch
from scipy import stats


class Codebook:
    def __init__(self, direction_bits:int = 12, magnitude_bits:int = 2):
        self.direction_bits = direction_bits
        self.magnitude_bits = magnitude_bits
        self.e8_basis = self._generate_e8_roots()

        self.codebook_magnitude = torch.randn(2**magnitude_bits, 8)
        self.codebook_direction = torch.randn(2**direction_bits, 8)

    def _generate_d8_half_vectors(max_sq_norm: float = 10.0) -> torch.Tensor:
        """Return a tensor of shape (227, 8) containing all vectors from E8 with norm less than max_sq_norm."""
        vectors = []

        for ks in product([0.5, 1.5, 2.5], repeat=8):
            vec = torch.tensor(ks)
            if vec.dot(vec) > max_sq_norm:
                continue

            vectors.append(vec)

        vectors = torch.stack(vectors, dim=0)
        vectors = torch.unique(vectors, dim=0)

        # QUIP#/VQ construction expects exactly 227 such vectors.
        assert vectors.shape[0] == 227, f"Expected 227 vectors, got {vectors.shape[0]}"

        return vectors

    def _generate_12(self)->torch.Tensor:
        """Generate the 12 for the d8 half vectors."""
        additional = torch.tensor([[3, 1, 1, 1, 3, 3, 3, 3], [1, 3, 1, 1, 3, 3, 3, 3], [1, 1, 3, 1, 3, 3, 3, 3],
                               [1, 1, 1, 3, 3, 3, 3, 3], [3, 3, 3, 1, 3, 3, 1, 1], [3, 3, 3, 1, 3, 1, 3, 1],
                               [3, 3, 3, 1, 1, 3, 3, 1], [3, 3, 3, 1, 3, 1, 1, 3], [3, 3, 3, 1, 1, 3, 1, 3],
                               [3, 3, 3, 1, 1, 1, 3, 3], [3, 3, 1, 3, 3, 3, 1, 1], [3, 3, 1, 3, 3, 1, 3, 1],
                               [3, 3, 1, 3, 1, 3, 3, 1], [3, 3, 1, 3, 3, 1, 1, 3], [3, 3, 1, 3, 1, 3, 1, 3],
                               [3, 3, 1, 3, 1, 1, 3, 3], [3, 1, 3, 3, 3, 3, 1, 1], [3, 1, 3, 3, 3, 1, 3, 1],
                               [3, 1, 3, 3, 1, 3, 3, 1], [3, 1, 3, 3, 3, 1, 1, 3], [3, 1, 3, 3, 1, 3, 1, 3],
                               [1, 3, 3, 3, 1, 1, 3, 3], [1, 3, 3, 3, 3, 3, 1, 1], [1, 3, 3, 3, 3, 1, 3, 1],
                               [1, 3, 3, 3, 1, 3, 3, 1], [1, 3, 3, 3, 3, 1, 1, 3], [1, 3, 3, 3, 1, 3, 1, 3],
                               [1, 1, 3, 3, 1, 3, 3, 3], [3, 3, 1, 1, 3, 3, 3, 1]]) / 2


        return additional

    def _generate_d8_signs(self,d8_half_vectors:torch.Tensor)->torch.Tensor:
        """Generate the signs for the d8 half vectors."""
        vectors = []

        for vec in d8_half_vectors:
            for signs in product([-1, 1], repeat=7):
                new_vec = vec.clone()
                new_vec[1:] = new_vec[1:] * torch.tensor(signs)

                if new_vec.sum() % 2 != 0:
                    new_vec[0] = -new_vec[0]

                if new_vec.sum() % 2 != 0:
                    raise ValueError("Invalid vector")

                vectors.append(new_vec)

        vectors = torch.stack(vectors, dim=0)

        assert vectors.shape[0] == 2 ** (7 + 8)

        return vectors

    def _get_direction_from_vectors(self, vectors:torch.Tensor)->torch.Tensor:
        """Get the direction from the vectors."""
        directions = [[torch.atan2(torch.sum(vec[i+1:] * vec[i+1:]), vec[i]) for i in range(7)] for vec in vectors]

        return torch.tensor(directions)

    def construct_direction_codebooks(self)->list[torch.Tensor]:
        """Construct the direction codebooks for the codebook."""
        d8_half_vectors = self._generate_d8_half_vectors()
        additional = self._generate_12()
        d8_full = torch.cat([d8_half_vectors, additional], dim=0)
        d8_signs = self._generate_d8_signs(d8_full)

        self.codebook_direction = self._get_direction_from_vectors(torch.cat([d8_signs, d8_signs + 0.25], dim=0))

    def find_max_r_bisection(self, k:int, tau:float, eps:float = 1e-8)->float:
        """Find the max r for the codebook using bisection from chi2 distribution."""
        upper_bound = 2 * torch.sqrt(k).item()
        lower_bound = 0.0

        while torch.abs(upper_bound - lower_bound) > eps:

            mid = (upper_bound + lower_bound) / 2
            p = stats.chi2.cdf(mid**2, df=k)
            if p < tau:
                lower_bound = mid
            else:
                upper_bound = mid

        return upper_bound

    def construct_magnitude_codebook(self, bits_for_magnitude:int, k:int, tau:float, tol:float = 1e-8, max_iters:int = 100)->torch.Tensor:
        """Construct the magnitude codebook using bisection from chi2 distribution."""
        num_centers = 2 ** bits_for_magnitude
        max_r = self.find_max_r_bisection(k, tau)
        codebook = torch.linspace(0, max_r, num_centers + 1)
        codebook = 0.5 * (codebook[:-1] + codebook[1:])
        for _ in range(max_iters):
            u = torch.empty(num_centers + 1)
            u[0] = 0.0
            u[-1] = max_r
            u[1: -1] = 0.5*(codebook[:-1] + codebook[1:])
            max_loss = 0.0
            codebook_tmp = torch.empty_like(codebook)
            for i in range(num_centers):
                num = scipy.special.gammainc((k + 1) / 2.0, np.float32((u[i + 1]**2) / 2.0)) - scipy.special.gammainc((k + 1) / 2.0, np.float32((u[i]**2) / 2.0))
                den = scipy.special.gammainc(k / 2.0, np.float32((u[i + 1]**2) / 2.0)) - scipy.special.gammainc(k / 2.0, np.float32((u[i]**2) / 2.0))
                cur = np.sqrt(2.0) * scipy.special.gamma((k + 1) / 2.0) / scipy.special.gamma(k / 2.0) * (num / (den + 1e-12))
                max_loss = max(max_loss, np.abs(cur - np.float32(codebook[i])))
                codebook_tmp[i] = cur

            codebook = codebook_tmp

            if max_loss < tol:
                break

        self.codebook_magnitude = codebook

    def construct_magnitude_codebooks(self)->list[torch.Tensor]:
        """Construct the magnitude codebooks for the codebook."""
        num_centers = 2 ** self.magnitude_bits
        max_r = self.find_max_r_bisection(num_centers, 0.99)
        codebook = torch.linspace(0, max_r, num_centers + 1)

    def save_codebooks(self, path: str = ""):
        """Save the codebooks to a file."""
        torch.save(self.codebook_direction, path + "codebook_direction.pt")
        torch.save(self.codebook_magnitude, path + "codebook_magnitude.pt")

    def load_codebooks(self, path:str = ""):
        """Load the codebooks from a file."""
        self.codebook_direction = torch.load(path + "codebook_direction.pt")
        self.codebook_magnitude = torch.load(path + "codebook_magnitude.pt")
