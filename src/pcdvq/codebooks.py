from itertools import product

import numpy as np
import scipy.special
import torch
from scipy import stats

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class Codebook:
    def __init__(self, direction_bits:int = 16, magnitude_bits:int = 2):
        """Initialize the codebook."""
        self.direction_bits = direction_bits
        self.magnitude_bits = magnitude_bits

        self.codebook_magnitude = torch.randn(2**magnitude_bits)
        self.codebook_direction = torch.randn(2**direction_bits, 7)

    def _generate_d8_half_vectors(self,max_sq_norm: float = 10.0) -> torch.Tensor:
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
        return torch.tensor([[3, 1, 1, 1, 3, 3, 3, 3], [1, 3, 1, 1, 3, 3, 3, 3], [1, 1, 3, 1, 3, 3, 3, 3],
                               [1, 1, 1, 3, 3, 3, 3, 3], [3, 3, 3, 1, 3, 3, 1, 1], [3, 3, 3, 1, 3, 1, 3, 1],
                               [3, 3, 3, 1, 1, 3, 3, 1], [3, 3, 3, 1, 3, 1, 1, 3], [3, 3, 3, 1, 1, 3, 1, 3],
                               [3, 3, 3, 1, 1, 1, 3, 3], [3, 3, 1, 3, 3, 3, 1, 1], [3, 3, 1, 3, 3, 1, 3, 1],
                               [3, 3, 1, 3, 1, 3, 3, 1], [3, 3, 1, 3, 3, 1, 1, 3], [3, 3, 1, 3, 1, 3, 1, 3],
                               [3, 3, 1, 3, 1, 1, 3, 3], [3, 1, 3, 3, 3, 3, 1, 1], [3, 1, 3, 3, 3, 1, 3, 1],
                               [3, 1, 3, 3, 1, 3, 3, 1], [3, 1, 3, 3, 3, 1, 1, 3], [3, 1, 3, 3, 1, 3, 1, 3],
                               [1, 3, 3, 3, 1, 1, 3, 3], [1, 3, 3, 3, 3, 3, 1, 1], [1, 3, 3, 3, 3, 1, 3, 1],
                               [1, 3, 3, 3, 1, 3, 3, 1], [1, 3, 3, 3, 3, 1, 1, 3], [1, 3, 3, 3, 1, 3, 1, 3],
                               [1, 1, 3, 3, 1, 3, 3, 3], [3, 3, 1, 1, 3, 3, 3, 1]]) / 2


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

    def construct_direction_codebook(self)->list[torch.Tensor]:
        """Construct the direction codebooks for the codebook."""
        logger.info("Constructing direction codebooks...")
        d8_half_vectors = self._generate_d8_half_vectors()
        logger.info("Generated d8 half vectors...")
        additional = self._generate_12()
        logger.info("Generating additional vectors...")
        d8_full = torch.cat([d8_half_vectors, additional], dim=0)
        d8_signs = self._generate_d8_signs(d8_full)

        logger.info("Generating direction vectors...")
        self.codebook_direction = self._get_direction_from_vectors(torch.cat([d8_signs, d8_signs + 0.25], dim=0))
        logger.info("Constructed direction codebooks successfully")

    def find_max_r_bisection(self, k:int, tau:float, eps:float = 1e-8)->float:
        """Find the max r for the codebook using bisection from chi2 distribution."""
        upper_bound = 2 * np.sqrt(k)
        lower_bound = 0.0

        while np.abs(upper_bound - lower_bound) > eps:

            mid = (upper_bound + lower_bound) / 2
            p = stats.chi2.cdf(mid**2, df=k)
            if p < tau:
                lower_bound = mid
            else:
                upper_bound = mid

        return upper_bound

    def construct_magnitude_codebook(self, k:int = 8, tau:float = 0.99, tol:float = 1e-8, max_iters:int = 100)->torch.Tensor:
        """Construct the magnitude codebook using bisection from chi2 distribution."""
        num_centers = 2 ** self.magnitude_bits
        logger.info(f"Constructing magnitude codebook with {num_centers} centers...")
        max_r = self.find_max_r_bisection(k, tau)
        logger.info(f"Found max r: {max_r}")
        codebook = torch.linspace(0, max_r, num_centers + 1)
        codebook = 0.5 * (codebook[:-1] + codebook[1:])
        for _ in range(max_iters):
            logger.info(f"Iteration {_} of {max_iters}...")
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
            logger.info(f"Current max loss: {max_loss}")
            if max_loss < tol:
                break

        logger.info(f"Constructed magnitude codebook with {num_centers} centers successfully")

        self.codebook_magnitude = codebook

    def save_codebooks(self, path: str = ""):
        """Save the codebooks to a file."""
        torch.save(self.codebook_direction, path + "codebook_direction.pt")
        torch.save(self.codebook_magnitude, path + "codebook_magnitude.pt")

    def load_codebooks(self, path:str = ""):
        """Load the codebooks from a file."""
        self.codebook_direction = torch.load(path + "codebook_direction.pt")
        self.codebook_magnitude = torch.load(path + "codebook_magnitude.pt")


if __name__ == "__main__":
    codebook = Codebook()
    codebook.construct_direction_codebook()
    codebook.construct_magnitude_codebook()
    codebook.save_codebooks()
