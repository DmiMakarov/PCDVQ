from itertools import product
from logging import getLogger
from tqdm import tqdm
import torch

logger = getLogger(__name__)


def construct_direction_codebook()->list[torch.Tensor]:
        """Construct the direction codebooks for the codebook."""
        logger.info("Constructing direction codebooks...")
        d8_half_vectors = generate_d8_half_vectors()
        logger.info("Generated d8 half vectors...")
        additional = generate_12()
        logger.info("Generating additional vectors...")
        d8_signs = generate_d8_signs(torch.cat([d8_half_vectors, additional], dim=0))
        d8_full = torch.cat([d8_signs, d8_signs + 0.25], dim=0)

        return d8_full / torch.norm(d8_full, dim=0)


def generate_d8_half_vectors(max_sq_norm: float = 10.0) -> torch.Tensor:
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

def generate_12()->torch.Tensor:
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

def generate_d8_12()->torch.Tensor:
        """Generate the 12 for the d8 half vectors."""
        vectors = []

        for ks in tqdm(product([0.5, 1.5, 2.5, 3.5], repeat=8)):
            vec = torch.tensor(ks)
            if vec.dot(vec) > 12 or vec.dot(vec) < 10:
                continue

            vectors.append(vec)

        vectors = torch.stack(vectors, dim=0)
        vectors = torch.unique(vectors, dim=0)

        return vectors


def generate_d8_signs(d8_half_vectors:torch.Tensor)->torch.Tensor:
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

        #assert vectors.shape[0] == 2 ** (7 + 8)

        return vectors

