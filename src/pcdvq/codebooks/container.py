from .lattice_e8 import generate_e8_candidates
from .optimization import *
from ..utils import to_polar
from .lattice_e8p import generate_e8p_candidates

from logging import getLogger

logger = getLogger(__name__)


class PCDVQCodebook:
    def __init__(self, k: int = 8, direction_bits: int = 14, magnitude_bits: int = 2):
        self.k, self.dir_bits, self.mag_bits = k, direction_bits, magnitude_bits
        self.directions = None
        self.magnitudes = None

    def build(self, use_e8p: bool = True):
        """Run the optimization pipeline."""
        # Increase max_norm if you need more candidates for higher bits
        candidates = generate_e8p_candidates() if use_e8p else generate_e8_candidates(max_norm=4.0)
        
        logger.info(f"Pool size: {len(candidates)}")

        logger.info("Optimizing Directions (Greedy)...")
        selected_vectors = optimize_direction_codebook(candidates, self.dir_bits)
        self.directions, _ = to_polar(selected_vectors)

        logger.info(f"Directions shape: {self.directions.shape}")

        logger.info("Optimizing Magnitudes (Lloyd-Max)...")
        self.magnitudes = optimize_magnitude_codebook(self.k, self.mag_bits)



    def save(self, path):
        torch.save({"d": self.directions, "m": self.magnitudes}, path)

    def load(self, path):
        data = torch.load(path)
        self.directions, self.magnitudes = data["d"], data["m"]

    def to(self, device):
        self.directions = self.directions.to(device)
        self.magnitudes = self.magnitudes.to(device)
        return self

