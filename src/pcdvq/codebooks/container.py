from .lattice_e8 import generate_e8_candidates
from .optimization import *
from ..utils import to_polar
from .lattice_e8p import generate_e8p_candidates

from logging import getLogger

logger = getLogger(__name__)


class PCDVQCodebook:
    def __init__(self, k: int = 8, direction_bits: int = 14, magnitude_bits: int = 2):
        self.k, self.dir_bits, self.mag_bits = k, direction_bits, magnitude_bits
        self.phis, self.magnitudes, self.directions = None, None, None

    def build(self, use_e8p: bool = False):
        """Run the optimization pipeline."""

        candidates = (
            generate_e8p_candidates()
            if use_e8p
            else generate_e8_candidates()
        )
        logger.info(f"Pool size: {len(candidates)}")

        logger.info("Optimizing Directions (Greedy)...")
        self.directions = optimize_direction_codebook(candidates, self.dir_bits)
        self.phis, _ = to_polar(self.directions)

        logger.info(f"Directions shape: {self.directions.shape}")

        logger.info("Optimizing Magnitudes (Lloyd-Max)...")
        self.magnitudes = optimize_magnitude_codebook(self.k, self.mag_bits)

    def save(self, path):
        torch.save({"phi": self.phis, "m": self.magnitudes, "d": self.directions}, path)

    def load(self, path):
        data = torch.load(path)
        self.phis, self.magnitudes, self.directions = data["phi"], data["m"], data["d"]

    def to(self, device):
        self.phis = self.phis.to(device)
        self.magnitudes = self.magnitudes.to(device)
        self.directions = self.directions.to(device)
        return self
