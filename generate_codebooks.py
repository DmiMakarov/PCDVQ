import logging
from src.pcdvq.codebooks import PCDVQCodebook


r_bits = 2
d_bits = [8, 10, 12, 14, 16]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    for d_bit in d_bits:
        logger.info(f"Generating codebook: magnitude - {r_bits}bit, direction - {d_bit}bit")
        codebook = PCDVQCodebook(direction_bits=d_bit, magnitude_bits=r_bits)
        codebook.build(use_e8p=False)
        codebook.save(f"codebooks/codebook_e8_{d_bit}_{r_bits}.pt")
        codebook.build(use_e8p=True)
        codebook.save(f"codebooks/codebook_e8p_{d_bit}_{r_bits}.pt")
