from src.pcdvq.codebooks import PCDVQCodebook


r_bits = 2
d_bits = [8, 10, 12, 14, 16]


for d_bit in d_bits:

    codebook = PCDVQCodebook(direction_bits=d_bit, magnitude_bits=r_bits)
    codebook.build(use_38p=True)
    codebook.save(f'codebooks/codebook_e8p_{d_bit}_{r_bits}.pt')
    codebook.build(use_38p=False)
    codebook.save(f'codebooks/codebook_e8_{d_bit}_{r_bits}.pt')
