from codebooks import Codebook
from quantizer import Quantizer
from standart_requlazition import RandomizedHadamard

import torch

codebook = Codebook()
codebook.load_codebooks()

quantizer = Quantizer(codebook)

# sample (n, k) matrix; transform acts on each (1, k) row
w = torch.randint(0, 256, (256, 128), dtype=torch.float32)
randomized_hadamard = RandomizedHadamard(w.shape[1])

standardized_w = randomized_hadamard.forward(w)

print(f"Standardized w = {standardized_w}")

for i in range(standardized_w.shape[0]):
    print(standardized_w[i].mean(), standardized_w[i].std())

recovered_w = randomized_hadamard.reverse(standardized_w)

max_err = (w - recovered_w).abs().max()
print(f"max reconstruction error: {max_err.item()}")
