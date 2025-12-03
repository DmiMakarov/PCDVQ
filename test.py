from codebooks import Codebook
from quantizer import Quantizer

import torch

codebook = Codebook()
codebook.load_codebooks()

quantizer = Quantizer(codebook)

w = torch.randn(256, 128)
print("Starting to quantize...")
w_q = quantizer.quantize_inplace(w)
print(w_q)

print(torch.linalg.norm(w - w_q))
