import torch
import torch.nn as nn
from .utils import get_linear_layers

class PureSVDApproximator:
    """
    Topic 7: Eckart-Young Theorem.
    Replaces the weight matrix W with its Rank-k approximation.
    W_approx = U_k @ S_k @ V_k.T
    """
    def __init__(self, target_bpw=2.0, original_dtype=torch.float16):
        self.target_bpw = target_bpw
        self.dtype = original_dtype

    def __call__(self, weight):
        # Calculate Rank k based on target BPW (Topic 5: Matrix Rank Storage)
        # BPW = (16 * k * (rows + cols)) / (rows * cols)
        # k = (BPW * rows * cols) / (16 * (rows + cols))
        rows, cols = weight.shape
        k = int((self.target_bpw * rows * cols) / (16 * (rows + cols)))
        k = max(1, k)
        
        # Compute SVD (Topic 6)
        w_float = weight.float()
        try:
            U, S, Vh = torch.linalg.svd(w_float, full_matrices=False)
            
            # Low Rank Approximation
            W_approx = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
            return W_approx.to(self.dtype), k
        except Exception as e:
            print(f"SVD failed: {e}")
            return weight, 0

class RTNQuantizer:
    """
    Simple Round-To-Nearest (MinMax) Quantizer.
    Serves as a baseline to see if PCDVQ geometry actually helps.
    """
    def __init__(self, bits=2):
        self.bits = bits
        self.qmax = 2**(bits-1) - 1
        self.qmin = -2**(bits-1)
    
    def __call__(self, weight):
        w = weight.float()
        # Per-channel scale (simple blocking)
        # Using block size 128 for fair comparison with vector dim
        # But simple MinMax is usually per-tensor or per-channel
        scale = w.abs().max(dim=1, keepdim=True)[0] / self.qmax
        scale = torch.clamp(scale, min=1e-8)
        
        w_q = torch.round(w / scale)
        w_q = torch.clamp(w_q, self.qmin, self.qmax)
        
        w_dequant = w_q * scale
        return w_dequant.to(weight.dtype)

def apply_baseline(model, method="svd", target_bpw=2.125):
    """
    Applies a baseline method to all linear layers.
    """
    total_params = 0
    total_k = 0
    
    linear_layers = get_linear_layers(model)
    
    for name, layer in linear_layers.items():
        w = layer.weight.detach()
        
        if method == "svd":
            approximator = PureSVDApproximator(target_bpw=target_bpw, original_dtype=w.dtype)
            w_new, k = approximator(w)
            total_k += k
            layer.weight.data = w_new
        elif method == "rtn":
            # Simple 2-bit RTN
            quantizer = RTNQuantizer(bits=2)
            w_new = quantizer(w)
            layer.weight.data = w_new
            
    return total_k / len(linear_layers) if method == "svd" else 0