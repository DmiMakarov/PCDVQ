from abc import ABC, abstractmethod
from math import sqrt

import torch
from torch import Tensor
import torch.nn.functional as F


class StandardRegularization(ABC):
    """
    Abstract base for regularization/transformation blocks applied to tensors.

    Subclasses should implement `forward(x)` which takes a tensor and returns
    a transformed tensor of the same shape (unless otherwise documented).
    """
    def __init__(
        self,
        p: int,
        seed: int = 42,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> None: 
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply the regularization/transform to `x`. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def reverse(self, y: Tensor):
        """Inverse of `forward`. Must be implemented by subclasses when invertible."""
        raise NotImplementedError

    def __call__(self, x: Tensor):
        return self.forward(x)


class RandomizedHadamard(StandardRegularization):
    """
    Randomized Hadamard transform per row (last dimension) for standard Gaussian regularization.
    """

    def __init__(
        self,
        p: int,
        seed: int = 42,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.p, self.dtype, self.device = p, dtype, device

        self.p = p
        self.eps = torch.finfo(dtype).eps

        # generator for reproducibility
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)

        # padding to next power of two (length along transform axis)
        self.n = 1 << (p - 1).bit_length()

        # generate random sign vector
        self.signs = (torch.randint(0, 2, (self.n,), generator=g, device=device) * 2 - 1).to(dtype)
        # permutation functionality
        self.perm = torch.randperm(self.n, generator=g, device=device)
        self.inv_perm = torch.argsort(self.perm)

    @staticmethod
    def fwht(x: Tensor) -> Tensor:
        """Fast Walsh–Hadamard transform along the last dimension (per row).

        Each row is transformed independently.
        Input: (Batch, N). N must be power of 2.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected a 2D tensor shaped (rows, cols), got {x.dim()}")

        rows, n = x.shape
        if n & (n - 1):
            raise ValueError("Hadamard length (cols) must be a power of two")

        h = 1
        y = x.clone()
        while h < n:
            y = y.view(rows, n // (2 * h), 2, h)
            a, b = y[:, :, 0, :], y[:, :, 1, :]
            y = torch.cat((a + b, a - b), dim=-1)
            y = y.view(rows, n)
            h *= 2

        return y / sqrt(n)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply the randomized Hadamard transform to each row of `x` (last dimension).

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor, scaling factor
        """
        k = x.shape[1]
        # pad columns to next power of two
        if k < self.n:
            x = F.pad(x, (0, self.n - k))
        # calcualte scaling factor for standardization (per row)
        s = (torch.linalg.vector_norm(x, dim=1).clamp_min(self.eps) / sqrt(self.n)).unsqueeze(1)
        # apply random sings
        y = x * self.signs.view(1, -1)
        # apply Hadamard (row-wise)
        y = self.fwht(y)
        # permute and normalize
        y = y[:, self.perm]
        return y / s, s

    def reverse(self, z: Tensor, s: Tensor) -> Tensor:
        """
        Inverse of the randomized Hadamard transform.

        Args:
            z: Transformed tensor.

        Returns:
            Original tensor before transformation.
        """
        y = z * s
        # inverse permutation
        y = y[:, self.inv_perm]
        # undo Hadamard (scaled)
        x_padded = self.fwht(y)
        # undo random signs
        x_padded = x_padded / self.signs.view(1, -1)
        # remove padding if original length < n
        return x_padded[:, : self.p]


class QRRotation(StandardRegularization):
    """
    Applies a random orthogonal rotation using a matrix Q derived from 
    QR decomposition of a random Gaussian matrix.
    """

    def __init__(
        self,
        p: int,
        seed: int = 42,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.p, self.dtype, self.device = p, dtype, device
        self.eps = torch.finfo(dtype).eps

        # Generator for reproducibility
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)

        # 1. Determine dimension 'n' 
        # We mimic the RHT logic: pad to next power of 2 to ensure 
        # downstream compatibility with block-based VQ.
        self.n = 1 << (p - 1).bit_length()

        # 2. Generate Random Orthogonal Matrix Q via QR Decomposition
        # Step A: Generate random Gaussian matrix A ~ N(0, 1)
        A = torch.randn(self.n, self.n, generator=g, device=device, dtype=dtype)
        
        # Step B: Compute QR decomposition (Topic 15)
        # Q is orthogonal, R is upper triangular. We only need Q.
        Q, _ = torch.linalg.qr(A)
        
        # Register Q as a buffer so it moves with the model (to GPU) 
        # and is saved in state_dict, but is not a learnable parameter.
        self.register_buffer('Q', Q)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply random rotation Q to x.
        Formula: y = (x @ Q.T) / s
        """
        k = x.shape[1]
        
        # 1. Pad columns to next power of two (matching RHT behavior)
        if k < self.n:
            x = F.pad(x, (0, self.n - k))
            
        # 2. Calculate scaling factor for standard Gaussian distribution
        # We divide by sqrt(n) to standardize the variance
        s = (torch.linalg.vector_norm(x, dim=1).clamp_min(self.eps) / sqrt(self.n)).unsqueeze(1)
        
        # 3. Apply Orthogonal Rotation (Topic 2: Matrix Mult)
        # x is (Batch, n), Q is (n, n). 
        # We want y = x @ Q.T (equivalent to Q @ x.T in column-vector notation)
        y = x @ self.Q.t()
        
        return y / s, s

    def reverse(self, z: Tensor, s: Tensor) -> Tensor:
        """
        Inverse of the QR rotation.
        Since Q is orthogonal, Q^-1 = Q.T.
        """
        # 1. Restore magnitude
        y = z * s
        
        # 2. Apply Inverse Rotation
        # Inverse of (x @ Q.T) is (y @ Q)
        x_padded = y @ self.Q
        
        # 3. Remove padding to return to original dimension p
        return x_padded[:, : self.p]
    
    # Helper for buffer registration if not inheriting from nn.Module
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)