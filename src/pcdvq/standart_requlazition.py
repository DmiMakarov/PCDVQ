from abc import ABC, abstractmethod
from math import sqrt

import torch
import torch.nn.functional as F


class StandardRegularization(ABC):
    """
    Abstract base for regularization/transformation blocks applied to tensors.

    Subclasses should implement `forward(x)` which takes a tensor and returns
    a transformed tensor of the same shape (unless otherwise documented).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the regularization/transform to `x`. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of `forward`. Must be implemented by subclasses when invertible."""
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
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
        super().__init__()

        self.p = p
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.eps = torch.finfo(dtype).eps
        self.s = None

        # generator for reproducibility
        g = None
        if seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(seed)

        # padding to next power of two (length along transform axis)
        self.n = 1 << (p - 1).bit_length()

        # generate random sign vector
        self.signs = (
            torch.randint(0, 2, (self.n,), generator=g, device=device) * 2 - 1
        ).to(dtype)

        # permutation functionality
        self.permute = torch.randperm(self.n, generator=g, device=device)

    @staticmethod
    def fwht(x: torch.Tensor) -> torch.Tensor:
        """Fast Walsh–Hadamard transform along the last dimension (per row).

        Each row is transformed independently. The last dimension length
        must be a power of two.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the randomized Hadamard transform to each row of `x` (last dimension).

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        k = x.shape[1]

        # pad columns to next power of two
        if k < self.n:
            x = F.pad(x, (0, self.n - k))

        # calcualte scaling factor for standardization (per row)
        sqrt_num_cols = sqrt(self.n)
        self.s = (
            torch.linalg.vector_norm(x, dim=1).clamp_min(self.eps) / sqrt_num_cols
        ).unsqueeze(1)

        # apply randomized Hadamard transform (row-wise)
        y = self.fwht(x)

        # apply random sings
        y_rand = y * self.signs.view(1, -1)

        # apply permutation
        y_rand = y_rand[:, self.permute]

        # scaling

        return y_rand / self.s

    def reverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse of the randomized Hadamard transform.

        Args:
            z: Transformed tensor.

        Returns:
            Original tensor before transformation.
        """
        if self.s is None:
            raise ValueError("Must call forward() before reverse().")

        # undo scaling
        y_rand = z * self.s

        # inverse permutation
        inv_permute = torch.argsort(self.permute)
        y_rand = y_rand[:, inv_permute]

        # undo random signs
        y = y_rand / self.signs.view(1, -1)

        # undo Hadamard (scaled)
        x_padded = self.fwht(y)

        # remove padding if original length < n
        return x_padded[:, : self.p]
