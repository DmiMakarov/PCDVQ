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


class RandomizedHadamard(StandardRegularization):
    """
    Randomized Hadamard transform per column for standard Gaussian regularization.
    """
    def __init__(
        self,
        p: int,
        seed: int = 42,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
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

        # padding to next power of two
        self.n = 1 << (p - 1).bit_length()

        # generate random sign vector
        self.signs = (torch.randint(0, 2, (self.n,), generator=g, device=device) * 2 - 1).to(dtype)

        # permutation functionality
        self.permute = torch.randperm(self.n, generator=g, device=device)

    @staticmethod
    def fwht(x: torch.Tensor) -> torch.Tensor:
        """Fast Walsh–Hadamard transform

        Modified version of function from https://github.com/amitport/hadamard-transform

        The hadamard transform is not numerically stable by nature (lots of subtractions),
        it is recommended to use with float64 when possible

        :param x: Either a vector or a batch of vectors where the first dimension is the batch dimension.
                Each vector's length is expected to be a power of 2! (or each row if it is batched)
        :return: The normalized Hadamard transform of each vector in x
        """
        original_shape = x.shape
        assert 1 <= len(original_shape) <= 2, 'input\'s dimension must be either 1 or 2'
        if len(original_shape) == 1:
            # add fake 1 batch dimension
            # for making the code a follow a single (batched) path
            x = x.unsqueeze(0)
        batch_dim, d = x.shape

        h = 2
        while h <= batch_dim:
            hf = h // 2
            x = x.view(batch_dim // h, d, h)

            half_1, half_2 = x[:, :, :hf], x[:, :, hf:]

            x = torch.cat((half_1 + half_2, half_1 - half_2), dim=-1)

            h *= 2

        return (x / sqrt(batch_dim)).view(*original_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the randomized Hadamard transform to each column of `x`.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        p = x.shape[0]

        # pad x to next power of two
        if p < self.n:
            x = F.pad(x, (0, 0, 0, self.n - p))
        else:
            x = x

        # calcualte scaling factor for standardization
        sqrt_num_cols = sqrt(self.n)
        self.s = (torch.linalg.vector_norm(x, dim=0).clamp_min(self.eps) / sqrt_num_cols).unsqueeze(0)

        # apply randomized Hadamard transform
        y = self.fwht(x)

        # apply random sings
        y_rand = y * self.signs.view(-1, 1)

        # apply permutation
        y_rand = y_rand[self.permute, :]

        # scaling
        z = y_rand / self.s
        return z

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
        y_rand = y_rand[inv_permute, :]

        # undo random signs
        y = y_rand / self.signs.view(-1, 1)

        # undo Hadamard (scaled)
        x_padded = self.fwht(y)

        # remove padding if original length < n
        return x_padded[:self.p, :]
