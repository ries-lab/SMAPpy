import torch


class ItemizedDist:
    def __init__(self, dist: torch.distributions.Distribution):
        """Helper class to output single values from torch dists"""
        self._dist = dist

    def __call__(self):
        return self.sample()

    def sample(self):
        return self._dist.sample((1,)).item()


class UniformInt(torch.distributions.Distribution):
    def __init__(
        self,
        low: int | torch.Tensor,
        high: int | torch.Tensor,
        dtype: torch.dtype = torch.long,
    ):
        """
        Integer uniform distribution derived from torch.distributions.Uniform.

        Args:
            low:
            high:
            dtype:
        """
        super().__init__()
        # this is needed because core does not work otherwise
        low = (
            torch.as_tensor(low, dtype=torch.float)
            if isinstance(low, torch.Tensor)
            else low
        )
        high = (
            torch.as_tensor(high, dtype=torch.float)
            if isinstance(high, torch.Tensor)
            else high
        )
        self._dtype_int = dtype
        self._core = torch.distributions.Uniform(low, high)

    def sample(self, sample_shape=torch.Size()):
        return self._core.sample(sample_shape).floor().int()

    def sample_n(self, n):
        return self._core.sample(n).floor().to(self._dtype_int)


class LinkedDist(torch.distributions.Distribution):
    def __init__(self, base: torch.distributions.Distribution, n_repeat: int):
        """
        Create a distribution that repeats the base distribution n times.

        Args:
            base: base distribution
            n_repeat: number of repeats
        """
        super().__init__()
        self._base = base
        self._n_repeat = n_repeat

    def sample(self, sample_shape=torch.Size()):
        return self._base.sample(sample_shape).repeat(1, self._n_repeat)

    def sample_n(self, n):
        return self._base.sample(n).repeat(1, self._n_repeat)


def permute_dim_to_pos(tensor, dim, pos):
    num_dims = tensor.ndim

    # Convert negative indices to positive equivalents
    dim = dim + num_dims if dim < 0 else dim
    pos = pos + num_dims if pos < 0 else pos

    if dim >= num_dims or pos >= num_dims:
        raise ValueError(
            "Dimension and position must be within the tensor's dimensions"
        )

    # Create a list of dimensions excluding the one to be moved
    dims = [i for i in range(num_dims) if i != dim]

    # Insert the moved dimension at the specified position
    dims.insert(pos, dim)

    return tensor.permute(*dims)
