import sklearn.neighbors
import torch


class DensityEstimator:
    def fit(self, xyz: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NearestNeighborsDensity(DensityEstimator):
    def __init__(self, radius: float):
        """
        Nearest neighbors density estimator, i.e. counting neighbors within a radius
        (excluding the point itself).

        Args:
            radius: max. radius
        """
        self.radius = radius
        self._nn = sklearn.neighbors.NearestNeighbors(radius=radius)

    def fit(self, xyz: torch.Tensor) -> torch.Tensor:
        self._nn.fit(xyz.numpy())
        counts = self._nn.radius_neighbors(xyz.numpy(), return_distance=False)
        nn_count = self._counts_to_nn(counts)
        return nn_count

    @staticmethod
    def _counts_to_nn(counts):
        nn_count = torch.tensor([len(c) for c in counts])
        nn_count = nn_count - 1
        return nn_count
