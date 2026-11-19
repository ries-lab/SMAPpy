from typing import Literal, Union, Optional

import torch


class FilterXYZ:
    def __init__(
        self,
        xextent: Union[tuple, torch.Tensor],
        yextent: Union[tuple, torch.Tensor],
        zextent: Optional[Union[tuple, torch.Tensor]] = None,
        mode: Optional[Literal["any", "all"]] = None,
    ):
        self.xextent = torch.as_tensor(xextent)
        self.yextent = torch.as_tensor(yextent)
        self.zextent = torch.as_tensor(zextent) if zextent is not None else None
        self._mode = mode

    def mask(self, xyz: torch.Tensor) -> torch.Tensor:
        in_fov = (
            (xyz[..., 0] >= self.xextent[..., 0])
            * (xyz[..., 0] < self.xextent[..., 1])
            * (xyz[..., 1] >= self.yextent[..., 0])
            * (xyz[..., 1] < self.yextent[..., 1])
        )

        if self.zextent is not None:
            in_fov *= (xyz[..., 2] >= self.zextent[..., 0]) * (
                xyz[..., 2] < self.zextent[..., 1]
            )
        return in_fov

    def filter(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Returns index of data that are inside the specified extent.

        Args:
            xyz: data tensor that is filtered
        """
        in_fov = self.mask(xyz)

        if self._mode == "any":
            return torch.any(in_fov, dim=1)
        elif self._mode == "all":
            return torch.all(in_fov, dim=1)
        elif self._mode is None:
            return in_fov
        else:
            raise ValueError(f"Unsupported mode: {self._mode}")
