from typing import Optional, Union

import torch


class ScalerModelChannel:
    def __init__(
        self,
        factor: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
        auto_view: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Scaling of channeled tensors.

        Note:
            First the factor is applied, then the offset.

        Args:
            factor: 1D, size as number of channels or broadcastable with model output
            offset: 1D, size as number f channels or broadcastable with model output
            auto_view: if True, the factor and offset are automatically reshaped
             assuming image space (N x C x H x W)
            device: device to move the tensors to
        """
        if auto_view:
            if factor is not None and factor.dim() == 1:
                factor = factor.view(1, -1, 1, 1)
            if offset is not None and offset.dim() == 1:
                offset = offset.view(1, -1, 1, 1)

        self._factor = factor
        self._offset = offset

        if device is not None:
            self._factor = self._factor.to(device)
            self._offset = self._offset.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._factor is not None:
            x = x * self._factor
        if self._offset is not None:
            x = x + self._offset
        return x


class ScalerAmplitude:
    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        """
        Simple Processing that rescales the amplitude, i.e. the pixel values.

        Args:
            scale (float): reference value
            offset: offset value
        """
        self.scale = scale if scale is not None else 1.0
        self.offset = offset if offset is not None else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward the tensor and rescale it.
        """
        return (x - self.offset) / self.scale
