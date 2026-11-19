from typing import Optional, Sequence, Union
import functools

import torch

from .. import utils
from ..generic import utils as gen_utils
from ..simulation.trafo.pos import trafo


class IndicatorChannelOffset:
    def __init__(
        self,
        xextent: tuple[float, float],
        yextent: tuple[float, float],
        img_shape: tuple[int, int],
        xy_trafo: Optional[trafo.XYZTransformation],
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Compute offset of pixel centers for each channel.

        Args:
            xextent: tuple of (min, max) of x extent (shared for all channels)
            yextent: tuple of (min, max) of y extent (shared for all channels)
            img_shape: tuple of (height, width) of image (shared for all channels)
            xy_trafo: transformation to apply to pixel centers
            device: device to compute on
        """
        # instance checking is for a later point when different channels can have
        # different extents and image shapes
        self._xextent = xextent if isinstance(xextent[0], Sequence) else (xextent,)
        self._yextent = yextent if isinstance(yextent[0], Sequence) else (yextent,)
        self._img_shape = (
            img_shape if isinstance(img_shape[0], Sequence) else (img_shape,)
        )
        self._xy_trafo = xy_trafo
        self._device = torch.device(device) if isinstance(device, str) else device

        self._ctr = None
        self._map = None

        self.validate()

    @functools.lru_cache(maxsize=128)
    def forward(
        self, xy_trafo: Optional[trafo.XYZTransformation] = None
    ) -> Sequence[torch.Tensor]:
        """

        Args:
            xy_trafo: overwrite xy_trafo attribute

        """
        xy_trafo = xy_trafo if xy_trafo is not None else self._xy_trafo

        self._ctr = self._compute_ctr()
        map = self._compute_map(xy_trafo=xy_trafo)
        # to keep flexibility of differently sized maps
        self._map = torch.unbind(map, dim=0)

        return self._map

    def validate(self):
        if any(len(x) != 1 for x in (self._xextent, self._yextent, self._img_shape)):
            raise NotImplementedError("Extents and image shapes must be of length 1.")

    def _compute_ctr(self) -> Sequence[torch.Tensor]:
        """
        Compute centers of pixels for each channel.

        Returns:
            list of tensors of size (2, *img_shape) with pixel centers (x and y)
        """
        ctr = []

        for i, (x, y, s) in enumerate(
            utils.future.zip(self._xextent, self._yextent, self._img_shape, strict=True)
        ):
            _, _, ctr_x, ctr_y = gen_utils.frame_grid(
                img_size=s, xextent=x, yextent=y, device=self._device
            )
            ctr.append(torch.stack(torch.meshgrid(ctr_x, ctr_y), dim=0))

        return ctr

    def _compute_map(self, xy_trafo: trafo.XYZTransformation) -> Sequence[torch.Tensor]:
        if self._ctr is None:
            raise ValueError(
                f"Pixel centers have not been computed yet."
                f"Call {self.__class__.__name__}._compute_ctr() first."
            )
        if len(self._ctr) != 1:
            raise NotImplementedError("Only one reference channel is supported.")

        ctr = self._ctr[0]
        xyz = ctr.view(2, -1).permute(1, 0)  # 2 = x, y
        xyz = torch.cat([xyz, torch.ones_like(xyz[:, :1]) * float("nan")], dim=1)

        if xyz.device != xy_trafo.device:
            raise ValueError(
                f"Device mismatch between xyz and xy_trafo. "
                f"xyz is on {xyz.device}, xy_trafo is on {xy_trafo.device}."
            )

        xyz = xy_trafo.forward(xyz)

        map = xyz[..., :-1].permute(1, -1, 0).reshape(xyz.size(1), *ctr.size())
        # we want offsets, i.e. relative to the reference channel
        map = map - map[0]
        return map


def trafo_aux_factory(
    trafo: trafo.XYZTransformation, img_size: tuple[int, int]
) -> torch.Tensor:
    """
    Compute auxiliary tensor for a given transformation under default conditions (i.e.
    no channel-specific extents or image shapes, px_size of 1).

    Args:
        trafo: transformation to compute auxiliaries for
        img_size: size of image

    """
    xextent, yextent = gen_utils.extent_by_sizes(img_size)
    ind = IndicatorChannelOffset(
        xextent=xextent,
        yextent=yextent,
        img_shape=img_size,
        xy_trafo=trafo,
    )
    aux = ind.forward()
    aux = torch.cat(aux, dim=0)
    return aux
