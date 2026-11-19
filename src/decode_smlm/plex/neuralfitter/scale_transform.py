import torch
from deprecated import deprecated

from . import spec


@deprecated(reason="Deprecated in favour of `channeled_scaler_from_tar`")
class ScalerTargetList:
    def __init__(self, phot: float, z: float, ch_map: spec.tar.MapListTar):
        """
        Rescale output of `ParameterListTarget`

        Args:
            phot: scale of photon
            z: scale of z
            ch_map: target channel map to scale the right channels
        """
        self.phot_max = phot
        self.z_max = z
        self._ch_map = ch_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()

        x[..., self._ch_map.ix_phot] /= self.phot_max
        x[..., self._ch_map.ix_z] /= self.z_max

        return x


@deprecated(
    reason="Deprecated in favour of ScalerModelChannel", version="0.11", action="error"
)
class ScalerModelOutput:
    ...


@deprecated(action="error")
class ScalerOffset:
    ...


@deprecated(action="error")
class ScalerInverseOffset(ScalerOffset):
    ...


@deprecated(action="error")
class InterpolationSpatial:
    ...
