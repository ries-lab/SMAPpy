from typing import Literal, Sequence

import torch

from . import base
from .. import spec


def channeled_scaler_from_output(
    ch_map: spec.ModelChannelMapGMM,
    phot: Sequence[float],
    z: float,
    bg: Sequence[float],
    sigma_factor: float = 3.0,
    sigma_eps: float = 0.001,
    device: str | torch.device | None = None,
) -> base.ScalerModelChannel:
    """
    Scaling of sigma channels is
    `(sigma * sigma_factor + sigma_offset) * scaling[phot/xyz]`

    Args:
        ch_map: semantic channel map
        phot: photon scaling
        z: z scaling
        bg: background scaling
        sigma_factor: sigma scaling factor
        sigma_eps: sigma offset (needed because of loss)
        device: device to put the scalers on
    """

    factor = torch.ones(ch_map.n)
    offset = torch.zeros(ch_map.n)

    # scale and offset sigma channels
    factor[ch_map.ix_sig] *= sigma_factor
    offset[ch_map.ix_sig] += sigma_eps

    # scale and offset attributes
    for p, ix_phot, ix_phot_sig in zip(
        phot, ch_map.ix_phot, ch_map.ix_phot_sig, strict=True
    ):
        factor[[ix_phot, ix_phot_sig]] *= p
        offset[[ix_phot, ix_phot_sig]] *= p  # is this really correct?

    factor[[ch_map.ix_xyz[-1], ch_map.ix_xyz_sig[-1]]] *= z
    offset[[ch_map.ix_xyz[-1], ch_map.ix_xyz_sig[-1]]] *= z

    for b, ix_bg in zip(bg, ch_map.ix_bg, strict=True):
        factor[ix_bg] *= b

    return base.ScalerModelChannel(factor=factor, offset=offset, device=device)


def channeled_scaler_from_tar(
    phot: Sequence[float],
    z: float,
    ch_map: spec.tar.MapListTar,
    device: str | torch.device | None = None,
) -> base.ScalerModelChannel:
    # dim is necessary because ScalerModelChannel by default operates on images
    factor = torch.ones(1, ch_map.n)
    offset = torch.zeros(1, ch_map.n)

    factor[..., ch_map.ix_phot] /= torch.as_tensor(phot)
    factor[..., ch_map.ix_z] /= z

    return base.ScalerModelChannel(factor=factor, offset=offset, device=device)


def channeled_scaler_from_tar_bg(
    bg: Sequence[float],
    device: str | torch.device | None = None,
) -> base.ScalerModelChannel:
    factor = torch.ones(len(bg), 1, 1)
    offset = torch.zeros(len(bg), 1, 1)

    factor /= torch.as_tensor(bg).view(-1, 1, 1)
    return base.ScalerModelChannel(
        factor=factor, offset=offset, auto_view=False, device=device
    )


def channeled_scaler_from_input(
    frame_scale: Sequence[float],
    frame_offset: Sequence[float],
    aux_scale: Sequence[float] | None,
    ch_map: spec.model_in.ModelChannelMapInput,
    order: Literal["reversed"] | None = "reversed",
) -> base.ScalerModelChannel:
    """
    Scaling of input channels.


    Args:
        frame_scale:
        frame_offset:
        aux_scale:
        ch_map:
        order: if "reversed", the frame offset is specified as applied before
         the frame scaling. In reverse mode, the behaviour is equivalent to
         `ScalerAmplitude` with `scale=frame_scale` and `offset=frame_offset`.

    """
    frame_scale = torch.as_tensor(frame_scale)
    frame_offset = torch.as_tensor(frame_offset)

    if order == "reversed":
        frame_offset = frame_offset / frame_scale

    offset = torch.zeros(ch_map.n)
    factor = torch.ones(ch_map.n)
    # loop because channels might have windows
    for ch, sc, off in zip(ch_map.ix_ch, frame_scale, frame_offset, strict=True):
        factor[ch] /= sc
        offset[ch] -= off

    if aux_scale is not None:
        factor[ch_map.ix_aux] /= torch.as_tensor(aux_scale)

    return base.ScalerModelChannel(factor=factor, offset=offset)
