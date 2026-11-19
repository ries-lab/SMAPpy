from typing import Sequence

import torch

from .. import frame_processing as fp
from ...generic import lazy
from ...generic import protocols
from ...simulation import camera as cam
from ...simulation.trafo import pos


class MultiChannelSplitter:
    def __init__(
        self,
        per_channel: list[Sequence[protocols.Forwardable]],
        pre_common: Sequence[protocols.Forwardable] | None = None,
        clone: bool = True,
    ):
        """
        Splits a tensor into multiple tensors,
         applies pre-processing to all and channel specific processing to each
         channel.

        Args:
            per_channel: per channel processing
            pre_common: common pre-processing before splitting
            clone: clone input tensor before per-channel processing (omit if input
             is already a list of tensors)
        """
        self._per_channel = per_channel
        self._pre_common = pre_common if pre_common is not None else []
        self._clone = clone

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        for p in self._pre_common:
            x = p.forward(x)

        if self._clone:
            x = [x.clone() for _ in range(len(self._per_channel))]
        else:
            x = list(x) if isinstance(x, Sequence) else x

        for i, (xx, p) in enumerate(zip(x, self._per_channel, strict=True)):
            for pp in p:
                xx = pp.forward(xx)
            x[i] = xx

        return x


def dual_splitter_factory(
    mirr_dim_gain: int | None,
    mirr_dim_ch: int | None,
    crop_to_fold: int | None = 8,
    crop_to: tuple[int | slice, int | slice]
    | tuple[tuple[int | slice, int | slice]]
    | None = None,
    camera: cam.CameraEMCCD | None = None,
    offset_xy: tuple[int, int] | None = None,
    _clone: bool = True,
) -> MultiChannelSplitter:
    """
    Factory to create multi-channel pipeline for dual channel data from a single camera.
    This includes gain correction, channel and em specific mirroring and cropping.

    Note:
        - Reversing the order of the channels can be achieved by simply cropping
          differently. Assumeing a 512x512 with the middle line at 256, left dark and
          right bright. Cropping (0, 256) will return in the order [dark, bright], while
          cropping (256, 512) will return [bright, dark].

    Args:
        mirr_dim_gain: mirror dimension due to em_gain
        mirr_dim_ch: mirror dimension due to channels
        crop_to_fold: crop to multiple of this number
        crop_to: crop to this size, or tuple of crops to specify crop for each channel
         individually
        camera: camera with gain to backward (i.e. photon units convert) before further
         processing
        offset_xy: offset x/y after de-mirroing between channels. The 2nd channel is moved.
        _clone: clone input tensor before per-channel processing
    """
    common_gain = [lazy.forward_relay("backward")(camera)] if camera is not None else []
    common_mirr = [fp.Mirror2D(dims=mirr_dim_gain)] if mirr_dim_gain is not None else []
    pre_common = common_gain + common_mirr
    per_channel = [
        [],
        [fp.Mirror2D(dims=mirr_dim_ch)] if mirr_dim_ch is not None else [],
    ]
    # shift before cropping (but of course after mirroring),
    # such that there is a chance that padded values are cropped
    # and not "content"
    if offset_xy is not None:
        per_channel[1].append(
            pos.frame.FrameShiftCropPad(x=offset_xy[0], y=offset_xy[1])
        )

    if crop_to is not None:
        if not isinstance(crop_to[0], Sequence):
            crop_to = [crop_to] * len(per_channel)
        _ = [
            p.append(fp.Crop(crop_to=c))
            for p, c in zip(per_channel, crop_to, strict=True)
        ]
    if crop_to_fold is not None:
        _ = [p.append(fp.AutoLeftUpperCrop(px_fold=crop_to_fold)) for p in per_channel]
    return MultiChannelSplitter(
        per_channel=per_channel, pre_common=pre_common, clone=_clone
    )


def multi_cam_factory(
    mirr_dim_gain: Sequence[int | None],
    mirr_dim_ch: Sequence[int | None],
    crop_to_fold: int | None = 8,
    crop_to: tuple[int | slice, int | slice] | None = None,
    camera: Sequence[cam.Camera] | None = None,
    offset_xy: Sequence[tuple[int, int]] | None = None,
):
    if offset_xy is None:
        offset_xy = [None] * len(mirr_dim_gain)

    per_channel = []
    for i, (dim_gain, dim_ch, off_xy) in enumerate(
        zip(mirr_dim_gain, mirr_dim_ch, offset_xy, strict=True)
    ):
        p = [lazy.forward_relay("backward")(camera[i])] if camera is not None else []
        p += [fp.Mirror2D(dims=dim_gain)] if dim_gain is not None else []
        p += [fp.Mirror2D(dims=dim_ch)] if dim_ch is not None else []
        p += (
            [pos.frame.FrameShiftCropPad(x=off_xy[0], y=off_xy[1])]
            if off_xy is not None
            else []
        )
        p += [fp.Crop(crop_to=crop_to)] if crop_to is not None else []
        p += (
            [fp.AutoLeftUpperCrop(px_fold=crop_to_fold)]
            if crop_to_fold is not None
            else []
        )
        per_channel.append(p)

    return MultiChannelSplitter(per_channel=per_channel, pre_common=None, clone=False)
