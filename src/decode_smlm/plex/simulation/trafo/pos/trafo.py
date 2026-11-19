from abc import abstractmethod
import copy
from typing import Optional, Union, Callable, Tuple, Sequence, Literal

import numpy as np
import torch
from deprecated import deprecated
from scipy.spatial.transform import Rotation as SciRot

from ....generic import mixin


class XYZTransformation(mixin.ForwardCallAlias, mixin.MultiDevice):
    """
    Base class of transformation of 3D coordinates.
    Defined transformations may produced multi-channeled from single-channeled
    coordinates or leave the coordinates dimension unchanged.
    """

    @abstractmethod
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Number of output channels.
        """
        raise NotImplementedError


class XYZTransformationNoOp(XYZTransformation):
    def __init__(
        self, n_channels: Optional[int] = None, device: Union[str, torch.device] = "cpu"
    ):
        """
        No-op transformation.

        Args:
            n_channels: number of channels, used for replication
        """
        self._n_channels = n_channels
        self._device = torch.device(device) if isinstance(device, str) else device

    def __repr__(self):
        return f"XYZTransformationNoOp(n_channels={self._n_channels})"

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> "XYZTransformationNoOp":
        return XYZTransformationNoOp(n_channels=self._n_channels, device=device)

    def __len__(self) -> int:
        if self._n_channels is None:
            raise ValueError("Ill-defined for non-specified repeats")

        return self._n_channels

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if self._n_channels is None:
            return xyz

        return xyz.unsqueeze(1).repeat(1, self._n_channels, 1)


class XYZCompositTransformation(XYZTransformation):
    def __init__(self, trafo: Sequence[XYZTransformation], _copy: bool = True):
        """
        Composit transformation.

        Args:
            trafo: sequence of transformations
            _copy: whether to copy the transformations, if not, `append` and `insert`
             would modify the original argument
        """
        self._trafo = copy.copy(trafo) if _copy else trafo

    def __len__(self) -> int:
        """
        Number of output channels,
        i.e. the number of output channels of the last transformation.
        """
        return len(self._trafo[-1])

    def __getitem__(self, item):
        return self._trafo[item]

    def __repr__(self):
        trafo_str = ",\n".join(f"    {item}" for item in self._trafo)
        return f"XYZCompositTransformation([\n{trafo_str}\n])"

    @property
    def device(self) -> torch.device:
        return self._trafo[0].device

    def to(self, device: Union[str, torch.device]) -> "XYZCompositTransformation":
        return XYZCompositTransformation([t.to(device) for t in self._trafo])

    def append(self, trafo: XYZTransformation):
        self._trafo.append(trafo)

    def insert(self, index: int, trafo: XYZTransformation):
        self._trafo.insert(index, trafo)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        for t in self._trafo:
            xyz = t.forward(xyz)
        return xyz


class XYZChanneledTransformation(XYZTransformation):
    def __init__(self, trafo: XYZTransformation, ch: int):
        """
        Wraps a transformation to only apply the wrapped transformation to a specific
        channel of a multi-channel input.

        Args:
            trafo: transformation to wrap
            ch: channel index to apply transformation to
        """
        self._trafo = trafo
        self._ch = ch

    def __repr__(self):
        return f"XYZChanneledTransformation({self._trafo}, {self._ch})"

    @property
    def device(self) -> torch.device:
        return self._trafo.device

    def to(self, device: str | torch.device) -> "XYZChanneledTransformation":
        return XYZChanneledTransformation(self._trafo.to(device), self._ch)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.dim() != 3:
            raise ValueError("Input must be N x C x D")

        xyz = xyz.clone()
        xyz[:, self._ch] = self._trafo.forward(xyz[:, self._ch])
        return xyz


class XYZTransformationGeneric(XYZTransformation):
    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        len: Optional[int] = None,
        device: str | torch.device | None = "cpu",
        device_arbitrary: bool = False,
    ):
        """
        Wrapper for a transformation function.

        Args:
            fn: transformation function
            len: manual length of the transformation, used downstream
            device: device the function is defined on
            device_arbitrary: whether the device is arbitrary, e.g. for a function that
             is itself arbitrary in device (e.g. lambda x: x[[1, 0]], which runs
             on the same device as the input)
        """
        self._fn = fn
        self._len = len
        self._device = torch.device(device) if isinstance(device, str) else device
        self._device_arbitrary = device_arbitrary

    def __len__(self) -> int:
        if self._len is None:
            raise ValueError("Length not specified.")
        return self._len

    def __repr__(self):
        return f"XYZTransformationGeneric({self._fn}, {self._len}, {self._device}, {self._device_arbitrary})"

    @property
    def device(self) -> torch.device:
        if self._device is None and not self._device_arbitrary:
            raise ValueError("Device not specified.")
        return self._device

    def to(self, device: Union[str, torch.device]) -> "XYZTransformationGeneric":
        """
        Change the device of the transformation. This is somewhat generic on generic
        functions and only does something if in no-op (i.e. same device) or if the
        function is arbitrary in device and device_arbitrary is True.

        Args:
            device: new device
        """
        device = torch.device(device) if isinstance(device, str) else device
        if self._device != device and not self._device_arbitrary:
            raise NotImplementedError(
                f"Cannot change device from {self._device} to {device}."
            )
        else:
            self._device = torch.device(device) if isinstance(device, str) else device
        return self

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self._fn(xyz)


class XYZShiftTransformation(XYZTransformation):
    def __init__(
        self,
        shift: float | Sequence[float] | torch.Tensor,
        device: str | torch.device = "cpu",
    ):
        """
        Shift coordinates by a given amount.

        Args:
            shift: shift amount
            device: device to use
        """
        super().__init__()
        self._shift = torch.as_tensor(shift, device=device)
        self._device = torch.device(device) if isinstance(device, str) else device

    def __repr__(self):
        return f"XYZShiftTransformation({self._shift}, {self._device})"

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: str | torch.device) -> "XYZShiftTransformation":
        return XYZShiftTransformation(self._shift, device=device)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return xyz + self._shift


class XYZScaleTransformation(XYZTransformation):
    def __init__(
        self,
        scale: float | Sequence[float | Sequence[float]],
        scope: Literal["global", "channeled"] = "global",
        device: str | torch.device = "cpu",
    ):
        """
        Scale coordinates by a given factor.

        Args:
            scale: scale factor, its length must be equal to the number of channels
            scope: `global`, one scale factor for all channels;
             future work: `channeled`, one scale per channel
            device:
        """
        super().__init__()
        self._scale = torch.as_tensor(scale, device=device)
        self._scope = scope

        if self._scope == "global" and self._scale.dim() >= 2:
            raise ValueError("Global scale must be a scalar or a vector of dim 1.")

    def __repr__(self):
        return f"XYZScaleTransformation({self._scale}, {self._scope})"

    @property
    def device(self) -> torch.device:
        return self._scale.device

    def to(self, device: Union[str, torch.device]) -> "XYZScaleTransformation":
        return XYZScaleTransformation(
            scale=self._scale, scope=self._scope, device=device
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        match self._scope:
            case "global":
                xyz = xyz * self._scale
            case "channeled":
                raise NotImplementedError("Not yet implemented.")
            case _:
                raise ValueError(f"Unknown scope: {self._scope}")

        return xyz


@deprecated(reason="use functional `offset_trafo` instead.")
class XYZOffsettedTransformation(XYZTransformation):
   def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

   def to(self, device: torch.device):
        raise NotImplementedError

   @property
   def device(self) -> torch.device:
        raise NotImplementedError


class XYZMirrorAt(XYZTransformation):
    def __init__(self, axis: int, at: float, device: str | torch.device = "cpu"):
        """
        Mirror coordinates at a given axis and position.

        Note:
            For a frame pixel flip of a frame of size `N` along axis `a`
            you should use `XYZMirrorAt(a, N - 1)`.

        Args:
            axis: axis to mirror at
            at: position to mirror at
        """
        self._axis = axis
        self._at = at
        self._device = torch.device(device) if isinstance(device, str) else device

    def __repr__(self):
        return f"XYZMirrorAt(axis={self._axis}, at={self._at}, device={self._device})"

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: Union[str, torch.device]) -> "XYZMirrorAt":
        # no real device here, so we can just modify inplace.
        self._device = torch.device(device) if isinstance(device, str) else device
        return self

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        xyz = xyz.clone()
        xyz[..., self._axis] = self._at - xyz[..., self._axis]

        return xyz

    @classmethod
    def from_frame_flip(cls, size: int, axis: int, device: str | torch.device = "cpu"):
        """
        Create a mirror transformation that is equivalent to a frame flip along the
        given axis for a frame with default reference.

        Args:
            size: size of the frame
            axis: axis to flip along
            device: device to use
        """
        return cls(axis=axis, at=size - 1, device=device)


class XYZTransformationMatrix(XYZTransformation):
    def __init__(self, m: torch.Tensor):
        """
        Transform coordinates by (Nx)3x3 matrix. Outputs are coordinates, if the
        transformation matrix is batched, the different transformations per coordinate
        are treated as channel dimension, e.g. xyz of size 5 x 3 with matrix of size
        2 x 3 x 3 will lead to new xyz of size 5 x 2 x 3.

        Args:
            m: (Cx)3x3 matrix
        """
        if m.dim() not in {2, 3}:
            raise NotImplementedError("Not supported dimension of m.")
        self._m = m

    def __len__(self) -> int:
        if self._m.dim() != 3:
            raise ValueError("Ill-defined length for non-batched transformation.")
        return self._m.size(0)

    def __repr__(self) -> str:
        return f"XYZTransformationMatrix({self._m})"

    def to(self, device: Union[str, torch.device]) -> "XYZTransformationMatrix":
        return XYZTransformationMatrix(self._m.to(device))

    @property
    def device(self) -> torch.device:
        return self._m.device

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """

        Args:
            xyz: coordinates of size `N x 3` (`N` being the batch dim).
        """
        if xyz.dim() != 2 or xyz.size(-1) != 3:
            raise ValueError(f"Expected xyz of size N x 3. Got {xyz.size()}.")

        z_in = xyz[..., -1].clone()  # backup z
        xyz = xyz.clone()
        xyz[..., -1] = 1  # leave z unchanged

        # apply actual transformation

        # former solution; not compatible with MATLAB, not sure how to test this
        # xyz = xyz @ self._m
        xyz = (self._m.transpose(-2, -1) @ xyz.T.unsqueeze(0)).transpose(-2, -1)

        # batched transformations are treated as channels
        if self._m.dim() == 3:
            xyz = xyz.permute(1, 0, -1)

        xyz[..., -1] = z_in.unsqueeze(-1)
        return xyz


class XYZRotation(XYZTransformation):
    def __init__(
        self,
        angle: Sequence[float],
        offset: Optional[Sequence[float]] = None,
        rebound: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        self._angle = angle
        self._offset = offset
        self._rebound = rebound
        self._device = torch.device(device) if isinstance(device, str) else device

        # construct core trafo
        mat = torch.stack([self._rot_matrix(a) for a in angle], 0).to(device)
        trafo = XYZTransformationMatrix(mat)
        if offset is not None:
            trafo = offset_trafo(trafo, offset, rebound=rebound)

        self._mat = mat
        self._trafo = trafo

    def __len__(self) -> int:
        return len(self._angle)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: Union[str, torch.device]) -> "XYZRotation":
        return XYZRotation(
            self._angle, self._offset, rebound=self._rebound, device=device
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self._trafo.forward(xyz)

    @staticmethod
    def _rot_matrix(angle: float) -> torch.Tensor:
        r = SciRot.from_rotvec(angle * np.array([0, 0, 1]))
        r = torch.from_numpy(r.as_matrix()).float()
        return r


def offset_trafo(
    t: XYZTransformation,
    offset: Union[Tuple[float, ...], torch.Tensor],
    rebound: bool = True,
) -> XYZCompositTransformation:
    """
    Offset a transformation. This is useful to simulate a coordinate system
    with a different origin, e.g. to sample a position on a camera.

    Args:
        t: transformation to offset
        offset: offset to apply
        rebound: whether to rebound the offset
    """
    offset = torch.as_tensor(offset, device=t.device)
    t = copy.deepcopy(t)

    shift_forward = XYZShiftTransformation(offset, device=t.device)
    shift_backward = XYZShiftTransformation(-offset, device=t.device)

    other = (
        XYZCompositTransformation([t])
        if not isinstance(t, XYZCompositTransformation)
        else t
    )

    other.insert(0, shift_forward)
    if rebound:
        other.append(shift_backward)

    return other
