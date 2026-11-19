from abc import ABC, abstractmethod

import torch


class FrameProcessing(ABC):
    @abstractmethod
    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Forward frame through processing implementation.

        Args:
            frame:

        """
        raise NotImplementedError


class Mirror2D(FrameProcessing):
    def __init__(self, dims: int | tuple[int, ...]):
        """
        Mirror the specified dimensions. Providing dim index in negative format is recommended.
        Given format N x C x H x W and you want to mirror H and W set dims=(-2, -1).

        Args:
            dims: dimensions

        """
        super().__init__()

        self.dims = dims

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return frame.flip(self.dims)


class Crop(FrameProcessing):
    def __init__(self, crop_to: tuple[int, int] | tuple[slice, slice]):
        """
        Crop frame to specified size

        Args:
            crop_to: size to crop to
        """
        super().__init__()
        self._crop_to = [self.as_slice(c) for c in crop_to]

    @staticmethod
    def as_slice(crop: int | tuple[int, int] | slice) -> slice:
        if isinstance(crop, int):
            return slice(crop)
        elif isinstance(crop, tuple):
            return slice(*crop)
        elif isinstance(crop, slice):
            return crop
        else:
            raise ValueError

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Process frames

        Args:
            frame: size [*, H, W]

        Returns:

        """
        # return frame[..., : self._crop_to[0], : self._crop_to[1]]
        return frame[..., self._crop_to[0], self._crop_to[1]]


def get_pxfold_tar(size_is: int | torch.Tensor, px_fold: int) -> int:
    """Compute target size if frame needs to be cropped to a multiple of px_fold"""
    tar = (size_is // px_fold) * px_fold
    if (isinstance(tar, int) and tar <= 0) or (
        isinstance(tar, torch.Tensor) and (tar <= 0).any()
    ):
        raise ValueError("Got negative target size")
    return tar


class AutoLeftUpperCrop(FrameProcessing):
    def __init__(self, px_fold: int):
        """
        Automatic cropping in left upper corner. Specify pixel_fold which the target frame size must satistfy
        and the frame will be left upper cropped to this size.

        Args:
            px_fold: integer in which multiple the frame must dimensioned (H, W dimension)

        """
        super().__init__()
        self.px_fold = px_fold

        if not isinstance(self.px_fold, int):
            raise ValueError

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Process frames

        Args:
            frame: size [*, H, W]

        Returns:

        """
        if self.px_fold == 1:
            return frame

        size_tar = get_pxfold_tar(torch.tensor(frame.size())[-2:], self.px_fold)
        return frame[..., : size_tar[0], : size_tar[1]]


class AutoCenterCrop(AutoLeftUpperCrop):
    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Process frames

        Args:
            frame: size [*, H, W]

        """
        if self.px_fold == 1:
            return frame

        size_is = torch.tensor(frame.size())[-2:]
        size_tar = get_pxfold_tar(size_is, self.px_fold)

        # crop
        ix_front = ((size_is - size_tar).float() / 2).ceil().long()
        ix_back = ix_front + size_tar

        return frame[..., ix_front[0] : ix_back[0], ix_front[1] : ix_back[1]]


class AutoPad(AutoCenterCrop):
    def __init__(self, px_fold: int, mode: str = "constant"):
        """
        Pad frame to a size that is divisible by px_fold. Useful to prepare
        an experimental frame for forwarding through network.

        Args:
            px_fold: number of pixels the resulting frame size should be divisible by
            mode: torch mode for padding. refer to docs of `torch.nn.functional.pad`
        """
        super().__init__(px_fold=px_fold)
        self.mode = mode

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        if self.px_fold == 1:
            return frame

        size_is = torch.tensor(frame.size())[-2:]
        size_tar = torch.ceil(size_is / self.px_fold) * self.px_fold
        size_tar = size_tar.long()

        size_pad = size_tar - size_is
        size_pad_div = size_pad // 2
        size_residual = size_pad - size_pad_div

        size_pad_lr_ud = [
            size_pad_div[1].item(),
            size_residual[1].item(),
            size_pad_div[0].item(),
            size_residual[0].item(),
        ]

        return torch.nn.functional.pad(frame, size_pad_lr_ud, mode=self.mode)


def get_frame_extent(size, func) -> torch.Size:
    """
    Get frame extent after processing pipeline

    Args:
        size:
        func:

    Returns:

    """

    if len(size) == 4:  # avoid to forward large batches just to get the output extent
        n_batch = size[0]
        size_out = func(torch.zeros(2, *size[1:])).size()
        return torch.Size([n_batch, *size_out[1:]])

    else:
        return func(torch.zeros(*size)).size()
