try:
    from findpeaks import findpeaks
except ImportError:
    findpeaks = None
from typing import Literal

import torch


class ModelPeakfinder:
    _formats_allowed = ["mask", "list"]
    def __init__(
        self,
        format: Literal["mask", "list"] = "mask",
        method: Literal["topology", "mask"] = "topology",
        whitelist=("peak", "valley"),
        lookahead=200,
        interpolate=None,
        limit=None,
        imsize=None,
        scale=True,
        togray=True,
        denoise="fastnl",
        window=3,
        cu=0.25,
        params_caerus={"window": 50, "minperc": 3, "nlargest": 10, "threshold": 0.25},
        verbose=3,
    ):
        """
        Very simple peakfinder model, that just creates a mask with the peaks
        of the input. Args stem from peakfinder.

        Args:
            format: output format, either like original frame as mask or
             as list of coordinates and its px values
            method : String, (default : None).
            Available methods for peak detection. In case method=None, the default is choosen.
            2d-array approaches:
                * 'topology' (default)
                * 'mask'
            whitelist : str or list ['peak','valley']
                Choose what to detect:
                    * 'peak'
                    * 'valley'
                    * ['peak','valley']
            lookahead : int, (default : 200)
                Looking ahead for peaks. For very small 1d arrays (such as up to 50 datapoints), use low numbers such as 1 or 2.
            interpolate : int, (default : None)
                Interpolation factor. The higher the number, the less sharp the edges will be.
            limit : float, (default : None)
                In case method='topology'
                Values > limit are active search areas to detect regions of interest (ROI).
            scale : bool, (default : False)
                Scaling in range [0-255] by img*(255/max(img))
            denoise : string, (default : 'fastnl', None to disable)
                Filtering method to remove noise:
                    * None
                    * 'fastnl'
                    * 'bilateral'
                    * 'lee'
                    * 'lee_enhanced'
                    * 'kuan'
                    * 'frost'
                    * 'median'
                    * 'mean'
            window : int, (default : 3)
                Denoising window. Increasing the window size may removes noise better but may also removes details of image in certain denoising methods.
            cu : float, (default: 0.25)
                The noise variation coefficient, applies for methods: ['kuan','lee','lee_enhanced']
            params : dict() (Default: None)
                caerus parameters can be defined in this dict. If None defined, then all default caerus parameters are used:
                {'window': 50, 'minperc': 3, 'nlargest': 10, 'threshold': 0.25}
            togray : bool, (default : False)
                Conversion to gray scale.
            imsize : tuple, (default : None)
                size to desired (width,length).
            verbose : int (default : 3)
                Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.
        """
        if findpeaks is None:
            raise ImportError("Optional dependency `findpeaks` not installed")

        if format not in self._formats_allowed:
            raise ValueError(f"Format must be one of {self._formats_allowed}")

        self._format = format

        self._model = findpeaks(
            method=method,
            whitelist=whitelist,
            lookahead=lookahead,
            interpolate=interpolate,
            limit=limit,
            imsize=imsize,
            scale=scale,
            togray=togray,
            denoise=denoise,
            window=window,
            cu=cu,
            params_caerus=params_caerus,
            verbose=verbose,
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self.forward_mask(x)

        match self._format:
            case "mask":
                return mask
            case "list":
                return self.forward_list(x, mask)

    def forward_mask(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            raise ValueError(f"Expected tensor of dim 3, got size {x.size()}")

        result = [self._model.fit(xx.numpy()) for xx in x]
        result = torch.stack([torch.as_tensor(r["Xdetect"]) for r in result])

        return result

    def forward_list(
        self, x: torch.Tensor, mask: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        det = mask.nonzero()
        frame_ix = det[:, 0]
        xyz = det[:, 1:]
        phot = x[frame_ix, xyz[:, 0], xyz[:, 1]]

        return frame_ix, xyz, phot
