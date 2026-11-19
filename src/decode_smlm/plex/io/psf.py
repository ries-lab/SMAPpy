import copy
from pathlib import Path
from typing import Union, Literal, Sequence

try:
    import mat73
except ImportError:
    pass
import scipy.io
import structlog
import torch
from deprecated import deprecated
import h5py
from omegaconf import OmegaConf, DictConfig
from dotted_dict import DottedDict
import numpy as np

from ..generic import deploy
from ..simulation import psf_kernel
from ..utils import types
from ..utils import spline

logger = structlog.get_logger()


@deploy.raise_optional_deps(
    "mat73", "mat73 not available, please use `pip install mat73`."
)
def load_spline_smap(
    path: Union[str, Path],
    xextent: tuple[float, float] | tuple[tuple[float, float], ...],
    yextent: tuple[float, float] | tuple[tuple[float, float], ...],
    img_shape: tuple[int, int] | tuple[tuple[int, int], ...],
    locality: Literal["SXY", "SXY_g", "auto"] = "auto",
    norm: float | Sequence[float] | None = None,
    device: str | torch.device = "cpu",
    kwargs_static: dict | None = None,
    kwargs_multi: dict | None = None,
) -> psf_kernel.CubicSplinePSF | list[psf_kernel.CubicSplinePSF]:
    """
    Load spline psf from calibration file. Currently old and new style `.mat` are supported.
    This loader most likely expects calibration files from SMAP. It is capable of loading
    multi-channel psf.

    Args:
        path: path to calibration file
        xextent: x extent of psf, or list of x extents for multi-channel
        yextent: y extent of psf, or list of y extents for multi-channel
        img_shape: image shape, or list of image shapes for multi-channel
        locality: whether to load global or local spline. If `auto`,
         global will be loaded if available
        norm: normalization factor, or list of normalization factors for multi-channel
        device: device to load psf on
        kwargs_static: arbitrary kwargs to pass on to `CubicSplinePSF`
        kwargs_multi: arbitrary kwargs to pass on to `CubicSplinePSF`
         per channel in multi-channel

    """
    kwargs_static = {} if kwargs_static is None else kwargs_static
    kwargs_multi = {} if kwargs_multi is None else kwargs_multi

    path = path if isinstance(path, Path) else Path(path)

    multi_channel = True if isinstance(xextent[0], (list, tuple)) else False

    try:
        data = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
        data = types.RecursiveNamespace(**data)
        if locality == "auto":
            locality = "SXY_g" if hasattr(data, "SXY_g") else "SXY"
        calib = getattr(data, locality)

        if multi_channel:
            coeff = [torch.from_numpy(c) for c in calib.cspline.coeff]
        else:
            coeff = torch.from_numpy(calib.cspline.coeff)

    except NotImplementedError:
        calib = mat73.loadmat(path, use_attrdict=False)
        calib = types.RecursiveNamespace(**calib).SXY
        coeff = torch.from_numpy(calib.cspline.coeff[0])

    ref0 = (
        calib.cspline.x0 - 1,
        calib.cspline.x0 - 1,
        float(calib.cspline.z0),
    )

    dz = calib.cspline.dz

    kwargs_static.setdefault("device", device)
    kwargs_static.setdefault("vx_size", (1.0, 1.0, dz))
    kwargs_static.setdefault("ref0", ref0)

    if norm == "auto":
        # load normalization from calibration file directly
        if not multi_channel:
            raise NotImplementedError(
                "Auto-loading normalization only supported for multi-channel psf"
            )
        norm = calib.cspline.normf.tolist()
        logger.info("Auto-loaded normalization from calibration file", norm=norm)

    if norm is not None and multi_channel:
        kwargs_multi["norm"] = norm
    if norm is not None and not multi_channel:
        kwargs_static["norm"] = norm

    if multi_channel:
        psf_multi = []
        for i, (c, x, y, s) in enumerate(
            zip(coeff, xextent, yextent, img_shape, strict=True)
        ):
            kwargs = copy.copy(kwargs_static)
            kwargs.update({k: v[i] for k, v in kwargs_multi.items()})
            psf = psf_kernel.CubicSplinePSF(
                coeff=c, xextent=x, yextent=y, img_shape=s, **kwargs
            )
            psf_multi.append(psf)
        return psf_multi

    return psf_kernel.CubicSplinePSF(
        coeff=coeff,
        xextent=xextent,
        yextent=yextent,
        img_shape=img_shape,
        **kwargs_static
    )

def h5_to_dict(group):
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = h5_to_dict(item)  # Recursive call for groups
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]  # Convert dataset to NumPy array or scalar
    return DottedDict(result)


def load_h5(path: Union[str, Path]) -> DictConfig:
    f = h5py.File(path, "r")
    res = h5_to_dict(f)
    params = OmegaConf.create(f.attrs["params"])
    return res, params

@deploy.raise_optional_deps(
    "h5py", "h5py not available, please use `pip install h5py`."
)
def load_spline_uiPSF(
    path: Union[str, Path],
    xextent: tuple[float, float] | tuple[tuple[float, float], ...],
    yextent: tuple[float, float] | tuple[tuple[float, float], ...],
    img_shape: tuple[int, int] | tuple[tuple[int, int], ...],
    device: str | torch.device = "cpu",
    norm: float | Sequence[float] | None = None,
    kwargs_static: dict | None = None,
    kwargs_multi: dict | None = None,
) -> list[psf_kernel.CubicSplinePSF] | psf_kernel.CubicSplinePSF:
    """
    Load spline psf from calibration file. Support '.h5' files.
    This loader most likely expects calibration files from uiPSF. It is capable of loading
    multi-channel psf.

    Args:
        path: path to calibration file
        xextent: x extent of psf, or list of x extents for multi-channel
        yextent: y extent of psf, or list of y extents for multi-channel
        img_shape: image shape, or list of image shapes for multi-channel
        device: device to load psf on
        kwargs_static: arbitrary kwargs to pass on to `CubicSplinePSF`
        kwargs_multi: arbitrary kwargs to pass on to `CubicSplinePSF`
        per channel in multi-channel

    """

    kwargs_static = {} if kwargs_static is None else kwargs_static
    kwargs_multi = {} if kwargs_multi is None else kwargs_multi

    path = path if isinstance(path, Path) else Path(path)

    multi_channel = True if isinstance(xextent[0], (list, tuple)) else False

    f, p = load_h5(path)

    if multi_channel:
        coeff = []
        for channel in f.res.keys():
            if "channel" in channel:
                coeff.append(
                    torch.from_numpy(
                        spline.psf_to_cspline(
                             f.res[channel].I_model.transpose(1, 2, 0) # f.res[channel].I_model.transpose(2, 1, 0) / 2  
                        ).transpose(1, 2, 3, 0)
                    )
                )
    else:
        coeff = torch.from_numpy(
            spline.psf_to_cspline(
                f.res.I_model.transpose(1, 2, 0)
            ).transpose(1, 2, 3, 0)
        )

    pixel_size_z = p.pixel_size['z'] * 1000
    # ToDO: check parameters
    kwargs_static.setdefault("device", device)
    kwargs_static.setdefault("vx_size", (1.0, 1.0, pixel_size_z))  # ?

    if norm == "auto":
        # load normalization from calibration file directly
        if not multi_channel:
            raise NotImplementedError(
                "Auto-loading normalization only supported for multi-channel psf"
            )
        norm = [1., f.res.channel1.I_model.sum()/f.res.channel0.I_model.sum()]
        logger.info("Auto-loaded normalization from calibration file", norm=norm)

    if norm is not None and multi_channel:
        kwargs_multi["norm"] = norm
            
    if multi_channel:
        kwargs_static.setdefault(
            "ref0",
            (
                f.res.channel0.I_model.shape[-2] // 2,
                f.res.channel0.I_model.shape[-1] // 2,
                f.res.channel0.I_model.shape[-3] // 2,
            ),
        )  # We assume here that the focal point lies at the middle slice in the bead data
    else:
        kwargs_static.setdefault(
            "ref0",
            (
                f.res.I_model.shape[-2] // 2,
                f.res.I_model.shape[-1] // 2,
                f.res.I_model.shape[-3] // 2,
            ),
        )

    if multi_channel:
        psf_multi = []
        for i, (c, x, y, s) in enumerate(
            zip(coeff, xextent, yextent, img_shape, strict=True)
        ):
            kwargs = copy.copy(kwargs_static)
            kwargs.update({k: v[i] for k, v in kwargs_multi.items()})
            psf = psf_kernel.CubicSplinePSF(
                coeff=c, xextent=x, yextent=y, img_shape=s, **kwargs
            )
            psf_multi.append(psf)
        return psf_multi

    return psf_kernel.CubicSplinePSF(
        coeff=coeff,
        xextent=xextent,
        yextent=yextent,
        img_shape=img_shape,
        **kwargs_static,
    )


@deploy.raise_optional_deps(
    "mat73", "mat73 not available, please use `pip install mat73`."
)
def load_spline(
    path: Union[str, Path],
    xextent: tuple[float, float] | tuple[tuple[float, float], ...],
    yextent: tuple[float, float] | tuple[tuple[float, float], ...],
    img_shape: tuple[int, int] | tuple[tuple[int, int], ...],
    locality: Literal["SXY", "SXY_g", "auto"] = "auto",
    norm: float | Sequence[float] | None = None,
    device: str | torch.device = "cpu",
    kwargs_static: dict | None = None,
    kwargs_multi: dict | None = None,
) -> psf_kernel.CubicSplinePSF | list[psf_kernel.CubicSplinePSF]:
    """
    Load spline psf from calibration file. Currently old and new style `.mat` are supported.
    This loader most likely expects calibration files from SMAP. It is capable of loading
    multi-channel psf.

    Args:
        path: path to calibration file
        xextent: x extent of psf, or list of x extents for multi-channel
        yextent: y extent of psf, or list of y extents for multi-channel
        img_shape: image shape, or list of image shapes for multi-channel
        locality: whether to load global or local spline. If `auto`,
         global will be loaded if available
        norm: normalization factor, or list of normalization factors for multi-channel
        device: device to load psf on
        kwargs_static: arbitrary kwargs to pass on to `CubicSplinePSF`
        kwargs_multi: arbitrary kwargs to pass on to `CubicSplinePSF`
         per channel in multi-channel

    """

    path = path if isinstance(path, Path) else Path(path)

    match path.suffix:
        case ".mat":
            return load_spline_smap(
                path,
                xextent=xextent,
                yextent=yextent,
                img_shape=img_shape,
                locality=locality,
                norm=norm,
                device=device,
                kwargs_static=kwargs_static,
                kwargs_multi=kwargs_multi,
            )

        case ".h5":
            return load_spline_uiPSF(
                path,
                xextent=xextent,
                yextent=yextent,
                img_shape=img_shape,
                device=device,
                norm=norm,
                kwargs_static=kwargs_static,
                kwargs_multi=kwargs_multi,
            )
        case _:
            raise ValueError("Unknown calibration file format.")
            

@deprecated(version="0.11.0", reason="use functional interface `load_spline` instead.")
class SMAPSplineCoefficient:
    pass
