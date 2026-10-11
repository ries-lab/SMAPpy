"""uiPSF's result file -> smappy's calibration file.

uiPSF writes the PSF it learnt as voxels (``res/I_model``, z, y, x) beside
the pupil and its Zernike coefficients, and a cubic spline of it that SMAP
reads (``locres/coeff``).  The spline is not used here: uiPSF normalises it by
the median plane sum after subtracting the minimum, which is not what the
fitter's photon number assumes, and its grid is uiPSF's.  The voxels are
taken instead and made into a calibration the way smappy's own bead
calibration makes one (`calibrate.core._single_model`,
`positive_pair_models`): clipped at zero, normalised, splined by
`spline_coefficients` -- so the fitter cannot tell where a calibration came
from, and the pupil and coefficients go along in ``parameters`` as the
record of what was learnt.

Conventions, each checked by fitting simulated data with the result
(tests/test_uipsf.py):

* **z.**  A bead model's planes are uiPSF's ``Zrange``, centred on the
  pupil's focus and in the order of the stack (objective moving up), which is
  smappy's: ``z0 = (nz - 1) / 2``, emitter ``z = -(index - z0) dz``.  An in
  situ model's planes run the other way -- the emitter's height above the
  coverslip -- so they are reversed, and z = 0 is put at the height the
  objective's focus sat at (uiPSF's learnt ``stagepos``, scaled by the
  refractive indices), which is the z a bead calibration calls 0.
* **x, y.**  numpy's (y, x) throughout; nothing is swapped, because smappy
  hands uiPSF arrays rather than files (uiPSF's ``swapxy`` is for SMAP's
  transposed TIFF reading).
* **two channels.**  uiPSF's ``T`` maps the reference channel's (y, x, 1) row
  vectors, about ``imgcenter``, to the other channel's, both in the arrays it
  was handed; `prepare.Split.to_chip` puts both into chip pixels, and the
  projective map smappy wants (secondary -> main, x, y) is fitted to points
  pushed through it.  The secondary PSF is flipped back to the camera's
  orientation, as smappy's dual calibration stores it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

NOLL_NAMES = {1: "piston", 2: "tilt x", 3: "tilt y", 4: "defocus",
              5: "astigmatism 45", 6: "astigmatism 0", 7: "coma y", 8: "coma x",
              9: "trefoil y", 10: "trefoil x", 11: "spherical",
              12: "2nd astigmatism 0", 13: "2nd astigmatism 45",
              22: "2nd spherical", 37: "3rd spherical"}


def read(path) -> Dict:
    """uiPSF's result: ``{"res": ..., "locres": ..., "params": ...}`` as dicts."""
    import h5py

    def group(g):
        return {k: group(v) if isinstance(v, h5py.Group) else v[()] for k, v in g.items()}
    with h5py.File(path, "r") as f:
        if "res" not in f or "params" not in f.attrs:
            raise ValueError(f"{path} is not a uiPSF result (no res group or params)")
        out = {k: group(f[k]) for k in f if isinstance(f[k], h5py.Group)}
        out["params"] = json.loads(f.attrs["params"])
    return out


def _is_insitu(params) -> bool:
    return "insitu" in str(params.get("PSFtype", ""))


def zernike_table(res: Dict, params: Dict) -> list:
    """(Noll index, name, magnitude, phase in nm rms) for each learnt term."""
    coeff = res.get("zernike_coeff")
    if coeff is None:
        return []
    coeff = np.asarray(coeff).reshape(2, -1)
    nl = params.get("option", {}).get("model", {}).get("zernike_nl") or []
    if nl:
        noll = [_nl2noll(n, l) for n, l in nl]
    else:
        noll = list(range(1, coeff.shape[1] + 1))
    wavelength_nm = 1000 * float(params["option"]["imaging"]["emission_wavelength"])
    return [(int(j), NOLL_NAMES.get(int(j), ""), float(m), float(p) * wavelength_nm / (2 * np.pi))
            for j, m, p in zip(noll, coeff[0], coeff[1])]


def _nl2noll(n, l):
    """uiPSF's `nl2noll`: Noll's index of radial order n, azimuthal l."""
    m = abs(l)
    j = n * (n + 1) // 2 + 1 + max(0, m - 1)
    if (l > 0 and n % 4 >= 2) or (l < 0 and n % 4 <= 1):
        j += 1
    return int(j)


def _focus_index(res: Dict, params: Dict, nz: int) -> float:
    """Where z = 0 lies in an in situ model, in its (reversed) planes.

    uiPSF's in situ planes are emitter heights ``zoffset + i`` (in z steps)
    above the coverslip, the objective's focus sitting at ``stagepos`` (um)
    of stage travel, which in the sample is the nominal focal plane
    ``stagepos * n_medium / n_immersion`` above it.  That plane is z = 0, as
    a bead stack's centre is, and the planes are reversed to run as a bead
    stack does; outside the model's range the middle is taken instead.
    """
    dz_um = float(params["pixel_size"]["z"])
    ri = params["option"]["imaging"]["RI"]
    stage = float(np.ravel(res.get("stagepos", [params["option"]["insitu"]["stage_pos"]]))[0])
    zoffset = float(np.real(np.ravel(res.get("zoffset", [0.0]))[0]))
    height = stage * float(ri["med"]) / float(ri["imm"]) / dz_um      # in planes
    index = height - zoffset                                           # unreversed
    reversed_index = (nz - 1) - index
    return float(reversed_index) if 0 <= reversed_index <= nz - 1 else (nz - 1) / 2


def _model(res: Dict, params: Dict):
    """The channel's PSF voxels in smappy's z order, and its z0."""
    psf = np.asarray(res["I_model"], np.float64)
    if psf.ndim != 3:
        raise ValueError(f"uiPSF's I_model has shape {psf.shape}; a 3D model was expected "
                         "(field-dependent and 4Pi models are not supported)")
    if _is_insitu(params):
        z0 = _focus_index(res, params, len(psf))
        psf = psf[::-1]
    else:
        z0 = (len(psf) - 1) / 2
    return np.real(psf), z0


def _record(params: Dict, res: Dict, path, **extra) -> Dict:
    imaging = params["option"]["imaging"]
    out = {"method": "uipsf_" + str(params.get("PSFtype")),
           "source": str(path),
           "uipsf_psftype": params.get("PSFtype"),
           "normalization": "maximum plane sum = 1",
           "z_reference": "uiPSF focus" if not _is_insitu(params)
                          else "nominal focal plane (stagepos * n_med / n_imm)",
           "z_convention": "emitter_nm = -(index-z0)*dz",
           "lateral_unit": "pixel",
           "NA": imaging["NA"], "refractive_index": imaging["RI"],
           "emission_wavelength_um": imaging["emission_wavelength"],
           "pixel_size_um": params["pixel_size"],
           "zernike": [list(row) for row in zernike_table(res, params)]}
    for key in ("stagepos", "zoffset", "sigma"):
        if key in res:
            out["uipsf_" + key] = np.real(np.ravel(res[key])).astype(float).tolist()
    out.update(extra)
    return out


def single_calibration(path, em_on: Optional[bool] = None):
    """A `SplineCalibration` of a one-channel uiPSF result."""
    from ..calibrate.core import spline_coefficients
    from ..io.calibration import SplineCalibration
    data = read(path)
    res, params = data["res"], data["params"]
    if "channel0" in res:
        raise ValueError(f"{path} is a multi-channel uiPSF result: use dual_calibration")
    psf, z0 = _model(res, params)
    psf = np.maximum(psf, 0)
    norm = float(psf.sum(axis=(1, 2)).max())
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("uiPSF's model has no positive signal")
    psf = (psf / norm).astype(np.float32)
    return SplineCalibration(spline_coefficients(psf).astype(np.float32),
                             1000 * float(params["pixel_size"]["z"]), z0,
                             x0=(psf.shape[1] - 1) / 2, psf=psf, em_mirror=False,
                             source=Path(path), em_on=em_on,
                             parameters=_record(params, res, path))


# the planes either side of focus the channels' light is compared over, as
# smappy's dual calibration does (calibrate.core.FOCAL_PLANES)
FOCAL_PLANES = 2


def dual_calibration(path, split, sources=(), em_on: Optional[bool] = None):
    """A `DualColorCalibration` of a two-channel uiPSF result.

    ``split`` is the `prepare.Split` the channels were cut with: without it
    the transformation cannot be put back on the chip.
    """
    from ..calibrate.core import spline_coefficients
    from ..calibrate.dual import DualColorCalibration
    from ..io.calibration import SplineCalibration
    data = read(path)
    res, params = data["res"], data["params"]
    if "channel1" not in res or "T" not in res:
        raise ValueError(f"{path} is not a two-channel uiPSF result")
    models, intensities, light = [], [], []
    for k in (0, 1):
        psf, z0 = _model(res[f"channel{k}"], params)
        models.append((psf, z0))
        centre = int(round(z0))
        window = slice(max(centre - FOCAL_PLANES, 0), centre + FOCAL_PLANES + 1)
        # the channel's light per emitter at focus: the model's focal signal
        # times the photons uiPSF fitted to it -- uiPSF normalises each
        # channel's model on its own, so the split between them is in the
        # intensities
        intensities.append(float(np.median(np.real(res[f"channel{k}"]["intensity"]))))
        light.append(float(psf[window].sum(axis=(1, 2)).mean()) * intensities[-1])
    if models[0][0].shape != models[1][0].shape or models[0][1] != models[1][1]:
        raise ValueError("uiPSF's two channel models differ in their grid")
    total = sum(light)
    if not np.isfinite(total) or total <= 0 or min(light) <= 0:
        raise ValueError("uiPSF's channel intensities give no valid photon split")
    mirror = split.geometry(sources)["mirror_axis_xy"]
    out = []
    for k, ((psf, z0), share) in enumerate(zip(models, light)):
        # one factor for the pair, as `positive_pair_models`: the channels'
        # light around focus sums to one, each keeping its share
        psf = np.maximum(psf, 0) * intensities[k] / total
        if k == 1 and mirror is not None:
            psf = np.flip(psf, axis=2 - mirror)       # back to the camera's orientation
        psf = np.ascontiguousarray(psf, np.float32)
        out.append(SplineCalibration(
            spline_coefficients(psf).astype(np.float32), 1000 * float(params["pixel_size"]["z"]),
            z0, x0=(psf.shape[1] - 1) / 2, psf=psf, em_mirror=False, source=Path(path),
            em_on=em_on, parameters=_record(
                params, res[f"channel{k}"], path,
                normalization="joint: both channels' light around focus sums to 1",
                photon_normalization=share / total)))
    transformation = channel_transformation(res["T"], res["imgcenter"], split)
    calibration_parameters = {
        "method": "uipsf_" + str(params.get("PSFtype")),
        "source": str(path),
        "secondary_main_brightness_ratio": light[1] / light[0],
        "transform_method": "uiPSF's affine channel transformation, learnt with the PSF",
        "transform_direction": "secondary to main, native camera coordinates",
        "uipsf_T": np.asarray(res["T"]).tolist(),
        "uipsf_imgcenter": np.asarray(res["imgcenter"]).tolist()}
    return DualColorCalibration(out[0], out[1], transformation,
                                split.geometry(sources), calibration_parameters)


def uipsf_to_main(T, imgcenter, yx) -> np.ndarray:
    """uiPSF's channel map: reference (y, x) -> the other channel's (y, x)."""
    T = np.asarray(T, float).reshape(-1, 3, 3)[0]
    c = np.asarray(imgcenter, float).ravel()
    rows = np.c_[np.asarray(yx, float), np.ones(len(yx))] - c
    return (rows @ T)[:, :2] + c[:2]


def channel_transformation(T, imgcenter, split) -> np.ndarray:
    """smappy's 3x3 secondary -> main chip (x, y) map from uiPSF's ``T``."""
    from ..calibrate.dual import fit_projective
    n = split.length
    other = (split.image_shape[1 - split.axis])
    shape = [0, 0]
    shape[split.axis], shape[1 - split.axis] = n, other
    yy, xx = np.meshgrid(np.linspace(0, shape[0] - 1, 9), np.linspace(0, shape[1] - 1, 9),
                         indexing="ij")
    main_yx = np.c_[yy.ravel(), xx.ravel()]
    secondary_yx = uipsf_to_main(T, imgcenter, main_yx)
    return fit_projective(split.to_chip(1, secondary_yx), split.to_chip(0, main_yx))


def save(path, calibration, overwrite: bool = True) -> Path:
    """Write either kind as smappy's calibration file."""
    from ..calibrate.dual import DualColorCalibration, save_dual_color_calibration
    from ..io.calibration import save_spline_calibration
    path = Path(path)
    if isinstance(calibration, DualColorCalibration):
        save_dual_color_calibration(path, calibration, overwrite=overwrite)
    else:
        save_spline_calibration(path, calibration, overwrite=overwrite)
    return path
