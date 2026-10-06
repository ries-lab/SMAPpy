"""PSF models learnt by uiPSF, made into smappy calibrations.

uiPSF (Liu et al. 2024, https://github.com/ries-lab/uiPSF) learns a PSF by
inverse modelling: a pupil -- Zernike polynomials, or free -- is propagated
to the camera, and its parameters are fitted to the data with the emitters'
positions.  From bead stacks that is a calibration like smappy's own,
with a physical model behind it; from the blinking molecules of the
acquisition itself (*in situ*) it is the PSF the data were actually taken
with, aberrations of the sample and all, which no bead on a coverslip shows.

The work is split four ways, so that each part can be read and tested alone:

* `prepare` reads the data, converts it to photons, and cuts a split camera
  into its two channels, recording exactly how;
* `uipsf_parameters` turns a microscope profile (`microscopes`) and the
  user's settings into uiPSF's configuration;
* `runner` runs uiPSF -- in its own Python, through `worker.py` -- and
  follows its progress;
* `convert` makes its result into smappy's calibration file, which the
  Spline 3D and Spline 3D 2C fitters read as they read a bead calibration.

`learn` does all four.  The plugin is ``Localize/uiPSF calibration``
(`smappy.plugins.uipsf`).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Optional

from .microscopes import Microscope

# uiPSF's configuration files by what is learnt from what
PSFTYPES = {("beads", "zernike"): ("zernike", None),
            ("beads", "voxel"): ("voxel", None),
            ("blinking", "zernike"): ("insitu", None),
            ("blinking", "pupil"): ("insitu", "insitu_pupil")}


def uipsf_parameters(profile: Microscope, *, data: str, pixelsize_xy_um, dz_nm: float,
                     roi_size_px: Optional[int] = None, emission_wavelength_nm: Optional[float] = None,
                     peak_height: float = 0.2, max_beads: int = 40, bead_diameter_nm: float = 0.0,
                     iterations: int = 100, zernike_order: int = 8,
                     depth_um: float = 1.0, z_range_um: float = 2.0, min_photon: float = 0.4,
                     z_bins: int = 21, per_bin: int = 100, repeat: int = 2,
                     start_from: str = "") -> dict:
    """uiPSF's parameters, nested as its YAML, for one run.

    What the profile does not say is uiPSF's default; what the profile's own
    ``uipsf`` section says is applied last, so a microscope with an unusual
    need (a deformable mirror's own Zernike terms) can say so there.
    """
    ri = profile.refractive_index
    roi = int(roi_size_px or profile.roi_size_px)
    params = {
        "pixel_size": {"x": float(pixelsize_xy_um[0]), "y": float(pixelsize_xy_um[1]),
                       "z": float(dz_nm) / 1000},
        "roi": {"roi_size": [roi, roi], "peak_height": float(peak_height),
                "max_bead_number": int(max_beads),
                "bead_radius": float(bead_diameter_nm) / 2000},
        "gain": 1.0, "ccd_offset": 0.0,        # smappy hands over photons
        "iteration": int(iterations),
        "option": {
            "imaging": {"emission_wavelength": float(emission_wavelength_nm
                                                     or profile.emission_wavelength_nm) / 1000,
                        "RI": {"imm": float(ri["immersion"]), "med": float(ri["medium"]),
                               "cov": float(ri["coverslip"])},
                        "NA": float(profile.NA)},
            "model": {"n_max": int(zernike_order), "init_pupil_file": start_from or ""},
        },
    }
    if data == "blinking":
        insitu = dict(profile.insitu)
        params["option"]["insitu"] = {
            "stage_pos": float(depth_um), "z_range": float(z_range_um),
            "min_photon": float(min_photon), "partition_size": [int(z_bins), int(per_bin)],
            "repeat": int(repeat),
            "zernike_index": list(insitu.get("zernike_index", [5])),
            "zernike_coeff": list(insitu.get("zernike_coeff", [0.5]))}
    _merge(params, profile.uipsf or {})
    return params


def _merge(target: dict, extra: dict) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


def learn(images, *, data: str, model: str, dual: bool, params: dict,
          workdir=None, python: str = "",
          progress: Optional[Callable[[str], None]] = None) -> Path:
    """Run uiPSF on ``images`` (`prepare`'s); the result file it wrote."""
    from . import runner
    psftype, override = PSFTYPES[(data, model)]
    if override:
        params = dict(params, PSFtype=override)
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="smappy-uipsf-"))
    runner.write_job(workdir, images, psftype, "2ch" if dual else "1ch", params)
    return runner.run(workdir, python=python, progress=progress)
