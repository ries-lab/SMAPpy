"""PSF calibration: the bead average, a pupil model learnt by uipsf, or that model in situ, in its own window.

Only a launcher (`Plugin.window`), as *Bead calibration* is: the calibration is
made by looking at it -- the beads, the pupil, the Zernikes, the residuals --
and that is a window.  The work is `smappy.calibrate.pupil.run_calibration`
(beads) and `smappy.calibrate.insitu_run.run_insitu` (in situ), which a script
calls the same.  *PSF calibration* rather than a mode of the bead calibration
because in situ learning learns from the blinking molecules of a measurement,
not from beads (`docs/uipsf/plan.md` §7.5).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import Context, Plugin, Result, register


@dataclass
class PSFCalibrationSettings:
    """Nothing: everything is chosen in the window."""


@register("Localize/PSF calibration")
class PSFCalibration(Plugin):
    """A PSF calibration: the bead average, or a pupil model learnt by uipsf from beads or in situ."""
    description = ("make the PSF calibration Spline 3D and Spline 3D 2C read: from bead z-stacks, the bead "
                   "average or a physical model of the pupil and its aberrations learnt with uiPSF v2; or that "
                   "model learnt in situ from the blinking molecules of a measurement; opens its own window")
    changes_locs = False
    Settings = PSFCalibrationSettings
    version = "1"
    window = "psf calibration"

    def run(self, ctx: Context, settings: PSFCalibrationSettings) -> Result:
        raise ValueError("the PSF calibration is interactive and opens a window; in a script, use "
                         "smappy.calibrate.pupil.run_calibration, or smappy.calibrate.insitu_run.run_insitu")
