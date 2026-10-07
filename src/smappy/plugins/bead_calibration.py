"""Bead calibration: the calibration window, as a section of the Localize tab.

Only a launcher (`Plugin.window`).  The calibration is made by looking at the
beads -- a table to exclude them from, a stack to browse, pairs to pick on
the transformation plot -- and that is a window, not a form with a Run
button; the work itself is `smappy.calibrate` (`calibrate.calibrate` for a
script).  Being a plugin is what lets it be pinned to a tab, or taken out of
one, like the fitters beside it.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import Context, Plugin, Result, param, register


@dataclass
class BeadCalibrationSettings:
    mode: str = param("single", choices=(("single", "single channel"),
                                         ("dual", "dual colour")),
                      help="dual colour: a camera split into two halves; the "
                           "calibration then carries the map between them")


@register("Localize/Bead calibration")
class BeadCalibration(Plugin):
    """An experimental spline PSF from bead z-stacks, in its own window."""
    description = ("build a spline PSF calibration for Spline 3D or Spline 3D 2C "
                   "from bead z-stacks; opens its own window")
    Settings = BeadCalibrationSettings
    version = "1"
    window = "bead calibration"

    def run(self, ctx: Context, settings: BeadCalibrationSettings) -> Result:
        raise ValueError("the bead calibration is interactive and opens a "
                         "window; in a script, use smappy.calibrate.calibrate "
                         "or calibrate_dual")
