---
version: "1"
covers: [smappy.calibrate.core.calibrate, smappy.calibrate.dual.calibrate_dual]
---

## What it does

A 3D fit needs to know what a molecule looks like at every height: the
**point spread function** (PSF).  The bead calibration measures it from
fluorescent beads on a coverslip, imaged while the objective steps through
focus, and saves it as the calibration file that
[Spline 3D](plugin:Localize/Spline 3D) and
[Spline 3D 2C](plugin:Localize/Spline 3D 2C) read.

This section only opens the [bead calibration window](panel:Bead calibration),
in the *mode* chosen here; everything else -- the bead stacks, the settings,
excluding beads and checking the result -- happens in that window, and its
page explains how the calibration is made.  The same window opens from
**Tools > Bead calibration...**, as it was last left.  The section is here so
that the calibration sits beside the fitters that use it; like any other
section it can be taken out of the tab.

For a PSF learnt as a model of the optics, or from the blinking molecules of
the sample itself, see [uiPSF calibration](plugin:Localize/uiPSF calibration).

## Parameters

### mode
Choose *dual colour* for a camera whose chip is split into two halves, each
seeing one dye: the calibration then holds a PSF for each half and the map
between them, for Spline 3D 2C.  The mode can still be switched in the
window, which keeps the files and the settings both modes share.
