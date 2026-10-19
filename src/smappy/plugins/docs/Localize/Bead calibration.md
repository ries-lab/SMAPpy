---
version: "3"
covers: [smappy.calibrate.core.calibrate, smappy.calibrate.dual.calibrate_dual]
---

## What it does

A 3D fit needs to know what a molecule looks like at every height: the
**point spread function** (PSF).  The bead calibration measures it from
fluorescent beads on a coverslip, imaged while the objective steps through
focus, and saves it as the calibration file -- a
[spline calibration](plugin:Panels/Glossary), also called a cspline or
experimental PSF model -- that
[Spline 3D](plugin:Localize/Spline 3D) and
[Spline 3D 2C](plugin:Localize/Spline 3D 2C) read.

This section only opens the [bead calibration window](plugin:Panels/Bead calibration);
everything else -- single channel or dual colour, the bead stacks, the
settings, excluding beads and checking the result -- happens in that window,
and its page explains how the calibration is made.  The window opens as it
was last left.  The section is here so that the calibration sits beside the
fitters that use it; like any other section it can be taken out of the tab.

Choose *Dual colour* in the window for a camera whose chip is split into two
halves, each seeing one dye (a [splitter](plugin:Panels/Glossary), also
called a dual-view): the calibration then holds a PSF for each half and the
map between them (the [registration](plugin:Panels/Glossary), also called
channel alignment), for Spline 3D 2C.
