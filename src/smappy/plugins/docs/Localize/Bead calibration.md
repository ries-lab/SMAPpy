---
version: "5"
covers: [smappy.calibrate.core.calibrate, smappy.calibrate.dual.calibrate_dual,
         smappy.plugins.bead_calibration.calibrate_beads, smappy.plugins.bead_calibration.save_path]
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

By hand, this section only opens the [bead calibration window](plugin:Panels/Bead calibration),
with the *Open* button in its title bar;
everything else -- single channel or dual colour, the bead stacks, the
settings, excluding beads and checking the result -- happens in that window,
and its page explains how the calibration is made.  The window opens as it
was last left.  The section is here so that the calibration sits beside the
fitters that use it; like any other section it can be moved or taken out of
the tab (right-click its title).

Choose *Dual colour* in the window for a camera whose chip is split into two
halves, each seeing one dye (a [splitter](plugin:Panels/Glossary), also
called a dual-view): the calibration then holds a PSF for each half and the
map between them (the [registration](plugin:Panels/Glossary), also called
channel alignment), for Spline 3D 2C.

## Without the window

The settings below are for a calibration made without the window: by the
AI assistant (`run_plugin`), in a [chain](plugin:Panels/Batch window), or from a
script.  The section in the tab never shows them: the window has its own.
Such a run does what the window's *Detect + calibrate* does with nothing
changed by hand.  In single channel, the beads marked doubles are left out,
as the window starts them.  With *fit quality* ticked, it also does
*Calculate fit quality*.  The calibration is then saved where the window's
*Save calibration...* would offer: beside the stacks, named after them and
the date.  *save as* puts it somewhere else.  The figures are the window's
pages, with no bead selected.

A run without the window does not put the calibration into the fitter.  Set
the fitter's *calibration* to the file it reports.

## Parameters

### stacks

A file is one stack.  A folder is searched as the window's *Add...* searches
it, so a folder holding several stacks calibrates from all of them.

### beads

The window's settings, by the same names.  *channels* and *registration*
are read only in dual colour.
