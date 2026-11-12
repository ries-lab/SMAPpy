---
version: "3"
covers: [smappy.calibrate.pupil.run_calibration, smappy.calibrate.insitu_run.run_insitu,
         smappy.calibrate.psf_run.inputs, smappy.calibrate.psf_run.calibrate]
---

## What it does

A 3D fit needs to know what one molecule looks like at every height: the
**point spread function** (PSF).  The PSF calibration makes it from bead
z-stacks by the **bead average** of the
[bead calibration](plugin:Localize/Bead calibration), or a **model of the
microscope's optics** learnt from the same stacks: the objective's pupil, with its
aberrations written as Zernike modes, from which the PSF at any height follows.
This is uiPSF (Liu et al. 2024, [doi:10.1038/s41592-024-02282-x](https://doi.org/10.1038/s41592-024-02282-x)),
re-implemented in PyTorch as uiPSF v2 and run inside SMAPpy.

The same model can also be learnt **in situ**, from the blinking molecules of
a measurement itself, with no beads: it then includes what the sample adds,
the spherical aberration that grows with depth above all.

A model is smooth without smoothing, says which aberrations the microscope
has and how sure it is of each, and shows where it does not fit the data.
Saved, it is turned into the same kind of
[spline calibration](plugin:Panels/Glossary) the bead calibration writes, so
[Spline 3D](plugin:Localize/Spline 3D) and
[Spline 3D 2C](plugin:Localize/Spline 3D 2C) read either alike, and the
model is kept inside the same file.

By hand, this section only opens the [PSF calibration window](plugin:Panels/PSF calibration),
with the *Open* button in its title bar; the window's page explains the
methods, their settings and what they show.  It replaces the *Bead calibration* section in the shipped GUI; *Bead calibration*
is still in the plugin tree (Plugins > Localize) for now.
The pupil model, from beads or in situ, needs PyTorch, an optional part of
SMAPpy; without it the window says what to type to install it.

## Without the window

The settings below are for a calibration made without the window: by the
AI assistant (`run_plugin`), in a [chain](plugin:Panels/Batch window), or from a
script.  The section in the tab never shows them: the window has its own.
They are the window's, by the same names.  Each part is one of the window's
sections, and it is read only where the chosen *method* reads it, as the
window shows only those sections.

Such a run does what the window's *Detect beads* and *Calibrate* do with no
bead marked by hand.  The beads the detection marks as doubles are left out.
The method's finish follows: for a pupil model, the fit quality and the
model against the data.  With *save model* on, the calibration is saved
beside the stacks, as the window saves it, unless *save as* names another
file.  Either way it is listed for the fitters' *auto*, as the window lists
it.  The figures are the window's pages, with no bead selected, and a page
that is a table (the Zernikes) goes into the result's data.

The *method* is named by its key, because two methods share a label.
*pupil*, *average* and *elliptical* read bead z-stacks.  *molecules* and
*molecules elliptical* read molecules on the coverslip in z sweeps.
*insitu* reads a blinking measurement.  A pupil model needs the
microscope's NA and emission wavelength.  If the database does not give
them for the stacks, type them under *microscope*; the run refuses before
any work and says which are missing.

A run without the window does not put the calibration into the fitter.  Set
the fitter's *calibration* to the file it reports, or leave it on *auto*.

## Parameters

### stacks

A file is one acquisition.  A folder is searched as the window's *Add...*
searches it, so a folder of bead stacks calibrates from all of them.

### mode

*Dual colour* and *4Pi* are offered only by the methods that have them, as
in the window.  Another mode is refused, with the modes the method has.

## References

- Liu S, Chen J, Hellgoth J, *et al.* (2024) Universal inverse modeling of point
  spread functions for SMLM localization and microscope characterization.
  *Nat Methods* 21:1082–1093.
  [doi:10.1038/s41592-024-02282-x](https://doi.org/10.1038/s41592-024-02282-x)
