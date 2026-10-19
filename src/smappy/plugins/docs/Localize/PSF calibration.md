---
version: "1"
covers: [smappy.calibrate.pupil.run_calibration]
---

## What it does

A 3D fit needs to know what one molecule looks like at every height: the
**point spread function** (PSF).  The PSF calibration makes it from bead
z-stacks by one of two methods: the **bead average** of the
[bead calibration](plugin:Localize/Bead calibration), or a **model of the
microscope's optics** learnt from the same stacks: the objective's pupil, with its
aberrations written as Zernike modes, from which the PSF at any height follows.
This is uiPSF (Liu et al. 2024, [doi:10.1038/s41592-024-02282-x](https://doi.org/10.1038/s41592-024-02282-x)),
re-implemented in PyTorch as uiPSF v2 and run inside SMAPpy.

A model is smooth without smoothing, says which aberrations the microscope
has and how sure it is of each, and shows where it does not fit the data.
Saved, it is turned into the same kind of
[spline calibration](plugin:Panels/Glossary) the bead calibration writes, so
[Spline 3D](plugin:Localize/Spline 3D) and
[Spline 3D 2C](plugin:Localize/Spline 3D 2C) read either alike, and the
model file is kept beside it.

This section only opens the [PSF calibration window](plugin:Panels/PSF calibration),
whose page explains the methods, their settings and what they show.  It will
replace the *Bead calibration* section once it does everything that one does.
The pupil model needs PyTorch, an optional part of SMAPpy; without it the
window says what to type to install it.

## References

- Liu S, Chen J, Hellgoth J, *et al.* (2024) Universal inverse modeling of point
  spread functions for SMLM localization and microscope characterization.
  *Nat Methods* 21:1082–1093.
  [doi:10.1038/s41592-024-02282-x](https://doi.org/10.1038/s41592-024-02282-x)
