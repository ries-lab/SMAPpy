---
title: PSF calibration
summary: Makes the PSF calibration the Spline 3D fitters read from bead z-stacks, by one of two methods -- the bead average, or a physical model of the microscope's pupil and aberrations learnt with uiPSF v2 -- and shows what each found and how well it fits.
widget: smappy.calibrate.psf_window.PSFCalibrationWindow
covers: [smappy.calibrate.psf_methods.BeadAverage, smappy.calibrate.psf_methods.PupilModel, smappy.calibrate.pupil.run_calibration, smappy.calibrate.pupil.bead_rois, smappy.calibrate.pupil.single_rois, smappy.calibrate.pupil.dual_rois, smappy.calibrate.pupil.spline_calibration, smappy.calibrate.pupil.dual_calibration, smappy.calibrate.pupil.save, smappy.calibrate.pupil.OpticsSettings, smappy.calibrate.pupil.BeadSettings, smappy.calibrate.core.find_stack_beads, smappy.calibrate.psf_window.PSFCalibrationWindow]
---

## What it does

A 3D fit needs to know what one molecule looks like at every height, the
*point spread function* (PSF).  This window makes that calibration from
z-stacks of beads, by one of two methods (*method*):

* **bead average**: the beads are aligned onto one another, averaged and
  smoothed -- the [bead calibration](plugin:Panels/Bead calibration)'s
  method, with its settings, pages and fit quality, explained on its page;
* **pupil model (uiPSF)**: the PSF is learnt as a **model of the optics**,
  explained here.

Both write the same kind of file and hand it to the same fitter, so they can
be compared on the same beads.  A later method will learn the model from the
blinking molecules of a measurement itself, with no beads.

The pupil model: the light from a point is collected by the objective, and
everything the microscope does to it -- the astigmatism of a cylindrical lens,
the objective's own aberrations, the refractive indices it passes through --
is described by the objective's **pupil**, a disc whose phase is written as a
sum of Zernike modes (standard shapes such as astigmatism, coma or spherical
aberration).  From the pupil the PSF at any height follows by the physics of
light, so the model is smooth without smoothing and also holds where the beads
were noisy.  The method is uiPSF (Liu et al. 2024,
[doi:10.1038/s41592-024-02282-x](https://doi.org/10.1038/s41592-024-02282-x));
this is its re-implementation in PyTorch (uiPSF v2), run inside SMAPpy.

Its input is the bead calibration's: z-stacks of sub-diffraction fluorescent
beads, imaged while the objective steps through focus.  Its output is the same
kind of [spline calibration](plugin:Panels/Glossary) file, so
[Spline 3D](plugin:Localize/Spline 3D) and
[Spline 3D 2C](plugin:Localize/Spline 3D 2C) read it as they read a bead
average, and beside it the model itself, which can start the next calibration
of the same microscope.

Use it to calibrate for fitting, and to look at the microscope: every run
shows the aberrations with their uncertainties and where the model and the
data disagree.  Untick *save model* to only look.

The pupil model needs PyTorch, an optional part of SMAPpy
(`pip install "smappy-smlm[uipsf]"`); without it the window says what to
type, and the bead average works as ever.  It uses the
GPU where PyTorch finds one (NVIDIA, or Apple's).

## How it works

This section and the next describe the pupil model; the bead average's are on
the [bead calibration's page](plugin:Panels/Bead calibration).

**Finding the beads** is the bead calibration's own: the same detection, the
same settings under *bead detection*, the same verdict on each bead's
brightness.  The verdict is only shown here; the model judges each bead by how
well it fits.  The camera's numbers come from the camera database, as a fit of
the same files would take them.

**Cutting** one box of raw pixels around each bead, converted to photons, with
saturated pixels left out of the fit rather than the bead dropped.  The first
slice of every stack is left out: on real data it is often brighter than the
rest.

**Learning** fits one model to all beads at once: the pupil's Zernike modes
(shared), an extra blur of the image (shared), and each bead's position,
brightness and background.  It runs in two stages, the low orders first, then
all of them.  With *start from*, a previous model of the microscope replaces
the first stage: every bead is placed on that model's pupil, and learning
continues from there.

**Rejecting beads** by their fit: a bead whose residuals, brightness or
position stand out from the others by more than *rejection (k)* robust
standard deviations is left out, and the model is learnt again without it.
The table says which beads were used and why the others were not.

**Saving.**  With *save model*, the model is turned into a spline: the PSF of a
point on the central pixel is computed on a grid through the stacks' z range,
each pixel integrated over its area, and normalised as the bead average is.
The calibration and the model file go beside the stacks, named as the bead
calibration names them, and the calibration goes to the fitter.

## In detail

The model is vectorial and high-NA: the pupil field carries the dipole
emission's polarisation through the immersion, coverslip and sample media
given under *microscope*, and the image is integrated over each camera pixel.
The likelihood is Poisson with the camera's read noise added as a shifted
Poisson, SMAPpy's own fitters' likelihood; with EM gain, the excess noise
factor 2.  Slices closer than 50 nm are thinned for learning.  The first
stage frees the modes up to radial order 4 and stops early; the second frees
all of them up to *Zernike order* (auto: 6).

The pupil magnitude is 1 plus Zernike modes: all of them up to the order
(*free*), only the rotationally symmetric ones (*symmetric*: Noll 4, 11, 22,
...), or none, so that only the apodisation of the objective remains
(*fixed*).  The blur is a Gaussian, σ for x and y, learnt (*free*) or held at
*blur σ* (*fixed*).  Which of these choices are right for which data is under
investigation; they are offered so that their effect can be seen.

Bead rejection scores each bead from the learning fit itself: its mean
deviance per pixel (about 1 where the model is right), |log| of its photons
against the median, its distance from its box's centre and the fraction of
its pixels that were saturated.  A score more than *rejection (k)* robust
standard deviations (1.4826 × the median absolute deviation) above the median
rejects the bead, with floors so that near-identical beads are not rejected
for tiny differences; *include* and *exclude* in the table always win.

The spline is sampled on the knots of SMAPpy's spline grid, a pixel apart,
each pixel integrated over 4 × 4 sub-samples, with z steps of *spline z step*
(auto: the stacks' step) through ± *spline z range* (auto: the stacks'
range).  One channel: clipped at zero and scaled so that the brightest plane
sums to 1.  Two channels: both sampled on one z grid and scaled together, so
that each channel's share of the light is the learnt one.

## Controls

### bead stacks
The files or folders of bead z-stacks.  All are learnt from together.

### Add files...
Adds stacks one by one.

### Add folder...
Adds a folder; with *search subdirectories* ticked, every acquisition inside it.

### Remove
Removes the selected entries.

### search subdirectories
Whether *Add folder...* looks inside the folder's subfolders.

### method
*bead average* or *pupil model (uiPSF)*.  The window shows only what the
method reads.  Switching keeps the files and settings and clears the result.

### mode
*Single channel*, or *Dual colour* for a camera split into two halves: a PSF
per half and the transformation between them.  The pupil model learns a pupil
per channel, with the photon share and the focus offset.

### start from
Pupil model only.  A previous model of this microscope: its `.uipsf.h5` file, or the calibration
saved beside it.  It starts the learning and is drawn beside the new
Zernikes, so a change in the microscope shows.  Empty, the learning starts
from the beads alone.

### save model
On: the calibration (and for the pupil model, the model file) is saved beside
the stacks, named as the bead calibration names it and never over another
file, and handed to the fitter.  Off: the figures only; *Save as...* still
saves.

### microscope
The optics, typed in until SMAPpy's microscope database supplies them.  There
is no default for the NA and the wavelength: a wrong value cannot be noticed
in the result, only in a model that does not fit.

### NA
The objective's numerical aperture, as engraved on it.

### emission
The beads' emission wavelength, the centre of the filter's band.

### emission, 2nd
For dual colour, the secondary channel's; auto: the same as *emission*.

### immersion index
The immersion oil's refractive index (1.518 for standard oil).

### model
The choices of the model, explained under *In detail*.

### Zernike order
The highest radial order of the pupil's modes.  Higher orders describe finer
aberrations and cost more time and more beads.

### pupil magnitude
*free*, *symmetric* or *fixed* (the apodisation only).

### blur
Whether the extra blur is learnt or held at *blur σ*.

### blur σ
The blur's start, or its value when held.

### bead diameter
The beads' size, so that it is taken out of the model: a 100 nm bead is not a
point.  0 for beads much smaller than the resolution.

### beads
How beads are found, and for the bead average how they are aligned and
averaged: the bead calibration's settings, which its
[page](plugin:Panels/Bead calibration) explains.  The pupil model reads only
the finding; *bead box* is also the size of the boxes it learns from and of
its spline.

### spline
How the model is sampled into the spline the fitter reads.

### spline z step
The spline's z spacing; auto, the stacks' own.

### spline z range
How far above and below focus the spline reaches; auto, the stacks' range.

### Detect + calibrate
Finds the beads and makes the calibration; with *save model*, saves it.  For
the pupil model the status line counts the iterations.

### Stop
Pupil model: stops the learning after the current iteration.  Nothing is
saved.

### Calculate fit quality
Bead average: fits the beads back with the fitter the data will meet, on the
*Fit quality* page.

### Save as...
Saves the calibration under another name, or one made with *save model* off.

### Browse average stack...
Bead average: the averaged bead stack, slice by slice, in its own window.

### Use in the Spline 3D fitter
Puts the saved calibration into the fitter again (it is put there when it is
saved).  For dual colour, the Spline 3D 2C fitter.

### Exclude / include
Marks the selected beads for the next *Recalculate*: a used bead to be
excluded, and for the pupil model a rejected one to be included; pressed
again, the mark is withdrawn.  Space does the same.

### Recalculate
Calibrates again from the same beads with the marks applied (the pupil model
from its current model) and saves over this run's files.

## Output

The pages appear after a run.  The bead average's are the bead calibration's
(*Overview*, *Transformation* for dual colour, *Fit quality*, *Field
diagnostics*, *Bead diagnostics*), explained on its page.  The pupil model's:

- **Pupil**: the learnt phase (radians) and magnitude over the pupil.  The
  phase of an astigmatic PSF is a saddle; the magnitude should be close to 1.
- **Zernikes**: every mode's coefficient with its uncertainty (CRLB) as an
  error bar, beside the *start from* model's when there is one.  A
  difference far beyond the error bars is a change in the microscope.
- **Residual atlas**: where the model and the data disagree, grouped by
  slice, defocus, distance from the spot's centre and position on the field.
  Each group's mean deviance is about 1 where the model is right; a group
  marked in red differs by more than noise, and the title says what that
  usually points to (a stage scale, the blur, field-dependent aberrations).
- **Beads**: where the beads were on the chip, coloured by whether they were
  used, rejected or excluded.

For dual colour, *Pupil* and *Zernikes* come once per channel.

The bead average's table is the bead calibration's.  The pupil model's lists each bead with its brightness verdict (as the bead
calibration would give it), its state, the reason for a rejection, and its
deviance, photon score and offset from the box's centre.

## References

- Liu S, Chen J, Hellgoth J, *et al.* (2024) Universal inverse modeling of point
  spread functions for SMLM localization and microscope characterization.
  *Nat Methods* 21:1082–1093.
  [doi:10.1038/s41592-024-02282-x](https://doi.org/10.1038/s41592-024-02282-x)
