---
version: "1"
covers: [smappy.uipsf.prepare.Split, smappy.uipsf.prepare.channel_shift, smappy.uipsf.prepare.subtract_background, smappy.uipsf.prepare.read_beads, smappy.uipsf.prepare.read_movie, smappy.uipsf.uipsf_parameters, smappy.uipsf.runner.run, smappy.uipsf.convert.single_calibration, smappy.uipsf.convert.dual_calibration, smappy.uipsf.convert.channel_transformation, smappy.uipsf.microscopes.Microscope, smappy.uipsf.plots.figures, smappy.uipsf.plots.data_and_model, smappy.uipsf.plots.localization_bias, smappy.uipsf.plots.pupil_images, smappy.uipsf.plots.zernike_phases, smappy.uipsf.plots.transformation_residuals]
---

## What it does

A 3D fit needs to know what a molecule looks like at every height: the
**point spread function** (PSF).  The [bead calibration](panel:Bead calibration)
measures it by averaging beads imaged at a series of focus positions.  This
plugin hands the same job to **uiPSF**
([Liu et al. 2024](https://doi.org/10.1038/s41592-024-02282-x)), which instead
fits a *model of the microscope* to the images.  The model is the light's
wavefront in the objective's pupil, written as a sum of Zernike polynomials
-- defocus, astigmatism, coma, spherical aberration and so on -- and
propagated to the camera with the physics of a high-NA objective.

That gives two things the bead average cannot:

* **A model from beads that is smooth without smoothing**, defined beyond the
  beads' noise, and that names the microscope's aberrations in nanometres of
  wavefront.
* **A model from the blinking molecules themselves** (*in situ*).  A bead on
  the coverslip does not see what a molecule a micrometre deep in a cell sees:
  the refractive-index mismatch between oil and water adds spherical
  aberration that grows with depth, and the sample adds its own.  uiPSF can
  learn the PSF from the raw frames of the acquisition you want to fit, so the
  model is the one the data were taken with.

The result is saved as an ordinary smappy calibration file, which
[Spline 3D](plugin:Localize/Spline 3D) and
[Spline 3D 2C](plugin:Localize/Spline 3D 2C) read exactly as they read a bead
calibration.  For a split camera it holds both channels' models and the
transformation between the halves, which uiPSF learns together with the PSF.

**What it needs.**  uiPSF itself, installed in a Python environment of its own
(see *Installing uiPSF* below), and a **microscope
profile**: the objective's numerical aperture, the refractive indices, the
emission wavelength and, for a split camera, how the halves are arranged.
From beads: one or more z-stacks, as for the bead calibration.  From blinking
molecules: a few thousand frames with molecules spread over the depth you want
to fit, and the depth the focus was at.

**Learning from blinking molecules needs checking.**  Unlike beads, the
molecules' heights are unknown, so uiPSF has to find them and the PSF
together.  Where that ends up depends on where it starts: the starting pupil
from the profile (`insitu:` in the profile) and the *depth*.  In tests on
simulated data the z scale of the learnt model came out up to a quarter
short, and a poor starting pupil gave a model whose z ran backwards.  Before
relying on an in situ calibration, compare it with a bead calibration of
the same microscope: fit both to the same data near the coverslip, where
they should agree.  Starting from that bead model (*start from*) helps.

**When not to use it.**  For a PSF no pupil describes -- a strongly
non-uniform illumination of the pupil, a damaged optic -- the voxel model or
the bead calibration is the safer choice.  Field-dependent and 4Pi models,
which uiPSF also learns, are not offered: the fitters here use one PSF for the
whole field.

## Installing uiPSF

smappy does **not** install uiPSF, and installing smappy does not bring it.
uiPSF needs TensorFlow, which is large and wants its own versions of numpy
and Python, so it lives in a Python environment of its own.  smappy starts
uiPSF in that environment when the plugin runs, and needs to be told once
where it is.  Everything below is done once per computer.

**1. An environment with uiPSF.**  With conda (Miniconda or Anaconda), in a
terminal (on Windows, the Anaconda Prompt):

```
conda create -n uipsf python=3.12
conda activate uipsf
git clone --branch claude/friendly-cerf-qt2sfj https://github.com/ries-lab/uiPSF
pip install -e uiPSF
```

The last line installs uiPSF with TensorFlow and everything else it needs.
It has to be this branch of uiPSF: its main branch needs Python 3.7 and
TensorFlow 2.9, which do not run on current Macs or with current numpy.
Without conda, `python3.12 -m venv uipsf-env` and that environment's `pip` do
the same.

**2. Check it.**  Still in that environment:

```
python -c "import psflearning.psflearninglib; print('uiPSF works')"
python -c "import sys; print(sys.executable)"
```

The second line prints the environment's Python, for example
`/Users/me/miniconda3/envs/uipsf/bin/python` on a Mac,
`/home/me/miniconda3/envs/uipsf/bin/python` on Linux, or
`C:\Users\me\miniconda3\envs\uipsf\python.exe` on Windows.

**3. Tell smappy.**  Once for all plugins and sessions, add a line with that
path to smappy's configuration file, `config.yaml` in
`~/Library/Application Support/smappy/` (Mac), `~/.config/smappy/` (Linux) or
`%APPDATA%\smappy\` (Windows), creating the file if it is not there:

```
uipsf_python: /Users/me/miniconda3/envs/uipsf/bin/python
```

Or give the path in the plugin itself, as *uiPSF python* (under *learning*,
"more"), or in the environment variable `SMAPPY_UIPSF_PYTHON`.  The plugin's
setting wins over the file, and the file over the variable.

If uiPSF can be imported in the Python smappy itself runs in, nothing needs to
be set, but a separate environment is what keeps TensorFlow from fighting
smappy's own packages.

**A graphics card.**  uiPSF runs on the CPU by default, which works
everywhere and takes minutes for beads (see *Running time* below).  On Linux
with an NVIDIA card, `pip install "tensorflow[and-cuda]"` in the uiPSF
environment lets it use the card.  On Windows TensorFlow uses no graphics card
after version 2.10, except under WSL2.  On a Mac uiPSF runs on the CPU, and
Apple-silicon Macs run it natively.

## How it works

**1. Reading the data.**  smappy reads the frames, not uiPSF: the bead
stacks' z positions come from the microscope's metadata as for the bead
calibration, and every pixel is converted to photons with the camera's gain
and offset, $N = (c - c_0)\,g$, as the fitter converts them.  Bead stacks of
different lengths are cut to the shortest about their centres.  Each stack's
median is subtracted.  uiPSF fits a background per bead anyway, but it finds
beads above a fraction (*threshold*) of the brightest pixel, and with the
background left in that fraction can fall into the noise.

```figure-setup
from smappy.simulate import dual_bead_stacks
from smappy.uipsf.prepare import make_split
stacks, z = dual_bead_stacks(1, seed=0, size_px=60)
frame = stacks[0].max(axis=0).astype(float)
```

**2. Splitting a two-channel camera.**  The frame is cut at the split into
two halves of equal size, the main channel first, and a mirrored half is
flipped back, so that uiPSF sees two images of the same molecules.  Exactly
how they were cut is recorded, because the transformation uiPSF learns
between the halves has to be put back into camera pixels afterwards.  For beads,
the shift between the halves is measured first, from the cross-correlation
of their projections, and handed to uiPSF as its starting point for pairing
the beads.

```figure Each bead appears in both halves of the split camera; cut and stacked, the two channels show the same beads, a little shifted and turned -- the transformation uiPSF learns along with the PSF.
fig.set_size_inches(7.5, 3.2)
split = make_split(frame.shape, {"layout": "up-down", "main_channel": "upper"})
channels = split.cut(frame[None])[:, 0]
axes = fig.subplots(1, 3)
axes[0].imshow(frame, cmap="magma")
axes[0].axhline(split.split_position - 0.5, color="w", lw=1)
axes[0].set_title("camera frame")
for ax, image, title in zip(axes[1:], channels, ("main", "secondary")):
    ax.imshow(image, cmap="magma")
    ax.set_title(title)
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
```

**3. Learning the PSF.**  uiPSF finds the beads or molecules, cuts a small
image around each, and fits all of them at once with one PSF model: every
emitter gets its own position, brightness and background, and all share the
pupil.  The pupil's Zernike coefficients are adjusted until the modelled
images match the measured ones.  From blinking molecules the heights are not
known beforehand.  They are estimated with a starting pupil (astigmatism
from the profile), then refined together with the pupil, over a few
*rounds*.  This is the slow step: minutes for beads, longer for blinking
molecules (see *Running time*).

**4. Making the calibration.**  The learnt PSF is taken as voxels, a stack of
images at the model's z planes, and turned into the cubic spline the fitters
use.  This is the same step as the bead calibration's: negative values set to
zero, the brightest plane normalised to one photon, and the spline computed
through every voxel.  The fitter cannot tell which way a calibration was made.
The Zernike coefficients, the refined depth and uiPSF's parameters are stored
with it, and uiPSF's own result file is saved beside it.

## In detail

**Where z = 0 is.**  For beads, uiPSF's model is centred on the pupil's
focus, and that plane is z = 0 -- the same convention as the bead calibration,
with the fitter's $z = -(i - z_0)\,\Delta z$ for plane $i$.  For blinking
molecules uiPSF's planes are heights above the coverslip, from the bottom up.
They are reversed to run as a bead stack does.  z = 0 is put at the nominal
focal plane: the focus moved $d$ into the sample sits at height
$d\, n_{\mathrm{med}} / n_{\mathrm{imm}}$, and $d$ is the *depth* as uiPSF
refined it.  A fitted z is then the molecule's height relative to that plane,
and its height above the coverslip is $z + d\, n_{\mathrm{med}} / n_{\mathrm{imm}}$.

**The scale of z from blinking molecules.**  From beads the z steps are
known, so the model's z scale is the stage's.  From blinking molecules it is
not: it comes only from the physics of the pupil model, from how defocus
changes the spot for this NA, wavelength and refractive index.  The profile's
numbers therefore set the z scale of an in situ calibration, and a wrong
refractive index scales every z.

**Two channels.**  uiPSF stores, for each pair of channels, an affine map
$T$ that takes a main-channel position $(y, x, 1) - c$ to the other
channel's, both in the arrays it was given and $c$ the image centre.  The
plugin pushes a grid of points through it, takes both sets back to camera
pixels through the recorded split, and fits the projective map, secondary to
main, that the two-colour fitter reads.  The two models are normalised
together, as the dual-colour bead calibration does: each channel's light
around focus is its model's signal times the photons uiPSF fitted to that
channel, and together they sum to one, so the photon split between the
channels is kept.  The secondary model is flipped back to the camera's
orientation when the splitter mirrors it.

**Running uiPSF.**  uiPSF is not imported by smappy.  It runs as a separate
program in its own Python, so that smappy needs neither TensorFlow nor its
version.  The plugin writes the photons and the configuration into a job
folder, starts uiPSF there, passes its progress on and reads back the file
uiPSF writes.  If it fails, the message names the log with everything uiPSF
printed (*keep job folder* keeps the folder).

**What uiPSF is told.**  The profile's NA, refractive indices and wavelength;
the pixel size, from the camera's metadata or else from the profile; the z
step; and the settings below.  Anything else uiPSF can be told goes into the
profile's `uipsf:` section, nested as in uiPSF's own YAML configuration, and
has the last word.

**Running time.**  On four CPU cores, 18 simulated beads in 41 planes took
three minutes.  uiPSF's README reports, on a graphics card, half a minute for
beads in one channel, five for two channels, and five to fifteen minutes in
situ.  On a CPU expect several times that.  On a Mac uiPSF runs on the CPU;
TensorFlow's Apple-GPU plugin has not been tried with it, and the pupil model
is built on complex numbers and Fourier transforms, which that plugin has
supported only in part.

**Microscope profiles.**  A profile is a YAML file, `example.yaml` ships as
a template:

```
name: M2, astigmatic, two colour
NA: 1.43
refractive_index: {immersion: 1.516, medium: 1.335, coverslip: 1.516}
emission_wavelength_nm: 680
roi_size_px: 25
dual: {layout: up-down mirrored, main_channel: upper}
insitu: {zernike_index: [5], zernike_coeff: [0.5]}
uipsf: {}
```

Profiles are found in the folders named by `SMAPPY_MICROSCOPES` (a group's
shared folder, say), then in `microscopes/` in smappy's configuration folder,
then among those that ship.  The first file of a name wins.

## Parameters

### data
Beads give the PSF on the coverslip, with the z scale of the stage.  Blinking
molecules give the PSF where the molecules are, aberrations of the sample
included; prefer it for samples deeper than about a micrometre in water.

### microscope
The profile sets the z scale of an in situ model (see *In detail*), so a
wrong NA or refractive index matters there more than for beads.

### model
A voxel model has a free value for every voxel, so it needs many bright beads
and has no aberrations to report.

### beads.z_step_nm
Only needed when the stack's metadata has no z positions, or has them wrong.

### beads.bead_diameter_nm
Beads of 100 nm or more broaden the measured PSF.  Given their size, uiPSF
models the bead and the PSF is learnt without it.

### blinking.depth_um
A starting value: uiPSF refines it.  A guess within a few hundred nanometres
is enough.

### blinking.z_range_um
Cover the depth the molecules are spread over, with a few hundred nanometres
to spare.  Wider costs time and leaves thinly sampled planes at the ends.

### blinking.last_frame
Three thousand frames with ten or more molecules each give enough molecules
per z slice; sparser data need more frames.

### blinking.start_from
A bead model of the same microscope is a good start, and the learning
converges in fewer rounds.  Either uiPSF's own result file or the calibration
this plugin saved, whose uiPSF file lies beside it, will do.

### blinking.min_photon
Raise it for data with many dim molecules, which carry little information
about the PSF.

### learning.roi_size_px
The fitters cut their ROI from inside the model, so it must be at least three
pixels wider than the fit's ROI.  Larger also captures the wings of a
defocused spot.

### learning.iterations
uiPSF's own default is 200.  100 is usually converged for a Zernike model,
and the output shows the loss falling.

### learning.zernike_order
Order 8 is 45 terms.  Order 6 (28 terms) is enough for most systems, and
lower orders learn faster.

## Output

**The calibration** (*save as*, by default `<acquisition>_uipsf_3dcal.h5`
beside the data) is what Spline 3D or Spline 3D 2C take as their
*calibration*.  Beside it is `<name>_uipsf.h5`, uiPSF's own result: the
pupil, the per-emitter fits and the model, which SMAP and uiPSF's own
notebooks read.

**The text** names both files, the model's z range, its largest
aberrations in nanometres of wavefront, and two numbers from uiPSF's own
check of the result.  The first is the median bias of the beads, refitted
with the model.  The second, for two channels, is how far the
transformation leaves each bead pair apart.  The figures are uiPSF's own
diagnostics, the ones its notebooks show, one tab each, and a bar chart of
the aberrations:

* **data vs model**: the measured spot above and uiPSF's model of it below,
  at planes through the model and in an x–z section, per channel.  For beads
  it is one bead, the one the model fits as well as it fits most, and the
  planes are labelled by stage position.  From blinking molecules the
  molecules are averaged per plane of the model by their fitted height, and
  the planes are labelled by height above the coverslip.  Planes no molecule
  fell into stay black.  Data and model should look alike in every plane; a
  difference that grows away from focus means the model misses an
  aberration.
* **localization bias** (beads): uiPSF refits every bead in every plane with
  the learnt model.  This is x, y and z of each fit minus where the bead
  was, against the stage position, one grey line per bead and their median
  in red.  A few nanometres in x and y, and a z bias of a few tens of
  nanometres at most over the range you will fit, are good.  A slope in z means the model's z scale
  is off.  A bias that grows towards the ends of the stack marks where the
  model stops being trustworthy.
* **pupil**: per channel, the pupil's magnitude, the aberration it adds to
  the wavefront (from Zernike term 5 up, in nm), and every Zernike term:
  phase as bars in nm rms, magnitude as dots.  The box names the terms
  uiPSF names.  Astigmatism (5, 6) is expected with an astigmatic lens, and
  spherical (11) grows with depth in water.  Large coma (7, 8) points at a
  tilted or misaligned optic.
* **Zernike**: the same phases as bars, from term 5 up, the channels side by
  side and the named terms labelled under the axis.  Two channels of one
  microscope should share most of their aberrations.  A term that differs
  between them belongs to the optics after the splitter.
* **beads** or **emitters**: where uiPSF found emitters (grey) and which it
  used for the model (red circles), in the channel as uiPSF saw it.  Beads
  too close to another or to the edge are dropped before they count as
  found.  Of those found, uiPSF leaves out the ones beyond *max beads* and
  those it rejects as outliers after a first round of learning.  From
  blinking molecules it also leaves out the dim and the poorly fitted, and
  more than each z slice takes (*per bin*).
* **channel transformation** (two channels): for every bead pair, where the
  secondary channel's bead lies minus where the transformation puts it, in
  nanometres.  On the left the residuals are drawn as arrows over the field,
  with the longest given in the title; on the right they are scattered.
  Residuals of a few nanometres are good.  Arrows that all point one way in
  part of the field mean the transformation cannot follow the splitter there.

**A bad result** shows as a model with structure that changes erratically
from plane to plane, or as fitted z that does not follow the stage when the
bead stack itself is fitted.  The usual causes are too few emitters, a wrong
pixel size or wavelength, and, from blinking molecules, a depth guess or z
range far from the data.

## Differences from SMAP

SMAP has no uiPSF plugin: uiPSF's result file is loaded directly into SMAP's
fitter and peak finder ("load 3D cal", "load T"), using the spline uiPSF
computes.  Here the spline is computed again from uiPSF's voxels, the way
smappy's bead calibration computes it.  uiPSF normalises its spline by the
median plane sum after subtracting the minimum, while the fitter's photon
count assumes the brightest plane sums to one.  The channel transformation is
converted to smappy's projective map in camera pixels, so a two-channel
calibration from uiPSF and one from beads are used in the same way.

## References

* Liu S, Chen J, Hellgoth J, *et al.* (2024) Universal inverse modeling of
  point spread functions for SMLM localization and microscope
  characterization. *Nat Methods* 21:1082–1093.
  [doi:10.1038/s41592-024-02282-x](https://doi.org/10.1038/s41592-024-02282-x)
  -- uiPSF.
* Li Y, Mund M, Hoess P, *et al.* (2018) Real-time 3D single-molecule
  localization using experimental point spread functions. *Nat Methods*
  15:367–369. [doi:10.1038/nmeth.4661](https://doi.org/10.1038/nmeth.4661)
  -- the cubic spline the fitters evaluate.
