---
title: Bead calibration
summary: Turns z-stacks of fluorescent beads into the measured 3D PSF model the Spline 3D fitters use, and for a split camera into two models and the transformation between the halves.
widget: smappy.calibrate.qt_gui.CalibrationWindow
covers: [smappy.calibrate.core.detect_beads, smappy.calibrate.core.collect_beads, smappy.calibrate.core.estimate_shift, smappy.calibrate.core.subvoxel_peak, smappy.calibrate.core.robust_shape_error, smappy.calibrate.core.build_calibration, smappy.calibrate.core._single_model, smappy.calibrate.core.spline_coefficients, smappy.calibrate.core.positive_pair_models, smappy.calibrate.core.FOCAL_PLANES, smappy.calibrate.dual.collect_dual_beads, smappy.calibrate.dual.fit_dual_transform, smappy.calibrate.dual.robust_projective, smappy.calibrate.dual.refine_projective, smappy.calibrate.dual.build_dual_calibration, smappy.calibrate.validation.fit_bead_diagnostics, smappy.calibrate.validation.fit_paired_bead_diagnostics, smappy.calibrate.qt_gui.CalibrationWindow]
---

## What it does

A 3D fit needs to know what one molecule looks like at every height.  On a
real microscope that image -- the *point spread function*, PSF -- is not a
textbook Gaussian: a cylindrical lens stretches it, the objective adds its
own aberrations, and both differ from one microscope to the next.  The
reliable way to know it is to measure it.

This window measures it from **beads**: sub-diffraction fluorescent beads
stuck to a coverslip, imaged while the objective is stepped through focus.
Each bead gives a small 3D image of the PSF (a *bead stack*).  The window
finds the beads, cuts them out, aligns them onto one another, averages
them, smooths the average and turns it into a cubic-spline model -- the
experimental-PSF method of [Li et al. 2018](https://doi.org/10.1038/nmeth.4661).
The model is saved as an `.h5` file, which the
[Spline 3D](plugin:Localize/Spline 3D) fitter reads; how the fit uses it is
explained on that page.

It has two modes (*mode*):

* **Single channel**: one camera image, one PSF model.
* **Dual colour**: a camera whose chip is split into two halves, each seeing
  the sample through a different filter (two colours) or at a different focus
  (biplane).  The window then makes **two** PSF models, one per half,
  normalised together so that they keep the beads' split of the light, and
  measures the **transformation** that says where a point in one half
  appears in the other.  This is what
  [Spline 3D 2C](plugin:Localize/Spline 3D 2C) fits with; its transformation
  alone also serves [Gaussian 2D 2C](plugin:Localize/Gaussian 2D 2C).

What it needs:

* **Bead stacks** taken with the same objective, lens, filters and camera as
  the data.  Tens of beads is better than a few: one stack of a field with
  ten well-separated beads works, several stacks from different places are
  better.  Beads must be isolated (no neighbour within *min distance*) and not
  saturated.
* **A known z step.**  It is read from the acquisition's metadata; if the
  file does not say, type it into *z step*.
* **For dual colour**, broadband beads that are visible in both halves, and
  the layout of the split (*layout*, *main channel*).  With the camera ROI in
  the file's metadata the transformation is in chip coordinates; without it
  the calibration is *ROI-local* and only applies to data of exactly the same
  image size (a warning says so).

It opens from **Tools > Bead calibration...** (or *Dual-colour
calibration...*, which starts in the dual-colour mode), from the button of
the same name on the Localize tab, or on its own as `smappy-calibrate`.

## How it works

```figure-setup
import warnings
from smappy.calibrate.core import (CalibrationSettings, build_calibration, collect_beads,
                                   detect_beads, FOCAL_PLANES)
from smappy.calibrate.dual import DualColorSettings, calibrate_dual
from smappy.calibrate.input import BeadStack
from smappy.calibrate.validation import fit_bead_diagnostics
from smappy.simulate import bead_stacks, dual_bead_stacks
# one simulated field of nine astigmatic beads, 60 nm steps -> a calibration
stacks, z_objective = bead_stacks(1, seed=0, dz_nm=60.0)
settings = CalibrationSettings(roi_size=21, smooth_z_nm=60.0)
image = stacks[0].astype(np.float32)
beads = collect_beads([BeadStack(image, z_objective, source="beads")], settings)
single = build_calibration(beads)
cal = single.calibration
# a split camera, 30 % of each bead's light in the lower half, 70 nm steps
dstacks, dz_objective = dual_bead_stacks(1, seed=0, dz_nm=70.0, z_range_nm=(-700.0, 700.0),
                                         secondary_share=0.3)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")        # no camera ROI in a simulation: ROI-local
    dual = calibrate_dual([BeadStack(dstacks[0].astype(np.float32), dz_objective)],
                          DualColorSettings(layout="up-down", main_channel="upper", dz_nm=70.0,
                                            roi_size=17, smooth_z_nm=40.0))
```

**1. Find the beads.**  The stack is collapsed to its brightest value per
pixel (a maximum projection), smoothed, and every local maximum clearly above
the background noise (*threshold*, in multiples of the noise) is a bead.  A
bead too close to the image edge is dropped, and so are **both** beads of a
pair closer than *min distance*: two overlapping PSFs are not one PSF.

**2. Cut them out.**  Around each bead a box of *ROI size* pixels (plus a
margin, *padding*, for the alignment to shift into) is cut from every plane.
One background value is subtracted from the whole box -- the median of its
border pixels over all planes, so that the PSF's faint tails, which change
with z, are kept -- and the box is divided by the light in its brightest
plane, so that a bright and a dim bead count alike.  A bead whose brightness
is far from the others' (*brightness range*) or that reaches the camera's
maximum (*saturation*) is set aside: it is either two beads, a clump, or
clipped.

```figure Left: the maximum projection of a simulated bead stack, with the beads that were found (boxes, the size of the ROI).  Right: one bead's box at three heights of the objective -- the astigmatic spot is stretched one way below focus and the other way above.
fig.set_size_inches(7.5, 2.7)
grid = fig.add_gridspec(1, 4, width_ratios=[1.5, 1, 1, 1])
ax = fig.add_subplot(grid[0])
found, projection = detect_beads(image, settings)
ax.imshow(projection, cmap="gray", vmax=np.percentile(projection, 99.8))
half = settings.roi_size / 2
for y, x in found:
    ax.add_patch(__import__("matplotlib").patches.Rectangle((x - half, y - half), 2 * half, 2 * half,
                 fill=False, color="#2ca02c", lw=1))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("beads found", fontsize=9)
volume = beads.volumes[0]
for n, z_nm in enumerate((-400.0, 0.0, 400.0)):
    k = int(np.argmin(np.abs(z_objective[beads.records[0]["start_plane"]:][:len(volume)] - z_nm)))
    a = fig.add_subplot(grid[n + 1])
    a.imshow(volume[k], cmap="magma", vmin=0, vmax=volume.max())
    a.set_xticks([]); a.set_yticks([])
    a.set_title(f"objective at {z_nm:+.0f} nm", fontsize=9)
```

**3. Align them.**  The beads do not sit at the same place in their boxes,
nor at the same height on the coverslip, so each is shifted in x, y and z
until it matches a reference -- the average of the half of the beads that
look most alike.  The shift is found by 3D cross-correlation, first over the
whole stack, then refined over the planes around the centre (*alignment
range*, *iterations*), and is applied with sub-pixel, sub-plane precision.  A
bead that has to move further than *max xy shift* or *max z shift* is taken
for a misdetection and dropped.

**4. Reject the odd ones out.**  Every aligned bead is compared with the
average of **all the others** (so it cannot vouch for itself).  A bead whose
shape differs much more than the typical bead's does (*rejection (MAD)*) is
left out of the average: a bead with a neighbour, a tilted one, a speck of
dirt.  The bead table lists every bead with its shift, its similarity to the
others and why it was used or not.

**5. Average and smooth.**  The accepted beads are averaged.  The average
still carries the noise of a finite number of photons, which a spline would
faithfully reproduce as ripples, so it is smoothed along z (*smoothing z*;
laterally only if *smoothing xy* is set), clipped at zero and scaled so that
its brightest plane holds exactly one photon.  A fitted photon number is then
the photons in the molecule's in-focus image.  Only the planes that every
accepted bead covers after its z shift are kept.

```figure Left: side views (x against emitter z) of one bead as it was cut out, of the aligned average of the accepted beads, and of the smoothed model the fitter uses; the spot is narrow in x on one side of focus and wide on the other.  Right: the light in each plane of the model -- nearly constant in z, since a bead loses little light out of the box, and 1 in the brightest plane by construction.
fig.set_size_inches(7.5, 4.2)
axes = fig.subplots(1, 4)
psf = cal.psf
z_model = cal.z_index_to_nm(np.arange(psf.shape[0]))
extent_z = (z_model[-1], z_model[0])
p = settings.padding
one = beads.volumes[int(np.flatnonzero(single.accepted)[0])][:, p:-p, p:-p]
one = one[single.z_crop_start:single.z_crop_start + psf.shape[0]]
panels = ((one, "one bead"), (single.raw_psf, "average"), (psf, "model"))
for ax, (volume, title) in zip(axes[:3], panels):
    section = volume[:, volume.shape[1] // 2, :]
    ax.imshow(section, cmap="magma", aspect="auto", vmin=0,
              extent=(-(psf.shape[2] // 2) - 0.5, psf.shape[2] // 2 + 0.5, *extent_z))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (pixels)", fontsize=8)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("emitter z (nm)", fontsize=8)
for ax in axes[1:3]:
    ax.set_yticklabels([])
axes[3].plot(psf.sum(axis=(1, 2)), z_model, color="#1f77b4")
axes[3].axvline(1.0, color="0.6", lw=0.8, ls="--")
axes[3].set_ylim(*extent_z)
axes[3].set_xlim(0, 1.15)
axes[3].set_xlabel("light per plane", fontsize=8)
axes[3].set_title("normalisation", fontsize=9)
axes[3].set_yticklabels([]); axes[3].tick_params(labelsize=7)
```

**6. The spline.**  The model is interpolated with cubic polynomials between
the grid points, so that the fitter can evaluate it, and its slopes, at any
sub-pixel position and height.  This is what *Save calibration...* writes.

**7. Check it.**  *Calculate fit quality* fits the calibration's own beads,
plane by plane, with the same fitter the data will meet, and compares the
fitted z with the objective's position.  A good calibration gives a straight
line of slope 1.  The check is on the beads that made the model, so it shows
that the model is self-consistent, not how precise a fit of dim molecules
will be.

```figure The calibration's own beads refitted plane by plane (thin lines, one per bead) and the averaged stack (black): fitted z against where the objective was.  Right: the same after removing each bead's own offset -- the part that says whether the model is distorted along z.
d = fit_bead_diagnostics(single)
fig.set_size_inches(7.0, 2.8)
left, right = fig.subplots(1, 2)
for bead in np.unique(d["bead_id"]):
    use = d["bead_id"] == bead
    order = np.argsort(d["expected_z_nm"][use])
    style = dict(color="black", lw=2.2) if bead < 0 else dict(lw=0.8, alpha=0.8)
    left.plot(d["expected_z_nm"][use][order], d["fitted_z_nm"][use][order], **style)
    right.plot(d["expected_z_nm"][use][order], d["centered_error_nm"][use][order], **style)
limits = [d["expected_z_nm"].min(), d["expected_z_nm"].max()]
left.plot(limits, limits, "k--", lw=0.6)
right.axhline(0, color="0.6", lw=0.6)
left.set_xlabel("expected z (nm)", fontsize=8); left.set_ylabel("fitted z (nm)", fontsize=8)
right.set_xlabel("expected z (nm)", fontsize=8); right.set_ylabel("error (nm)", fontsize=8)
right.set_ylim(-40, 40)
for ax in (left, right):
    ax.tick_params(labelsize=7)
```

**Dual colour: two halves, one calibration.**  The frame is cut at the
*split position* into the main half (*main channel*) and the other, the
secondary.  Steps 1 and 2 run in each half on its own.  Then:

* **Pairs.**  Each bead is located precisely by a 2D Gaussian fit to its
  eleven central planes.  After undoing the declared layout (the split, and
  a mirror if the layout says *mirrored*), the most common offset between
  the beads of the two halves is found, and a bead is paired with its
  nearest neighbour in the other half when each is the other's nearest.
* **The transformation** from the secondary half to the main half is a
  projective map, a 3x3 matrix -- enough for a shift, a rotation, a scale
  and a small tilt of the image plane.  It is fitted to the paired
  positions robustly, in two rounds: pairs that do not fit (a wrong
  pairing, a bad position) are found and left out, so they cannot bend the
  map.
* **The two PSFs** are built from the pairs, with the secondary half's image
  of each bead brought onto the main half's (mirrored if need be, shifted by
  what the transformation predicts) and **one** shift per pair for both
  halves: both images are of the same bead at the same height.  The two
  averages are normalised **together**, so that the models keep the beads'
  split of the light between the halves.

```figure Dual colour.  Left: the light in each plane of the two models; the grey band is the five planes around the stack's centre over which the two together are set to one photon, so the main half (70 % of the beads' light in this simulation) and the secondary (30 %) keep their share.  Right: how far each bead pair lands from its partner after the transformation, in pixels; the box is the *dx/dy limit* a pair must meet.
cal2 = dual.calibration
fig.set_size_inches(7.2, 3.0)
left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.4, 1]})
z2 = cal2.main.z_index_to_nm(np.arange(cal2.main.psf.shape[0]))
sums = [m.psf.sum(axis=(1, 2)) for m in (cal2.main, cal2.secondary)]
for s, name, color in zip(sums, ("main", "secondary"), ("#1f77b4", "#d62728")):
    left.plot(z2, s, color=color, label=name)
left.plot(z2, sums[0] + sums[1], color="0.4", lw=1, label="both")
centre = (len(z2) - 1) // 2
dz2 = cal2.main.dz
left.axvspan(z2[centre] - (FOCAL_PLANES + 0.5) * dz2, z2[centre] + (FOCAL_PLANES + 0.5) * dz2,
             color="0.85", zorder=0)
left.set_xlabel("emitter z (nm)", fontsize=8); left.set_ylabel("light per plane", fontsize=8)
left.set_ylim(0, 1.15)
left.legend(fontsize=7, frameon=False, loc="center right")
fig.subplots_adjust(wspace=0.35)
used = dual.transform_accepted
delta = dual.transform_fit.dxdy
limit = dual.beads.settings.transform_axis_limit_px
right.scatter(*delta[used].T, s=14, color="#2ca02c")
right.plot([-limit, limit, limit, -limit, -limit], [-limit, -limit, limit, limit, -limit],
           "k--", lw=0.7)
right.axhline(0, color="0.7", lw=0.5); right.axvline(0, color="0.7", lw=0.5)
right.set_xlim(-1.3 * limit, 1.3 * limit); right.set_ylim(-1.3 * limit, 1.3 * limit)
right.set_aspect("equal")
right.set_xlabel("dx (pixels)", fontsize=8); right.set_ylabel("dy (pixels)", fontsize=8)
for ax in (left, right):
    ax.tick_params(labelsize=7)
```

## In detail

**Detection.**  With $P$ the maximum projection smoothed by a Gaussian of
width *detection sigma*, a pixel is a bead if it is a $3\times3$ local
maximum and

$$P > \mathrm{median}(P) + t \cdot 1.4826\, \mathrm{median}\left|P - \mathrm{median}(P)\right| ,$$

$t$ the *threshold*: 1.4826 times the median absolute deviation estimates the
noise's standard deviation without being pulled up by the beads.  The bead's
position is the centre of mass of its maximum, rounded to a pixel.

**Extraction.**  A box of $n + 2p$ pixels ($n$ the *ROI size*, $p$ the
*padding*) is cut from every plane; $b$, the median of its outermost pixels
over all planes, is subtracted, and the box $V$ is divided by its brightness
$B = \max_k \sum_{x,y} V_k(x, y)$, the light of its brightest plane $k$.
The brightness limits are $B_\mathrm{med}/\sqrt{r}$ and
$B_\mathrm{med}\sqrt{r}$, with $B_\mathrm{med}$ the median over unsaturated
beads and $r$ the *brightness range*, so the brightest accepted bead is at
most $r$ times the dimmest.  Stacks of different lengths are cropped to the
shortest, centred.

**Alignment.**  A shift is the maximum of the normalised linear (not
circular) cross-correlation

$$C(\Delta) = \frac{\sum_\mathbf{r} R(\mathbf{r})\, M(\mathbf{r} - \Delta)}{\sqrt{\sum_{\mathrm{overlap}} R^2 \sum_{\mathrm{overlap}} M^2}} ,$$

each lag normalised by the energy in the part of the two volumes that
actually overlap at that lag, so an edge contributes nothing it does not
have.  The search is within a central $13 \times 13$ pixel window
laterally.  The integer peak is refined to a fraction of a pixel and a
plane from the correlation itself, without resampling the bead: $C$ is
taken, over one fixed overlap, at the 27 whole-voxel lags around the peak,
the triquadratic through their logarithms (every $z^i y^j x^k$ with
$i, j, k \leq 2$) is found, and its maximum within $\pm1$ voxel is the shift.
The logarithm, because the peak is close to a Gaussian; the triquadratic
rather than a quadratic, because at 20 nm steps the peak is some two hundred
times flatter along z than across, and its z curvature changes with the
lateral offset (an astigmatic PSF's width changes with z) -- a quadratic
folds that change into the z shift and pulls every bead towards a whole
plane.  On simulated beads at 20 nm steps the shifts come back within 1.4 nm
rms in z and 0.004 pixels laterally.  Maximising the correlation of a
linearly interpolated bead, as this did before, took nine times as long and
left 5.7 nm: interpolation smooths by an amount that depends on the
fraction, and draws the optimum towards whole planes.
The first pass covers the whole stack; the median shift is
subtracted, so that *max xy shift* and *max z shift* are measured from the
consensus of the beads and not from the reference; the refinement passes
(*iterations* minus one) correlate the central planes only, $\max(7, A/dz)$ of them, $A$ the *alignment range*.  Every bead is resampled once, with
cubic interpolation, from its original box by its accumulated shift.

**Shape score.**  Each bead is scaled to the current average $T$ by
$a_j = \mathrm{median}(V_j / T)$ over the brightest quarter of $T$'s voxels
(four passes), and compared with the average of the other accepted beads,
$T_{-j}$:

$$e_j = \sqrt{\frac{\left\langle (V_j/a_j - T_{-j})^2\right\rangle}{\left\langle T_{-j}^2 \right\rangle}}, \qquad s_j = \frac{e_j}{c_j} ,$$

$c_j$ their correlation coefficient (the table's *correlation*, and $e_j$
its *residual*).  Dividing by $c_j$ keeps a bead that is broadly wrong but
locally smooth from scoring well.  A bead is kept if
$s_j \leq \bar{s} + k\,\mathrm{sd}(s)$, $k$ the *rejection (MAD)*, where
the mean and standard deviation are taken over the beads within 4 MAD of the
median when there are at least six.  This is repeated three times after the
last alignment pass.

**The model.**  With $\bar V$ the average of the accepted, aligned beads over
the planes all of them cover and the central $n \times n$ pixels,

$$\mathrm{PSF} = \frac{\max\left(0,\ G * \bar V\right)}{\max_k \sum_{x,y} \max\left(0,\ G * \bar V\right)_k} ,$$

$G$ a Gaussian of standard deviation *smoothing z* $/\,dz$ planes along z and
*smoothing xy* pixels laterally (reflected at the edges).  The spline
coefficients are the not-a-knot cubic interpolant along x, then y, then z;
the file stores them with the model, the unsmoothed average, every bead's
shift and the reasons.  z = 0 is the centre plane of the cropped stack, and
the convention is that of the fitter: z is where the emitter is relative to
the focal plane, in objective nanometres.

**Dual colour: the normalisation.**  Both averages are smoothed the same way.
Before clipping, each half's light is read as the mean plane sum over the
centre plane and `FOCAL_PLANES` = 2 planes either side,

$$L_c = \frac{1}{5} \sum_{k = k_0 - 2}^{k_0 + 2}\ \sum_{x,y} \left(G * \bar V_c\right)_k , \qquad \mathrm{PSF}_c = \frac{\max\left(0,\ G * \bar V_c\right)}{L_1 + L_2} ,$$

so that the two models **together** hold one photon around focus and keep
the beads' ratio.  Each also stores its share,
$f_c = L_c / (L_1 + L_2)$, as `photon_normalization`.  Fitted with one photon
number for both halves (biplane), that number is the molecule's total;
fitted with a number per half (two colours), each is multiplied by its $f_c$
to give that half's photons.  The light is counted *before* clipping because
clipping keeps the noise in the tails above zero, and counted as light that
noise read the photons about 1.5 % high; counted before, it averages out.
Several planes are used so that one plane's noise does not set the scale, and
few enough that light defocusing out of the box does not.  The lowest value
the spline takes between knots, relative to its peak, is stored as
`spline_minimum` (the fitter floors its model for that).

**Dual colour: pairs.**  A bead's position is the elliptical Gaussian fit
(75 iterations) to the mean of the central 11 planes, 13 by 13 pixels; a fit
that fails or lands more than 4 pixels off rejects the bead.  The offset
between the halves is voted for in a histogram of 2-pixel bins of every
difference between a main-half and a secondary-half bead of the same stack,
smoothed; pairs are mutual nearest neighbours within 12 pixels of each other
after that offset, and never cross stacks.  Fewer than *min pairs* stops the
calibration.

**Dual colour: the transformation.**  In homogeneous coordinates the map is
$\mathbf{x}_m \propto H\, \mathbf{x}_s$ ($H$ a $3\times3$ matrix, secondary to
main, in chip pixels).  Round one: RANSAC
([Fischler & Bolles 1981](https://doi.org/10.1145/358669.358692)) -- 500
draws of four pairs (a fixed seed, so it is reproducible), each solved by the
normalised direct linear transform, keeping the draw with the most pairs
within *RANSAC radius*; the inlier set is refitted until it stops changing,
and then refined by least squares on the reprojection error in the main half
with the robust soft-L1 loss

$$\rho(r) = 2 s^2 \left(\sqrt{1 + (r/s)^2} - 1\right) ,$$

$s$ half the *dx/dy limit*, which is quadratic for small errors and grows only linearly for large ones.
Round two keeps the round-one inliers whose $|dx|$ **and** $|dy|$ are both
within the *dx/dy limit* and fits again on that fixed set.  The map must not
cross infinity anywhere on the camera image, forwards or backwards.  The
PSFs never enter this fit: a pair rejected for its shape still counts for
the transformation, and the transformation list and the PSF list are kept
apart (the table's *state* says which a pair is in).

**Dual colour: the paired stacks.**  The secondary box is mirrored if the
layout is, and shifted by the sub-pixel difference between where it was cut
and where the transformation puts the partner of the main box -- a shift,
not a projective warp of the image.  Both halves are divided by one number,
the sum of their brightnesses, and then aligned with one $(z, y, x)$ shift
per pair: the correlation adds each half's numerator and energies, never
correlating one half with the other.  A pair whose correction exceeds the
padding is kept for the transformation but not the PSF.  The secondary model
is mirrored back to the camera's orientation before it is saved.

**Fit quality.**  Single channel: every fifth plane of each accepted bead
(ROI of *ROI size* minus 4 pixels, the camera values restored) and of the
average stack, fitted with the spline fitter from five z starts, keeping the
best likelihood.  Dual colour: the pairs fitted with the two-channel global
fit, x, y and z shared, one z per pair
([Li et al. 2022](https://doi.org/10.1038/s41467-022-30719-4)), which also
shows the fitted split of the photons.  The *centred* error removes each
bead's median error, since a bead's absolute height on the coverslip is not
known; it measures how distorted the z scale is, not how accurate.

**Compared with the paper.**  The method is that of
[Li et al. 2018](https://doi.org/10.1038/nmeth.4661): beads registered in
3D, averaged, and interpolated by cubic splines.  The main changes are those
listed under *Differences from SMAP*: a Gaussian smoothing in place of a
smoothing B-spline, a linear correlation, a leave-one-out shape rejection,
and the joint normalisation of the two halves.

## Controls

### bead stacks
The files or folders of bead z-stacks (TIFF or OME-TIFF).  All stacks are
pooled into one calibration, so they must share the z step.

### Add files...
Adds stacks one by one.

### Add folder...
Adds a folder; with *search subdirectories* ticked, every acquisition found
inside it.

### Remove
Removes the selected entries from the list.

### search subdirectories
Whether *Add folder...* looks for acquisitions inside the folder's
subfolders, or adds the folder as it is.

### mode
*Single channel* or *Dual colour*.  Switching keeps the files and the
settings the two modes share, clears the result, and shows or hides the
split-camera settings; the *Transformation* and *Field diagnostics* pages
exist only in dual colour.

### settings
The calibration's settings.  The common ones are shown; *more* has the
rest, whose defaults rarely need changing.

### ROI size
The model's width, in camera pixels; odd, at least 7.  It must hold the
spot at the ends of the z range, where it is widest: too small cuts off the
tails, too large lets neighbouring beads in.  The fitter's ROI may be
smaller.

### z step
Leave it on auto when the files record the step.  A wrong step stretches or
squeezes every fitted z by the same factor.

### threshold
Lower it if clear beads are missed, raise it if noise or dim debris is
detected; 6 is a good start for bright beads.

### min distance
Larger than twice the half-width of a defocused bead's spot, or the tails
of a neighbour end up in the model.  Beads closer than this are both
dropped, so dense fields lose many beads.

### smoothing z
The width of the smoothing along z.  About one to two z steps is usual.
Too little leaves noise ripples that become stripes in z; too much blurs
the PSF's change with z and costs z precision.

### padding
Extra pixels around the ROI for the alignment to shift into; must exceed
*max xy shift* by more than one pixel.

### detection sigma
The smoothing before detection; about the spot's width.

### max xy shift
Beads that need a larger lateral shift than this are rejected.

### max z shift
Beads that sit further than this from the others in height are rejected: a
bead stuck higher up, or one the correlation mismatched.

### alignment range
The z range around the centre that the refinement correlates over.  Around
the focus the PSF has most structure, far from it least.

### iterations
The number of alignment passes: one coarse, then refinements.

### rejection (MAD)
The shape cut, in standard deviations of the beads' scores (see *In
detail*).  Smaller rejects more beads, larger keeps more.

### smoothing xy
Lateral smoothing in pixels; 0, the default, is usually right, because the
beads' alignment already averages laterally.

### min beads
The fewest accepted beads the calibration will be built from.  Fewer than ten
gives a warning in any case.

### brightness range
The largest ratio between the brightest and the dimmest accepted bead,
centred on the median; auto disables the check.

### saturation
The camera value at which a bead counts as saturated; auto takes the file's
integer maximum (65535 for 16-bit).

### layout
How the camera chip is split: *right-left* or *up-down*, and whether the
secondary half is mirrored along the split (an image splitter with an odd
number of mirrors in one path).  Look at a bead frame: if the pattern of
beads in one half is the other's mirror image, choose a *mirrored* layout.

### main channel
Which half is the reference that positions are reported in.  It must match
the layout: *left*/*right* for right-left, *upper*/*lower* for up-down.

### split position
The first pixel of the second half; auto is the middle of the image.

### min pairs
The fewest bead pairs the transformation may be fitted from; at least 4,
which is what a projective map needs.  More pairs, spread over the field,
make the map reliable towards the edges.

### RANSAC radius
The first round's tolerance, in pixels.  It only has to separate right
pairings from wrong ones, so it can be generous.

### dx/dy limit
The second round's tolerance on each axis, in pixels.  Good bead data sits
well below 0.1 pixel; if too few pairs pass, look at the *Transformation*
page before raising it.

### Detect + calibrate
Runs everything: detection, extraction, alignment, rejection, the model (and
in dual colour the pairs and the transformation).  It runs in the
background; the status bar says how many beads were used.

### Calculate fit quality
Refits the beads with the model just built and draws the *Fit quality*
page.

### Save calibration...
Writes the calibration as `.h5`, by default named after the bead data's
folder and saved beside it.  Opened from the main window, saving also puts
it into the fitter's *calibration* field -- Spline 3D for a single-channel
calibration, Spline 3D 2C for a dual-colour one.

### Use in the Spline 3D fitter
Puts the calibration last saved into the fitter again, for when the field
was changed since.  In dual-colour mode the button says *Spline 3D 2C*.

### Browse average stack...
Opens the averaged bead stack -- the raw average, before smoothing, so that
a bad bead or a misalignment still shows -- slice by slice in its own
window; in dual colour both halves side by side.

### Overview
The stack's projection with its beads (green used, red rejected), the
brightness of every bead against its lateral shift with the accepted
brightness band, a side view of the model, and each bead's shape residual
against its z shift.  In dual colour: the split frame with its pairs, and
the main half's field with pairs used for the PSF and the transformation
(green), for the transformation only (orange), rejected (red) and unpaired
beads.  Select a bead in the table to see its own stack.

### Transformation
Dual colour: how far each pair lands from its partner after the
transformation (round two), and the first round's residuals with the *dx/dy
limit* box.  A tight, round cloud centred on zero is good; a cloud with a
tail or two clusters means wrong pairs or the wrong layout.

### Fit quality
After *Calculate fit quality*: fitted against expected z for every bead and
for the average, the centred error, and the lateral and axial profiles of the
beads against the model.  Good: a straight line of slope 1 over the
useful range and a centred error of a few nanometres; the ends of the range,
where the spot is faint and wide, are always worse.  Dual colour adds the
fitted photon split, secondary over total, which should be flat in z at one
half: the models already carry the beads' split.

### Field diagnostics
Dual colour: the main half's beads overlaid with the transformed secondary
ones, and how the beads' shape mismatch varies over the field.  A mismatch
that grows towards one side is a PSF that changes over the field, which one
model cannot describe.

### Bead diagnostics
Dual colour: per half, brightness against lateral shift, and the shape
residual against the z shift and against the correlation, coloured by what
each pair was used for.

### Exclude / include
Marks the selected beads in the table as excluded (or includes them again);
space does the same.  Takes effect on *Recalculate*.

### Recalculate
Builds the calibration again from the beads already found, without the
excluded ones.  In dual colour it refits the transformation too.

## Differences from SMAP

Based on SMAP's bead calibrator, `calibrate3D_GUI_g` in `fit3Dcspline`
(`calibrate3D_g` and `getstackcal_g` for one channel,
`calibrate_globalworkflow` for two;
[Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  The main changes:

* **Background.**  SMAP subtracts the minimum of each stack when reading it,
  and the minimum of the averaged PSF around focus before normalising.  Here
  each bead's own background, the median of its box's border over all planes,
  is subtracted before averaging; a minimum is set by the noisiest pixel.
* **Smoothing** is a Gaussian of a width in nanometres, where SMAP fits a
  smoothing B-spline with a penalty *lambda* whose effect depends on the z
  step.  The width here means the same at any step.
* **Alignment** uses a linear, overlap-normalised correlation rather than a
  circular one, the shift limits reject a bead after the search instead of
  bounding it, and a bead's shape is scored against the average of the
  *other* beads.  SMAP finds the sub-voxel peak by upsampling both stacks
  fourfold and interpolating their correlation cubically; here the
  correlation is taken at whole voxels only, and its peak interpolated by
  the triquadratic above.
* **Two halves, normalised together.**  SMAP divides both averages by the
  main half's brightest plane (main at 1, secondary at its ratio).  Here the
  two halves' light around focus together is one photon, counted before
  clipping, and each half stores its share (`photon_normalization`), so a
  fit with the photons linked (biplane) returns the molecule's total and a
  fit with a number per half returns each half's photons.
* **The transformation** comes from a Gaussian fit to each bead's central
  planes and is projective only, fitted by RANSAC and a soft-L1 refinement in
  two rounds, independently of the PSFs.  SMAP first calibrates each half
  separately, takes the bead positions from those 3D fits, and fits the
  transformation type chosen in the GUI with progressively tighter cutoffs.

## References

* Li Y, Mund M, Hoess P, et al. Real-time 3D single-molecule localization
  using experimental point spread functions. *Nat Methods* 15, 367 (2018).
  [doi:10.1038/nmeth.4661](https://doi.org/10.1038/nmeth.4661) -- the
  method: beads registered, averaged and interpolated by cubic splines.
* Li Y, Shi W, Liu S, et al. Global fitting for high-accuracy multi-channel
  single-molecule localization. *Nat Commun* 13, 3133 (2022).
  [doi:10.1038/s41467-022-30719-4](https://doi.org/10.1038/s41467-022-30719-4)
  -- the global fit of both halves, which the dual-colour fit quality uses.
* Fischler MA, Bolles RC. Random sample consensus: a paradigm for model
  fitting with applications to image analysis and automated cartography.
  *Commun ACM* 24, 381 (1981).
  [doi:10.1145/358669.358692](https://doi.org/10.1145/358669.358692) --
  RANSAC, the robust first fit of the transformation.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
