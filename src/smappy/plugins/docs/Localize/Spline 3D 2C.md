---
version: "4"
covers: [smappy.psf.GlobalSplinePSF, smappy.dualfit.combine_peaks, smappy.dualfit.build_link, smappy.dualfit.cut_paired_rois, smappy.dualfit.paired_to_localizations, smappy.dualfit.DualChannelEngine, smappy.calibrate.dual.build_dual_calibration, smappy.calibrate.dual.fit_dual_transform, smappy.calibrate.core.positive_pair_models, smappy.calibrate.dual.load_dual_color_calibration, smappy.plugins.fit.DualSplineFit, smappy.plugins.fit.finish_localizations]
---

## What it does

This is the fitter for **two halves of one camera in 3D**, and it serves two
experiments:

* **Two colours.**  A dichroic sends each molecule's light to both halves,
  in a proportion that depends on the dye.  The photons of each half are
  fitted separately, and their split tells the dyes apart.
* **Biplane.**  A beam splitter sends the light to both halves with the two
  halves focused at different heights.  The split is the same for every
  molecule and known from the beads, so only **one photon number** is
  fitted, and both halves together pin z.  Tick *link photons* for this.

For two colours it combines two things other pages explain: the split camera of
[Gaussian 2D 2C](plugin:Localize/Gaussian 2D 2C) -- a dichroic sends each
molecule's light to both halves of the chip, and the proportion in each half
tells the dyes apart -- and the measured, z-dependent PSF of
[Spline 3D](plugin:Localize/Spline 3D), which gives every localization a
height.

Each molecule is fitted in **both halves at once, as one emitter**: one x,
one y and **one z** for both halves, and a photon number for each.  Sharing z
is the main gain.  The two spots are two independent looks at the same
height, so z comes out up to $\sqrt{2}$ times more precise than from one half
alone, while the photon split between the halves -- the colour -- is left
free.  At the end of the run the colours are assigned from that split
([Assign colours](plugin:Analysis/Dual-Color/AssignColors)) and, if asked,
the drift is corrected ([RCC](plugin:Analysis/Drift/RCC) or
[COMET](plugin:Analysis/Drift/COMET)).

What it needs:

* **A dual-colour bead calibration** of this microscope, made with
  **Tools > Dual-colour calibration** (the Bead calibration window in its
  *Dual colour* mode): z-stacks of beads that are seen in both halves.  It
  holds everything the fit needs about the two halves -- which half is which,
  the transformation between them, and a PSF model for each.  A
  single-channel calibration or a SMAP `_3dcal.mat` will not do.
* **The camera**, as for any fit.
* **Sparse data**, as for [Spline 3D](plugin:Localize/Spline 3D): a
  neighbour inside the ROI pulls z towards focus.

For 2D data, use [Gaussian 2D 2C](plugin:Localize/Gaussian 2D 2C), which
needs no bead calibration at all.

## How it works

```figure-setup
import warnings
from scipy.special import erf
from smappy.calibrate.dual import DualColorSettings, calibrate_dual
from smappy.calibrate.input import BeadStack
from smappy.io.calibration import evaluate_spline
from smappy.plugins.fit import DualModelSettings
from smappy.psf import SplinePSF
from smappy.simulate import ASTIGMATISM, astigmatic_sigmas, dual_bead_stacks, dual_transformation
# one simulated bead z-stack on a split camera (nine beads in each half, an
# astigmatic PSF, 50 nm steps) -> a dual-colour calibration; kept small, so quick
stacks, z_objective = dual_bead_stacks(1, seed=0, dz_nm=50.0, z_range_nm=(-700.0, 700.0))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")        # no camera ROI in a simulation: ROI-local
    cal = calibrate_dual([BeadStack(stacks[0].astype(np.float32), z_objective)],
                         DualColorSettings(layout="up-down", main_channel="upper", dz_nm=50.0,
                                           roi_size=17, smooth_z_nm=40.0)).calibration
rng = np.random.default_rng(1)

def true_psf(z_nm, x0, y0, sigma_nm, size=13):
    """What the beads were drawn with: an astigmatic, pixel-integrated Gaussian."""
    sx, sy = (w / 100.0 * np.sqrt(2.0) for w in astigmatic_sigmas(np.asarray(z_nm, float), sigma_nm, *ASTIGMATISM))
    k = np.arange(size)[None]
    ex = 0.5 * (erf((k + 0.5 - x0[:, None]) / sx[:, None]) - erf((k - 0.5 - x0[:, None]) / sx[:, None]))
    ey = 0.5 * (erf((k + 0.5 - y0[:, None]) / sy[:, None]) - erf((k - 0.5 - y0[:, None]) / sy[:, None]))
    return ey[:, :, None] * ex[:, None, :]

def pairs(z_true, secondary_share, photons=2000.0, background=10.0):
    """ROI pairs of one molecule each, both halves cut on the spot, drawn from
    the true PSF of each half (130 and 145 nm wide)."""
    x0, y0 = 6 + rng.uniform(-0.5, 0.5, (2, len(z_true)))
    shares = (1 - secondary_share, secondary_share)
    images = np.stack([rng.poisson(background + photons * share * true_psf(z_true, x0, y0, sigma))
                       for share, sigma in zip(shares, (130.0, 145.0))], axis=1)
    link = np.zeros((len(z_true), 2, 2, 5), np.float32)
    link[:, 1] = 1.0                      # no sub-pixel offset, unit factors
    return images.astype(np.float32), link, x0

levels = np.arange(-400.0, 401.0, 100.0)
z_true = np.repeat(levels, 250)
images, link, x0 = pairs(z_true, 0.5)
one = SplinePSF(cal.main)
alone = one.unpack(one.fit(images[:, 0]))
both = DualModelSettings().model(cal)
linked = both.unpack(both.fit(images, link))
```

The detection, the pairing of the two halves' peaks, the ROIs and the global
fit are those of [Gaussian 2D 2C](plugin:Localize/Gaussian 2D 2C), which
explains them step by step: each half is thresholded on its own, the
secondary peaks are mapped onto the main half, peaks within 4 pixels are
merged into one candidate and peaks without a partner are kept, and a ROI is
cut in each half and both are fitted at once by maximum likelihood.  What is
new here is the model in each ROI, where it comes from, and z.

**1. The calibration.**  The Dual-colour calibration steps the objective
through focus over beads, which appear in both halves.  From the stacks it
takes three things:

* **The transformation.**  Each bead is found in both halves; the pairs are
  matched, and a projective map from the secondary half to the main half is
  fitted to their positions -- first robustly (RANSAC), then refined on the
  pairs whose residual is within 0.15 pixels in both x and y.
* **A PSF per half, on one z grid.**  The two images of a bead are aligned to
  the other beads with **one** shift in x, y and z for both halves together,
  then averaged and interpolated with a cubic spline, as for
  [Spline 3D](plugin:Localize/Spline 3D).  Because the shift is shared, the
  two PSFs have the same focal plane and the same planes: z = 0 means the
  same height in both halves.  Any real difference between the halves -- one
  half focused a little higher, say -- stays in the models rather than being
  aligned away.
* **Their relative brightness.**  Both PSFs are scaled by **one** factor,
  chosen so that the two together hold one photon around the focal plane.
  So the pair keeps the beads' split: with a fifth of the light in the
  secondary half, the main PSF holds 0.8 of a photon and the secondary 0.2.
  Each half's share is stored with its PSF.  The median of the beads'
  brightness ratio is saved too.

**2. Two uses of one calibration.**  How the photons are read depends on
whether they are linked.

* **Linked (biplane).**  One photon number $N$ is fitted, and each half's
  model is $N$ times its PSF.  Because the PSFs carry the beads' split and
  hold one photon together, $N$ is the molecule's total photons, split as
  the beads split -- nothing has to be typed in.
* **Free (two colours).**  A photon number is fitted in each half against
  that half's PSF.  A number fitted against a PSF that holds 0.2 of a photon
  is five times the photons it stands for, so each half's number is
  multiplied by its stored share.  `photons_ch0` and `photons_ch1` are then
  the photons in each half, `photons` their sum, and `ratio` the dye's own
  fraction in the secondary half -- whatever the beads' split was.

```figure Both uses give photons back as photons.  Left, two colours: molecules of 2000 photons with a quarter in the secondary half, fitted with the photons free -- each half's photons and their sum, against the truth (grey).  Right, biplane: molecules that split as the beads did, fitted with one photon number, which is the total.
dye = pairs(np.repeat(levels, 60), 0.25)[0]
free = both.unpack(both.fit(dye, link[:len(dye)]), link[:len(dye)])
from smappy.psf import GlobalSplinePSF
biplane = GlobalSplinePSF((cal.main, cal.secondary), (True,) * 5)
bi_images = pairs(np.repeat(levels, 60), 0.5)[0]
bi = biplane.unpack(biplane.fit(bi_images, link[:len(bi_images)]), link[:len(bi_images)])
fig.set_size_inches(7, 2.8)
axes = fig.subplots(1, 2, sharey=True)
bins = np.linspace(0, 3200, 65)
for name, truth, colour in (("photons_ch0", 1500, "#1f77b4"), ("photons_ch1", 500, "#d62728"),
                            ("photons", 2000, "#2ca02c")):
    axes[0].hist(free[name], bins, histtype="step", color=colour, label=name)
    axes[0].axvline(truth, color="0.6", lw=1, zorder=0)
axes[1].hist(bi["photons"], bins, histtype="step", color="#2ca02c", label="photons")
axes[1].axvline(2000, color="0.6", lw=1, zorder=0)
axes[0].set_title("two colours: photons free", fontsize=9)
axes[1].set_title("biplane: one photon number", fontsize=9)
for ax in axes:
    ax.set_xlabel("photons")
    ax.legend(frameon=False, fontsize=7)
axes[0].set_ylabel("molecules")
```

```figure The calibration.  Left: the PSF model of each half at three heights, as the fit sees them -- the same astigmatism, the secondary half's spot a little wider.  Right: how far the calibration's transformation, measured on nine bead pairs, is from the true one over the secondary half, in pixels.
fig.set_size_inches(7.5, 3.3)
grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.25, 2.2])
heights = (-400.0, 0.0, 400.0)
for row, (name, model) in enumerate((("main", cal.main), ("secondary", cal.secondary))):
    for col, z in enumerate(heights):
        ax = fig.add_subplot(grid[row, col])
        ax.imshow(evaluate_spline(model, 6.0, 6.0, float(model.z_nm_to_index(z)), 13), cmap="magma")
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f"z = {z:+.0f} nm", fontsize=8)
        if col == 0:
            ax.set_ylabel(name, fontsize=8)
ax = fig.add_subplot(grid[:, 4])
gx, gy = np.meshgrid(np.arange(0.0, 100.0), np.arange(100.0, 200.0))
points = np.c_[gx.ravel(), gy.ravel()]
truth_map = np.c_[points, np.ones(len(points))] @ dual_transformation().T
error = cal.to_reference(points) - truth_map[:, :2] / truth_map[:, 2:]
size = np.hypot(error[:, 0], error[:, 1]).reshape(gx.shape)
shown = ax.imshow(size, cmap="viridis", vmin=0)
ax.set_title(f"map error: at most {size.max():.3f} px", fontsize=8)
ax.set_xlabel("x (px)", fontsize=8); ax.set_ylabel("y within the half (px)", fontsize=8)
ax.tick_params(labelsize=7)
fig.colorbar(shown, ax=ax, fraction=0.046, label="px").ax.tick_params(labelsize=7)
```

**3. The model in each half.**  Each ROI is compared with its own half's
PSF: for half $c$, the calibrated PSF of that half at the molecule's position
and height, scaled to $N_c$ photons, on a background $b_c$.  x and y reach the
secondary half through the transformation, exactly as in
[Gaussian 2D 2C](plugin:Localize/Gaussian 2D 2C).  z needs no mapping at all:
the two PSFs share one z grid, so the same z is the same plane in both.

**4. What is linked.**  By default x, y and z are linked -- one value each for
both halves -- and the photons and the background are free, one per half.
This is the choice that the two-colour experiment needs: every photon of both
spots informs the position and the height, and nothing forces the photons to
divide in a fixed way, so their split is measured and is the colour.  Linking
the photons too (*link photons*) would make the fit more precise still, but
then the split is imposed rather than measured and there is no colour left to
read -- the right choice for two halves that see the *same* dye (biplane),
not for two dyes.

```figure What linking buys.  Molecules simulated at known heights from the true PSF (2000 photons split evenly between the halves, 10 background photons per pixel in each) and fitted with the calibration above: the scatter about the truth (dots) and the precision the fit reports (lines), from the main half alone (orange) and from both halves with x, y and z linked (blue).  z is about 1.4 times more precise, near the $\sqrt{2}$ of two equal looks.
fig.set_size_inches(7.5, 3.1)
axes = fig.subplots(1, 2)
for ax, (name, err, truth, scale) in zip(axes, (("z_nm", "z_err_nm", z_true, 1.0),
                                              ("x_roi", "x_err_pix", x0, 100.0))):
    for fitted, colour, label in ((alone, "#ff7f0e", "main half alone"),
                                  (linked, "#1f77b4", "both halves, linked")):
        ok = np.isfinite(fitted[name])
        scatter = [np.std((fitted[name] - truth)[ok & (z_true == z)]) * scale for z in levels]
        bound = [np.median(fitted[err][ok & (z_true == z)]) * scale for z in levels]
        ax.plot(levels, bound, color=colour, label=label)
        ax.plot(levels, scatter, "o", ms=4, color=colour)
    ax.set_xlabel("true z (nm)")
    ax.set_ylim(0, None)
axes[0].set_ylabel("z precision (nm)")
axes[1].set_ylabel("x precision (nm, 100 nm pixels)")
axes[0].legend(frameon=False, fontsize=8)
```

**5. z, and the colour.**  z is converted from the calibration's planes as
in [Spline 3D](plugin:Localize/Spline 3D): relative to the calibration's
focal plane, in the nanometres the objective moved.  The photons of the two
halves come out as `photons_ch0` and `photons_ch1`, their sum as `photons`,
and `ratio` is the fraction in the secondary half, $N_1 / (N_0 + N_1)$.
Because the fitted height enters both halves' models, the photon split is
measured with the PSF of the right height in each half; the colour does not
drift with z.

```figure The colour does not depend on the height.  Two dyes simulated at known heights, putting 25 % and 75 % of their light into the secondary half, fitted as above: the fitted ratio of each molecule against its true z.
dye_z = np.repeat(levels, 60)
fig.set_size_inches(5.2, 2.9)
ax = fig.subplots()
for share, colour in ((0.25, "#d62728"), (0.75, "#2ca02c")):
    pair_images, pair_link, _ = pairs(dye_z, share)
    fitted = both.unpack(both.fit(pair_images, pair_link))
    ax.plot(dye_z + rng.uniform(-25, 25, dye_z.size), fitted["ratio"], ".", ms=3,
            alpha=0.5, color=colour, label=f"{share:.0%} in the secondary half")
    ax.axhline(share, color="k", ls="--", lw=0.8)
ax.set_ylim(0, 1)
ax.set_xlabel("true z (nm)")
ax.set_ylabel("ratio")
ax.legend(frameon=False, fontsize=8, loc="center right")
```

**6. Finishing.**  As in [Gaussian 2D 2C](plugin:Localize/Gaussian 2D 2C):
once the last frame is fitted, drift correction (off by default; RCC or
COMET, which corrects z as well when the table has it) and colour assignment
(on by default) run over the whole table, with the Analysis tab's own
plugins, before the file is finished.  A step that fails is a line in the
output, and the table is saved regardless.

## In detail

**The model.**  For pixel $i$ of the ROI of half $c$,

$$\mu_{c,i} = b_c + N_c\, \mathrm{PSF}_c(x_i - x_c,\; y_i - y_c,\; z) ,$$

with $\mathrm{PSF}_c$ the cubic spline of half $c$, $(x_0, y_0) = (x, y)$
the linked position in the main ROI, and $(x_1, y_1)$ the same position seen
through the link: $x_1 = f_x\, x + o_x$, with $f_x$ the local scale of the
transformation and $o_x$ the sub-pixel remainder of cutting the partner ROI at
a whole pixel (and the same for y).  z has factor 1 and offset 0 in both
halves; the calibration's two PSFs are checked on loading to have the same
grid, plane spacing and focal plane, and the fit refuses them otherwise.  By
default the fitted vector is $(x, y, N_0, N_1, b_0, b_1, z)$, seven numbers
for the ten of two separate fits.  The likelihood, the Levenberg-Marquardt
steps and the Cramér-Rao bound are those of
[Gaussian 2D 2C](plugin:Localize/Gaussian 2D 2C), summed over the pixels of
both ROIs, with the spline's derivatives of
[Spline 3D](plugin:Localize/Spline 3D).  z starts at the calibration's focal
plane (there is no *start z* setting here), and is held inside the
calibration's range.

**Why $\sqrt{2}$.**  The information about z is summed over both ROIs:

$$I_{zz} = \sum_{c} \sum_i \frac{1}{\mu_{c,i}} \left( N_c\, \frac{\partial\,\mathrm{PSF}_c}{\partial z} \right)^2 .$$

With the photons split evenly and two halves whose PSFs change with z in the
same way, each half contributes the same, and the variance of z is half that
of one half alone: $\sigma_z$ shrinks by $\sqrt{2}$.  A dye that puts most of
its light into one half gains less, since the other half adds little; and
because x, y and z are linked, the photons of both halves count in every one
of them.

**The normalisation.**  Each half's PSF is the average of its bead images,
smoothed and clipped at zero: the tails of an average of a few beads scatter
about zero, and a PSF has no negative light.  Half $c$'s signal is

$$S_c = \left\langle \sum_{\mathrm{ROI}} \mathrm{PSF}_c(z) \right\rangle_{|z| \leq 2\,\Delta z} ,$$

the plane sum averaged over the focal plane and the two planes either side
of it ($\Delta z$ the calibration's step).  Both PSFs are divided by
$S_0 + S_1$, and $s_c = S_c / (S_0 + S_1)$ is stored with each, so
$s_0 + s_1 = 1$.

Clipping keeps a little of the noise, the part above zero: on a simulated
calibration from 9 beads the photons read 1.5% high and the ratio 0.005 high,
and from 27 beads 0.8% and 0.003.  Between its knots the cubic spline can dip
slightly below zero next to a clipped tail; the calibration records the
lowest value as `spline_minimum` (a fraction of the peak, a few per mille),
and the fit floors its model at $10^{-3}$ photons, so a low background never
makes it negative.  Before version 4 the PSFs were not clipped but lifted: a
constant was added to every pixel, large enough to prove the spline positive
everywhere.  That constant was the worst noise dip anywhere in the volume --
11% of a bright half's signal and 40% of a half with a fifth of the light,
from 9 beads -- and the fit's background took it: the photons survived, but
the fitted background read low by the photons times the constant, and at a
photon per pixel it sat on its floor.

**Photons.**  With the link's photon factor $f_c$ ($f_0 = 1$), half $c$'s
model is $f_c \hat{N}\,\mathrm{PSF}_c$ when the photons are linked, and
$\hat{N}_c\,\mathrm{PSF}_c$ when they are free.  What is reported is

$$N = \hat{N} \sum_c f_c\, s_c \quad \mathrm{(linked)}, \qquad N_c = s_c\, \hat{N}_c ,\quad N = \sum_c N_c \quad \mathrm{(free)} .$$

With no *photon ratio* given, $f_1 = 1$: the split is the one the PSFs carry,
the beads', and linked $N = \hat{N}$.  A *photon ratio* $r$ (secondary over
main photons) sets $f_1 = r\, s_0 / s_1$, which splits the one photon number
as $r$ says, and $N$ is still the total.  On simulated beads with a fifth of
their light in the secondary half, a dye with a quarter comes back at a
`ratio` of 0.25 and its true photons, and a biplane fit of molecules that
split as the beads did gives their total, both within 1%.  (Before version
2 the secondary's photons were read in the beads' units -- that dye came
back at 0.57, with 3.4 times its photons -- and before version 3 a linked
fit reported the main half's share rather than the total.)  A calibration
saved before the shares were stored had the main half's brightest plane
at 1, and its shares are worked out from that on loading.

**Mirrored splitters.**  A splitter that mirrors one half is handled by the
link: the local scale of the transformation along the mirrored axis is
$-1$, and the partner ROI is evaluated about its centre.  The calibration
stores the secondary PSF in the camera's own orientation, so no image is
flipped.

## Parameters

### model.calibration
Made once per microscope configuration: the same objective, dichroic,
filters, camera settings and split as the data.  It records whether the beads
were taken with EM gain, and data taken the other way is warned about (an EM
register mirrors the image, and the PSF with it).  The first frames are
checked against its geometry: a calibration made on another camera ROI is
refused, and a movie with no camera ROI in its metadata is taken to start at
the chip's corner, with a warning.

### model.link_xy
Unlinked, each half fits its own position.  The table's position is the
main half's, and `dx_nm_ch1`, `dy_nm_ch1` say how far the secondary half put
the molecule from where the calibration's transformation expects it: a check
of the calibration on this data, whose median should be near 0.  Relink for
the real fit.

### model.link_z
Unlinked, each half fits its own z.  `z_nm` is the main half's -- the
precision of one half -- and `dz_nm_ch1` is the secondary half's z minus it,
which should scatter about 0 at every height if the two PSF models share
their focus.

### model.link_photons
On for biplane, off for two colours.  Linked, `photons` is the total and
`ratio` is 0 for every localization -- there is no split left to measure, and
colour assignment has nothing to work with -- but z is more precise still,
since both halves' photons count as one.

### model.photon_ratio
Only used with *link photons* on; with the photons free it has no effect.
Leave it empty: the calibration's PSFs already carry the beads' split, which
is the molecules' too when the photons can be linked at all (biplane, one
dye).  Give it only when the molecules split differently from the beads --
a narrow-band dye behind a splitter whose transmission depends on colour.

### fit.roisize
Both ROIs have this size.  As for [Spline 3D](plugin:Localize/Spline 3D) it
should hold the widest, most defocused spot, and stay inside the
calibration's lateral size.

### finish.drift
With z in the table, both RCC and COMET correct z as well unless their
*correct z* says otherwise.

## Output

The table has the columns of [Spline 3D](plugin:Localize/Spline 3D), with the
photons of each half beside the totals:

| column | meaning |
| --- | --- |
| `x_nm`, `y_nm`, `z_nm` | the linked position, in the main half's coordinates, and the height |
| `x_err_nm`, `y_err_nm`, `xy_err_nm`, `z_err_nm` | their precision (CRLB), from both halves |
| `photons`, `photons_err` | the photons of both halves together: the sum of the two, or, linked, the one fitted number |
| `photons_ch0`, `photons_ch1` | photons free: the photons of the main and the secondary half, and `photons_err_ch0`, `photons_err_ch1` |
| `ratio` | $N_1 / (N_0 + N_1)$, the colour |
| `background_ch0`, `background_ch1` | the background per pixel of each half (`background` is the main half's) |
| `channel` | the colour Assign colours gave it: 1, 2, or 0 for none |
| `logl`, `logl_rel` | the log-likelihood, and per pixel of both ROIs |
| `peak_x_nm`, `peak_y_nm`, `iterations` | the candidate in the main half, and the steps taken |
| `dx_nm_ch1`, `dy_nm_ch1`, `dz_nm_ch1` | unlinked only: the secondary half's position against the calibration's |

What to check, beyond the checks of [Spline 3D](plugin:Localize/Spline 3D)
(the z histogram, `logl_rel`, `z_err_nm`):

* **The preview** draws the secondary half's peaks mapped onto the main half
  with the calibration's transformation (cyan crosses); a cross on its
  partner's circle means the calibration fits this data.  A calibration made
  with a different split, camera ROI or splitter alignment misses.
* **The ratio histogram** should show one peak per dye, and should not change
  with z: a ratio that drifts with height means the two PSFs do not describe
  the two halves at the same z.

## Differences from SMAP

The workflow is SMAP's `fit_global_dualchannel` -- `PeakFinder`,
`PeakCombiner`, `RoiAdder`, `RoiCutterWF` and the `MLE_global_spline` fitter,
the GlobLoc fitter of Li et al. (2022) -- run here on the CPU.  The
differences:

* **The calibration** is the Dual-colour calibration's HDF5 file, not a
  `_3dcal.mat` from SMAP's global calibration; one calibration serves the
  whole field (SMAP can hold several regions), and there is no refractive
  index factor (SMAP's is 0.8 when switched on).
* **The link is fixed by the form**: x, y, z, photons and background, each
  linked or not.  SMAP's global table also sets a factor per parameter;
  channel weights, a choice of which channel's x and y are reported (SMAP:
  either, or the mean), several z starts and sCMOS variance maps are not
  offered.  x and y are always the main half's.
* **The photons are scaled back the same way.**  SMAP stores a
  normalisation per channel with the global calibration (`normf`),
  multiplies each channel's fitted photons by it and divides a typed-in
  photon ratio by the secondary's.  Here the PSFs are normalised together
  so that they hold one photon at focus, the share $s_c$ is what is stored,
  and a linked fit with no ratio given takes the beads' split and reports
  the total.
* **The tails are clipped, not lifted.**  SMAP's calibration (`getstackcal`)
  subtracts the volume's minimum -- lifting every pixel by the worst noise
  dip -- counts that lift into the normalisation, and then clips at zero.
  Here the PSFs are only clipped, so the fitted background is the
  background.
* **No image is mirrored.**  SMAP flips the second channel's ROIs for a
  mirrored splitter; here the secondary PSF is kept in the camera's
  orientation and the mirror is a factor of $-1$ in the link.
* **Detection per half, pairing, and the finishing steps** are those of
  [Gaussian 2D 2C](plugin:Localize/Gaussian 2D 2C): the dynamic cutoff set for
  each half on its own, a pair merged at a mean weighted by the square root of
  the peak heights (SMAP's two-channel `PeakCombiner` weights by their square),
  and drift correction and colour assignment as part of the fit.

## References

* Li Y, Shi W, Liu S, et al. Global fitting for high-accuracy multi-channel
  single-molecule localization. *Nat Commun* 13, 3133 (2022).
  [doi:10.1038/s41467-022-30719-4](https://doi.org/10.1038/s41467-022-30719-4)
  -- globLoc, the linked multi-channel fit this one is ported from, and the
  precision it gains.
* Li Y, Mund M, Hoess P, et al. Real-time 3D single-molecule localization
  using experimental point spread functions. *Nat Methods* 15, 367 (2018).
  [doi:10.1038/nmeth.4661](https://doi.org/10.1038/nmeth.4661) -- the bead
  calibration, the cubic spline model and the fitter.
* Bossi M, Fölling J, Belov VN, et al. Multicolor far-field fluorescence
  nanoscopy through isolated detection of distinct molecular species. *Nano
  Lett* 8, 2463 (2008).
  [doi:10.1021/nl801471d](https://doi.org/10.1021/nl801471d) -- ratiometric
  multicolour localization with two detection channels.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
