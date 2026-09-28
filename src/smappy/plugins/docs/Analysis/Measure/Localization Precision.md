---
version: "3"
covers: [smappy.plugins.precision.measure, smappy.plugins.precision.displacement_pairs, smappy.plugins.precision.fit_radial, smappy.plugins.precision.fit_axis, smappy.plugins.precision.radial_density, smappy.plugins.precision.axis_density, smappy.plugins.precision.drift_free_sigma, smappy.plugins.precision.sigma_at_photons, smappy.plugins.precision.crlb_statistics, smappy.plugins.precision.summary, smappy.plugins.precision.draw_radial, smappy.plugins.precision.draw_gaps, smappy.plugins.precision.draw_crlb, smappy.plugins.precision.draw_frc, smappy.frc.frc_resolution, smappy.frc.blur_envelope, smappy.frc.envelope_resolution]
---

## What it does

How well was a molecule placed?  The fitter gives one answer for every
localization: its precision, `xy_err_nm`, the Cramer-Rao bound (CRLB)
computed from the photons, the background and the PSF.  That number is what
an ideal fit of an ideal spot could reach.  It knows nothing of what happens
after the fit: drift, vibration, a PSF that is not quite the model, a
wrong camera calibration.

This plugin measures the precision **on the data itself** and puts it beside
what the fitter claimed.  It reports three kinds of number, and they are
meant to disagree -- the way they disagree is the diagnosis:

* **The CRLB histogram** -- what the fitter believes.  The distribution of the
  precision column, with its typical value $\sigma_c$.
* **The pairwise displacement** (NeNA,
  [Endesfelder et al. 2014](https://doi.org/10.1007/s00418-014-1192-3)) --
  what the experiment did.  A
  fluorophore is usually on for more than one frame, so the same molecule is
  localized again in the next frame, a few nanometres away.  How far apart
  those two localizations are depends only on how precisely each was placed.
  Nothing about photons or PSFs is assumed.
* **FRC**, Fourier ring correlation
  ([Nieuwenhuizen et al. 2013](https://doi.org/10.1038/nmeth.2448)) -- what
  the picture resolves.  The acquisition is split in two
  halves, both are rendered, and the plugin finds up to which level of detail
  the two images still agree.  This folds in the labelling density, the drift
  and the number of localizations as well as the precision.

By default it measures twice: the localizations as they are shown (the
layer's filter and the ROI) and all of them, side by side (*localizations*).
If nothing is filtered, it measures once.

**What it needs.**  A `frame` column and positions for the pairwise
displacement; a precision column (`xy_err_nm`, and `z_err_nm` for 3D) for the
CRLB histogram and for comparing each pair with its bound; `photons` for the
photon law (below).  The displacement fit wants at least 200 pairs in the
search radius.  FRC wants at least 1000 localizations.

It always reads the **ungrouped** table, even when the layer shows the
grouped one: grouping merges the repeated localizations of a blink into one,
and those repeats are exactly what the pairwise displacement measures.

Photons and on-times are described by
[Localization Statistics](plugin:Analysis/Measure/Localization Statistics),
which fits the same model to the precision histogram as this plugin does.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.plugins.precision import (Measurement, measure, displacement_pairs,
    fit_radial, drift_free_sigma, sigma_at_photons, REACH_Z, median_precision,
    draw_radial, draw_crlb, draw_gaps, draw_frc)
sim = simulate(n_frames=3000, seed=1)
# the simulation keeps spots down to 10 photons; a cut at 200 stands in for a
# fitter's detection threshold.  Its xy_err_nm is honest: every position was
# displaced by a random error of exactly that size.
locs = sim[np.asarray(sim["photons"]) > 200]
found = measure(locs, name="simulated", max_gap=1, frc=True,
                frc_repeats=3)
x, y = np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])
z, frame = np.asarray(locs["z_nm"]), np.asarray(locs["frame"])
reach = found.radial.half_width
# the truth: the pairs the simulation knows are one molecule (a z far apart for
# every other emitter keeps them out of the same search cylinder)
emitter = np.asarray(locs["emitter"], dtype=float)
same = displacement_pairs(x, y, frame, reach, z=emitter * 1e6 + z,
                          dz_max=REACH_Z * median_precision(locs, ("z_err_nm",)))[1]
```

**1. Pairs.**  For every localization, every localization in the next frame
within the *search radius* is paired with it -- not only the nearest one.
By default the radius is 6 times the median precision.  In 3D the search is a
cylinder: the disc laterally, and within the *axial search* in z (6 times the
median axial precision by default).

**2. Signal and background.**  Most pairs are one molecule seen twice.  Some
are two different molecules that happen to be close.  Those are spread evenly
over the search disc, so their share of the distance histogram rises in a
straight line with the distance $d$.  The pairs of one molecule follow a
known curve instead: if each localization is off by a random error of size
$\sigma$ in x and in y, the distance between two of them has the density

$$p_{\mathrm{same}}(d) = \frac{d}{2\sigma^2}\, e^{-d^2/4\sigma^2} ,$$

which peaks at $d = \sqrt{2}\,\sigma$.  The plugin fits the sum of the two
curves, the fraction $f$ of same-molecule pairs and $\sigma$ together.  The
fit is maximum likelihood on the distances themselves, so the histogram's
*bins* change nothing but the picture.

```figure The distances between localizations in consecutive frames of a simulated dataset (blue steps), and the fit: the whole model (solid), its background of different molecules (dashed), the peak at $\sqrt{2}\,\sigma$ (dotted), and the photon law (dash-dot, step 3).  Grey: the pairs the simulation knows are one molecule -- nearly all of them.  The single $\sigma$ misses the long tail of dim pairs and hands it to the background; the photon law, with a width per pair, follows it.
fig.set_size_inches(6.0, 3.0)
ax = fig.subplots()
ax.hist(same.d, bins=80, range=(0, reach), color="0.85", label="truth: same molecule")
draw_radial(ax, [found])
```

**3. The pairs are not a typical localization, so the photons are used too.**
A single $\sigma$ describes the pairs only if every localization is equally
good.  In real data some are bright and some are dim, and the pairs are a
particular subset of them: the frames of a blink that is on in two frames in
a row.  So the plugin fits the same distances twice more, each time letting
every pair have its own expected width:

* against the **photons** of the two localizations.  A precision falls as
  $\sqrt{A/N}$ with the photon count $N$, and one number $A$ is fitted for all
  pairs.  The law can then say what a localization of *any* brightness is
  worth -- in particular the median one of the table, which the text reports.
  It needs no camera calibration: a wrong gain changes $N$ and $A$ by opposite
  factors and leaves $\sqrt{A/N}$ where it was.
* against the **CRLB** of the two localizations.  The one number fitted,
  $\kappa$, says how far the data misses the fitter's own bound.
  $\kappa = 1$: the fitter reaches its bound and the photon calibration is
  consistent with the data.  $\kappa > 1$: something is worse than the model
  says -- a wrong gain or offset, a PSF that does not match, residual motion.
  $\kappa$ does not say which.

```figure What the photon law says each localization is worth (red), against the true precision of each localization of the simulation (grey dots).  The plain NeNA $\sigma$ (blue) is one number for all of them.  Diamond: the law at the median photon count, which is what the text reports.
law = found.photon_law
photons = np.asarray(locs["photons"], dtype=float)
true = np.asarray(locs["xy_err_nm"], dtype=float)
pick = np.random.default_rng(0).choice(len(locs), 3000, replace=False)
fig.set_size_inches(6.0, 3.0)
ax = fig.subplots()
ax.scatter(photons[pick], true[pick], s=2, color="0.7", label="simulated precision")
n = np.geomspace(200, photons.max(), 200)
ax.plot(n, [sigma_at_photons(law, v) for v in n], color="#d62728", label="$\\sqrt{A/N}$, fitted to the pairs")
ax.axhline(found.radial.sigma, color="#1f77b4", ls="--", label=f"NeNA $\\sigma$ = {found.radial.sigma:.2f} nm")
median = float(np.median(photons))
ax.plot([median], [sigma_at_photons(law, median)], "D", color="#d62728",
        label=f"median photons: {sigma_at_photons(law, median):.2f} nm "
              f"(true median {np.median(true):.2f} nm)")
ax.set_xscale("log"); ax.set_ylim(0, 12)
ax.set_xlabel("photons"); ax.set_ylabel("precision (nm)")
ax.legend(fontsize=7, frameon=False)
```

**4. Per axis.**  The same mixture is fitted to the signed displacements in
x, in y and, for 3D data, in z, each on its own.  This is how the axial
precision is measured: the z fit needs no model of how the PSF changes with
depth.  The mean of each displacement is printed as well: it is the drift
over one frame.

**5. Frame gap.**  The whole fit is repeated for localizations 2, 3, ... up to
*frame gaps to* frames apart.  If nothing moves, a molecule is placed just as
well two frames later and $\sigma$ stays flat.  If the sample drifts or
vibrates, the displacement grows with the time between the two
localizations, and so does $\sigma$.  A straight line through $\sigma^2$
against the gap, extrapolated to gap zero, gives the precision with the
motion taken out, and its slope gives how far the sample moves per frame.
After drift correction, a rising curve says the correction missed the fast
part of the motion -- which nothing else in the program shows.

```figure $\sigma$ against the frame gap for the simulated data as it is (blue) and with a random walk of 2 nm per frame per axis added to every position (red).  The star is the extrapolation to gap zero.  Grey: what the walk that was put in should give.
walk = np.cumsum(np.random.default_rng(3).normal(0, 2.0, (frame.max() + 1, 2)), axis=0)
series = []
for name, (xx, yy) in (("still", (x, y)),
                       ("moving", (x + walk[frame, 0], y + walk[frame, 1]))):
    pairs = displacement_pairs(xx, yy, frame, reach, gaps=(1, 2, 3))
    fits = {g: fit_radial(p.d, p.d_max, sigma0=found.radial.sigma)
            for g, p in pairs.items()}
    line = drift_free_sigma(list(fits), [f.sigma for f in fits.values()],
                            [f.sigma_error for f in fits.values()])
    series.append(Measurement(name=name, n_locs=len(locs), by_gap=fits,
                              gap_line=line))
fig.set_size_inches(6.0, 3.0)
ax = fig.subplots()
g = np.linspace(0, 3, 50)
ax.plot(g, np.sqrt(series[0].by_gap[1].sigma ** 2 + 2.0 ** 2 * g / 2), color="0.6",
        lw=3, alpha=0.5, label="truth: 2 nm per frame")
draw_gaps(ax, series)
```

**6. The CRLB histogram.**  The precision column is histogrammed and the
distribution that exponentially distributed photons imply is fitted over it,
exactly as in [Localization Statistics](plugin:Analysis/Measure/Localization Statistics):
the single number $\sigma_c$ is the precision at the mean photon count.  The
pairwise $\sigma$ is drawn over it as a dashed line, which is the comparison
the plugin exists for.

```figure The CRLB histograms of the simulation, lateral and axial, with the fitted distribution (solid) and the $\sigma$ the pairs measured (dashed).
fig.set_size_inches(6.0, 4.4)
draw_crlb(fig, [found])
fig.subplots_adjust(hspace=0.6)
```

**7. FRC.**  The frames are cut into *FRC blocks* -- equal stretches of the
acquisition -- and the blocks are dealt at random into two halves, so that
all the frames of one blink land in the same half.  Both halves are rendered,
and their Fourier transforms are compared ring by ring: the correlation is 1
at coarse detail, where the two images agree, and falls to 0 at fine detail,
where each shows only its own noise.  The resolution is where the smoothed
curve first falls below 1/7, the threshold of Nieuwenhuizen et al.  This is
done *FRC splits* times with different deals; the curves are averaged, and
the spread of the resolutions is the error bar.

Drawn beside it, dashed, is what the blur of the measured precision alone
leaves of the correlation.  It is not a fit.

```figure The FRC of the simulation (the light line raw, the dark one smoothed), the 1/7 threshold, and the blur that the pairwise precision alone implies (dashed).
fig.set_size_inches(6.0, 3.0)
draw_frc(fig.subplots(), [found])
```

## In detail

**The pair model.**  Two localizations of one molecule are at the same true
place, so their displacement is the difference of two independent errors: a
variance $v = 2\sigma^2$ per axis.  The lateral distance $d$ of a pair follows
Churchman's law at zero true distance, truncated at the search radius
$d_{\max}$, mixed with the uniform background:

$$p(d) = f\, \frac{d}{v}\, \frac{e^{-d^2/2v}}{1 - e^{-d_{\max}^2/2v}} + (1 - f)\, \frac{2d}{d_{\max}^2} .$$

Per axis it is a truncated Gaussian of variance $v$ on a background that is
the chord of the search disc, $2\sqrt{d_{\max}^2 - \Delta^2} / \pi d_{\max}^2$,
for x and y, and flat for z (the search is a slab of $\pm$ the *axial search*
there).  Displacements are *later minus earlier*, so their mean is the drift
over the gap; the fit ignores the mean.

**Why all pairs, not the nearest neighbour.**  With every pair inside the
radius, the pairs of different molecules are uniform in the disc and their
density is exactly $2d/d_{\max}^2$.  Keeping only the nearest neighbour makes
that background depend on the density of the sample: a molecule with a close
neighbour hides its true partner.

**The scaled fits.**  For the photon law each pair has its own variance
$v_k = A\,(1/N_i + 1/N_j)$, for the CRLB fit
$v_k = \kappa^2 (\sigma_i^2 + \sigma_j^2)$, with $N$ the photons and $\sigma$
the precision column of the two members.  The density is the same mixture,
one Rayleigh per pair.  The photon law assumes shot noise dominates; with a
heavy background the dim end wants an extra term in $1/N^2$, and that is
where a deviation from the law shows (the red curve above, at the dim
end).

**The fit.**  Maximum likelihood, by a Nelder-Mead simplex in
$(\log\sigma, \mathrm{logit}\, f)$, so neither can leave its range.  It starts
from three signal fractions (0.5, 0.15 and 0.85) and keeps the best, then
restarts the simplex once at its own optimum, because on a shallow valley a
simplex stops early and would remember where it started.  The error is from
the curvature of the log-likelihood in both parameters together; if that
comes out singular it falls back to $\sigma/\sqrt{2 n f}$ for $n$ pairs.
Fewer than 200 pairs are not fitted, and a fit in which fewer than 100 pairs
are signal says so.

**The frame gap line.**  A random walk of step $w$ per frame and axis adds
$w^2 g$ to the displacement variance over $g$ frames, so

$$\sigma(g)^2 = \sigma_0^2 + \frac{w^2}{2}\, g .$$

The line is fitted to $\sigma(g)^2$ by weighted least squares, each gap
weighted by its fitted error; $\sigma_0$ is its intercept and $w$ is printed
as the motion "per root frame".  The slope counts as motion only when it is
more than twice its own standard error (from the same weighted fit) above
zero; otherwise the text says there is no motion beyond the scatter of the
gaps and $w$ is 0.  Five points that scatter by their errors always have some
slope, and on still simulated data it came out as half a nanometre per root
frame before this test.

**The CRLB histogram fit** is Localization Statistics' least-squares fit of
$p(\sigma) = (2a/\sigma^3)\, e^{-a/\sigma^2}$, $a = \sigma_c^2$, to the
histogram from zero to its 99.5th percentile.  If $\sigma_c$ lands outside 0.3
to 3 times the median, the photons are not exponential and the text says to
read the median instead.  When the selection is measured and the layer's
filter cuts the precision column, the histogram is fitted between the cuts
only: a lower cut would otherwise leave empty bins the model reads as a
distribution moved up (11.3 nm for a true 8.9 nm, with a cut at 7 nm), and
the fit of the part that survived recovers the whole.

**FRC.**  With the half-images $F_1$ and $F_2$, the correlation over a ring of
spatial frequency $q$ is

$$\mathrm{FRC}(q) = \frac{\sum \mathrm{Re}(F_1 F_2^*)}{\sqrt{\sum |F_1|^2 \sum |F_2|^2}} .$$

The images are tapered at the edges first (a Tukey window over the outer
eighth), and the curve is smoothed with a Savitzky-Golay filter before the
crossing is read.  The picture is transformed in tiles of 512 pixels whose
sums are added, so the pixel can be small whatever the size of the field;
tiles with fewer than 200 localizations are skipped.  With the *FRC pixel* at
0 the resolution is first measured on a coarse grid (the field over 1024
pixels) and then again with a pixel a fifth of what that found.  A coarse
pixel cannot see a resolution finer than about two of it, so when the coarse
curve never falls through 1/7 the coarse pass is repeated at a quarter of the
pixel, down to 1 nm.  The blur
envelope is $e^{-4\pi^2\sigma^2 q^2}$, and the text reports where it alone
crosses 1/7, at $2\pi\sigma/\sqrt{\ln 7} = 4.50\,\sigma$.  That is not a limit
on the FRC: a structure localized many times per molecule resolves better
than it, a sparsely labelled one worse.

**FRC per axis** (3D).  Instead of rings of a 2D picture, planes of the 3D
transform perpendicular to one axis (a Fourier plane correlation) give a
resolution along x, y and z separately, each plane summed only over the band
of frequencies the other two axes resolve.  The volume is far sparser than
the projection, so the lateral numbers come out worse than the ring FRC,
often by a factor of two; they are the ones to quote for a 3D measurement.

**A ROI or a slab** costs only the pairs that straddle its edge, which for a
search radius of tens of nanometres is nothing -- except in z, where a slab
(a filter on `z_nm`, or the 3D view's slab while plugins use it) thinner than
four axial precisions loses the partners that fell outside and $\sigma_z$
reads low; the text warns.  The pairs within one frame
gap are searched with one KD-tree per frame.

**Compared with the papers.**  NeNA
([Endesfelder et al. 2014](https://doi.org/10.1007/s00418-014-1192-3))
estimates the precision from nearest neighbours.  Here:

* every pair within the search radius is taken, not only the nearest one,
  so that the background of different molecules has the known shape
  $2d/d_{\max}^2$ (above);
* the photon law and the CRLB-scaled fit are added to the single-$\sigma$
  (NeNA) fit, because the pairs are the frames in which a blink began or
  ended and are not a typical localization (step 3).

FRC follows [Nieuwenhuizen et al. 2013](https://doi.org/10.1038/nmeth.2448),
with the fixed threshold of 1/7, and departs in three places:

* The paper divides the curve's numerator by the average blur of the
  localization uncertainties.  Here nothing is divided: the blur envelope is
  drawn beside the curve, because dividing amplifies the noise at exactly the
  frequencies that are read and uses the precision that is being checked.
* The halves are dealt in blocks of frames, so a blink stays in one half
  (all but the few that straddle a block boundary) and cannot correlate with
  itself.
* Each Fourier plane is summed only over the band of frequencies the other
  two axes resolve: over the whole plane, the axial resolution of one
  dataset read 89 nm on a 20 nm lateral voxel and 417 nm on a 4 nm one.

## Parameters

### source
*as plotted* answers "how well is what I am looking at placed", *all* "how
good is this sample".  A filter on precision or photons does not bias the
pairwise fit, it selects: both partners of a pair must survive it, and the
first and last frames of a blink are the dim ones, so a hard cut removes
true pairs preferentially.  The fitted signal fraction shows it; $\kappa$ survives a cut better than the plain $\sigma$.

### pairwise
It is the only one of the three methods that sees drift, vibration and a
wrong calibration, and the only one that needs molecules on in consecutive
frames.

### crlb
Needs a precision column; without one the histogram is left out.

### frc
The slowest part of a run on a large field; untick it when only the
precision is wanted.

### max_gap
5 is enough to see a trend.  At long gaps fewer molecules are still on, so
the pairs thin out and the error bars grow.

### reach_nm
The peak sits at $\sqrt{2}$ times the precision, and the background must be
visible beyond the tail for the fit to tell them apart.  Much wider costs time
and adds background; narrower than about three times the precision cuts into
the signal.  In a very dense sample a smaller radius keeps the background
share down.

### reach_z_nm
Kept narrower than four axial precisions, it cuts off the tail of the z
displacements and $\sigma_z$ reads low; the text warns.

### frc_axes
The volume is transformed tile by tile, each tile the full depth of the data,
so a whole field of view is many tiles and a long run.

### frc_blocks
More blocks mix the two halves more evenly over the acquisition, which
matters when the sample changes over time; fewer keep more blinks whole.

### frc_repeats
At least 3 are needed for the spread to be the error bar; with fewer, the
error comes from the ring statistics of the curve alone.

### frc_pixel_nm
Leave it at 0 unless two measurements must be compared at the same pixel.  A
pixel coarser than about a fifth of the resolution reads the resolution too
large; one much finer only costs time.

### max_frame_pairs
Only a speed limit: the default covers most acquisitions whole.

## Output

The **text** has one block per set of localizations, a line per number:

* *CRLB*: $\sigma_c$ and the median of the precision column, lateral and
  axial.
* *pairwise*: $\sigma$ at gap 1, the number of pairs and the percentage in
  its peak (the signal fraction $f$, see below); then $\sigma_x$, $\sigma_y$, $\sigma_z$ and
  the mean shift over one frame.
* *law*: $\sqrt{A}$ and the percentage of pairs that are one molecule by
  this fit, and what it gives for the median paired localization and
  for the median localization of the table.
* *kappa*, and the bound the pairs claimed -- the root mean square of their
  precisions, which is larger than the median because dim localizations weigh
  more.
* *FRC*: the resolution and its error, and where the blur of the pairwise
  $\sigma$ alone crosses 1/7; *per axis*: the resolutions and the
  axial-to-lateral ratio.
* *gap -> 0*: $\sigma_0$ and the motion per root frame, or that there is no
  motion beyond the scatter of the gaps.
* Notes: the search radius used, too few pairs, a cut-off axial search or a
  thin slab.

A table in pixels (`x_pix`, `xy_err_pix`) is measured and reported in pixels.

Besides the text:

* **Figures**: the distance histogram with the fit (the main one), one panel
  per axis, $\sigma$ against the frame gap (when *frame gaps to* is above 1),
  FRC, FRC per axis, and the CRLB histograms.
* The numbers (not the pairs) are kept with the result.

**Reading them.**  Pairwise $\sigma$ close to the CRLB median and $\kappa$
near 1: the data are as good as the fitter says.  $\kappa$ well above 1: the
fitter's precision is optimistic -- check the camera gain and offset, the PSF
model, and the frame-gap curve for motion.  A flat gap curve means nothing
moved; a rising one, residual drift or vibration.  An FRC curve that falls
well before the dashed blur envelope means the picture is limited by
labelling density, drift or too few localizations, and better precision would
not help; if the two fall together, the precision is what limits it.

The percentage in the *pairwise* line is a property of the single-$\sigma$
model, not a count.  When the precisions vary, as they always do, the one
Rayleigh describes the core of the distribution and hands part of the broad
tail of dim pairs to the background: in the simulation above all but a few
tenths of a percent of the pairs are one molecule, and about 85% are in the
single $\sigma$'s peak.  The photon law, which gives every pair its own
width, finds a fraction close to one, and that is the one the *law* line
reports.

## Differences from SMAP

The FRC is based on SMAP's `Analyze/measure/FRCresolution`, and the CRLB
histogram on its `Analyze/measure/Locstatistics`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)); the pairwise
displacement (NeNA) and the comparison with the CRLB are new.

* **The CRLB histogram.**  SMAP reads the maximum, median and rising edge off
  the histogram.  Here it is the fit of
  [Localization Statistics](plugin:Analysis/Measure/Localization Statistics),
  with $\sigma_c$.
* **FRC halves.**  SMAP splits the table into blocks of equal numbers of
  localizations.  Here the blocks are equal stretches of *frames*, so a blink
  is never split between the halves.
* **No clipping.**  SMAP clips the brightest pixels at the 0.9999 quantile;
  here nothing is clipped, because clipping changes the spectrum.

## References

* Endesfelder U, Malkusch S, Fricke F, Heilemann M. A simple method to
  estimate the average localization precision of a single-molecule
  localization microscopy experiment. *Histochem Cell Biol* 141, 629 (2014).
  [doi:10.1007/s00418-014-1192-3](https://doi.org/10.1007/s00418-014-1192-3)
  -- NeNA.
* Churchman LS, Ökten Z, Rock RS, Dawson JF, Spudich JA. Single molecule
  high-resolution colocalization of Cy3 and Cy5 attached to macromolecules
  measures intramolecular distances through time. *PNAS* 102, 1419 (2005).
  [doi:10.1073/pnas.0409487102](https://doi.org/10.1073/pnas.0409487102)
  -- the distribution of the distance between two localizations.
* Nieuwenhuizen RPJ, Lidke KA, Bates M, et al. Measuring image resolution in
  optical nanoscopy. *Nat Methods* 10, 557 (2013).
  [doi:10.1038/nmeth.2448](https://doi.org/10.1038/nmeth.2448) -- FRC, the
  1/7 threshold.
* Mortensen KI, Churchman LS, Spudich JA, Flyvbjerg H. Optimized
  localization analysis for single-molecule tracking and super-resolution
  microscopy. *Nat Methods* 7, 377 (2010).
  [doi:10.1038/nmeth.1447](https://doi.org/10.1038/nmeth.1447) -- the
  precision a fit can reach, and its $1/\sqrt{N}$ scaling.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
