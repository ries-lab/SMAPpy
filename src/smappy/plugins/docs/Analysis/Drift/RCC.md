---
version: "1"
covers: [smappy.rcc.estimate_drift_rcc, smappy.rcc._pair_shifts, smappy.rcc._axial_shifts, smappy.rcc._solve]
---

## What it does

During an acquisition of many minutes the sample moves by tens to hundreds of
nanometres: the stage creeps, the objective warms up, the coverslip settles.
Every localization is then off by however far the sample had moved in its
frame, and the super-resolved image is smeared by the whole path.

RCC (*redundant cross-correlation*,
[Wang et al. 2014](https://doi.org/10.1364/OE.22.015982)) measures that path from the
localizations themselves -- no beads or fiducial markers are needed -- and
subtracts it.  It cuts the acquisition into time windows, makes an image of
each, and asks how far one image has to be shifted to match another.  The
answer, for every pair of windows, is the drift between them.

Use it on **fixed** samples.  It assumes the structure itself does not change,
only where it is, so anything that moves by itself (live cells, diffusing
molecules) is taken for drift.  It needs enough localizations per window for
each image to show structure: a few thousand in the selection at the very
least (below 5000 the plugin warns), and many more for a precise
curve.

It is the classic method, and fast -- seconds, where
[COMET](plugin:Analysis/Drift/COMET) takes minutes.  The two share no code
and almost no assumptions, so running both is a useful second opinion on a
drift curve.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.rcc import RCCSettings, estimate_drift_rcc, _pair_shifts, _solve
locs = simulate(n_frames=10000, seed=1, drift=True)
truth = np.array(locs.metadata["drift_truth"])
settings = RCCSettings(n_timepoints=10)
x, y = np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])
frame = np.asarray(locs["frame"])
edges = np.linspace(0, frame.max() + 1, settings.n_timepoints + 1)
window = np.clip(np.searchsorted(edges, frame, side="right") - 1,
                 0, settings.n_timepoints - 1)
```

**1. Time windows.**  The frames are split into a number of equal windows
(*time windows*, 20 by default).  Blinks are first grouped, so that a molecule
that is on for five frames counts once rather than five times (*group
blinks*).

**2. An image per window.**  The localizations of each window are binned into
a 2D histogram with small pixels (15 nm by default).

**3. Cross-correlation.**  For two images $A$ and $B$ the cross-correlation

$$C(\Delta) = \sum_{\mathbf{r}} A(\mathbf{r})\, B(\mathbf{r} - \Delta)$$

is large at the shift $\Delta$ that best lays one onto the other.  Its peak is
the drift between the two windows.  The correlation is computed with Fourier
transforms, which is what makes it fast, and the peak is looked for only
within *max drift* of zero.  Its position is refined below a pixel by fitting
a paraboloid to the few pixels around the maximum.

```figure The first and the last time window of a simulated acquisition with drift, and their cross-correlation: the peak sits where the last window has to be shifted to match the first -- the drift between them.
from smappy.rcc import _grid, _render
occupied = np.array([0, settings.n_timepoints - 1])
gx, grid_x = _grid(x, 30.0, 2048)
gy, grid_y = _grid(y, 30.0, 2048)
images = _render(gx, gy, window, occupied, (grid_x, grid_y))
spectra = np.fft.rfft2(images, s=(256, 256))
corr = np.fft.fftshift(np.fft.irfft2(spectra[0] * np.conj(spectra[1]), s=(256, 256)))
fig.set_size_inches(7.5, 2.6)
axes = fig.subplots(1, 3)
for ax, image, title in zip(axes, images, ("first window", "last window")):
    ax.imshow(image, cmap="magma", vmax=np.percentile(image, 99.7), origin="lower")
    ax.set_title(title, fontsize=9)
reach = 12
patch = corr[128 - reach:128 + reach + 1, 128 - reach:128 + reach + 1]
extent = np.array([-reach - 0.5, reach + 0.5]) * 30
axes[2].imshow(patch, cmap="viridis", origin="lower", extent=(*extent, *extent))
axes[2].axhline(0, color="w", lw=0.5); axes[2].axvline(0, color="w", lw=0.5)
axes[2].set_title("cross-correlation", fontsize=9)
axes[2].set_xlabel("shift x (nm)", fontsize=8)
for ax in axes[:2]:
    ax.set_xticks([]); ax.set_yticks([])
axes[2].tick_params(labelsize=7)
```

**4. Every pair, not just neighbours.**  Correlating only consecutive
windows and adding up the shifts would add up their errors too, and one bad
correlation would offset everything after it.  Instead every window is
correlated with every other: with $T$ windows that is $T(T-1)/2$
measurements $s_{kl}$ of the difference $d_k - d_l$ between the drift of two
windows, for only $T$ unknowns.  This redundancy is the *R* in RCC.

```figure Left: the measured shift in x of every window against every other -- one number per pair, $T(T-1)/2$ of them.  Right: the drift per window that best explains all of them (dots), against the drift that was simulated (line).
lateral = _pair_shifts(x, y, window, np.arange(settings.n_timepoints),
                       settings.pixelsize_nm, settings, False, "x-y")
fig.set_size_inches(7.5, 2.8)
left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.4]})
shown = left.imshow(lateral[0], cmap="RdBu_r", vmin=-120, vmax=120)
left.set_xlabel("window l", fontsize=8); left.set_ylabel("window k", fontsize=8)
left.set_title("$s_{kl}$, x (nm)", fontsize=9)
fig.colorbar(shown, ax=left, shrink=0.8).ax.tick_params(labelsize=7)
centres = 0.5 * (edges[:-1] + edges[1:])
dx = _solve(lateral[0], settings.n_timepoints)
true_x = truth[:, 0] - truth[:, 0].mean()
right.plot(np.arange(len(truth)), true_x, color="0.6", label="simulated")
right.plot(centres, dx - dx.mean() + (true_x[centres.astype(int)].mean()),
           "o", color="#d62728", label="RCC, per window")
right.set_xlabel("frame", fontsize=8); right.set_ylabel("drift x (nm)", fontsize=8)
right.legend(fontsize=7, frameon=False)
left.tick_params(labelsize=7); right.tick_params(labelsize=7)
```

**5. The least-squares solution.**  The drift per window is the set of
$d_k$ that fits all pair measurements best.  A pair whose correlation peak
landed in the wrong place -- a window with little structure, a coincidence
of shapes -- is then outvoted by the others instead of believed.  How that is
done is under *In detail*.

**6. z, if the table has it.**  The axial drift is measured the same way, but
on z histograms rather than images: the lateral drift is taken out first, the
field of view is cut into small tiles (200 nm), each tile's z histogram is
correlated between the two windows, and the correlations of all tiles are
added up before the peak is found.  Tiles matter: a z histogram of the whole
field mixes hundreds of structures into a featureless slab with nothing for
the correlation to lock onto.

**7. Per frame.**  The drift per window is interpolated to every frame with a
smooth curve that does not overshoot between the windows (PCHIP), and
subtracted from every localization -- of the whole table, not only of the
selection it was measured on.

```figure The drift estimated from the simulation (coloured) against the drift that was put in (grey).  Only differences between windows are measured, so the curves are compared after removing their mean.
drift = estimate_drift_rcc(locs, settings)
fig.set_size_inches(6.5, 2.8)
ax = fig.subplots()
frames = np.arange(len(truth))
for axis, (name, color) in enumerate(zip("xyz", ("#1f77b4", "#ff7f0e", "#2ca02c"))):
    true = truth[:, axis] - truth[:, axis].mean()
    found = drift.drift[:len(truth), axis] - drift.drift[:len(truth), axis].mean()
    ax.plot(frames, true, color="0.7", lw=2.5)
    ax.plot(frames, found, color=color, lw=1.2, label=name)
ax.set_xlabel("frame"); ax.set_ylabel("drift (nm)")
ax.legend(frameon=False, fontsize=8)
```

## In detail

**The pair equations.**  With $s_{kl}$ the measured shift of window $k$
against window $l$, the drifts $d_1, \ldots, d_T$ solve, in the least-squares
sense,

$$d_k - d_l = s_{kl}\quad (k < l), \qquad \sum_k d_k = 0 .$$

Only differences are measured, so the drift is fixed to average zero; the
absolute position of the sample over the acquisition is not known and is not
needed.  x, y and z are solved separately.

**Robust weights.**  The system is solved five times, each time weighting a
pair by how well the previous solution explained it.  With residuals
$r_{kl} = d_k - d_l - s_{kl}$ and their robust spread
$\hat s = 1.4826\,\mathrm{median}\,|r - \mathrm{median}\, r|$, a pair's row
of the system is multiplied by the Cauchy weight

$$w_{kl} = \frac{1}{1 + \left(r_{kl} / 3\hat s\right)^2} ,$$

so a pair three robust standard deviations off counts half, and one far off
counts almost nothing.  The equations are linear in the unknowns, so each
pass is a plain linear least-squares solve.

**The peak.**  The correlation is smoothed with a $5\times5$ box before the
maximum pixel is picked (the largest pixel of a noisy correlation can sit one
off the ridge), then a quadratic surface
$c_0 + c_1u + c_2v + c_3u^2 + c_4uv + c_5v^2$ is fitted to the unsmoothed
correlation over $(2h+1)^2$ pixels around it, $h$ the *peak fit half-width*,
and its stationary point is the sub-pixel shift.  If the fit is not a maximum
or lands outside the patch, the pixel is kept.

**The images.**  The images are padded before the Fourier transform so that
the circular correlation does not wrap onto itself within the search range.
A field of view wider than *max image size* pixels is folded back onto
itself: the structure repeats, but a shift of the whole is still a shift, and
the transform stays bounded.

**z.**  Each tile's z histogram (bins of *z bin*) has its mean subtracted, so
that the correlation follows structure rather than the number of
localizations.  The sample at zero shift is left out of the peak search and
the fit (*skip zero shift*): anything that sits at the same z in both windows
regardless of drift -- the same molecule on in both, or localizations piled
up at one z by the fitter -- would pull the answer towards no drift.  The
axial peak is searched within *max axial drift*, which is kept smaller than
the lateral range because a z profile is much less structured than an image
and a wide range lets the maximum wander.

**Compared with the paper.**  The method is that of
[Wang et al. 2014](https://doi.org/10.1364/OE.22.015982): images of time
windows, every window correlated with every other, and the drift solved from
the redundant set of shifts.  The main changes are SMAP's and this code's
(*Differences from SMAP*): blinks are grouped before the images are made, a
pair that disagrees with the rest is weighted down by the Cauchy weight above,
and z comes from the tiled z histograms.

**What limits the precision.**  The noise of the curve falls with more
localizations per window, and rises with more windows (fewer localizations
in each); more windows follow faster drift.  Twenty windows is a good start
for an acquisition of 10 000 to 50 000 frames.  If the curve looks noisy,
use fewer windows; if it looks like it cuts corners, use more, or COMET.

## Parameters

The ones shown without *more* are the ones worth looking at; the defaults of
the rest rarely need changing.

### n_timepoints
Twenty is a good start for 10 000 to 50 000 frames.  A noisy curve wants
fewer; a curve that cuts the corners of the real drift wants more.

### pixelsize_nm
Much smaller makes the images sparse and the correlation noisy; much larger
blurs the structure the correlation locks onto.

### max_drift_nm
Set it above the total drift over the whole acquisition.  A larger range
costs little, but gives a spurious peak more room.

### group
It removes the correlation of a molecule with itself in consecutive frames,
and makes the estimate faster.

### axial_max_drift_nm
Smaller than the lateral range, because the axial correlation is broad and a
wide range lets its maximum wander.

### exclude_zero_lag
See *In detail*.

## Output

* **The table**, with the drift subtracted from `x_nm`, `y_nm` and `z_nm` (or
  the pixel columns) of every localization.  The correction is recorded in the
  file's history and can be undone.
* **The drift curve**, x, y and z against frame (the *Plot* button).  It is
  saved with the file and drawn again when it is reopened.
* **The text** gives the range of the drift in x, y and z.

A curve that is smooth and plausible (a few hundred nanometres at most, slow
changes) is a good sign.  A curve that jumps between windows means the
correlations had too little to work with: fewer windows, more
localizations, or a larger *pixel size*.

## Differences from SMAP

Based on SMAP's `Drift/driftcorrectionXYZ`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), its
`finddriftfeature` in particular.

* The axial pass tiles the field in x *and* y and removes the lateral drift
  first; with slices spanning the whole field in y the axial shifts came out
  uncorrelated with COMET's, against a correlation of 0.96 with square tiles.
* Interpolation to frames is PCHIP, which does not overshoot between windows.

## References

* Wang Y, Schnitzbauer J, Hu Z, et al. Localization events-based sample drift
  correction for localization microscopy with redundant cross-correlation
  algorithm. *Opt Express* 22, 15982 (2014).
  [doi:10.1364/OE.22.015982](https://doi.org/10.1364/OE.22.015982)
* Mlodzianoski MJ, Schreiner JM, Callahan SP, et al. Sample drift correction
  in 3D fluorescence photoactivation localization microscopy. *Opt Express*
  19, 15009 (2011).
  [doi:10.1364/OE.19.015009](https://doi.org/10.1364/OE.19.015009)
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
