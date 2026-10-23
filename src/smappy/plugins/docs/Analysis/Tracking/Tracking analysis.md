---
version: "1"
covers: [smappy.tracking.analysis.gather, smappy.tracking.analysis.steps_of, smappy.tracking.analysis.lag_sums, smappy.tracking.analysis.msd_covariance, smappy.tracking.analysis.lag_errors, smappy.tracking.analysis.optimal_lags, smappy.tracking.analysis.fit_msd, smappy.tracking.analysis.msd_analysis, smappy.tracking.analysis.jumps, smappy.tracking.analysis.fit_jumps, smappy.tracking.analysis.jump_density, smappy.plugins.tracking_analysis.TrackingAnalysis.run]
---

## What it does

Measures how [tracks](plugin:Panels/Glossary) move.  It needs tracks, an
id column that says which localizations belong to one particle -- the
*track id*: on *auto*, `track_id` from
[Link tracks](plugin:Analysis/Tracking/Link tracks), else MINFLUX's `tid`,
else a simulation's `track_id_true`.  Two analyses, each ticked on or off:

* **MSD** ([mean squared displacement](plugin:Panels/Glossary)): how far a
  particle has moved, squared and averaged, after 1, 2, 3, ... frames.  For
  free diffusion it grows in a straight line whose slope is the
  [diffusion coefficient](plugin:Panels/Glossary) *D*.  It is fitted for
  every track and for all tracks pooled.  The number of points fitted is
  chosen per track as Michalet showed it must be
  ([Michalet 2010](https://doi.org/10.1103/PhysRevE.82.041914)): few when
  the localization error is small compared with a frame's motion, more
  when it is large.  It also gives the
  [anomalous exponent](plugin:Panels/Glossary) α, which tells free
  diffusion (1) from confined (below 1) and directed motion (above).
* **Jump distances**: the distances particles move between frames, as a
  mixture of one to three [diffusive states](plugin:Panels/Glossary), each
  with its *D* and the fraction of jumps in it -- a bound and a free
  population, say.

Both take the [localization precision](plugin:Panels/Glossary) into
account: a molecule that does not move still seems to, by its localization
error, and an analysis that ignores it finds every molecule mobile.

Time comes from the file's frame time (recorded by the camera, set in
*File > File properties*, or here under *frame time*); a table with its own clock (`time_s`, MINFLUX)
uses that.  It analyses the localizations the layer's filter and the ROI
keep.  Per region it runs too: in each ROI of the ROI manager, or in a
block over `track_id`, once per track.  With *write per track*, each
track's *D* and α are written onto its localizations
(`track_D_um2_per_s`, `track_alpha`), to colour or filter the tracks by.

## How it works

```figure-setup
from smappy.simulate import simulate_tracks
from smappy.tracking.analysis import (gather, msd_analysis, lag_errors, jumps,
                                      fit_jumps, jump_density)
one = simulate_tracks(n_frames=1000, density_um2=0.02, d_um2_s=(0.1,),
                      track_frames=40.0, precision_nm=30.0, seed=1)
two = simulate_tracks(n_frames=1000, density_um2=0.04, d_um2_s=(0.01, 0.5),
                      fractions=(0.35, 0.65), track_frames=40.0,
                      precision_nm=20.0, seed=2)

def tracks(locs):
    return gather(locs["track_id_true"], locs["frame"], locs["x_nm"],
                  locs["y_nm"], err_xy=locs["xy_err_nm"], frame_time_s=0.01)
```

**1. Displacements.**  The tracks are sorted by id and time.  For every
lag of k frames, the squared distance between every two localizations of a
track exactly k frames apart is averaged: that track's MSD at lag k.
Pairs across a dark frame count at their true lag.  The MSD of all tracks
pools the pairs of every track.

**2. The offset.**  A particle's measured position is off by its
localization error, so even at a lag of zero frames the MSD would not be
zero: in two dimensions it starts at $4\sigma^2$, with $\sigma$ the
precision.  It is taken from the precisions of the localizations
(*offset*: from the precisions), or fitted with the slope.

**3. How many points.**  The first MSD points are averaged over many
pairs but carry the localization error; the later ones are averaged over
few.  Somewhere between is the number of points that gives the most
precise *D*.  It depends on how long the track is and on how large the
error is compared with a frame's motion, and it is computed for each track
(see *In detail*).

**4. D and α.**  A straight line through the offset fitted to those points
gives *D*; the slope of the MSD above the offset on a log-log plot gives α.

```figure Left: simulated tracks at D = 0.1 µm²/s with 30 nm precision; the MSD of all of them (dots), the points the fit used (blue) and the line through the offset. Right: how precise D is when fitted to the first p points, for a track of 50 positions, at three ratios of the localization error to a frame's motion: when the error is small one point is best, when it dominates many are, and fitting the wrong number can be several times worse.
fig.set_size_inches(9, 3.6)
left, right = fig.subplots(1, 2)
r = msd_analysis(tracks(one))
k = int(r.pooled.lags[0])
left.plot(r.lag_s[:15], r.msd_nm2[:15] / 1e6, "o", ms=3, color="0.6")
left.plot(r.lag_s[:k], r.msd_nm2[:k] / 1e6, "o", ms=4, color="C0")
line = np.linspace(0, r.lag_s[14], 50)
D = r.pooled.D_um2_per_s[0]
left.plot(line, r.pooled.offset_nm2[0] / 1e6 + 4 * D * line, color="C3",
          label=f"D = {D:.3f} µm²/s (truth 0.1)")
left.set_xlim(0, line[-1])
left.set_ylim(0, None)
left.set_xlabel("lag (s)")
left.set_ylabel("MSD (µm²)")
left.legend(fontsize=8)
for x, colour in ((0.1, "C0"), (10.0, "C1"), (100.0, "C3")):
    err = lag_errors(50, x)
    right.plot(np.arange(1, len(err) + 1), err, color=colour, label=f"x = {x:g}")
    best = int(np.nanargmin(err))
    right.plot(best + 1, err[best], "o", color=colour)
right.set_yscale("log")
right.set_xlabel("MSD points fitted")
right.set_ylabel("relative error of D (one axis)")
right.legend(fontsize=8)
```

**5. Jump distances.**  The distances between localizations of a track
exactly *lag* frames apart are collected.  Each diffusive state gives
jumps of a known shape -- wider for a larger *D*, and widened by the two
localizations' errors -- and the states' *D* and fractions are those that
make the observed jumps most likely.

```figure Two populations, 35% at 0.01 µm²/s and 65% at 0.5 µm²/s. Left: D per track, on a log axis -- two groups, broad because each track is short. Right: the jumps between consecutive frames, the two states fitted to them and their sum (black), which give back both coefficients and fractions.
fig.set_size_inches(9, 3.6)
left, right = fig.subplots(1, 2)
t = tracks(two)
r = msd_analysis(t)
d = r.per_track.D_um2_per_s
d = d[np.isfinite(d) & (d > 0)]
left.hist(d, bins=np.logspace(-3.5, 0.5, 40), color="0.6")
left.set_xscale("log")
for truth in (0.01, 0.5):
    left.axvline(truth, color="C3", lw=1)
left.set_xlabel("D per track (µm²/s)")
left.set_ylabel("tracks")
d2, err = jumps(t, 1)
fit = fit_jumps(d2, err, 2, 0.01, 2)
jump = np.sqrt(d2)
grid = np.linspace(0, np.quantile(jump, 0.995), 200)
right.hist(jump, bins=grid[::4], density=True, color="0.75")
parts = jump_density(grid, fit, float(np.median(err)), 2, 0.01)
for i, part in enumerate(parts):
    right.plot(grid, part, color=f"C{i}",
               label=f"{fit.D_um2_per_s[i]:.3f} µm²/s, {100 * fit.fraction[i]:.0f}%")
right.plot(grid, parts.sum(0), color="k", lw=1, label="both")
right.set_xlabel("jump (nm)")
right.set_yticks([])
right.legend(fontsize=8)
```

## In detail

**The MSD.**  For a track with positions $\mathbf{r}_i$ in frames $f_i$,

$$\mathrm{MSD}(k) = \frac{1}{N_k} \sum_{f_j - f_i = k} |\mathbf{r}_j - \mathbf{r}_i|^2 ,$$

over all $N_k$ pairs $k$ frames apart, overlapping -- the usual estimator.
Pooled, the sums and counts of all tracks are added.  Lags up to 50 frames
are computed, every one over the whole table at once, so 50,000 tracks of a
million localizations take a few seconds.

**The model.**  For free diffusion in $d$ dimensions, with frame time
$\Delta t$ and a localization error $\sigma_a$ per axis,

$$\mathrm{MSD}(k) = 2 d D k \Delta t + 2 \sum_a \sigma_a^2 ,$$

the offset $4\sigma^2$ in 2D.  *from the precisions* takes $\sigma^2$ as
the mean of the track's `xy_err_nm`² (and `z_err_nm`² in 3D).  It is the
static error; a particle that moves during the exposure is blurred, which
makes the true offset smaller (Michalet 2010).  *fitted* fits the
intercept too, which needs no precisions but leaves *D* less certain
([Michalet & Berglund 2012](https://doi.org/10.1103/PhysRevE.85.061916)).

**How many points.**  With the offset known, $D$ is the slope through it,
$\hat{D} = \sum_k k \, (\mathrm{MSD}(k) - o) / (2 d \Delta t \sum_k k^2)$
over $k = 1 \ldots p$; with it fitted, the slope of an ordinary straight
line.  Michalet (2010) showed that the error of $\hat{D}$ has a minimum
in $p$, set by the track's length $N$ and the reduced localization error

$$x = \frac{\sigma^2}{D \Delta t} ,$$

and gave an empirical formula for it.  That formula is not used here:
the optimum is computed from the model itself.  The displacements of a
Brownian track with Gaussian errors are jointly Gaussian, with covariances
$2 D \Delta t$ times the frames two displacements share plus $\sigma^2$
where their ends meet; the MSD points are means of their squares, and for
Gaussian variables $\mathrm{Cov}(a^2, b^2) = 2 \, \mathrm{Cov}(a, b)^2$, so
the full covariance of the MSD points follows exactly.  From it, the
variance of $\hat{D}$ for every $p$, and the $p$ with the least.  (Against
20,000 simulated tracks the covariance agrees to a few percent.)  Since
$x$ needs $D$, Michalet's iteration: $D$ from $p = N/10$, $x$ from $D$, $p$
from $x$, until $p$ repeats, at most 10 rounds.  A track with dark frames
is taken as the gap-free track over its span.  For $x \ll 1$ the optimum
is one point with the offset known (two with it fitted); for $x = 100$ and
50 positions it is about 35.

**α** is the slope of $\log(\mathrm{MSD}(k) - o)$ against $\log k$ over
the same points, or the first two at least, where the MSD is above the
offset; NaN for a track that hardly moves.  Per track it is noisy: read
the median, or the pooled value.

**Jump distances.**  For the jumps $r$ between localizations exactly $L$
frames apart, a state with coefficient $D_s$ gives per axis a Gaussian of
variance

$$s^2 = 2 D_s L \Delta t + \sigma_1^2 + \sigma_2^2 ,$$

the two localizations' errors from their precisions, so $r^2 / s^2$ is
chi-square with $d$ degrees of freedom.  The fractions and coefficients
of the states maximise the likelihood of all jumps, by expectation
maximisation: the fractions are the mean memberships, and each $D_s$
solves its weighted likelihood equation by a fixed point.  At most 200,000
jumps are fitted (a random subset, the same each run).  The BIC
(Bayesian information criterion, $(2S - 1) \ln n - 2 \ln \mathcal{L}$ for
$S$ states and $n$ jumps) compares fits with different *states*: the
lower is better.  The jumps of one track are not independent, so the
likelihood is overconfident, and no errors are given.

**MINFLUX.**  A table with `time_s` has no regular frames; its
localizations are put on a grid of the median interval between
consecutive localizations of a track, each at the nearest step.

## Parameters

### id_column
*auto* takes the first of `track_id`, `tid` and `track_id_true` the table
has, and the list says which.  Choose another integer column to compare
two linkings (Link tracks' *output column*), or `group_id` to treat each
merged blink as a track.  Id 0 and below is no track.

### msd_fit.offset
*from the precisions* when the fitter's precisions are trustworthy (the
default); *fitted* when they are not, or when motion blur is large -- a
fast particle over a long exposure.

### msd_fit.lags
Leave on *auto*.  A fixed number is what most papers use and what Michalet
showed to be often far from the best: too few points for slow particles
with large errors, too many for fast ones.

### msd_fit.min_length
In 2D, a track of 5 localizations gives a *D* uncertain by about 55%
even with a small localization error, one of 20 by about 25%: for
per-track values to mean something, 20 or more.  The pooled MSD uses every
track.

### jump_fit.states
Fit 1, 2 and 3 and compare the BIC: a state more that does not lower it by
at least about 10 (a common rule of thumb) is not supported by the data.  Two states fitted to one
population come out as two close values of *D*.

### jump_fit.lag
1 for the most jumps.  A larger lag spreads the states further apart
relative to the localization error, which helps for slow states, at the
cost of fewer jumps and of more particles changing state within the lag.

## Output

A line per analysis: the pooled *D* and α with the lags used and the
offset, the median *D* and α per track; the states' *D* and fractions
with the BIC.  A figure: the pooled MSD with the line fitted, the
histogram of *D* per track, and the jumps with the fitted states.  Per
region, the values (`msd_D_um2_per_s`, `msd_alpha`,
`track_D_median_um2_per_s`, `jump_D1_um2_per_s`, `jump_fraction1`, ...)
and the pooled MSD curve (`lag_s`, `msd_nm2`).

Run on the view, the fit of every track is also kept as a
[region table](plugin:Panels/Glossary) of the track id ("per track_id:
Tracking analysis"): `n_locs`, `D_um2_per_s`, `alpha`, `lags` and
`reduced_error` (the localization error over D times the step), NaN for a
track shorter than *minimum length*.  *Summarize/Histograms* shows their
distributions, and the [region filter](plugin:Analysis/Regions/Filter regions)
keeps the tracks of one population -- `alpha` below 0.7, say -- without a
column per value; *write per track* is for colouring the tracks.  Not when
an ROI is drawn, since the table stands for every track the layer shows.

A good result: the MSD a straight line over its first points, α near 1
for free diffusion, the states' curves following the jumps' histogram.  A
pooled MSD that bends down is confinement or a mixture of slow and fast
particles; one that bends up, drift or directed motion.  Per-track *D*
of short tracks is broad even for one population: compare populations by
the jumps' states, not by the width of that histogram.

## Differences from SMAP

Based on SMAP's *AnalyzeSPT*
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), rewritten.  The
main changes:

* SMAP fitted the MSD at lags 2 to 4, always, with *D* the slope over 4
  and the frame time hard-coded at 30 ms.  Here the lags are the optimum
  per track, the offset is the localization error's, *D* is divided by
  $2d$ (3D as well), and the frame time is the file's.
* SMAP fitted a lognormal to the histogram of *D* per track.  Here the
  jumps are fitted with diffusive states, which separates populations
  that the per-track *D* of short tracks blurs together.
* SMAP's diffusion maps and its interactive track view are not ported;
  the tracks are coloured by `track_D_um2_per_s` instead.

## References

* Michalet X. Mean square displacement analysis of single-particle
  trajectories with localization error: Brownian motion in an isotropic
  medium. *Phys Rev E* 82, 041914 (2010).
  [doi:10.1103/PhysRevE.82.041914](https://doi.org/10.1103/PhysRevE.82.041914)
  -- the optimal number of MSD points.
* Michalet X, Berglund AJ. Optimal diffusion coefficient estimation in
  single-particle tracking. *Phys Rev E* 85, 061916 (2012).
  [doi:10.1103/PhysRevE.85.061916](https://doi.org/10.1103/PhysRevE.85.061916)
  -- a known localization error makes D more precise.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
