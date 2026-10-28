# Drift correction: measurements

The investigation log behind `smappy.drift` and `smappy.rcc`, moved out
of `NOTES.md`, which keeps the design decisions ("Drift correction").
The dataset throughout is the clathrin one unless a section says
otherwise.

## Where the time goes

Measured on the clathrin dataset (410 k localizations after filtering, 92 time
windows, 314 M neighbour pairs):

| stage | time |
|---|---|
| segmentation | 0.03 s |
| pair-count estimate (the safety check) | 0.6 s |
| KD-tree neighbour search | 5 s |
| **optimization** | **97 s** (was 309 s) |
| applying the drift and writing the file | ~5 s |

The optimization is everything: L-BFGS-B evaluates cost and gradient a few
hundred times, and each evaluation is one `exp()` per pair over all 314 M of
them -- ~0.6 s, compute-bound at ~2 ns/pair across the cores.  So there are only
two levers: the work per evaluation (the pair count) and the number of
evaluations.  Two changes to the vendored COMET, both opt-in from `drift.py`
(`CPU_WRAPPER` and `LBFGSB_OPTIONS` in `comet/core/drift_optimizer.py` keep
upstream's values for anyone importing COMET directly):

* **`cpu_wrapper_chunked_fast`** -- the original converts its arguments on every
  call, and the pair indices arrive as int32 and are copied to int64: 5 GB of
  allocation and copying per cost evaluation, for nothing, since the compiled
  kernel specialises on whatever dtypes it gets.  0.73 -> 0.61 s per call, same
  answer to 5e-10.
* **`ftol = 1e-7`** instead of COMET's `1e3 * eps` (~2e-13).  423 -> 154 cost
  evaluations.  The drift changes by 0.12 nm rms (2.2 nm on one window out of
  92) and the rendered image by 0.03% -- against a localization precision of
  10-20 nm, that tolerance was buying nothing.

Together: 309 s -> 97 s (and 75 s with the approximate kernel below).  Things that were tried and did **not** help, so that
they are not tried again:

* float32 coordinates, int32 indices, and sorting the localizations spatially so
  that paired ones sit near each other in memory: no change at all.  The kernel
  is compute-bound on `exp`, not waiting on memory, which is also why the
  gradient scatter-add is not worth restructuring.
* a chunked NumPy version, to get NumPy's SIMD `exp`: **44x slower** -- it is
  single-threaded and materialises the per-pair temporaries.
* dropping same-segment pairs (they add a constant and exactly zero gradient):
  correct, but only 6.8% of the pairs.
* a smaller `max_drift_nm`.  Pairs grow with r^3 only for uniform data; this is
  clustered, and going 300 -> 150 nm removes just 21% of the pairs.
* the PyTorch/MPS backend on Apple silicon: the index tensors alone are 5 GB on
  the device and a single cost evaluation did not finish in ten minutes.
  (COMET's CUDA and torch backends were removed in 2026-10: they served only
  the time-window fit, nobody ran them, and the torch one could not import.)

## Saying what it will cost, before it costs it

A 8.4 M-localization, 93 k-frame dataset with `max_drift_nm = 500` does not run
at all: 2.75 M localizations survive the standard filter, grouping leaves
1.7 M, and the neighbour search within 500 nm wants **4 x 10^9 pairs**.  scipy
hands those back as one int64 `(N, 2)` array and `pair_indices_kdtree` copies
them into two int32 arrays -- 24 bytes a pair alive at once, 97 GB, before
counting the vector scipy builds the array from.  On a 34 GB machine that is
tens of minutes in swap and then a dead process, and the optimization it never
reached would have been another 19 minutes (4 x 10^9 pairs x 136 evaluations x
2 ns).  Nothing said so: `pair_indices_safety_check` is only on the segmented
path, and the spline is the default; it warns through `warnings.warn`, which in
the GUI goes to a terminal nobody is reading; and its `estimate_pairs` counts
only within-cell pairs, which came out 40% low on the two datasets checked.

`estimate_cost` answers the question in **0.2-0.4 s** instead, for both paths.
Pairs are counted on a sample and scaled: the count grows as N^2 for a fixed
field, so a sample of 60 k localizations answers for millions.  Measured
against the pair count the real run then built: **0.1%** out ungrouped, 7%
grouped -- and grouped needs the sample to be grouped too, since grouping is
exactly what removes the densest, shortest-range pairs.  The sample is a
stretch of frames from the middle of the acquisition, which is a fair sample of
the field because every frame images the same one, and the ratio it groups by
is the ratio the whole table groups by.

From the pair count both numbers follow, with constants measured here and kept
in one place at the top of the estimator: ~2 ns per pair per cost evaluation,
~16 ns per pair for the KD-tree search, ~23 evaluations per sigma level (136
over 6 levels on the dataset above, 154 over 6 on the clathrin one), 24 bytes
per pair.  They are only ever used to decide whether to warn.  The GUI asks
before starting when the estimate is over five minutes, or when the pairs want
more than half the machine's memory -- the table, its index and the rendered
view are already resident, and the pair figure is a floor.

The asking is `Plugin.preflight`, called on the session's own thread before the
worker exists, so a "no" is no work at all rather than a thread started and
abandoned, and the dialog is an ordinary modal one rather than something
marshalled off a worker.  A preflight that raises is ignored: an estimate is a
courtesy, and one that cannot be made must not be what stops a run.

## What it says while it runs

A drift estimate is one call that can take minutes, and silence is
indistinguishable from a hang -- which is what a 4 x 10^9-pair run looks like
from the outside.  `correct_drift`, `estimate_drift` and the spline estimator
take a `progress` callback (the GUI passes the plugin panel's `ctx.report`, the
command line still uses `display`), and report the stages that take the time:
grouping and what it left, the neighbour search and its radius *before* it
starts -- that call is uninterruptible, so a run that dies in it at least says
what it was asked for -- the pair count, and then each sigma level.

Inside the optimizer the evaluations all cost the same, one pass over the
pairs, so two of them are enough to say how long the rest will take: the line
is "evaluation 12 of about 138, 8.1 s each, ~17 min left", rewritten at most
once a second because each one is a queued signal into the GUI thread and on a
small dataset they arrive hundreds of times a second.  The total is
approximate -- L-BFGS-B decides when it is done -- which is why it says "about".

## Two more that were tried

* **A looser tolerance for the coarse sigma steps** -- the idea being that they
  only have to land the estimate in the right basin.  **Refuted.**  Coarse
  `ftol=1e-4` is 2.3x faster but moves the drift by 4.8 nm rms and 52 nm on
  individual windows, and a *tighter* final step does not repair it: a local
  optimizer at small sigma cannot undo a wrong basin.  Loosening `gtol` instead
  traces the same curve (48 evaluations, 2.4 nm rms), so it is not an artefact
  of `ftol` being a relative criterion.  Anything below ~140 evaluations costs
  tens of nm.  The knob survives as `optimizer_ftol_coarse`, defaulting to off.
* **An approximate kernel** (`cpu_wrapper_chunked_approx`).  Kept: **1.35x**,
  100.8 -> 74.8 s end to end, drift unchanged (0.12 nm rms against the reference,
  the same as the exact kernel) and image sharpness identical to four digits.
  `exp` is 54% of the kernel on ARM, so it is tabulated on the integer part and
  Taylor-expanded on the fraction; pairs beyond 6 sigma are skipped, and the
  four divisions per pair become one hoisted reciprocal.  A 4 sigma cutoff was
  measured too: no faster, and two orders of magnitude less accurate.
  *Since the kernel moved to C++ (`csrc/drift.hpp`) only the cutoff is left*:
  it calls `std::exp`, so the Gaussian is exact and "approximate" means only
  that pairs beyond 6 sigma are skipped -- which is where the saving was, once
  sigma has tightened.  The table and the Taylor series were numba's problem.

## The noise floor, and how to measure it

Deviation from a previous run is a bad yardstick without knowing how much a
drift estimate varies by chance.  Split the filtered localizations into two
disjoint halves, estimate from each, and their difference is pure noise by
construction; since noise goes as 1/sqrt(N), the full estimate's noise is half
the difference between the halves:

| estimate | halves differ by | noise of the full estimate |
|---|---|---|
| ungrouped | 3.25 nm rms | **1.6 nm** |
| grouped | 6.19 nm rms | **3.1 nm** |

So a change that moves the drift by less than ~2 nm rms is not measurable, which
is what makes `ftol=1e-7` (0.12 nm) and the approximate kernel (0.12 nm) safe,
and what makes the coarse-tolerance experiment (4.8 nm) a real degradation.

## Grouping before estimating

`DriftSettings.group` estimates from grouped localizations -- one entry per
blink instead of one per frame.  On the clathrin dataset 410 k -> 146 k
localizations, and **314 M -> 21 M pairs**: the whole correction takes 5 s
instead of 1:17.  It beats the 2.8^2 that the localization count alone suggests,
because it removes exactly the dense very-short-range pairs, which is where the
pair count grows fastest.

Grouping links in x and y only and averages z, which is right: one localization
per frame is one molecule (multi-emitter fits are removed by the log-likelihood
cut), and a molecule does not move in z between consecutive frames, so averaging
z only makes it more precise.

**Filter before estimating, and include z.**  (On simulations the precision
weights now do most of this filter's work; see "Precision weights and the
width in z".  This dataset has not been run again with them.)  Without a z cut the two estimates
disagreed by 8.2 nm rms and the ungrouped z drift range came out 113 nm against
the grouped 75 nm -- which looked like grouping damaging the axial estimate, and
was not.  `z_err_nm` has a median of 41 nm and a 95th percentile of 108 nm: the
out-of-focus tail carries the axial estimate away.  With
`logl_rel > -2`, `xy_err_nm < 15`, `|z_nm| < 300` (323 k of 844 k
localizations) the ranges agree (69 vs 75 nm) and the two curves lie on top of
each other: **median |difference| 0.9 nm in x, 0.6 in y, 1.4 in z**.

What is left is that grouping doubles the noise of the estimate (3.1 nm against
1.6 nm, split-half) because there are fewer, though better, localizations per
window.  On the image that costs a few percent: of the improvement the full
estimate makes over no correction, grouping recovers 93% in the x-y projection
and 82% in x-z.  So the full estimate stays the default and grouping is the
switch for a first look, for large datasets, and wherever 35x matters more than
a few percent of resolution.

## A failed window, and quality control on the CPU

The ungrouped estimate on this dataset contains one **bad time window** (49,
centred on frame 24 819): the drift swings 46 nm in y and 34 nm in z and comes
straight back, over 2500 frames, where the grouped estimate -- same data, more
precise positions -- moves 6 nm.  A stage does not do that.  It is what inflates
the grouped-vs-ungrouped rms (4.3 nm in y) over its median (0.6 nm).

COMET's answer to this is quality control: per segment, compare the overlap the
fitted drift achieves against the overlap with no drift correction at all, and
NaN the segments that fail, so the spline bridges them.  It was wired into the
`cuda_qc` and `torch_qc` backends only (since removed); the CPU path raised "not implemented".
It now works there (`cpu_wrapper_chunked_qc`, `DriftSettings.quality_control`,
`--qc`), one extra pass over the pairs, about 4 s of the 150 s run.

**The criterion, however, does not work as written, and the replacement only
half works.**  Both findings are worth keeping:

* Upstream flags a segment when ``q_obs < q_null + std(q_null)``, where the
  standard deviation is taken *across* segments.  That spread is not an
  uncertainty on any one segment: on a bleaching sample the pair count per
  window falls from 14 M to 2 M, so the null varies five times more between
  segments than the correction improves any of them.  It flagged **90 of 92**
  windows here; with 18 left the spline bridged gaps of thousands of frames and
  the estimated drift range blew up from 53/75/69 nm to 491/267/591 nm.
* Normalising instead by the null itself -- the **lift**,
  ``q_obs / q_null - 1`` (`flag_flawed_segments_by_lift`, the default here) --
  removes that density dependence, and at sigma = 20-30 nm it does single out
  window 49 as the only segment whose fitted drift makes its own overlap
  *worse* than no correction.  But the lift is a function of sigma, and at the
  sigma this run actually finishes at (13 nm) window 49 comes out at +0.06 and
  is not flagged.  So the flag is real but marginal, and it catches this failure
  only at some sigmas.

The reason neither works well is that overlap-versus-null is not the quantity
one wants.  Window 49 has few pairs (2.7 M against 14 M early in the movie), so
its cost landscape is flat: the estimate is badly *constrained*, while its
overlap is not obviously bad.  What that asks for is the **curvature of the cost
in each segment's three parameters**, which is an uncertainty per window in nm,
accumulated in the same single pass and needing a threshold in nm rather than a
tuned dimensionless one.  That is the thing to implement next; it would also
give the drift curve error bars, which nothing currently does.

## Redundant cross-correlation

`smappy.rcc` is the other classical estimator -- bin the acquisition into time
windows, render each, and measure how far one has moved against another by
cross-correlation -- ported from SMAP's `finddriftfeature.m` /
`finddisplacementZ2.m`.  It shares no code and almost no assumptions with the
overlap estimator, which is the point: it is the second opinion on a drift
curve, and it takes seconds.

*Redundant* means every pair of windows is correlated, not just consecutive
ones, giving T(T-1)/2 measurements of `d_k - d_l` for T unknowns; the drift is
the robust least-squares solution of that overdetermined system, so a
correlation peak found in the wrong place is outvoted instead of accumulated.

Compared on the same footing -- same filter, all estimating from **grouped**
localizations, each judged on the half of the molecules it did not see:

| estimator | time | noise x/y/z (nm) | out-of-sample x-y | x-z |
|---|---|---|---|---|
| **COMET spline, 2000 fr knots (default)** | 3.6 s | 0.69 / 0.61 / 0.73 | **4.4249e-5** | **3.7097e-5** |
| COMET spline, 2300 fr knots (= RCC window) | 2.0 s | 0.63 / 0.58 / 0.61 | 4.4178e-5 | 3.7068e-5 |
| COMET per-window | 1.7 s | 1.25 / 1.70 / 1.45 | 4.3983e-5 | 3.6563e-5 |
| RCC, 20 windows | 7.4 s | 0.61 / 0.90 / 0.86 | 4.3843e-5 | 3.6580e-5 |
| no correction | | | 3.5268e-5 | 3.1192e-5 |

At matched smoothing -- spline knots at the RCC window spacing -- the two agree
to **1.2 / 1.7 / 1.6 nm rms**, at the level of their own noise, from methods
sharing no code and no assumptions.  The spline is the better of the two on both
counts here, but not by much, and RCC's axial noise is close to it.

Grouping matters as much to RCC as it does to COMET, so it is on by default
there too.  Comparing a grouped estimator against an ungrouped one, which is
what the first version of this table did, mostly measures the grouping.

Two things were needed to get there, both of which cost a factor of ten or more
in accuracy when wrong:

* **The axial correlation must be tiled in both x and y.**  SMAP's
  `finddisplacementZ` slices only in x, each slice spanning the whole field;
  each z profile then pools hundreds of structures into a slab whose correlation
  peak has nothing to lock onto.  Measured: pairwise shifts *uncorrelated* with
  COMET's (-0.09) with x-slices, +0.96 with 200 nm square tiles.  The tiled
  version, `finddisplacementZ2`, is the one to follow.
* **The lateral drift must be taken out before the axial pass**, or a tile holds
  different structure in different time windows and the z profiles being
  correlated are not of the same thing.

`exclude_zero_lag` leaves the zero-lag sample out of the axial peak search and
fit, on the grounds that anything sitting at the same z in both windows --
a molecule on in both, localizations pinned to a z plane by the fit -- pulls the
estimate towards zero drift.  Measured here it makes no difference (0.86 nm
axial noise excluded against 0.70 nm kept, identical agreement with COMET),
because grouping already collapses each molecule to one entry and this data has
no pinning: no 5 nm z bin holds more than three times the median count.  It
stays on as insurance for data that does; a z histogram is how to check.

Other departures: the correlation peak is located by a quadratic fit rather than
a 2D Gaussian (the peak is not Gaussian, and this needs no starting values); the
overdetermined system is solved by iteratively reweighted least squares rather
than MATLAB's robust `nlinfit`, since `d_k - d_l` is linear in the unknowns and
needs no nonlinear fit; per-frame interpolation is PCHIP, which does not
overshoot between windows.

## Drift as a spline in time

`DriftSettings.spline` (`--spline`) replaces the free vector per time window
with a cubic B-spline in time, fitted directly: the optimizer's variables are
the spline's coefficients, the cost and gradient are COMET's over the same
pairs, and the chain rule to the coefficients is one matrix product.  There is
**no segmentation** -- each localization sits at its own frame -- and no
interpolation afterwards, because the fitted curve *is* the per-frame drift.
B-splines are variation-diminishing, so it cannot overshoot the way an
interpolating spline through noisy window estimates does.

Drift is slow creep with the occasional small jump, so a coefficient every 2000
frames (26 of them here, against 276 free window parameters) describes it, and
each coefficient is constrained by every localization near it in time rather
than by one window's worth.

Judged on halves that are **disjoint in molecules** (see below), and by
correcting each half with the *other* half's drift so that no estimate is graded
on localizations it was fitted to:

| estimator | time | noise x/y/z (nm) | out-of-sample x-y | x-z |
|---|---|---|---|---|
| windows, ungrouped | 34 s | 1.87 / 3.73 / 5.19 | 94.0% | 100.0% |
| windows, grouped | 2 s | 1.25 / 1.70 / 1.45 | 100.0% | 90.0% |
| spline, ungrouped | 31 s | 1.62 / 3.39 / 5.03 | 94.1% | 100.7% |
| **spline, grouped** | **2 s** | **0.93 / 0.81 / 0.99** | **102.9%** | 95.9% |

Knot spacing on the full data: 200 frames gives 1.7-2.0 nm noise, 2000 gives
0.6-0.7 nm, and the out-of-sample sharpness is flat from 1000 to 4000 -- no sign
of over-smoothing, which is what a slowly drifting stage should look like.
`spline_penalty` adds a second-difference penalty on top if a dataset needs it.

## Time resolution, and the first seconds

The estimate can be made as fine as one likes -- `spline_knot_frames` for the
overlap fit, `n_timepoints` for RCC -- and on this dataset finer is worse
everywhere except at the very start:

| curve | ranges x/y/z (nm) | dz over frames 0-1000 | time |
|---|---|---|---|
| spline, 2000 fr knots (default) | 62.5 / 70.1 / 69.9 | 4.4 nm | 2.6 s |
| spline, 500 fr knots | 63.5 / 74.8 / 72.7 | 3.9 nm | 5.1 s |
| RCC, 20 windows of 2300 fr (default) | 56.0 / 71.5 / 61.1 | 0.0 nm | 3.8 s |
| RCC, 92 windows of 500 fr | 58.8 / 75.6 / 73.6 | 7.7 nm | 72 s |

Past frame 30 000 both 500-frame estimators break into +-5 nm oscillations that
the defaults do not have, and the two disagree on their phase -- noise, not
signal.  That is where the localization density has fallen from 31 to 4.4 per
frame and the median precision has gone from 10 to 23 nm, so a 500-frame window
has the least to work with.  Split-half noise says the same: 0.69/0.61/0.73 nm
at 2000-frame knots against 1.14/1.10/1.27 at 500.  RCC's cost also grows with
the square of the window count -- 4186 correlations instead of 190.

**The drift really is faster at the start**, which is worth knowing before
trying to regularise it away.  Over the first 5000 frames the lateral drift
moves 31 nm in x and 39 nm in y, against ~10 nm per 5000 frames later.  It is
not an artefact of density or of mislocalization: a strict filter
(precision < 10 nm, `logl_rel > -1.4`) makes it *larger* (37 / 56 nm), and
matching the localization density across the movie changes nothing (33 / 52 nm).
Both estimators show it, grouped and ungrouped alike.  The first 2000 frames
also carry 4x the density of the rest at 2.5x fewer photons each -- a sample
that has just been mounted, settling.

There is a separate, sharper feature in **z over the first ~1000 frames**.  RCC
at its default 20 windows appears flat there, but only because its first
estimate sits at frame 1150 and PCHIP extrapolates: given 500-frame windows it
shows 7.7 nm, larger than either spline (3.9-4.4 nm) and in the same direction.
Three bases agree it is there.  What is *not* real is the additional ~6 nm spike
at frame 0 that only the per-window fit has (39 nm against 28-32 for everything
else): its first window has no left neighbour to constrain it and the cubic
interpolation through it overshoots at the edge.  The spline has no such edge.

Not a way to fix any of this: `spline_penalty`.  At 1e3 it flattens the initial
lateral excursion from 31 nm to 3.6 nm and simultaneously moves the curve 12.9 nm
away from RCC over those frames and 6.7 nm away over the whole movie.  It
removes the correction, not an artefact.

## Two ways to be wrong about which estimate is better

Both of these produced a wrong conclusion earlier in this work, so they are
written down:

* **Split by molecule, not by localization.**  A blink is several localizations
  of one molecule in consecutive frames.  Splitting localizations at random puts
  one molecule in both halves, the halves share their errors, and the noise
  comes out too low -- it said the ungrouped estimate had a 1.6 nm noise floor
  against the grouped one's 3.1 nm.  Splitting whole molecules reverses it:
  1.9/3.7/5.2 nm ungrouped against 1.3/1.7/1.5 nm grouped.  Grouping does not
  throw information away; treating a molecule's repeats as independent
  measurements adds correlated noise, and over-weights bright molecules.
* **Grade out of sample.**  Rendering the same localizations the drift was
  fitted to rewards the estimator that fitted their noise.  In sample the
  ungrouped estimate looked sharper (3.9761e-5 against 3.9056e-5); correcting
  each half with the other half's drift reverses that laterally.

The remaining honest gap is axial: the ungrouped estimate is still slightly
better in x-z (100% against 95.9%), even though its z drift is by far the
noisiest.  Worth understanding rather than explaining away.

## Two passes: grouped, then a small radius

*Both passes, and the RCC prepass, used to run only with the time-window fit.*
`estimate_drift` took the spline branch before looking at `two_stage` or
`rcc_prepass`, so with the spline on -- the default since it was added -- both
were skipped without a word.  The slow-run dialogue's "RCC first, then COMET
within 50 nm" then ran one spline fit over 50 nm and no RCC: on 10 000 simulated
frames with drift, 29 / 42 / 10 nm rms error in x / y / z where the default
gives 10 / 6 / 6.  The passes are now checked first and each runs the fit that
was asked for; the same offer gives 8.7 / 5.5 / 5.5 nm.  Every test of the
passes had set `spline=False`, which is why none saw it.  COMET's version went
to 2.  (Found writing its documentation page, 2026-09-27.)

`DriftSettings.two_stage` (`--two-stage`) runs the grouped estimate first, takes
that drift out, and then runs an ungrouped estimate over a **30 nm** search
radius.  The pair count is what costs time, and once the coarse pass has removed
the drift the fine pass only has to look a few nm: 314 M pairs become 42 M.

| | time | vs the single pass (x/y/z) | sharpness x-y | x-z |
|---|---|---|---|---|
| single pass, ungrouped | 153 s | reference | 3.9761e-5 | 3.3623e-5 |
| grouped only | 6 s | 1.8 / 4.3 / 4.7 nm | 3.9056e-5 | 3.2388e-5 |
| grouped + fine, r=100 nm | 91 s | 1.5 / 6.6 / 6.3 nm | 3.9465e-5 | 3.3769e-5 |
| grouped + fine, r=50 nm | 26 s | 1.2 / 4.0 / 3.5 nm | 3.9583e-5 | 3.3626e-5 |
| **grouped + fine, r=30 nm** | **18 s** | 0.8 / 3.6 / 3.0 nm | **3.9743e-5** | 3.3513e-5 |

**8.7x faster for 99.8% of the lateral and 98.4% of the axial improvement**, and
theremaining difference is at the noise floor (1.6 nm).  Note that the earlier estimate
of what this would gain -- 2.5-3.5x -- was too pessimistic: it assumed the fine
pass would need as many cost evaluations as the single pass, and it does not,
because it starts a few nm from the answer instead of a hundred.

It is also **more robust**, which was not the point of it.  The fine pass is
bounded by its own radius, so it cannot make a large excursion: over the window
where the single pass fails (frames 24 000-26 500) the single-pass estimate
swings 46 nm in y, and the two-stage one 7.9 nm.  A small search radius is a
speed knob and a regularizer at once.

The radii are tuned on one dataset.  `two_stage_radius_nm` sets the fine pass's
radius and, from it, its sigma schedule (radius/3 down to radius/5); it must
stay well above the residual the grouped pass leaves, which was a few nm here.

What is left is a real trade, not a free win, so it stays a parameter:
`max_locs_per_segment` caps the localizations per time window and the pair count
falls roughly quadratically -- 2000 per window is 10x faster (33 s) and moves
the drift by 2.5 nm rms, 1000 per window is 15x faster and moves it by 6 nm.
Worth it for a first look at a large dataset, not for the final result.

## Precision weights and the width in z

COMET's overlap counts every pair alike, with one Gaussian width for x, y
and z.  Two things were tried against that (survey candidate #4,
`docs/survey/drift_averaging_fitting.md`, DME): **(a)** a wider Gaussian in
z, $\sigma_z = f\sigma$, with $f = 3$ -- the median `z_err_nm` /
`xy_err_nm` of these data, which was the plan for its default; **(b)** DME's weighting, each pair a Gaussian of variance
$2\sigma^2 + p_i^2 + p_j^2$ per axis from the two localizations' own
precisions, normalised as a density, so an imprecise pair is low and wide.
Between them, the same widening with every localization given the median
precision.  Both are in the kernel (`csrc/drift.hpp`: `z_factor`,
`variances`).

**On simulations, not the clathrin dataset**: `studies/drift_weights`
(README there): 3D filaments through 600 nm of z, median precision 14 nm
laterally and 41 nm in z, 90-150 nm of slow drift over 20 000 frames,
defaults otherwise (merged blinks, 2000-frame knots, target sigma 30 nm).
rms error of the curve against the drift put in, nm, x / y / z, over four
seeds:

| data | COMET | z width 3 | median precision | **precision weights** |
|---|---|---|---|---|
| dense (100 k locs), `xy_err_nm < 15` | 0.51 / 0.44 / 0.59 | 0.51 / 0.43 / 0.71 | 0.50 / 0.43 / 0.58 | **0.49 / 0.42 / 0.54** |
| dense, everything | 0.48 / 0.50 / 0.56 | 0.56 / 0.54 / 1.16 | 0.51 / 0.54 / 0.65 | **0.50 / 0.40 / 0.49** |
| dense, 20% imprecise | 0.52 / 0.53 / 0.68 | 0.59 / 0.58 / 1.68 | 0.57 / 0.59 / 0.98 | **0.53 / 0.44 / 0.57** |
| sparse (35 k locs) | 0.78 / 0.89 / 1.13 | 0.82 / 0.94 / 1.80 | 0.86 / 0.92 / 1.22 | **0.71 / 0.79 / 0.98** |
| sparse, 20% pinned | 0.83 / 0.87 / 1.42 | 0.88 / 0.95 / 2.41 | 1.02 / 0.98 / 2.20 | **0.75 / 0.85 / 1.14** |
| the same, `z_err_nm < 100` | 0.72 / 0.91 / 1.26 | | | 0.72 / 0.88 / 1.09 |
| drift x2, 40% pinned | 4.35 / 1.94 / 4.62 | 5.65 / 2.61 / 15.4 | | **4.19 / 2.15 / 2.46** |
| the same, `z_err_nm < 100` | 3.24 / 1.56 / 1.82 | | | 3.37 / 1.67 / 1.63 |

*Imprecise*: a fraction of the localizations given 20-40 nm laterally and
150-300 nm in z, recorded so and moved so at random.  *Pinned*: out-of-focus
fits held at z = ±400 nm whatever the z drift, recorded as imprecise -- the
fit's edge, which is what "the out-of-focus tail carries the axial estimate
away" above most likely was.  (The x errors of the last two rows are the
drift reaching *max drift*: see below.)

* **(a) is worse at the default target sigma**, in z always and by a lot
  where there is junk: 30 nm is already wider than most z precisions, and a
  Gaussian three times wider lets the pinned planes, which do not move with
  the drift, outvote the structure (15 nm against 4.6; with $f$ taken from
  the junk-laden data's own median ratio, above 3, it was 23 nm).  Widening by the
  median precision is the same mistake, smaller.
* **(b) is better everywhere**: 10-20% on every axis on clean data, and
  half of COMET's axial error with 40% pinned junk -- within a nanometre of
  what the hand-set cut on z gives, which is what it was for.  With the cut
  as well it is better again.  It costs nothing measurable (one more load
  and a square root per pair).

The ranking of (a) turns only at a target sigma near the lateral precision,
where 30 nm in z is narrower than z's precision (sparse, four seeds):

| target sigma | COMET | z width 3 | precision weights |
|---|---|---|---|
| 30 nm | 0.78 / 0.89 / 1.13 | 0.82 / 0.94 / 1.80 | **0.71 / 0.79 / 0.98** |
| 15 nm | 0.66 / 0.64 / 1.12 | 0.64 / 0.62 / 1.09 | **0.58 / 0.51 / 0.97** |
| 8 nm | 0.58 / 0.52 / 1.83 | 0.53 / 0.46 / **1.17** | **0.45 / 0.36** / 1.20 |

and there the weights do as well.  So **(b) is on by default**
(`DriftSettings.precision_weights`), and (a) is a setting at 1 (*z width*).
Two things are worth a look on real data: a smaller default target sigma
(15 nm was better on every axis with the weights here, but these
simulations have no aberrations, no mislocalized multi-emitter fits and no
uneven structure), and DME's other ingredient, the logarithm of each
localization's neighbour sum, which saturates a dense cluster.

## Every run says its noise

The noise floor above was measured by hand, once.  It is now part of every
COMET and RCC run (*error bar*, on by default): the estimate is made again
from each half of the field, cut at the median of x or y, whichever the
selection spreads along further -- DME's split, which is a split by
molecule, as "Two ways to be wrong" asks -- and the noise is half the rms
of the halves' difference.  It doubles a COMET run (each half has about
half the pairs), and the cost estimate says so.  On the simulations of the
table above the reported noise and the true error agreed to within a
factor of about 1.5 in either direction wherever the error was noise (2 in
the unit tests' small simulation, `tests/test_drift.py`); it
does not see a bias the halves share -- 40% pinned junk had 4.6 nm of
axial error and reported 1.4 nm of noise.

## Wavy drift where there is none

COMET on a simulation with no drift returns a curve that wanders by a few
nm.  Is that the noise of the estimate or a bias of the spline (knots every
2000 frames, no penalty)?  Noise, on every count (`studies/drift_weights`,
`driftless`; rms of the curve about its mean, x / y / z, against the noise
the error bar reports, three seeds):

| density | locs | knots 500 | 1000 | 2000 | 4000 | noise at 2000 |
|---|---|---|---|---|---|---|
| 0.01 / nm | 18 k | 1.9 / 1.9 / 2.5 | 1.3 / 1.3 / 1.6 | 0.8 / 1.0 / 1.2 | 0.6 / 0.9 / 1.0 | 1.0 / 1.0 / 1.3 |
| 0.02 / nm | 35 k | 1.4 / 1.4 / 1.7 | 0.9 / 1.0 / 1.2 | 0.6 / 0.8 / 0.9 | 0.5 / 0.7 / 0.7 | 0.8 / 0.6 / 0.8 |
| 0.04 / nm | 72 k | 0.9 / 1.0 / 1.1 | 0.6 / 0.8 / 0.8 | 0.4 / 0.6 / 0.6 | 0.3 / 0.5 / 0.6 | 0.4 / 0.5 / 0.6 |
| 0.08 / nm | 140 k | 0.6 / 0.7 / 0.9 | 0.4 / 0.4 / 0.7 | 0.3 / 0.3 / 0.5 | 0.2 / 0.3 / 0.3 | 0.3 / 0.4 / 0.5 |

* it falls as $1/\sqrt{\mathrm{density}}$ (0.83, 0.61, 0.41, 0.30 nm in x at
  2000-frame knots: a factor 0.7 per doubling);
* it grows as the knots get finer, by about $\sqrt{2}$ per halving;
* it is the split-half noise, cell by cell: the ratio of the two had a
  median of 1.0 and a largest value of 1.7 over 144 curves' axes.

Peak to peak that is 3-15 nm at the sparsest -- "a few nm of wavy drift" --
and a spline whose coefficients fit noise looks wavy with a period of a few
knots.  A spline bias would not fall with density, and with no drift there
is nothing for the spline to miss.  The precision weights take 10-15% off
it (0.01 / nm: 1.01 / 1.03 / 1.20 nm without, 0.83 / 0.97 / 1.20 with).

What is done about it: **the text says when a curve is not drift.**  An
axis whose rms about its mean is below twice its split-half noise is
reported as "no drift above the noise" (`Drift.insignificant_axes`,
`SIGNIFICANT`); the largest null ratio, 1.7, is below that, and a real drift
is tens of times its noise.  The simulation's default scene with no drift
now says so on all three axes, and with its drift on none.

Not done, and the next step if wanted: **choosing the knot spacing from the
data.**  The halves give an out-of-sample criterion for free -- fit the
curve on one half, score the overlap of the other half's pairs under it,
and take the spacing where that is largest -- which is the out-of-sample
sharpness this file used to pick 2000 frames by hand.  It costs a fit of
each half per candidate spacing (four candidates: about four times a run),
and on the clathrin data that sharpness was flat from 1000 to 4000 frames,
so it would mostly confirm the default.  A penalty chosen the same way is
the alternative; the knot spacing is the more readable knob.

## Drift close to max drift is underestimated

Found along the way: with 150 nm of drift range and *max drift* 150 nm the
curve came out with 9.5 nm rms error in x, shaped like a shrunken copy of
the drift; at 300 nm the same data gave 0.2 nm.  The pairs are found once,
in the uncorrected positions, so a molecule seen at two times further apart
than *max drift* never pairs with itself, and the pairs that remain favour
smaller drifts.  With 305 nm of drift range and *max drift* 300 nm the bias
was 3-4 nm (the last two rows of the first table).  *max drift* wants
margin over the largest drift, not equality.

