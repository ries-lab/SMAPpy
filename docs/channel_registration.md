# Registering two channels from localizations

The 2D two-colour workflow (`Localize/Gaussian 2D 2C`) has no PSF model to
calibrate.  What it needs instead is a **registration**: where on the chip the
second channel puts the same molecule.  In a ratiometric experiment that can be
read off the data itself, because every molecule is already imaged twice in the
same frame.

    Localize/Gaussian 2D          fit the whole frame, ignoring the split
    Analysis/Register/Calibrate transform   pair those localizations, fit the map
    Localize/Gaussian 2D 2C       fit both halves as one emitter

or, in one step, tick **calibrate from this movie** on the 2C fit and it runs
the first two itself on the leading frames.  This is SMAP's `fit_dualcolor`
sequence -- fit, `RegisterLocs2`, refit -- with the middle step measured
differently, and the final fit a global one rather than an intensity readout.

## What it does

### 1. Vote for the channel offset

For every pair of localizations in a frame, the vector between them.  Over
thousands of frames the offset between the channels is the vector that shows up
over and over; everything else is spread thin.  Accumulate them into a 2D
histogram, take a difference of Gaussians (the same filter the peak finder uses
on an image, for the same reason: the answer is a *sharp* peak and the things
one is not interested in are broad), and the argmax is the offset.

This is a cross-correlation.  `RegisterLocs2` computes the same thing by
rendering both channels into 500 nm histograms and correlating the images; doing
it on the point pairs directly costs nothing in accuracy and buys one thing:

> **no part of it assumes where the split is.**

The offset is found first and the split position follows from it.  A splitter
whose boundary sits nowhere near the middle of the chip needs no initial shift,
no magnification guess and no correctly-set image centre -- the settings that
most often make a registration fail quietly.

A mirrored layout is the same argument with one sign changed.  Reflected about
a line at `m`, partners satisfy `a + b = 2m - 1` rather than `b - a = d`, so the
vote is taken on the **sum** along the mirrored axis.  Voting in all three
feature spaces -- plain difference, x summed, y summed -- and keeping the
sharpest peak is what lets the layout be detected instead of declared.

### 2. Pair and refit, twice

Which frames go in matters as much as how many.  The calibration pass takes its
frames as a handful of evenly spaced blocks across the **whole** movie rather
than as one run at the start, and skips the opening frames outright.  Both were
measured on a real dataset:

| frames used | result |
|---|---|
| 600, contiguous from frame 0 | **fails**: "singular projective transformation" |
| 600, contiguous from frame 500 | 6,303 pairs, 100% coverage, 0.140 px |
| 600, six blocks across 46,000 frames | 3,620 pairs, 97% coverage, **0.130 px**, vote contrast 47 against 19 |

The opening frames of a movie are not single molecules -- every fluorophore is
still on and the frames are often saturated -- and what a fit makes of them is
a dense mat of spurious positions that swamps the vote and leaves the
projective fit collinear rubbish.  It is skipped rather than down-weighted
because none of it is usable.  Spreading the rest costs nothing (reads stay
sequential inside each block) and buys a vote that is more than twice as sharp,
because a registration measured on one minute of a movie describes whichever
molecules were blinking in that minute -- which in a sample that is not
uniformly labelled is *a part of the field*, the one thing a transformation
must not be fitted on.


Round one pairs by the vote alone, mutually nearest within 20 px, frame by
frame, and fits a projective map through `fit_dual_transform` (RANSAC, then a
soft-L1 refinement under a componentwise dx/dy screen -- the same function the
bead calibration uses).  Round two re-pairs through that map at 1.5 px and
refits: SMAP's coarse pass followed by clean matches only.

Two stages, not three.  Measured on simulated data with a 3° rotation and 2%
scale:

| schedule | error over the field | pair residual |
|---|---|---|
| coarse only (fine = 0) | 0.036 px | 0.25 px |
| 20 then 1.5 px (the default) | 0.005 px | 0.029 px |
| 20, 5, then 1.5 px | 0.005 px | 0.029 px |

The tight round is worth a factor of seven; an intermediate one is worth
nothing, because once the coarse round has the map to a fifth of a pixel the
tight round keeps the same pairs whether or not it was approached gradually.

The vote only ever finds a translation.  Rotation, scale and the projective
terms are the refit's job, and the first tolerance is what has to cover them:
3° over 512 px is 27 px of spread, which is why it defaults to 20 px and is the
first thing to raise if a badly rotated channel finds no pairs.

### 2b. Keeping the edges of the field

A tight second round has a failure mode worth naming, because nothing in the
residual reveals it.  If the first round's transformation is slightly wrong,
it is wrong *most* at the edges of the field; those pairs are then screened
out, the transformation is refitted on what is left -- the middle -- and the
second round, pairing at a tolerance the edges can no longer meet, never sees
them again.  The registration ends up measured on a disc in the middle of the
chip and extrapolated over everything else, while reporting an excellent
residual on the pairs it kept.

Two things prevent it.

**The tight tolerance follows the misfit.**  After the coarse round, the
residual is measured over *every* pair it matched -- not over the ones it
chose to keep, which is the distinction that matters, because the rejected
ones are the edges.  The tight round then opens up to `PAIR_MARGIN` times the
98th percentile of that, so it can still reach what the coarse fit missed.
Measured against ground truth, with a field distortion a projective map cannot
absorb:

| distortion | fixed 1.5 px | tolerance follows the misfit |
|---|---|---|
| barrel 0.01 | 71% of edge pairs kept | 100% |
| barrel 0.03 | 0% | 99.9% |
| barrel 0.08 | 0% | 91.6% |

On a clean field it changes nothing at all -- the same pairs, to the pair.

**A pooled inlier test that actually converges.**  `robust_projective`
re-fitted on its inliers until the inlier *set repeated exactly*.  With the
tens of pairs a bead calibration has that happens; with the thousands a
localization registration has it never does, because a handful of pairs always
sit within noise of the threshold and flip every iteration.  It raised rather
than converged, which is why any field it could not fit perfectly failed
outright instead of degrading.  It now settles for a set that is no longer
changing meaningfully.

### 2b'. Weighting by localization precision

The fit is inverse-variance weighted: a pair's weight is
`1 / (sigma_ref^2 + sigma_sec^2)`, taken from the fit's own
`loc_precision_pix`.  (The bead path stays unweighted — there the soft-L1 loss
expresses geometric robustness alone, which is right for beads, all of which
are bright.)

This matters because of the dim channel.  A two-colour splitter usually has
one, the detection threshold has to come down to find its partners at all, and
what that buys is pairs with poor precision.  Unweighted they count the same as
the good ones and win by sheer number.  Measured against ground truth:

| pairs per frame | unweighted | weighted |
|---|---|---|
| 3 bright only | 0.0017 px | 0.0017 |
| 3 bright + 3 dim | 0.0058 | **0.0018** |
| 3 bright + 12 dim | 0.0125 | **0.0016** |
| 1 bright + 12 dim | 0.0128 | **0.0030** |
| 15 dim, no bright | 0.0160 | 0.0160 |

Unweighted, lowering the threshold makes the registration **seven times
worse**, and adding still more dim pairs does not compensate -- they dominate
by count.  Weighted, they cost nothing and help slightly (3 bright + 12 dim
beats 3 bright alone).  So the rule is: lower the threshold freely, but only
because the fit knows which pairs to believe.

### 2c. Judging a registration fairly

The overall residual is **not** comparable between two runs that kept
different pairs: widen the tolerance, the edges come back, and the number
grows -- an improvement reported as a regression.  Measured on a clean field,
widening the tight round from 1.5 px to 20 px leaves the true accuracy
unchanged at 0.0014 px while the reported residual inflates fivefold.

So what is recorded is the residual in the **middle** of the paired region and
at its **edge**, separately.  Flat means noise; rising means the model is
wrong, and a warning says so, because the remedies are opposite -- more pairs
will not help a model that cannot describe the field, and a wider tolerance
only spreads the same error more evenly.

The coarse tolerance is set by the rotation it must cover and nothing else.
A looser one does not degrade the result: measured against ground truth, the
true error after the tight round is flat from 6 px to 30 px at every density
tried, because the tight round removes what the loose one let in.

### 2d. How many frames, and which

Not a fixed count.  What a registration needs is enough *pairs*, and a pair
needs both partners -- so the binding constraint is the **dim** channel, and
that is what is counted.  Blocks are read until the dimmer half has
`calibrate_locs` localizations (10,000 by default) or the frame cap is
reached, so a bright dataset stops early and a sparse one keeps going instead
of quietly registering on too little.

Per-half thresholding is worth a great deal here.  On the NPC benchmark, at
the same cutoff, it takes the dim channel from 4,100 localizations to 14,632
and the pairs from 1,780 to 4,364 -- and lowering the cutoff on top of that
reaches 32,308 and 5,973.

The detection during that pass thresholds each half separately too, against a
*provisional* seam -- the middle of the chip's longer axis, or whatever the
settings already say.  There is a chicken-and-egg here: per-half thresholding
is what lets the dim channel be found, and the real seam is not known until
the registration that needs those detections has run.  A rough seam settles
it, because the threshold only needs the two populations kept apart: a few
pixels off costs a thin band judged against the wrong half, while pooling the
two costs the dim channel outright.

### 3. The split position

Measured, not assumed -- with one honest exception.

**Unmirrored**, the pairs place it.  A localization whose partner lies above it
is below the seam and vice versa, so any value between the two labelled
populations classifies all of them correctly and the midpoint is the pick.  This
is asked of the *fitted* map at the tight tolerance, not of the vote at the
loose one: at 20 px and a few thousand localizations, chance matches reach right
across the frame and it would be their tails being measured.

How wide the consistent range is, is the thing to look at, and it is recorded as
`split_interval_px`.  It is narrow when the two halves see the same field --
what a splitter is for -- and wide when a misalignment leaves a strip of one
half with nothing to pair against.  A wide one warns, and means: set
`split_position` by hand.  Nothing in the pairs can place a boundary inside a
gap where neither channel has a partner.

**Mirrored**, the pairs cannot place it at all.  A seam at `c` with a residual
shift `t`, and a seam at `c + t/2` with no shift, predict exactly the same
pairs; no amount of data separates them.  What the vote does measure is the
mirror line, which *is* the seam for a perfectly aligned splitter and within
half the misalignment of it otherwise.  That is what gets used, and it is
reported as an estimate rather than a measurement.

The split position is written in **chip** coordinates, which is how
`dualfit.which_channel` reads it.

## Reading the diagnostics

Three panels, in the order the questions come:

1. **The vote.**  Did it find the channels at all?  A sharp peak means one
   constant offset explains thousands of pairs.  The contrast number is the
   peak over the map's noise; tens are healthy.
2. **The residual scatter** (`RegisterLocs2`'s `dxy` panel).  Is the map right?
   A round cloud a few hundredths of a pixel across is a good registration; a
   structured one means the projective model is not capturing the distortion.
3. **Coverage.**  Is it right *where I care*?  A projective map is only measured
   where there are pairs, and everywhere else it is extrapolation however small
   the residual looks.  The pairs should span the frame.

## Which transformation

Two models, both global.

**Projective** (8 coefficients) is the default and extrapolates gracefully.

**Polynomial, order three** (20) describes a distorted field far better.  Order
three and not two, for a reason that is an identity rather than a preference:
radial distortion is by definition

    r' = r (1 + k1 r^2 + ...)

which in components is `u' = u + k1 (u^3 + u v^2)`.  The leading term is
**cubic**.  A quadratic polynomial spans `{1, u, v, u^2, uv, v^2}` and contains
no cubic monomial at all, so it cannot describe radial distortion to any
degree -- measured, it is indistinguishable from projective:

| field | projective | poly 2 | poly 3 |
|---|---|---|---|
| clean | 0.0014 px | 0.0024 | 0.0028 |
| barrel 0.01 | 0.307 | 0.298 | **0.0035** |
| barrel 0.03 | 0.862 | 0.880 | **0.026** |
| barrel 0.08 | 2.04 | 2.10 | **0.133** |

The cost is extrapolation.  With pairs confined to the middle of the field, a
cubic is ~85x worse than the projective map outside them.  So it is refused
when the pairs cannot support it -- fewer than ten per coefficient, or covering
less than half the reference channel -- and the projective fit is kept with the
reason said out loud.  The projective fit is written to the file either way, so
a reader that knows nothing of polynomials still gets a usable transformation.

A polynomial has no closed-form inverse, and both directions are needed:
`to_reference` pairs the peaks, `to_secondary` places the partner ROI.  A cubic
is fitted in the reverse direction too -- and is stored -- but it is only the
seed, because a cubic is not the inverse of a cubic: the round-trip error is
0.02 px at 1% distortion and 0.53 px at 5%.  Half a pixel is not affordable
here, since whatever `to_secondary` gets wrong is handed to the fitter as a
link offset as if it were real.  Two Newton steps against the forward map take
it to machine precision; where the Jacobian is singular -- a cubic folding
over, far outside the pairs -- the step is dropped and the fitted seed stands.

## The fit it feeds

`Localize/Gaussian 2D 2C` is `Spline 3D 2C` with the PSF model removed.
Candidates are found over the whole frame and combined, each gets a ROI in both
halves, and the pair goes to `fit_gauss_global` with x and y shared through the
registration -- so the two photon numbers that come back belong to one molecule
and their ratio is its colour, ready for `Analysis/Dual-Color/AssignColors`.

The width is fitted **free per channel** by default.  The two halves of a
ratiometric splitter see different wavelengths and rarely share a focus, and
forcing one width on both would bias the photon split, which is the very thing
being measured.

Beyond that the two workflows share everything: `combine_peaks`, the ROI
cutting and the link array are the same code, because the Gaussian's fifth
parameter (sigma) sits in the same slot as the spline's (z) and the link leaves
that slot alone in both cases.  See `docs/dual_color_calibration.md` for the
bead route, which produces a transformation this fit will also accept.

## Files

A measured transformation is written as `*_2ct.h5`
(`smappy-channel-transform`, version 1): the 3x3 matrix, the geometry, what it
was measured from, and the vote histogram and pair residuals for the plots.
`load_transform()` reads it, and also reads a dual-colour bead calibration
(`*_2c.h5`) and takes just the registration out of it.
