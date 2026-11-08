---
version: "5"
covers: [smappy.calibrate.dual.resolve_arrangement, smappy.microscope_db.Microscope.arrangement, smappy.plugins.registration.chip_pixels, smappy.plugins.registration.default_transform_path, smappy.plugins.registration._draw_vote, smappy.plugins.registration._draw_residuals, smappy.plugins.registration._draw_coverage, smappy.calibrate.transform.register_channels, smappy.calibrate.transform.find_alignment, smappy.channels.ChannelLayout.nominal_map, smappy.plugins.registration.table_channels, smappy.calibrate.transform._pair, smappy.calibrate.transform._refit, smappy.calibrate.transform.pair_weights, smappy.calibrate.transform.fit_polynomial, smappy.calibrate.transform._polynomial_round, smappy.calibrate.transform.save_channel_transform, smappy.calibrate.dual.fit_dual_transform, smappy.calibrate.dual.robust_projective, smappy.calibrate.dual.fit_projective, smappy.calibrate.dual.refine_projective]
---

## What it does

A two-colour [splitter](plugin:Panels/Glossary) (also called a dual-view or
image splitter) images the sample twice, side by side on one camera:
each half of the chip sees the same field through a different filter.  The
two images never lie exactly on top of each other.  The second is shifted,
slightly turned and slightly magnified, and at the corners of the field the
difference is several pixels.  To combine the two halves -- to fit each
molecule in both, as [Gaussian 2D 2C](plugin:Localize/Gaussian 2D 2C) does --
the program needs the **transformation** that maps a position in one half
onto the same position in the other -- the
[registration](plugin:Panels/Glossary) of the two channels, also called
channel alignment, mapping or chromatic correction.

This plugin measures that transformation from localizations, without beads.
In a [ratiometric](plugin:Panels/Glossary) experiment -- dyes told apart by
how their light divides between the halves -- every molecule is imaged in
both halves at once.
Fit the movie with a plain [Gaussian 2D](plugin:Localize/Gaussian 2D) fit
as two channels (*channels* > *read*: *two channels*): each half is then
detected and fitted on its own, and each localization keeps the channel it
was found in, `camera_channel`.  Every molecule shows up twice in the same
frame, once in each channel.  The plugin finds those pairs, and fits the map
that takes one partner onto the other.

It needs no starting values: not the shift and not the magnification.  What
it does need is how the chip is split -- left and right or upper and lower,
whether one half is mirrored, which half is the main one.  That is a fact
about the microscope, and it comes from the microscope's entry in the
[Microscope manager](plugin:Panels/Microscope manager) for the microscope
the table was fitted on; for a microscope without one, choose it (*channels*:
*layout*, *main channel*), in the fit and here.  It is never guessed.  It also needs a
table with `frame` and `camera_channel` attributes and positions that can be
put back into camera pixels, and enough molecules seen in both channels: a
few hundred pairs at the least, from frames spread over the movie.

The 2C fit can do this step itself, on the first frames of the movie
(*registration* > *from this movie*).  Run it here to see what it found before
trusting it, or to save a transformation for movies too sparse to register
on.

## How it works

```figure-setup
import warnings
from smappy.simulate import simulate, dual_transformation
from smappy.calibrate.dual import map_points
from smappy.calibrate.transform import ArrangementSettings, register_channels
from smappy.channels import split_halves
from smappy.plugins.registration import _draw_vote, _draw_residuals, _draw_coverage
# a two-channel fit of a split chip, 100 x 100 pixels a half: every molecule
# is found twice in its frame, in the upper (main) channel and, through the
# simulator's splitter, in the lower one; each channel misses some of them
locs = simulate(n_frames=2000, seed=1)
early = np.asarray(locs["frame"]) < 100
main = np.c_[locs["x_nm"], locs["y_nm"]][early] / 100.0      # chip pixels
sigma = np.asarray(locs["xy_err_nm"])[early] / 100.0
frame = np.asarray(locs["frame"])[early]
truth = dual_transformation(100)                              # lower -> upper
secondary = map_points(np.linalg.inv(truth), main)
rng = np.random.default_rng(2)
seen = [rng.random(len(main)) > 0.15, rng.random(len(main)) > 0.15]
points = np.r_[main[seen[0]], secondary[seen[1]]]
sigma_all = np.r_[sigma[seen[0]], sigma[seen[1]]]
points = points + rng.normal(size=points.shape) * sigma_all[:, None]
frames = np.r_[frame[seen[0]], frame[seen[1]]]
channel = np.r_[np.zeros(seen[0].sum(), int), np.ones(seen[1].sum(), int)]   # camera_channel
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    told = ArrangementSettings(layout="up-down", main_channel="upper")   # the split, as an entry says
    layout = split_halves({"layout": "up-down", "main_channel": "upper"}, (0, 0, 100, 200))
    result = register_channels(points[:, 0], points[:, 1], frames, channel, layout, (200, 100),
                               arrangement=told, precision=sigma_all)
```

**1. Positions in camera pixels.**  A transformation is a property of the
camera, so it is measured in chip pixels.  A table in nanometres is converted
back with the pixel size the fit recorded.  Only the localizations of the
current [selection](plugin:Panels/Glossary) are used -- those that pass the
layer's filter (the lowest and highest values of attributes, set in the Render
tab), inside the drawn ROI (region of interest).

**2. The vote.**  The layout alone says roughly where the second channel's
image of a molecule is: its half moved onto the main one, flipped back if
the optics mirror it.  Every localization of the second channel is put there
first.  Then, for every localization of the main channel and every one of
the second in the same frame, the vector from one to the other is collected.
Most of these vectors connect unrelated molecules and point anywhere.  But
every molecule seen in both channels adds the same vector -- what the
splitter adds to the layout's rough map -- so over many frames that vector
turns up again and again, and stands out in a histogram of all of them as a
sharp peak.  Its position is that offset, found without any guess.

```figure Left: the simulated localizations on the chip; every molecule of the upper channel appears again in the lower one.  Right: the vote, the vectors from each localization of the lower channel, moved up by the layout, to each of the upper one in the same frame -- the ring marks the one offset that thousands of them share.
fig.set_size_inches(7.5, 3.4)
left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.6]})
left.plot(points[:, 0], points[:, 1], ".", ms=1, color="#1f77b4")
left.axhline(100, color="k", lw=1, ls="--")
left.set(xlim=(0, 100), ylim=(200, 0), xlabel="x (chip px)", ylabel="y (chip px)")
left.set_aspect("equal")
_draw_vote(right, result)
for ax in (left, right):
    ax.title.set_fontsize(9)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
```

**3. Pair and fit, twice.**  First, every localization of the second
channel is paired, frame by frame, with the nearest one of the main channel
where the layout and the offset say its partner should be,
within *coarse matching* (20 pixels).  The offset is only a shift; the wide
tolerance is what lets the pairs still be found when the halves are also
turned or scaled.  A **projective** transformation is fitted to these pairs,
robustly, so that the chance pairs a wide tolerance lets in are voted out
(*In detail*).  Then everything is paired again, this time through the
fitted map and within *fine matching* (1 pixel, or wider if the first fit
shows it must be), and the map is fitted again on these clean pairs.  The
second round is where the accuracy comes from.

```figure Left: how far the partners of each pair are apart after the fit.  The first round, paired within 20 px on the offset alone, lets in chance partners (the tail), which the robust fit rejects; the second, paired through the fitted map, keeps clean pairs only.  Right: the error of the final map over the field, against the transformation that was simulated -- a hundredth of a pixel, far below the scatter of a single pair.
fig.set_size_inches(7.5, 2.9)
left, right = fig.subplots(1, 2)
bins = np.geomspace(0.005, 20, 60)
for r, color, name in zip(result.rounds, ("0.5", "#d62728"), ("first round", "second round")):
    left.hist(r.residual_px, bins=bins, histtype="step", lw=1.5, color=color,
              label=f"{name}: {r.n_pairs} pairs, {int(r.accepted.sum())} kept")
left.set_xscale("log")
left.set_yscale("log")
left.set(xlabel="distance after the fit (px)", ylabel="pairs")
left.legend(fontsize=7, frameon=False, loc="upper left")
gx, gy = np.meshgrid(np.linspace(5, 95, 40), np.linspace(5, 95, 40))
grid = np.c_[gx.ravel(), gy.ravel()]
back = result.transform.to_reference(map_points(np.linalg.inv(truth), grid))
error = np.linalg.norm(back - grid, axis=1).reshape(gx.shape)
shown = right.imshow(error, origin="upper", extent=(5, 95, 95, 5), cmap="viridis")
bar = fig.colorbar(shown, ax=right, shrink=0.85)
bar.set_label("map error (px)", fontsize=8)
bar.ax.tick_params(labelsize=7)
right.set(xlabel="x (chip px)", ylabel="y (chip px)", title="error against the truth")
for ax in (left, right):
    ax.title.set_fontsize(9)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
```

**4. Optionally, a polynomial.**  A projective map describes a shift, a
rotation, a scale, a shear and a tilt of the field, but not the curved
distortion a lens can add towards the edges.  With *transformation* set to
*polynomial*, a third-order polynomial is fitted on top of the projective
result, the pairs are matched again through it and it is fitted again.  It is
refused -- the projective map is kept, and the reason given -- when there are
too few pairs or they cover too little of the field, because outside its
pairs a polynomial is not to be trusted.

**5. Saving.**  With *save transformation*, the map, the two channels'
rectangles on the chip and what it was measured from are written to a
`_2ct.h5` file, which the two-colour fit's *transformation* reads.

## In detail

**Which table.**  The localizations, whatever the layer shows: the
pairs are one molecule seen in both channels of the same frame, and the
selection is the layer's bounds applied to each localization.  With several
files selected, a pair is of one frame of one file (their frame numbers are
moved apart, `frames_apart`), so bead files recorded at several positions
pool into one registration.

**The channels** are not measured.  A two-channel fit records the two
rectangles it cut the chip into, and their flips (`metadata["channels"]`);
those are used.  A table without them has the arrangement cut from the
frame: *layout* and *main channel* on *microscope* are read from the entry
of the microscope the table records (`metadata["camera"]["microscope"]`) --
the side each arm lands on, which arm the optics mirror, which is the main
one, and a split position if the entry gives one (`resolve_arrangement`) --
and the frame is cut there, or in its middle.  Without an entry that says,
the run stops and asks for the layout.  Which channel a localization is in
is always the fit's, `camera_channel`; a table without it is refused.  Until
2026-11 the table was a fit of the whole chip, the layout was voted for and
the seam between the halves measured from the pairs; on a wide, low camera
ROI the vote chose the wrong axis, and a seam measured from the pairs could
not be placed under a mirror.

**The vote.**  The second channel's positions are first put through the
layout's nominal map $N$ (`ChannelLayout.nominal_map`): from the chip into
its own rectangle's pixels, flipped back along its mirror, and onto the main
channel's rectangle -- where a molecule's second image would be for a
perfect splitter.  For each frame, every pair $(i, j)$ with $i$ in the main
channel and $j$ in the second gives the vector $p_i - N(p_j)$.  Two
localizations of one channel are never a candidate pair.  Frames are taken
in order until 4 000 000 vectors have been collected.  The vectors are binned
into a histogram $H$ with bins of *vote bin* (2 px), and scored with a
difference of Gaussians

$$S = G_{\sigma} * H - G_{4\sigma} * H ,$$

$\sigma$ being *vote smoothing* (4 px), so that the sharp peak of the pairs
stays and everything broad subtracts away.  The *contrast* is the peak of $S$
over the standard deviation of $S$, and the peak is refined to below a bin by
the centroid of the counts around it.  The first map is $N$ followed by that
shift.  A mirrored arm needs nothing of its own here: the flip is in $N$.

**Pairing.**  Two localizations of one frame, one of each channel, are
partners when each is the other's nearest neighbour and they are within the
tolerance, after the second channel has been mapped onto the main one -- so the tolerance is
a distance in the reference channel in every round.  At most *pairs used in
the fit* (20 000) pairs, drawn at random, go into a fit.

**The projective map.**  With $(x, y)$ a position in the second channel and
$(x', y')$ in the main channel,

$$x' = \frac{h_{11} x + h_{12} y + h_{13}}{h_{31} x + h_{32} y + h_{33}} ,\qquad y' = \frac{h_{21} x + h_{22} y + h_{23}}{h_{31} x + h_{32} y + h_{33}} ,$$

eight free coefficients, since the nine are fixed only up to a common factor.
The linear solution for a set of pairs is the normalised direct linear
transform: both point sets are centred and scaled to a root-mean-square
distance of $\sqrt{2}$ from their centre first, which keeps the solve well conditioned
([Hartley 1997](https://doi.org/10.1109/34.601246)).

**The robust fit**, in each round:

1. **RANSAC** ([Fischler & Bolles 1981](https://doi.org/10.1145/358669.358692)):
   500 times, four pairs are drawn at random and the map through them is
   computed; the draw that brings the most pairs within the *outlier
   threshold* wins (ties go to the smaller truncated squared error).  The
   map is then refitted on its inliers until the inlier set changes by no
   more than 0.2% of the pairs, at most 10 times.
2. **A soft-L1 refinement** of the eight coefficients on those inliers,
   minimising $\sum_k w_k\, \rho(r_k)$ over the x and y misfits $r_k$, with
   $\rho(r) = 2c^2 \left( \sqrt{1 + (r/c)^2} - 1 \right)$ -- quadratic for small
   misfits, linear for large ones -- and $c$ half the *dx/dy limit*.
3. **A screen**: a pair whose misfit in x or in y exceeds the *dx/dy limit*
   is dropped, and the map is fitted again, as in 2, on what is left.

The outlier threshold and the dx/dy limit widen with the round's tolerance,
to at least a half and a quarter of it: 10 and 5 pixels in the first round,
the set values in a tight second round.

**Weights.**  If the table has a [localization
precision](plugin:Panels/Glossary) (also called the CRLB) -- `xy_err_nm`,
converted with the pixel size it records -- a pair's weight is the inverse
variance of its separation,

$$w_k = \frac{1}{\sigma_{\mathrm{ref},k}^2 + \sigma_{\mathrm{sec},k}^2} ,$$

normalised to a mean of one.  A dim channel usually brings many poorly
localized partners; unweighted, they would outnumber the good ones.  RANSAC
itself is unweighted: a precise outlier is still an outlier.

**The second round's tolerance.**  With *adapt_fine_tolerance* on, the
second round is widened to 3 times the 98th percentile of how far the first
fit misses its own accepted pairs, up to the coarse tolerance.  Where a
projective map misfits a distorted field, it misfits most at the edges, and a
fixed tight tolerance would lose exactly those pairs.  If the first fit
accepted less than 60% of its pairs, its misfit comes from chance partners
rather than distortion, and the tolerance is left alone.

**The polynomial.**  Third order: all 10 monomials $u^i v^k$ with
$i + k \leq 3$ in centred, scaled coordinates, for each of x and y -- 20
coefficients.  Order three, because radial distortion is cubic in the
coordinates, and a quadratic cannot represent it at all.  It needs at least
10 pairs per coefficient (200), and the pairs' convex hull must cover at
least half of the reference channel's localizations' hull.  It is a linear
least-squares fit, reweighted 4 times by $1/\sqrt{1 + (r/c)^2}$ against
outliers, with $c$ a quarter of the tolerance (at least 0.05 px); both
directions are fitted, since a polynomial has no closed-form inverse.

**The checks.**  Two warnings are raised on the final pairs: when the median
misfit in the outer half of the paired region is more than twice that in the
inner half (and above 0.1 px), which means the model does not describe the
field; and when the kept pairs cover less than half of the reference channel,
which means the map is an extrapolation over the rest.

## Parameters

### channels.layout
Leave it on *microscope* for a microscope in the database whose entry gives
its arms' sides; choose it for any other.  A wrong layout finds no clear
peak, and the run says so.

### channels.main_channel
The half the other is mapped onto.  For two far-red dyes imaged
ratiometrically it is the transmitted arm, the brighter one; the
microscope's entry names it.

### channels.split_position
Read only for a table that does not record its channels: set it as the fit
had it.  A two-channel fit records its rectangles, and they win; so do the
microscope's arms' regions, where its entry gives them and the layout is the
microscope's.

### registration.model
Stay with projective unless the residual grows towards the edges of the
field (the warning, or the residual panel) and the pairs cover the whole
chip.

### registration.coarse_tolerance_px
The first thing to raise if no pairs are found and the halves are strongly
rotated.  Wider costs more chance pairs, which the robust fit then has to
reject.

### registration.fine_tolerance_px
One pixel is roughly seven times the residual of a good registration.  Much
tighter strips the edges of the field; skipping the second round costs a
factor of three to seven in accuracy.

### registration.adapt_fine_tolerance
See *In detail*.  Off, the second round pairs at exactly *fine matching*.

### registration.max_pairs
Tens of thousands of pairs buy no more accuracy than a well-spread twenty
thousand.

### path
With no source recorded in the table, the automatic name is
`channels_2ct.h5` in the working folder.  Saving refuses to replace an
existing file unless *overwrite* is ticked.

## Output

* **The text**: the layout and which half is the main one; how many pairs were kept of how many, from how many localizations and
  frames; and the residual in x and y, in pixels, with the median misfit in
  the middle of the paired region and at its edge.
* **The figure**, three panels; the *Preview* button draws it without
  writing a file.
* *The pair vote*: a single sharp peak (the ring) against a faint background
  is a found offset.  A smear, or a peak barely above the rest (a low
  *contrast*), means the channels were not paired.
* *The residual*: where each kept pair lands after the fit, the rejected ones
  in red.  A round cloud a fraction of a pixel wide, about the localization
  precision, is good; a stretched or split cloud means the map is wrong.
* *The pairs used*: where in the main channel the pairs are.  The map is
  measured only there; anywhere else it is extrapolated, however small the
  residual.
* **The file**, with *save transformation*: the projective map (always, also
  for a polynomial), the two channels' rectangles and flips on the chip, the numbers of the text, the
  polynomial if one was fitted, and the pairs and the vote for a later look.

```figure The plugin's residual and coverage panels for the simulated splitter: a round cloud about 0.15 px wide, and pairs over the whole of the upper half.
fig.set_size_inches(7.5, 3.4)
left, right = fig.subplots(1, 2)
_draw_residuals(left, result)
_draw_coverage(right, result)
for ax in (left, right):
    ax.title.set_fontsize(8)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
```

## Differences from SMAP

Based on SMAP's `Process/Register/RegisterLocs2`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), whose sequence --
a global alignment, then pairing and refitting at a shrinking tolerance -- it
keeps.

* **The global alignment.**  SMAP renders both halves into images (500 nm
  pixels) and cross-correlates them; the user says where the second half is
  (*target position*) and whether it is mirrored, and the split is the middle
  of the ROI.  Here the offset is voted for on the pair vectors themselves,
  which is the same cross-correlation computed on points.  The layout and the
  mirroring are the microscope's, as SMAP's user states them, and so is the
  split; which channel a localization is in is the fit's, each channel
  detected on its own.
* **The fit.**  SMAP fits the pairs with MATLAB's `fitgeotrans` by plain
  least squares.  Here the fit is robust -- RANSAC, a soft-L1 refinement and
  a dx/dy screen -- and weighted by the localization precision.
* **Models.**  SMAP offers MATLAB's projective, affine, similarity, polynomial
  and local (`lwm`, `pwl`) transformations.  Here: projective, or a
  third-order polynomial that is refused when the pairs do not support it.
* **Scope.**  SMAP can register two files (two cameras) and fits a z
  transformation for 3D data.  Here the two channels are the two halves of
  one chip, in one two-channel table, and the map is 2D.
* **The second round's tolerance** follows the first round's misfit instead
  of a fixed value (SMAP: 1250 nm, then 150 nm).

## References

* Fischler MA, Bolles RC. Random sample consensus: a paradigm for model
  fitting with applications to image analysis and automated cartography.
  *Commun ACM* 24, 381 (1981).
  [doi:10.1145/358669.358692](https://doi.org/10.1145/358669.358692) --
  RANSAC, the robust first fit.
* Hartley RI. In defense of the eight-point algorithm. *IEEE Trans Pattern
  Anal Mach Intell* 19, 580 (1997).
  [doi:10.1109/34.601246](https://doi.org/10.1109/34.601246) -- the
  normalisation before the linear solve.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
