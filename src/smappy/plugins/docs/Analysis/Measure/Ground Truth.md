---
version: "4"
covers: [smappy.plugins.ground_truth.truth_for, smappy.plugins.ground_truth.match, smappy.plugins.ground_truth._near_any, smappy.plugins.ground_truth.compare, smappy.plugins.ground_truth.Comparison, smappy.plugins.ground_truth.GroundTruth.run, smappy.plugins.ground_truth.draw_all, smappy.plugins.ground_truth._draw_recall, smappy.plugins.ground_truth._draw_error, smappy.plugins.ground_truth._draw_pull, smappy.plugins.ground_truth._draw_z, smappy.simulate.camera.camera_truth, smappy.simulate.camera.neighbour_distance]
---

## What it does

On real data nobody knows where the molecules were, so a fit can only be
judged by how the picture looks.  A simulation does know: every fluorophore,
every frame it was on in, how many photons it gave.  Ground Truth uses that
to score a fit of a simulation.  It pairs each localization with the true
spot it came from and reports two things:

* **Detection** -- how many molecules were found, how many localizations
  are false, how many molecules were missed, and the Jaccard index that sums
  the three up.  These are the measures of the SMLM software challenge
  ([Sage et al. 2015](https://doi.org/10.1038/nmeth.3442);
  [Sage et al. 2019](https://doi.org/10.1038/s41592-019-0364-4)).
* **Accuracy** -- for the localizations that were found, how far they are
  from the truth: the systematic offset (bias), the typical error, and
  whether the error each localization *claims* (its precision) is the error
  it really has.

Use it to test a fitter or its settings on data where the answer is known,
to see what a filter on fit quality costs and gains, or to check that the
precision a table reports can be trusted before relying on it in an
analysis.  It is of no use on real data.

**What it needs.**  A table that came from a simulation, in nanometres
(`x_nm`, `y_nm`, `frame`):

* a fit of simulated camera frames -- a `*.sim.yaml` opened in a fitter,
  which writes the file's name into the table as its `source`;
* or a table made directly by
  [Blinking Structure](plugin:File/Simulate/Blinking Structure), which
  carries its simulation settings with it;
* or any table, with the simulation file named under *simulation*.

The truth is not stored with the table.  It is worked out again from the
simulation's settings and seed, which fix it exactly and take about a
second to redraw.  A fit that names a `*.sim.yaml` that has since been moved
is refused with a message saying so; name the file under *simulation*.

The comparison takes a fraction of a second.  For the precision to be
scored, the table needs `x_err_nm` and `y_err_nm`, or `xy_err_nm` for both
(and `z_err_nm` for z).

## How it works

```figure-setup
from smappy.locs import Localizations
from smappy.simulate import (ISOLATED_NM, LocalizationOutputSettings,
                             SimulationSettings, simulate)
from smappy.plugins.ground_truth import (_draw_error, _draw_pull, _draw_recall,
                                         _draw_z, compare, match, truth_for)

# the demo structure, 1000 frames; close emitters are fitted as one spot at
# their mean, as a single-emitter fitter would, so there is crowding to see
settings = SimulationSettings(n_frames=1000, seed=1, localizations=
                              LocalizationOutputSettings(close="average"))
locs = simulate(settings)
truth = truth_for(locs)                   # redrawn from the table's settings
true_photons = np.asarray(truth["photons"])
bright = true_photons >= 100              # the plugin's default: counted
isolated = bright & (np.asarray(truth["neighbour_nm"]) > ISOLATED_NM)
c = compare(locs, truth, 100.0, bright, min_photons=100.0)
ci = compare(locs, truth, 100.0, isolated, max(100.0, ISOLATED_NM / 2),
             min_photons=100.0)

def spoiled(locs, rng, bias_x=4.0, extra_y=1.2, z_scale=0.9, false=0.05):
    """The honest simulated table, made worse in known ways: x shifted by
    `bias_x` nm, y given extra noise of `extra_y` times its precision,
    z compressed by `z_scale`, and a fraction `false` of false localizations
    at random places and frames."""
    c = {k: np.asarray(v).copy() for k, v in locs.columns.items()}
    err = c["xy_err_nm"].astype(float)
    c["x_nm"] = c["x_nm"] + bias_x
    c["y_nm"] = c["y_nm"] + rng.normal(0.0, extra_y * err)
    c["z_nm"] = c["z_nm"] * z_scale
    n = int(false * len(err))
    pick = rng.integers(0, len(err), n)
    extra = {k: v[pick] for k, v in c.items()}
    for key in ("x_nm", "y_nm"):
        extra[key] = rng.uniform(c[key].min(), c[key].max(), n)
    extra["frame"] = rng.integers(0, c["frame"].max() + 1, n)
    return Localizations({k: np.concatenate([c[k], extra[k]]) for k in c},
                         locs.metadata)
```

**1. The truth.**  The simulation is run again, up to the point where
the camera frames would be drawn, and every spot in every frame becomes a
row: where the fluorophore was (drift included, if the simulation had
drift), how many photons it emitted in that frame, and how far away the
nearest other spot on in the same frame was.  Only the frames from the
table's first to its last are used, so a fit of part of a stack is scored
against that part.

**2. Which true spots count.**  A fluorophore that switched on for a
sliver of a frame gave a handful of photons, and no fitter finds it.
Counting it as missed would measure the simulation, not the fit.  So only
true spots with at least *count spots from* photons (100 by default) are
counted.  With *isolated spots only*, only spots with no other spot within
1 µm in their frame count, which measures the fit without crowding.  An ROI
drawn in the render window limits the truth as well as the localizations,
so the score is of what is being looked at.  Every true spot still takes
part in the pairing; the rule only decides which are scored.

**3. Pairing, frame by frame.**  In each frame, localizations and true spots
are paired one to one: each localization with at most one spot, each spot
with at most one localization.  Of all the ways to do that, the one taken
pairs as many as possible and, among those, keeps the total distance
smallest.  No pair may be further apart than *match within* (100 nm); with
*and in z within* set, a pair must also agree that well in z.  Matching on
the least total distance matters where spots are close: taking the nearest
pair first can leave a second localization with no partner that it would
otherwise have had.

```figure The crowded part of one simulated frame.  Circles: true spots, drawn with the 100 nm radius a partner must lie within.  Blue dots: localizations, joined to their partner.  Where two or three emitters were on close together they were fitted as one localization between them: it pairs with one of them, and the others count as missed (red).  The orange cross is a false localization added by hand, with no spot near it.
merged = np.flatnonzero(np.asarray(locs["n_merged"]) > 1)
fx, fy = np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])
tx, ty = np.asarray(truth["x_nm"]), np.asarray(truth["y_nm"])
lf, tf = np.asarray(locs["frame"]), np.asarray(truth["frame"])
# the merged localization with the most true spots near it
near = [np.sum((tf == lf[i]) & (np.hypot(tx - fx[i], ty - fy[i]) < 1200))
        for i in merged[:400]]
centre = merged[:400][int(np.argmax(near))]
frame, cx, cy = lf[centre], fx[centre], fy[centre]
fi = np.flatnonzero(lf == frame)
ti = np.flatnonzero(tf == frame)
fxy = np.column_stack([fx[fi], fy[fi]])
fxy = np.vstack([fxy, [cx + 700.0, cy - 650.0]])          # a false one
txy = np.column_stack([tx[ti], ty[ti]])
pf, pt = match(fxy, np.zeros(len(fxy)), txy, np.zeros(len(txy)), 100.0)
fig.set_size_inches(4.6, 4.6)
ax = fig.subplots()
from matplotlib.patches import Circle
counts = true_photons[ti] >= 100
for k, (x, y) in enumerate(txy):
    colour = "black" if counts[k] else "0.65"
    if counts[k] and k not in pt:
        colour = "#d62728"
    ax.add_patch(Circle((x, y), 100.0, fill=False, lw=1.2, color=colour))
for a, b in zip(pf, pt):
    ax.plot([fxy[a, 0], txy[b, 0]], [fxy[a, 1], txy[b, 1]], color="#1f77b4", lw=1)
real = np.arange(len(fxy)) < len(fi)
ax.plot(fxy[real, 0], fxy[real, 1], "o", color="#1f77b4", ms=4)
ax.plot(fxy[~real, 0], fxy[~real, 1], "x", color="#ff7f0e", ms=8, mew=2)
ax.set_xlim(cx - 1100, cx + 1100); ax.set_ylim(cy - 1100, cy + 1100)
ax.set_aspect("equal")
ax.set_xlabel("x (nm)", fontsize=8); ax.set_ylabel("y (nm)", fontsize=8)
ax.tick_params(labelsize=7)
ax.set_title(f"frame {frame}", fontsize=9)
```

**4. Found, false, missed.**  A pair whose true spot counts is a **found**
molecule (a true positive).  A localization with no partner is **false** (a
false positive), and a counted spot with no partner is **missed** (a false
negative).  Some localizations are neither right nor wrong and are **set
aside**: those paired with a spot that does not count, and those left
without a partner that lie next to such a spot or are themselves dimmer than
*count spots from*.  Without this, raising the photon threshold would turn
good fits of dim spots into false positives.

**5. The Jaccard index.**  Detection in one number: the found molecules
over everything that is found, false or missed.  It is 1 when every counted
molecule was found and nothing else, and it falls with false localizations
and missed molecules alike.  It is what a filter on fit quality should be
judged by: tightening the filter removes false localizations and
also found ones, and the Jaccard index says whether the trade paid.

```figure The fraction of true spots found, by the photons they emitted in the frame (the first panel of the plugin's figure), scored on every spot from 100 photons (left) and on isolated spots only (right).  In this dense simulation the dim spots are missed most often, and it is crowding that misses them: two emitters on close together were fitted as one spot at their photon-weighted mean, nearer the brighter one, which takes the pair.  Isolated spots are found whatever their brightness.
fig.set_size_inches(7.5, 2.6)
left, right = fig.subplots(1, 2, sharey=True)
_draw_recall(left, c)
_draw_recall(right, ci)
left.set_title("all counted spots: " + left.get_title(), fontsize=8)
right.set_title("isolated only: " + right.get_title(), fontsize=8)
for ax in (left, right):
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
```

**6. How far off.**  For every found molecule the error is the fitted
position minus the true one, per axis.  Its mean is the **bias**, a
systematic shift that a good fit does not have.  Its root mean square is the
**RMS error**, the typical distance from the truth, bias included.

**7. Is the precision honest?**  Every localization carries a precision,
the error the fit believes it has (`x_err_nm`, `y_err_nm`, `z_err_nm`).  Dividing each
error by its own precision gives a number that should scatter like a
standard normal distribution -- spread 1 -- when the precisions are right.
The plugin reports that spread as **error / reported precision**.  Above 1,
the localizations are worse than they claim, and anything that weights by
precision (grouping, a line profile, a drift correction) trusts them too
much.  Below 1, they are better than they claim.  The spread is measured
robustly, so a few wrong pairs do not change it.  A bias widens it too,
since a shift of a few nanometres is several times the precision of a
bright spot and a fraction of a dim one's.

**8. The z scale.**  With z in both, fitted z is plotted against true z and
a straight line fitted.  Its slope is the scale of the axial calibration: 1
is right, below 1 the fitted z range is compressed.  Crowded spots pull
astigmatic z towards the focus and so lower the slope, which is one reason
to look at isolated spots too.

```figure The plugin's figure for the simulated table spoiled in known ways, scored on isolated spots: x shifted by 4 nm, extra noise on y of 1.2 times the precision (so a spread of $\sqrt{1 + 1.2^2} \approx 1.56$), z compressed to 90%, and 5% false localizations at random places.  Top right: the measured error lies above the reported precision, most of all for the bright spots, whose precision is smallest next to the 4 nm shift.  Bottom left: the x errors are shifted, the y errors too wide, and the z errors widened by the compression.  Bottom right: z on a slope of 0.9.  The false localizations lower the Jaccard index below the recall, and the extra noise takes a few of the dimmest spots beyond 100 nm, where they count as missed.
rng = np.random.default_rng(0)
bad = spoiled(locs, rng)
cb = compare(bad, truth, 100.0, isolated, max(100.0, ISOLATED_NM / 2),
             min_photons=100.0)
fig.set_size_inches(7.5, 5.4)
axes = fig.subplots(2, 2).ravel()
_draw_recall(axes[0], cb)
_draw_error(axes[1], cb)
_draw_pull(axes[2], cb)
_draw_z(axes[3], cb)
for ax in axes:
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
fig.subplots_adjust(hspace=0.5, wspace=0.3)
```

**9. Crowding, and isolated spots.**  Two molecules on close together in
one frame are fitted as one spot, or twice with each fit pulled towards the
other.  Their errors are larger than their precision says.  With *isolated
spots only*, the score is of the spots that had no neighbour within 1 µm,
and localizations near the crowded ones (within 500 nm, or *match within*
if that is larger) are set aside.  Comparing the two scores separates what
the fitter does on a clean spot from what crowding does to it.

```figure Error over reported precision in x and y for the same fit, scored on every spot from 100 photons (left) and on isolated spots only (right).  The localizations fitted between two close emitters widen the distribution on the left; the isolated spots scatter as the unit Gaussian (dashed).
import copy
lateral = []
for c_ in (c, ci):
    c_ = copy.copy(c_)
    c_.pairs = {k: v for k, v in c_.pairs.items() if k != "pull_z"}
    lateral.append(c_)
fig.set_size_inches(7.5, 2.5)
left, right = fig.subplots(1, 2, sharey=True)
_draw_pull(left, lateral[0])
_draw_pull(right, lateral[1])
left.set_title("all counted spots", fontsize=9)
right.set_title("isolated spots only", fontsize=9)
for ax in (left, right):
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_fontsize(8)
```

## In detail

**The truth table.**  The simulation settings come from the file named in
*simulation*, else from the `*.sim.yaml` in the table's `source`, else from
the `simulation_settings` a simulated table carries.  From them the ground
truth is redrawn (`simulate.ground_truth`) and turned into one row per spot
per frame (`camera_truth`): `x_nm`, `y_nm`, `z_nm` including the simulated
drift, `photons` emitted in that frame (the expected number, before shot
noise), and `neighbour_nm`, the distance to the nearest other spot in the
same frame.  With *drift corrected*, the simulated drift of each spot's
frame is subtracted from its position, the table is matched against that,
the mean offset of the pairs in each axis is taken out of the truth as well,
and the table is matched again.

**Which spots count.**  True spot $j$ counts when

$$f_{min} \leq f_j \leq f_{max}, \qquad N_j \geq N_{min}, \qquad \mathbf{r}_j \in \mathrm{ROI}, \qquad \left( d_j > 1000\ \mathrm{nm} \right) ,$$

with $f_{min}$ and $f_{max}$ the first and last frame of the whole table,
$N_j$ the photons it emitted in its frame, $N_{min}$ *count spots from*, and
the last condition, on the neighbour distance $d_j$, only with *isolated
spots only*.  The layer's filter applies to the localizations only.

**Pairing.**  In each frame that has both, the candidate pairs are those
within the radius $R$ (*match within*), found with a k-d tree, and, with a
z radius $R_z$, also with $|z_i - z_j| \leq R_z$ (applied only when both
tables have `z_nm`).  The cost of a candidate is its lateral distance
$\sqrt{\Delta x^2 + \Delta y^2}$; every other pairing costs $10^{12}$.
The assignment of least total cost (`scipy.optimize.linear_sum_assignment`,
exact) is taken and the pairs of forbidden cost are dropped.  Because a
forbidden pair costs more than any sum of allowed ones, this is the largest
set of allowed pairs, and of those the one of least total distance.  A frame
with one localization and one spot pairs them if they are within reach.

**The counts.**  With $M$ the pairs, $C$ the counted spots:

* found: $TP = |\{(i, j) \in M : j \in C\}|$;
* missed: $FN = |C| - TP$;
* set aside: the pairs whose spot $j \notin C$, plus every unpaired
  localization that has a spot not in $C$ within $\rho$ in its frame, or has
  fewer than $N_{min}$ photons (when the table has `photons`).  $\rho$ is
  $R$, or $\max(R, 500\ \mathrm{nm})$ with *isolated spots only*;
* false: $FP$, the unpaired localizations not set aside.

A localization that *is* paired with a counted spot is found, however dim it
was fitted.  Then

$$\mathrm{recall} = \frac{TP}{|C|}, \qquad \mathrm{correct} = \frac{TP}{TP + FP}, \qquad J = \frac{TP}{TP + FP + FN} .$$

The text reports the false fraction, $1 - \mathrm{correct}$; "correct" is
what the SMLM challenge calls precision, a word kept here for the
localization precision.

**Accuracy.**  Over the $TP$ found pairs, with $\delta_i = \hat{x}_i - x_i$
the fitted minus the true coordinate:

$$\mathrm{bias} = \frac{1}{TP} \sum_i \delta_i, \qquad \mathrm{RMS} = \sqrt{\frac{1}{TP} \sum_i \delta_i^2} .$$

The RMS error therefore contains the bias:
$\mathrm{RMS}^2 = \mathrm{bias}^2 + \mathrm{variance}$.  x is divided by
`x_err_nm`, y by `y_err_nm` and z by `z_err_nm`, for the pull $p_i = \delta_i /
\sigma_i$ (pairs with $\sigma_i \leq 0$ left out), and the reported spread
is the scaled median absolute deviation

$$s_p = 1.4826\ \mathrm{median}_i \left| p_i - \mathrm{median}_j\, p_j \right| ,$$

which is the standard deviation for a Gaussian, ignores a few wrong pairs,
and is unmoved by a shift common to all $p_i$ -- though not by a bias in
nanometres, which is a different $p$ for every precision.  Each axis by its
own precision, because an astigmatic spot is narrower in x than in y on one
side of focus and the other way round on the other: divided by `xy_err_nm`,
the RMS of the two, a spline fit to frames drawn from its own PSF read an x
pull of 0.86 to 0.96.  A table with `xy_err_nm` only has both axes divided
by it.  An axis without a precision column has bias and RMS only.

**The z slope.**  With more than 10 found pairs, `z_nm` in both tables and
true z not all equal, the least-squares line $\hat{z} = a z + b$ gives the
slope $a$ and the offset $b$.

**The figures.**  The recall panel bins the counted spots' true photons in
24 logarithmic bins from the dimmest (at least 1 photon) to the brightest,
and draws the fraction found in each non-empty bin.  The error panel,
drawn with at least 20 found pairs, splits them into ten equal-sized groups
by their *fitted* photons and draws, per group, the measured error per axis
$\sqrt{\langle (\delta_x^2 + \delta_y^2)/2 \rangle}$ against the median
reported precision per axis, $\sqrt{(\sigma_x^2 + \sigma_y^2)/2}$ (which is
`xy_err_nm`).  The pull panel has 60 bins from $-5$ to $5$, with
the standard normal density dashed.  The z panel is a 60 by 60 histogram of
fitted against true z, with the identity dashed and the fitted line drawn
over the 0.5th to 99.5th percentile of true z.

**Compared with the challenge.**  The SMLM challenge
([Sage et al. 2019](https://doi.org/10.1038/s41592-019-0364-4)) also pairs
frame by frame and scores recall, precision, the Jaccard index and the RMS
error.  Its assessment differs from this one in four places:

* It pairs by a presorted nearest-neighbour search within 250 nm; here the
  exact optimal assignment is taken, within 100 nm by default and, if asked,
  within a z radius as well.
* It keeps the brightest 75% of the true activations; here a spot counts
  from a fixed number of emitted photons.
* It leaves out the fluorophores within 450 nm of the border of the field;
  here none is.
* It corrects the depth-dependent lateral shift of its experimental PSFs
  (the "wobble") before comparing; here such a shift shows in the bias and
  the error.

## Parameters

### truth
Needed for a table fitted from the TIFF a simulation also wrote (its
`source` is the TIFF, which holds no truth), and for a fit whose simulation
file has since moved.

### radius_nm
A few times the localization precision, and well below the distance between
spots in a frame.  Too small, and a poorly fitted spot becomes one false and
one missed; too large, and a false localization is paired with a missed
spot nearby and counted as found, with a large error.

### z_radius_nm
For 3D data, where a spot fitted at the wrong z -- on the wrong side of
focus, say -- would otherwise count as found and only show as a large z
error.

### min_photons
It compares the *true* photons of a spot, what the fluorophore emitted in
the frame.  The fitted photons of a localization only decide whether an
unpaired one is false or set aside, so a false localization dimmer than the
threshold is never counted as false: a fitter that makes many dim false
detections looks clean at 100.  At 0 every spot counts and every unpaired
localization not beside an uncounted spot is false; the recall figure shows
from how many photons on the fit finds its spots.

### isolated
1 µm is the distance the simulation calls isolated: nearer, a neighbour's
light falls into a fit's 13-pixel box.

### drift_corrected
The truth includes the simulated drift, and a table that has been drift
corrected no longer does; tick it after a drift correction.  A drift
correction measures drift only up to a constant -- RCC and COMET make it
average zero over the acquisition, while the simulated drift starts at zero
-- so a corrected table sits a constant offset from the truth that nothing
in it can tell.  That offset is measured from the pairs, taken out, and
reported; the bias of a drift-corrected table is therefore zero by
construction and says nothing, while the error / reported precision above 1
that remains is the drift correction's own error.

## Output

* **The text** gives, line by line: the localizations compared, the true
  spots counted (and drawn in all) and the radius; found (with the recall),
  false (with the fraction of the scored localizations that are false) and
  missed, and the Jaccard index; the number set aside, if any; per axis the
  bias, the RMS error and the error / reported precision; and, with z, the
  slope and offset of fitted against true z.
* **The figure** (the *Plot* button): the recall by photons; the measured
  error against the reported precision, by photons; the distribution of
  error over reported precision per axis, with the spread in the legend;
  and, with z, fitted against true z.

A good fit of an honest simulation has a recall and a Jaccard index near 1
on the isolated spots, biases well below a nanometre, an error / reported
precision near 1 (the measured-error points on the precision line) and a z
slope near 1.  A bias of tens of nanometres usually means the table and the
truth disagree about something other than the fit: a drift correction
without *drift corrected*, or a pixel convention.  An error / reported
precision well above 1 on isolated spots means the fitter's precision
cannot be trusted; if it is near 1 there and above 1 on all spots, the
excess is crowding.

## Differences from SMAP

Does the job of SMAP's `other/CompareToGroundTruth`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), which compares two
loaded layers, one of which is the ground truth.  Here:

* **Pairing is the optimal assignment.**  SMAP's `matchlocshd` takes the
  candidate pairs of a frame nearest first, skipping any whose partner is
  taken, which can pair fewer spots in a crowded frame.
* **The z radius makes or breaks a pair.**  SMAP pairs laterally, counts
  every pair as a true positive, and leaves pairs more than 300 nm apart in
  z out of the errors only.
* **Not every true spot counts.**  SMAP scores every localization of the
  truth layer; the photon threshold, *isolated spots only* and the
  localizations set aside are new here.
* **Bias, error and pull are computed directly.**  SMAP fits a Gaussian to
  the histogram of the errors and reports its centre, the lateral RMS error
  after taking that shift out, and the fitted width of the normalised
  errors.  Here the bias is the mean, the RMS error includes it, per axis,
  and the pull's spread is the robust MAD.

## References

* Sage D, Pham T-A, Babcock H, et al. Super-resolution fight club:
  assessment of 2D and 3D single-molecule localization microscopy software.
  *Nat Methods* 16, 387 (2019).
  [doi:10.1038/s41592-019-0364-4](https://doi.org/10.1038/s41592-019-0364-4)
  -- one-to-one pairing within a tolerance, and the Jaccard index, as the
  measures of detection.
* Sage D, Kirshner H, Pengo T, et al. Quantitative evaluation of software
  packages for single-molecule localization microscopy. *Nat Methods* 12,
  717 (2015).
  [doi:10.1038/nmeth.3442](https://doi.org/10.1038/nmeth.3442)
* Kuhn HW. The Hungarian method for the assignment problem. *Naval Res
  Logist Q* 2, 83 (1955).
  [doi:10.1002/nav.3800020109](https://doi.org/10.1002/nav.3800020109)
  -- the assignment problem that the pairing solves.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
