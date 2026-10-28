---
version: "5"
covers: [smappy.drift.estimate_drift, smappy.drift._estimate_spline, smappy.drift._spline_basis, smappy.drift._two_stage, smappy.drift._rcc_prepass, smappy.drift._grouped, smappy.drift.merge_from_session, smappy.drift.estimate_cost, smappy.drift._sigma_levels, smappy.drift.Drift.apply, smappy.drift.Drift.referenced, smappy.drift.Drift.then, smappy.drift.Drift.parameters, smappy.drift.Drift.noise_nm, smappy.drift.Drift.insignificant_axes, smappy.drift.split_halves, smappy.drift.precision_variances, smappy.drift._TransformationFit, smappy._comet.core.drift_optimizer.comet_run_kd, smappy._comet.core.drift_optimizer.optimize_3d_chunked_better_moving_avg_kd, smappy._comet.core.cpu_wrapper.cpu_wrapper_chunked_approx, smappy._comet.core.segmenter.segmentation_wrapper, smappy._comet.core.interpolation.interpolate_drift, smappy._comet.core.qc_utils.flag_flawed_segments_by_lift]
---

## What it does

Over a long acquisition the sample drifts by tens to hundreds of nanometres,
and every localization is off by however far it had moved in its frame (the
[RCC](plugin:Analysis/Drift/RCC) page says more about where drift comes from
and what it does to the image).  COMET
([Reinkensmeier et al. 2026](https://doi.org/10.64898/2026.03.27.714864))
measures that drift from the localizations themselves -- no beads, no fiducial markers -- and subtracts it.

It does not make images.  It asks a simpler question of the localizations
directly: if the drift were known and taken out, a molecule seen early and
the same structure seen late would sit on top of each other.  So COMET looks
for the drift curve that makes localizations from different times **overlap
as much as possible**, and moves every localization by that curve.  A
precise localization counts for more in that overlap than an imprecise one,
as in [DME](plugin:Panels/Glossary)
([Cnossen et al. 2021](https://doi.org/10.1364/OE.426620)), so
out-of-focus and dim localizations pull little on the result.

Every run also says how far to trust the curve, with a
[split-half error bar](plugin:Panels/Glossary): it estimates the drift
again from each half of the field and draws both halves beside the curve.
Where they differ is noise.  When the curve is no larger than that noise
the text says so: there is then no drift to correct, and correcting it would
only add noise.

Use it on **fixed** samples: anything that moves by itself is taken for
drift.  It needs a few thousand localizations in the selection at the least
(it warns below 5000), and does better with many more.  The drift is
estimated from the [selection](plugin:Panels/Glossary) -- the localizations
that pass the layer's filter (the lowest and highest values of columns such
as precision, set in the Render tab), inside the drawn
[ROI](plugin:Panels/Glossary) (region of interest) if there is one -- and
subtracted from **every** localization of the table,
so a clean filter (bright, well-fitted localizations) helps the estimate and
costs the corrected table nothing.

A sample that **changes shape** during imaging -- it expands, or creeps
differently in different places -- or optics whose magnification drifts
with temperature or focus, cannot be corrected by one shift per frame: a
correction that is right in the middle of the field is wrong at its edges.
For these, *transformation* fits a change of scale, a rotation, an
[affine](plugin:Panels/Glossary) transformation of x and y (with a scale
of z of its own if wanted) or a full 3D one over time alongside the
drift.  The default is the shift alone.

In the comparisons made while it was developed, COMET was slightly the more
precise of the two drift plugins, but it can take minutes where RCC takes
seconds, and on a large dataset much longer.  It says what a run will cost before it starts.  The two
share no code and almost no assumptions, so running both is a useful second
opinion.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.drift import DriftSettings, estimate_drift, _grouped, _positions_nm
from smappy.locs import Localizations
from smappy._comet.core.pair_indices import pair_indices_kdtree
from smappy._comet.core.cpu_wrapper import cpu_wrapper_chunked_approx
locs = simulate(n_frames=10000, seed=1, drift=True)
truth = np.array(locs.metadata["drift_truth"])
# the default estimate, with its error bar, for the figures that show it
spline = estimate_drift(locs, DriftSettings())
settings = DriftSettings(split_half=False)
grouped = _grouped(locs, settings, None)
x, y, z = _positions_nm(grouped, None, None)
frame = np.asarray(grouped["frame"])
```

**1. Blinks become molecules.**  A fluorophore that is on for five frames
gives five localizations of the same molecule.  They are first merged into
one, at their precision-weighted mean position (*merge blinks*): [merge
blinks](plugin:Panels/Glossary), also called grouping or linking, combines
the localizations in consecutive frames that come from one blinking event
into one localization.  This
removes the densest, shortest-range pairs of localizations, which makes the
estimate much faster, and stops one bright molecule from counting as five
independent measurements.

**2. Neighbours.**  Every pair of localizations closer than *max drift* is
found once, at the start.  Only these pairs are ever compared: two
localizations further apart than the largest drift cannot be the same
structure seen at two times.

**3. Overlap.**  Each localization is treated as a small Gaussian blob of
width $\sigma$.  For a trial drift curve, every localization is moved back by
the drift of its own frame, and the overlap of each pair of blobs is added
up.  The overlap is largest when the drift curve is right, because then
localizations of the same structure land on top of each other whatever time
they were recorded at.

With *precision weights* (on by default) each blob is also as wide as its
localization is uncertain, in each axis on its own: its
[localization precision](plugin:Panels/Glossary) is added to $\sigma$, and a blob keeps
the same total amount however wide it is.  Two precise localizations of a
molecule then overlap strongly, and a dim or out-of-focus one -- wide, and
in z often three times wider than in x and y -- spreads its little overlap
thinly and pulls little on the curve.  It is what used to need a hand-set
filter on z.

```figure Out-of-focus fits are often held at the edge of the z range the fit covers, whatever the drift: here 40% of the simulation's localizations were moved to z = ±400 nm and given the poor precision such fits have (30 nm laterally, 200 nm in z).  Drawn is how far the z drift found with them lies from the z drift found without them.  Counted like any other (COMET as published, orange) they hold the axial drift back by up to a few nm where it is largest; weighted by their precision (blue) they hardly count.
rng = np.random.default_rng(3)
cols = {k: np.array(v, dtype=float) if k.endswith("nm") else np.array(v)
        for k, v in locs.columns.items()}
f = np.asarray(cols["frame"], int)
bad = rng.random(len(f)) < 0.4
cols["z_nm"][bad] = np.where(cols["z_nm"][bad] - truth[f[bad], 2] >= 0, 400.0, -400.0)
cols["xy_err_nm"][bad], cols["z_err_nm"][bad] = 30.0, 200.0
pinned = Localizations(cols, locs.metadata)
fig.set_size_inches(6.5, 2.4)
ax = fig.subplots()
clean = spline.drift[:, 2]
for weights, color, label in ((False, "#ff7f0e", "every localization alike"),
                              (True, "#1f77b4", "weighted by precision")):
    found = estimate_drift(pinned, DriftSettings(split_half=False,
                                                 precision_weights=weights)).drift[:, 2]
    n = min(len(found), len(clean))
    ax.plot(np.arange(n), (found[:n] - found[:n].mean()) - (clean[:n] - clean[:n].mean()),
            color=color, lw=1.2, label=label)
ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
ax.set_xlabel("frame", fontsize=8)
ax.set_ylabel("z drift, difference (nm)", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, frameon=False)
```

**4. Coarse to fine.**  With small blobs the overlap has many small peaks,
and an optimiser started far from the answer would stop at the nearest one.
So the search starts with wide blobs, $\sigma$ a third of *max drift*, where
the overlap has one broad maximum near the answer, and narrows them by a
factor 1.5 at a time down to *target sigma*, each step starting from the
last one's answer.

```figure The overlap between the first and the last 1000 frames of a simulated acquisition, as the last ones are shifted in x (the other axes held at their true values).  At every width the maximum sits at or near the true drift between the two (dashed); wide blobs give a broad hill that is easy to climb from far away, narrow ones a sharp peak that pins the answer down.
early, late = frame < 1000, frame >= 9000
keep = early | late
coords = np.ascontiguousarray(np.column_stack([x, y, z])[keep], dtype=np.float32)
window = np.ascontiguousarray(late[keep].astype(np.int32))
idx_i, idx_j, _ = pair_indices_kdtree(coords, settings.max_drift_nm)
true_shift = truth[9000:].mean(0) - truth[:1000].mean(0)
shifts = np.linspace(-300, 300, 121)
fig.set_size_inches(6.5, 2.8)
ax = fig.subplots()
sigma = settings.max_drift_nm / 3
colors = iter(("#9ecae1", "#6baed6", "#3182bd", "#08519c"))
while True:
    cost = []
    for dx in shifts:
        mu = np.array([[0.0, 0.0, 0.0], [dx, true_shift[1], true_shift[2]]])
        value, _ = cpu_wrapper_chunked_approx(mu, coords, window, idx_i, idx_j, sigma, 1.0)
        cost.append(-value)
    cost = np.array(cost)
    ax.plot(shifts, (cost - cost.min()) / (cost.max() - cost.min()),
            color=next(colors), lw=1.5, label=f"$\\sigma$ = {sigma:.0f} nm")
    if sigma <= settings.target_sigma_nm:
        break
    sigma = max(sigma / 1.5, settings.target_sigma_nm)
ax.axvline(true_shift[0], color="0.3", ls="--", lw=1)
ax.set_xlabel("shift of the last frames in x (nm)", fontsize=8)
ax.set_ylabel("overlap (scaled)", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, frameon=False)
```

**5. A smooth curve in time.**  By default (*fit spline*) the drift is a
smooth curve with one adjustable point every *knot spacing* frames (2000),
and the overlap is maximised over those points directly.  Every localization
is moved by the drift of its own frame; there are no time windows.  Drift is
slow creep, so a few tens of numbers describe an acquisition of tens of
thousands of frames, and each is constrained by all the localizations near
it in time.

With *fit spline* off, the method is COMET as published: the acquisition is
cut into **time windows** (*window size*, 500 frames by default), one drift
is fitted per window, and a curve through the windows' values gives the drift
of every frame.  This follows faster changes and can check each window
(*quality control*), at the price of a noisier curve.

```figure The drift estimated from the simulation, against the drift that was put in (grey, with the frame-to-frame jitter of the simulation).  The spline at its default 2000-frame knots (blue) follows the slow course of the drift and smooths over wiggles shorter than its knot spacing; the time-window fit (orange, dashed) follows more of them and is noisier.  Only differences between frames are measured, so the curves are compared after removing their mean.
windows = estimate_drift(locs, DriftSettings(spline=False, split_half=False))
fig.set_size_inches(7.5, 2.6)
axes = fig.subplots(1, 3)
frames = np.arange(len(truth))
for axis, ax in enumerate(axes):
    ax.plot(frames, truth[:, axis] - truth[:, axis].mean(), color="0.75", lw=2.5,
            label="simulated")
    for drift, color, ls, label in ((windows, "#ff7f0e", "--", "time windows"),
                                    (spline, "#1f77b4", "-", "spline")):
        found = drift.drift[:len(truth), axis]
        ax.plot(frames, found - found.mean(), color=color, ls=ls, lw=1.2, label=label)
    ax.set_title("xyz"[axis], fontsize=9)
    ax.set_xlabel("frame", fontsize=8)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("drift (nm)", fontsize=8)
axes[0].legend(fontsize=7, frameon=False)
```

**6. Correcting.**  The drift of each frame is subtracted from `x_nm`,
`y_nm` and `z_nm` (or the pixel columns) of every localization in that
frame, whether or not it was selected.

**7. The error bar.**  With *error bar* on (the default), the field is cut
in two at its middle, and the drift is estimated again from each half on
its own.  The two halves hold different molecules, so whatever they
disagree on is chance: the noise of the curve.  Both are drawn dashed beside
the curve, and the text gives the noise in nm.  Where the curve moves no
more than about its noise, there is no drift to speak of.

```figure Left: the simulation's drift as estimated (solid), with the estimates from each half of the field (dashed); they agree to about a nanometre, which is the noise the text reports.  Right: the same on a simulation with no drift at all.  The curve still wanders by a few nm -- that is its noise, as the halves show -- and the text says that there is no drift above the noise.
still = estimate_drift(simulate(n_frames=10000, seed=1), DriftSettings())
fig.set_size_inches(7.5, 2.6)
left, right = fig.subplots(1, 2)
spline.plot(left)
still.plot(right)
for ax, title in ((left, "drift put in"), (right, "no drift")):
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.legend(fontsize=7, frameon=False)
```

**8. A sample that changes size.**  With a *transformation*, each frame
also gets a small linear transformation about the middle of the data -- a
change of scale, a rotation, or the full affine one -- and its parameters
are smooth curves in time too, with the drift's knots or, with
*transformation knots*, their own.  Every localization is moved by the
transformation and the drift of its own frame, and the overlap is
maximised over both at once.  The overlap of a pair is measured in units
of the two frames' own scale: otherwise shrinking everything towards the
middle would make every pair overlap more, and the fit would collapse the
image.

```figure A simulated sample that grows steadily by 1% (10 000 ppm) over the acquisition, about the middle of its 9 µm field: its edges move outwards by 45 nm while it also drifts.  The scale change COMET finds with *transformation* "xy scale" (blue) follows the one put in (grey), within a few hundred ppm -- a nanometre or two at the edge of the field -- and least well at the two ends, where the spline has data on one side only.  Like the drift, only changes are measured, so both are drawn about their mean.
frame_all = np.asarray(locs["frame"])
grow = 1e-2 * frame_all / frame_all.max()
xy = np.column_stack([np.asarray(locs["x_nm"], float), np.asarray(locs["y_nm"], float)])
middle = (xy.min(0) + xy.max(0)) / 2
xy = middle + (xy - middle) * (1 + grow)[:, None]
grown = Localizations(dict(locs.columns, x_nm=xy[:, 0], y_nm=xy[:, 1]), locs.metadata)
found = estimate_drift(grown, DriftSettings(transformation="xy scale"))
scale = found.parameters()["scale"] * 1e6
put_in = 1e4 * found.frames / frame_all.max()
fig.set_size_inches(6.5, 2.6)
ax = fig.subplots()
ax.plot(found.frames, put_in - put_in.mean(), color="0.75", lw=2.5, label="simulated")
ax.plot(found.frames, scale - scale.mean(), color="#1f77b4", lw=1.2, label="found")
ax.set_xlabel("frame", fontsize=8)
ax.set_ylabel("scale change (ppm)", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, frameon=False)
```

**Before it starts**, the plugin estimates in a fraction of a second how long
the run will take and how much memory it needs.  Both follow from the number
of neighbour pairs, which grows with the square of the number of
localizations and steeply with *max drift*.  If the answer is more than five
minutes, or more than half the computer's memory, it asks first, and offers
an alternative: RCC over a few time windows to take out the bulk of the
drift in seconds, then COMET within 50 nm of what is left.

## In detail

**Which table.**  The localizations, whatever the layer shows.  The
estimate is made from those the layer's filter and the ROI select -- the
bounds applied to each localization, also on a layer that merges blinks -- and the
correction is applied to every localization of the file, selected or not.

**Several files.**  Files opened together keep their own frame numbers, so
one curve over two of them would average frame 100 of one with frame 100
of the other.  Each file the selection holds gets a drift curve of its own,
estimated from its localizations and subtracted from its localizations
only.  A file the selection does not hold (the layer shows only the
others) is left as it was.  The result then draws a curve per file, under
its name.  To correct pieces of one acquisition with one curve, make them
one file first with [Combine files](plugin:Analysis/Process/Combine files).

The blinks the estimate merges for itself (*merge blinks*) are merged with
the session's merge settings -- the distance and dark frames of the Render
tab's *merge settings...* -- unless *within* and *dark frames* are set here;
a session that merges by precision lends its largest distance, *at most*.
Without a session (a script) they are 50 nm and 1 frame, the session's own
defaults.  The merged blinks are thrown away afterwards.

**The cost.**  With $\mathbf{r}_i$ the position of localization $i$ (x, y
and z, with z zero for a 2D table), $t_i$ its frame, $\mathbf{d}(t)$ the
drift and $\Delta_k$ axis $k$ of the corrected separation
$(\mathbf{r}_i - \mathbf{d}(t_i)) - (\mathbf{r}_j - \mathbf{d}(t_j))$, the
overlap is

$$C = \sum_{(i,j)} \frac{1}{\sigma} \sqrt{\prod_k \frac{2 s_k^2}{V_{ij,k}}}\, \exp\left( -\sum_k \frac{\Delta_k^2}{2 V_{ij,k}} \right), \qquad V_{ij,k} = 2 s_k^2 + p_{i,k}^2 + p_{j,k}^2 ,$$

summed over the pairs with $|\mathbf{r}_i - \mathbf{r}_j| \leq R$ in the
uncorrected positions, $R$ the *max drift*.  The blob widths are
$s_x = s_y = \sigma$ and $s_z = f\sigma$, $f$ the *z width* (1 by
default), and $p_{i,k}$ is localization $i$'s precision in axis $k$:
`xy_err_nm` for x and y, `z_err_nm` for z.  Each term is the overlap of two
Gaussians, whose separation has the variance $V_{ij,k}$ in each axis, as a
density: it integrates to the same whatever the precisions, so a precise
pair is tall and narrow and an imprecise one low and wide.  With *precision
weights* off, the $p$ are zero; with $f = 1$ as well this is COMET's
$\exp(-|\Delta|^2 / 4\sigma^2) / \sigma$.  The weights take the
precisions of what the estimate runs on -- the merged blinks', with *merge
blinks* -- with z's zero where z is not corrected or the table has no
`z_err_nm`, and a missing value replaced by the median.  A table without
`xy_err_nm` is run unweighted, and the run says so.  The weights are for
the spline fit of the shift alone: the time-window fit and a
*transformation* run unweighted.  This is DME's weighting, a Gaussian per
pair from the two precisions; DME's cost differs otherwise -- the logarithm
of a sum over each localization's neighbours rather than one sum over pairs.  The pair list is built once with a KD-tree
and not updated as the drift changes, so $R$ must be at least as large as the
drift between any two times: a pair that starts further apart is never seen.
$C$ does not change when the same constant is added to the drift of every
frame, so only differences are determined.

$-C$ and its gradient are computed by a compiled, multithreaded kernel and
minimised with L-BFGS-B, stopping when the relative change of the cost falls
below *ftol* ($10^{-7}$; the gradient tolerance is $10^{-5}$), with every
parameter bounded to $\pm 2R$.  With *approximate kernel* on, pairs whose
exponent $\sum_k \Delta_k^2 / 2V_{ij,k}$ is above 9 are skipped -- further
apart than $6\sigma$, for one width and no precisions: each contributes
less than $e^{-9}$ of a coincident pair, and once $\sigma$ is small they
are most of the pairs.

**The spline.**  With $F$ frames and knot spacing $\Delta$, the drift in each
axis is a clamped cubic B-spline

$$d(t) = \sum_{k=1}^{K} c_k\, B_k(t), \qquad K = \max\left(4,\ \mathrm{round}(F/\Delta) + 3\right),$$

with its knots evenly spaced over frames $0$ to $F-1$.  The optimiser's
variables are the $3K$ coefficients $c_k$; the kernel returns the gradient
with respect to the drift of every frame, and the chain rule to the
coefficients is one sparse matrix product.  A B-spline cannot overshoot its
coefficients, so the curve does not ring between knots the way an
interpolating spline through noisy estimates can.  A *spline penalty*
$\lambda$ adds $\lambda \sum_k (c_{k+1} - 2c_k + c_{k-1})^2$ per axis to the
cost.

The widths are $\sigma_0 = \sigma_\mathrm{init}$ (default $R/3$), then
$\sigma_{n+1} = \max(\sigma_n / 1.5,\ \sigma_\mathrm{target})$, one full
L-BFGS-B at each, ending with the one at $\sigma_\mathrm{target}$: with the
defaults 100, 67, 44 and 30 nm.  The spline fit always runs on the CPU and
always uses the $6\sigma$ cutoff.

**The error bar** (*error bar*).  The selected localizations are cut at the
median of x or of y, whichever they spread further along -- DME's split --
and the whole estimate, passes and blink merging included, is run on each
half with the same settings.  Each half's curve is moved to the mean of the
whole curve.  The noise per axis is the root mean square over frames of the
difference between the two halves' curves, each made zero mean, divided by
2: each half has half the data, so its noise is $\sqrt{2}$ times the
whole's, and the difference of two such is $\sqrt{2}$ times that again.
An axis whose curve has a root mean square about its mean below twice its
noise is reported as having no drift above the noise.  On simulations with
no drift, that ratio had a median of 1.0 and was at most 1.7 over 144
axes; with drift it is in the tens.  A transformation has no error bar: the
halves have different centres.  The error bar measures noise only.  A bias
common to both halves -- drift beyond *max drift*, or junk that does not
follow the drift -- does not show in it.

**Time windows** (*fit spline* off).  The windows are, by *window unit*:
*window size* consecutive frames that contain localizations; or frames
accumulated until a window holds at least *window size* localizations; or
*window size* windows, each with about the same share of the localizations.  Each
window $s$ has one drift vector $\mathbf{d}_s$, and a localization's
$\mathbf{d}(t_i)$ is that of its window.  *max locs / window* keeps a random
subset of each window.  The width schedule is COMET's own: after each
successful L-BFGS-B at width $\sigma$, the run stops if
$\sigma \leq \sigma_\mathrm{target}$ and the median squared change of the
parameters over that step was larger than over the step before (or
$\sigma \leq 1$ nm); otherwise $\sigma$ is divided by 1.5.  So it can end
below *target sigma*.  Before each step the window drifts are smoothed with a
running mean over *smoothing* windows.  Finally a curve through the windows'
values, placed at the mean frame of each window's localizations, gives every
frame: a cubic spline (*cubic*) or a Catmull-Rom spline, which is held at the
second and second-to-last window's value beyond them.

**Quality control** (time windows only).  After the fit, each window's
overlap with its neighbours in other windows is computed with the fitted
drift, $Q_\mathrm{obs}$, and with none, $Q_\mathrm{null}$, both per pair.  A
window whose lift $Q_\mathrm{obs}/Q_\mathrm{null} - 1$ is at most *min lift*
is discarded, and the curve through the remaining windows bridges it.

**Two passes.**  *Two stage* runs the estimate on merged blinks,
subtracts it, and then runs it again on every localization with *max drift* set to
*fine radius* $r$, starting at $\sigma = r/3$ and ending at $r/5$; the two
drifts add.  *RCC first* runs [RCC](plugin:Analysis/Drift/RCC) over *RCC
windows* windows (merging blinks as set here), subtracts it, and runs COMET on
what is left; again the two add.  In both, the second pass searches a small
radius, which is what makes it fast.  The fit in each pass is the one
asked for, the spline or the time windows.

**Compared with the paper.**
[Reinkensmeier et al. 2026](https://doi.org/10.64898/2026.03.27.714864)
present COMET as working directly on 2D or 3D localizations and following
drift with a finer time resolution than image cross-correlation.  The
default here gives some of that time resolution up on purpose: the drift is
a smooth B-spline with a coefficient every 2000 frames, not one value per
time window, which was less noisy on a real dataset.  *fit spline* off is
the time-window fit of the paper's software; *Differences from COMET and
SMAP* lists the other changes.

**A transformation over time** (*transformation*).  Each frame has a
matrix $A(t)$ besides the drift, and a localization at $\mathbf{r}$ in
frame $t$ is corrected to
$\mathbf{c} + A(t)\,(\mathbf{r} - \mathbf{c}) - \mathbf{d}(t)$, with
$\mathbf{c}$ the midpoint of the bounding box of the localizations the
estimate used.  The drift is free, so $\mathbf{c}$ does not change the
correction, only how it is split between drift and transformation.  The
matrix is $A(t) = I + \sum_p \theta_p(t)\, E_p$ over the model's
generators $E_p$: for *xy scale* the identity on x and y; for *xy scale +
rotation* that and the rotation generator, which together make $A$ exactly
a scaled rotation; for *xy affine* the four entries of x and y; for *xy
affine + z scale* those and $A_{zz}$, so that nothing turns about the x or
y axis, which neither a sample nor a lens does; for *3D affine* all nine.
In the xy models z is only shifted.  Each $\theta_p$ is a
clamped cubic B-spline like the drift's, over *transformation knots* frames
(the drift's knot spacing on auto).

In the cost, the difference of a pair's corrected positions is divided,
in x and y, by $(\det A_{xy}(t_i) \det A_{xy}(t_j))^{1/4}$, the geometric
mean of the two frames' scales in the plane ($A_{xy}$ the 2 x 2 block),
and in z by the same mean of z's scale: 1 where z is only shifted,
$A_{zz}$ in *xy affine + z scale*.  In *3D affine*, where the matrix mixes
the axes, all three are divided by the mean of $(\det A)^{1/3}$.  Without
it the overlap grows as everything shrinks, and the optimum is the image
collapsed onto its middle.  With it, a transformation common to both frames
of a pair cancels from the scale, and only its change between frames is
measured, as only the change of the drift is.  So the spline coefficients
of every $\theta_p$ are kept at zero sum, by optimising in an orthonormal
basis of the zero-sum vectors.  For L-BFGS-B the variables are each
$\theta_p$ times the half-width of the data along the axis its generator
multiplies -- how far it moves the outermost localizations, in nm -- so
that they and the drift's coefficients are of one size; they share the
drift's bound $\pm 2R$ and the *spline penalty*.  Along z that half-width
is a few hundred nm, not the field's ten µm.

A *transformation penalty* $\lambda$ adds
$\lambda\, \frac{S}{n \sigma^2} \sum_p (r_\mathrm{max}/r_p)^2 \sum_k u_{pk}^2$
to the cost, with $u_{pk}$ the variables in nm, $r_p$ the half-width of
parameter $p$'s axis and $r_\mathrm{max}$ the largest, $S$ the overlap at
the first evaluation of the width step, $n$ the number of variables.  It
is a pull on the relative change $\theta$, the same for every parameter,
towards no change: in nm an entry acting on a thin z is held
$(r_\mathrm{max}/r_z)^2$ -- a thousand times, for 600 nm of depth in a 20
µm field -- harder than one acting across the field.  The kernel
returns the gradient with respect to $A$, $\mathbf{d}$ and the scale factor
of every frame, and the chain rule through $\det$ and the splines is done
per frame.  A frame whose $\det A_k \leq 0$ is refused (its cost is
infinite).  The time-window fit has no transformation.

*Reference* first or last frame makes $A$ the identity and the drift zero
there: the correction becomes $A(t_r)^{-1}$ applied after the estimated
one.  Two passes (*two stage*, *RCC first*) compose: the second
correction is applied after the first, which for shifts alone is the sum.

**The cost estimate.**  The pairs within $R$ are counted with a KD-tree on a
fixed random sample of at most 60 000 of the selected localizations and
scaled by $(N/n)^2$.  With *merge blinks* on, a stretch of frames from the middle of
the acquisition is merged first, and the fraction it keeps is the fraction
the whole table is expected to keep.  From the pair count $P$, the time is
$P\,(16\ \mathrm{ns} + E \times 2.1\ \mathrm{ns})$, plus 0.3 µs per
localization of the table, with $E = 23$ cost evaluations per width in the
schedule above, and twice that with the *error bar*; the memory is 24 bytes
per pair.  The constants were measured
on one computer, so the answer is an order of magnitude, and it is only used
to decide whether to ask.

## Parameters

The ones shown without *more* are the ones worth looking at.  Several
settings under *more* apply only to the time-window fit and are ignored
while *fit spline* is on; their tooltips say so.

### max_drift_nm
The setting that decides the running time.  Set it to what the stage
actually drifts over the whole acquisition, not a safe-looking round number:
the pair count, and with it the time and memory, grows steeply with it.  Too
small, and the drift beyond it cannot be found at all.  RCC, or a first run
with blinks merged, is a quick way to find out how large the drift is.

### target_sigma_nm
Smaller sharpens the last step, but the overlap then has more small maxima
for the optimiser to stop in.  The time-window fit can end below it (see *In
detail*).

### initial_sigma_nm
It must be wide enough that the overlap has a single maximum within reach
of zero drift, which is why it scales with *max drift*.

### spline_knot_frames
2000 frames was the best choice on a real 46 000-frame dataset: finer knots
were noisier without being sharper, and there was no sign of
over-smoothing up to 4000.  Drift that changes faster wants a smaller
spacing.  The noise grows by about $\sqrt{2}$ for every halving of the
spacing; the *error bar* shows whether a finer spacing still leaves the
drift well above it.

### precision_weights
Leave it on.  On simulations it lowered the error by 10-20% in every axis,
and in z by a fifth where a fifth of the localizations were out-of-focus
fits pinned in z, by half where two fifths were -- within a nanometre of
what a hand-set filter on z gave.  Off is COMET as
published, for a table whose precisions cannot be trusted.

### sigma_z_factor
One width for x, y and z is right at the default *target sigma*: 30 nm is
already wider than most z precisions, and making z three times wider made
the axial error worse on simulations (1.1 to 1.8 nm, and 4.6 to 15 nm with
pinned out-of-focus fits).  It helps only with a *target sigma* near the
lateral precision (8 nm: 1.8 to 1.2 nm), where the *precision weights* do
as well.

### split_half
It doubles the run.  Turn it off for a large ungrouped estimate where the
time matters more than knowing the noise.

### spline_penalty
Rarely needed: the knot spacing already sets how smooth the curve is.  On
real data a large penalty flattened genuine early drift rather than removing
an artefact.

### transformation
Leave it at shift unless the sample or the optics change during the
acquisition: a drift-corrected image that is sharp in the middle and
smeared at the edges, or a local correction (in a small ROI) that works
where the global one does not.  Start with the fewest parameters that
describe the change.  On a simulated 20 µm field with 30 000
localizations, scale and rotation came back to within 150 ppm (1.5 nm at
the edge of the field); the more parameters, the more an uneven density
of localizations can pull on them.  For a sample that changes along z
too, *xy affine + z scale* is the first step beyond the xy models, *3D
affine* the most free; on a few hundred nanometres of depth, z's entries
are determined to about a nanometre, which in ppm is thousands (see
*transformation penalty*).  The fit costs about twice as long as the
shift alone.

### transformation_penalty
For a thin sample in *xy affine + z scale* or *3D affine*.  Measured in nm,
z is as well determined as x and y: the z entries wander by a few
thousand ppm of a few hundred nm of depth, which is a nanometre, and the
correction is as good with them as without.  The penalty keeps them near
1 when that matters to the reading: on a simulated 600 nm deep, 20 µm
wide sample, $10^{-3}$ took the z entries of *3D affine* from 2300-4600
ppm to below 700 and left the xy scale as found.  It cannot tell noise
from a real change, though: a 5% change of z's scale over that depth --
15 nm at the top and bottom -- was mostly held back by it too.  Above
$10^{-2}$ it starts to pull on the xy entries as well.

### transformation_knot_frames
A change of the optics is slow, often much slower than the drift of the
stage; fewer coefficients for it are fewer parameters to determine from
the same data.

### segmentation_var
A window must hold enough localizations to show the structure -- a few
thousand is a good target -- and the acquisition must give at least a few
tens of windows.  With *window unit* set to localizations, windows get longer
as the sample bleaches.

### group
Merging blinks made a real estimate 20 to 35 times faster.  On real data it also
gave a less noisy drift curve, because the repeated localizations of one
molecule are not independent measurements.

### two_stage
It suits a large dataset where a
single pass over every localization is slow: the pass over merged blinks
gets within a few nanometres, and the pass over every localization refines
that over a small radius.

### two_stage_radius_nm
Well above what the first pass leaves, which is a few nanometres.

### rcc_prepass
It is the lever when the
estimate would otherwise take hours: RCC's cost does not depend on how far it
looks, COMET's does.

### rcc_prepass_max_drift_nm
When the plugin offers *RCC first* before a long run, it puts the *max
drift* that was asked for here and uses 50 nm for COMET.

### max_locs_per_segment
Fewer per window is roughly quadratically faster and less precise; on real
data 2000 per window was ten times faster and moved the drift by 2.5 nm rms.
The subset is drawn with a fixed seed per window, so a run repeats exactly.

### boxcar_width
A running mean over windows also flattens real, fast drift.

### optimizer_ftol
COMET's own tolerance is about $2 \times 10^{-13}$; $10^{-7}$ changed the
drift by about a tenth of a nanometre on real data for less than half the
cost evaluations.

### optimizer_ftol_coarse
Loosening it is measured to be a bad trade: faster, but the early steps can
land in the wrong maximum, and the fine steps cannot undo that.

### quality_control
Useful for spotting a window that failed.  The spline fit has no windows,
and the run refuses the combination.

### min_lift
0 keeps every window whose fit is better than no correction at all.

## Output

* **The table**, with the drift subtracted from the position columns of every
  localization.  The run is recorded in the file's history.  *Stop* ends the
  fit at its next step and leaves the table as it was: a curve refined half
  way is not one to subtract.
* **The drift curve**, x, y and z against frame (*Show result*); an axis
  with no drift at all is not drawn.  With the *error bar*, the halves'
  curves are drawn dashed in the same colours.  One curve per file when the selection
  holds several.  It is saved with the file and drawn
  again when it is reopened.
* **The transformation**, with *transformation*, in a second panel under the
  drift curve: each of its parameters against frame, in ppm -- nm of
  movement per mm from the centre, which is what a scale change of
  $10^{-6}$ and a rotation of a microradian both are.  It is the
  transformation the sample underwent, as the drift is how far it moved:
  an expanding sample has a rising scale, and a positive rotation is
  anticlockwise.  It is saved with the curve, and Transform's *drift of
  another file* applies it with the drift.
* **The text** gives the range of the drift in x, y and z, with the *error
  bar* its noise in each axis and the axes with no drift above the noise,
  with a transformation its centre and the range of each parameter in ppm,
  and with *quality control* the time windows that were discarded.  The
  noise is kept in the table's metadata with the correction.

A good result is a smooth curve of plausible size, a few hundred nanometres
at most, well above its noise, with halves that follow it, and an image
that is visibly sharper after correction.  A noise of a nanometre or two is
typical; a curve that wanders by a few nanometres with a noise of the same
size is noise, and *no drift above the noise* says so.  A curve
that sits flat at twice *max drift* has run into the optimiser's bound, and
*max drift* was too small.  Sharp excursions that come straight back usually
mean the data at that time had too little to go on: merge blinks, filter
more cleanly, or compare with RCC.

## Differences from COMET and SMAP

SMAP has no COMET.  Its nearest relative there is the DME drift correction,
which also works on the localizations rather than on images.

The estimator is COMET 1.1.0 (github.com/gpufit/Comet), vendored with its
cost function and optimiser.  Around it, this plugin differs from upstream
COMET in these ways:

* **The spline is the default.**  Upstream COMET fits one drift per time
  window and interpolates.  Here the drift is fitted directly as a B-spline
  in time, with COMET's cost.  On a real dataset, both with blinks merged, its noise
  was 0.8-1.0 nm against the time-window fit's 1.3-1.7 nm.
* **Merging blinks first**, by default, which COMET does not do.
* **Quality control** has a different criterion: upstream discards a
  window whose overlap does not beat its no-drift overlap by at least the
  standard deviation of the no-drift overlap *across windows*.  On a
  bleaching sample that spread mostly measures the falling density, and
  flagged 90 of 92 windows on one dataset.  The lift divides the density out.
* **Precision weights** are an addition, DME's weighting of each pair by
  the two localizations' precisions on COMET's cost; COMET counts every
  pair alike, with one width for all three axes.
* **The error bar** is an addition, DME's split-half precision.
* **Two passes** and **RCC first** are additions.
* **A transformation over time** is an addition.  SMAP's
  *driftcorrectionM* ([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)) fitted a magnification alone: it rendered
  time blocks, found each pair of blocks' relative magnification from the
  peak of their cross-correlation at a few trial scales about the middle of
  the ROI, splined the blocks' values over frames and rescaled the
  positions -- without a shift.  Here the change of scale, and optionally a
  rotation or a full affine transformation, are fitted with the drift in
  COMET's one overlap, so the two do not have to be run one after the
  other and cannot pull against each other.

## References

* Reinkensmeier L, Aufmkolk S, Farabella I, Egner A, Bates M. Cost-function
  optimized maximal overlap drift estimation for single molecule localization
  microscopy. *bioRxiv* (2026), preprint.  COMET, the method used here.
  [doi:10.64898/2026.03.27.714864](https://doi.org/10.64898/2026.03.27.714864)
* Cnossen J, Cui TJ, Joo C, Smith C. Drift correction in localization
  microscopy using entropy minimization. *Opt Express* 29, 27961 (2021).
  DME, the localization-based method in SMAP; its precision weighting and
  split-half error bar.
  [doi:10.1364/OE.426620](https://doi.org/10.1364/OE.426620)
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870-872 (2020).  SMAP's *driftcorrectionM*.
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
* Wang Y, Schnitzbauer J, Hu Z, et al. Localization events-based sample drift
  correction for localization microscopy with redundant cross-correlation
  algorithm. *Opt Express* 22, 15982 (2014).  RCC, used by *RCC first*.
  [doi:10.1364/OE.22.015982](https://doi.org/10.1364/OE.22.015982)
