---
version: "3"
covers: [smappy.fiducials.find_beads, smappy.fiducials.fit_drift, smappy.fiducials.estimate_drift_fiducials, smappy.fiducials.find_and_choose, smappy.fiducials.bead_names, smappy.beads.select_beads, smappy.beads.bead_metrics, smappy.beads.MadRule, smappy.beads.draw_bead_tracks, smappy.tracking.link.link_tracks, smappy.drift._spline_basis, smappy.drift.Drift.apply, smappy.simulate.beads.simulate_beads]
---

## What it does

Over a long acquisition the sample [drifts](plugin:Panels/Glossary) by tens
to hundreds of nanometres, and every localization is off by however far it
had moved in its frame (the [RCC](plugin:Analysis/Drift/RCC) page says more
about where drift comes from).  When the sample carries
[fiducials](plugin:Panels/Glossary) -- bright, still particles on the
coverslip such as fluorescent beads or gold particles, seen in nearly every
frame -- the drift can be read off them directly: where a bead appears to
move, the sample moved.

This plugin finds the beads among the localizations, follows each one
through the frames, and fits one smooth drift curve to all of them.  It
then subtracts that curve from every localization of the file, the beads
included.

Use it when the sample has beads that stay in focus over the whole
acquisition, or most of it.  A bead is localized to a few nanometres in
every frame, so a handful of beads gives a curve that follows faster drift
than [COMET](plugin:Analysis/Drift/COMET) or RCC, which infer the drift
from the overlap of blinking molecules and need thousands of them per
stretch of time.  Without beads, use one of those two.  It needs `x_nm`,
`y_nm`, `frame` and `photons`; with `z_nm` it corrects z too.

The beads are looked for in the [selection](plugin:Panels/Glossary): the
localizations that pass the layer's filter, inside the drawn
[ROI](plugin:Panels/Glossary) (region of interest) if there is one.  So an
ROI drawn around the beads, or a filter that keeps only the brightest
localizations, tells the plugin where to look.  The selection must hold one
file.

## How it works

```figure-setup
from smappy.simulate import simulate, with_beads
from smappy.fiducials import FiducialSettings, estimate_drift_fiducials
locs = simulate(n_frames=8000, seed=2, drift=True)
truth = np.asarray(locs.metadata["drift_truth"])
truth = truth - truth[0]
locs = with_beads(locs, n_beads=5, gaps=[(0, 2500, 2700), (1, 1000, 1030)],
                  late=[(2, 4500)], seed=2)
drift, beads, fit = estimate_drift_fiducials(locs)
frame = np.asarray(locs["frame"])
bead_true = np.asarray(locs["bead_true"])
```

**1. Bright localizations.**  A bead gives far more photons than a dye
molecule -- tens of thousands, against hundreds to a few thousand.  Only
localizations brighter than *min photons* are considered.

**2. Tracks.**  These are linked into tracks with the simple linker of
[Link tracks](plugin:Analysis/Tracking/Link tracks): in each frame, a
localization continues the nearest track that ended within *maximum jump*
of it.  A bead can be missed in a frame, or fitted badly; a track survives
up to *memory* frames without it (50 by default).  A longer gap ends the
track, and the bead's next appearance starts a new one.

**3. Long tracks are beads.**  A track must be seen in at least *min
length* of the frames -- a quarter, by default.  A dye molecule blinks for a
few frames and is gone, so this is what separates the two.  A bead that
lands on the glass halfway through the acquisition is a bead from then on,
as long as the rest of the acquisition is long enough.

```figure The localizations brighter than *min photons* (grey) and the beads found among them (coloured), x against frame: one bead is interrupted (the gap near frame 2500 is longer than the memory, so its two parts are separate tracks), one starts late.
fig.set_size_inches(7, 2.8)
ax = fig.subplots()
bright = np.asarray(locs["photons"]) >= FiducialSettings().min_photons
x = np.asarray(locs["x_nm"])
ax.plot(frame[bright & (beads.bead < 0)], x[bright & (beads.bead < 0)] / 1000,
        ".", ms=1, color="0.7")
for b in range(len(beads)):
    mine = beads.bead == b
    ax.plot(frame[mine], x[mine] / 1000, ".", ms=1)
ax.set_xlabel("frame"); ax.set_ylabel("x (µm)")
```

**4. One curve through all beads.**  Each bead sits at its own place, its
*offset*; on top of that every bead moves by the same drift.  So the
position of a bead $b$ in frame $t$ is modelled as

$$x = o_b + d(t),$$

with $o_b$ the bead's offset and $d(t)$ the drift.  The drift is a smooth
curve (a cubic B-spline, the same kind of curve as COMET's), with a free
value every *knot spacing* frames.  The offsets of all beads and the curve
are fitted together, by least squares, to every bead's positions at once.
A frame where one bead is missing is carried by the others, and a bead that
starts late adds to the curve from its first frame on.  x, y and z are
fitted separately.

**5. Beads that disagree are left out.**  A bead that came loose, or a
cluster of long-lived molecules taken for one, does not move with the
others.  Each bead is compared with the curve the *other* beads give, by
three numbers: its *noise* (scatter from one frame to the next), its
*wander* (slow motion of its own) and its *jump* (largest sudden step).  A
bead is left out when one of these is well above the other beads', and the
curve is fitted again without it.  The same rule judges the MBM beads of
[MBM drift correction](plugin:Analysis/Drift/MBM).  *Select beads...*
under the settings opens the [bead selector](plugin:Panels/Bead selector),
where each bead's track can be looked at and a bead included or excluded by
hand.

```figure Five beads, one of which creeps 40 nm away in x over the acquisition (red, dashed): against the curve the others give (black), its deviation (right) is a slope, and it is left out.
from smappy.fiducials import estimate_drift_fiducials as _estimate
from smappy.beads import draw_bead_tracks
creep = with_beads(simulate(n_frames=2000, seed=3, drift=True), n_beads=5, seed=3)
xs = np.asarray(creep["x_nm"], float).copy()
mine = np.asarray(creep["bead_true"]) == 3
xs[mine] += 40.0 * np.asarray(creep["frame"])[mine] / 2000
creep = creep.__class__({**creep.columns, "x_nm": xs}, dict(creep.metadata))
_, _, creep_fit = _estimate(creep)
fig.set_size_inches(9, 5.4)
draw_bead_tracks(fig, creep_fit.chosen, ("x", "y", "z"), time_unit="frame")
```

**6. Subtracted from everything.**  The curve is subtracted from every
localization of the table: the beads, the structure, and what the filter
hides.

```figure Top two: each bead's positions less its fitted offset (dots, a colour per bead) lie on one curve (black), which follows the drift that was simulated (grey, under it).  Bottom: the curve less the simulated drift, about 2.5 nm rms -- less than one bead's precision (3 nm, the band) -- and no worse over the gaps or where the late bead joins.  What is left is the frame-to-frame jitter of the simulated drift, a random walk, which a curve with a value every 100 frames does not follow.
fig.set_size_inches(7, 5.4)
axes = fig.subplots(3, 1, sharex=True)
used = frame[fit.rows]
for axis, ax in enumerate(axes[:2]):
    values = fit.residuals_nm[:, axis] + fit.drift_nm[used, axis]
    for b in np.unique(fit.bead):
        mine = fit.bead == b
        ax.plot(used[mine], values[mine], ".", ms=0.8, alpha=0.15)
    ax.plot(truth[:, axis], color="0.55", lw=3.5)
    ax.plot(drift.drift[:, axis], color="k", lw=0.8)
    ax.set_ylabel(f"drift {'xy'[axis]} (nm)")
error = drift.drift[:, :2] - truth[:, :2]
axes[2].axhspan(-3, 3, color="0.9")
axes[2].plot(error[:, 0], lw=0.8, label="x")
axes[2].plot(error[:, 1], lw=0.8, label="y")
axes[2].set_ylabel("error (nm)")
axes[2].legend(frameon=False, fontsize=7, ncol=2)
axes[-1].set_xlabel("frame")
```

## In detail

**Which table.**  The localizations, whatever the layer shows.  The beads
are found among those the layer's filter and the ROI select; the correction
is applied to every localization of their file.  With several files open,
the others keep their positions: their frame numbers are their own, and the
curve says nothing about them.

**The fit.**  With $B_k(t)$ the cubic B-spline basis on knots evenly spaced
over the frames, $K$ of them for $N$ frames,

$$K = \max\left(4,\ \mathrm{round}\left(\frac{N}{\Delta}\right) + 3\right),$$

$\Delta$ the *knot spacing*, the drift is $d(t) = \sum_k c_k B_k(t)$, and
the offsets $o_b$ and coefficients $c_k$ minimise

$$\sum_i r_i^2 + \lambda \sum_k (c_{k+1} - 2 c_k + c_{k-1})^2, \qquad r_i = x_i - o_{b_i} - d(t_i),$$

the first sum over every localization $i$ of every bead, $r_i$ its
residual.  The problem is
linear, so it is solved in one step.  Every localization of the beads
used counts the same; a weighting by precision is left for later.

**Which beads.**  Each bead's residual is taken about the fit of the
*other* used beads, with its own offset fitted over its frames: with few
beads, one inside the fit pulls the curve towards itself and hides part of
its own error.  From it, *noise* is $1.4826\,\mathrm{MAD}(\Delta r) /
\sqrt{2}$ (the differences between consecutive points, blind to slow
motion), *wander* $\sqrt{\mathrm{rms}^2 - \mathrm{noise}^2}$, and *jump*
the largest distance between the medians of the points before and after
any point, over five points or a fiftieth of the bead's points, whichever
is more.  For each of the three, a bead is out when it exceeds the used
beads' median by more than *robust sigmas* times their robust standard
deviation (1.4826 MAD) and by more than *margin*.  Rejecting and refitting
repeat until the set of beads no longer changes, never below *min beads*.
With every bead used, the curve is exactly the fit above.

A bead is named by its median position in micrometres (`x12.3y4.5`), and a
choice by hand (*by hand*) goes to the nearest bead within 0.3 µm of that
place: the linker numbers the beads in the order it meets them, and a
filter or *min photons* changes that order.  A bead whose track a gap
broke in two is two beads at one place, the second named with `#2`.

**The reference.**  A constant can move from every offset into the curve
without changing the fit: only how the beads moved is measured, not where
the sample was.  The curve is fixed by setting it to zero at the first
frame, $d(0) = 0$.  *reference* can move that zero to the last frame
afterwards, by subtracting the curve's last value from all of it: for two
consecutive files corrected separately, the first referenced to its last
frame and the second to its first meet where they join.

**The penalty.**  The second term keeps the system solvable where no bead
was seen at all: there the curve runs straight from one side of the gap to
the other.  Its weight $\lambda$ is $10^{-3}$ of the mean data weight per
coefficient.  Where beads were seen it is a slight smoothing: on the
simulation above it moves the curve by a third of a nanometre on average,
well below the beads' precision.

**Why fit the offsets.**  SMAP subtracted each bead's mean position.  For a
bead seen in every frame that is the same thing up to a constant, but the
mean of a bead that starts late is its position plus the mean drift over
*its* frames, not over all of them, and subtracting it shifts its part of
the curve by the difference.  Fitting the offsets with the curve avoids
that, and a bead's first point -- which carries its whole localization
error -- is no better.

**What limits the precision.**  With $n$ beads each localized to $\sigma$,
a stretch of $\Delta$ frames sets each coefficient from about $n\Delta$
points, so the curve is good to roughly $\sigma / \sqrt{n\Delta}$ -- far
better than one bead's precision -- as long as the drift itself is smooth
on the scale of $\Delta$.  What the drift does faster than that is not
followed and is left in the table: in the simulation above, a random walk
of 0.6 nm per frame, it is most of the 2.5 nm that remain.  Faster drift
wants a smaller *knot spacing* (50 frames brings it under 2 nm there); the
*Output* says how to tell.

## Parameters

### min_photons
Set it between the brightest dye localizations and the beads: the
`photons` histogram of a file with beads usually has a second hump far to
the right.  Too low lets bright dye blinks in, which seldom last long
enough to be taken for beads but cost linking time; too high loses the
frames where a bead was dim.

### min_fraction
Lower it when beads come and go -- one that lands late, or one that
bleaches -- and the shorter tracks are still beads.  Too low, and a
cluster of long-lived dye molecules could pass.

### max_jump_nm
A bead moves only by the drift between two frames it is seen in, nanometres
in most acquisitions; the jump must be larger than its localization error
and the drift over *memory* frames, and smaller than the distance between
beads.

### memory
Longer bridges a bead that was out of focus for a while, as long as the
drift over that time stays within *maximum jump*.  A track broken by a
longer gap is not lost if both parts are long enough.

### spline_knot_frames
100 frames suits beads seen in every frame: a few hundred points per
coefficient.  Smaller follows faster drift and is noisier; larger is
smoother and cuts the corners of a quick drift.

### rejection.k
Larger keeps more beads.  With a handful of beads the spread is uncertain,
and *margin* usually decides.

### rejection.margin_nm
A bead is never left out for being less than this worse than the others'
median.  Set it about the beads' precision.

### rejection.min_beads
At least two are needed to tell which one moved.

### beads
Written by the bead selector; it can be typed too, as `x12.3y4.5 out`.

### use_z
Off corrects x and y only, for a 3D table whose z drift is corrected
otherwise, or not wanted.

## Output

* **The table**, with the drift subtracted from `x_nm`, `y_nm` and `z_nm`
  (or the pixel columns) of every localization.  The beads stay in the
  table, corrected like everything else.  The correction is recorded in the
  file's history and can be undone.
* **The drift curve**, x, y and z against frame, the same figure as COMET's
  and RCC's.  It is saved with the file and drawn again when it is reopened.
* **The beads** figure: each bead's positions less its fitted offset, over
  the curve (left), and its deviation from the curve (right), a row per
  axis; the beads left out red and dashed.  The beads used should lie on
  the curve, scattered about it by their precision.  A used bead whose
  deviation has a slope or a step is not a fiducial -- it is loose, or a
  cluster of molecules -- and is one to exclude in the bead selector.
* **The text** gives the range of the drift, the number of beads (and how
  many were left out) and of the localizations used, and the scatter of the beads about the curve, per
  axis.  That scatter should be close to the beads' own precision
  (`xy_err_nm`); clearly more means the curve is too stiff for the drift
  (a smaller *knot spacing*) or a bead disagrees with the others.

## Differences from SMAP

Based on SMAP's `Drift/driftcorrection_beads`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* SMAP took the beads from its grouping (localizations merged over
  consecutive frames, with more than a number of them); here they are
  tracks of the bright localizations with a memory, and must be seen in a
  fraction of the frames.
* SMAP subtracted each bead's mean and averaged all beads in a running
  window of frames (5 by default).  Here each bead's offset is fitted
  together with one spline, so a bead that starts late or is interrupted
  lands where it belongs, and the curve is smooth rather than a running
  mean.
* SMAP used every bead it found.  Here a bead that disagrees with the
  others is left out, and any bead can be chosen by hand.
* The beads are those of the selection; SMAP also offered the sites of the
  ROI manager.  SMAP saved the corrected file itself; here the correction
  is saved with the file like any other.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
