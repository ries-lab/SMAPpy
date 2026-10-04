---
version: "1"
covers: [smappy.plugins.npc.segment_npcs, smappy.plugins.npc.ring_kernel, smappy.plugins.npc.ring_filtered, smappy.plugins.npc.ring_quality, smappy.plugins.npc.fit_circle, smappy.roi_manager.core.ROIProject.find]
---

## What it does

Nuclear pore complexes (NPCs) are rings of eight corners, about 110 nm
across, in the flat nuclear envelope.  Imaged at a pore protein such as
Nup96, they are the reference structure of quantitative SMLM
(Thevathasan et al. 2019, [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)):
the labelling efficiency, the resolution and the scale of a microscope can all
be measured on them.  For that, each pore needs its own region of interest
(ROI) in the ROI manager, centred on the pore.

This plugin finds the pores in a file and puts an ROI on each.  It looks for
rings of the pore's radius, fits a circle to every candidate, and keeps those
that look like a pore: the right radius, few localizations inside the ring and
few outside it.  The ROIs are centred on the fitted circle, which is what the
evaluators that follow need, such as
[NPC Corners](plugin:ROIManager/Evaluate/NPC Corners).

Use it on a 2D view of the nuclear envelope at the bottom of the nucleus,
where the pores are seen face on.  Pores at the side of the nucleus are seen
edge on and do not look like rings; they are rejected.  Pores closer together
than the *separation* (120 nm) cannot both be found.  For structures that are
not rings, [Density Peaks](plugin:ROIManager/Segment/Density Peaks) finds
sites by density alone.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.plugins.npc import (NPCSegmentSettings, segment_npcs, ring_filtered,
                                ring_kernel, ring_quality)
from matplotlib.patches import Circle
sim = SimulationSettings(n_frames=3000, seed=3,
                         structure=StructureSettings(preset="npc"),
                         labelling=LabellingSettings(efficiency=0.6))
locs = simulate(sim)
x, y = np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])
xy = np.column_stack((x, y))
settings = NPCSegmentSettings()
sites = segment_npcs(locs, settings)
image, origin = ring_filtered(xy, settings)
truth = np.column_stack((locs.metadata["copies"]["x_nm"],
                         locs.metadata["copies"]["y_nm"]))
```

**1. A density image.**  The localizations are binned into an image with
pixels of *bin* (10 nm).  Its square root is taken, so that one molecule that
blinks many times in one place does not outweigh a whole pore.

**2. A ring filter.**  The image is filtered with a ring of the pore's
*radius* (55 nm), minus a disc of the same total weight.  At each position the
filter gives the density on a ring around it minus the mean density over the
disc.  An empty area gives zero, and so does an evenly covered one.  A filled
blob gives little or less than nothing.  A ring gives a strong peak at its
centre.

**3. Candidates.**  The peaks of the filtered image are candidates, taken
strongest first.  A candidate is dropped at once if fewer than *min
localizations* lie in the band of width ±*ring width* around its ring: that is
background, not a pore.

**4. Circle fit.**  A circle of the nominal radius is fitted to the
localizations within the *window* (100 nm) of the candidate, to place its
centre.  This is done twice, the second time around the first answer.  The
centre found is where the ROI goes.  A second fit, with the radius free as
well, measures the radius of the structure.

**5. Quality control.**  The localizations within the *window* are sorted by
their distance $r$ from the centre: *inside* the ring ($r < R - \Delta R$),
*on* it, or *outside* it ($r > R + \Delta R$), with $R$ the *radius* and
$\Delta R$ the *ring width*.  A pore passes when its fitted radius is within
*min fitted radius* and *max fitted radius*, at most *max inside* of the
localizations are inside the ring, at most *max outside* are outside, and at
least *min localizations* are on it.

**6. Separation.**  A pore whose centre is closer than the *separation* to a
pore already found, or to an ROI already in the file, is dropped.  Running the
plugin a second time adds only what is new.

```figure A 2 µm piece of a simulated nuclear envelope (Nup96, 60 % labelling).  Left: the localizations.  Middle: the ring-filtered density, bright at the centre of every pore.  Right: the pores found, each with its fitted circle, and the true centres (+).  Every pore is found, and nothing else.
lo, hi = 2500.0, 4500.0
inside = (x > lo) & (x < hi) & (y > lo) & (y < hi)
fig.set_size_inches(7.5, 2.7)
axes = fig.subplots(1, 3)
axes[0].scatter(x[inside], y[inside], s=1.2, c="k", linewidths=0)
extent = (origin[0], origin[0] + image.shape[1] * settings.bin_nm,
          origin[1], origin[1] + image.shape[0] * settings.bin_nm)
axes[1].imshow(np.clip(image, 0, None), origin="lower", extent=extent, cmap="magma")
axes[2].scatter(x[inside], y[inside], s=1.2, c="0.6", linewidths=0)
t = truth[(truth[:, 0] > lo) & (truth[:, 0] < hi) & (truth[:, 1] > lo) & (truth[:, 1] < hi)]
axes[2].plot(t[:, 0], t[:, 1], "+", color="k", ms=6)
for s in sites:
    cx, cy = s["center"]
    if lo < cx < hi and lo < cy < hi:
        axes[2].add_patch(Circle((cx, cy), s["radius_nm"], fill=False,
                                 color="#2f7fd0" if s["use"] else "#d62728", lw=1))
for ax, title in zip(axes, ("localizations", "ring-filtered", "pores found")):
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
axes[0].plot([lo + 100, lo + 600], [lo + 100, lo + 100], "k", lw=2)
axes[0].text(lo + 350, lo + 160, "500 nm", ha="center", va="bottom", fontsize=7)
```

```figure What the checks see.  Four made-up structures, each with the circle fitted with free radius (blue), the ring band (dotted) and the checks it fails.  A full pore passes.  A filled disc has too much inside its ring.  A single blob fits a tiny circle.  Three neighbouring corners fit a circle of the right size: only the *min spread* check, off by default, rejects them.
rng = np.random.default_rng(0)
def corners(which, n=15):
    return np.concatenate([np.column_stack((53.7 * np.cos(k * np.pi / 4) + rng.normal(0, 6, n),
                                            53.7 * np.sin(k * np.pi / 4) + rng.normal(0, 6, n)))
                           for k in which])
r = 70 * np.sqrt(rng.uniform(0, 1, 300)); a = rng.uniform(0, 2 * np.pi, 300)
cases = (("pore", corners(range(8))), ("filled disc", np.column_stack((r * np.cos(a), r * np.sin(a)))),
         ("blob", corners([0])), ("three corners", corners([0, 1, 2])))
check = NPCSegmentSettings(min_spread_nm=25.0)
fig.set_size_inches(8, 2.4)
for ax, (name, pts) in zip(fig.subplots(1, 4), cases):
    q = ring_quality(pts[:, 0], pts[:, 1], (0, 0), check)
    c = np.asarray(q["center"])
    ax.scatter(pts[:, 0] - c[0], pts[:, 1] - c[1], s=3, c="k", linewidths=0)
    for rr in (check.radius_nm - check.ring_width_nm, check.radius_nm + check.ring_width_nm):
        ax.add_patch(Circle((0, 0), rr, fill=False, ls=":", color="0.5", lw=0.8))
    if 0 < q["radius_nm"] < 500:
        ax.add_patch(Circle((0, 0), q["radius_nm"], fill=False, color="#2f7fd0", lw=1.2))
    ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}\n" + ("passes" if q["use"] else "fails: " + ", ".join(q["failed"])),
                 fontsize=8)
```

## In detail

**The filter.**  With $b$ the *bin*, $R$ the *radius* and $w$ the *ring
width*, the kernel is

$$K(r) = \frac{e^{-(r - R)^2 / 2w^2}}{\sum e^{-(r - R)^2 / 2w^2}} - \frac{[r \leq R + 2w]}{\sum [r \leq R + 2w]},$$

on a grid of pixels of size $b$ reaching to $R + 3w$, with $[\cdot]$ one where
the condition holds and zero elsewhere.  It sums to zero.  It is applied by an
FFT convolution to $\sqrt{n}$, $n$ the localizations per pixel.  The image
covers the localizations with a margin of $R + 3w + b$ and may have at most 16
million pixels.

**Candidates.**  A pixel is a candidate when it is above zero and is the
largest value within a square of about the *separation* on a side.
Candidates are taken in order of the filter's value, ties broken by position.
The ring band count, the localizations with
$R - \Delta R \leq r \leq R + \Delta R$ of the pixel's centre, must reach
*min localizations*.

**The fits.**  Both fits minimise $\sum_i \rho(d_i - R)$ over the
localizations within the *window*, with $d_i$ the distance of localization
$i$ from the centre.  The loss is soft-L1,
$\rho(z) = 2s^2\left(\sqrt{1 + z^2/s^2} - 1\right)$ with $s = \Delta R$: a
localization far from the circle counts with its distance, not its square, so
the background in the window does not pull the centre.  It is minimised by
Gauss-Newton with reweighting, up to 30 iterations.  The fixed-radius fit runs
twice: the first time on the window around the filter's peak, then on the
window around the result.  The free fit starts from the median distance of
the localizations from that centre.  On one blob the free radius collapses,
which is what rejects a blob.

**The fractions.**  With $N$ the localizations within the *window* of the
centre, the *fraction inside* is the number with $r < R - \Delta R$ over $N$,
and the *fraction outside* the number with $r > R + \Delta R$ over $N$.  The
window should reach past the ring but not into the next pore: at the default
100 nm it does not, for pores more than 170 nm apart.

**The spread** is $(\det C)^{1/4}$, $C$ the covariance of the window's
positions: the geometric mean of the two standard deviations.  A full ring of
radius $R$ has a spread of $R/\sqrt{2}$, about 39 nm; a blob has its
localization precision.  It is off by default (0).  It is the one check that
rejects an arc, but the arcs it rejects are mostly real pores with only a few
corners labelled.  Those are the pores the labelling efficiency is measured
from, so rejecting them would bias it.  With the separation above $2R$, the
arcs of neighbouring pores it was meant for no longer appear in simulations.

**Separation.**  A candidate inside the separation of a pore already accepted
is skipped before it is fitted, and its fitted centre is checked again
afterwards.  A circle through an arc of a pore can be centred up to $2R$ from
it, which is why the default, 120 nm, is a little above $2R$.

**What it runs on.**  In the ROI manager, the plugin runs on the file being
shown, on its localizations after the layer's filters, and on the grouped
localizations when the layer is grouped.  The ROIs get the manager's size and
shape (a circle of 300 nm by default).  Each ROI records how it was found
(this plugin, its version, its settings, the filters) and its own numbers: the
fitted radius, the ring count, both fractions, the spread and the checks it
failed.

## Parameters

The defaults are for Nup96 and other proteins of the pore's cytoplasmic and
nuclear rings, at a radius of about 54 nm.  For a protein at another radius,
change *radius* and the radius range together.

### ring_width_nm
About the localization precision plus the label's size, 10 to 20 nm.
Narrower and the ring loses localizations to *inside* and *outside*; wider and
the band reaches the centre.

### min_locs
How many localizations a pore must show.  Set it well below what a real pore
has, but above what a patch of background of the size of a ring holds.

### min_radius_nm
With *max fitted radius*, the band around the expected radius.  Look at the
fitted radius in the *checks* figure: real pores make a narrow peak, and the
limits go on either side of it.

### max_inside
Look at the *checks* figure.  Real pores have a few percent inside, from
localizations with poor precision.  Blobs and filled structures have far more.

### max_outside
Raise it on data with much background between the pores, or for a larger
*window*.

### min_spread_nm
About 25 nm rejects arcs and pairs of corners, if data with crowded or broken
pores need it.  It biases the labelling efficiency (see *In detail*).

### keep_rejected
Tick it while choosing the thresholds: each rejected candidate becomes an ROI
that is not used.  Its origin lists the checks it failed, and it can be looked
at, or included, in the ROI manager.

## Output

* **ROIs**, one per pore, added to the current file in the ROI manager and
  centred on the fitted circle.  They are used (*use*) from the start.  With
  *keep rejected*, the rejected candidates are added too, not used.
* **The text** says how many pores were kept of how many candidates, and how
  often each check failed.
* **The main figure** shows the localizations with every candidate's fitted
  circle: kept in blue, rejected in grey.
* **The *checks* figure** has one histogram per check, the fitted radius, the
  two fractions and the spread, with kept and rejected candidates stacked and
  the limits in red.  A good choice of limits cuts through the gaps between
  the peak of the pores and the rest.

A good result has one ROI on every pore that is seen face on, and none
elsewhere.  Pores missed: lower *min localizations*, or widen the limits that
the *checks* figure shows them failing.  ROIs on things that are not pores:
tighten those limits.

Run without the ROI manager, from a script (`Context(locs=locs)`), it returns
every candidate with its numbers in `data["sites"]`, and the centres of the
pores that passed in `data["centers"]`.

## Differences from SMAP

Ported from SMAP's `ROIManager/Segment/segmentNPC` and its clean-up evaluator
`ROIManager/Evaluate/NPCsegmentCleanup`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* **The filter.**  SMAP filters the rendered image of each cell with a ring
  kernel, $e^{-|r^2 - R^2| / 4w^2}$, follows it with a difference of
  Gaussians, and keeps the maxima above a *cutoff* in units of the filtered
  image.  Here a zero-sum kernel does both jobs at once, and what decides is
  a number of localizations on the ring, which means the same at any pixel
  size or rendering.
* **One step.**  In SMAP, the segmenter proposes sites and a separate
  evaluator fits and judges them, marking the bad ones as not used.  Here
  both happen in one run, and the rejected candidates are dropped unless
  *keep rejected* is ticked.
* **The fits** use a soft-L1 loss; SMAP's `fitposring` is plain least
  squares, which the background in the window can pull.
* **The checks.**  SMAP compares the counts inside and outside to the count on
  the ring (*in/ring* < 0.3, *out/ring* < 0.75); here they are fractions of all
  localizations in the window.  SMAP's *min size* is the *min spread* here,
  off by default for the reason given above.  SMAP's *max average PSF* is
  left out: a filter on `sigma_nm` in the layer does the same.
* SMAP works per cell, and needs cells first; this works on the whole file.

## References

* Thevathasan JV, Kahnwald M, Cieśliński K, et al. Nuclear pores as
  versatile reference standards for quantitative superresolution
  microscopy. *Nat Methods* 16, 1045 (2019).
  [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)
  -- the method, and the Nup96 geometry simulated in the figures.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
