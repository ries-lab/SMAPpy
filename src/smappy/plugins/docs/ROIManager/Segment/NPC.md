---
version: "2"
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
that look like a ring: the right radius, and few localizations far from it.
The ROIs are centred on the fitted circle, which is what
[NPC Corners](plugin:ROIManager/Evaluate/NPC Corners) and
[NPC Labeling Efficiency](plugin:ROIManager/Analyze/NPC Labeling Efficiency)
need.

The checks are made to keep sparse pores.  A pore with only a few corners
labelled is still a pore, and those pores are what the labelling efficiency
is measured from; what has to go is what is not a ring at all -- blobs,
filled clumps, pieces of filament.  Use it on a 2D view of the nuclear
envelope at the bottom of the nucleus, where the pores are seen face on, on
grouped localizations (one per blink).  For structures that are not rings,
[Density Peaks](plugin:ROIManager/Segment/Density Peaks) finds sites by
density alone.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.plugins.npc import (NPCSegmentSettings, segment_npcs, ring_filtered,
                                ring_quality)
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
disc.  An empty area gives zero, and so does an evenly covered one.  A ring
gives a strong peak at its centre.

**3. Candidates.**  The peaks of the filtered image are candidates, taken
strongest first.  A candidate with fewer than *min localizations* (4) in the
band of ±*ring width* around its ring is dropped at once.

**4. Circle fit, on the precise localizations.**  Only the localizations
better than *judge on precision* (20 nm) within the *window* (100 nm) are
used from here on.  A circle of the set *radius* is fitted to them to place
the centre, twice; the ROI goes there.  Then the radius is fitted as well.

**5. Is it a ring?**  The fitted radius must lie between *min fitted radius*
and *max fitted radius*: a blob fits a small circle.  And the localizations
must lie on that circle: each one's distance from it is compared with its own
precision.  A localization more than three of its precisions off the ring is
*far*.  On a pore that is rare; on a filled clump many are.  The candidate is
rejected when it has more far ones than *max far* (1 %) can explain by
chance -- judged by count, so a pore with ten localizations is not rejected
for one stray.

**6. Separation.**  A pore whose centre is closer than the *separation*
(120 nm) to a pore already found, or to an ROI already in the file, is
dropped.  Running the plugin a second time adds only what is new.

```figure A 2 µm piece of a simulated nuclear envelope (Nup96, 60 % labelling).  Left: the localizations.  Middle: the ring-filtered density, bright at the centre of every pore.  Right: the pores found, each with its fitted circle, and the true centres (+).
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
    if lo < cx < hi and lo < cy < hi and s["use"]:
        axes[2].add_patch(Circle((cx, cy), s["radius_nm"], fill=False,
                                 color="#2f7fd0", lw=1))
for ax, title in zip(axes, ("localizations", "ring-filtered", "pores found")):
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
axes[0].plot([lo + 100, lo + 600], [lo + 100, lo + 100], "k", lw=2)
axes[0].text(lo + 350, lo + 160, "500 nm", ha="center", va="bottom", fontsize=7)
```

```figure What the checks see, on made-up structures with a precision of 5 nm: the fitted circle (blue), the localizations more than 3 precisions from it (red), and the verdict.  A full pore passes, and so does a sparse one with three corners.  A single blob fits a small circle.  A filled disc fails on its far localizations, whether the site is on it or beside it -- where the ring filter often puts it.
rng = np.random.default_rng(0)
def corners(which, n=12):
    return np.concatenate([np.column_stack((53.7 * np.cos(k * np.pi / 4) + rng.normal(0, 4, n),
                                            53.7 * np.sin(k * np.pi / 4) + rng.normal(0, 4, n)))
                           for k in which])
r = 60 * np.sqrt(rng.uniform(0, 1, 300)); a = rng.uniform(0, 2 * np.pi, 300)
filled = np.column_stack((r * np.cos(a), r * np.sin(a)))
cases = (("pore", corners(range(8)), (4, -4)), ("three corners", corners([0, 1, 2], 5), (0, 0)),
         ("blob", corners([0]), (0, 0)), ("filled disc", filled, (0, 0)),
         ("beside a disc", filled, (55, 0)))
fig.set_size_inches(9, 2.3)
for ax, (name, pts, start) in zip(fig.subplots(1, 5), cases):
    sigma = np.full(len(pts), 5.0)
    q = ring_quality(pts[:, 0], pts[:, 1], start, settings, sigma)
    c = np.asarray(q["center"])
    d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    far = np.abs(d - q["radius_nm"]) > 3 * np.hypot(5.0, settings.radial_extra_nm)
    ax.scatter(pts[~far, 0] - c[0], pts[~far, 1] - c[1], s=3, c="k", linewidths=0)
    ax.scatter(pts[far, 0] - c[0], pts[far, 1] - c[1], s=3, c="#d62728", linewidths=0)
    if 0 < q["radius_nm"] < 500:
        ax.add_patch(Circle((0, 0), q["radius_nm"], fill=False, color="#2f7fd0", lw=1.2))
    ax.set_xlim(-120, 120); ax.set_ylim(-120, 120); ax.set_aspect("equal")
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
million pixels.  A candidate is a pixel above zero that is the largest within
a square of about the *separation* on a side.

**The fits.**  Both minimise $\sum_i \rho(d_i - R)$, $d_i$ the distance of
localization $i$ from the centre, with the soft-L1 loss
$\rho(z) = 2s^2\left(\sqrt{1 + z^2/s^2} - 1\right)$, $s$ the *ring width*: a
localization far from the circle counts with its distance, not its square.
They are minimised by Gauss-Newton with reweighting, up to 30 iterations.
They use the localizations better than *judge on precision* within the
*window*; when fewer than 3 are, or the table has no precision column, all of
them.

**The far test.**  With $\hat{R}$ the fitted radius and $\sigma_i$ the
precision (`xy_err_nm`) of each of the $n$ precise localizations,

$$z_i = \frac{d_i - \hat{R}}{\sqrt{\sigma_i^2 + e^2}},$$

$e$ the *ring spread* (5 nm, for the tilt of the pore and the size of the
label).  A localization is far when $|z_i| > 3$.  With $m$ far ones, the
candidate fails when $P(X \geq m) <$ *far test level* (0.05) for $X$
binomial with $n$ trials and the chance *max far*.  Two things follow.  The
test is the same at 500 photons as at 5000, because each localization is
judged by its own precision; and it needs nothing about how often a
fluorophore blinks, which is not known at this point.  Without a precision
column the test is off, and only the radius is checked.

**Why not the fractions inside and outside.**  Counted on every
localization, as SMAP's clean-up does, those fractions do not tell a pore
from a clump at low photon numbers: a pore's imprecise localizations land
inside and outside too.  And a filled clump the size of a pore passes them,
because the ring filter answers a filled disc with a ring of response
*around* it, so the site lands beside the clump, and from there the clump is
an arc of the right radius with nothing inside.  On simulations with
pore-sized clumps among the pores, the far test kept the clumps that reached
the analysis to 3 to 5 per 300 pores at 5000 photons (16 to 19 with the
fractions); see `studies/npc_le`.

**Min localizations.**  Every localization in the band counts here, precise
or not.  A pore that shows $k$ corners has at least $k$ of them, so with the
analysis fitted from 5 corners, any value up to 5 loses no pore the analysis
uses; 4 removes most of the background.  A higher value removes the sparse
pores that still show 5 corners: the earlier default of 10 found 27 % of the
pores showing 4 corners in the simulations of `studies/npc_le`, and 14 % with
one blink per fluorophore at 500 photons.

**What it runs on.**  In the ROI manager, the plugin runs on the file being
shown, on its localizations after the layer's filters, and on the grouped
localizations when the layer is grouped.  The ROIs get the manager's size and
shape (a circle of 300 nm by default).  Each ROI records how it was found
(this plugin, its version, its settings, the filters) and its numbers: the
fitted radius, the localizations on the ring, the precise ones, the fraction
far, and the checks it failed.

## Parameters

The defaults are for Nup96 and other proteins of the pore's cytoplasmic and
nuclear rings, at a radius of about 54 nm.  For a protein at another radius,
change *radius* and the radius range together.

### ring_width_nm
About the localization precision plus the label's size, 10 to 20 nm.

### min_locs
At most the fewest corners the analysis is fitted from (5): see *In detail*.

### precision_nm
The same as NPC Corners' *corner precision*, so that the pores are judged on
the localizations they are counted with.  Looser lets more imprecise ones
into the far test, which they pass anyway; tighter leaves sparse pores too
few to judge (3 are needed).

### min_radius_nm
With *max fitted radius*, the band around the expected radius.  Look at the
fitted radius in the *checks* figure: real pores make a narrow peak.

### max_far
On a pore about 0.3 % of the localizations are more than three precisions
off; 1 % leaves room for neighbours and the tilt.  Raise it if the *checks*
figure shows real pores rejected for *far*.

### keep_rejected
Tick it while choosing the settings: each rejected candidate becomes an ROI
that is not used, and its origin lists the checks it failed.

### replace
On, a second run with other settings gives a fresh answer: the ROIs an
earlier run of this segmenter made on the file go first (their origin says
which).  Off, to add the pores of a second region or a second pass to those
already found -- a candidate near an existing ROI is suppressed.  ROIs drawn
by hand or made by another segmenter stay either way.

## Output

* **ROIs**, one per pore, added to the current file in the ROI manager and
  centred on the fitted circle, used from the start; with *keep rejected*,
  the rejected candidates too, not used.
* **The text** says how many pores were kept of how many candidates, and how
  often each check failed.
* **The main figure** shows the localizations with every candidate's fitted
  circle: kept in blue, rejected in grey.
* **The *checks* figure** has the fitted radius, the fraction far from the
  ring and the precise localizations per candidate, kept and rejected
  stacked, the radius limits in red.

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
  a number of localizations, which means the same at any pixel size.
* **The checks.**  SMAP's clean-up compares the localizations inside and
  outside the ring with those on it (*in/ring* < 0.3, *out/ring* < 0.75),
  checks the fitted radius, a minimum size and the mean PSF width, and
  marks the bad sites as not used.  Here the radius is kept and the fractions
  are replaced by the far test on the precise localizations, for the reasons
  in *In detail*; the PSF width is a filter on `sigma_nm` in the layer.
* **One step.**  Segmenting and judging happen in one run, and the rejected
  candidates are dropped unless *keep rejected* is ticked.
* **The fits** use a soft-L1 loss on the precise localizations; SMAP's
  `fitposring` is plain least squares on all of them.
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
