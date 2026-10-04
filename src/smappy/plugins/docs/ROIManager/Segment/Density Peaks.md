---
version: "1"
covers: [smappy.plugins.roi.density_peaks, smappy.roi_manager.core.ROIProject.find]
---

## What it does

Many SMLM experiments image the same small structure hundreds of times over
one cell: nuclear pore complexes, clathrin-coated pits, centrioles,
kinetochores.  The ROI manager analyses them one by one.  It first needs a
list of *sites*: one region of interest (ROI) centred on each structure.
Clicking hundreds of them by hand is slow, and not reproducible.

Density Peaks proposes those sites automatically.  It looks for places where
localizations pile up, and puts an ROI on each one that holds enough
localizations and is not too close to another.  It is the first of the ROI
manager's three steps:

1. **Segment** (this plugin): find the sites in a file.
2. **Evaluate**: measure every site, with plugins that run once per ROI,
   such as [Statistics](plugin:ROIManager/Evaluate/Statistics).
3. **Analyze**: summarise the collection of sites, for example as
   [Histograms](plugin:ROIManager/Analyze/Histograms).

Use it for compact, well separated structures of roughly 50 to 200 nm, on a
low background.  It looks only at how dense the localizations are, not at
their shape.  A ring, a blob and a piece of filament are all the same to it,
so on filaments or a dense meshwork it puts sites along every line (see the
second figure).  The ROIs it adds can be removed, or excluded with *use*, in
the ROI manager like any other.

Despite the name, this is not the "density peaks" clustering of
[Rodriguez & Laio 2014](https://doi.org/10.1126/science.1242072).  It is a
blurred density image, its local maxima, and a count threshold.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.plugins.roi import DensityPeakSettings, density_peaks
sim = SimulationSettings(n_frames=3000, seed=3,
                         structure=StructureSettings(preset="npc"),
                         labelling=LabellingSettings(efficiency=0.6))
locs = simulate(sim)
x, y = np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])
truth = np.column_stack((locs.metadata["copies"]["x_nm"],
                         locs.metadata["copies"]["y_nm"]))
settings = DensityPeakSettings()
centres = np.array(density_peaks(locs, settings))
```

**1. A density image.**  The localizations are binned into an image with
pixels of *bin* (20 nm).  Each pixel holds the number of localizations that
fell into it.

**2. Blur.**  The image is smoothed with a Gaussian of width *smoothing*
(50 nm).  A single structure then becomes one smooth hill, however its
localizations are arranged inside it.  A nuclear pore, a ring of about
110 nm across, turns into one peak at its centre.

**3. Local maxima.**  Every pixel that is at least as high as its eight
neighbours is a candidate.  They are taken in order, highest first.

**4. Enough localizations?**  For each candidate the localizations within
*count radius* (75 nm) are counted.  A candidate with fewer than *minimum
count* is dropped: that is what removes the many small maxima of the
background.

**5. Re-centring.**  The candidate is moved to the mean position of those
localizations, which is more precise than the pixel it was found in.  The
count is checked again at the new centre.

**6. Separation.**  A candidate closer than *separation* (150 nm) to a site
already accepted is dropped.  Because the highest peaks come first, of two
maxima on one structure the stronger one wins.  ROIs that are already in the
file count as accepted too, so running the plugin a second time adds only
what is new.

```figure Simulated nuclear pores (Nup96, 60 % labelling) in a 2 µm piece of the field.  Left: the localizations.  Middle: the blurred density image the maxima are found in.  Right: the sites found (circles: the *count radius*) and the true pore centres (+).  Every pore is found, and nothing else.
from scipy.ndimage import gaussian_filter
lo, hi = 2500.0, 4500.0
inside = (x > lo) & (x < hi) & (y > lo) & (y < hi)
edges = np.arange(lo, hi + settings.bin_nm, settings.bin_nm)
image = np.histogram2d(y[inside], x[inside], bins=(edges, edges))[0]
blurred = gaussian_filter(image, settings.sigma_nm / settings.bin_nm, mode="constant")
fig.set_size_inches(7.5, 2.7)
axes = fig.subplots(1, 3)
axes[0].scatter(x[inside], y[inside], s=1.2, c="k", linewidths=0)
axes[1].imshow(blurred, origin="lower", extent=(lo, hi, lo, hi), cmap="magma")
axes[2].scatter(x[inside], y[inside], s=1.2, c="0.6", linewidths=0)
t = truth[(truth[:, 0] > lo) & (truth[:, 0] < hi) & (truth[:, 1] > lo) & (truth[:, 1] < hi)]
axes[2].plot(t[:, 0], t[:, 1], "+", color="k", ms=6)
from matplotlib.patches import Circle
for cx, cy in centres:
    if lo < cx < hi and lo < cy < hi:
        axes[2].add_patch(Circle((cx, cy), settings.count_radius_nm, fill=False,
                                 color="#d62728", lw=1))
for ax, title in zip(axes, ("localizations", "blurred density", "sites found")):
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
axes[0].plot([lo + 100, lo + 600], [lo + 100, lo + 100], "k", lw=2)
axes[0].text(lo + 350, lo + 160, "500 nm", ha="center", va="bottom", fontsize=7)
```

```figure The same plugin on the ring and crossed lines of the `demo` structure, with a *minimum count* of 10 (left) and 100 (right).  It finds density, not shape: sites line the ring and the lines at about the *separation*, and at the low threshold also land on the scattered background molecules.
demo = simulate(n_frames=3000, seed=4)
dx, dy = np.asarray(demo["x_nm"]), np.asarray(demo["y_nm"])
fig.set_size_inches(7, 3.3)
for ax, count in zip(fig.subplots(1, 2), (10, 100)):
    found = np.array(density_peaks(demo, DensityPeakSettings(min_count=count)))
    ax.scatter(dx, dy, s=0.2, c="0.6", linewidths=0)
    ax.plot(found[:, 0], found[:, 1], "o", mfc="none", mec="#d62728", ms=4, mew=0.8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"minimum count {count}: {len(found)} sites", fontsize=9)
```

## In detail

**The image.**  Only localizations with finite `x_nm` and `y_nm` are used;
if there are fewer of them than *minimum count*, nothing is proposed.  The
image covers the localizations plus a margin of $4\sigma + b$ on every side
($\sigma$ the *smoothing*, $b$ the *bin*), so that a structure at the edge of
the field still gets its full peak; outside the image the density is taken
to be zero.  The image may have at most 16 million pixels (for 20 nm pixels,
a field of 80 µm by 80 µm); a larger one is refused with a request to
increase the bin.

**The maxima.**  A pixel is a maximum when it equals the largest value in its
$3\times3$ neighbourhood and is above zero.  A flat top of several equal
pixels that touch is one maximum, placed at their mean position.  Maxima are
ordered by the height of the blurred density, ties broken by position, so the
result does not depend on the order of the table.

**Counting and re-centring.**  With $\mathbf{c}$ the centre of the maximum's
pixel and $N(\mathbf{c})$ the localizations within the *count radius* $r$ of
it, a candidate needs $|N(\mathbf{c})| \geq n_{\mathrm{min}}$.  It is then
moved to

$$\mathbf{c}' = \frac{1}{|N(\mathbf{c})|} \sum_{i \in N(\mathbf{c})} \mathbf{x}_i ,$$

and must pass the same count at $\mathbf{c}'$.  The mean is taken over the
compact neighbourhood rather than over the analysis ROI, so a neighbouring
structure just outside $r$ does not pull the centre.  It is still a mean of
what was detected: on an incompletely labelled pore the centre moves towards
the labelled side, by about 15 nm (median) in the simulation above.  An
evaluator that fits a model to the site refines it further.

**Separation.**  $\mathbf{c}'$ is rejected if it lies closer than the
*separation* to any accepted centre, including the centres of the ROIs
already in that file.  This is a greedy non-maximum suppression in
nanometres: which of two close candidates survives is decided by the order,
highest density first.

**What it runs on.**  In the ROI manager, the plugin runs on the file being
shown, on its localizations after the layer's filters, and on its grouped
localizations when the layer is grouped.  The new ROIs get the manager's
global size and shape (a circle of 300 nm by default), which has nothing to
do with the *count radius*: detection and analysis are set separately.  Each
ROI records how it was found: this plugin and its version, its settings, the
filters and the grouping.

## Parameters

The defaults suit compact structures of about 100 nm, such as nuclear pores.

### bin_nm
Smaller than the *smoothing*, or the blur has nothing to smooth over; 10 to
25 nm is typical.  Only the speed and the memory depend on it much.

### sigma_nm
About the radius of the structure.  Much smaller splits one structure into
several maxima (the *separation* then has to remove them); much larger merges
neighbouring structures into one.

### separation_nm
A bit more than the diameter of the structure and less than the typical
distance between two of them.  Too small gives two sites on one structure;
too large loses one of two close neighbours.

### count_radius_nm
About the radius of the structure plus the localization precision, so that
the count and the re-centring see the whole structure but not its
neighbours.

### min_count
The most useful setting for a given data set: it separates structures from
background.  Look at how many localizations a real site has (Statistics
reports it) and set it well below that, and well above what a patch of
background of the same size holds.

## Output

* **ROIs**, one per site found, added to the current file in the ROI
  manager, brightest first.  They are included (*use*) from the start.
* **The text** says how many candidates were found.

A good result has one site on each structure and none on the background.
Sites on empty background: raise *minimum count*.  Faint structures missed:
lower it.  Two sites on one structure: raise *separation* or *smoothing*.
Two structures under one site: lower them.

Run without the ROI manager (from a script, `Context(locs=locs)`), it
returns only the list of centres, in `data["centers"]`.

## Differences from SMAP

SMAP has no single counterpart; its segmenters are specific to a structure.
This plugin replaces the generic part of `ROIManager/Segment/segmentCME` and
`segmentNPC` ([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* SMAP finds maxima in the rendered super-resolution image of each cell,
  after a Gaussian filter (`segmentCME`) or a ring filter and a difference of
  Gaussians (`segmentNPC`), and keeps those above an intensity *cutoff*.
  Here the threshold is a number of localizations within a radius in
  nanometres, which does not change with the pixel size, the blur or the
  rendering.
* SMAP works per cell, and needs cells first; this works on the whole file.
* The centre is re-centred on the localizations rather than left at the
  pixel, and candidates too close to each other, or to ROIs already there,
  are suppressed.
* There is no ring filter: a blur of about the structure's radius already
  gives a pore a single peak.  For nuclear pores,
  [NPC](plugin:ROIManager/Segment/NPC) keeps SMAP's ring filter and judges
  each candidate's shape as well.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
* Thevathasan JV, Kahnwald M, Cieśliński K, et al. Nuclear pores as
  versatile reference standards for quantitative superresolution
  microscopy. *Nat Methods* 16, 1045 (2019).
  [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)
  -- the Nup96 pores simulated in the figures.
* Rodriguez A, Laio A. Clustering by fast search and find of density peaks.
  *Science* 344, 1492 (2014).
  [doi:10.1126/science.1242072](https://doi.org/10.1126/science.1242072)
  -- a different method of the same name, cited so as not to be confused
  with it.
