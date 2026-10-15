---
version: "1"
covers: [smappy.plugins.segment.SegmentRegions.run, smappy.plugins.segment.SegmentRegions.react, smappy.plugins.segment.SegmentRegions.replaced, smappy.plugins.segment.presets, smappy.plugins.segment.made_by, smappy.plugins.npc.segment_npcs, smappy.plugins.npc.ring_kernel, smappy.plugins.npc.ring_filtered, smappy.plugins.npc.ring_quality, smappy.plugins.npc.fit_circle, smappy.plugins.roi.density_peaks, smappy.plugins.region_column.column_regions, smappy.plugins.region_column.hull, smappy.roi_manager.core.ROIProject.find]
---

## What it does

Puts an [ROI](plugin:Panels/Glossary) (region of interest: a shape kept in
the [ROI manager](plugin:Panels/ROI manager), inside which things are
measured) on every structure of a kind -- every nuclear pore, every cluster
-- so that the structures can be measured one by one, reviewed and
annotated.  It is the first of the ROI manager's three steps:

1. **Find** (this plugin): put ROIs on the structures of a file.
2. **Measure**: measure every ROI, with the per-region steps -- plugins that
   run once per ROI, such as
   [Localization Statistics](plugin:Analysis/Measure/Localization Statistics)
   or [NPC Corners](plugin:Analysis/Measure/NPC Corners).
3. **Summarise**: summarise the collection of ROIs, for example as
   [Histograms](plugin:Analysis/Regions/Summarize/Histograms).

How the structures are found is the *method*; the form shows only the chosen
method's settings:

* **ring template** -- rings of a given radius, each checked for being a
  ring: nuclear pores.
* **density peaks** -- places where localizations pile up: any compact
  structure.
* **from an id column** -- an ROI per region an id column already names,
  such as the clusters [Find clusters](plugin:Analysis/Regions/Find clusters)
  found.

A **preset** fills the settings for one kind of structure: choosing *NPC*
sets the ring template to the nuclear pore's size.  A preset is a small
text file of settings; a lab adds its own structure as a file under
`segment_presets` in smappy's configuration folder, and it appears in the
list (one of the same name replaces the shipped one).  The settings, not
the preset, are what a run uses; change any after choosing one.

*Preview* draws the ROIs it would make, grey, in the ROI manager without
adding them; *Run* adds them, as one step of the ROI manager's *Undo*.

### Ring template

Nuclear pore complexes (NPCs) are rings of eight corners, about 110 nm
across, in the flat nuclear envelope.  Imaged at a pore protein such as
Nup96, they are the reference structure of quantitative SMLM
(Thevathasan et al. 2019, [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)):
the [labelling efficiency](plugin:Panels/Glossary) (also called degree of
labelling: the fraction of target proteins that carry a detected label), the
resolution and the scale of a microscope can all be measured on them.  For
that, each pore needs its own [ROI](plugin:Panels/Glossary) (region of
interest) in the ROI manager, centred on the pore.

The ring template finds the pores in a file and puts an ROI on each.  It looks for
rings of the pore's radius, fits a circle to every candidate, and keeps those
that look like a ring: the right radius, and few localizations far from it.
The ROIs are centred on the fitted circle, which is what
[NPC Corners](plugin:Analysis/Measure/NPC Corners) and
[NPC labelling efficiency](plugin:Analysis/Regions/Summarize/NPC labelling efficiency)
need.

The checks are made to keep sparse pores.  A pore with only a few corners
labelled is still a pore, and those pores are what the labelling efficiency
is measured from; what has to go is what is not a ring at all -- blobs,
filled clumps, pieces of filament.  Use it on a 2D view of the nuclear
envelope at the bottom of the nucleus, where the pores are seen face on,
with blinks merged (one localization per blink) -- [merge
blinks](plugin:Panels/Glossary), also called grouping or linking, combines
the localizations in consecutive frames that come from one blinking event
into one.  For structures that are not rings, *density peaks* finds ROIs by
density alone.

### Density peaks

Many SMLM experiments image the same small structure hundreds of times over
one cell: nuclear pore complexes, clathrin-coated pits, centrioles,
kinetochores.  The ROI manager analyses them one by one.  It first needs a
list of [ROIs](plugin:Panels/Glossary) (regions of interest): one circle or
square centred on each structure.  Clicking hundreds of them by hand is slow,
and not reproducible.

Density peaks proposes those ROIs automatically.  It looks for places where
localizations pile up, and puts an ROI on each one that holds enough
localizations and is not too close to another.

Use it for compact, well separated structures of roughly 50 to 200 nm, on a
low background.  It looks only at how dense the localizations are, not at
their shape.  A ring, a blob and a piece of filament are all the same to it,
so on filaments or a dense meshwork it puts ROIs along every line (see the
figure of a ring and two lines under *How it works*).  The ROIs it adds can be removed, or excluded with *use*, in
the ROI manager like any other.

Despite the name, this is not the "density peaks" clustering of
[Rodriguez & Laio 2014](https://doi.org/10.1126/science.1242072).  It is a
blurred density image, its local maxima, and a count threshold.

### From an id column

Makes an ROI for every [region](plugin:Panels/Glossary) an id column of the table names -- usually
every [cluster](plugin:Panels/Glossary) (localizations lying closer to each
other than to the rest) a clusterer found (`cluster_id`, from
[Find clusters](plugin:Analysis/Regions/Find clusters) or any other) -- so the
clusters can be looked at one by one, reviewed, excluded and annotated.

It works on any column of whole numbers: an id of 0 or below means "in no
region" and gets no ROI, and so does an id with fewer than *at least*
localizations.  Use it when the clusters need a person's eye or a note; to
measure every cluster, a block of a [chain](plugin:Panels/Glossary) (plugins
run one after another as one) repeating over `cluster_id` does it without
ROIs, and is much faster for thousands of clusters.

## How it works

### Ring template

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
ring_settings = NPCSegmentSettings()
sites = segment_npcs(locs, ring_settings)
image, origin = ring_filtered(xy, ring_settings)
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
better than *judge on precision* (20 nm) -- the
[localization precision](plugin:Panels/Glossary), also called the CRLB, how
far a fitted position is expected to lie from the true one -- within the
*window* (100 nm) are
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
extent = (origin[0], origin[0] + image.shape[1] * ring_settings.bin_nm,
          origin[1], origin[1] + image.shape[0] * ring_settings.bin_nm)
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

```figure What the checks see, on made-up structures with a precision of 5 nm: the fitted circle (blue), the localizations more than 3 precisions from it (red), and the verdict.  A full pore passes, and so does a sparse one with three corners.  A single blob fits a small circle.  A filled disc fails on its far localizations, whether the centre is on it or beside it -- where the ring filter often puts it.
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
    q = ring_quality(pts[:, 0], pts[:, 1], start, ring_settings, sigma)
    c = np.asarray(q["center"])
    d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    far = np.abs(d - q["radius_nm"]) > 3 * np.hypot(5.0, ring_settings.radial_extra_nm)
    ax.scatter(pts[~far, 0] - c[0], pts[~far, 1] - c[1], s=3, c="k", linewidths=0)
    ax.scatter(pts[far, 0] - c[0], pts[far, 1] - c[1], s=3, c="#d62728", linewidths=0)
    if 0 < q["radius_nm"] < 500:
        ax.add_patch(Circle((0, 0), q["radius_nm"], fill=False, color="#2f7fd0", lw=1.2))
    ax.set_xlim(-120, 120); ax.set_ylim(-120, 120); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}\n" + ("passes" if q["use"] else "fails: " + ", ".join(q["failed"])),
                 fontsize=8)
```

### Density peaks

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
peak_settings = DensityPeakSettings()
centres = np.array(density_peaks(locs, peak_settings))
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

**6. Separation.**  A candidate closer than *separation* (150 nm) to a centre
already accepted is dropped.  Because the highest peaks come first, of two
maxima on one structure the stronger one wins.  ROIs that are already in the
file count as accepted too, so running the plugin a second time adds only
what is new.

```figure Simulated nuclear pores (Nup96, 60 % labelling) in a 2 µm piece of the field.  Left: the localizations.  Middle: the blurred density image the maxima are found in.  Right: the ROIs found (circles: the *count radius*) and the true pore centres (+).  Every pore is found, and nothing else.
from scipy.ndimage import gaussian_filter
lo, hi = 2500.0, 4500.0
inside = (x > lo) & (x < hi) & (y > lo) & (y < hi)
edges = np.arange(lo, hi + peak_settings.bin_nm, peak_settings.bin_nm)
image = np.histogram2d(y[inside], x[inside], bins=(edges, edges))[0]
blurred = gaussian_filter(image, peak_settings.sigma_nm / peak_settings.bin_nm, mode="constant")
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
        axes[2].add_patch(Circle((cx, cy), peak_settings.count_radius_nm, fill=False,
                                 color="#d62728", lw=1))
for ax, title in zip(axes, ("localizations", "blurred density", "ROIs found")):
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
axes[0].plot([lo + 100, lo + 600], [lo + 100, lo + 100], "k", lw=2)
axes[0].text(lo + 350, lo + 160, "500 nm", ha="center", va="bottom", fontsize=7)
```

```figure Density peaks on the ring and crossed lines of the `demo` structure, with a *minimum count* of 10 (left) and 100 (right).  It finds density, not shape: ROIs line the ring and the lines at about the *separation*, and at the low threshold also land on the scattered background molecules.
demo = simulate(n_frames=3000, seed=4)
dx, dy = np.asarray(demo["x_nm"]), np.asarray(demo["y_nm"])
fig.set_size_inches(7, 3.3)
for ax, count in zip(fig.subplots(1, 2), (10, 100)):
    found = np.array(density_peaks(demo, DensityPeakSettings(min_count=count)))
    ax.scatter(dx, dy, s=0.2, c="0.6", linewidths=0)
    ax.plot(found[:, 0], found[:, 1], "o", mfc="none", mec="#d62728", ms=4, mew=0.8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"minimum count {count}: {len(found)} ROIs", fontsize=9)
```

### From an id column

**The regions.**  The localizations are sorted by their id; each id above
0 with enough localizations is one region.  Membership is the column alone:
the layers' filters do not decide which localizations a cluster has.

**The outline.**  With *the cluster's own outline* ticked (the default), the
ROI is the convex hull of the cluster's positions -- the smallest shape with
no dents that holds them all -- so it fits a cluster of any shape and size.
A localization exactly on the hull's edge or at a corner counts as inside.
Unticked, only the cluster's centre is placed, and the ROI is the circle or
square of the Regions tab, at its size: the right choice when the clusters
are alike (pores, say) and every ROI should be measured in the same window.
Such ROIs follow the Regions tab's shape and size when they are changed
later, and *Clear shape* in the ROI manager turns a hull ROI into one.  A
cluster whose localizations all lie on one line has no hull; it gets the
centre and the Regions tab's shape even with the outline ticked, and the
text says how many did.

**The ROI.**  Its centre is the mean position of the cluster's
localizations, its file the one its localizations came from, and it
remembers the column, the id and the number of localizations (`origin`).
All the ROIs of a run are one step of the ROI manager's undo.

## In detail

### Ring template

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
*around* it, so the centre lands beside the clump, and from there the clump is
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
shown, on its localizations after the layer's filters (the lowest and
highest values of columns, set in the Render tab), and on the merged blinks
when the layer merges them.  The ROIs get the manager's size and
shape (a circle of 300 nm by default).  Each ROI records how it was found
(the plugin and the method, its version, its settings, the filters) and its numbers: the
fitted radius, the localizations on the ring, the precise ones, the fraction
far, and the checks it failed.

### Density peaks

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
the labelled side, by about 15 nm (median) in the simulation above.  A
per-region step that fits a model to the ROI refines it further.

**Separation.**  $\mathbf{c}'$ is rejected if it lies closer than the
*separation* to any accepted centre, including the centres of the ROIs
already in that file.  This is a greedy non-maximum suppression in
nanometres: which of two close candidates survives is decided by the order,
highest density first.

**What it runs on.**  In the ROI manager, the plugin runs on the file being
shown, on its localizations after the layer's filters (the lowest and
highest values of columns, set in the Render tab), and on its merged blinks
when the layer merges them -- [merge blinks](plugin:Panels/Glossary), also
called grouping or linking, combines the localizations in consecutive frames
that come from one blinking event into one.  The new ROIs get the manager's
global size and shape (a circle of 300 nm by default), which has nothing to
do with the *count radius*: detection and analysis are set separately.  Each
ROI records how it was found: the plugin and the method, its version, its
settings, the filters and the merge settings.

### From an id column

**Which table.**  The localizations, whatever the layer shows: the id
column is theirs, and a region's outline is drawn round all of its
localizations.

## Parameters

### replace
On, a second run gives a fresh answer: the ROIs the same *method* made on
the file before go first (each ROI's origin says which) -- for *from an id
column*, those it made of the same column, in every file.  ROIs drawn by
hand or made by another method stay either way: density peaks run after the
ring template keeps the pores.  Off, a run adds to what is there; for the
ring template and density peaks a candidate near an existing ROI is
suppressed, so running again with the same settings finds nothing new,
which is why it is on by default.  The ROI manager's *Undo* brings replaced
ROIs back.

**Ring template.**  The defaults are for Nup96 and other proteins of the pore's cytoplasmic and
nuclear rings, at a radius of about 54 nm.  For a protein at another radius,
change *radius* and the radius range together.

### ring.ring_width_nm
About the localization precision plus the label's size, 10 to 20 nm.

### ring.min_locs
At most the fewest corners the analysis is fitted from (5): see *In detail*.

### ring.precision_nm
The same as NPC Corners' *corner precision*, so that the pores are judged on
the localizations they are counted with.  Looser lets more imprecise ones
into the far test, which they pass anyway; tighter leaves sparse pores too
few to judge (3 are needed).

### ring.min_radius_nm
With *max fitted radius*, the band around the expected radius.  Look at the
fitted radius in the *checks* figure: real pores make a narrow peak.

### ring.max_far
On a pore about 0.3 % of the localizations are more than three precisions
off; 1 % leaves room for neighbours and the tilt.  Raise it if the *checks*
figure shows real pores rejected for *far*.

### ring.keep_rejected
Tick it while choosing the settings: each rejected candidate becomes an ROI
that is not used, and its origin lists the checks it failed.  *Preview*
shows them too, dotted, without making them ROIs.

**Density peaks.**  The defaults suit compact structures of about 100 nm, such as nuclear pores.

### peaks.bin_nm
Smaller than the *smoothing*, or the blur has nothing to smooth over; 10 to
25 nm is typical.  Only the speed and the memory depend on it much.

### peaks.sigma_nm
About the radius of the structure.  Much smaller splits one structure into
several maxima (the *separation* then has to remove them); much larger merges
neighbouring structures into one.

### peaks.separation_nm
A bit more than the diameter of the structure and less than the typical
distance between two of them.  Too small gives two ROIs on one structure;
too large loses one of two close neighbours.

### peaks.count_radius_nm
About the radius of the structure plus the
[localization precision](plugin:Panels/Glossary) (also called the CRLB), so that
the count and the re-centring see the whole structure but not its
neighbours.

### peaks.min_count
The most useful setting for a given data set: it separates structures from
background.  Look at how many localizations a real structure has (Statistics
reports it) and set it well below that, and well above what a patch of
background of the same size holds.

### ids.column
Any column of whole numbers: `cluster_id`, an id a dropped-in clusterer
wrote, or `roi_id` from *Regions to column* to bring ROIs back from a table.

### ids.min_size
Large clusters are usually the interesting ones and small ones are noise:
a few times the localizations one fluorophore gives.  The column keeps every
cluster either way, and a filter on `n_in_cluster` does the same on the
table.

### ids.outline
The hull follows each cluster but makes ROIs of different sizes; the circle
or square makes them all alike, cutting off any cluster larger than it, and
taking in the localizations around a smaller one.

## Output

### Ring template

* **ROIs**, one per pore, added to the current file in the ROI manager and
  centred on the fitted circle, used from the start; with *keep rejected*,
  the rejected candidates too, not used.
* **The text** says how many pores were kept of how many candidates, and how
  often each check failed.
* **The main figure** shows the localizations with every candidate's fitted
  circle: kept in blue, rejected in grey.
* **A preview** (the *Preview* button) finds the same pores, writes the
  same text and figures, and adds nothing: the pores are drawn grey and
  dashed in the ROI manager (with *keep rejected*, the rejected candidates
  dotted), and with *replace* the earlier pores it would remove are hidden.
  The next preview replaces it; *Run* adds it.
* **The *checks* figure** has the fitted radius, the fraction far from the
  ring and the precise localizations per candidate, kept and rejected
  stacked, the radius limits in red.

Run without the ROI manager, from a script (`Context(locs=locs)`), it returns
every candidate with its numbers in `data["sites"]`, and the centres of the
pores that passed in `data["centers"]`.

### Density peaks

* **ROIs**, one per structure found, added to the current file in the ROI
  manager, brightest first.  They are included (*use*) from the start.
* **The text** says how many candidates were found.

*Preview* finds the same ROIs and adds none: they are drawn grey and
dashed in the ROI manager, and the next preview replaces them.  Try
settings with it until the ROIs sit on the structures, then *Run*.

A good result has one ROI on each structure and none on the background.
ROIs on empty background: raise *minimum count*.  Faint structures missed:
lower it.  Two ROIs on one structure: raise *separation* or *smoothing*.
Two structures under one ROI: lower them.

Run without the ROI manager (from a script, `Context(locs=locs)`), it
returns only the list of centres, in `data["centers"]`.

### From an id column

The ROIs in the ROI manager; the text says how many were made, the smallest
cluster that became one, and how many clusters had no outline.  It cannot run
inside a block, which repeats over regions: make the ROIs after it.

## Differences from SMAP

### Ring template

Ported from SMAP's `ROIManager/Segment/segmentNPC` and its clean-up evaluator (SMAP's word for a plugin run in every ROI)
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
  marks the bad sites (ROIs) as not used.  Here the radius is kept and the fractions
  are replaced by the far test on the precise localizations, for the reasons
  in *In detail*; the PSF width is a filter on `sigma_nm` in the layer.
* **One step.**  Segmenting and judging happen in one run, and the rejected
  candidates are dropped unless *keep rejected* is ticked.
* **The fits** use a soft-L1 loss on the precise localizations; SMAP's
  `fitposring` is plain least squares on all of them.
* SMAP works per cell, and needs cells first; this works on the whole file.

### Density peaks

SMAP has no single counterpart; its segmenters are specific to a structure.
Density peaks replaces the generic part of `ROIManager/Segment/segmentCME` and
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
  gives a pore a single peak.  For nuclear pores, the ring template keeps
  SMAP's ring filter and judges each candidate's shape as well.

### From an id column

New: SMAP has no segmenter that makes sites of clusters; its DBSCAN writes
the cluster index and makes no ROIs.

### One plugin

SMAP has a segmenter per structure in `ROIManager/Segment` -- `segmentNPC`,
`segmentCME`, `segmentBeads`, ... -- each with its own settings.  Here one
plugin has a method per algorithm, and a structure is a preset of values, so
a structure the methods already find needs a file, not a plugin.
`segmentCME` blurs the rendered image and keeps the maxima above a cutoff,
which is what density peaks does: clathrin-coated pits are a preset of it.
`segmentBeads` judges beads by their frames and PSF in a bead stack, which
none of the methods reads, so beads would be a method of their own.

## References

* Thevathasan JV, Kahnwald M, Cieśliński K, et al. Nuclear pores as
  versatile reference standards for quantitative superresolution
  microscopy. *Nat Methods* 16, 1045 (2019).
  [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)
  -- the ring template's method, and the Nup96 pores simulated in the figures.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
* Rodriguez A, Laio A. Clustering by fast search and find of density peaks.
  *Science* 344, 1492 (2014).
  [doi:10.1126/science.1242072](https://doi.org/10.1126/science.1242072)
  -- a different method of the same name, cited so as not to be confused
  with it.
