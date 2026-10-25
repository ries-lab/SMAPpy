---
version: "2"
covers: [smappy.plugins.segment.SegmentRegions.run, smappy.plugins.segment.SegmentRegions.react, smappy.plugins.segment.SegmentRegions.replaced, smappy.plugins.segment.SegmentRegions.describe, smappy.plugins.segment.candidates, smappy.plugins.segment.presets, smappy.plugins.segment.made_by, smappy.plugins.segment.draw_as, smappy.plugins.segment.place_positions, smappy.plugins.segment.draw_values, smappy.segmentation.find_circles, smappy.segmentation.ring_kernel, smappy.segmentation.disc_kernel, smappy.segmentation.radii, smappy.segmentation.density_image, smappy.segmentation.filter_radii, smappy.segmentation.peaks, smappy.segmentation.too_close, smappy.segmentation.measure_ring, smappy.segmentation.measure_disc, smappy.segmentation.measure_beads, smappy.segmentation.mark_touching, smappy.segmentation.judge, smappy.segmentation.keep_test, smappy.segmentation.clauses, smappy.segmentation.read_positions, smappy.segmentation.fit_circle, smappy.segmentation.roundness, smappy.fiducials.find_beads, smappy.roi_manager.order.condition, smappy.roi_manager.core.segment_columns, smappy.roi_manager.core.ROIProject.draws_track, smappy.plugins.region_column.column_regions, smappy.plugins.region_column.hull, smappy.roi_manager.core.ROIProject.find]
---

## What it does

Puts an [ROI](plugin:Panels/Glossary) (region of interest: a shape kept in
the [ROI manager](plugin:Panels/ROI manager), inside which things are
measured) on every structure of a kind -- every nuclear pore, every
clathrin-coated pit, every bead, every cluster -- so that the structures can
be measured one by one, reviewed and annotated.  It is the first of the ROI
manager's three steps:

1. **Find** (this plugin): put ROIs on the structures of a file.
2. **Measure**: measure every ROI, with the per-region steps -- plugins that
   run once per ROI, such as
   [Localization Statistics](plugin:Analysis/Measure/Localization Statistics)
   or [NPC Corners](plugin:Analysis/Measure/NPC Corners).
3. **Summarise**: summarise the collection of ROIs, for example as
   [Histograms](plugin:Analysis/Regions/Summarize/Histograms).

How the structures are found is the *method*, a model of the structure; the
form shows only the chosen method's settings:

* **circle template** -- rings or filled discs, of one radius or of any
  radius in a range: nuclear pores, clathrin-coated pits, any round
  structure.
* **beads** -- bright spots seen in many frames: fiducial beads, for drift
  correction and stability measurements.
* **from an id column** -- an ROI per region an id column already names,
  such as the clusters [Find clusters](plugin:Analysis/Regions/Find clusters)
  found.
* **positions from a file** -- an ROI at each position of a table, such as
  one written by [Export table](plugin:Analysis/Regions/Summarize/Export table).

The circle template and beads measure every candidate they find -- its
radius, how round it is, how many localizations it has, and what its model
adds -- and **keep** decides which candidates become ROIs: one condition on
those values, written as in [Filter ROIs](plugin:Analysis/Regions/Filter ROIs),
such as `n_ring >= 4 and 40 <= radius_nm <= 70`.  The values stay with each
ROI and are columns of the ROI manager's results table, so the ROIs can
later be sorted and filtered by them like anything measured.

A **preset** fills the settings for one kind of structure, *keep* included:

* *NPC* -- a ring of 55 nm, and the checks that a candidate is a pore;
* *CME* -- filled discs of 30 to 150 nm radius, with at least 20
  localizations;
* *Circles* -- rings of 100 to 500 nm, round, with at least 10
  localizations on the ring.

Without a preset, the settings are discs of 30 to 150 nm with at least 10
localizations: compact structures of any kind, as a first look.

A preset is a small text file of settings; a lab adds its own structure as a
file under `segment_presets` in smappy's configuration folder, and it appears
in the list (one of the same name replaces the shipped one).  The settings,
not the preset, are what a run uses; change any after choosing one.

*Preview* draws the ROIs it would make, grey, in the ROI manager without
adding them; *Run* adds them, as one step of the ROI manager's *Undo*.

### Circle template

**Nuclear pores.**  Nuclear pore complexes (NPCs) are rings of eight
corners, about 110 nm across, in the flat nuclear envelope.  Imaged at a pore
protein such as Nup96, they are the reference structure of quantitative SMLM
(Thevathasan et al. 2019, [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)):
the [labelling efficiency](plugin:Panels/Glossary) (also called degree of
labelling: the fraction of target proteins that carry a detected label), the
resolution and the scale of a microscope can all be measured on them.  For
that, each pore needs its own ROI, centred on the pore, which is what
[NPC Corners](plugin:Analysis/Measure/NPC Corners) and
[NPC labelling efficiency](plugin:Analysis/Regions/Summarize/NPC labelling efficiency)
read.  The *NPC* preset looks for rings of the pore's radius, fits a circle
to every candidate, and keeps those that look like a ring: enough
localizations on it, the right radius, and few localizations far from it.

The checks are made to keep sparse pores.  A pore with only a few corners
labelled is still a pore, and those pores are what the labelling efficiency
is measured from; what has to go is what is not a ring at all -- blobs,
filled clumps, pieces of filament.  Use it on a 2D view of the nuclear
envelope at the bottom of the nucleus, where the pores are seen face on,
with blinks merged (one localization per blink) -- [merge
blinks](plugin:Panels/Glossary), also called grouping or linking, combines
the localizations in consecutive frames that come from one blinking event
into one.

**Pits and other filled structures.**  With *shape* set to *disc*, the
template is a filled disc: clathrin-coated pits, small clusters, anything
compact and roundish.  Over a range of radii it finds each structure at its
own size, and measures that size.  `centre_ratio` says whether a disc is
filled evenly (about 1) or empty in the middle (near 0): a pit late in its
growth, which looks like a ring, has a low one -- a value to sort or filter
on, not a second class.

**Touching structures** are flagged (`touching` is 1) and not split; a
*keep* with `touching == 0` leaves them out.

### Beads

Fiducial beads -- fluorescent beads on the coverslip -- are bright and seen
in nearly every frame, which makes them the reference for
[drift](plugin:Panels/Glossary) and for the stability of the microscope.
The beads method finds them as bright, long tracks, with the same linker as
[Fiducials drift correction](plugin:Analysis/Drift/Fiducials), and puts an
ROI on each.  [Fiducials](plugin:Analysis/Drift/Fiducials) with *beads from*
set to *ROI manager* then takes each used ROI as a bead, and
[Stability](plugin:Analysis/Measure/Stability) runs once per ROI -- so the
beads can be looked at, excluded or annotated before either.

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

It is also how **tracks** are explored.  With *an ROI holds* set to *its id
alone* and the column `track_id` (from
[Link tracks](plugin:Analysis/Tracking/Link tracks)), each
[track](plugin:Panels/Glossary) becomes an ROI that holds that track's
localizations and nothing else -- not the other tracks that cross its path,
which any shape drawn round it would take in.  The ROI manager draws it as
its line, selects it by a click on the line, and lists it beside the other
ROIs, with *use*, keywords, and a frame range and z range of its own.

### Positions from a file

Puts an ROI at each position of a table: a csv or tsv with `x_nm` and
`y_nm`, and optionally the file each position belongs to.  The ROIs get the
ROI manager's size and shape.  Use it to bring back the sites of an earlier
analysis -- *Export table* with *ROI position* ticked writes such a table --
or sites chosen in another program.

## How it works

### Circle template

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy import segmentation
from smappy.segmentation import (VALUES, density_image, filter_radii, find_circles,
                                 judge, measure_ring, radii)
from matplotlib.patches import Circle
NPC_KEEP = "n_ring >= 4 and 40 <= radius_nm <= 70 and far_p >= 0.05"
sim = SimulationSettings(n_frames=3000, seed=3,
                         structure=StructureSettings(preset="npc"),
                         labelling=LabellingSettings(efficiency=0.6))
locs = simulate(sim)
x, y = np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])
xy = np.column_stack((x, y))
sites = find_circles(locs, "ring", 55.0, 55.0)
judge(sites, NPC_KEEP, VALUES["ring"])
image, origin = density_image(xy, 10.0, 55.0 + 3 * 15.0)
score, response, _ = filter_radii(image, 10.0, "ring", [55.0], 15.0)
truth = np.column_stack((locs.metadata["copies"]["x_nm"],
                         locs.metadata["copies"]["y_nm"]))

def disc_field(radii_nm, seed=0, precision=8.0, background=400):
    """Filled discs on a grid with uniform background, as the tests make them."""
    rng = np.random.default_rng(seed)
    radii_nm = np.asarray(radii_nm, float)
    centres = np.array([(600.0 * (i % 7), 600.0 * (i // 7)) for i in range(len(radii_nm))])
    pts = []
    for (cx, cy), R in zip(centres, radii_nm):
        n = int(400 * (R / 100) ** 2) + 60
        r = R * np.sqrt(rng.uniform(0, 1, n)); a = rng.uniform(0, 2 * np.pi, n)
        pts.append(np.column_stack((cx + r * np.cos(a), cy + r * np.sin(a))))
    pts = np.concatenate(pts) + rng.normal(0, precision, (sum(map(len, pts)), 2))
    lo, hi = centres.min(axis=0) - 300, centres.max(axis=0) + 300
    pts = np.concatenate([pts, rng.uniform(lo, hi, (background, 2))])
    from smappy.locs import Localizations
    n = len(pts)
    return Localizations({"x_nm": pts[:, 0], "y_nm": pts[:, 1], "frame": np.arange(n),
                          "xy_err_nm": np.full(n, precision)}), centres, radii_nm
```

**1. A density image.**  The localizations are binned into an image with
pixels of *bin* (10 nm).  Its square root is taken, so that one molecule that
blinks many times in one place does not outweigh a whole structure.

**2. A template at each radius.**  The image is filtered with the shape's
template.  The ring is a ring of the radius, minus a disc of the same total
weight; the disc is a filled disc, minus the ring-shaped band around it, of
the same weight.  At each position the filter gives the localizations on the
shape minus what the surround says is background: an empty area gives zero,
and so does an evenly covered one, while a structure of the template's shape
gives a strong peak at its centre.  With *min radius* and *max radius*
different, this is done at several radii, each about 15 % (or a *ring
width*) larger than the last.

**3. The best radius.**  At each position the radius that matches best wins
-- the one with the best signal against the background's noise -- and it is
the structure's first size estimate.

**4. Candidates.**  The peaks of the filtered image are candidates, taken
strongest first.  A candidate whose circle overlaps a stronger one's by more
than half the smaller radius is dropped: two pores of 55 nm may be as close
as 82.5 nm, and then they touch.

**5. Measuring each candidate.**  Each candidate is measured on its
localizations within about 1.5 times its radius (for a ring, as far as its
template reaches).  A ring's centre is placed
by fitting a circle of the template's radius to the localizations better
than *judge on precision* (20 nm) -- the [localization
precision](plugin:Panels/Glossary), also called the CRLB, how far a fitted
position is expected to lie from the true one; a disc's is the mean position
of its localizations.  Then its values: the radius (for a ring, a circle
fitted with its radius free; for a disc, from how far its localizations lie
from the centre), how round it is, how many localizations it has, and its
shape's own values -- for a ring how many lie on it, inside it and outside
it, and whether too many lie far from it for their precision; for a disc how
dense its centre is against its edge.  Candidates are measured strongest
first, and each one kept claims its localizations: a weaker candidate is
measured only on what is left, so the arc of a pore that the window of a
candidate beside it takes in is not a ring of its own.

**6. Keep.**  *keep* is checked on every candidate's values; those that pass
become ROIs.  A candidate that overlaps one that passes is marked `touching`.

```figure A 2 µm piece of a simulated nuclear envelope (Nup96, 60 % labelling), with the *NPC* preset.  Left: the localizations.  Middle: the ring-filtered density, bright at the centre of every pore.  Right: the pores kept, each with its fitted circle, and the true centres (+).
lo, hi = 2500.0, 4500.0
inside = (x > lo) & (x < hi) & (y > lo) & (y < hi)
fig.set_size_inches(7.5, 2.7)
axes = fig.subplots(1, 3)
axes[0].scatter(x[inside], y[inside], s=1.2, c="k", linewidths=0)
extent = (origin[0], origin[0] + image.shape[1] * 10.0,
          origin[1], origin[1] + image.shape[0] * 10.0)
axes[1].imshow(np.clip(response, 0, None), origin="lower", extent=extent, cmap="magma")
axes[2].scatter(x[inside], y[inside], s=1.2, c="0.6", linewidths=0)
t = truth[(truth[:, 0] > lo) & (truth[:, 0] < hi) & (truth[:, 1] > lo) & (truth[:, 1] < hi)]
axes[2].plot(t[:, 0], t[:, 1], "+", color="k", ms=6)
for s in sites:
    cx, cy = s["center"]
    if lo < cx < hi and lo < cy < hi and s["use"]:
        axes[2].add_patch(Circle((cx, cy), s["values"]["radius_nm"], fill=False,
                                 color="#2f7fd0", lw=1))
for ax, title in zip(axes, ("localizations", "ring-filtered", "pores kept")):
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
axes[0].plot([lo + 100, lo + 600], [lo + 100, lo + 100], "k", lw=2)
axes[0].text(lo + 350, lo + 160, "500 nm", ha="center", va="bottom", fontsize=7)
```

```figure What the *NPC* preset's *keep* sees, on made-up structures with a precision of 5 nm: the fitted circle (blue), the localizations more than 3 precisions from it (red), and the verdict -- the clauses of *keep* a candidate fails.  A full pore passes, and so does a sparse one with three corners.  A single blob fits a small circle.  A filled disc fails on its far localizations, whether the centre is on it or beside it -- where the ring filter often puts it.
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
fig.set_size_inches(9, 2.5)
for ax, (name, pts, start) in zip(fig.subplots(1, 5), cases):
    sigma = np.full(len(pts), 5.0)
    q = measure_ring(pts[:, 0], pts[:, 1], start, 55.0, 15.0, sigma)
    judge([q], NPC_KEEP, VALUES["ring"])
    c = np.asarray(q["center"]); radius = q["values"]["radius_nm"]
    d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    far = np.abs(d - radius) > 3 * np.hypot(5.0, segmentation.RING_SPREAD_NM)
    ax.scatter(pts[~far, 0] - c[0], pts[~far, 1] - c[1], s=3, c="k", linewidths=0)
    ax.scatter(pts[far, 0] - c[0], pts[far, 1] - c[1], s=3, c="#d62728", linewidths=0)
    if 0 < radius < 500:
        ax.add_patch(Circle((0, 0), radius, fill=False, color="#2f7fd0", lw=1.2))
    ax.set_xlim(-120, 120); ax.set_ylim(-120, 120); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    verdict = "passes" if q["use"] else "fails:\n" + "\n".join(q["failed"])
    ax.set_title(f"{name}\n{verdict}", fontsize=7)
```

```figure Filled discs of 30 to 150 nm radius (8 nm precision, scattered background), found with *shape* disc between 30 and 150 nm.  Left: the localizations and the discs found, each drawn at its measured radius.  Right: the measured radius against the true one, within 15 % (dashed) of it.
field, centres, true_r = disc_field([30, 45, 60, 80, 100, 120, 150] * 2)
found = find_circles(field, "disc", 30.0, 150.0)
judge(found, "n_locs >= 20", VALUES["disc"])
kept = [s for s in found if s["use"]]
from scipy.spatial import cKDTree
_, which = cKDTree(centres).query([s["center"] for s in kept])
fig.set_size_inches(8, 3.0)
left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2.2, 1]})
left.scatter(field["x_nm"], field["y_nm"], s=0.4, c="k", linewidths=0)
for s in kept:
    left.add_patch(Circle(s["center"], s["values"]["radius_nm"], fill=False,
                          color="#2f7fd0", lw=1))
left.set_aspect("equal"); left.set_xticks([]); left.set_yticks([])
measured = np.array([s["values"]["radius_nm"] for s in kept])
right.plot(true_r[which], measured, "o", ms=4, color="#2f7fd0")
line = np.array([20, 160])
right.plot(line, line, "k-", lw=0.8)
right.plot(line, 1.15 * line, "k--", lw=0.6); right.plot(line, 0.85 * line, "k--", lw=0.6)
right.set_xlabel("true radius (nm)"); right.set_ylabel("measured radius (nm)")
```

### Beads

**1. Bright localizations.**  Only localizations of at least *min photons*
can be a bead.

**2. Tracks.**  They are linked into tracks frame to frame, a step being at
most *maximum jump* and a track allowed to miss up to *memory* frames -- the
linker of [Link tracks](plugin:Analysis/Tracking/Link tracks).

**3. Long ones.**  A track seen in at least *min length* of the frames is a
bead.

**4. The ROI.**  Each bead gets an ROI at the median position of its
localizations, at the ROI manager's size and shape, and is measured: the
fraction of the frames it was seen in (`frames_fraction`), its median
`photons`, and the spread of its positions (`spread_nm`), which includes the
drift.  *keep* then decides, as for the circle template.

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

**Its id alone.**  With *an ROI holds* set to *its id alone*, the ROI has
no shape: it holds the localizations with its id, wherever they are, and
the outline settings do not apply.  The ROI manager draws it from its
localizations -- with *draw it as* a track, a line through them in frame
order; as an outline, the hull round them.  On *auto* a column whose name
contains `track` (or MINFLUX's `tid`) is drawn as tracks and any other as
outlines.  Drawing a polygon on such an ROI in the ROI manager makes it the
part of its id inside the polygon; *Clear shape* makes it the whole id
again.

**The ROI.**  Its centre is the mean position of the cluster's
localizations, its file the one its localizations came from, and it
remembers the column, the id and the number of localizations (`origin`).
All the ROIs of a run are one step of the ROI manager's undo.

### Positions from a file

**The table.**  The columns for x and y are found by their headers as the
csv loader finds them, in nm unless the header says otherwise.

**The file.**  A position goes to the file its row names, by `file_id` (as
*Export table* writes it, for the same session), `file` (the file's name) or
`filenumber`; without any of these, to the file the ROI manager shows.

## In detail

### Circle template

**The templates.**  With $b$ the *bin*, $R$ the radius and $w$ the *ring
width*, the ring's template is

$$K_{\mathrm{ring}}(r) = g(r) - [r \leq R + 2w] \frac{\sum g}{\sum [r \leq R + 2w]}, \quad g(r) = e^{-(r - R)^2 / 2w^2},$$

on a grid of pixels of size $b$ reaching to $R + 3w$, with $[\cdot]$ one where
the condition holds and zero elsewhere.  The disc's is

$$K_{\mathrm{disc}}(r) = [r \leq R] - [R < r \leq \sqrt{2} R] \frac{\sum [r \leq R]}{\sum [R < r \leq \sqrt{2} R]},$$

a disc minus the band out to $\sqrt{2} R$, which has the same area.  Both sum
to zero.  They are applied by an FFT convolution to $\sqrt{n}$, $n$ the
localizations per pixel, which where a pixel holds one localization at most
-- nearly everywhere at 10 nm -- is $n$ itself.  The disc's response is then
the localizations inside it less those in its band; the ring's, whose
surround is a disc that takes in the ring too, is the localizations on the
ring less the disc's mean density times the ring's weight, about 0.4 of the
localizations on the ring at the pore's size.  The image covers the
localizations with a margin of the largest template and may have at most 16
million pixels (80 µm by 80 µm at 20 nm); a larger one is refused with a
request to increase the bin.

**The radii and the best one.**  From *min radius*, each radius is the last
plus the larger of 15 % of it and, for a ring, the *ring width*, up to *max
radius*.  At each pixel the radius with the largest
$\mathrm{response} / \sqrt{W}$ wins, $W$ the sum of the template's positive
part -- the signal against the Poisson noise of a background over the same
area.  The response alone would not do for a disc: once the template holds
the whole disc, every larger radius answers the same count.

**Candidates.**  A pixel is a candidate when it is the largest score within a
square about *min radius* wide and its response is above zero.  Taken by
score, a candidate is dropped when its circle -- of its template radius --
and a stronger one's overlap by more than half the smaller radius,

$$d < r_1 + r_2 - \frac{1}{2} \min(r_1, r_2),$$

checked at the filter's maximum and again where its fit moved it.  An ROI
already in the file counts as a circle of the candidate's own radius.  A
candidate needs 3 localizations of its own in its window, the fewest a
circle is fitted to.

**Claimed localizations.**  Each candidate claims its localizations -- a
ring those out to $R + 2w$ from its centre, as far as the filter's disc
reaches, a disc those inside its measured radius -- whether *keep* passes it
or not.  Every weaker candidate is fitted and counted only on the
localizations not claimed (a ring's far test still sees all of its window);
one with fewer than 3 left is dropped.  An ROI already in the file claims
what a candidate of *min radius* would.  Without this, a candidate about
100 nm beside a pore, its window taking in an arc of the pore, passed the
*NPC* checks next to 6 of 141 simulated pores, and at 500 photons, where a
pore's localizations spread far past its band, next to a third of them.

**A ring's values.**  The window is $\max(1.5 R, R + 3w)$, as far as the
template reaches -- 100 nm for a pore: a sparse pore found 50 nm off its
centre, as happens at 500 photons, fails the far test only when the window
takes in the far side of the pore.  With these rules the *NPC* preset finds
what the ring template it replaced found: on the simulations of
`studies/npc_le` with junk, over 18 conditions of labelling, photons and
blinks, the pores found and the junk kept were within 1 % and 2 % of its,
and the fitted labelling efficiency within 0.017.  The centre is
fitted twice with the radius fixed at $R$, the second time on the window
around the first answer; the fits minimise $\sum_i \rho(d_i - R)$, $d_i$ the
distance of localization $i$ from the centre, with the soft-L1 loss
$\rho(z) = 2w^2\left(\sqrt{1 + z^2/w^2} - 1\right)$: a localization far from
the circle counts with its distance, not its square.  They are minimised by
Gauss-Newton with reweighting, up to 30 iterations, on the localizations
better than *judge on precision*; when fewer than 3 are, or the table has no
precision column, on all of them.  `radius_nm` is the same fit with the
radius free.  `n_ring`, `n_inside` and `n_outside` count every localization
in the window in the band $R \pm w$, nearer the centre, and further out.

**The far test** (`far_p`).  With $\hat{R}$ the fitted radius and $\sigma_i$
the precision (`xy_err_nm`) of each of the $n$ precise localizations,

$$z_i = \frac{d_i - \hat{R}}{\sqrt{\sigma_i^2 + e^2}},$$

$e$ = 5 nm for the tilt of the pore and the size of the label.  A
localization is far when $|z_i| > 3$, which on a pore happens 0.3 % of the
time.  With $m$ far ones, $\mathrm{far\_p} = P(X \geq m)$ for $X$ binomial
with $n$ trials and a chance of 1 %.  Two things follow.  The test is the
same at 500 photons as at 5000, because each localization is judged by its
own precision; and it needs nothing about how often a fluorophore blinks.
Without a precision column it is 1.  Two touching pores fail it: each sees
the other's localizations in its window, far from its own ring.

**Why not the fractions inside and outside** as the checks.  Counted on
every localization, as SMAP's clean-up does, those fractions do not tell a
pore from a clump at low photon numbers: a pore's imprecise localizations
land inside and outside too.  And a filled clump the size of a pore passes
them, because the ring filter answers a filled disc with a ring of response
*around* it, so the centre lands beside the clump, and from there the clump
is an arc of the right radius with nothing inside.  On simulations with
pore-sized clumps among the pores, the far test kept the clumps that reached
the analysis to 3 to 5 per 300 pores at 5000 photons (16 to 19 with the
fractions); see `studies/npc_le`.  `n_inside` and `n_outside` are there to
look at all the same.

**Why `n_ring >= 4`.**  Every localization in the band counts, precise or
not.  A pore that shows $k$ corners has at least $k$ of them, so with the
analysis fitted from 5 corners, any value up to 5 loses no pore the analysis
uses; 4 removes most of the background.  A higher value removes the sparse
pores that still show 5 corners: 10 found 27 % of the pores showing 4
corners in the simulations of `studies/npc_le`, and 14 % with one blink per
fluorophore at 500 photons.

**A disc's values.**  The window is $1.5 R$.  The centre is the mean
position of the localizations in the window, taken twice, the second time
around the first.  With $d_i$ the distances from it of the precise
localizations in the window,

$$\hat{R} = \sqrt{2 \left( \langle d^2 \rangle - \langle \sigma^2 \rangle \right)},$$

the radius of a uniform disc with that spread, less the localizations' own
($\langle \sigma^2 \rangle$; nothing without a precision column).  It is
taken three times, the last two over the localizations within
$\hat{R} + 2\sqrt{\langle \sigma^2 \rangle}$ (or $1.1 \hat{R}$ without a
precision): one background localization at the window's edge, counting with
its distance squared, made a 30 nm disc 16 % too large.  On the simulated
discs above the radius is within 15 % from 30 to 150 nm.  `centre_ratio` is
the density within $\hat{R}/2$ over the density between $\hat{R}/2$ and
$\hat{R}$: three times the ratio of the counts, as the inner circle is a
quarter of the area.

**The shared values.**  `n_locs` is the localizations in the window that no
stronger candidate claimed.  `roundness` is $\sigma_{\mathrm{short}} / \sigma_{\mathrm{long}}$,
the square roots of the eigenvalues of the covariance of the precise
localizations in the window -- as [Cluster shape](plugin:Analysis/Measure/Cluster shape)
computes them: 1 for a round structure, $b/a$ for an ellipse of half-axes
$a$ and $b$.  A ring's own window takes in some background, which makes it
read a little rounder than it is.  `psf_nm` is the mean `sigma_nm` in the
window -- SMAP's check that a structure is in focus -- and NaN without that
column.  `touching` is 1 when the candidate's circle (of `radius_nm`, or of
the template's radius when that is not a number) overlaps at all the circle
of another candidate that passes *keep* -- *keep* taken first with every
`touching` 0.  The rejected candidates are not counted: most are background,
beside every structure, and would flag them all.

**What it runs on.**  In the ROI manager, the file being shown, on its
localizations after the layer's filters (the lowest and highest values of
columns, set in the Render tab) -- so a z or frame range set there limits
the search -- and on the merged blinks when the layer merges them.  The ROIs
get the manager's size and shape (a circle of 300 nm by default).

### Keep

*keep* is read by the parser of *Filter ROIs*, never run as code:
comparisons (chained too, `40 <= radius_nm <= 70`), `and`, `or`, `not`,
arithmetic, numbers and `in [...]`.  A value that is not a number (NaN, as
`radius_nm` is with fewer than 3 precise localizations) makes every
comparison with it false.  A name the method does not measure is an error
naming those it does, and so is `psf_nm` on a table without `sigma_nm`: a
condition on a value that is not there would otherwise reject, or keep,
every candidate silently.  An empty *keep* keeps every candidate.  A
rejected candidate lists the parts of *keep* joined by its top-level `and`
that it fails.

### Beads

**The tracks** are `find_beads` of [Fiducials](plugin:Analysis/Drift/Fiducials):
the localizations of at least *min photons*, linked frame to frame by the
greedy linker, kept when seen in at least the *min length* fraction of the
frames -- the last frame plus one, or the file's frame count when it records
one.  A bead within three times its spread of an ROI already in the file is
that ROI's, and not proposed again.

**Which table.**  The localizations of the file shown, under the layer's
filter.  Beads are tracks of single localizations, so a layer that merges
blinks is refused: a bead seen in every frame would be one merged blink.

**The values.**  `frames_fraction` is the number of different frames the
bead was seen in over all frames; `photons` the median; `spread_nm` is
$\sqrt{(\mathrm{var}\, x + \mathrm{var}\, y) / 2}$, drift included;
`radius_nm` $\sqrt{2 \langle d^2 \rangle}$ about the median position, as a
disc's; `n_locs` its localizations; `roundness` and `psf_nm` as above.

### From an id column

**Which table.**  The localizations, whatever the layer shows: the id
column is theirs, and a region's outline is drawn round all of its
localizations.

**An ROI of its id alone** keeps `{column: id}` (`ROI.ids`) with its shape
switched off (`ROI.spatial`).  Its rows are found through one sort of the
column, kept while the column is unchanged, so thousands of such ROIs cost
a binary search each rather than a pass over the table.  On a layer that
merges blinks, it holds the blinks its localizations belong to, through
`group_id`.

### Positions from a file

**The columns** are the csv loader's: a header `x`, `xnm` or `x_nm` in any
case for x and the same for y, the separator -- comma, semicolon or tab --
found from the header line, and a unit in brackets (`x [um]`) in nm, um or
mm, nm when none is given; pixels are refused.  Rows without a number for x
or y are skipped.

**The file of each position** is the first of `file_id`, `file` and
`filenumber` that the row fills and the ROI manager has; a row that names a
file the manager does not have is skipped, and the text says how many were.
With *replace*, the ROIs this method made in the files it adds to go first.

## Parameters

### replace
On, a second run gives a fresh answer: the ROIs the same *method* made on
the file before go first (each ROI's origin says which) -- for *from an id
column*, those it made of the same column, in every file.  ROIs drawn by
hand or made by another method stay either way: beads found after the
pores keep the pores.  Off, a run adds to what is there; for the circle
template and beads a candidate on an existing ROI is suppressed, so running
again with the same settings finds nothing new, which is why it is on by
default.  The ROI manager's *Undo* brings replaced ROIs back.

### circle.shape
*ring* for structures labelled on their rim (nuclear pores, rings of a
protein); *disc* for filled ones (pits, clusters, blobs).  A ring is also
found by a disc of its size, as a disc with a low `centre_ratio`.

### circle.min_radius_nm
With *max radius*, the sizes looked for.  Equal for one radius, which is
fastest and most selective -- *NPC* uses 55 nm for Nup96 at about 54 nm.  A
range costs a filter per radius (each about 15 % apart).  The smallest also
sets how close two maxima may be, so too small a value for the structures
lets a large one break into several candidates, which the overlap rule
mostly removes.

### circle.max_radius_nm
Larger than the largest structure you want; a much larger value lets
neighbouring structures merge into one large candidate where they are dense.

### circle.ring_width_nm
About the localization precision plus the label's size, 10 to 20 nm.  Read
only for a ring.

### circle.precision_nm
The same as NPC Corners' *corner precision*, so that the pores are judged on
the localizations they are counted with.  Looser lets more imprecise ones
into the fits and the far test; tighter leaves sparse structures too few to
judge (3 are needed).

### circle.bin_nm
Smaller than the ring width; 10 nm is right for most data, larger for a
field too big for 16 million pixels.  Only the speed and the memory depend
on it much.

### beads.min_photons
Between the brightest molecules and the beads: a bead gives several times
the photons of a dye molecule.  Look at the photons of a few beads.

### beads.min_fraction
Most beads are seen in nearly every frame; 0.25 keeps one that came loose
or bleached part of the way, which Fiducials can still use.  *keep* with
`frames_fraction >= 0.9` keeps only the steady ones.

### keep
A preset writes it; change it after looking at the *keep* figure of a
*Preview*, where every value it reads is drawn for the kept and the rejected
candidates.  The form lists the values the method measures.

### keep_rejected
Tick it while choosing *keep*: each rejected candidate becomes an ROI that is
not used, its origin lists what it failed, and its values are in the results
table to sort by.  *Preview* shows them too, dotted, without making them
ROIs.

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
taking in the localizations around a smaller one.  Read only when *an ROI
holds* is *what is in its shape*.

### ids.membership
*Its id alone* for tracks, and for clusters that overlap, where a shape would
mix two of them; *what is in its shape* where the area is the point -- the
localizations of every kind near a cluster, measured in the same window
round each.

### ids.draw
Read only for *its id alone*.  A track's line shows its direction and its
steps; an outline is the better picture of a cluster, whose localizations
have no order.

### file.path
A table written by *Export table* (with *ROI position*), or any csv with
`x_nm` and `y_nm`.

## Output

### Circle template and beads

* **ROIs**, one per candidate *keep* passes, added to the current file in the
  ROI manager, used from the start; with *keep rejected*, the rejected
  candidates too, not used.
* **The values** of each -- `n_locs`, `radius_nm`, `roundness`, `psf_nm`,
  `touching`, and the method's own: `n_ring`, `n_inside`, `n_outside`,
  `far_p` for a ring, `centre_ratio` for a disc, `frames_fraction`, `photons`,
  `spread_nm` for beads -- are columns of the ROI manager's results table
  (`segment.<name>` where a measurement already has the name), so
  [Filter ROIs](plugin:Analysis/Regions/Filter ROIs),
  [Sort ROIs](plugin:Analysis/Regions/Sort ROIs) and the summaries read them.
  They describe the candidate as it was found, and stay when the ROI is
  moved.
* **The text** says how many were kept of how many candidates, and how often
  each part of *keep* failed.
* **The main figure** shows the localizations with every candidate's circle,
  at its measured radius: kept in blue, rejected in grey.
* **The *keep* figure** has a histogram of each value *keep* reads (the
  radius and the localizations when *keep* is empty), kept and rejected
  stacked, with the numbers *keep* compares each with in red.
* **A preview** (the *Preview* button) finds the same candidates, writes the
  same text and figures, and adds nothing: they are drawn grey and dashed in
  the ROI manager (with *keep rejected*, the rejected ones dotted), and with
  *replace* the earlier ROIs it would remove are hidden.  The next preview
  replaces it; *Run* adds it.

A good result has one ROI on each structure and none on the background.
ROIs on the background: tighten *keep* (more localizations, a narrower
radius).  Faint structures missed: loosen it, after checking in the *keep*
figure that they are among the candidates.  None at all: the radii do not
match the structures.

Run without the ROI manager, from a script (`Context(locs=locs)`), it
returns every candidate with its values, `use` and what it failed in
`data["sites"]`, and the centres of those kept in `data["centers"]`.

### From an id column

The ROIs in the ROI manager; the text says how many were made, the smallest
cluster that became one, and how many clusters had no outline.  It cannot run
inside a block, which repeats over regions: make the ROIs after it.

### Positions from a file

The ROIs in the ROI manager; the text says how many were made, from which
file, and how many rows named a file the manager does not have.  It cannot
run inside a block.

## Differences from SMAP

### Circle template

Ported from SMAP's `ROIManager/Segment/segmentNPC` and its clean-up evaluator (SMAP's word for a plugin run in every ROI)
`ROIManager/Evaluate/NPCsegmentCleanup`, and replacing the script
`segmentCME` ([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* **The filter.**  SMAP filters the rendered image of each cell with a ring
  kernel, $e^{-|r^2 - R^2| / 4w^2}$, follows it with a difference of
  Gaussians, and keeps the maxima above a *cutoff* in units of the filtered
  image; `segmentCME` blurs the rendered image with a Gaussian of 2 pixels
  and keeps the maxima above 80, inside a region drawn by hand.  Here a
  zero-sum template does both jobs at once, a disc for pits, over a range of
  radii, and what decides is *keep* on numbers of localizations, which mean
  the same at any pixel size.
* **The checks.**  SMAP's clean-up compares the localizations inside and
  outside the ring with those on it (*in/ring* < 0.3, *out/ring* < 0.75),
  checks the fitted radius, a minimum size and the mean PSF width, and marks
  the bad sites (ROIs) as not used.  Here the checks are *keep*: the radius
  is kept, the fractions are replaced by the far test on the precise
  localizations, for the reasons in *In detail* (the counts inside and
  outside are values to filter on all the same), and the PSF width is
  `psf_nm`, for the user to add where the table has `sigma_nm`.
* **One step.**  Segmenting and judging happen in one run, and the rejected
  candidates are dropped unless *keep rejected* is ticked.
* **The fits** use a soft-L1 loss on the precise localizations; SMAP's
  `fitposring` is plain least squares on all of them.
* SMAP works per cell, and needs cells first; this works on the whole file.

### Beads

SMAP's `ROIManager/Segment/segmentBeads` merges the localizations within
150 nm into groups, allowing *minimum frames* (30) of gap, takes a group of
more than that many localizations as a bead, and puts a site at its position
in each cell of a grid it makes first; a *Filter* button then excludes the
beads whose smallest PSF width, or the frame of it, is far from the others'
-- a check for a bead stack scanned in z.  Here the beads are the bright,
long tracks of Fiducials' linker, on the whole file without cells, with a
threshold in photons and a length as a fraction of the frames, and the
check is *keep* on the values measured; a bead stack is the
[PSF calibration](plugin:Localize/PSF calibration)'s business,
which finds its own beads.

### Positions from a file

SMAP's `ROIManager/Analyze/ImportSitePositions` copies the sites and cells
of another SMAP file into the current one, the files matched by their
order.  Here the positions come from any table, a csv written by *Export
table* among them, and each goes to the file it names.

### From an id column

New: SMAP has no segmenter that makes sites of clusters; its DBSCAN writes
the cluster index and makes no ROIs.

### One plugin

SMAP has a segmenter per structure in `ROIManager/Segment` -- `segmentNPC`,
`segmentCME`, `segmentBeads`, ... -- each with its own settings.  Here one
plugin has a method per model of a structure, and a structure is a preset of
values, so a structure the methods already find needs a file, not a plugin.

## References

* Thevathasan JV, Kahnwald M, Cieśliński K, et al. Nuclear pores as
  versatile reference standards for quantitative superresolution
  microscopy. *Nat Methods* 16, 1045 (2019).
  [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)
  -- the ring-shaped template and checks for nuclear pores, and the Nup96
  pores simulated in the figures.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
