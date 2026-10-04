---
title: ROI tab
summary: The shape every analysis ROI shares, and the way into the ROI manager and the evaluation of the sites.
widget: smappy.gui.roi_tab.ROIHeader
covers: [smappy.roi_manager.core.ROIProject.set_geometry, smappy.roi_manager.core.ROIProject.indices, smappy.roi_manager.core.ROIProject.geometry, smappy.roi_manager.core.ROIProject.evaluate, smappy.roi_manager.core.ROIProject.latest, smappy.roi_manager.core.inside_polygon, smappy.roi_manager.link.SessionROIs, smappy.roi_manager.pipeline.default_instances]
---

## What it does

Many experiments image many copies of one structure -- nuclear pores,
clathrin pits, centrioles -- and the question is about all of them: how big,
how many molecules, how often a ring is complete.  The way to answer it is to
mark each copy with a small region, an **ROI** (region of interest), measure
what is inside each one, and summarise the measurements.  A marked copy is
called a *site*.

The ROI tab is where that begins.  It holds what all ROIs share -- their shape
and size -- and opens the two windows the work is done in: the
[ROI manager](plugin:Panels/ROI manager), where sites are found, drawn,
included or excluded, and looked at one by one, and the *evaluation* window,
where the measurements to make on every site are chosen and run.  Below the
tab's own controls sit the ROI plugins pinned to it: by default one that finds
sites (*Density Peaks*) and one that summarises the measurements
(*Histograms*).

An ROI sees what the image shows: the localizations of its file that pass a
layer's filter, grouped if that layer is grouped.  So clean the data in the
[Render tab](plugin:Panels/Render tab) first; there is no separate filter for
ROIs.  The ROIs, their measurements and the evaluation settings are saved into
the localization file with *File > Save* and come back when it is opened.

## How it works

```figure-setup
from matplotlib.patches import Circle, Rectangle, Polygon
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.roi_manager import ROIProject
from smappy.render import (FieldOfView, RenderSettings, SigmaSettings,
                           DisplaySettings, render_locs)
sim = SimulationSettings(n_frames=3000, seed=3,
                         structure=StructureSettings(preset="npc"),
                         labelling=LabellingSettings(efficiency=0.6))
locs = simulate(sim)
project = ROIProject()
source = project.add_source(locs)
rois = project.find(source.id, reviewed=True)     # Density Peaks
state = project.state(source.id)
kept = state.locs[state.filter.indices]           # what passes the filter
# the site with the nearest neighbour, so that the geometry matters
centres = np.array([r.center for r in rois])
gap = np.hypot(*(centres[:, None] - centres[None]).transpose(2, 0, 1))
np.fill_diagonal(gap, np.inf)
nearest = gap.min(axis=1)
roi = rois[int(np.argmin(np.where(nearest > 250, nearest, np.inf)))]
cx, cy = roi.center
size = project.size_nm
view = 3 * size                                   # ROI view left at its default
render = RenderSettings(sigma_settings=SigmaSettings(factor=0.5))
display = DisplaySettings(contrast=2.5)
```

**1. Sites.**  ROIs are made in the ROI manager: found automatically, clicked
by hand, or taken from a region drawn in the 2D view (*Add the drawn region as
an ROI*).  Each ROI is a centre in one file; its outline is the shape and
size set here, the same for all of them, unless the ROI has an outline of its
own (a polygon).

```figure Left: simulated nuclear pores, with the ROIs Density Peaks proposed on them, each a circle of the default 300 nm.  Right: one site as the ROI manager shows it -- a field three ROIs wide around the ROI, with its neighbour outside.
fig.set_size_inches(7.5, 3.4)
a, b = fig.subplots(1, 2)
whole = FieldOfView.from_range((0, 10000), (0, 10000), 20.0)
a.imshow(display.apply(render_locs(kept, whole, render, display)), extent=whole.extent)
for r in rois:
    a.add_patch(Circle(r.center, size / 2, fill=False, color="c", lw=0.6))
near = FieldOfView.from_range((cx - view / 2, cx + view / 2),
                              (cy - view / 2, cy + view / 2), 3.0)
b.imshow(display.apply(render_locs(kept, near, render, display)), extent=near.extent)
b.add_patch(Circle(roi.center, size / 2, fill=False, color="c", lw=1))
for ax, title in ((a, f"{len(rois)} sites"), (b, "one site, ROI view 900 nm")):
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
```

**2. The shape decides what is measured.**  A localization belongs to a site
if it lies inside the ROI's outline, boundary included.  The outline is a
circle of the given diameter or a square of the given side, around the
ROI's centre; an ROI with its own polygon uses that instead.  ROIs may
overlap, and a localization in the overlap counts for both.  Make the ROI
large enough to hold the whole structure with its scatter, and small enough to
leave the neighbours out.

```figure The same site with four outlines: the localizations counted (red) and the ones left out (grey).  The square's corners and the larger circle reach the neighbouring pore; a polygon drawn in the 2D view keeps its own outline whatever the shape and size.
fig.set_size_inches(7.5, 2.1)
axes = fig.subplots(1, 4)
x, y = np.asarray(kept["x_nm"]), np.asarray(kept["y_nm"])
around = (np.abs(x - cx) < view / 2) & (np.abs(y - cy) < view / 2)

def show(ax, title, patch):
    inside = project.extract(roi)
    ax.plot(x[around], y[around], ".", ms=1.5, color="0.7")
    ax.plot(inside["x_nm"], inside["y_nm"], ".", ms=1.5, color="#d62728")
    ax.add_patch(patch)
    ax.set_aspect("equal")
    ax.set_xlim(cx - view / 2, cx + view / 2); ax.set_ylim(cy - view / 2, cy + view / 2)
    ax.set_title(f"{title}: {len(inside)}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

show(axes[0], "circle, 300 nm", Circle(roi.center, size / 2, fill=False, lw=0.8))
project.set_geometry(size, "square")
show(axes[1], "square, 300 nm",
     Rectangle((cx - size / 2, cy - size / 2), size, size, fill=False, lw=0.8))
project.set_geometry(500, "circle")
show(axes[2], "circle, 500 nm", Circle(roi.center, 250, fill=False, lw=0.8))
project.set_geometry(size, "circle")
outline = [[cx - 250, cy - 120], [cx + 200, cy - 200], [cx + 120, cy + 180],
           [cx - 150, cy + 150]]
roi.polygon = outline
show(axes[3], "drawn polygon", Polygon(outline, fill=False, lw=0.8))
roi.polygon = None
```

**3. Evaluation.**  *Evaluation pipeline...* opens a window with a list of
*evaluators*: plugins that measure one site and return a row of numbers for
it (the built-in one, *Statistics*, counts the localizations and averages
their precision and photons).  The list is run over every included ROI of
every file, one row per site; together the rows are the *site table*, which
the Analyze plugins such as Histograms summarise.

**4. What is still true.**  Each stored measurement remembers what it was made
from: the file's data, the ROI and its geometry, the filter, the grouping,
and the evaluator with its version and settings.  When any of these changes
-- a new size here, a new filter bound, a drift correction -- the measurements
it affects are marked out of date and left out of the site table until the
sites are evaluated again.

## In detail

**Inside an ROI.**  For an ROI with centre $(x_c, y_c)$ and size $d$, a
localization at $(x, y)$ is inside a circle if

$$(x - x_c)^2 + (y - y_c)^2 \leq (d/2)^2 ,$$

and inside a square if $|x - x_c| \leq d/2$ and $|y - y_c| \leq d/2$.  A
polygon uses the even-odd rule, with points on an edge counted as inside.
Only localizations that pass the filter are considered; they are looked up
through a spatial index over the ROI's bounding box first, so the test runs
on a few hundred points rather than on the whole table.  Coordinates are in
nanometres, whatever the Render tab's axes show.

**The data an ROI sees.**  The session's table, cut to the ROI's own file
(an ROI knows its file, so the layer's choice of files is ignored), filtered
by the first localization layer's bounds and grouped if that layer is
grouped.  On a
grouped layer a site's count is a count of blinks, and its photons are the
blinks' summed photons.

**A drawn region.**  *Add the drawn region as an ROI* turns the region drawn
in the 2D view into an ROI of the file chosen in the ROI manager: a rectangle
becomes its four corners, a line the rectangle of its width around it, a
polygon its vertices.  The ROI's centre is the mean of the vertices.  Such an
ROI keeps its outline when the shape or size is changed here.

**What is out of date.**  A stored measurement is signed with a digest of its
inputs -- the data's fingerprint (row count, columns and a sample of the
positions), the ROI's geometry, the filter bounds, the grouping and its
parameters -- together with the evaluator, its version and its settings, but
not its name.  A step is *current* when that signature matches what it would
be now, and *stale* when not; the check is per step, so changing one
evaluator's settings leaves the others' numbers standing.  Changing the size
or shape here makes the measurements of every circle and square ROI stale,
but not those of polygon ROIs.

**Which ROIs count.**  Every ROI counts from the moment it is made; the *use*
tick in the ROI manager is what excludes one.  Evaluation and the site table
pass over excluded ROIs; their earlier results stay in the file.

## Controls

### Open ROI manager
The window where ROIs are found, drawn, moved, included or excluded, and
where each site is shown with what the evaluators measured on it (also
*Tools > ROI manager*, Ctrl+R).  Under the button, the file chosen there and
how many of its ROIs there are and are used.

### geometry
What every ROI's outline is, unless it has a polygon of its own.

### shape
*circle* for round structures; *square* for structures that are square or
when the corners do not matter.  A square of side $d$ is 27 % larger in area
than a circle of diameter $d$, so its corners reach further towards a
neighbour.

### size
The circle's diameter, or the square's side.  Start at about twice the
structure's diameter: the default 300 nm fits a nuclear pore (about 110 nm)
with room for the localization scatter and a misplaced centre.  Too small,
and the edge of the structure is cut off where the centre is off; too large,
and neighbours and background come in.  Changing it applies to every
circle and square ROI and puts their measurements out of date.

### ROI view
A display setting only: it changes nothing that is measured.  The width of
the field the ROI manager shows around the selected ROI.  At 0 it is three
times the ROI size.  Wider shows more context; narrower, more detail of the
site.

### Add the drawn region as an ROI
Draw a rectangle, a line or a polygon in the 2D view first.  Useful for
structures that are not round, or a region the automatic finder would not
propose; the outline is kept as it is.

### evaluate
The measurements.  The table lists what was measured on the ROI selected in
the ROI manager, one line per number, marked with its state -- *(stale)*, say -- where the
step that produced it is not current; below it, how many sites have a complete, current
row in the site table.

### Evaluation pipeline...
Opens the evaluation window: which evaluators run, in what order, with which
settings, and the buttons that run them on every ROI or re-run only what
changed.  A new pipeline starts with the general evaluators, once each;
a specialised one, such as NPC Corners, is added here when the data call
for it.  The
pipeline is saved in the file with the results, because the site table's
columns mean nothing without it.

## Differences from SMAP

Based on SMAP's ROI manager (the SiteExplorer, with its `ROIManager` plugins)
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* **One outline, circle or square.**  SMAP's site settings have an ROI size
  (300 nm, as here) and a site field of view (500 nm); an evaluator gets the
  localizations of a square the size of the field, or of a size it asks for,
  and cuts out what it wants itself.  Here the shape is chosen once and every
  evaluator gets the localizations inside the same outline; a polygon
  replaces it per ROI.
* **The view is not a setting of the analysis.**  SMAP's site field of view
  is both what is shown and what an evaluator gets by default; here *ROI
  view* only changes what is shown, three ROI widths unless it is set.
* **No review step.**  An ROI counts once it is made, and *use* excludes one.
* **Evaluation is a pipeline with provenance.**  Each result is stored with
  what it was computed from, is marked out of date when that changes, and is
  saved in the localization file rather than in a separate site file.
* **Filters and grouping come from a layer** of the Render tab, not from
  the ROI manager's own settings.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
