---
title: Regions tab
summary: The shape every analysis ROI shares, and the way into the ROI manager and the steps that measure every ROI.
widget: smappy.gui.roi_tab.ROIHeader
covers: [smappy.roi_manager.core.ROIProject.set_geometry, smappy.roi_manager.core.ROIProject.indices, smappy.roi_manager.core.ROIProject.geometry, smappy.roi_manager.core.ROIProject.evaluate, smappy.roi_manager.core.ROIProject.latest, smappy.roi_manager.core.ROIProject.evaluations, smappy.regions.inside_polygon, smappy.roi_manager.core.ROIProject.sync, smappy.roi_manager.pipeline.default_block, smappy.gui.region_filter_panel.RegionFilterPanel, smappy.gui.region_results_window.RegionResultsWindow]
---

## What it does

Many experiments image many copies of one structure -- nuclear pores,
clathrin pits, centrioles -- and the question is about all of them: how big,
how many molecules, how often a ring is complete.  The way to answer it is to
mark each copy with an [ROI](plugin:Panels/Glossary) (region of interest: a
shape -- here a circle, a square or a polygon -- around one copy of the
structure), measure what is inside each one, and summarise the measurements.

The Regions tab is where that begins.  It opens the
[ROI manager](plugin:Panels/ROI manager), where ROIs are found, drawn,
included or excluded, and looked at one by one; it holds the *per-region
steps*, the measurements to make in every ROI, with what they measured in
the ROI selected in the manager; and it holds what all ROIs share -- their
shape and size.  Below the tab's own controls sit the ROI plugins pinned to
it.  By default these are
the one that finds ROIs -- [Segment regions](plugin:Analysis/Regions/Segment regions), by a circle
template (a ring for nuclear pores, a disc for pits and other compact
structures), beads, an id [attribute](plugin:Panels/Glossary) (one value
every localization carries), positions from a file, or like a few ROIs
put on good examples -- and
[Export table](plugin:Analysis/Regions/Summarize/Export table), which writes
what was measured in every ROI to a file.  The summaries, such as
[Histogram](plugin:Analysis/Measure/Histogram) with *source* on region results, and
[NPC Analysis](plugin:Analysis/Regions/Workflow/NPC Analysis), which finds
the pores, measures them and fits their
[labelling efficiency](plugin:Panels/Glossary) (also called degree of
labelling: the fraction of the target proteins that carry a detected label)
in one run, are pinned with the tab's **+**.

An ROI sees what the image shows: the localizations of its file that pass a
layer's filter, with blinks merged if that layer merges them.  A
[layer](plugin:Panels/Glossary) is one picture of the localizations, with its
own filter and colours; its [filter](plugin:Panels/Glossary) (also called
bounds) is the lowest and highest values of attributes a localization must lie
between; [merge blinks](plugin:Panels/Glossary) (also called grouping or
linking) combines the localizations in consecutive frames that come from one
blinking event into one localization.  So clean the data in the
[Render tab](plugin:Panels/Render tab) first; there is no separate filter for
ROIs.  The ROIs, their measurements and the per-region steps are saved into
the localization file with *File > Save* and come back when it is opened.

## How it works

```figure-setup
from matplotlib.patches import Circle, Rectangle, Polygon
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.session import Session
from smappy.render import (FieldOfView, RenderSettings, SigmaSettings,
                           DisplaySettings, render_locs)
sim = SimulationSettings(n_frames=3000, seed=3,
                         structure=StructureSettings(preset="npc"),
                         labelling=LabellingSettings(efficiency=0.6))
locs = simulate(sim)
session = Session(locs)                           # the GUI's model, no window
session.show_grouped(0, False)                    # localizations, not blinks
project = session.rois
source = project.source_of(0)                     # the one file
rois = project.find(source.id)                    # Segment regions: its default discs
state = project.state(source.id)
kept = state.locs[state.filter.indices]           # what passes the filter
# the ROI with the nearest neighbour, so that the geometry matters
centres = np.array([r.center for r in rois])
gap = np.hypot(*(centres[:, None] - centres[None]).transpose(2, 0, 1))
np.fill_diagonal(gap, np.inf)
nearest = gap.min(axis=1)
roi = rois[int(np.argmin(np.where(nearest > 250, nearest, np.inf)))]
cx, cy = roi.center
size = project.size_nm
view = 2 * size                                   # ROI view left at its default
render = RenderSettings(sigma_settings=SigmaSettings(factor=0.5))
display = DisplaySettings(contrast=2.5)
```

**1. ROIs.**  ROIs are made in the ROI manager: found automatically, clicked
by hand, or taken from a region drawn in the 2D view (*Add drawn region*).  Each ROI is a centre in one file; its outline is the shape and
size set in the ROI manager (*shape*, *size*), the same for all of them,
unless the ROI has an outline of its own (a polygon or an ellipse).

```figure Left: simulated nuclear pores, with the ROIs Segment regions proposed on them with its defaults (discs of 30 to 150 nm), each a circle of the default 300 nm.  Right: one ROI as the ROI manager shows it -- a field two ROIs wide around it, with its neighbour outside.
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
for ax, title in ((a, f"{len(rois)} ROIs"), (b, "one ROI, ROI view 900 nm")):
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
```

**2. The shape decides what is measured.**  A localization belongs to an ROI
if it lies inside its outline, boundary included.  The outline is a
circle of the given diameter or a square of the given side, around the
ROI's centre; an ROI with its own polygon uses that instead.  ROIs may
overlap, and a localization in the overlap counts for both.  Make the ROI
large enough to hold the whole structure with its scatter, and small enough to
leave the neighbours out.

```figure The same ROI with four outlines: the localizations counted (red) and the ones left out (grey).  The square's corners and the larger circle reach the neighbouring pore; a polygon drawn in the 2D view keeps its own outline whatever the shape and size.
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

**3. Measuring.**  *per-region steps* is a list of
steps that run once in every included ROI of every file -- plugins that measure a
region and return a row of numbers for it, such as Localization Statistics,
the same plugin as on the whole view.  One row per ROI, the rows together
are the *results table*, which the summaries such as Histogram on region results read.  The
list is a *block*, as in a [chain](plugin:Panels/Glossary) (also called a
pipeline or workflow: plugins run one after another as one): it can repeat
over the [regions](plugin:Panels/Glossary) of an id attribute instead (a region
is a set of localizations measured together -- an ROI, or a cluster that an
attribute such as `cluster_id` names; *for each*).  *Measure all* measures every ROI
(or, while several are selected in the ROI manager, those); with
*measure the selected ROI* ticked, an ROI is also measured as soon as it is
selected or added in the ROI manager, and its numbers appear under the
steps.

**Proposals.**  Some steps say not only what they measured but what the
ROI could be: the [Trace inspector](plugin:Analysis/Tracking/Trace inspector)
proposes a track's direction as the ROI's angle, its axis as the line
*axis*, its centre; [Trace statistics](plugin:Analysis/Tracking/Trace statistics)
the trace's direction as the angle and, with a tail rule, the range after
its tail; [Circle fit](plugin:Analysis/Measure/Circle fit) the fitted
centre.  Under the selected ROI's numbers, a line per step says what
it proposes for the ROI ("Trace inspector proposes for the ROI: angle 37°,
line axis, its centre"), and *Apply* sets it on the ROI -- on the ROIs selected in the ROI
manager's list, or all of them when none is, each ROI its own proposals --
as one step of the ROI manager's undo.  What was set by hand (an angle
turned, a range set, a line drawn) is left alone.  Once applied, they are
the ROIs' own properties, as if set by hand or by
[Write results](plugin:Analysis/Regions/Write results); nothing is
applied by measuring alone.  In a chain,
[Apply proposals to ROIs](plugin:Analysis/Regions/Accept proposals) does the
same.

**4. What is still true.**  Each stored measurement remembers what it was made
from: the file's data, the ROI and its geometry, the filter, the merge
settings, and the step's plugin with its version and settings.  When the data
an ROI sees change -- a new size here, a new filter bound, a drift correction
-- its measurements are marked out of date and left out of the results table
until the ROIs are measured again.  A measurement belongs to its ROI, whether
the per-region steps or a chain made it, and steps under different
names stand side by side in the results table.

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
by the first localization layer's bounds, with blinks merged if that layer
merges them.  With blinks merged an ROI's count is a count of blinks, and its
photons are the blinks' summed photons.

**A drawn region.**  *Add drawn region*, in the ROI manager, turns the region drawn
in the 2D view into an ROI of the file chosen in the ROI manager: a rectangle
becomes its four corners, a line the rectangle of its width around it, a
polygon its vertices.  The ROI's centre is the mean of the vertices.  Such an
ROI keeps its outline when the shape or size is changed here.

**What is out of date.**  A stored measurement is signed with a digest of its
inputs -- the data's fingerprint (row count, attributes and a sample of the
positions; not the attributes written from regions' results, such as those of
*Write results* (to the localizations)), the ROI's geometry, the filter bounds, whether blinks
are merged and with which settings -- together with the step's plugin, its
version and its settings, but not its name.  A step is *current* when that
signature matches what it would be now, and *stale* when not; the check is
per step, so changing one step's settings leaves the others' numbers
standing.  The results table is looser: it holds every measurement still
true of the ROI's data, the newest under each step's name, with the settings
it was made with.  Changing the size
or shape here makes the measurements of every circle and square ROI stale,
but not those of polygon ROIs.

**Which ROIs count.**  Every ROI counts from the moment it is made; the *use*
tick in the ROI manager is what excludes one.  The per-region steps and the
results table pass over excluded ROIs; their earlier results stay in the file.

## Controls

### Open ROI manager
The window where ROIs are found, drawn, moved, included or excluded, and
where each ROI is shown with what the per-region steps measured in it (also
Ctrl+R).  Under the button, the file chosen there and
how many of its ROIs there are and are used.

### Results table...
Opens the [Region results](plugin:Panels/Region results) window: what was
measured in every region -- the ROIs, or the clusters, tracks or traces of
an id attribute -- as one table, a row per region and a column per value.
Sort it, copy it to a spreadsheet, or colour the localizations by a column
from its header's right click.  The ROI manager's list header offers it too.

### per-region steps
Folded when the tab opens; unfold it to choose and run the steps.  At the
top is what applies to every step: *Measure all*, *Measure this ROI*, and
the ticks that measure and draw the ROI selected in the ROI manager.
*Measure all* runs the steps in every ROI (or region) whose result is
missing or out of date, and carries the rest forward.  While two or more
ROIs are selected in the ROI manager's list it says *Measure n selected*
and runs them in those of them with *use* ticked; one selected ROI is only
the ROI being looked at, and stepping through the ROIs selects each, so it
does not narrow the run.  For the quick numbers of every ROI at once --
counts, spreads, medians of attributes -- [Region
statistics](plugin:Analysis/Measure/Region statistics) with *regions* set
to *ROIs* takes a second where the steps take an ROI at a time.  While it measures,
it is Stop: the ROI being measured is finished, the ones measured so far are
kept, and the rest wait for the next *Measure all*.  The grey line under them says
how far a run has got -- the share and number of ROIs (or regions)
measured, how many a second and about how long is left -- and, after a
Stop or a failure, how many are still to measure.  A run that will take
more than a minute says how long before it starts.

Below them, the steps: each a folded section with a tick in its title.
Ticked, the step runs; unticked, it stays in the list with its settings and
is skipped.  Unfolding a step shows its settings, and a right click on its
title renames, moves or removes it.  With an ROI selected in the ROI
manager, a step's title says how its result for that ROI stands --
*(stale)*, *(failed)* -- and says nothing when it is current.  A new list
starts with the general measurements, once each, with their own defaults;
a specialised one, such as NPC Corners, is added there when the data call
for it.  The list is saved in the file with the results, because the
results table's columns mean nothing without it; a file opened without
steps of its own takes the ones shown here.  The **⋯** menu in the
section's title bar opens a saved list (*Open chain...*: a chain's steps, or
a block's) and keeps this one as a chain (*Save as...*).

### for each
What the steps are repeated over: every ROI of the ROI manager, or each
region of an id attribute of the table (`cluster_id`, `track_id`, `tid`).

### build
Folded: the tools that change the steps, but for *add...*, which stands in
build's own title line and adds a plugin as a step, after the step last
opened.  *check* times the steps in a few ROIs and says
what the whole will take.

What a step measures in an ROI stays with that ROI, in the results table.
To make a measured value one of the ROIs' own properties -- its angle, a
range, a line, a keyword -- run
[Write results](plugin:Analysis/Regions/Write results) after the
steps; to write it onto each ROI's localizations, as an attribute to filter
or colour by, run [Write results](plugin:Analysis/Regions/Write results).
A step's own *write ...* settings (Trace statistics' rotated track, say)
change only the ROI's working copy of the localizations, which the next
steps of the same ROI read.  A project saved by an earlier version with
*write back* or *write values* set shows those two fields here, and still
writes them until they are emptied.

Under it, how many ROIs have a complete, current row in the results table,
and, once an ROI is selected in the ROI manager, a table of what was
measured on it, one line per number, marked with its state -- *(stale)*,
say -- where the step that produced it is not current.  The keywords it was given
in the ROI manager (*annotate*) are listed with them.

### measure the selected ROI
On by default.  An ROI selected or added in the ROI manager is measured at
once: the steps whose result for it is missing or out of date run, and what
they measure is stored, so the same work is not done on the next visit.
Untick it to scroll a large project without anything being computed behind
you; the numbers are then shown as stored.

### Apply proposals
At the top of the per-region steps, beside *Measure all*: sets on the ROIs
what the steps proposed for them (see *Proposals* above) -- on the ROIs
selected in the ROI manager, all of them when none is, each its own -- as
one undo step.  With several steps proposing, it asks which, or all.
Greyed while no step has proposed anything.  The line under a selected
ROI's numbers does the same for that one step.

### Measure this ROI
Runs the steps on the ROI selected in the ROI manager now, whether or not
*measure the selected ROI* is ticked: the steps whose result for it is
missing or out of date, which a step whose settings were just changed is.
It is the way to see what a new setting gives on the ROI being looked at
before running all of them.

### draw its figures
Opens the *ROI results* window beside the ROI manager (and the manager, if
it is not open): what the steps draw for the selected ROI, a tab per
enabled step, redrawn in place by the next ROI.  A step that keeps the data
its figure is drawn from with its result -- Localization Statistics,
Binding kinetics -- draws it from what is stored, without measuring again;
any other step whose tab is open runs again for its figure.  Ticking it
brings the window to the front with the keyboard, so ↑ and ↓ step through
the ROIs from there (← and → through the tiles), as they do in the ROI
manager; redrawn as the ROIs are stepped, from either window, it never takes
the keyboard from the one the keys are pressed in.  Closing that window
unticks it.

### measured values
What the steps measured in the ROI selected in the ROI manager, a row per
value, and the keywords given it; a value whose step's result is out of
date says so.  Shown once an ROI is selected; click the title to fold it
away while stepping through the ROIs.

### region filter
Keeps the localizations of the [regions](plugin:Panels/Glossary) whose
results pass conditions -- ROIs with six corners or more, traces whose
spread is below 10 nm, clusters of 20 localizations or more -- and drops
the rest from every layer, view and plugin.  The conditions are the same
as [Filter regions](plugin:Analysis/Regions/Filter regions) sets in a
chain, whose page has the rules: a region must pass every condition on its
source, a localization is kept if it is in any region that passes, and one
in no region of a filtered source is dropped.  Folded when the tab opens.

From the top: the conditions applied, each with how many regions it keeps
of how many (click one to show its value, × to remove it); a warning where
results are out of date; a search box and the tree of everything that can
be filtered -- the ROIs and each id attribute, each evaluation, its values,
and the statistics of one quantity (`cfr` › `median`) as a level of their
own; and the chosen value's distribution over all the regions, with the
range to keep, or for a keyword or a flag (an ROI's *use*) the texts to
accept.  The Render tab's *regions* button turns the filter off in one
layer.

### clear all
Removes every condition: every localization is shown again.

### add
Adds the chosen value with its range as a condition, or with the texts
ticked; for a value that already has one it reads *update* and replaces
it.  The range is the two numbers beside it, or the shaded band dragged on
the histogram -- an end dragged to the edge is no bound.

## Differences from SMAP

Based on SMAP's ROI manager (the SiteExplorer, with its `ROIManager` plugins)
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), which calls an ROI
a *site* and a measuring plugin an *evaluator*.

* **One outline, circle or square.**  SMAP's settings have an ROI size
  (300 nm, as here) and a site field of view (500 nm); an evaluator gets the
  localizations of a square the size of the field, or of a size it asks for,
  and cuts out what it wants itself.  Here the shape is chosen once and every
  step gets the localizations inside the same outline; a polygon replaces it
  per ROI.
* **The view is not a setting of the analysis.**  SMAP's site field of view
  is both what is shown and what an evaluator gets by default; here *ROI
  view* only changes what is shown, three ROI widths unless it is set.
* **No review step.**  An ROI counts once it is made, and *use* excludes one.
* **The steps keep their provenance.**  Each result is stored with what it
  was computed from, is marked out of date when that changes, and is saved in
  the localization file rather than in a separate site file.
* **Filters and merged blinks come from a layer** of the Render tab, not from
  the ROI manager's own settings.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
