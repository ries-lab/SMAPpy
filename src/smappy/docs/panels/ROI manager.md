---
title: ROI manager
summary: The window where ROIs are found, looked at one by one, kept or excluded, and measured.
widget: smappy.gui.roi_window.ROIManagerWindow
covers: [smappy.gui.roi_window.ROIManagerWindow.evaluate_active, smappy.gui.roi_window.ROIList, smappy.gui.roi_window.ROIManagerWindow._in_view, smappy.regions.Region.rim_distance, smappy.regions.Region.outline, smappy.roi_manager.core.ROIProject.tiles, smappy.roi_manager.core.ROIProject.indices, smappy.roi_manager.core.ROIProject.inputs, smappy.roi_manager.core.ROIProject.entries, smappy.roi_manager.core.ROIProject.needs_evaluation, smappy.roi_manager.core.ROIProject.results, smappy.roi_manager.core.ROIProject.evaluate_one, smappy.roi_manager.core.ROIProject.sync, smappy.roi_manager.core.ROIProject.view, smappy.roi_manager.core.ROIProject._fingerprint, smappy.roi_manager.core.ROIProject.propose, smappy.plugins.roi.SegmenterPlugin.found, smappy.roi_manager.core.ROIHistory, smappy.roi_manager.core.ROIProject.edit, smappy.roi_manager.core.ROIProject.set_line, smappy.roi_manager.core.ROIProject.reverse_line, smappy.roi_manager.core.line_angle_length, smappy.roi_manager.core.ROIProject.geometry, smappy.roi_manager.core.ROIProject.members, smappy.roi_manager.core.narrow, smappy.roi_manager.core.ROIProject.set_ranges, smappy.roi_manager.core.ROIProject.outline, smappy.roi_manager.core.ROIProject.rim_distance, smappy.roi_manager.core.ROIProject.summary, smappy.gui.roi_ranges.RangePanel, smappy.roi_manager.core.ROIProject.set_angle, smappy.roi_manager.core.ROIProject.set_rotate, smappy.roi_manager.core.ROIProject.frame, smappy.roi_manager.core.ROIProject.view_region, smappy.roi_manager.core.normal_angle, smappy.regions.SiteFrame, smappy.regions.Region.site, smappy.gui.roi_window.arrow, smappy.gui.roi_window.ImagePane.set_turn]
---

## What it does

Many SMLM experiments image the same small structure hundreds of times:
nuclear pores, clathrin-coated pits, centrioles.  The answer is then not one
image but a table, one row per structure, and a distribution over the rows.
The ROI manager is where that table is built.  Each structure gets an
[ROI](plugin:Panels/Glossary) (region of interest: a circle, square or
polygon around it, inside which it is measured).  You look at the ROIs one
after another, keep the good ones, exclude the bad ones, and let plugins
measure each one.

The work goes in four steps:

1. **Find** candidate ROIs, with a segmentation plugin in the
   [Regions tab](plugin:Panels/Regions tab) such as
   [Segment regions](plugin:Analysis/Regions/Segment regions), or by clicking
   them in the zoom.
2. **Review** them: step through the list, look at each ROI, and untick
   *use* for the ones that are not what they should be.
3. **Measure** every ROI with the per-region steps, plugins such as
   [Localization Statistics](plugin:Analysis/Measure/Localization Statistics) -- the same plugin as on the
   whole view, run once in each ROI.  Each step adds columns to the
   ROI's row.
4. **Analyse** the table, for example with
   [Histogram](plugin:Analysis/Measure/Histogram) with *source* on region
   results, pinned to the Regions tab with its **+**.

An ROI need not be a shape.  [Segment
regions](plugin:Analysis/Regions/Segment regions)' *from an id attribute*, set
to *its id alone*, makes an ROI of each id of an
[attribute](plugin:Panels/Glossary) (one value every localization carries) -- each
[track](plugin:Panels/Glossary) of `track_id`, say -- which holds that
track's localizations and none of the tracks crossing it.  Such an ROI is
drawn as its track's line, selected by a click on the line, and reviewed,
kept and annotated like any other.  While the layer is rendered as tracks,
the track ROIs are marked in the picture rather than by a line of one
colour over it, which would hide their own colours: each is drawn twice
as wide, in its colours when used and grey when not, and the selected or
drafted one has a white rim besides.  The tracks that are no ROI are dimmed
a little, to 80 %, so that they stay easy to pick from.  One track at a time is picked by hand
with *shape* set to *track* -- its default while the
layer is rendered as tracks: a click in the zoom drafts the track drawn
nearest it, and *Add* keeps it.  The list shows *all* ROIs, only the
*shapes*, or only those *of an id*.

Any ROI can also be cut on its attributes -- in time and in depth by
default, on any other the table has as well (`time_s` for MINFLUX,
`sigma_nm`, `photons`): ranges, set under *ranges* in *annotate*, are part
of what the ROI holds, so what is measured in it -- and what *Regions to
attribute* writes -- is that part only.  The layers' filters still apply;
an ROI's ranges cut further, inside the ROI, on every layer.  No range,
the default, is no cut.

The Regions tab holds the plugins that find, measure and summarise ROIs;
this window is for making, choosing and judging them, and holds the shape
and size all ROIs share.

An ROI sees the localizations the image shows.  The first layer's filters
and whether it merges blinks, set in the
[Render tab](plugin:Panels/Render tab), apply to every ROI, so a filter on
precision or photons is set there, once.  (A [layer](plugin:Panels/Glossary)
is one picture of the localizations; its filter, also called bounds, is the
lowest and highest values a localization's attributes may have; [merge
blinks](plugin:Panels/Glossary), also called grouping or linking, combines
the localizations in consecutive frames that come from one blinking event
into one.)

The images in this window show what the main view shows of the file: every
visible layer that shows this file, each with its own filter, merging and
colours, added up as the main view adds them -- two channels in two layers
are drawn in their two colours.  A layer hidden in the Render tab, or set to
show only other files, is left out, so the molecules of another file are
never drawn over this file's ROIs; if that leaves no layer, the first layer
is drawn anyway.  Three things still differ from the main view.  The images
are always of x and y, whatever the layers' axes, because an ROI is a
place.  The 2D view's own ROI and the 3D view's slab are not applied.  And
contrast is set for each image separately, so the same molecules can look
brighter or dimmer than in the main view at another zoom; set the
layer's scale to *absolute* in the Render tab to put both on one scale.  The ROIs
are measured on the first layer alone, whatever the images show.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.roi_manager import ROI
from smappy.session import Session
from smappy.render import FieldOfView
from smappy.chain import Block, Step
from smappy.gui.roi_window import (arrow, OTHER_PEN, EXCLUDED_PEN, ROI_PEN,
                                   DRAFT_PEN, TILE_PEN, TILE_HERE_PEN, LINE_PENS,
                                   STALE_COLOUR)
sim = SimulationSettings(n_frames=3000, seed=3,
                         structure=StructureSettings(preset="npc"),
                         labelling=LabellingSettings(efficiency=0.6))
locs = simulate(sim)
session = Session(locs)                           # the GUI's model, no window
session.show_grouped(0, False)                    # localizations, not blinks
project = session.rois
source = project.source_of(0)                     # the one file
project.set_geometry(200, "circle")
found = project.find(source.id)
project.set_tiles(2500)
tiles = project.tiles(source.id)
state = project.state(source.id)
pen = lambda p: p.color().name()

def image(ax, x0, x1, y0, y1, n=400):
    fov = FieldOfView.fit((x0, x1), (y0, y1), n, n)
    rgb, _ = state.image(fov)
    ax.imshow(rgb, extent=(fov.x0, fov.x1, fov.y1, fov.y0))
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([])

def ring(ax, roi, colour, lw=1.0, ls="-"):
    ox, oy = project.region(roi).outline()
    ax.plot(ox, oy, color=colour, lw=lw, ls=ls)
```

**1. Three images and a list.**  Three pictures side by side, each about
square: the whole file, a zoom, and the ROI being looked at.  Under the ROI
image is one row of what to draw and *Add*; under the zoom, *2D view* and
*rotate*, how the ROI is turned.  Under
the pictures, the files and the ROIs on the left and *annotate* on the
right.  The
dividers between them can be dragged.  A click in the file moves the zoom
there.  Moving around never moves an ROI.

**2. Finding ROIs.**  A segmentation plugin run from the Regions tab adds its
candidates to the file shown here.  Its *Preview* button proposes them
without adding them: they are drawn grey and dashed (the ones the finder
rejected dotted), each preview replaces the last, and *Run* adds them -- or
*Add previewed ROIs* adds those shown, without searching again.  That
is how a finder's settings are tuned without removing the last try's ROIs by
hand; and a run that was not wanted is taken back with *Undo*.  A click on empty space in the zoom drafts
an ROI by hand (blue, dashed); *Add* stores it.  A click on an ROI's
*outline* selects it, while a click *inside* one drafts a new ROI, so that
overlapping ROIs can be drawn.

**3. A systematic walk.**  *Tiles* lays a grid over the file.  The arrow keys
step the zoom from one square to the next, row by row, so that every part of
the data is looked at once, in a fixed order, and not only where the eye
happened to land.

```figure Simulated nuclear pores (Nup96, 60 % labelling) after Segment regions with its defaults, with ROIs of 200 nm.  Left: the file, every ROI found (amber) and a grid of 2.5 µm tiles, the current one in cyan.  Middle: the zoom on that tile, with an ROI excluded from use (grey), the selected ROI (green) and a draft not yet added (blue, dashed).  Right: the ROI image of the selected ROI, two ROI widths across, with a line called *line* drawn on it (magenta).
x0, y0, x1, y1 = state.index.bounds
here = 5
tx0, ty0, tx1, ty1 = tiles[here]
mine = [r for r in found if tx0 <= r.center[0] < tx1 and ty0 <= r.center[1] < ty1]
excluded, selected = mine[0], mine[1]
excluded.use = False
selected.lines = {"line": [[selected.center[0] - 70, selected.center[1] + 50],
                                [selected.center[0] + 70, selected.center[1] - 50]]}
draft = ROI(source.id, [tx0 + 0.25 * (tx1 - tx0), ty0 + 0.8 * (ty1 - ty0)])
fig.set_size_inches(7.5, 2.7)
a, b, c = fig.subplots(1, 3)
image(a, x0, x1, y0, y1)
for tx in tiles:
    a.plot([tx[0], tx[2], tx[2], tx[0], tx[0]], [tx[1], tx[1], tx[3], tx[3], tx[1]],
           color=pen(TILE_PEN), lw=0.6)
a.plot([tx0, tx1, tx1, tx0, tx0], [ty0, ty0, ty1, ty1, ty0], color=pen(TILE_HERE_PEN), lw=1.5)
for r in found:
    ring(a, r, pen(OTHER_PEN), 0.4)
a.set_title("file", fontsize=9)
image(b, tx0, tx1, ty0, ty1)
for r in mine:
    ring(b, r, pen(EXCLUDED_PEN) if not r.use else
         pen(ROI_PEN) if r is selected else pen(OTHER_PEN), 1.0)
ring(b, draft, pen(DRAFT_PEN), 1.2, "--")
b.set_title("zoom", fontsize=9)
half = project.size_nm * 1.0
cx, cy = selected.center
image(c, cx - half, cx + half, cy - half, cy + half)
ring(c, selected, pen(ROI_PEN), 1.5)
dx, dy = arrow(selected.lines["line"])
c.plot(dx, dy, color=pen(LINE_PENS[0]), lw=1.5)
c.set_title("ROI", fontsize=9)
```

**4. Reviewing.**  Selecting an ROI in the list shows its file, centres the
zoom on it and draws it in the ROI image.  The *use* box in the list, or the
space bar, includes or excludes it.  An excluded ROI stays in the list, grey,
and is left out of the per-region steps and the results.  There is no
separate review flag: *use* is the decision, from the moment an ROI is made.

**Many at once.**  The list selects as a file browser does: Shift-click
selects every ROI between the last one clicked and this one, Ctrl-click (Cmd
on a Mac) adds an ROI to the selection or takes it out, and Ctrl+A (Cmd+A),
with the list in focus, selects them all.  The panes still show one ROI, the
last one clicked, and drawing and *annotate* work on it alone.  The space
bar, *Remove* and the list's right-click menu -- *Use*, *Don't use* and
*Remove* -- act on every selected ROI, as one step of *Undo*.  The space bar
excludes them all when all are included, and otherwise includes them all.
Up and Down, or a click in the zoom, select one ROI again.  The selection
is also what setting ROIs from measurements acts on -- the Regions tab's
*Apply* beside a step's proposals,
[Write results](plugin:Analysis/Regions/Write results),
[Accept proposals](plugin:Analysis/Regions/Accept proposals) -- and with
none selected, all ROIs: *use* is for the analysis, what is changed is
chosen by selecting.

**Annotating.**  Beside the list, *annotate* gives the selected ROI a
keyword from each of a few lists the user writes -- *quality: good, bad,
unsure*, say, or the stage of a structure.  The number keys give a keyword
of the active list (1 the first, 0 none), Tab moves to the next list, and
Up and Down to the next and previous ROI; with *next ROI after a keyword* ticked, a
keyword moves on by itself, so a few hundred ROIs are a key each.  The
keywords show in the list's *annotations* column, under the steps in the
Regions tab, and as columns of the results table, named after their lists,
where an analysis can read them.  They are no part of what is measured: giving
one puts no result out of date.  An annotation is undone with the other ROI
edits.

A right click on an annotated ROI in the list adds its keywords to the menu,
*quality: bad* say, each with *Use only these* (every ROI with that keyword
used, every other not), *Use these too*, *Don't use these* (the others left
as they are), *Select these*, and a sort by that list either way.  They act
on every ROI of every file, as one step of *Undo*, and are not logged: for a
step the history should keep, write the same condition in
[Filter ROIs](plugin:Analysis/Regions/Filter ROIs) (`quality == "bad"`).

**Order.**  The list is in the order of the ROI numbers until it is sorted:
by [Sort ROIs](plugin:Analysis/Regions/Sort ROIs), on what was measured or
the keywords, or by a click on a column of the list (again for the other
way round; on *ROI*, back to the numbers).  A click on *annotations* sorts by
the first three keyword lists together; a right click on the list's header
sorts by any one of them, or back to the numbers.  The results table and the
summaries follow the same order, which is saved with the ROIs.  Sorting
never changes an ROI's number.

**5. Shapes and lines.**  All ROIs share the circle or square set under
*shape* and *size*, above the files.  One ROI can have its own shape instead: a polygon, a
freehand outline or an ellipse (*Polygon*, right-click for the other two).

An ROI can also carry lines drawn by hand, each under a name: the axis of
an elongated structure, the direction a filament runs, the distance between
two spots.  The names are written once for the project, under *line names*
(*line* to start with), and each ROI can have one line of each name.
Pick the name with a right-click on the *Line* button, then click the
button and twice in the ROI image, the start and then the end.  A line is an arrow, so it has a direction: a small ring
marks its start and a head its end, both the same size on the screen at any
zoom.  *Reverse* turns it round.  Each name has its own colour, and its name
is written at the arrow's tip.

Each line gives two numbers in the ROI's row of the results table:
`<name>_angle_deg`, its angle, and `<name>_length_nm`, its length.  So
[Filter ROIs](plugin:Analysis/Regions/Filter ROIs),
[Sort ROIs](plugin:Analysis/Regions/Sort ROIs), the summaries and
[Write results](plugin:Analysis/Regions/Write results) read them
like anything measured.  The keys: L draws the chosen line, Shift+L chooses
the next name, R reverses it.

Every ROI also has an **angle**: the direction its structure points in, 0
until it is set.  It is the ROI's own frame: every step run in the ROI gets
the localizations turned by it (see *In detail*), and it is in the results
table as `roi_angle_deg`.  It is set by hand, here; by the segmenter that
made the ROI (*Segment regions* from an id attribute, *angle*); or by
[Write results](plugin:Analysis/Regions/Write results) from any measured angle.
An angle set by hand is kept apart, and a method run again leaves it alone
unless told otherwise.  With *rotate* ticked, each ROI is shown turned by its
angle, so that its structure lies along $x$ (to the right), and a green
handle sits above it: drag the handle round and the picture turns with it
(Shift turns in steps of 15°), or type the angle, or press [ and ] to turn
by a degree, or take it from a line drawn along the structure (*From
line*).  A square ROI stays square on the screen, so it turns with the
structure; a circle is the same at any angle; a polygon drawn by hand stays
round the localizations it was drawn round.  Lines and polygons are kept in
the file's coordinates, so they stay where they are whichever way the
picture is turned.

*Rotate* is a way of looking, with one exception: with it, a square takes
in the localizations of the turned square, so ticking or unticking it puts
square ROIs' results out of date.  The steps are given each ROI's own
coordinates either way: a step that measures along the structure cannot be
fed the file's axes by a box left unticked.  Turning an ROI puts its
results out of date, ticked or not.

**6. Measuring.**  The steps run in every ROI are chosen, ordered, set up
and run in the [Regions tab](plugin:Panels/Regions tab), under *per-region
steps*.  An ROI selected or added here is measured at once (*measure the
selected ROI*, there), and its numbers are shown under the steps.  With
*draw its figures* ticked there, a second window, *ROI results*, shows what
the steps draw for the selected ROI, one tab per step.  Stepping down the
list redraws it in place, so that ROIs are compared like with like.

**7. Out of date.**  Every result is stored with a record of what it was
computed from.  When the ROI's data change -- the ROI moved, the data were
corrected, a filter was edited -- its results no longer describe it: its
number in the list turns orange, the status bar counts how many wait, and
the ROI drops out of the results until it is measured again.  Editing a
step's parameter also turns every ROI orange, since the steps would now
compute something else, but the results it has stay in the table until
then: they are still true of the ROI, made with the old settings, and say
so.

```figure The first 14 numbers of the ROI list, and the size of the results table, through four edits: after measuring with Localization Statistics everything is current (black); moving ROI 3 by 40 nm puts only that ROI out of date (orange) and out of the results; excluding ROI 6 greys it and takes it out of the results; editing a parameter of the step puts every included ROI out of date, while the results made with the old setting stay until they are replaced.
for r in found:
    r.use, r.lines = True, {}
project.block = Block(steps=[Step(plugin="Analysis/Measure/Localization Statistics", values={"precision_fit": "none"})])
project.evaluate()

def column(ax, title):
    waiting = set(project.needs_evaluation())
    for k, r in enumerate(found[:14]):
        colour = "#8a8a8a" if not r.use else STALE_COLOUR if r.id in waiting else "k"
        ax.text(0.5, 1 - (k + 0.8) / 15, str(k + 1), color=colour, ha="center",
                fontsize=8, transform=ax.transAxes,
                weight="bold" if r.id in waiting else "normal")
    ax.set_title(f"{title}\n{len(project.results())} results", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

fig.set_size_inches(6, 3.2)
axes = fig.subplots(1, 4)
column(axes[0], "measured")
project.move_roi(found[2].id, np.asarray(found[2].center) + 40)
column(axes[1], "ROI 3 moved")
found[5].use = False
column(axes[2], "ROI 6 excluded")
project.block = Block(steps=[Step(plugin="Analysis/Measure/Localization Statistics",
                                  values={"precision_fit": "none", "bins": 50})])
column(axes[3], "a parameter edited")
```

**8. Saving.**  The ROIs are part of the session.  *File > Save* writes them
into the localization file, with their shapes, the *use* flags, how each was
found, the runs of the per-region steps, the steps themselves and where the window was, and
opening the file brings them back.  A draft that was not added is not saved.

## In detail

**Which localizations an ROI holds.**  A circle of diameter $s$ holds the
localizations with $(x - c_x)^2 + (y - c_y)^2 \leq (s/2)^2$, a square of side
$s$ those with $|x - c_x| \leq s/2$ and $|y - c_y| \leq s/2$, and a polygon
those inside it by the even-odd rule, edges included.  Only rows that pass
the layer's filter count, from the merged blinks when the layer merges them.
Overlapping ROIs are independent: a localization in the overlap belongs to
both.  An ROI belongs to one file and only sees that file's localizations.

**Selecting on the outline.**  A click selects the ROI whose outline is
within 6 screen pixels of it, the nearest one if there are several.  For a
circle that distance is $\big| |\mathbf{p} - \mathbf{c}| - s/2 \big|$, for a
square and a polygon the distance to the nearest edge.  A click on a tile's
edge in the file image (also within 6 pixels) jumps to that tile; a click
inside a tile just centres the zoom.

**A line's numbers.**  A line from $(x_0, y_0)$ to $(x_1, y_1)$ has the
angle $\mathrm{atan2}(y_1 - y_0, x_1 - x_0)$ in degrees, from $-180$ to
$180$, counted from $+x$ towards $+y$ -- SMAP's convention.  The images show
$y$ downwards, so a positive angle turns clockwise on the screen.  Its length
is $\sqrt{(x_1 - x_0)^2 + (y_1 - y_0)^2}$ in nm.  Where a step already
writes a column of the same name, the line's is called
`annotation.<name>_angle_deg`.  An old file's one *direction* opens as the
line called `direction`, and its stored results stay current.

**An ROI's own coordinates.**  An ROI at the centre $\mathbf{c}$ with the
angle $\theta$ (the same convention as a line's) has its own frame: a
position $\mathbf{p}$ is, in it,
$\mathbf{u} = R(-\theta)\,(\mathbf{p} - \mathbf{c})$, where $R(\alpha)$
turns by $\alpha$ -- so a structure at $\theta$ lies along $+u$.  Every
step run in an ROI gets these as two extra columns, `x_site_nm` and
`y_site_nm`, beside `x_nm` and `y_nm`, which stay the file's positions: a
step that reports a position to be found again in the file (a circle's
centre, which *Centre ROIs* moves the ROI to) keeps using those, and one
that wants the structure's own axes uses these (SMAP's `xnmrot`).  They
are turned whether or not *rotate* is ticked; an ROI never given an angle
has $\theta = 0$, and `x_site_nm` is then only the distance from the
centre.  The columns exist only while a step runs in an ROI; they
are never written to the file's table.  A square ROI of side $s$ holds what
has $|u_x| \leq s/2$ and $|u_y| \leq s/2$.  An angle is kept between
$-180$° and $180$°.

**What turns the picture.**  The ROI image is drawn as it is and turned on
the screen by $-\theta$ about the ROI's centre, with the outlines and lines
turned with it, so dragging the handle turns it at once and the picture is
rendered again only when the handle is let go.  To fill the image at any
angle, it is rendered $\sqrt{2}$ times as wide at the same pixel size.  A
click is turned back before it is used, so what is drawn is stored in the
file's coordinates.

**Many ROIs.**  Thousands of ROIs -- every [cluster](plugin:Panels/Glossary)
(localizations lying closer to each other than to the rest) of a cell, made by
[Segment regions](plugin:Analysis/Regions/Segment regions) from `cluster_id` -- cost what is on
screen, not what is in the project.  A pane draws only the ROIs that reach
into its view, and when more than 1000 do, a dot at each centre instead of
the outlines, with the selected ROI still outlined; the pane's title says
so, and zooming in brings the outlines back.  The list makes a row only when
it is scrolled to, and checks then whether that ROI is out of date; past
2000 ROIs the status line stops counting the ones waiting, which would mean
checking them all, and the Regions tab counts the ROIs with results rather
than checking each is current.  A click tests only the ROIs near it.

**Tiles.**  The grid starts at the lower corner of the file's extent and has
$\lceil (x_1 - x_0)/t \rceil$ by $\lceil (y_1 - y_0)/t \rceil$ squares of
side $t$, numbered row by row.  It is not stored: it follows the file and
the one size, so it never goes out of date.  The walk wraps from the last
tile to the first, and a new file starts at its first tile.

**What a result depends on.**  For each ROI and each per-region step
the result is stored with a signature, a hash of

* the data: in the GUI, the number of rows of the ROI's file, the attribute
  names, and the rounded positions of up to 512 localizations spread through
  it, so that a drift correction or a refit changes it and a rename does not;
* the geometry: the centre, and the global shape and size or the ROI's own
  polygon, and its lines, since a step may read them;
* the layer's filter ranges, whether blinks are merged, and the merge
  settings when they are;
* the step's plugin, its version and all its parameters, but not the name of
  the step.

A step is *current* when a result with the same signature
exists, *out of date* when there is one under the step's name with another
signature, and *missing* otherwise.  Results of a plugin that is not
installed here cannot be checked and are *unverified*.  An ROI is waiting
(orange) when it is included and any step is not current.  Because the
signature is the content and not the name, running a renamed step with
unchanged settings copies the numbers rather than measuring again, and
changing the global ROI size does not affect ROIs with their own polygon.

**What the results table holds.**  A result belongs to its ROI, whoever ran
it -- the per-region steps, a [chain](plugin:Panels/Glossary) (plugins run
one after another as one), a script.  An ROI's row holds, under each
step's name, the newest result that was computed from the ROI's data as they
are now (the signature without the step's part); the per-region steps' own
come first.  The name is what tells two sets of results apart: running the
same plugin again under its name replaces its numbers, and running it
renamed, with other settings, puts a second set beside the first, so the two
can be compared.  A column keeps its plain name while only one step writes
it, and is called `<name>.<column>` when two do.  Each
result keeps the settings it was made with, which is what an analysis such
as NPC labelling efficiency reads its cutoffs from.

**What runs when an ROI is selected.**  With *measure the selected ROI* on
(the Regions tab), the steps whose stored result for the ROI is missing or
out of date run when it is selected or added, and what runs is stored as a
one-ROI run, so the same work is not done again on the next visit.  With
*draw its figures* on, the step whose tab is open is drawn as well.  A step
that keeps the data its figure is drawn from with its result
(Localization Statistics, Binding kinetics) draws it from what is stored;
any other runs again, current or not, since its figure is drawn from the
localizations.  A step re-run only for its figure, with an unchanged result,
is not stored again.  With both off, selecting an ROI computes nothing: stored results are
shown as they are, still marked as out of date.  A step that fails costs its
own columns, not the ROI or the other steps, and its tab turns orange with
the error in it.

## Controls

### file
The whole file, with every ROI (past 1000, their centres) and, with *Tiles*
on, the grid.  The yellow
frame is what the zoom shows.  Click to centre the zoom there.

### zoom
Three micrometres across to start with.  The wheel zooms, a drag pans, and a
click drafts an ROI or, on an outline, selects one.  An ROI is drawn amber
when included, grey when excluded and green when selected.

### ROI
The selected ROI or the draft, as wide as *ROI view* (two ROI
sizes by default); the wheel here changes that width for all ROIs.  An ROI of
an id is shown as wide as its localizations reach, unless *ROI view* is set.
Since the width changes, a scale bar in the bottom-left corner gives the
size: a round length, about a fifth of the image.
The title says what the ROI holds: its shape or id, its range, how many
localizations over which frames, and for a track the length of its path
and its median step.  A click centres the draft or the selected ROI
there: its centre only, so a polygon and lines stay on the structure they
were drawn on and move with the localizations (one undo step for a stored
ROI; an ROI of an id alone stays where its id is).  A click on one of the
ROI's lines chooses it, and a click on another ROI's outline selects that
one.  It is also where *Polygon* and *Line* are drawn, and where they
stay editable: drag a polygon's corner or an ellipse's handle (the small
squares) or a line's start or end, one undo step each.  Dotted in
orange is what the per-region steps drew over the ROI -- a fitted circle,
a track's axis -- as kept with its results; a picture only, which no
click edits.

### shape
The three fields above the files: what every ROI's outline is, unless it
has a shape of its own, and how much of its surroundings the ROI image
shows.  They were the Regions tab's *geometry*.

*circle* for round structures; *square* for structures that are square or
when the corners do not matter.  A square of side $d$ is 27 % larger in area
than a circle of diameter $d$, so its corners reach further towards a
neighbour.

*track* is not an outline: a click in the ROI manager then drafts an ROI of
the [track](plugin:Panels/Glossary) whose line passes nearest it -- that
track's localizations and none of those crossing it -- and *Add* keeps it.
The tracks are those of the attribute the layer draws them by (the Render
tab's *track id*), and every ROI that has an outline keeps it.  Until a
shape is chosen, *track* is what the list shows whenever the layer is
rendered as *tracks*, and the shape otherwise.

### size
The circle's diameter, or the square's side.  Start at about twice the
structure's diameter: the default 300 nm fits a nuclear pore (about 110 nm)
with room for the localization scatter and a misplaced centre.  Too small,
and the edge of the structure is cut off where the centre is off; too large,
and neighbours and background come in.  Changing it applies to every
circle and square ROI and puts their measurements out of date.

### ROI view
A display setting only: it changes nothing that is measured.  The width of
the field the ROI manager shows around the selected ROI.  At 0 it is twice
the ROI size.  Wider shows more context; narrower, more detail of the
ROI.

### Tiles
Covers the file with a grid of squares of the size beside it and puts the
zoom on the first.  `<` and `>` beside it, or the left and right arrow keys,
step to the previous and next tile.  A tile about ten times the size of a
structure keeps each step to a few dozen ROIs.

### Add
The green button at the end of the row under the ROI image.  Stores the
draft as an ROI, with its shape, lines and ranges if it has them.
Enter does the same.

### Remove
Deletes the selected ROIs -- every one selected in the list, as one step of
*Undo*, without asking: *Undo* brings them all back.  To keep an ROI out of the analysis without
losing it, untick *use* instead (the space bar, or *Don't use* in the list's
right-click menu, for several).  The other ROIs keep their numbers: an ROI
is given its number when it is made and keeps it, and a number is never given
again, so ROI 5 is ROI 5 in the list, in the results and in the `roi_id`
attribute of *Write results* (to the localizations), whatever was removed before it.  *Undo*
brings a removed ROI back under its own number.

### Undo
Under the ROI list.  Takes back the last edit of the ROIs: a segmenter's run (all the ROIs it
added, and any it replaced), ROIs added, removed, included or excluded (several at once are one step),
an outline, a line drawn, reversed or cleared.  Ctrl+Z in this window does the same.  Its tooltip
names the edit it would undo -- *undo: Remove ROI 3*, *undo: Segment regions: 141 ROIs* --
as the main window's *Edit > Undo* names a change of the table (*Undo
RCC*).  This is the ROI manager's own undo, apart
from the main window's, which undoes changes to the localizations: one
never undoes the other.  Measuring is not an undo step, since it can be run
again, and its results stay with the ROI, so an ROI brought back brings
its numbers back.  Twenty steps are kept, and opening another file starts
afresh.

### Redo
Puts back what *Undo* took back, and names it in its tooltip as *Undo* does (Ctrl+Shift+Z, or the platform's own redo key, Ctrl+Y on Windows and Linux).  A new edit
after an undo makes the undone step unrecoverable.

### Add previewed ROIs
Shown while a segmenter's preview is drawn.  Adds the grey proposals as they
were proposed -- with what was measured on each, and the ROIs a replacing
run would remove removed -- as one step of *Undo*, without running the
segmenter again: on a large file a search can take a minute, and the ROIs
added are the ones looked at even if the filter has changed since.  For *from
an id attribute*, the preview and so what is added are the file shown; *Run*
makes the ROIs of every file.

### Discard preview
Shown while a segmenter's preview is drawn.  Takes the grey proposals away
without adding any; *Run* in the segmenter, or *Add previewed ROIs*, adds
them instead.

### Polygon
Draws the ROI's own shape in the ROI image; a right-click on the button
chooses which -- the button is named after it -- or clears the shape.
Escape abandons a shape half drawn.

* *Polygon*: click once per vertex; a double-click, or a click near the
  first vertex again, after at least three, closes the outline.
* *Freehand*: drag round the structure and let go; the path closes itself.
  It is kept as a polygon of the corners that matter: a point is left out
  when it lies within a pixel of the line through those round it
  (Ramer-Douglas-Peucker), so a smooth outline keeps a few dozen corners
  rather than one per mouse event, and its corners can be dragged.
* *Ellipse*: drag the box it fills, upright on the screen.  It is kept as an
  ellipse -- centre, two semi-axes and tilt -- so what it holds is decided
  exactly.  Its four handles, at the ends of its axes, stretch that axis
  and turn the ellipse with the mouse; the other axis stays.  It is not
  turned with the ROI's angle.

The ROI's centre moves to the mean of the vertices, or the ellipse's
centre.  On a draft, the shape goes with it when it is added.  Each shape
replaces the one before.  A right click on a polygon's corner in the ROI
image removes that corner (an outline keeps at least three), one undo step.

### Clear shape
In the *Polygon* button's right-click menu.  Back to the circle or square
of the Regions tab.  An ROI of an id with a
polygon or an ellipse drawn on it (its id inside the shape) goes back to
its id alone.

### Line
Draws the chosen line, which the button is named after (*Line: line*): two
clicks in the ROI image, the start and then the end (L).  A right-click on
the button chooses which line -- the project's names, and any name the
shown ROI has a line of besides -- or reverses or clears the chosen one.
Shift+L chooses the next name; a click on a line in the ROI image chooses
that one.  Drawing a
line the ROI already has replaces it.  A line does not turn the image or
the ROI.  It goes into the ROI's signature, so drawing one puts the ROI's
results out of date, as moving the ROI would.

### Reverse
In the *Line* button's right-click menu.  Swaps the chosen line's start
and end (R): its angle turns by 180°, its
length stays.  For a line drawn the wrong way round, when the direction
matters, as for averaging ROIs.

### Clear line
In the *Line* button's right-click menu.  Removes the chosen line from the
ROI.

### rotate
Under the zoom.  Shows every ROI turned by its angle, with the angle and the handle to turn
it, and turns a square ROI with it.  The steps are given each ROI's turned
coordinates whether or not it is ticked.  The choice is saved with the
ROIs.  Unticked, the pictures are not turned, a square stays upright, and
the angle's controls are hidden.  The
field beside it is the angle of the ROI shown, in degrees from $+x$ towards
$+y$ (clockwise on the screen); on a draft, it goes with the draft when it
is added.

### From line
Sets the angle to the chosen line's, start to end: draw a line along the
structure, then turn the ROI by it.  The line stays, as any other.

### 2D view
Under the zoom.  What selecting an ROI does to the main picture.  *off* (the default)
leaves it alone.  *centred* centres it on the ROI, at the zoom
it has, and filters every localization layer to the ROI's file, as the
Render tab's file list would -- so other files are not drawn over it.
*ROI, in 3D too* does the same and also makes the ROI the 2D
view's own ROI, which the 3D view's box follows while its *follow 2D ROI*
is on: an ROI with a line becomes a line ROI along it, as wide as the ROI
reaches across it -- the chosen line under *Line*, or the first it has
-- so the 3D view looks along the line; an ROI without one is put there
as its shape.  With *rotate* ticked, an ROI with no line chosen is looked
at along its angle instead, through its centre.  That replaces whatever ROI the 2D view had.

### Add drawn region
Under the ROI list, beside *Redo*.  Draw a rectangle, a circle, a line or a polygon in the 2D view first; it
becomes an ROI of the file shown here, and is selected.  Useful for
structures that are not round, or a region the automatic finder would not
propose; the outline is kept as it is.

### annotate
What the selected ROI is, in the user's own words.  The keys work wherever
the window has the keyboard, except in a text field, which keeps its own:
typing a 1 into *lists* writes a 1.

### use
The same switch as the list's *use* box, and the space bar: whether the
per-region steps and the results include the ROI.  This box is the shown
ROI's alone; the space bar sets every selected ROI.

### next ROI after a keyword
On, a keyword selects the next ROI once given: one key an ROI.  Off, the
ROI stays selected, for giving it keywords from several lists.

### lists
One list a line: a name, a colon, and its keywords separated by commas
(`quality: good, bad, unsure`).  The number keys reach the first nine of a
list.  Saved with the ROIs.  An ROI keeps a keyword its list no longer has,
so a list rewritten by mistake loses nothing already annotated.  Keywords
can also be given from what was measured, by a condition
([Write results](plugin:Analysis/Regions/Write results)): one
given here by hand is left alone by it.

### ranges
The ranges of the ROI the ROI image shows -- the selected one, or the draft
while there is one: the part of it that is measured.  A row per attribute
listed under *attributes*, frame and z unless changed.  Each ROI has its
own; a new draft starts with none, and *Add* gives the ROI the draft's.  A
range can also be set from what was measured -- a tail's last frame, a
structure's z and its spread -- by
[Write results](plugin:Analysis/Regions/Write results), which
leaves a range set here by hand alone.  The ROI image leaves out the
localizations inside the ROI that the ranges leave out, so it shows what is
measured, with everything round the ROI as it is; the file image and the
zoom show the file whatever an ROI's ranges.

Each row has two fields, the lowest and highest value kept, both included
(blank is an open end, which only a field emptied by hand or *Clear* gives), and *Clear* and *To all used* for that range.
Under them a plot of the ROI as it is without its
ranges, in one of two kinds:

* **Bars** for an attribute where each value has many localizations -- a
  time or an id (`frame`, `time_s`, `tid`, `..._id`): the localizations
  counted over it.  Frames and ids are whole numbers, and a band let go
  is rounded to them.
* **Dots** for a measured quantity -- `z_nm`, `sigma_nm`, `photons`: the
  attribute against x, or against the distance along the ROI's line called
  *line* (*direction* in an older project) when it has one, so a tilted
  structure is seen edge on.

The green band is the range; drag its edges to set it.  A drag inside the
band moves it whole only with Shift held, so a drag near an edge sets that
end rather than moving the window.  The plot's scale stays where it is
while an edge is dragged, and takes in a range set beyond the data.  An
end is kept as it was let go or typed, even at or beyond the edge of the
data; only *Clear* makes the range all of the attribute again.  Each change is one step
of the ROI undo, and puts the ROI's results out of date (a draft's, no
step: it is not stored yet).  An attribute the table does not have is said
so, and its range cuts nothing.

### attributes
The attributes ranges are offered on, separated by commas, in the order
the rows are shown: `frame, z_nm` by default.  For MINFLUX, `time_s` in
place of `frame` cuts a trace in seconds.  Saved with the ROIs.  Taking an
attribute out of the list keeps the ranges already set on it.

### Clear
On each range's row.  Clears that range of the ROI shown: all of the
attribute again.

### To all used
On each range's row.  Gives every ROI ticked *use* the ROI shown's range of
that attribute -- or none, if it has none -- in one undo step: for a range
that should be the same everywhere, set once.  The other ranges stay.

### line names
The names an ROI's lines are drawn under, separated by commas
(`line, axis`).  Each is the start of two column names, so spaces
become `_` and other signs are dropped.  Saved with the ROIs.  Removing a
name keeps the lines already drawn under it.

## Differences from SMAP

Based on SMAP's ROI manager, the *site explorer*
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), its `SEExploreGui`
and `SEEvaluationGui`, which calls an ROI a *site* and a measuring plugin an
*evaluator*: the images beside the lists, the global ROI size, the *use* flag
and a list of evaluators with a tab each are its design.

* **Files and ROIs, no cells.**  SMAP has three levels, file, cell and site,
  each with its own list and image.  Here an ROI belongs directly to its file.
  *Tiles* takes the place of cells for walking through a file, and is a grid
  rather than a set of objects to create.
* **Out-of-date results.**  SMAP stores each evaluator's output with the site
  and, with evaluation switched on, runs every enabled evaluator again when a
  site is shown; it does not record what a stored result was computed from.  Here every result carries
  the signature of its data, geometry, filters and parameters, results that
  no longer match are marked and left out of the table, and only the
  step being looked at is run on selection.
* **An angle and named lines.**  SMAP turns a site by a rotation line
  (`rotationpos`, set by clicks beside the site image or a typed angle) and
  has two fixed lines per site, *line 1* and *line 2*.  Here the turn is the
  ROI's angle, set with a handle, and a site has any number of lines, each
  under a name the project defines; a SMAP file's rotation line becomes
  both the angle and the line `direction`.  The angle is SMAP's,
  `atan2(dy, dx)` in degrees; the angle and the lines' angles and lengths
  are columns of the results table, where SMAP keeps them in the site's
  annotation for evaluators to read.  SMAP gives every evaluator the
  rotated coordinates (`xnmrot`, `ynmrot`) whether or not its image is
  turned, and so does this: `x_site_nm` and `y_site_nm` are turned by the
  ROI's angle whether or not *rotate* shows it.  SMAP stores a shape
  drawn on a turned site in the turned image's coordinates; here it is
  stored in the file's.  SMAP's polar angle, the turn in 3D, is not
  ported.  An ROI can have its own polygon outline.
* **Annotation lists.**  SMAP has four lists, loaded from a text file, each
  a list box, with the arrow keys moving between lists and entries.  Here the
  lists are written in the window, as many as wanted, with a name each, and
  saved with the ROIs; a number key gives a keyword, and the keywords are
  columns of the results table.
* **Not ported.**  SMAP's free comment.  Its sorting and filtering of sites are the plugins
  [Sort ROIs](plugin:Analysis/Regions/Sort ROIs) and
  [Filter ROIs](plugin:Analysis/Regions/Filter ROIs).

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
* Thevathasan JV, Kahnwald M, Cieśliński K, et al. Nuclear pores as
  versatile reference standards for quantitative superresolution microscopy.
  *Nat Methods* 16, 1045 (2019).
  [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)
  -- the Nup96 geometry of the simulated pores.
