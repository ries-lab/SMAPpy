---
version: "1"
covers: [smappy.plugins.average_rois.AverageROIs.run, smappy.plugins.average_rois.superimpose, smappy.plugins.average_rois.placement, smappy.plugins.average_rois.line_point, smappy.plugins.average_rois.average, smappy.plugins.average_rois.draw_average]
---

## What it does

One copy of a small structure -- a nuclear pore, a kinetochore, a synapse --
holds few localizations, and a missing label or a stray blink can make it
look like something else.  Many copies, laid on top of each other, hold
many, and show the structure as it is on average.  This plugin does that:
it takes every used [ROI](plugin:Panels/Glossary) (region of interest: a
shape kept in the [ROI manager](plugin:Panels/ROI manager), inside which
things are measured), moves its centre to one point, turns it so that it
points one way, and writes all of them, superimposed, into one new table.

The centre is the ROI's own centre, a centre that was measured in it (by
[Cluster shape](plugin:Analysis/Measure/Cluster shape), say), or a point of
a line drawn on it in the ROI manager.  The orientation is none, a line drawn
on it, or an angle that was measured.  For a structure without a direction,
such as a pore seen from the top, a centre is enough.  For one with a
direction, draw a line along it on each ROI (*Draw line* in the ROI
manager), and turn by that line.

The new table is added to the session as a file of its own, beside the data,
and can be rendered, filtered and measured like any other.  It holds the
localizations themselves, not merged blinks, with every column they had.

## How it works

```figure-setup
import tempfile, yaml
from matplotlib.patches import Circle
from pathlib import Path
from smappy.simulate import simulate
from smappy.simulate.settings import SimulationSettings, StructureSettings
from smappy.locs import Localizations
from smappy.plugins.average_rois import superimpose, average, draw_average
# an arrow of three labels, 25 copies each turned at random
ARROW = [[-50, 0, 0], [50, 0, 0], [50, 40, 0]]
spec = {"name": "arrow", "elements": [{"points": ARROW}],
        "copies": {"n": 25, "placement": "grid", "rotation": "random",
                   "field": [0, 0, 5000, 5000]}}
path = Path(tempfile.mkdtemp()) / "arrow.yaml"
path.write_text(yaml.safe_dump(spec))
locs = simulate(SimulationSettings(n_frames=2000, seed=3,
                                   structure=StructureSettings(file=str(path))))
locs = locs[np.asarray(locs["photons"]) > 200]
copies = locs.metadata["copies"]
rng = np.random.default_rng(1)
x, y = np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])
parts_centre, parts_line, lines = [], [], []
for k, (cx, cy, gamma) in enumerate(zip(copies["x_nm"], copies["y_nm"],
                                        copies["gamma_deg"])):
    inside = np.hypot(x - cx, y - cy) < 125
    # a line drawn by hand along the arrow: a few degrees off
    angle = gamma + rng.normal(0, 4)
    lines.append((cx, cy, angle))
    parts_centre.append((locs[inside], (cx, cy), 0.0, k + 1))
    parts_line.append((locs[inside], (cx, cy), angle, k + 1))
centred = average(parts_centre)
turned = average(parts_line)
```

**1. Each ROI's place and turn.**  For every used ROI the plugin works out
two things.  Its centre: the ROI's centre, the measured *x value* and *y
value*, or the start, middle or end of a named line.  Its angle: 0, the
angle of a named line, or a measured *angle value*.  An ROI that lacks one
of them -- no line of that name, nothing measured -- is left out, and the
line under the panel says how many and why.

**2. Moving and turning.**  Each localization of the ROI is moved so that
the centre lies at (0, 0), and turned about it by minus the angle.  A line
at 30° then lies along +x, so every ROI ends up pointing the same way.  The
localizations are all those inside the ROI's shape, not only those a layer's
[filter](plugin:Panels/Glossary) (the lowest and highest values a
localization's columns may have) shows -- unless *only what the layers show*
is ticked.

**3. One table.**  The moved localizations of all ROIs go into one table,
each with `roi_id`, the number of the ROI it came from.  The table keeps
every other column, so a layer that filters on `channel` still shows one
colour of a two-colour average.

```figure Twenty-five copies of an arrow of three labels (circles), each turned at random.  Left: four of them in their ROIs (green), with the line drawn along each (magenta).  Middle: superimposed on their centres only, the arrow is smeared into rings.  Right: turned by their lines as well, the three labels come back where they are.
fig.set_size_inches(9, 3.1)
a, b, c = fig.subplots(1, 3)
for k, (cx, cy, angle) in enumerate(lines[:4]):
    # each drawn in a slot of a 2 x 2 grid, 300 nm apart, not where it lies
    ox, oy = 300 * (k % 2), 300 * (k // 2)
    near = np.hypot(x - cx, y - cy) < 125
    a.scatter(x[near] - cx + ox, y[near] - cy + oy, s=2, color="0.4")
    a.add_patch(Circle((ox, oy), 125, fill=False, color="#2e9e4f", lw=1))
    t = np.radians(angle)
    a.annotate("", (ox + 70 * np.cos(t), oy + 70 * np.sin(t)), (ox, oy),
               arrowprops=dict(arrowstyle="->", color="#e05fd0", lw=1.5))
a.set_aspect("equal")
a.set_title("four ROIs and their lines", fontsize=9)
a.set_xlim(-150, 450); a.set_ylim(450, -150)
a.set_xlabel("x (nm)")
for ax, table, title in ((b, centred, "centred"), (c, turned, "centred and turned")):
    tx, ty = np.asarray(table["x_nm"]), np.asarray(table["y_nm"])
    ax.scatter(tx, ty, s=1, color="0.3", alpha=0.5)
    for p in ARROW:
        ax.add_patch(Circle(p[:2], 12, fill=False, color="#1f77b4", lw=1.2))
    ax.set_xlim(-110, 110); ax.set_ylim(110, -110); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (nm)")
```

## In detail

**The turn.**  A localization at $(x, y)$ of an ROI with centre $(c_x, c_y)$
and angle $\theta$ goes to

$$x' = (x - c_x)\cos\theta + (y - c_y)\sin\theta, \qquad y' = -(x - c_x)\sin\theta + (y - c_y)\cos\theta,$$

SMAP's `rotcoord`.  A line's angle is $\mathrm{atan2}(y_1 - y_0, x_1 - x_0)$
from its start to its end, as the ROI manager writes it into
`<line>_angle_deg`, so the line's start goes to $-x$ and its end to $+x$.  A
line drawn the wrong way round turns its ROI by 180°: *Reverse* in the ROI
manager fixes it.  The images show $y$ downwards, so the average comes out as
the ROI manager shows the ROIs, turned until their lines point to the right.
A measured *angle value* is read the same way, in degrees from $+x$ towards
$+y$; an axis measured without a direction, such as Cluster shape's
`angle_deg` (0 to 180°), leaves the average with a two-fold ambiguity.

**Measured values** are read from the results table as the ROI manager
shows it: the current results of each included ROI, by the column's name
there (`centre_x_nm`, or `Cluster shape.centre_x_nm` when two steps write
it), and the lines' `<line>_angle_deg`.  A centre is in nm, in the
coordinates of the data, not relative to the ROI.  An ROI whose results are
out of date has none, and is left out.

**Frames.**  Merging blinks links localizations in consecutive frames that
lie close together.  After superimposing, two ROIs' blinks in the same
frames lie close together, and would be merged into one.  So each ROI's
frames are moved past the previous one's: the $k$-th ROI's frames have
$k \cdot F$ added, with $F$ one more than the largest frame of the table.
The frame in its own file is kept as `frame_in_file`.

**Where it goes.**  The average is centred on (0, 0), so `x_nm` is the
position along the orientation and `y_nm` across it.  *z* is not turned or
moved.

## Parameters

### centre.by
*the ROI's centre* suits ROIs placed on their structures by a segmenter,
or moved onto them by [Centre ROIs](plugin:Analysis/Regions/Centre ROIs).
For ROIs drawn by hand, or a structure whose centre is better found than
guessed, measure the centre first and use *measured values*: `x_nm` and
`y_nm` of [Circle fit](plugin:Analysis/Measure/Circle fit) for a ring,
`centre_x_nm` and `centre_y_nm` of Cluster shape for the mean position.
*a point of a line* centres on, say, one end of a kinetochore pair.

### orientation.by
*as it is* averages without turning: right for a structure without a
direction, or one imaged always the same way round.

### filtered
Off, as SMAP: every localization in the ROI, whatever the layers' filters.
On, those that pass the filter of at least one localization layer -- the
same photons or precision cut as the picture.

### output
*a new file* keeps the data and puts the average beside it, as file of its
own: show it in a layer that shows that file.  *the table* replaces the
session's table with the average; *Edit > Undo* brings the data back.

## Output

A new file, *average of N ROIs*, holding the superimposed localizations
with `roi_id` and `frame_in_file` beside their own columns.  The figure
shows the average as a 2D histogram, with $x$ along the orientation, and
its profiles along $x$ and $y$.  A sharp average with the structure's
features in place is good; rings or a smear where the structure has
corners means the orientation is wrong for some ROIs -- lines drawn the
wrong way round, or an angle without a direction.

## Differences from SMAP

Based on SMAP's *AverageSites* and its variants *AverageSites_kinetochore*,
*_window* and *_NPC* ([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* **One plugin, the centre and the turn as choices.**  *AverageSites* only
  centres; *_window* and *_NPC* turn by the site's own rotation and a
  LocMoFit model; *_kinetochore* centres on a point of one annotation line
  and turns by the other, and draws images rather than writing a table.
  Here any of the centres goes with any of the turns.  LocMoFit's model is
  not ported.
* **Frames per ROI.**  SMAP regroups the averaged table as it is, so blinks
  of different sites in the same frames can be merged.  Here each ROI has its
  own frames.
* **Centred on (0, 0).**  SMAP moves the average to the median position of
  the data.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
