---
version: "1"
covers: [smappy.plugins.remove_locs.region_mask, smappy.plugins.remove_locs.current_roi, smappy.plugins.remove_locs.drop, smappy.plugins.remove_locs.hide, smappy.plugins.remove_locs.check_field, smappy.regions.Region.mask]
---

## What it does

Some localizations are not wanted: a fluorescent dirt particle, a bead, a
cell that was out of focus, or simply everything except the one cell out of
a field of twenty that is to be analysed.  This plugin takes a region and
either throws away what is **inside** it or what is **outside** it.

It can do that in two ways (*action*):

* **remove** -- the localizations are deleted from the table.  Every other
  plugin, every layer and a save afterwards see the smaller table.
* **hide** -- nothing is deleted.  The localizations are marked in a column
  (`use` by default: 1 kept, 0 hidden) and the filter is set on that column,
  so they disappear from the picture and from what the other plugins get, but
  are still in the file and come back when the filter is opened again.

Hiding is the safer of the two.  A removal can be undone only as long as the
session lasts; once the table is saved, the removed localizations are gone
from the file.  A hidden localization is one slider away, now or next week.

It needs a table and, for the usual use, an ROI drawn in the render window.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.regions import Region
from smappy.group import group
from smappy.plugins.remove_locs import region_mask, drop, hide
locs = simulate(n_frames=3000, seed=4)
roi = Region.rect(3000, 3000, 6500, 6500)
inside = region_mask(locs, roi)

def show(ax, table, title):
    x, y = np.asarray(table["x_nm"]), np.asarray(table["y_nm"])
    image, _, _ = np.histogram2d(y, x, bins=200, range=[[0, 10000], [0, 10000]])
    ax.imshow(image, cmap="magma", origin="lower", extent=(0, 10, 0, 10),
              vmax=max(1, np.percentile(image[image > 0], 98)))
    x0, y0, x1, y1 = np.array(roi.bounds) / 1000
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#4cc9f0", lw=1)
    ax.set_title(f"{title}\n{len(table)} localizations", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
```

**1. Which localizations are in the region.**  With *region* set to *the
drawn ROI*, a localization is inside when it lies within the shape drawn in
the render window, in the picture's own coordinates -- its position (`x_nm`,
`y_nm`, or the pixel columns of a pixel table) on the ordinary picture, and
whatever the axes show when the picture is of other columns (photons against
frame, say): a rectangle, a polygon, or a line with a width (a line is a rectangle
of that width around the segment).  The ROI is taken on its own: the layer's
filter does not narrow it, so every localization of the table in the drawn
shape counts, whichever file or layer it belongs to.

With *region* set to *the selection*, "inside" means everything that
decides what the current layer shows: its filter, the ROI if one is drawn,
and the 3D slab while the 3D window is set to have plugins use it.  This is
the way to say "keep exactly what I see" (*remove* the localizations
*outside* the selection), or to remove the localizations a filter already
hides.

**2. Which side goes.**  The setting *remove* chooses: *the localizations
inside* empties the region, *the localizations outside* keeps it and nothing
else.

```figure A simulated field of view with an ROI drawn over it (blue).  Removing the localizations inside the ROI cuts a hole; removing those outside keeps the ROI alone.
fig.set_size_inches(7, 2.8)
axes = fig.subplots(1, 3)
show(axes[0], locs, "before")
show(axes[1], drop(locs, ~inside), "remove inside")
show(axes[2], drop(locs, inside), "remove outside")
```

**3a. Remove.**  A new table is made with only the localizations that stay,
and it replaces the old one.  The old one goes on the undo stack, under the
plugin's name, so *File > Undo* brings it back.

**3b. Hide.**  The flag column is written: 1 for a localization that stays,
0 for one that is hidden.  If the column is already there, the two are
combined -- a localization already hidden stays hidden -- so hiding one dirt
particle and then another hides both.  Then the filter of every localization
layer is set to keep `use` $\geq$ 0.5.  No row is deleted, and moving that
bound back (or removing it) shows everything again.

Before anything is written the plugin counts what would go, and refuses a
run that would take nothing (nothing is on that side of the region) or
everything (it would leave an empty table).  *Preview* says the same count
without changing the table.

## In detail

**Inside a shape.**  A rectangle is a test of $x$ and $y$ against its edges,
the edges included.  A polygon, and a line (stored as the four corners of its
rectangle), is tested with the even-odd rule: a point is inside when a ray
from it crosses the outline an odd number of times.  Only the points in the
polygon's bounding box are tested, so a small ROI on a large table is quick.

**The whole table.**  Both actions work on the whole ungrouped table, not on
the selection of one layer: a localization inside the drawn ROI goes (or is
hidden) whatever layer or file it is in.  To act on one file only, set
*region* to *the selection* with the layer showing that file.

**A hidden flag and grouping.**  Once the localizations are linked into
blinks, a blink is kept only if **all** of its localizations are.  The flag
is stored with a rule for grouping (a recipe with the rule *all* and no
expression, in the table's `metadata["derived"]`), so it is not averaged
like a measured column: a blink that straddles the edge of the hidden region
is hidden as a whole rather than coming back as a blink with half a
flag.  See [Math Parser](plugin:Analysis/Process/Math Parser) for how
recipes work.

**The flag's name.**  *flag* must be a word of letters, digits and
underscores that does not start with a digit, so that it can be used in a
filter and in an expression.  `group_id` and `n_in_group` are refused,
because grouping writes them itself and would overwrite the flag.

## Parameters

### which
*the localizations outside* together with *the selection* keeps what is on
screen and nothing else.

### region
*the drawn ROI* needs an ROI in the render window; the plugin refuses to run
without one.

### action
Prefer *hide* while deciding; *remove* makes a smaller table and file, which
is worth it only once the decision is final.

### field
Only used when hiding.  Keep the default unless two different kinds of
hiding should be kept apart -- `dirt` and `cell`, say -- each with its own
filter.

## Output

* **The table** -- smaller after *remove*; with the flag column after
  *hide*, and the filter set on it in every localization layer.  A layer that
  was showing the grouped table is grouped again from the new one.
* **The text** says how many of how many localizations went, and on which
  side of which region: "removed 5619 of 24693 localizations inside the rect
  ROI 3500 x 3500; 19074 left".  After hiding it also says how many are
  hidden in all, counting earlier runs.

The run is recorded in the file's history, with its settings, and can be
undone.  The flag column and its grouping rule are saved with the file, so
the decision survives a reload.  So does the filter on it: the bound is kept
with the table, and a reopened file hides the same localizations again.

## Differences from SMAP

SMAP's `Process/Modify/RemoveLocs` does the same job, with three modes
("remove inside ROI", "remove outside ROI", "keep visible inside ROI"), an
*all files* checkbox and a *set property: active* checkbox.  Here:

* **What counts as inside** is one setting.  SMAP's third mode quietly adds
  the layers' filters to the ROI; here that is *region* = *the selection*
  (the filter, the ROI and the slab), and it combines with either side.
* **Which files** are acted on is not a setting.  SMAP restricts the first two
  modes to the files shown in the layers unless *all files* is ticked; here
  *the drawn ROI* acts on every file, and *the selection* on what the layer
  shows, which is how one file is picked.
* **Hiding.**  SMAP's *set property: active* writes an `active` field,
  combined with an earlier one, as here.  Here the column is named by *flag*,
  the filter is set on it so the hidden localizations actually disappear, and
  it carries the rule *all* for grouping.  SMAP regroups the table after
  every run; here nothing is relinked unless a layer is showing the grouped
  table.
* A run that would remove nothing or everything is refused.
