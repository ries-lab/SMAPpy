---
version: "1"
covers: [smappy.plugins.remove_locs.region_mask, smappy.plugins.remove_locs.current_roi, smappy.plugins.remove_locs.drop, smappy.plugins.remove_locs.hide, smappy.plugins.remove_locs.check_field, smappy.plugins.remove_locs.included_rois_mask, smappy.roi_manager.core.ROIProject.rows_of, smappy.regions.Region.mask, smappy.plugins.remove_locs.edges_mask, smappy.group.event_cores]
---

## What it does

Some localizations are not wanted: a fluorescent dirt particle, a bead, a
cell that was out of focus, or simply everything except the one cell out of
a field of twenty that is to be analysed.  This plugin takes a region and
either throws away what is **inside** it or what is **outside** it.

It can do that in two ways (*action*):

* **remove** -- the localizations are deleted from the table.  Every other
  plugin, every layer and a save afterwards see the smaller table.
* **hide** -- nothing is deleted.  The localizations are marked in an attribute
  (`use` by default: 1 kept, 0 hidden) and the [filter](plugin:Panels/Glossary)
  -- also called bounds: the lowest and highest values of attributes a
  localization must lie between -- is set on that attribute,
  so they disappear from the picture and from what the other plugins get, but
  are still in the file and come back when the filter is opened again.

Hiding is the safer of the two.  A removal can be undone only as long as the
session lasts; once the table is saved, the removed localizations are gone
from the file.  A hidden localization is one slider away, now or next week.

It needs a table and, for the usual use, an [ROI](plugin:Panels/Glossary)
(region of interest: a rectangle, polygon or line with a width) drawn in the
render window.

The region can also be one in time: the **binding-event edges** of
DNA-PAINT, in which a short DNA strand carrying the dye (the imager) binds
to its target for a few frames and leaves again.  The imager rarely binds
exactly as a frame starts or leaves exactly as one ends, so the first and
the last frame of each binding event saw it for only part of the exposure:
those localizations have too few photons and a worse precision.  Removing
them -- keeping the event's *core* -- is the selection of Steen et al.
([2024](https://doi.org/10.1038/s41592-024-02374-8)), as Picasso
([Schnitzbauer et al. 2017](https://doi.org/10.1038/nprot.2017.024)) does
it, and is worth doing before anything that weighs localizations by their
photons or their precision, or averages them into one position per site.
It needs the localizations' frames and, for the default linking distance,
the table's pixel size.

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
`y_nm`, or the pixel attributes of a pixel table) on the ordinary picture, and
whatever the axes show when the picture is of other attributes (photons against
frame, say): a rectangle, a polygon, or a line with a width (a line is a rectangle
of that width around the segment).  The ROI is taken on its own: the layer's
filter does not narrow it, so every localization of the table in the drawn
shape counts, whichever file or layer it belongs to.

With *region* set to *the selection*, "inside" means everything that
decides what the current [layer](plugin:Panels/Glossary) (one picture of the
localizations, with its own filter and colours) shows: its filter, the ROI if one is drawn,
and the 3D slab while the 3D window is set to have plugins use it.  This is
the way to say "keep exactly what I see" (*remove* the localizations
*outside* the selection), or to remove the localizations a filter already
hides.

With *region* set to *all included ROIs*, "inside" means inside any ROI
of the [ROI manager](plugin:Panels/ROI manager) whose *use* is ticked, each
ROI in its own file, by its shape alone (the layer's filter does not
narrow it, as for a measurement in the ROI).  *remove* the localizations
*outside* them and a field of a few hundred segmented pores shrinks to the
pores, which makes a much smaller file; for clusters, a filter on
`cluster_id` $\geq$ 1 does the same without the ROI manager.

With *region* set to *binding-event edges*, the localizations are first
linked into binding events: a localization joins an event when it lies in
the frame after the event's last one (with *dark frames* above 0, up to
that many frames later) and within *within* of the event's position --
by default a tenth of a camera pixel, 13 nm for a 130 nm pixel.  "Inside"
is then every localization in the first or the last frame of its event,
and every localization of an event with fewer than *min localizations*
(with three, such an event has no frame that is neither).  Files and
channels are linked apart.

```figure The photons of simulated binding events, each localization's photons in proportion to the part of the frame the imager was bound (the truth: the whole frame, 3000 photons, dashed).  All localizations (grey) have a tail of dim ones -- the edges; the cores (orange) are the whole frames.
from smappy.simulate.binding import bound_intervals
from smappy.group import event_cores
rng = np.random.default_rng(1)
site, start, end = bound_intervals(30, 20000, 8.0, 1500.0, rng)
first = np.floor(start).astype(int)
span = np.ceil(end).astype(int) - first
which = np.repeat(np.arange(len(site)), span)
frame = first[which] + np.concatenate([np.arange(n) for n in span])
cover = np.minimum(end[which], frame + 1.0) - np.maximum(start[which], frame)
lit = cover >= 0.5
which, frame, cover = which[lit], frame[lit], cover[lit]
order = np.argsort(frame, kind="stable")
which, frame, cover = which[order], frame[order], cover[order]
s = site[which]
xy = np.column_stack([s % 6, s // 6]) * 1000.0 + rng.normal(0, 2, (len(s), 2))
photons = rng.poisson(3000 * cover)
core, _ = event_cores(xy[:, 0], xy[:, 1], frame, 13.0)
ax = fig.subplots()
bins = np.linspace(1000, 3400, 49)
ax.hist(photons, bins, color="0.6", label=f"all, mean {photons.mean():.0f}")
ax.hist(photons[core], bins, color="#e07a1f", alpha=0.8,
        label=f"cores, mean {photons[core].mean():.0f}")
ax.axvline(3000, color="k", ls="--", lw=1)
ax.set_xlabel("photons"); ax.set_ylabel("localizations"); ax.legend(fontsize=8)
```

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

**3b. Hide.**  The flag attribute is written: 1 for a localization that stays,
0 for one that is hidden.  If the attribute is already there, the two are
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

**The whole table.**  Both actions work on the whole table of single
localizations, not on
the selection of one layer: a localization inside the drawn ROI goes (or is
hidden) whatever layer or file it is in.  To act on one file only, set
*region* to *the selection* with the layer showing that file.  The selection
is then the layer's bounds applied to each localization, also when the
layer shows blinks.

**A hidden flag and merged blinks.**  Once the localizations of a blink are
merged -- [merge blinks](plugin:Panels/Glossary), also called grouping or
linking -- a blink is kept only if **all** of its localizations are.  The
flag is stored with a rule for merging (a recipe with the rule *all* and no
expression, in the table's `metadata["derived"]`), so it is not averaged
like a measured attribute: a blink that straddles the edge of the hidden region
is hidden as a whole rather than coming back as a blink with half a
flag.  See [Math Parser](plugin:Analysis/Process/Math Parser) for how
recipes work.

**Binding-event edges.**  The events are linked by the same walk as
[merge blinks](plugin:Panels/Glossary) uses, given no precision, so that
its test is a circle: a localization joins when its distance from the
event's running position is less than *within*.  The running position is
the mean of the last position and the new one, each time a localization
joins; Picasso tests the distance to the last localization alone.  Two
consecutive localizations of one event are $\sqrt{2}\sigma$ apart along
each axis (with $\sigma$ the precision), and one from the running position
about $1.15\sigma$, so at the same radius an event breaks in two a little
less often here.  Either way the radius must be several times the
precision: at 13 nm, localizations precise to 5 nm break an event between
two frames about one time in eight (one in five with Picasso's test), and
each break makes two more edges; precise to 3 nm, about one time in 800.  In both programs the first localization in range in the next frame
joins, not the nearest.  Picasso's constants are the defaults: a tenth of
a pixel, no dark frame, three localizations.

Hidden rather than removed, the edges are flagged with the rule *any* for
merging, not *all*: every merged blink has edges, so with *all* every one
would be hidden.  A merged blink is then shown when it has a core, but its
position and photons are still merged from all its localizations, the
edges included.  To merge blinks from their cores alone, *remove* the
edges.

**The flag's name.**  *flag* must be a word of letters, digits and
underscores that does not start with a digit, so that it can be used in a
filter and in an expression.  `group_id` and `n_in_group` are refused,
because merging blinks writes them itself and would overwrite the flag.  So is a
attribute the table already has, unless it is a flag itself: hiding writes 0 and
1 over it, and a flag named `photons` would have replaced the photons.

## Parameters

### which
*the localizations outside* together with *the selection* keeps what is on
screen and nothing else.

### region
*the drawn ROI* needs an ROI in the render window, *all included ROIs* at
least one used ROI in the ROI manager; the plugin refuses to run without
them.

### edges.within_nm
The default, a tenth of a camera pixel, is Picasso's and suits DNA-PAINT,
whose bright events are localized to a few nanometres.  For dimmer data
raise it to about three times the precision; too small and events break
into pieces, each with edges of its own, too large and two sites close
together are linked into one event.

### edges.dark_frames
0, as in Picasso: one frame without a localization ends the event.  With 1,
an event missed in one frame (a localization lost to the filter) stays one
event, but two bindings of one site a frame apart are joined.

### edges.min_locs
Three is the least that leaves a core.  Raise it to keep only long events,
whose cores are many frames.

### action
Prefer *hide* while deciding; *remove* makes a smaller table and file, which
is worth it only once the decision is final.

### field
Only used when hiding.  Keep the default unless two different kinds of
hiding should be kept apart -- `dirt` and `cell`, say -- each with its own
filter.

## Output

* **The table** -- smaller after *remove*; with the flag attribute after
  *hide*, and the filter set on it in every localization layer.  A layer that
  merges blinks has them merged again from the new one.
* **The text** says how many of how many localizations went, and on which
  side of which region: "removed 5619 of 24693 localizations inside the rect
  ROI 3500 x 3500; 19074 left".  After hiding it also says how many are
  hidden in all, counting earlier runs.

The run is recorded in the file's history, with its settings, and can be
undone.  The flag attribute and its merging rule are saved with the file, so
the decision survives a reload.  So does the filter on it: the bound is kept
with the table, and a reopened file hides the same localizations again.

## Differences from SMAP

Based on SMAP's `Process/Modify/RemoveLocs`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  Here:

* **What counts as inside** is one setting.  SMAP's "keep visible inside ROI"
  mode quietly adds the layers' filters to the ROI; here that is *region* =
  *the selection* (the filter, the ROI and the slab), and it combines with
  either side.
* **Hiding.**  SMAP's *set property: active* writes an `active` field and
  regroups the table after every run.  Here the attribute is named by *flag*,
  the filter is set on it so the hidden localizations actually disappear, and
  it carries the rule *all* for merging blinks, so nothing is merged again
  unless a layer merges blinks.
* **The ROIs.**  SMAP's *RemoveLocsOutsideSites*, a plugin of the ROI
  manager, keeps the localizations in a square of the site size around each
  site, whatever the sites' shape, and only deletes.  Here it is *region* =
  *all included ROIs*: each ROI's own shape, the used ROIs only, and it can
  hide as well as remove, and remove the inside instead.

## References

* Steen PR, Unterauer EM, Masullo LA, Kwon J, Perovic A, Jevdokimenko K,
  Opazo F, Fornasiero EF, Jungmann R. The DNA-PAINT palette: a
  comprehensive performance analysis of fluorescent dyes. *Nat Methods*
  21, 1755 (2024).
  [doi:10.1038/s41592-024-02374-8](https://doi.org/10.1038/s41592-024-02374-8)
  -- the selection of binding-event cores (Extended Data Fig. 1f).
* Schnitzbauer J, Strauss MT, Schlichthaerle T, Schueder F, Jungmann R.
  Super-resolution microscopy with DNA-PAINT. *Nat Protoc* 12, 1198 (2017).
  [doi:10.1038/nprot.2017.024](https://doi.org/10.1038/nprot.2017.024)
  -- Picasso, whose `select_binding_event_cores` this follows.

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
