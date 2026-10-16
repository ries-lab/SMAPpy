# Regions: ROIs, clusters and what is measured in them

SMAP measured the same things three times: in an Analyze plugin over the
view, in an ROI manager evaluator per site, and in cluster code per cluster.
Here a **region** is a set of localizations with an id -- an ROI of the ROI
manager, or the rows that share a value of an id column such as
`cluster_id` -- and every measurement is an ordinary plugin that runs the
same on the view, on one region, or on every region in turn.  Running a list
of plugins on every region is a **block** of a chain; what it measured is a
**region table**, a row per region.  The ROI manager finds, reviews,
annotates and organises ROIs; it is not the only place they can be measured.

## A region

* **An ROI** (`roi_manager.core.ROI`, kept by the session's `ROIProject`,
  `Session.rois`, which takes its data, filters and grouping from the
  session).  Its shape is a `regions.Region` (rect, polygon, line, circle):
  its own polygon, or the project's circle or square of `size_nm` round its
  centre, worked out when asked for (`ROI.region`) so that changing the
  project's shape is one edit for every ROI.  Its named *lines*
  (`ROI.lines`, names from `ROIProject.line_names`), keywords, *use*,
  *comment* and *origin* are annotations beside it; the lines' angles and
  lengths and the keywords are columns of its results row
  (`core.annotation_columns`).  The ROIs a run covers
  are the **included** ones: *use* ticked.  `ROI.number`
  is given once and never reused, so ROI 5 stays 5 in the list, in a
  `roi_id` column and in the results after ROI 3 is deleted.
  An ROI is a filter more than a shape: its rows are its shape narrowed by
  `ROI.ids` (``{column: id}``), `ROI.frames` and `ROI.z` (``[low, high]``,
  either end open) -- or, with `ROI.spatial` off, its id alone, which is
  how a track is an ROI (*Segment regions*, *from an id column*, *its id
  alone*): its localizations, and not the tracks crossing its path, which a
  shape round it would take in (`ROIProject.members`, `core.narrow`; the
  id through `core.IdLookup`, one sort per column array).  A shapeless ROI
  has no `ctx.region`, as a cluster has none; the ROI manager draws it as
  its track's line or its hull (`ROIProject.outline`, `draws_track`).  The
  id and the ranges are in its signature, and a project holding such ROIs
  is saved as format 2, which an older smappy refuses rather than reading a
  track as a circle.
* **An id column** (`region_runs.ColumnRegions`; `ROIRegions` is the same
  interface over the ROIs, and `regions_of` picks one): one sort splits the table
  (`split_ids`).  Any integer column serves -- `cluster_id`, `roi_id`,
  `channel`.  An id of 0 or below (DBSCAN's noise), or NaN, is in no region
  and is never measured.

**Membership is the region alone**; each layer's filter then applies inside
it, and a region left with nothing is not measured.  ROIs may overlap, so an
ROI's rows are found from its shape through the spatial index every time
(`core.rows_in`) and never stored as a column: a column cannot hold a
localization in two ROIs, and would go stale whenever an ROI moved.  One
point-in-polygon test (`regions.inside_polygon`, the outline counted inside)
serves the drawn ROI and the ROI manager alike.

## A measurement runs anywhere

A plugin that sets `per_region = True` hands back `Result.values` (numbers,
a column each of the region table) and `Result.arrays` (named arrays), and
can then run once per region as well as on the view.  There is no separate
kind of evaluator plugin: in a region it gets **the ordinary context with
the region where the drawn ROI would be** (`ROIProject.region_context`,
`ColumnRegions.tables`).  `ctx.locs` is the region's rows of the table the
layer shows, `ctx.table(i)` and `ctx.layer_selection(i)` each layer cut to
the region with its own filter, `ctx.region` the ROI's shape (None for a
cluster), `ctx.region_id` the region's id.  No drawn ROI or 3D slab narrows
it.  A layer's grouped table is cut too -- for an ROI by its shape, for an
id column through `group_id`, so a blink whose localizations fall in two
regions is in both.  Whether a plugin measures per layer or several layers
together stays its own business, as on the view.

### Slow work is timed

Nothing in a step is switched off for running per region (Localization
Statistics' *precision fit* runs in each ROI unless set to *none*); the
block is timed instead: `BlockPlugin.estimate` runs
the steps in up to five regions spread over the source (or a second's worth),
keeps nothing, and multiplies each step's median out ("about 1 h over 100000
regions ...").  The editor's *check* shows it; a run estimated over a minute
asks first.  While it runs the status line
gives the percentage, the rate and the time left (`region_runs.Progress`).
*Stop* (`ctx.stop`) ends it between regions and keeps what was measured; a
region that has begun finishes, since half its numbers would be kept as if
they were all of them.

## Evaluation is a block of a chain

A `chain.Block` is a list of steps run once in each region: `regions` is
`rois` or the name of an id column.  In each region the steps share one
`region_runs.RegionScratch`, so a table a step hands back is what the next
one measures (filter a region, then measure it), and all of it is thrown
away afterwards.  Because of that, once one step must run in a region every
step before it that changes the table runs too, and a step's signature
includes the identities of those steps.  The per-region steps (Regions tab)
(`gui/region_steps.py`) is the chain panel opened on the ROI project's own
block (`ROIProject.block`, a `chain.Block` saved with the project);
its Run measures what is out of date and carries the rest.

**Nesting.**  A step may be a chain or a block.  A block inside a block
repeats over an id column of the outer region's table
(`region_runs.ContextRegions`): regions narrow, and a `cluster_id` an inner
DBSCAN writes exists only in that region's scratch copy.  The inner block
makes a table per outer region, kept beside the outer step's with rows named
`outer:inner`.  Nothing an inner block changes reaches the real table:
writing back inside a nested block is not supported, and a table change
meant to be kept -- the inner block's *write back*, its *write values*
unless the outer block writes those columns back, a last step that changes
the table -- is warned about in the editor (`BlockPlugin.describe`) and when
the chain runs (`blocks.nested_losses`); a filter before a measurement,
whose effect is meant to stay in the region, is not.  A chain that contains itself is refused when loaded
(`chain.load_class`), and a segmenter is refused in a block: ROIs found in
each region would have nowhere to go.

**Write-back is opt-in, by name** (`write_back`, `region_runs.WriteBack`).
The scratch table carries `_row`, the row of the real table each row came
from, and the named columns are merged by it once every region has run.  A
row in no region keeps its value (NaN, or 0 for an id or a flag, in a new
column); a row in overlapping regions takes the `merge` rule's value --
`last` (the default), `first`, `mean`, `any` -- so overlap is decided, not an
accident.  With `use` among the columns, a localization a step removed is
flagged (`use` False), not deleted: a deletion merged over overlapping
regions means nothing.
Not automatic, because a temporary column of one step would otherwise leak
into the real table: a column a step added or changed that is not written
back is warned about once, after the first region that has one, and in the
editor's check.  A block that writes back is one table change -- one undo
step, one log line; one that writes nothing changes nothing.

## What a region keeps

A `region_table.RegionTable` per evaluation, a row per region: the values in
columns, the arrays end to end with their offsets (10^5 profiles are two
datasets), the step that made each row (plugin, version, parameters), the
signature its freshness is checked by, and the error that took its place
where the step failed.  `Result.data` and the figures stay with the run that
made them.

A step whose arrays are curves on one x axis is summarised over regions by
*Summarize/Curves* (`plugins/curves.py`): the mean and spread per x, a line
per ROI keyword.  A curve that is a ratio of counts should be pooled, not
averaged; a plugin that hands back the numerator and denominator among its
arrays implements `Plugin.pool(sums, x)`, which gets every array summed over
the regions and returns the curves made from those sums (Spatial statistics:
g, K and L - r from pairs and expected pairs).

### Figures

A row of an ROI also keeps what `Plugin.keep` returns, and `Plugin.restore`
draws the figure again without measuring; a row of an id column does not --
ROIs are looked at one by one, clusters are not (to look at clusters, make
ROIs of them).  What `keep` holds is binned data, never
per-localization arrays, which would be the table again.  Off, a figure
comes back by running the step on that region again, which gives the same
figure since the row is signed with what made it.

## Where results live, and how they are read

Two stores, and both stay, because what differs is ownership:

* **`ROIProject.tables`** -- per ROI, one table per evaluation name.  An ROI
  is edited alone, so its row is checked alone: `ROIProject.evaluations(roi)`
  is what still describes it (its file, place, every layer's filters and
  grouping, and the step).  The name makes the evaluation: the same plugin
  renamed and run with other settings is a second evaluation beside the
  first; run again under its old name, it replaces the old numbers.  Rows of
  removed ROIs stay until a save, so an undo brings an ROI back with them.
* **`Session.id_regions`** -- `{column: {name: RegionTable}}`, per region of
  an id column (`Session.evaluate_regions`).  The regions change together --
  a new table, a new clustering -- so a table is current or not as a whole:
  one signature per step and table (`region_runs.column_inputs`), which is why
  checking 10^5 clusters costs one digest.

An analysis reads neither store directly: **`region_results`** is the one
interface.  An evaluation is `Evaluation(source, label)`, written "Corners"
on the ROIs and "per cluster_id: Corners" per cluster -- what *results from*
holds and a chain file saves.  `available`, `rows` (current rows only, the
region's id as `roi_id`), `steps`, `is_current` and `missing` answer the
same way for both; the summaries (`roi.SiteAnalysisPlugin`) read through
it.  It exists because every reader used to branch on the store, and the
per-cluster branch never checked freshness: three clusters relabelled as
one, and Histograms still counted three.  In the file the tables are HDF5
groups under `/regions` and `/id_regions`; the saver can leave them, or only
their figure data, out.

## Region values as columns

To filter or colour localizations by what was measured in their region --
the pores with six corners, the clusters with a good precision -- each
region's number is written onto its rows as a column
(`region_runs.write_values`): by *write values* on a block, at the end of
every run so a chain keeps it current, or by *Regions to column* for results
already there, the ROIs' own numbers (`roi_id`) among them.  NaN outside any
region or where the region has no value; where ROIs overlap, the last, the
first or the mean (*any* is for write back, and refused here).  Everything that reads a column then works on it: each layer's
filter, colour by a field, merge blinks (a blink takes the mean, the highest
of a `roi_id`), the file.

This replaced a planned region table in the session with a filter of its
own: that filter would belong to no layer, a second kind of table would need
its own UI and its own link to the localizations, and filtering through it
would be the column anyway.  The cost: a histogram of such a column counts
localizations, not regions (*Summarize/Histograms* is the distribution over
regions), and it is a snapshot until written again.  The table remembers
which columns came from regions (`metadata["region_columns"]`) and writes
only over those, never over a column of the localizations' own; those
columns are left out of both stores' data signatures, so writing them does
not put the results they came from out of date.

## Clusters

A clusterer writes **`cluster_id`** (1, 2, ... by size, 0 for none) and
**`n_in_cluster`** onto every localization, as recipes with the rule `max`
so a blink keeps its cluster (`plugins/clusters.py`, *Find clusters*: DBSCAN,
Voronoi or HDBSCAN).  Filtering
clusters by size is then an ordinary filter on `n_in_cluster`, and a block
repeats over `cluster_id`.  A cluster carries no annotations: to review one,
*Segment regions* (*from an id column*) makes an ROI of each id of any
column -- so every clusterer gets it and none makes ROIs itself -- with at least 10 localizations by
default, shaped as the cluster's convex hull, or the project's circle or
square on its centre.

## The plugin tree

```
Analysis/Measure/...                every measurement: on the view, per ROI, per cluster
Analysis/Regions/Segment regions    ROIs: a ring template, density peaks, or an id column
Analysis/Regions/Find clusters      DBSCAN, Voronoi or HDBSCAN (cluster_id)
Analysis/Regions/Summarize/...      Histograms, NPC labelling efficiency: read a region table
Analysis/Regions/Regions to column  what was measured per region onto the table
Analysis/Regions/Average ROIs       the used ROIs centred, turned and superimposed: a new file
Analysis/Regions/Workflow/...       NPC Analysis, a shipped chain with a block
```

A finder and a clusterer are one kind of thing, *find regions*, differing in
what they write.  *Regions*, not *Groups*, which is the blink linking
(`group_id`).  The Regions tab seeds from `Analysis/Regions/`, which the
Analysis tab leaves out.
