# Regions: ROIs, clusters and what is measured in them

Decisions taken 2026-10-11, in a discussion that started from one question:
SMAP measured the same things three times -- in an Analyze plugin over the
view, in an ROI manager evaluator per site, and in cluster code per cluster --
and SMAPpy was on its way to doing the same.  This note replaces the section
"The ROI evaluation pipeline is a separate window" of
`docs/plugin-architecture.md` and the section "ROIs: segment, evaluate,
analyse" of `docs/batch.md`.  Nothing here is built yet; "Order of work" at
the end says in what steps it will be.

The idea in one paragraph: a **region** is a set of localizations with an id
-- an ROI drawn or found in the ROI manager, or a cluster named by a column
such as `cluster_id`.  Every measurement is an ordinary plugin, and runs the
same on the view, on one region, or on every region in turn.  Running a list
of plugins on every region is a **block** of a chain.  What the block measures
is a **region table**, one row per region, which is a table like any other.
The ROI manager is what it should be: a tool to find, review, annotate and
organise ROIs, not the only place where they can be measured.

## What is there now, and what is duplicated

* **Two region types and two copies of their geometry.**  `regions.Region` is
  the ROI drawn in the session (rectangle, polygon, line); it goes into
  `ctx.selection`, and `line_profile.py` reads it.  `roi_manager.core.ROI` is
  the ROI manager's: a centre, a polygon or the project's shared shape, a
  direction, and the annotations (`reviewed`, `use`, `comment`, `origin`).
  `roi_manager.core.inside_polygon` and `regions._in_polygon` are the same
  test written twice.
* **An evaluator is an ordinary plugin with a flag.**  `scope = "site"`; the
  driver (`ROIProject._one_step`) hands it `Context(locs=<the ROI's locs>,
  site=geometry)` and stores `Result.data` as the ROI's row.  What keeps it
  apart from the plugins of the Analysis tab is where the region comes from
  (`ctx.site` against `session.roi`), that a row has to be flat numbers, and
  the chooser's filter on `scope`.  So `ROIManager/Evaluate/Statistics` and
  `Analysis/Measure/Localization Statistics` are two plugins.
* **A third filter path.**  `SessionROIs` follows one layer (`self.layer`),
  copies that layer's filter into a `ViewState` of its own per file, and the
  context it hands an evaluator has no session -- so a plugin that measures two
  layers together cannot run per ROI at all.
* **The ROI analyses read a table that is not a table.**  `SiteAnalysisPlugin`
  reads `ROIProject.results()`, a list of dicts; `Histograms` is a histogram of
  columns rebuilt for that one kind of table.
* **Two list editors.**  `gui/chain_panel.py` edits a chain's steps (add, move,
  rename, remove, tick, save); `gui/evaluation.py` edits the evaluation
  pipeline with the same operations, filtered to evaluators.
* **No clusters yet.**  The ids on the table are `group_id` (blinks), `tid`,
  `channel` and `filenumber`; nothing writes a `cluster_id`.  Deciding now
  costs nothing that a second set of per-cluster plugins would.

## Decisions

### A region is an id and a set of localizations

Two **region sources** give the same thing -- `(id, rows, region or None,
annotations)` for each region:

* **ROIs** (`ROIGroups(project)`): the reviewed, included ROIs of the ROI
  manager, their rows found from the geometry through the spatial index.
  ROIs may overlap, so membership is computed, never stored as a column -- a
  column cannot hold a localization that is in two ROIs, and would go stale
  whenever an ROI moves.
* **An id column** (`ColumnGroups(locs, "cluster_id")`): one sort splits the
  table into regions.  Any integer column serves -- `cluster_id`, `roi_id`,
  `group_id`, `channel`.  Id 0 is "in no region" (DBSCAN's noise) and is never
  evaluated.

**Membership comes from the region alone**; each layer's own filter then
applies inside it.  A region with no localizations left after the filter is
not evaluated.

### One region type

`Region` grows what the ROI manager needs -- a circle, a fixed-size shape
around a centre, a line with a direction -- and an ROI manager entry becomes a
`Region` plus its file and annotations.  One point-in-polygon test, one
`ctx.region` for a plugin to read whether the region was drawn, found, or is
None (a cluster).

### A measurement runs anywhere: no evaluators

`scope = "site"` goes, and with it the distinction between evaluation and
analysis plugins.  Running on a region means **the normal context with that
region where the drawn ROI goes**, in every layer: `ctx.selection` is already
the layer's filter and the ROI and the slab, so `ctx.layer_selection(i)` is
layer *i* inside the region.  Whether a plugin measures per layer, several
layers together or the current one stays the plugin's business, exactly as on
the whole view (`line_profile._groups` is the shape).  The ROI manager's own
`ViewState` copies go.

A plugin can be a step in a block if what it returns can be stored per region
(see "What a region keeps"); NPC Corners, the line profile and the statistics
become one plugin each, usable on the view, per ROI and per cluster.

Slow, optional parts of a measurement (a bootstrap, a figure that needs a
fit) are **settings**, so a step in a block, which has its own copy of the
plugin's settings, turns them off without any new mechanism.  A plugin does
not need to know it is running per region, though it may (`ctx.region`).

### Evaluation is a block of a chain

A chain step can be a **block**: a list of steps run once per region of a
region source, each region in a scratch copy narrowed to it.

```
Density Peaks                       once: finds ROIs
for each ROI:                       a block
    Remove outliers                 may change the region's table
    Localization Statistics         sees what the step before did
    NPC Corners                     a row per ROI
NPC Labeling Efficiency             once, on the region table
```

Against the present loop (each evaluator step runs over every ROI on its own)
the block lets a step see what the step before did *to that region* -- filter
it, then measure it -- which the loop cannot.  The row of a region is its
steps' values side by side; a column is qualified with the step's name when
two steps write it, as now.  Results are signed per step and region as now, so
editing one step re-runs that step and not the block.

**Nesting.**  A saved chain is a plugin, so a chain can be a step in a block,
and a block can be a step in a block.  The rules that keep it well defined:

* **Regions narrow.**  A block in a block runs on the outer region and the
  inner one.  *For each ROI -> DBSCAN -> for each cluster -> measure* is
  clusters within each ROI, and the `cluster_id` the inner DBSCAN writes
  exists only in that ROI's scratch copy.
* **An inner block makes a table, not a row**: one per outer region, keyed by
  (outer id, inner id), kept as the outer region's child results.  Later steps
  in the outer block may summarise it -- clusters per ROI, mean cluster size --
  which is what makes nesting worth anything.
* **No cycles.**  A chain that contains itself, directly or through another
  chain, is refused when it is loaded.  `chain.py` has no such check today,
  and needs one as soon as a chain is a step.

### What a block does to the localization table: opt-in write-back

Everything a step does inside a block is scratch: a table it changes, ROIs it
makes, are thrown away when the region is done.  What comes back is what the
block says comes back:

* **The region's scratch table carries the original row** (a hidden `_row`
  column), so whatever a step does, a surviving localization still knows
  where it came from.
* **A block writes columns back by name** -- a setting of the block, *write
  back: `corner`, `r_nm`, `use`*.  When the block is done they are merged into
  the real table by `_row`.  A row in no region keeps its value (NaN, or 0 for
  an id or a flag, in a new column).  A row in two overlapping regions takes
  the **later region's** value by default; the rule is a setting of the block
  (last, first, mean, any) so overlap is decided rather than an accident, and
  never excluded.  An existing column (`use`, a per-region `z_nm`) is written
  back the same way.
* **A localization a step removed is a flag, not a deletion**: `use = False`
  on the rows the region dropped, as `remove_locs.py` flags.  A deletion
  merged over overlapping regions has no meaning.
* **New localizations made in a block are not written back.**  Their numbers
  may reach the region table; a step that makes localizations belongs after
  the block.
* **Opt-in, with a warning.**  Columns a step adds or changes that are not
  written back are reported: when the block runs (the scratch table's columns
  against the original's, after the first region), and in the editor's check,
  which runs the block on the current region.  `changes_locs` says only *that*
  a plugin changes the table; which columns is found by running it.  Not
  automatic, because a temporary column of one step would otherwise leak into
  the real table.
* **One table change**: one undo step and one line in the session's log,
  naming the columns and the rule.  A block that writes nothing back changes
  nothing.

The same merge, outside a chain, is *Regions/ROI to column*: `roi_id` written
onto the table, the later ROI winning where they overlap, 0 in none.  It is a
snapshot -- moving an ROI afterwards does not change it -- and logged.  After
it, "per ROI" and "per cluster on `roi_id`" are the same run, without the
overlaps and the annotations.

### What a region keeps

A plugin declares what it stores per region by shape -- `Result.values`
(scalars) and `Result.arrays` (named arrays) -- and the runner stores it in
columns, so 10^5 clusters cost what one array of 10^5 does:

| per region | stored as |
| --- | --- |
| a number | a column of the region table |
| a fixed-size array (eight corner counts, a profile on fixed bins) | an (n_regions, k) column |
| a variable-length array (a profile over the region's own extent) | ragged: the values end to end, and offsets |
| a value per localization | not a region result: a write-back column |
| anything else (a dict of fits, an object) | JSON per region, optional in the saver, off by default past a few thousand regions |
| figures | never: drawn again on request |

What is not declared is there in the session for the region being looked at,
and not saved.  Nothing is lost by not saving it: a stored row is signed with
its inputs and step, so running the step on that region again gives back the
structure and the figure exactly.  Most of what SMAP's evaluators kept per
site is arrays -- profiles, corner counts, fit parameters -- and falls in the
first three rows; this should be checked against SMAP's `ROIManager/Evaluate`
plugins before the storage is written.

### The region table is a table

**In the file**: one columnar table per evaluation (per block and region
source), in HDF5 beside the localizations, the steps' provenance -- plugin,
version, settings -- written once, a signature column per row so that one
moved ROI reads as out of date and the rest do not.  Saved whenever a run went
over a list of regions; the saver offers to leave region results out, for the
large cluster sets.  This replaces the JSON records (`runs -> records -> roi ->
steps`) the ROI project writes now.

**In the session**: a table of its own with its own filter, as the grouped
table is (`state.sets["grouped"]`).  So the plugins that summarise regions are
ordinary table plugins -- a histogram of any column replaces `Histograms`;
NPC Labeling Efficiency reads `n_corners` -- *results from* keeps its job of
choosing which evaluation's columns, and the filter UI filters regions
(`n_corners >= 6`) as it filters localizations.  For clusters that filter can
reach the localizations through `cluster_id`; for ROIs it is an option, "only
the localizations in the regions that pass".

### Clusters

A clusterer always writes **`cluster_id`** and **`n_in_cluster`** (the
localizations in the cluster, the analogue of `n_in_group`) onto the
localization table.  Filtering clusters by size is then an ordinary filter on
`n_in_cluster`.  A cluster carries no annotations of its own: to review or
comment on clusters, a *Regions/Find* plugin makes ROIs of them (a polygon
around each, `origin` the clusterer).  Opening the ROI manager stays optional,
and past some number of ROIs it lists them without drawing their boxes on the
picture.

### The plugin tree

Paths change, and nothing old is kept for compatibility: SMAPpy has one user.

```
Analysis/Measure/...            every measurement: on the view, per ROI, per cluster
Analysis/Regions/Find/...       makes regions: Density Peaks and NPC (ROIs),
                                DBSCAN (cluster_id, ROIs, or both)
Analysis/Regions/Summarize/...  reads a region table
Analysis/Regions/ROI to column  roi_id onto the table
```

A finder and a clusterer are one kind of thing, *find regions*, differing in
what they write.  *Regions*, not *Groups*, which is the blink linking
(`group_id`).

### One step-list editor

The chain panel's list and the evaluation window become one editor, in which
a block is a step with a list of its own (and any step may be a chain).  The
evaluation window is that editor opened on a block, renamed to say so, with
the region source -- ROIs, `cluster_id`, any id column -- chosen at its top.
The ROI manager keeps a button that opens it on its block.  The ROI tab
becomes *Regions*, to match the tree.

## How fast the runner is, and what is slow

Measured 2026-10-11 on `ROIProject.evaluate` as it stands: one file, 200 nm
ROIs, 200 localizations per site, the ROI `Statistics` evaluator.

| sites | evaluate | per site | again, current results reused | `results()` |
| --- | --- | --- | --- | --- |
| 1 000 | 0.94 s | 0.9 ms | 0.14 s | 0.08 s |
| 10 000 | 6.1 s | 0.6 ms | 1.5 s | 0.7 s |

The evaluator's own work is about 0.1 ms a site; the rest is bookkeeping --
signing each site's inputs and step for the staleness check (about eleven JSON
dumps and a SHA-256 a site) and extracting its localizations.  2000 stored
one-site runs, which walking through the list leaves behind, make `results()`
slower but not quadratic.  So for ROIs the overhead is fine; for 10^5 clusters
it would be a minute of bookkeeping, which an id column does not need:
membership is one sort, and freshness is one signature per (table, step), not
per region.  What is slow per region is a plugin's own work -- a bootstrap, a
fit, a link -- and that is a setting (above).  Figures cost nothing until they
are drawn, and a run over regions keeps rows, never `Result`s and their
closures.

## Order of work

Each step leaves the program working and the suite green.

1. **One region type.**  `Region` takes the ROI manager's shapes; `ROI` holds
   one; one point-in-polygon test.  Nothing a user sees changes.
2. **The context.**  `ctx.region`; a per-region context that is the session's
   with the region in place of the drawn ROI; `SessionROIs`' own filter copies
   go.  `Result.values` and `Result.arrays`.  `scope` goes, the two statistics
   plugins become one, the chooser offers every plugin that can be a step.
3. **Region sources and one runner**, ROIs first, then id columns, with the
   region table in columns, in the file and as a session table with its filter.
4. **Blocks** in chains, the nesting rules and the cycle check, write-back and
   its warning; one step-list editor, the evaluation window becoming it.
5. **Clusters**: DBSCAN under *Regions/Find* writing `cluster_id` and
   `n_in_cluster`, clusters to ROIs, *ROI to column*.
6. **The tree** moved to `Analysis/Measure` and `Analysis/Regions`, the tab
   renamed, the summarising plugins turned into table plugins, the shipped
   `npc_analysis.chain.yaml` rewritten as a block.

Every step that moves a plugin or changes what it computes brings its page
along (`CLAUDE.md`, "The plugin's page"), and the exhaustive plugin-list tests
with it.

## Not decided

* Whether a cluster should ever carry annotations without becoming an ROI.
  For now, no: filter by `n_in_cluster`, or make ROIs.
* The threshold past which the saver leaves JSON region results out by
  default, and past which the ROI manager stops drawing boxes: to be set by
  measuring, not guessed.
