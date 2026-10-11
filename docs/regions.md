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
around a centre -- and an ROI manager entry gives its shape as a `Region`
(`ROI.region(shape, size_nm)`, `ROIProject.region(roi)`).  It works it out
rather than holding it: the circle or square around the centre is the
project's, and changing it is one edit for every ROI.  An ROI's *direction*
stays an annotation beside the shape -- an arrow inside it, not its outline.
One point-in-polygon test (`regions.inside_polygon`, the outline counted as
inside), one outline and one rim distance for the ROI manager to draw and
pick by, and one `ctx.region` for a plugin to read whether the region was
drawn, found, or is None (a cluster).

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
The plugin **declares** them -- `param(..., slow=True)`, presentation like
`advanced=True` -- and a step added to a block **starts with them off**; the
editor marks them, and its check warns when one is on in a block over many
regions (with an estimate, once the step has run on the current region).
With that default no "fast options" button is needed: ticking one back on is
a choice.  Localization Statistics' fits are the example: about 10 ms a
region whatever its size, 17 minutes over 10^5 clusters.

**Necessary slow work is timed, not declared.**  A step that is slow because
the measurement *is* slow (the fit is the result) has nothing to switch off
and gets no flag.  Instead the block is timed:

* **Before a run.**  The editor shows each step's estimated time and the
  block's total, and a *Check* button brings it up to date: it runs the block
  on the current region (as the write-back check does anyway), times each
  step, and multiplies by the regions the source holds -- "NPC fit: ~40 ms a
  region, ~1 h over 10^5 clusters".  One region is a noisy sample, so the
  check times a few (up to five, or a second's worth) and reports the median.
  The block's preflight asks with the same estimate when it is long, the way
  COMET asks before an afternoon, and offers "run with slow options off" as a
  choice when the block has any.
* **During a run.**  A status line with the percentage of regions done, the
  regions per second and the time left, from the regions so far; the step
  being run when a block has several.  It goes through `ctx.report`, so the
  plugin panel, the chain panel, the batch window and a script all see it.
  Today's runner already reports "i of n sites"; this adds the rate and the
  estimate.  A run can be stopped (`ctx.stop`) and keeps the regions it has.

Timing covers every plugin, including one nobody marked, and measures the
cost on the user's own data; `slow=True` remains only for what timing cannot
tell -- which part of a step could be dropped.

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

Write-back carries columns the steps made per localization.  What the steps
measured per region goes onto the table by **write values** (see "Region
values as columns" below).  The ROIs' own numbers are one such value: *Regions
to column* writes `roi_id`, the later ROI winning where they overlap, 0 in
none, a snapshot, logged.  After it, "per ROI" and "per cluster on `roi_id`"
are the same run, without the overlaps and the annotations.

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
| the data behind its figures | optional, a switch per step (*Figures*, below) |
| the figures themselves | only as files, when a step asks (*Figures*, below) |

What is not declared is there in the session for the region being looked at,
and not saved.  Nothing is lost by not saving it: a stored row is signed with
its inputs and step, so running the step on that region again gives back the
structure and the figure exactly.  Most of what SMAP's evaluators kept per
site is arrays -- profiles, corner counts, fit parameters -- and falls in the
first three rows; this should be checked against SMAP's `ROIManager/Evaluate`
plugins before the storage is written.

#### Figures

A per-region figure comes back in one of three ways, and the step chooses
which it pays for.  The mechanism is the one a drift correction's curve
already uses to outlive its session: `Plugin.keep(result)` returns the plain
data behind the figure, `Plugin.restore(saved)` draws it again
(`plugins/__init__.py`).  It is used per region rather than a second
mechanism invented beside it.

1. **Numbers**, `Result.values` and `Result.arrays`: always stored -- the
   region table.
2. **Figure data**, what `keep` returns: a switch on each step of a block,
   *keep figure data*.  On, any region's figure is back at once through
   `restore`.  Off, it is back by running the step on that region again,
   which gives the same figure, since the row is signed with what made it --
   milliseconds for most plugins, so the switch matters only for the slow
   ones.  A plugin with no `keep` has the switch greyed out and is always
   re-run.
3. **Figure files**: a step setting, *save figures to folder*, written as the
   run goes -- for a plugin slow enough to run overnight, where re-running to
   look is not an option.  The batch runner already writes each plugin's
   figures per file; this is the same per region.

The default of *keep figure data* follows the **region source, not the
plugin**: on for ROIs (hundreds, looked at one by one), off for an id column
(to look at clusters, make ROIs of them).  The same Statistics plugin is cheap
on 300 ROIs and too much on 10^5 clusters, so the plugin cannot decide it.
What the plugin does decide is what `keep` holds, and the rule is **binned
data, never per-localization arrays**: a histogram of 100 bins is 800 bytes
a region, 80 MB for 10^5; the photons themselves would be the table again.

It follows that a plugin's figures are drawn from the data it keeps, not from
closures over the table -- `keep` is "the arrays the plot function takes",
`restore` "call the plot function on them".  That is the house rule for
algorithms already; the ROI `Statistics` and NPC Corners draw from closures
today and change.

**Saving.**  Kept figure data is saved with the region table by default.  A
save past a threshold (to be measured; of the order of 100 MB) says so in the
save dialog, with the size, and offers to leave the figure data out; figure
data present and not saved is one line in the dialog -- "figure data of N
regions not saved: their figures will be redrawn by running the step again"
-- not a warning to dismiss, since nothing is lost but time.  The one real
loss is a slow plugin's figure data left out, which is what figure files are
for.

### The region table is a table

**In the file**: one columnar table per evaluation (per block and region
source), in HDF5 beside the localizations, the steps' provenance -- plugin,
version, settings -- written once, a signature column per row so that one
moved ROI reads as out of date and the rest do not.  Saved whenever a run went
over a list of regions; the saver offers to leave region results out, for the
large cluster sets.  This replaces the JSON records (`runs -> records -> roi ->
steps`) the ROI project writes now.

**In the session**: not a table with a filter of its own -- that was the
plan, and was dropped (see the next section).  The summaries read a region
table through *results from*, which names an evaluation on the ROIs or one
of an id column's regions.

### Region values as columns

To filter or colour localizations by what was measured in their region --
the pores with six corners, the clusters with a good precision -- the
region's number is written onto its localizations as a column: *write
values* on a block (at the end of every run, so a chain keeps it current), or
*Regions/Regions to column* for results already there
(`region_runs.values_onto`).  Everything that reads a column then works on
it: each layer's filter, *colour by*, grouping (a blink takes the mean), the
file.

Why not the region table as a session table with its filter, as first
planned: a filter on regions would belong to no layer (membership is the
region alone), a second kind of table would need its own UI and its own
link to the localizations, and filtering localizations through it would be
the column anyway.  What the column costs: its histogram counts
localizations, not regions (a large cluster weighs more) -- the distribution
over regions is *Summarize/Histograms* -- and it is a snapshot until written
again.  A localization in no region, or in one without a value, is NaN; one
in overlapping ROIs takes the block's *overlaps* rule.  The table remembers
which columns came from regions (`metadata["region_columns"]`) and writes
only over those, never over a column of the localizations' own.

### Clusters

A clusterer always writes **`cluster_id`** and **`n_in_cluster`** (the
localizations in the cluster, the analogue of `n_in_group`) onto the
localization table.  Filtering clusters by size is then an ordinary filter on
`n_in_cluster`.  A cluster carries no annotations of its own: to review or
comment on clusters, *Regions/Column to ROIs* makes ROIs of them -- of any id
column, so every clusterer gets it and none makes ROIs itself (a convex hull
around each, or the project's ROI shape on its centre; `origin` the column
and the id).  Opening the ROI manager stays optional,
and past some number of ROIs it lists them without drawing their boxes on the
picture.

### The plugin tree

Paths change, and nothing old is kept for compatibility: SMAPpy has one user.

```
Analysis/Measure/...            every measurement: on the view, per ROI, per cluster
Analysis/Regions/Find/...       makes regions: Density Peaks and NPC (ROIs),
                                DBSCAN (cluster_id)
Analysis/Regions/Summarize/...  reads a region table
Analysis/Regions/Column to ROIs     an ROI per id of a column (a cluster)
Analysis/Regions/Regions to column  what was measured per region (or roi_id) onto the table
```

A finder and a clusterer are one kind of thing, *find regions*, differing in
what they write.  *Regions*, not *Groups*, which is the blink linking
(`group_id`).

### One step-list editor

The chain panel's list and the evaluation window become one editor, in which
a block is a step with a list of its own (and any step may be a chain).  The
evaluation window is that editor opened on a block, renamed to say so, with
the region source -- ROIs, `cluster_id`, any id column -- chosen at its top.
The ROI manager keeps a button that opens it on its block.  The Regions tab
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

1. **One region type.**  `Region` takes the ROI manager's shapes; `ROI` gives
   one; one point-in-polygon test.  Nothing a user sees changes.  *Done
   2026-10-11*; the one difference is that a localization exactly on the edge
   of a drawn polygon or line now counts as inside, as it already did for a
   rect and for the ROI manager.
2. **The context.**  `ctx.region`; a per-region context that is the session's
   with the region in place of the drawn ROI.  `Result.values` and
   `Result.arrays`.  `scope` goes (a flag, `per_region`, says a plugin returns
   values), the two statistics plugins become one, the chooser offers every
   plugin that can be a step.  In three parts:
   * *2a, done 2026-10-11*: `ROIProject.region_context(roi)`.  `ctx.locs` is
     the ROI's rows of its file, unfiltered and ungrouped; each layer's
     selection is its filter over them; the grouped table is offered cut the
     same way where the layer has a current one; neither the drawn ROI nor
     the slab narrows it.  It reuses the shape of the live fit's pinned
     tables, so `ctx.table(i)` and `ctx.layer_selection(i)` work unchanged.
     `ctx.site` is gone; an ROI's centre is `ctx.region.center` (the
     `Region`'s `anchor`).  A stored ROI result is signed with every layer's
     filters and grouping, so results saved before read as out of date once.
     `SessionROIs`' per-file states stay for now: they draw the manager's
     images and hold its index, and decide nothing a measurement sees.
   * *2b, done 2026-10-11*: `Result.values` / `Result.arrays`, and
     `per_region = True` in place of `scope = "site"`.  The runner stores only
     those (`core.region_entry`: a value that is not a number is refused, an
     array goes in `arrays`, kept as a JSON list until step 3); `data` and the
     figures stay with the run.  A tab's chooser offers every plugin, the
     evaluation window the `per_region` ones.  A chain step of such a plugin
     says where it runs, `regions: view` (the default) or `rois` -- the
     step's own choice until blocks replace it; the shipped NPC chain says
     `rois`.
   * *2c, done 2026-10-11*: one statistics plugin.  Localization Statistics
     is `per_region`, returns its numbers as values (`statistics_values`),
     gives an empty site a row of zero and NaNs, never takes a site's
     on-time from the session's whole grouped table, and has *fit the
     precision model* (`slow=True`: the fit is 10 of its 11 ms).  The ROI
     `Statistics` is gone.  `param(..., slow=True)` and `fast_values` exist;
     the evaluation pipeline adds a plugin with its slow switches off
     (`pipeline.instance_for`).
3. **Region sources and one runner**, ROIs first, then id columns, with the
   region table in columns, in the file and as a session table with its filter;
   `keep` and `restore` per region, and the switch.
   * *3a, done*: `region_table.RegionTable`, a table per evaluation, rows set
     whole, through plain data and HDF5 (10^5 rows of two numbers: 2.7 MB).
   * *3b, done*: the ROI project keeps its results in `tables`, one per
     evaluation name, the newest result of each ROI with the step that made
     it; `runs` is a log without records.  One thing is lost: a result
     replaced under the same name is gone, so editing a parameter back no
     longer finds the old numbers current -- it measures again.  An older
     file's records are read into tables (`absorb_run`).
   * *3c, done*: region tables as HDF5 groups under `/regions` in the
     localization file, not in the metadata attribute (read back into the
     ROI project's JSON as columns); *save the ROI results* on the saver
     leaves them out.
   * *3d, done*: `region_runs`: `ColumnRegions` (an id column's regions,
     one sort; each layer cut to them, the grouped table through
     `group_id`), `evaluate_regions` (one signature per step and table),
     `Session.evaluate_regions` and `Session.id_regions`, saved under
     `/id_regions`.  `ctx.region_id` and `ctx.per_region` say a run is one
     region of many, a cluster included, which has no `ctx.region`.  The ROI
     runner and a chain's evaluator step report percentage, rate and time
     left and stop when asked; the evaluation window's *stop* button now
     works.  Measured on 10^5 clusters of 20 localizations: 0.3 s to split,
     0.1 ms a cluster of runner, 1.3 ms a cluster for Localization
     Statistics with its fit off.  No GUI runs a column's regions yet: that
     is the block (step 4).
   * *3e, done*: a region keeps what `Plugin.keep` returns (`kept`, beside
     its values; in HDF5 a JSON string per row) when its step's `keep` is
     on -- on for ROIs, off for an id column's regions unless
     `keep_figures`; a forced figure of a current ROI comes back through
     `Plugin.restore` without measuring.  Localization Statistics keeps its
     histograms (equal bins as their range) and fitted numbers, ~2 kB, and
     draws its curves again from the numbers (`model_curve`).  The saver's
     *save the ROI figures' data* leaves it out and says so.  The switch per
     step is in the model (`Step.keep`); its checkbox comes with the editor
     (step 4), and so does the size warning, which needs the session the
     form's `notice` is not given.

   The region table as a session table with its own filter, planned here,
   was dropped for region values as columns (step 7).
4. **Blocks** in chains, the nesting rules and the cycle check, write-back and
   its warning; one step-list editor, the evaluation window becoming it.
   * *4a, done*: `chain.Block` (its file form in `docs/batch.md`) and
     `blocks.block_class`, a `BlockPlugin` built on `ChainPlugin`, so
     preflight, choices, hints and `react` reach a block's steps as a
     chain's.  The steps of a region run in one `region_runs.RegionScratch`
     (a table handed on; `_row` ties rows to the real table); once one step
     must run, every step before it that changes the table runs too, and a
     step's signature includes the identities of those before it.
     `WriteBack` merges named columns by `_row` with the rule, `use` also
     flagging dropped rows; a changed column not written back is warned
     about once, the warning reaching the chain's caller.  A block in a
     block repeats over an id column of the region's table
     (`ContextRegions`); its tables are kept beside the outer's, rows named
     ``outer:inner``, and survive a save with their outer ROI.  A segmenter
     in a block is refused.  `chain.load_class` refuses a chain that
     contains itself, naming the loop.  The interim `regions: rois` step
     field is gone; the shipped NPC chain is a block (`each pore`).
     Per-cluster results come back from a chain's scratch session as
     `data["id_regions"]`.
   * *4b, done*: `BlockPlugin.estimate` runs the steps in up to five
     regions spread over the source (or a second's worth), nothing kept,
     and multiplies the median per step out (`Estimate.text`: "about 1 h
     over 100000 regions (NPC fit 40 ms ... a region, timed on 5)").  A
     block estimated at over a minute asks first, offering *Slow options
     off* (`without_slow`); a chain keeps a step's choices in its own
     question now, rather than only its text.  The *Check* button that
     shows the estimate is the editor's (4c).
   * *4c, done*: one editor.  The chain panel's steps are a tree (a block's
     steps under it; *block...*, add into the selected block, *check* with
     the estimate and the unwritten columns).  The evaluation window is gone:
     *Region steps* (`gui/region_steps.py`) is the chain panel opened on the
     ROI project's block -- its steps `ROIProject.pipeline` (a nested block
     an instance of `Chain/Block`), its own fields `ROIProject.block_options`,
     both saved with the project; `pipeline.block_of` / `from_block`
     convert.  Run measures what is out of date and carries the rest; there
     is no separate "run everything" any more.  The Regions tab, the ROI manager
     and the Tools menu open it as *Region steps...*; the Regions tab itself is
     renamed with step 6.
5. **Clusters**: DBSCAN under *Regions/Find* writing `cluster_id` and
   `n_in_cluster`, clusters to ROIs, *ROI to column*.
   *Done*: `plugins/clusters.py`.  DBSCAN (SMAP's definition of core and
   border; a border point joins its nearest core point; clusters numbered
   by size; each file alone; on a grouped layer the blinks, mapped back
   through `group_id`) writes `cluster_id` and `n_in_cluster` as recipes
   with the rule `max`, and with *output* ROIs makes a convex-hull ROI of
   each cluster (*replace*, *ROIs from*), refused inside a block.  Its
   radius defaults to 50 nm: SMAP's automatic one is the density of the
   whole field and merges everything in a sparse cell.  *ROI to column*
   writes `roi_id` (the ROI's number; the later ROI where two overlap; 0 in
   none; ROIs ticked *use*), membership the shape alone.  Both are under
   `Analysis/Regions/`, where step 6 moves the rest.
6. **The tree** moved to `Analysis/Measure` and `Analysis/Regions`, the tab
   renamed, the summarising plugins turned into table plugins, the shipped
   `npc_analysis.chain.yaml` rewritten as a block.
   *Done*: `Analysis/Regions/Find/{Density Peaks, NPC, DBSCAN}`,
   `Analysis/Regions/ROI to column`, `Analysis/Regions/Summarize/{Histograms,
   NPC Labeling Efficiency}`, `Analysis/Regions/Workflow/NPC Analysis`,
   `Analysis/Measure/NPC Corners`; their pages moved with them.  The ROI
   tab is the *Regions* tab, seeded from `Analysis/Regions/`, which the
   Analysis tab leaves out (`exclude` in `DEFAULT_TABS`); its page is
   *Regions tab*.  The summaries read a table of an id column's regions as
   well as the ROIs' (*results from*: "per cluster_id: ..."); NPC Labeling
   Efficiency, which needs the ROIs' settings and truth, stays with the
   ROIs.  Panel pages' `covers` are now checked, as plugin pages' are.

7. **Region values as columns**, in place of a session region table with
   its own filter.
   *Done*: `region_runs.values_onto` (and `merge_pieces`, the merge
   `WriteBack` uses too), *write values* on a block (`Block.write_values`,
   in its file form and the project's `block_options`), and *Regions to
   column* (`plugins/region_column.py`), which replaces *ROI to column*:
   any numeric value of an evaluation on the ROIs or of an id column's
   regions, or the ROIs' numbers, written onto the localizations, NaN
   outside, with a *prefix* and the overlap rule.  ROIs of clusters moved
   out of DBSCAN into *Column to ROIs*, which takes any id column, a
   minimum size (10 localizations) and the hull or the project's shape.

Every step that moves a plugin or changes what it computes brings its page
along (`CLAUDE.md`, "The plugin's page"), and the exhaustive plugin-list tests
with it.

## Not decided

* Whether a cluster should ever carry annotations without becoming an ROI.
  For now, no: filter by `n_in_cluster`, or make ROIs.
* The thresholds past which the saver leaves JSON region results out by
  default, past which it offers to leave figure data out, and past which the
  ROI manager stops drawing boxes: to be set by measuring, not guessed.
