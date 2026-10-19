# Region results as the standard, and a filter on them

A design note, 2026-10, not yet built.  It starts from MINFLUX (a value per
trace -- its number of localizations, its scatter, its mean `cfr` -- and
keeping the traces whose values are good), but the question is general:
whatever is measured per region -- an ROI, a cluster, a trace, a track, a
bead -- should be kept in one kind of table, filtered in one place, and the
answer passed to every consumer of the localizations without a column per
measured quantity.  `docs/regions.md` is the design this extends; the
changes it needs there are listed at the end.

## Where things stand

* **What is measured per region is already a region table**
  (`region_table.RegionTable`): per ROI in `ROIProject.tables`, per region of
  an id column in `Session.id_regions`, both read through `region_results`.
  A block of a chain fills them by running a `per_region` plugin once in
  each region.
* **Filtering on a region's value goes through a column**: *Regions to
  column* or a block's *write values* copies the value onto the region's
  localizations, and a layer bound on that column filters.  `regions.md`
  ("Region values as columns") chose this over a region table with a filter
  of its own, and names the costs: a histogram of such a column counts
  localizations, not regions, and the column is a snapshot until written
  again.  A third cost is the table's width: every value one wants to
  filter on is a column of every localization, 4 bytes each, 230 MB a
  column on a 57 M table.
* **Some plugins write per-id values onto the localizations themselves**:
  `n_in_cluster` (*Find clusters*), `track_length` (*Link tracks*),
  `track_D_um2_per_s` and `track_alpha` (*Tracking analysis*, *write per
  track*).
* **ROIs are filtered by their results** in *Filter ROIs*
  (`roi_order.py`): a parsed condition sets each ROI's *use*.  There is no
  such thing for the regions of an id column.
* **Beads have their own per-region structure** (`beads.BeadSet`): metrics
  per bead, an automatic rule (`MadRule`), choices by hand by name that are
  never overruled, and a window (`gui/bead_window.py`).

## The decision

**Region tables are the standard store for anything measured per region.**
Statistics per trace, per track, per cluster, per bead and per ROI go there,
whoever computes them -- a block, or a plugin that computes every region at
once.  **One region filter**, a section of the Regions tab, keeps or drops
regions by their values; the answer reaches the localizations as **one
column**, `region_filter`, which every consumer honours through the layers'
selection, as it honours `use`.

What stays a column of the localizations: what is not a region analysis
(`n_in_group`, the grouping's own), the ids themselves (`cluster_id`,
`tid`, `track_id`), and what someone writes there on purpose (*Math parser*
with *to localization table*, *Regions to column*) -- to colour by a value,
to compute with it per localization, or to compare two filters in two
layers.

## The filter

### What it holds

A list of **conditions**, each a path to one value and a range (or, for
text such as a keyword, a set of accepted values):

    ROIs › NPC Corners › n_corners          6 … ∞        412 / 530
    per tid › MINFLUX traces › cfr › mean   0 … 0.8    41 210 / 52 004
    ROIs › annotations › use                  true        498 / 530

The list is what is filtered, always visible, each row with an × that
removes it and the number of regions it keeps.  The filter is session
state, saved with the session like a layer's bounds; a plugin
(`Analysis/Regions/Filter regions`) sets the same state, so a chain and a
batch can reproduce it.

### How conditions combine

* **Conditions on one source are judged per region**: a region passes if it
  passes all of them.
* **Within a source, a localization is in if it is in any passing region**
  (OR).  Two NPC ROIs overlap; ROI 5 has six corners and ROI 6 three.  A
  localization in the overlap belongs to a good pore, so it stays.  AND
  would cut good pores wherever a bad neighbour overlaps them.
* **Judging per region first matters where regions overlap.**  ROI 5 has
  the corners but not the diameter, ROI 6 the diameter but not the corners.
  A localization in their overlap is out -- neither pore passes.  Combining
  condition by condition (in some ROI with enough corners, and in some ROI
  with the right diameter) would keep it.
* **Different sources combine by AND**: with a condition on the ROIs and one
  on `cluster_id`, a localization must be in a passing ROI and in a passing
  cluster.  One source is the usual case.
* **A localization in no region of a filtered source is out**: DBSCAN's
  noise (`cluster_id` 0) once clusters are filtered, a localization outside
  every ROI once the ROIs are.
* **The layers' own filters still apply**: the flag is one more condition,
  AND-ed with each layer's ranges.  A localization a layer's `cfr` bound
  removes stays removed, whatever its region says.
* **A merged blink takes `any`** (in if any of its localizations is): OR, as
  across ROIs.  The opposite of `use`, whose rule is `all` because hiding
  dirt should hide the whole blink.  For a cluster or a trace it does not
  arise: a blink's `cluster_id` is one value.

### What it writes

1. **A pass column in each region table it reads**, so the summaries
   (*Histograms*, *Curves*) count the passing regions only.
2. **`region_filter` on the localizations** (1 in, 0 out, grouped rule
   `any`) and a bound on it in every layer.  One column, not one per layer:
   comparing two filters in two layers is a column written on purpose.
3. **Its definition** (the conditions), from which both are recomputed --
   not a snapshot.

### When it is recomputed

* **When a result it reads changes** (a block measures again, a plugin
  hands back a new table): the flag and the pass columns are rebuilt from
  the definition.  From pass/fail per region to the localizations is a
  lookup by id for an id column, and the members of each ROI for ROIs.
* **After a table change that keeps the ids** (drift correction, say), the
  region tables are out of date as a whole.  The filter **keeps applying,
  with a warning**, as long as the id column's hash is unchanged; a changed
  id column (a new clustering) leaves nothing to apply, and the filter says
  so.
* **During a live fit**, appended rows get the flag of their region by id; a
  row of a region not yet measured (a trace still running) is out until it
  is.

### The feedback loop, kept out

A region's results depend on every layer's filters (`region_runs.
_layer_inputs`).  If the region filter were among them, measuring again
would run inside the filter: the regions that failed would be empty, their
values NaN, and the next pass would drop them for good -- a filter on a
result rewriting the result.  So **`region_filter`, and every column
written from region results, is neither an input of region results nor
applied while a region is measured.**  Today the columns written from
regions are already left out of the *columns* input (`REGION_COLUMNS`), but
a bound on them is not left out of the *filters* input; that is the first
thing to fix, and to cover by a test, whatever else is built.

## The panel

A section of the Regions tab, from the top:

1. **The applied conditions** (above), with ×.
2. **A tree selector with a search box** over everything filterable: source,
   evaluation, value, and as many levels as the data has -- `cfr › mean`,
   `cfr › max` where a plugin hands back statistics of a quantity, nothing
   more where it hands back a plain value.  Out-of-date evaluations are
   marked.  ROI annotations are an evaluation like any other (`use`,
   keywords, comment, the lines' lengths and angles,
   `core.annotation_columns`).
3. **The selected value's histogram over regions**, with draggable bounds
   (the layer filter's widget, on region values) and *add*.

In the **Render tab**, a *regions* button switches the region filter off and
on in a layer, on by default, so that seeing everything is one click away
without losing the conditions.

### Levels come from the data

The tree is as deep as the data.  The region table stays flat -- a column
per number, `cfr_mean`, `cfr_max` -- and a plugin that hands back
statistics of a quantity says so (`RegionTable.groups = {"cfr": {"mean":
"cfr_mean", "max": "cfr_max"}}`, saved with the table); the tree splits there
and nowhere else.  Guessing levels from underscores would split `n_corners`
and `x_nm`.  (Built: the groups are the table's own, `RegionTable.groups`,
saved with it, rather than a field of the result.)

## Maths on region values: *Math parser*

*Math parser* gets a **source**: the localizations, as now, or a region
source (the ROIs, or per an id column).  On a region source an expression
reads that source's values and makes a new one, kept in the region table as
a recipe and recomputed when its inputs are measured again.  Values of
several evaluations of one source meet by region id.

Names are **aliased**: an evaluation is called by a short alias the tree
selector fills in (`corners.n_corners / stats.n`), since an evaluation's
label (`NPC Corners`, `per cluster_id: Cluster shape`) is not an identifier.

With ***to localization table***, the value goes onto the localizations
(`region_runs.write_values`).  Unlike pass/fail a number cannot be OR-ed, so
overlapping ROIs keep the overlap rule (`last`, `first`, `mean`).  *Regions
to column* stays as a small plugin on the same function, for chains.

## Writing region tables fast

A block runs its plugin once per region in Python, so 10^5 traces are
minutes.  Statistics per id need no loop: a sort by id and a `bincount`
per quantity (`tracking.analysis.Tracks` already does this for tracks).  So
plugins that compute every region at once hand back a whole table in
`Result.id_regions` (`{column: {label: RegionTable}}`, which existed for
chains), made by `region_runs.column_table`: signed as a block's run would
sign it, so that `region_results`, *results from* and the filter's tree find
it current however it was computed.  *Id statistics*
(`plugins/id_statistics.py`, "per tid: Id statistics") is the first.

The table itself, measured with four values per region and 20 localizations
each (this container):

| regions | `split_ids` | `RegionTable.set`, row by row | HDF5 write | file | HDF5 read |
| --- | --- | --- | --- | --- | --- |
| 10^5 | 0.6 s | 0.2 s | 0.15 s | 24 MB | 0.2 s |
| 10^6 | 8.9 s | 3.0 s | 2.8 s | 236 MB | 4.4 s |

Fine at 10^5; at 10^6, two fixable things:

* **The file is mostly text**: every row stores its id, its signature and
  its data signature as strings, and for an id column both signatures are
  the same for every row.  Stored once, the four values are about 16 MB.
* **`split_ids` sorts every group again** after a stable sort that has
  already ordered them, and builds 10^6 small arrays.  Order and offsets
  instead.

So: a bulk writer from arrays (`RegionTable.from_columns(ids, values,
step, signature)`), shared strings stored once, `split_ids` without the
second sort, and a test at 10^6 regions.

Built (steps 1-3): the file at 10^6 regions is 44 MB (the ids and four
values; it was 236), written in 0.8 s and read in 1.8 s; `split_ids` is
6.5 s, the stable sort of 2 x 10^7 ids that remains -- *Id statistics*
does not use it, finding regions by a lookup table on the ids.

## What moves onto this

| now | then |
| --- | --- |
| *Filter ROIs* (`roi_order.py`) | conditions on ROIs in the region filter; its parser and its rule for missing values (an unmeasured ROI makes a comparison false) are kept.  *Use* stays an annotation -- which ROIs are measured at all -- that the filter can read. |
| *Sort ROIs* | stays; picks its columns with the tree selector |
| *results from* (`SiteAnalysisPlugin`: *Histograms*, *Curves*) | the tree selector, and honours the pass column |
| *Regions to column*, a block's *write values* | *Math parser* on a region source with *to localization table*; *Regions to column* a small plugin on top, for chains |
| `track_D_um2_per_s`, `track_alpha` (*Tracking analysis*, *write per track*) | a region table per track, from the vectorised fit it already does |
| *Stability* per bead | a region table per bead |
| MINFLUX per-trace statistics | a region table per `tid` (the first client) |
| `n_in_cluster` | not written by default once *Find clusters* hands back a table per cluster with `n`; kept while filtering on it before a block is how small clusters are skipped (*Cluster shape*'s page says so) |
| `track_length` | as `n_in_cluster` |
| `n_in_group` | stays: the grouping's, not a region analysis |

### Beads: open

`beads.BeadSet` is a region table in all but name -- metrics per bead,
pass/fail by a rule, choices by hand -- and its window is a region filter
with a figure.  What differs: the selection is inside the drift fit (each
bead is judged against the curve of the other used beads, and the curve
changes with the selection), and a choice by hand is a setting of the
plugin, by bead name, so that it survives the beads being numbered again.
Options: the metrics become a region table that the filter can show while
the selection stays in `select_beads`; or the selection moves to the filter
and the drift plugins read it.  To be discussed.

## Order of work

0. This note, and the changes to `docs/regions.md` (below), agreed.  The
   feedback loop fixed, with a test.
1. Region tables at scale: bulk writer, shared strings, `split_ids`, a test
   at 10^6 regions.  *Done.*
2. Whole tables through `Result.id_regions` (`region_runs.column_table`)
   and the groups of a value (`RegionTable.groups`).  *Done.*
3. *Done* (*Id statistics*, `Analysis/Measure/Id statistics`).  Statistics per id (`smappy/id_statistics.py` and a plugin): n, the
   standard deviation per axis, extent, first and last time, duration, and
   mean, median and max of chosen columns.  MINFLUX traces first, clusters
   with the same plugin.
4. The region filter: session state and `region_filter`, the panel, the
   Render tab's button, the plugin for chains; *Filter ROIs* folded in.
   *Done*, with step 0's feedback loop closed (`region_runs.unmeasured`).
   As built: the conditions are kept in the table's metadata (`KEY`), not
   in a separate session state, so a plugin's new table, an undo and a
   saved file carry them; a plugin hands them back as
   `Result.region_filter`, and a chain's copy keeps them in a table of its
   own and forwards them.  The pass column is not written into the region
   tables: `region_results.rows` asks the filter which regions pass, so a
   table measured again is never out of step with it.  The regions of the
   ROIs are every ROI, *use* among the annotations to filter on.  *Filter
   ROIs* stays as it was -- it sets *use*, which ROIs are measured at all --
   and its conditions on ROIs are now also the region filter's.
5. *Math parser* on region sources, aliases, *to localization table*.
6. Clients: Tracking analysis per track, Stability per bead, MINFLUX DCR per
   trace; beads once decided.
7. Pages, panel docs, `default.gui.yaml`, the shapes table in `CLAUDE.md`.

Steps 1-3 stand on their own and can be one change.

## Changes to `docs/regions.md`

* **"Region values as columns"**: from the route to filtering to an option.
  Old: "To filter or colour localizations by what was measured in their
  region ... each region's number is written onto its rows as a column".
  New: filtering goes through the region filter and one column,
  `region_filter`; a value is written onto the rows to colour by it, to
  compute with it per localization, or to compare filters between layers.
* **"a snapshot until written again"**: no longer true of the flag, nor of
  values written by a recipe; true of *Regions to column*'s columns only.
* **"Where results live"**: a plugin may hand back a whole table
  (`Result.id_regions`, `region_runs.column_table`), stored under the label
  a block would give it.  *Done.*
* **A new section, "Filtering regions"**: the combination rules above, and
  the feedback-loop rule.
