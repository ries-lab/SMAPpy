---
version: "1"
covers: [smappy.region_filter.evaluate, smappy.region_filter.Condition.passes, smappy.region_filter.region_values, smappy.region_filter.write, smappy.region_filter.passing_rows, smappy.plugins.region_filter.FilterRegions.run, smappy.region_runs.unmeasured, smappy.session.Session.set_region_filter, smappy.session.Session.refresh_region_filter]
---

## What it does

Keeps the localizations of the [regions](plugin:Panels/Glossary) whose
results pass a condition, and drops the rest from every layer, view and
plugin: the nuclear pores with six corners or more, the MINFLUX
[traces](plugin:Panels/Glossary) whose spread is under 10 nm, the clusters
of more than 20 localizations.  The results are those the regions were
measured with -- by the per-region steps, *Id statistics*, a block of a
chain -- and stay in their [region table](plugin:Panels/Glossary); what
reaches the localizations is one column, `region_filter` (1 in, 0 out),
and a bound on it in every layer.

This is the step for a chain or a batch.  By hand, the Regions tab's
[region filter](plugin:Panels/Regions tab) section edits the same
conditions, with the distribution of each value to set its range on, and
the Render tab's *regions* button turns the filter off in one layer to
see what it takes away.

The summaries -- *Histograms*, *Curves*, *Export table* -- then count the
regions the filter keeps, as the views show them.

## How it works

**1. A condition** is one value of one evaluation and a range -- `n_locs`
of *Id statistics* per `cluster_id`, from 20 up -- or, for a keyword or a
flag, the texts it accepts (an ROI's *use*, `true`).  A missing value (not
measured, NaN) fails.

**2. A region passes** if it passes every condition on its source -- the
ROIs, or the regions of one id column.  Of two overlapping pores, one with
the corners and the other with the diameter, neither passes.

**3. A localization is kept** if it is in any region that passes (an ROI
overlapping a passing one does not take its localizations away), and in a
passing region of every source that has conditions.  A localization in no
region of a filtered source -- DBSCAN's noise once clusters are filtered,
one outside every ROI -- is dropped.  The layers' own filters still apply
on top.

**4. A merged blink** is kept if any of its localizations is.

## In detail

The conditions are kept with the table, in its metadata, as the recipes of
derived columns are; a saved file opens filtered, and an undo brings back
the table's own.  The flag is written again from them whenever the table,
the ROIs or their results change -- never stored as a snapshot.

After a table change that keeps the regions' ids -- a drift correction --
the regions' results are out of date; the filter keeps applying them, and
says so, as long as the id column is the one they were applied to (its
hash is remembered).  A new clustering leaves nothing to apply: that
source's conditions are suspended, with a warning, until it is measured
again.  An ROI's values are its newest stored ones, current or not, as
*Filter ROIs* reads them.

A region is always measured whatever the filter says: neither
`region_filter` nor a value written from region results onto the
localizations (*Regions to column*) narrows a region's measurement, or
counts among what its results depend on.  Otherwise a filter on a result,
applied while the result is measured again, would empty the regions that
failed it, and the next run would drop them for good.

The filter is no undo step -- it changes no localization -- but it is
logged, since what a later measurement saw depends on it.

## Parameters

### mode
*add* keeps the conditions there are and adds this one, all of which must
hold; *replace* leaves only this one; *clear* removes all, and needs
nothing else.

### evaluation
The evaluations offered are those the session has: the ROIs' by name, an
id column's as "per cluster_id: ...", and *annotations*, an ROI's *use*,
keywords and lines.

### accept
For a keyword list, the keywords to keep (`good, unsure`); for an ROI's
*use*, `true` or `false`.

## Output

The text gives each condition with how many regions it keeps of how many,
and how many localizations are in regions that pass.  A warning says when
results are out of date, or a source's conditions are suspended.
