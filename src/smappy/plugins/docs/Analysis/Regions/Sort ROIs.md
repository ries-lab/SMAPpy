---
version: "1"
covers: [smappy.plugins.roi_order.SortROIs.run, smappy.roi_manager.order.sort_ids, smappy.roi_manager.core.ROIProject.sort, smappy.roi_manager.core.ROIProject.sort_rows, smappy.roi_manager.core.ROIProject.ordered_ids]
---

## What it does

Puts the [ROIs](plugin:Panels/Glossary) (regions of interest: shapes kept in
the [ROI manager](plugin:Panels/ROI manager), inside which things are
measured) in order of something measured in them -- the number of corners
of a nuclear pore, the localizations in a cluster -- or of a keyword they
were given in the ROI manager (*annotate*).  The ROI manager's list, the
results table and every summary that reads it, such as
[Histograms](plugin:Analysis/Regions/Summarize/Histograms), then go through
the ROIs in that order.  Use it to look at the best or the worst sites
first, or to step through them in the ROI manager grouped by a keyword.

It changes the order only.  Each ROI keeps its number, its results and
whether it is used; *Filter ROIs* is the plugin that includes and excludes.

## How it works

**What it reads.**  For each ROI, the newest stored values of every
measurement made in it -- whatever ran it: the per-region steps, a chain or
a script -- named as in the results table, and its keywords, one column per
list.  A result that is out of date still counts, so the order does not
change under the user while the ROIs are measured again.

**The order.**  ROIs are ordered by *sort by*, those with equal values by
*then by*, then by *and then by*, and finally by their number.  A keyword
sorts by its place in its list: with *quality: good, bad, unsure*, good
comes first, then bad, then unsure (*largest first* reverses that).  An ROI
with no value -- not measured, the step failed, no keyword -- goes to the
end either way.

**Where it shows.**  The order is kept with the ROIs, saved in the file, and
used by the ROI manager's list (so Up and Down step through it) and by the
results table.  ROIs added later come after the sorted ones.  Clicking a
column of the ROI manager's list sorts by that column too, and clicking *ROI*
returns to the order of the numbers, as does running this with *sort by*
set to *none*.

## Parameters

### by
The lists offer every column some ROI has a value for, and every keyword
list.  A column two steps both write is named `step.column`.

### descending
For a count or a score, ticked puts the largest first; for a keyword list,
the last keyword of the list first.

### then_by
Useful when the first column takes few values, such as a count of corners:
the ROIs with the same count are then ordered by this.

### then_descending
As *largest first* above, for *then by*.

### last_by
Rarely needed; ties that remain after it are ordered by ROI number.

### last_descending
As *largest first* above, for *and then by*.

## Output

The line under the panel says how many ROIs were sorted, by what, and how
many had no value for the first column and went to the end.  Nothing is
drawn; the ROI manager's list shows the new order.

## Differences from SMAP

Ported from SMAP's *SortROIs* ([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* **Columns, not fields.**  SMAP browses a site's structure for a field and
  sorts by a keyword list by its value, the keyword's number.  Here the
  choices are the results table's columns and the keyword lists by name;
  a keyword still sorts by its place in its list.
* **Missing values last.**  SMAP sorts a missing value as 0.
* **Three keys, not four**, and the order is saved with the ROIs.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
