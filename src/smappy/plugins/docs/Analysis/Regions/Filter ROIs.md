---
version: "1"
covers: [smappy.plugins.roi_order.FilterROIs.run, smappy.plugins.roi_order.new_use, smappy.roi_manager.order.condition, smappy.roi_manager.core.ROIProject.filter_rois]
---

## What it does

Includes or excludes [ROIs](plugin:Panels/Glossary) (regions of interest:
shapes kept in the [ROI manager](plugin:Panels/ROI manager), inside which
things are measured) by a condition on what was measured in them and the
keywords they were given in the ROI manager (*annotate*): keep the nuclear
pores with six corners or more, the clusters of more than 50 localizations,
the sites annotated *good*.  It sets each ROI's *use*, which is what the
per-region steps, the results table and the summaries respect.  Use it in
place of excluding ROIs one by one, and in a chain, after the ROIs were
measured.

## How it works

**The condition.**  A line such as

    n_corners >= 6 and quality == "good"

Names are the results table's columns and the keyword lists; a keyword is
written in quotes and compared as text.  Comparisons (`==`, `!=`, `<`, `<=`,
`>`, `>=`, also chained, `2 < n < 9`), `and`, `or`, `not`, arithmetic and
`in ["good", "unsure"]` are understood; nothing else is.  A column whose
name has spaces in it -- `NPC Corners.n_corners`, when two steps write
`n_corners` -- is written in backquotes.

**What it reads.**  For each ROI, the newest stored values of every
measurement made in it, current or not, and its keywords, as Sort ROIs reads
them.  An ROI with no value for a name fails every comparison with it: an
unmeasured ROI is left out by `n_corners >= 6`.  A name no ROI has is
refused, since it is nearly always a typing error that would exclude
everything.

**Before it runs.**  Under the panel, as the condition is typed, how many
ROIs it would use, and how many are used now.

**Running it** sets *use* on every ROI, as one step of the ROI manager's
undo (*Undo ROI edit*).  An ROI keeps its number and its results either way.

## Parameters

### condition
Start from one column and check the count under the panel before adding
more.  The columns the ROIs have are listed there while the field is empty.

### mode
*exclude the others* only ever excludes: a filter run after another
narrows the set further, and an ROI excluded by hand stays excluded.
*use exactly these* also includes every ROI that passes, undoing earlier
exclusions -- SMAP's behaviour.

## Output

The line under the panel says how many ROIs are used and how many changed.
Excluded ROIs turn grey in the ROI manager and drop out of the results.

## Differences from SMAP

Ported from SMAP's *FilterSites* ([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* **Keywords by their text.**  SMAP's conditions compare a list's value,
  the keyword's number, so reordering a list changes what a filter keeps.
* **Parsed, not evaluated.**  SMAP builds a MATLAB expression of six
  variables and evaluates it; here the condition names the columns itself
  and only comparisons, logic and arithmetic are allowed.
* **Undoable, and narrowing by default.**  SMAP overwrites *use* for every
  site.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
