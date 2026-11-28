---
title: Region results
summary: What was measured in every region, as one table -- sorted, copied, or turned into the colour of the localizations.
widget: smappy.gui.region_results_window.RegionResultsWindow
covers: [smappy.gui.region_results_window.results_table, smappy.region_filter.region_values, smappy.region_filter.sources, smappy.region_colour.name, smappy.region_colour.numeric_values]
---

## What it does

A per-region step measures something in each
[region](plugin:Panels/Glossary) -- each ROI, or each cluster, track or
trace of an id attribute -- and keeps the answer with that region: a
region table, a row per region and a column per value.  The Regions tab
shows the values of the one ROI selected in the ROI manager; this window
shows all of them at once, for any source: the ROIs, or the regions of
an id attribute (`cluster_id`, `track_id`, `tid`).

Use it to look over the numbers before summarising them -- which pores have
too few corners, which clusters are unusually large -- to sort the regions
by one value and step through the extremes, to copy the table into a
spreadsheet, and to see a value on the picture: a column's header colours
the localizations by it.

It is also where the table is saved for a spreadsheet: *Save...* writes
it as a CSV, the columns chosen under *select*.  (In a chain or a batch,
where there is no window, [Export table](plugin:Analysis/Regions/Summarize/Export table)
writes the same.)

It opens from the Regions tab's *Results table...*, and from the header of
the ROI manager's list.

## How it works

**One table per source.**  *regions* chooses whose regions: the ROIs, or
the regions of one id attribute.  Each row is a region -- an ROI by its
number, a cluster or track by its id -- and each column one value of one
evaluation, its name over the evaluation's in the header.  An empty cell is
a value that region does not have: not measured, or measured by a step that
gives nothing there.

**Out of date.**  A value whose result is older than the region or the
step's settings is drawn in orange: for an ROI, a value of a step whose
result for it is out of date (the ROI was moved, or the step changed); for
an id attribute, every value of an evaluation that is no longer current
(the localizations changed since it ran).  The numbers are those stored,
which is what the summaries and the region filter read too.

**An ROI's own columns.**  For the ROIs the first columns are the ROI's
own, not measured: its centre (`x_nm`, `y_nm`), its file's name and its
comment, under *ROI* -- what is needed to find an ROI again from a row of
a spreadsheet, or to put the sites on a map.

**It follows the regions.**  The table is built again whenever the ROIs or
their results change, and when the window is shown again.

**From a column to the picture.**  *Colour localizations by this* sets the
Render tab's current layer to colour by that value (*colour by* › *region
value*): each localization takes its region's value, the mean where ROIs
overlap, grey where it is in none.  Nothing is written onto the
localizations; the [Render tab](plugin:Panels/Render tab)'s page explains
how.

## Controls

### regions
The source whose regions are listed: *ROIs*, or *per* an id attribute, for
every attribute something was measured over.  Only sources with results are
offered.

### used ROIs only
On by default: the ROIs whose *use* is ticked in the ROI manager, the ones
every summary and the region filter read.  Off lists every ROI.  Shown only
for the ROIs.

### select
Choosing the columns *Copy* and *Save...* write.  While it is pressed each
column's header carries a tick, a click on a header ticks or unticks it
(instead of sorting), the unticked columns are drawn grey, and *all* and
*none* appear above the table with how many columns are ticked.  Press it
again to sort and read as before; the choice stays, also when the table is
built again, and a column a step adds later is ticked.  Every column is
ticked to begin with.

### Copy
The ticked columns, in the order the table is sorted, tab-separated, to
the clipboard: pasted into a spreadsheet it keeps its columns.  The header
is the value and its evaluation, `n_locs / Region statistics`.  The
numbers are exact, not the four digits the cells show, and a value that is
out of date is left empty: a file cannot show it orange, and it no longer
describes its region.

### Save...
The same as a CSV file, for statistics and figures made elsewhere.

A right click on a column's header offers:

* *Colour localizations by this* -- the Render tab's current layer coloured
  by that value, and the Render tab brought forward.  Greyed for a column
  of texts (a keyword), which no LUT can show.
* *Filter regions on this...* -- the Regions tab's region filter unfolded
  on that value, with its distribution over the regions, ready for the
  range to keep.
* *Copy this column* -- the regions' names and that column alone, to the
  clipboard.

A click on a row shows the region: an ROI is selected in the ROI manager,
which zooms to it; a cluster or track is centred in the render window.
