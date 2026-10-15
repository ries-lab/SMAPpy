---
version: "2"
covers: [smappy.plugins.region_column.RegionsToColumn.run, smappy.plugins.region_column.rows_values, smappy.region_runs.write_values, smappy.region_runs.roi_numbers, smappy.region_runs.region_members, smappy.region_runs.values_onto, smappy.region_runs.merge_pieces, smappy.region_runs.with_region_columns]
---

## What it does

Takes what was measured once per [region](plugin:Panels/Glossary) -- in each
[ROI](plugin:Panels/Glossary) (region of interest, a shape in the ROI
manager), or in each cluster an id column such as `cluster_id` names -- and
writes it onto the localizations of that region, as a column of the
table.  Every localization in a pore gets the pore's number of corners;
every localization in a cluster gets that cluster's size or precision.

The point is that a column is something every other tool already reads.
With the region's result on its localizations

* the layer's **[filter](plugin:Panels/Glossary)** (also called bounds) keeps the localizations of regions that measured
  well -- the pores with 6 or more corners, say -- and each layer filters
  on its own;
* *colour by* set to the column in the Render tab colours each region by its
  result;
* the column is saved with the table and kept when blinks are merged.

It can also write each ROI's own number, as the
[ROI manager](plugin:Panels/ROI manager) shows it (`roi_id`): the ROIs then
live in the table itself, and a block set to *for each* `roi_id` repeats
over them like over clusters.  An ROI's number is given when the ROI is made
and kept when others are removed, and the results' rows name the ROI by the
same number, so ROI 5 in the list, in the column and in the results is the
same ROI.

The results must be there first: the ROIs measured (by the
[per-region steps](plugin:Panels/Regions tab), or a block of a
[chain](plugin:Panels/Glossary) -- plugins run one after another as one --
over the
ROIs), or a block over an id column such as `cluster_id`.  A block can write
the same columns itself at the end of its run -- its *write values* field --
which keeps them up to date whenever the chain runs; this plugin is for
results that are already there.

It is a snapshot.  Moving an ROI, re-clustering, or measuring again does not
change the column; run it again (or let the block write it) to bring it up
to date.  The run is written into the table's history.

## How it works

**Which localizations.**  A region holds the localizations inside its
outline -- whatever the layers' filters -- or, for an id column, the rows
with its id: membership is the region alone, as when a block runs.  Only the
ROIs ticked *use* are written.

**The values.**  Each region's number is written over its rows.  A
localization in no region, or in a region with no result (the measurement
failed there, or the region came after it), gets no value (NaN): a filter
on the column drops it.  The ROI number is 0 outside every ROI.

**Overlaps.**  Where two ROIs overlap, a localization takes the later ROI's
value -- the one made after the other -- or, with *overlaps*, the first's or
the mean of both.  The regions of an id column never overlap.  A block's
*write values* writes by the same rules, the block's own *overlaps*; its
fourth choice, *any*, is for a flag the block writes back and is refused
for written values.

**Merged blinks.**  The columns are kept when blinks are merged -- [merge
blinks](plugin:Panels/Glossary), also called grouping or linking, combines
the localizations in consecutive frames that come from one blinking event
into one: a blink takes the mean of its localizations' values (all the same
unless the blink crosses two regions), and the highest ROI number.

## In detail

**Which table.**  The localizations, whatever the layer shows: the
values are written onto every localization of a region, and the merged
blinks take them when they are merged again (*Merged blinks*, above).

**What may be written over.**  The table remembers which columns were
written from regions (`metadata["region_columns"]`, with where each came
from).  Running again writes over those, but never over a column the
localizations had of their own: a value named like one (`photons`, say) is
refused, and *prefix* gives it another name.

**Which ROI results.**  For the ROIs, the values are those of the results
that still describe each ROI as it is now -- the same rows the analyses read
-- so an ROI moved since it was measured gets no value until it is
measured again.

## Parameters

### source
Lists every current set of results on the ROIs by its step's name, then
every current set for an id column's regions as *per cluster_id: ...*.  Left empty it
writes only the ROIs' numbers.  Results that no longer describe their regions
-- re-clustered, the table or a filter changed -- are refused with a message
saying so; a column written by this plugin is not such a change, so a second
value can be written from the same results.

### values
The names are those of the results' columns, as *Summarize/Histograms*
shows them.  The lines drawn on the ROIs in the ROI manager give
`<line>_angle_deg` and `<line>_length_nm` with or without *results from*:
they are the ROIs' own, as `roi_id` is, and need no measurement.  Writing every number of a set of results adds a column for each:
name the few you will filter or colour by.

### prefix
Needed when a value has a name the table already uses, and useful to keep
two snapshots side by side: `before_`, `after_`.

### merge
Only matters where ROIs overlap.  *mean* suits a value measured over both;
*last* keeps the newer ROI's, as drawn.

## Output

The table with the new columns; the text names them and says over how many
regions and localizations they were written, and how many localizations are
in none.

A histogram of such a column in the filter counts **localizations**, not
regions: a large cluster weighs more than a small one.  For the distribution
over regions, use *Summarize/Histograms* on the same results.
