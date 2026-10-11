---
version: "1"
covers: [smappy.plugins.clusters.roi_column, smappy.plugins.clusters.ROIToColumn.run]
---

## What it does

Writes the number of the ROI each localization lies in onto the table, as a
column (`roi_id`): the number the [ROI manager](plugin:Panels/ROI manager)
shows the ROI by, 0 for a localization in no ROI.  The ROIs then live in the
table itself -- they go wherever the table goes, a filter on `roi_id` shows
one of them, and a block of a chain set to *for each* `roi_id` repeats over
them like over clusters.

It is a snapshot.  Moving an ROI afterwards, adding one or removing one does
not change the column; run it again to bring it up to date.  The run is
written into the table's history, so the file says when the column was
made.

Only the ROIs ticked *use* are written.  Where two of them overlap, a
localization takes the number of the later one -- the one made after the
other, which has the higher number.

## How it works

**Which localizations.**  An ROI holds the localizations inside its outline
-- the project's circle or square, or its own polygon -- whatever the
layers' filters, as when a block repeats over the ROIs: membership is the
shape alone.

**The column.**  Every row starts at 0; the ROIs are then gone through in
the order they were made, each writing its number over the rows inside it.
The column is kept with the table when it is grouped, a blink taking the
highest number among its localizations.

## Parameters

### field
Another name keeps two snapshots side by side, before and after an edit of
the ROIs, say.

## Output

The table with the column; the text says how many ROIs were written, over
how many localizations, and how many are in none.
