---
version: "1"
covers: [smappy.plugins.region_export.ExportTable.run, smappy.plugins.region_export.with_positions, smappy.plugins.region_export.write_table, smappy.plugins.region_export.table_columns]
---

## What it does

Writes the results table -- a row per [ROI](plugin:Panels/Glossary) (region
of interest), or per [region](plugin:Panels/Glossary) of an id column such
as a cluster, and a column per number the per-region steps measured -- to a
csv file, for the statistics and figures made in a spreadsheet, in Python
or in R.

The table is what the ROI manager's results show: every measurement made on
the ROIs side by side, or, with *results from*, one named measurement --
on the ROIs, or per cluster.  The ROIs' annotations (their keywords) are
columns of it too.  Tick the columns to write in *columns*.

With *ROI position*, each ROI's row also has its centre, its file and its
comment: what is needed to find an ROI again from a row of the spreadsheet.

## How it works

**1. The rows.**  The included ROIs (*use* ticked) whose results still
describe them, in the order of the ROI list -- as
[Sort ROIs](plugin:Analysis/Regions/Sort ROIs) left it -- or the regions of
the id column the chosen measurement was made per.

**2. The columns.**  `roi_id` (the ROI's number, or the region's id) and
`file_id` first; with *ROI position* then `x_nm`, `y_nm`, `file` and
`comment`; then every number measured and every annotation, in the order
they first appear.  Only the ticked ones are written; all ticked writes
every column there is when it runs.

**3. The file.**  Commas in a `.csv`, tabs in a `.tsv` or `.txt`, one header
line.  A value an ROI does not have -- a measurement that failed there, a
keyword it was not given -- is an empty cell.

## Parameters

### positions
Off for a table that only feeds a statistic; on to go back to the ROIs, or
to draw the sites on a map of the sample.  A region of an id column has no
position of its own and gets none: its centre is a measurement
(Localization Statistics' mean position).

## Output

The file, and a line saying how many rows and columns were written where.
The export is written into the history, since where a file went is not
visible in the file.

## Differences from SMAP

Based on SMAP's `ExportEvaluationsTable`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), which writes
every single number one evaluator measured per site, with the site, its
cell and its file, to a csv.  The main changes:

* Every measurement on the ROIs can be written side by side, and the
  ROIs' annotations with them; SMAP wrote one evaluator at a time.
* The columns are ticked in a list; SMAP wrote all of them.
* The clusters' results (per region of an id column) are written the same
  way.
* SMAP's `SaveSiteParameters`, which saved the sites with everything they
  held as a MATLAB file, has no counterpart: the ROIs and their results are
  saved with the localizations by *Save*.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
