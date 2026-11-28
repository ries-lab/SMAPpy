---
version: "1"
covers: [smappy.plugins.histogram.value_range, smappy.plugins.histogram.bin_edges, smappy.plugins.histogram.mode, smappy.plugins.histogram.describe_values, smappy.plugins.histogram.draw, smappy.plugins.histogram.histograms_of, smappy.plugins.histogram.region_histograms, smappy.plugins.histogram.Histogram.run, smappy.plugins.histogram.Histogram.evaluation, smappy.region_groups.group_labels, smappy.region_groups.numeric_fields, smappy.region_results.rows]
---

## What it does

Counts the values of any attribute -- the axial position `z_nm`, the fit
quality `logl_rel`, the PSF width `sigma_nm`, DECODE's detection
probability `prob`, an attribute made with
[Math Parser](plugin:Analysis/Process/Math Parser) -- and draws them as a
histogram, with the mean, median, standard deviation and mode.  It is the
quick look at an attribute before filtering on it: where most values lie, where
the tail starts, whether there are two populations.

[Localization Statistics](plugin:Analysis/Measure/Localization Statistics)
does the same for photons, precision and on-time, and fits the law each of
them follows; this plugin fits nothing and takes any attribute.

Each visible [layer](plugin:Panels/Glossary) (one picture of the
localizations with its own filter and colours) gets its own histogram, read
as the layer shows it -- its [filter](plugin:Panels/Glossary), the drawn
[ROI](plugin:Panels/Glossary), and merged blinks if the layer merges them --
so two channels, or one channel filtered two ways, are compared on the same
bins.

It also runs once per [region](plugin:Panels/Glossary) -- in every ROI, or
every cluster -- giving the numbers of the attribute in each.

**Region results.**  With *source* set to *region results* it draws the
same histograms of what was measured *per region* instead: the results
table the per-region steps leave, one row per ROI (or per cluster, trace or
bead of an id attribute), one column per number a step returned.  That is
where a population shows itself -- the typical number of localizations of a
nuclear pore, how much it varies, which ROIs are outliers (empty, doubled,
on background) and deserve a look before they go into an average.  With
*attribute* empty every measure is drawn, a grid of panels; with one
chosen, its histogram alone.  In place of layers it overlays **groups** of
regions (*group by*): per file, per ROI keyword, per value of a results
column or of a condition of the localizations.  It needs results: measure
the regions first (*per-region steps* in the Regions tab, or a plugin such
as [Region statistics](plugin:Analysis/Measure/Region statistics)).

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.plugins.histogram import (LayerHistogram, bin_edges, describe_values,
                                      draw, value_range)
locs = simulate(n_frames=3000, seed=3)
photons = np.asarray(locs["photons"], float)
err = np.asarray(locs["xy_err_nm"], float)[photons > 200]     # two layers' filters
photons = photons[photons > 200]
groups = [("photons > 200", "#1f77b4", err), ("photons > 2000", "#d62728", err[photons > 2000])]
```

**1. The range.**  All the layers' values together decide it: by default
from the 0.1th to the 99.9th percentile, so that a handful of failed fits
far away do not squeeze everything else into one bin.

**2. The bins.**  A *bin width* you set is used.  Otherwise an attribute of
whole numbers (a frame, a count, an id) gets one bin per value, and any
other attribute a width that suits the number of values: more values, narrower
bins, between 20 and 200 of them.

**3. The numbers.**  Per layer, of the values inside the range: how many,
the mean, the median, the standard deviation and the mode (the peak, read
between the bins: the dashed line).

```figure The localization precision of a simulated table in two layers filtered at 200 and at 2000 photons, as fractions so the two sizes compare, with 1 % left off each end: the bright localizations sit at a smaller precision, with a shorter tail.
fig.set_size_inches(5.0, 3.0)
ax = fig.subplots()
arrays = [v for _, _, v in groups]
lo, hi = value_range(arrays, quantile=0.01)
edges = bin_edges(arrays, lo, hi)
layers = [LayerHistogram(name, colour, *describe_values(v, lo, hi, edges))
          for name, colour, v in groups]
draw(ax, "xy_err_nm", edges, layers, fraction=True, style="steps")
```

**Region results.**  The table is the current results of *results from*:
on *auto* everything measured on the ROIs, side by side, or the one table
of an id attribute when nothing was measured on the ROIs (asked, when there
are several).  Only regions still described by their results count: an ROI
moved, or a filter changed, since it was measured is left out, and a table
of clusters relabelled since is refused as out of date.  Each measure is
binned as an attribute is: its own range, its own bins, the numbers per
group.

```figure Two hundred simulated pores in two files, their radius and their count of labelled corners (whole numbers: a bin each), grouped by file: the second file's pores are a little larger.
fig.set_size_inches(7.0, 2.6)
rng = np.random.default_rng(1)
rows = [{"roi_id": i, "file_id": "a" if i < 100 else "b",
         "radius_nm": float(rng.normal(50 if i < 100 else 54, 3)),
         "n_corners": int(rng.binomial(8, 0.6))} for i in range(200)]
from smappy.plugins.histogram import region_histograms
found = region_histograms(rows, ["radius_nm", "n_corners"],
                          [r["file_id"] for r in rows])
for ax, (name, d) in zip(fig.subplots(1, 2), found.items()):
    draw(ax, name, d["edges"], d["groups"], style="steps")
```

## In detail

**Range.**  With *range* on *without the extremes*, the range is
$[q_{p}, q_{1-p}]$, the quantiles of all the layers' finite values pooled,
$p$ the *leave off* fraction (0.001).  *Smallest to largest* takes the
minimum and maximum; *from ... to ...* takes what is typed, an empty end
being the data's.  A range of one value becomes one bin around it.

**Bins.**  A whole-number attribute (an integer attribute, or floats that are all
whole numbers) over a range of at most 500 values gets the edges
$k - \frac{1}{2}$, one bin per value.  Otherwise the width is the
Freedman-Diaconis rule
([Freedman and Diaconis 1981](https://doi.org/10.1007/BF01025868)),
$h = 2\,\mathrm{IQR}\, n^{-1/3}$, with IQR the interquartile range and $n$
the number of values inside the range, and the number of bins is held
between 20 and 200.  A typed bin width that would make more than $10^5$
bins is refused.

**Numbers.**  Of the values inside the range; the text says how many fell
outside.  The standard deviation is the sample one ($n - 1$).  The mode is
the vertex of the parabola through the five bins around the tallest, kept
within them (otherwise the tallest bin's centre), as SMAP finds it: finer
than a bin for a smooth peak, and as noisy as the five bins it rests on.

**Per region.**  The values are named after the attribute:
`<column>_n`, `_mean`, `_median`, `_std`, `_mode` for the first layer and
with `_L2`, `_L3`, ... for the others.  A region with no values of the
attribute gives `<column>_n` = 0 and nothing else.

**Region results.**  The rows are `region_results.rows`': for the ROIs the
included ones whose results are current, for an id attribute every region
where the steps did not fail, and where the
[region filter](plugin:Analysis/Regions/Filter regions) has conditions on
that table, only the regions it keeps.  With *attribute* empty, a column is
drawn when every row holds a number there (`NaN` counts; text, `True` and
`False` do not), except `roi_id`, `file_id`, the *group by* column and the
counts behind means (columns ending in `_n`).  *Attribute* may also name
several columns, separated by commas.  A region without a finite value of a
column is left out of that histogram and counted in its panel; a column no
region has a value for is named in the text instead of drawn.  On *without
the extremes* the quantiles are values of the table, never between two
(numpy's "lower" and "higher"): what is left off is whole regions, and none
while the fraction is under one region's share -- interpolated, the 0.1th
percentile of 40 regions would leave the smallest out.  The groups are
`smappy.region_groups`': a region's file is the file its ROI was drawn on,
or, for an id attribute, the file of its localizations; a results column or
an attribute *(of the localizations)* is offered when it has 2 to 12
different values, and an attribute is read from each region's first
localization.  A row without the column is in the group "(none)".  The
region source runs on the whole table only: run per region, it is refused.

## Parameters

### source
*Localizations* for any attribute of the table; *region results* for the
numbers the per-region steps measured, one per region.  The form shows only
what the chosen one reads.

### column
For region results, leave it empty for an overview of every measure, then
choose the one to look at closely.  When two steps return a column of the
same name, the table keeps both with the step's name in front
(`label.column`); that is the name to choose.

### evaluation
*auto* for everything on the ROIs.  Pick one set of results to see only its
columns -- one of two settings of the same plugin, say, or an id
attribute's table such as *per tid: Region statistics*.

### group_by
*File* compares acquisitions; an ROI keyword list compares what was
annotated by hand; a condition written by
[File conditions](plugin:Analysis/Process/File conditions) compares
conditions across files.  Set *y* to *fraction* when the groups differ in
size.  For a table of medians and boxes per group, use
[Compare groups](plugin:Analysis/Regions/Summarize/Compare groups).

### range
*Without the extremes* suits nearly every attribute.  Use *smallest to
largest* to see the outliers themselves, and *from ... to ...* to compare
several runs on the same axis.

### bin_width
Leave it on *auto*.  Set it to compare histograms of different files
bin for bin, or when an attribute of whole numbers is wanted in coarser bins.

## Output

For region results, the figure is a grid of panels, up to four across, one
per measure, titled with the measure and its unit (`radius_nm` is *radius
(nm)*), each with its groups overlaid and the number of regions without a
value; the text says how many regions and measures went in, then per measure
the range and bins and per group the count, mean, median, standard
deviation and mode.  For a script, `Result.data["histograms"]` holds per
measure the bin edges, the histogram of each group, every value and the
region it came from (`roi_ids`), and the number missing.  A population of
similar structures gives one peak per measure; a second peak at twice the
typical count suggests ROIs holding two structures, a tail at low counts
ROIs on background.

For the localizations:

The figure, one histogram per layer (bars, or outlines with *draw as*;
counts, or each layer's fraction with *y*), with each layer's mode dashed
and its median in the legend.  The text gives the range and the number of
bins, and per layer the count, mean, median, standard deviation and mode,
and how many values fell outside the range.

## Differences from SMAP

Based on SMAP's `Analyze/measure/Histograms`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  The main changes:

* SMAP makes 200 bins over the range unless a width is set; here a whole
  number gets a bin of its own and anything else a width that suits the
  number of values, so a few hundred values are not a comb of empty bins.
* The range is worked out once over all the layers, so their bins line up;
  SMAP worked it out per layer.
* It runs per ROI and per cluster as well.
* It draws what was measured per region too (*source*), which SMAP looked
  at in the ROI manager's evaluation window or exported
  (`ROIManager/Analyze/ExportEvaluationsTable`).  This took over SMAPpy's
  own *Summarize/Histograms*, whose 20 equal bins from the smallest to the
  largest value are now the binning described above; a chain or a pinned
  tab naming the old plugin opens this one, its *columns* and *results
  from* kept, its *bins* dropped.

## References

* Freedman D, Diaconis P. On the histogram as a density estimator: L2
  theory. *Z Wahrscheinlichkeitstheorie verw Gebiete* 57, 453 (1981).
  [doi:10.1007/BF01025868](https://doi.org/10.1007/BF01025868) -- the bin
  width.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
