---
version: "1"
covers: [smappy.plugins.histogram.value_range, smappy.plugins.histogram.bin_edges, smappy.plugins.histogram.mode, smappy.plugins.histogram.describe_values, smappy.plugins.histogram.draw, smappy.plugins.histogram.Histogram.run]
---

## What it does

Counts the values of any column -- the axial position `z_nm`, the fit
quality `logl_rel`, the PSF width `sigma_nm`, DECODE's detection
probability `prob`, a column made with
[Math Parser](plugin:Analysis/Process/Math Parser) -- and draws them as a
histogram, with the mean, median, standard deviation and mode.  It is the
quick look at a column before filtering on it: where most values lie, where
the tail starts, whether there are two populations.

[Localization Statistics](plugin:Analysis/Measure/Localization Statistics)
does the same for photons, precision and on-time, and fits the law each of
them follows; this plugin fits nothing and takes any column.

Each visible [layer](plugin:Panels/Glossary) (one picture of the
localizations with its own filter and colours) gets its own histogram, read
as the layer shows it -- its [filter](plugin:Panels/Glossary), the drawn
[ROI](plugin:Panels/Glossary), and merged blinks if the layer merges them --
so two channels, or one channel filtered two ways, are compared on the same
bins.

It also runs once per [region](plugin:Panels/Glossary) -- in every ROI, or
every cluster -- giving the numbers of the column in each.

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

**2. The bins.**  A *bin width* you set is used.  Otherwise a column of
whole numbers (a frame, a count, an id) gets one bin per value, and any
other column a width that suits the number of values: more values, narrower
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

## In detail

**Range.**  With *range* on *without the extremes*, the range is
$[q_{p}, q_{1-p}]$, the quantiles of all the layers' finite values pooled,
$p$ the *leave off* fraction (0.001).  *Smallest to largest* takes the
minimum and maximum; *from ... to ...* takes what is typed, an empty end
being the data's.  A range of one value becomes one bin around it.

**Bins.**  A whole-number column (an integer column, or floats that are all
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

**Per region.**  The values are named after the column:
`<column>_n`, `_mean`, `_median`, `_std`, `_mode` for the first layer and
with `_L2`, `_L3`, ... for the others.  A region with no values of the
column gives `<column>_n` = 0 and nothing else.

## Parameters

### range
*Without the extremes* suits nearly every column.  Use *smallest to
largest* to see the outliers themselves, and *from ... to ...* to compare
several runs on the same axis.

### bin_width
Leave it on *auto*.  Set it to compare histograms of different files
bin for bin, or when a column of whole numbers is wanted in coarser bins.

## Output

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

## References

* Freedman D, Diaconis P. On the histogram as a density estimator: L2
  theory. *Z Wahrscheinlichkeitstheorie verw Gebiete* 57, 453 (1981).
  [doi:10.1007/BF01025868](https://doi.org/10.1007/BF01025868) -- the bin
  width.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
