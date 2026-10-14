---
version: "1"
covers: [smappy.plugins.roi.histograms, smappy.roi_manager.core.ROIProject.results, smappy.region_results.rows, smappy.region_results.is_current]
---

## What it does

The ROI manager works in three steps: [ROIs](plugin:Panels/Glossary)
(regions of interest, one around each copy of a structure) are found
([Density Peaks](plugin:Analysis/Regions/Find/Density Peaks), or by hand), each
ROI is measured by the per-region steps, plugins that run once per ROI (such as
[Localization Statistics](plugin:Analysis/Measure/Localization Statistics)), and the collection of
results is analysed.  The per-region steps leave a *results table*: one row
per ROI, one column per number a step returned.

Histograms is the simplest analysis of that table.  It draws, for each
column, how the values are distributed over the ROIs.  The table may as
well be one of clusters: a block of a [chain](plugin:Panels/Glossary)
(plugins run one after another as one) over `cluster_id` -- the
clusters [Find clusters](plugin:Analysis/Regions/Find clusters) found -- leaves a row
per cluster, and *results from* offers it as *per cluster_id: ...*.  ROIs and
clusters are both [regions](plugin:Panels/Glossary): sets of localizations
measured together.  That is where a population shows itself: the typical
number of localizations of a nuclear pore, how much it varies, and which ROIs
are outliers -- empty, doubled, on background -- and deserve a look before
they go into an average.

It needs results: measure the ROIs first (*per-region steps* in the
Regions tab).

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.session import Session
from smappy.plugins import Context
from smappy.plugins.roi import Histograms, HistogramSettings
sim = SimulationSettings(n_frames=3000, seed=3,
                         structure=StructureSettings(preset="npc"),
                         labelling=LabellingSettings(efficiency=0.6))
locs = simulate(sim)
session = Session(locs)                           # the GUI's model, no window
session.show_grouped(0, False)                    # localizations, not blinks
project = session.rois
project.find(project.source_of(0).id)
project.evaluate()
rows = project.results()
```

**1. The results table.**  The rows are the current results of every included
ROI (*use* ticked).  An ROI whose result is out of date -- the ROI was
moved, a filter or a step's setting changed -- is left out until it is
measured again.  An ROI where a step failed has no values for that
step's columns, and no row at all if nothing else ran.  Nothing is
measured here: the histograms show what the table holds when the plugin is
run, and are a snapshot.

A table of clusters goes out of date as a whole, since the clusters change
together: when the table, its `cluster_id` or a layer's
[filter](plugin:Panels/Glossary) (the lowest and highest values of columns a
localization must lie between) is no longer
what the clusters were measured on -- three clusters relabelled as one, say
-- the plugin says the results are out of date instead of drawing clusters
that are not there any more, and *results from* stops offering them.  Run the
block again.  A column written from the results themselves (*Regions to
column*, a block's *write values*) does not count as a change.

**2. The columns.**  With *columns* empty, every column that holds a number
in every row is drawn, except the ROI and file identifiers.  Otherwise the
columns named, in that order.

**3. The histograms.**  Each column's values are split into *bins* equal
bins from the smallest to the largest value, and the number of regions in each
bin is drawn: a grid of panels, up to four across, each titled with the
measure and its unit (`radius_nm` is *radius (nm)*).

```figure The histograms drawn for about 140 simulated nuclear pores (Nup96, 60 % labelling) measured with Statistics, one panel per column.  The number of localizations per pore varies several-fold, from the random labelling and blinking alone; precision and photons, which are averages over a pore, vary much less.
fig.set_size_inches(7.5, 2.4)
panels = fig.subfigures(1, 3)
for panel, name in zip(panels, ("n_localizations", "mean_precision_nm", "mean_photons")):
    one = Histograms().run(Context(site_table=rows), HistogramSettings(fields=name))
    one.plot.draw(panel)
    for ax in panel.axes:
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
```

## In detail

**Which table.**  None of its own: it reads what the per-region steps
measured in each region, and those measured the table each layer showed.

**Which columns.**  A column is taken by default when the first row has it
and every row holds an integer or a floating-point number there (a `NaN`
counts as a number; `True`/`False` and text do not).  `roi_id` and
`file_id` are never drawn.  The rows' columns are stored in alphabetical
order, and the panels follow it.  With Statistics the default columns are
five: the three numbers and the two counts of finite values behind the
means (`mean_photons_n`, `mean_precision_nm_n`), which equal
`n_localizations` unless the table holds `NaN`s.  Name the columns to leave
those out.

**Missing values.**  A value that is not finite (`NaN`, or a row without that
column) is left out of its histogram, and the panel says how many regions
have no value.  A column no region has a value for is not drawn: the text
names it instead.

**Bins.**  The bins are $B$ equal intervals from the smallest to the largest
finite value, as `numpy.histogram` makes them; the largest value falls into
the last bin.

**Two steps with the same column.**  When two per-region steps return a
column of the same name, the results table keeps both, prefixed with the
step's label (`label.column`), and that is the name to give in *columns*.

## Parameters

### bins
Around the square root of the number of regions is a fair start: 10 for 100
ROIs, 30 for 1000.  Too many leaves most bins with one region or none; too
few hides a second population.

### fields
Names as in the results table, for example
`n_localizations, mean_photons`.  A name no row has is not drawn, and the
text names it.

### evaluation
*auto* for an overview of everything measured.  The list offers every
set of results on the ROIs, by the name of the step that made it; pick one
to see only its columns, under their own names -- one of two settings of the
same plugin, say, when both are on the ROIs.

## Output

* **The figure**: one histogram per column, up to four across, the number
  of regions on the vertical axis.
* **The text** says how many regions and columns went in.
* **The data**, for a script: for each column the counts, the bin edges, the
  values and which ROI each value came from (`roi_ids`), and the number
  missing; and the rows themselves.

A population of similar structures gives one peak per column.  A second peak
at twice the typical count suggests ROIs holding two structures; a tail at
low counts, ROIs on background or on incompletely labelled structures.

## Differences from SMAP

SMAP has no plugin of this name; its results per site (SMAP's word for an
ROI) are looked at in the ROI manager's evaluation window or taken out with
`ROIManager/Analyze/ExportEvaluationsTable`, a table of one evaluator's
results to plot elsewhere
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  This plots the
columns of all the per-region steps' results directly, and leaves out ROIs
whose result is out of date.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
