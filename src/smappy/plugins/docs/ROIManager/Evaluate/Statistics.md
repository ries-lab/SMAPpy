---
version: "1"
covers: [smappy.plugins.roi.site_statistics, smappy.roi_manager.core.ROIProject.evaluate]
---

## What it does

Once the ROI manager has a list of sites -- found by
[Density Peaks](plugin:ROIManager/Segment/Density Peaks) or drawn by hand --
each site is *evaluated*: a plugin looks at the localizations inside one ROI
and returns a row of numbers for it.  The rows of all sites together make the
*site table*, which the Analyze plugins, such as
[Histograms](plugin:ROIManager/Analyze/Histograms), summarise.

Statistics is the simplest evaluator.  A new pipeline starts with the
general evaluators, and in a plain installation that is this one.  For each site it reports three numbers: how many
localizations it holds, their mean localization precision and their mean
photon count.  They are the first thing to look at in a set of sites: a
count far from the others marks a site that is empty, doubled or sits on
background, and the precision and photons say whether the sites were imaged
alike.

It needs `photons` and the precision column (`xy_err_nm` by default) in the
table.  A site where one of them is missing gets an error instead of a row,
and the other sites carry on.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.roi_manager import ROIProject
from smappy.plugins import Context
from smappy.plugins.roi import Statistics, StatisticsSettings
sim = SimulationSettings(n_frames=3000, seed=3,
                         structure=StructureSettings(preset="npc"),
                         labelling=LabellingSettings(efficiency=0.6))
locs = simulate(sim)
project = ROIProject()
source = project.add_source(locs)
rois = project.find(source.id)
for roi in rois:
    roi.reviewed = True          # the GUI does this when an ROI is made
project.evaluate()
rows = {row["roi_id"]: row for row in project.results()}
```

**1. The site's localizations.**  The ROI manager cuts out the localizations
inside the ROI: a circle of 300 nm by default, or the ROI's own square or
polygon.  They are the localizations the layer shows, after its filters, and
one row per blink when the layer is grouped.

**2. The numbers.**  Statistics counts them, and averages the precision
column and `photons` over them.

**3. The figure.**  The site is drawn as its localizations, with the count and
the precision in the title.  The ROI manager redraws it for each site as the
list is walked through.  The three numbers cannot tell a ring from a smear of
the same size; the picture can.

```figure The figure Statistics draws, for the site with the most (left) and the fewest (right) localizations among the simulated nuclear pores.  Both are single pores: the left one shows 25 fluorophores, the right one 9 of its 32 Nup96 copies -- labelling is random, and so is how often a dye blinks.
order = sorted(rois, key=lambda r: rows[r.id]["n_localizations"])
fig.set_size_inches(7, 3.4)
for ax, roi in zip(fig.subplots(1, 2), (order[-1], order[0])):
    result = Statistics().run(Context(locs=project.extract(roi),
                                      site=project.geometry(roi)),
                              StatisticsSettings())
    result.plot(ax)
    ax.title.set_fontsize(9)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
```

## In detail

For a site with localizations $i = 1, \ldots, N$ it reports

$$N, \qquad \bar{\sigma} = \frac{1}{N'} \sum_{i} \sigma_i , \qquad \bar{P} = \frac{1}{N''} \sum_{i} P_i ,$$

where $\sigma_i$ is the precision column and $P_i$ the photons.  The means
are arithmetic, over the finite values only: $N'$ and $N''$ are how many of
the $N$ values were finite, and are reported beside the means
(`mean_precision_nm_n`, `mean_photons_n`).  They differ from $N$ only when
the table holds `NaN`s, for example from a fit that failed to give a
precision.  An empty site has $N = 0$ and `NaN` means, and no figure.

The mean precision is the mean of the per-localization precisions, not the
precision of the site's centre, which is much smaller.  On a grouped layer
each row is a whole blink, so $N$ counts blinks, $P$ is a blink's summed
photons and $\sigma$ the precision the grouping gave the blink.

The evaluation of a site is recorded with this plugin's name, version and
settings, and with the site's inputs (the file, the ROI, the filters, the
grouping).  When any of them changes, the stored row is out of date and is
left out of the site table until the site is evaluated again.

## Parameters

### precision_column
`xy_err_nm` is the lateral precision; `z_err_nm` gives the axial one
instead, on a 3D table.  Whichever it is, the result is called
`mean_precision_nm`, so a column in pixels (`xy_err_pix`) gives a number in
pixels under that name.

## Output

One row per site, in the site table, with the columns

* `n_localizations`: the number of localizations (or blinks) in the ROI;
* `mean_precision_nm`: their mean precision, and `mean_precision_nm_n` the
  number of finite values it is the mean of;
* `mean_photons`: their mean photon count, and `mean_photons_n` likewise.

The text says how many localizations the site has, and the figure shows
them.  In a set of similar structures the counts scatter widely -- the
labelling, the number of blinks and the photons all vary from site to site
-- so look for the outliers rather than the spread: a site with a fraction
of the typical count, or several times it, is worth a look before it goes
into an average.

## Differences from SMAP

Based on SMAP's `ROIManager/Evaluate/generalStatistics`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* SMAP evaluates every layer that is switched on and reports, per layer,
  the count, the *median* precision, the mean PSF size, mean and median
  photons and background, the mean on-time, and counts per channel, grouped
  and ungrouped.  This reports three numbers for the layer the ROI manager
  follows, and takes the *mean* precision, over finite values only.
* The precision column can be chosen.
* The ROI is the manager's own geometry (circle, square or polygon); SMAP
  has a checkbox for a circular ROI.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
