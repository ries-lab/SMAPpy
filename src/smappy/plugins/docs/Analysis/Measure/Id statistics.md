---
version: "1"
covers: [smappy.id_statistics.id_statistics, smappy.id_statistics.dense_codes, smappy.id_statistics.stat_name, smappy.plugins.id_statistics.IdStatistics.run, smappy.plugins.id_statistics.summarised_columns, smappy.region_runs.column_table]
---

## What it does

Gives a few numbers for every [trace](plugin:Panels/Glossary), track or
[cluster](plugin:Panels/Glossary) at once: how many localizations it has,
where its centre is, how far its localizations spread around it, how long
it lasted, and the mean, median or maximum of any column -- a MINFLUX
trace's `cfr` or `efo`, the photons, the precision.

A MINFLUX microscope measures a molecule many times in a row, and calls
those localizations one trace (`tid`).  Whether a trace is good shows in
its numbers: one emitter gives a tight spread close to the
[localization precision](plugin:Panels/Glossary); two emitters at once, or a
molecule that jumped between sites, give a spread several times wider; a
trace of two or three localizations is often background.  The same holds
for any [id column](plugin:Panels/Glossary) -- a linked track's
`track_id`, a cluster's `cluster_id`, a blink's `group_id`.

The numbers are kept as a [region table](plugin:Panels/Glossary), a row per
trace, named "per tid: Id statistics", not as columns of the localizations.
*Summarize/Histograms* then shows how they are distributed over the traces,
and *Regions to column* writes one onto the localizations where it is wanted
there, to colour by it or to filter on it.

It measures every region in one pass.  The same numbers from
*Localization statistics* run per region in a block take minutes for 10^5
traces; here they take a second.  It reads the localizations, not merged
blinks, and only those the layer's filter keeps; a drawn ROI does not
narrow it, as it does not narrow a block's regions.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import SimulationSettings
from smappy.group import group
from smappy.id_statistics import id_statistics

# a blink, the localizations of one emitter in consecutive frames, is a
# region of group_id, as a trace is one of tid
locs = simulate(SimulationSettings(n_frames=2000, seed=1))
locs = locs[np.asarray(locs["photons"]) > 200]
group(locs)
err = np.asarray(locs["xy_err_nm"], float)
found = id_statistics(locs["group_id"], {"x": locs["x_nm"], "y": locs["y_nm"]},
                      columns={"err2": err ** 2,
                               "emitter": np.asarray(locs["emitter"], float)},
                      stats=("mean", "min", "max"))
v = found.values
enough = v["n_locs"] >= 5
one = enough & (v["emitter_min"] == v["emitter_max"])
two = enough & (v["emitter_min"] != v["emitter_max"])
ratio = v["std_xy_nm"] / np.sqrt(v["err2_mean"])
```

**1. The regions.**  Every distinct value above 0 of the id column is one
region; a localization with 0 (DBSCAN's noise, say) is in none.  Only the
localizations the layer's filter keeps count, and a region with none kept
is left out.

**2. Count, centre and spread.**  For each region: its number of
localizations, the mean of each coordinate, and the standard deviation of
each coordinate around that mean.  The lateral spread `std_xy_nm` is the
root mean square of the x and y ones, as `xy_err_nm` is of the two
precisions.

**3. Time.**  With a clock (`time_s`, MINFLUX) the first and last time and
the duration; with frames, the first and last frame and the span.

**4. Columns.**  For each chosen column, the chosen statistics of the
values that were measured -- a localization whose `cfr` is missing counts
for neither the mean nor the median.

```figure Blinks of a simulated data set, each a region of `group_id`: the spread of each blink's localizations over the precision they report (the root mean square of their `xy_err_nm`), for blinks of five or more localizations.  A blink of one emitter spreads by its precision (the ratio near 1); a blink that merged two emitters, which the simulation knows, mostly spreads two to five times wider; only where its emitters were closer than their precision does it look like one.
fig.set_size_inches(5.0, 3.0)
ax = fig.subplots()
bins = np.linspace(0, 5, 41)
ax.hist(ratio[one], bins=bins, color="0.35", label=f"one emitter ({one.sum()})")
ax.hist(ratio[two], bins=bins, color="tab:red", alpha=0.7,
        label=f"two or more ({two.sum()})")
ax.axvline(1, color="k", linewidth=0.8, linestyle=":")
ax.set_xlabel("spread / reported precision (std_xy_nm / RMS xy_err_nm)")
ax.set_ylabel("blinks")
ax.legend(fontsize=8, frameon=False)
```

## In detail

The standard deviation is the sample one,

$$s = \sqrt{\frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2},$$

so that its square is on average the true variance; for one localization it
is NaN.  It is computed in two passes, the mean first, since positions are
$10^4$ nm and their spread a few nm.  $s_{xy} = \sqrt{(s_x^2 + s_y^2)/2}$.

The spread is the region's own extent and the localization precision
together; nothing is subtracted.  For one emitter it is the measured
precision -- which is why it is worth comparing with the precision the fit
or the MINFLUX bound reports, `xy_err_median_nm` beside `std_xy_nm`.  For a
moving track it measures the motion, and is no quality measure.

A region's median is the middle value of its finite values, or the mean of
the two middle ones.  All regions' values are put in order with one sort,
of the region's number plus the value scaled into [0, 0.5]; two values of a
region closer than $2^{-52}$ times the number of regions times the
column's whole range can swap, which moves a median by no more than that.
Every other statistic needs no sort: the rows are put in region order once
(not at all for a MINFLUX file, whose traces follow each other), and the
regions are found by a lookup table on the ids, not by sorting them.
2 × 10^7 localizations in 10^6 traces take about 12 s with three medians,
10^5 traces about a second.

A statistic is named before the unit, so that a name keeps the table's
`<quantity>_<unit>` form: `cfr_mean`, `xy_err_median_nm`.  The statistics of
one column are listed together in the table, so a reader can show
`cfr › median` as a level of its own.

The table is signed as a block's run would sign it: by the table, the id
column and every layer's filter.  Once any of those changes it is out of
date, as a block's would be, and *Summarize* no longer offers it until it
is run again.

## Parameters

### id_column
*auto* takes MINFLUX's `tid`, else a linked `track_id`, else `cluster_id`;
choose `group_id` for blinks, or any other id column.

### columns
Names the table does not have are refused with the list of those it has.
On a MINFLUX file *auto* gives `cfr`, `efo`, `dcr`, `photons` and
`xy_err_nm`.

### use_z
Untick it for a 2D measurement whose `z_nm` is only noise (a MINFLUX 2D
file has z at ~10^-13 m).

### statistics.median
Leave it on for `cfr` and `efo`, whose distributions within a trace have a
long tail; it is the slowest statistic, a sort per column.

### statistics.mean
The mean of a column with a long tail is pulled by its few large values;
the median is the safer summary there.

### statistics.std
The spread of a column within a region: a trace whose `efo` jumps between
two levels has a large one.

### statistics.min
Mostly for times and frames written as columns of their own.

### statistics.max
A trace's worst `cfr`: one bad iteration in a trace shows here and not in
its mean.

## Output

The text gives the number of regions and localizations, the median count
per region, the median spread and duration, and the name the table is
kept under.  The figure shows the distribution over the regions of the
count, the spread, the duration (or the span in frames), and of the median
(else the mean) of each chosen column.

The table has, per region: `n_locs`; `centre_x_nm`, `centre_y_nm`
(`centre_z_nm`); `std_x_nm`, `std_y_nm` (`std_z_nm`) and `std_xy_nm`;
`start_s`, `end_s` and `duration_s`, or `first_frame`, `last_frame` and
`frames`; and for each column its statistics, `cfr_mean`, `cfr_median`,
`cfr_max`, ...  A good MINFLUX measurement has a narrow peak of `std_xy_nm`
near the precision with a tail of a few wide traces; a second peak at twice
the first is two emitters measured as one.
