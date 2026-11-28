---
version: "4"
covers: [smappy.id_statistics.id_statistics, smappy.id_statistics.member_statistics, smappy.id_statistics.region_shapes, smappy.id_statistics.covariance_axes, smappy.id_statistics.hull_measure, smappy.id_statistics.dense_codes, smappy.id_statistics.stat_name, smappy.id_statistics.parse_percentiles, smappy.plugins.region_statistics.RegionStatistics.run, smappy.plugins.region_statistics.regions_source, smappy.plugins.region_statistics.summarised_columns, smappy.plugins.region_statistics.roi_table, smappy.region_runs.column_table]
---

## What it does

Gives a few numbers for every region at once -- every ROI of the
[ROI manager](plugin:Panels/ROI manager), or every
[trace](plugin:Panels/Glossary), track or [cluster](plugin:Panels/Glossary): how many localizations it has,
where its centre is, how far its localizations spread around it, how long
it lasted, and the mean, median, maximum or percentiles of any attribute --
a MINFLUX trace's `cfr` or `efo`, the photons, the precision, z.

A MINFLUX microscope measures a molecule many times in a row, and calls
those localizations one trace (`tid`).  Whether a trace is good shows in
its numbers: one emitter gives a tight spread close to the
[localization precision](plugin:Panels/Glossary); two emitters at once, or a
molecule that jumped between sites, give a spread several times wider; a
trace of two or three localizations is often background.  The same holds
for any [id attribute](plugin:Panels/Glossary) -- a linked track's
`track_id`, a cluster's `cluster_id`, a blink's `group_id`.

For ROIs, choose *ROIs* under *regions*: the count, spread and attribute
statistics of every ROI, the quick summary before or instead of measuring
the ROIs one at a time.

The numbers are kept as a [region table](plugin:Panels/Glossary), a row per
region, not as attributes of the localizations: for an id attribute named
"per tid: Region statistics", for the ROIs kept with them as the per-region
steps' results are, under "Region statistics", where *Filter ROIs* and
*Sort ROIs* read them too.
[Histogram](plugin:Analysis/Measure/Histogram) on region results then shows how they are distributed over the traces,
and *Write results* (to the localizations) writes one onto the localizations where it is wanted
there, to colour by it or to filter on it.

With *shape* ticked it also describes each region's shape, with the
numbers and names of [Cluster shape](plugin:Analysis/Measure/Cluster shape):
the area it covers, how elongated it is and in which direction, how far it
extends.  *Cluster shape* run in a block measures one cluster at a time and
draws it; this gives every cluster's numbers at once.

It measures every region in one pass.  The same numbers from
*Localization statistics* run per region in a block take minutes for 10^5
traces; here they take a second.  That is the line between the two: this
plugin gives cheap numbers -- counts, means, spreads -- for many regions at
once; the [per-region steps](plugin:Panels/Regions tab) run any
measurement in each region with that region's own context (its shape, its
frame, a fit, a figure), one region at a time.  *Localization statistics*
there fits the photon and precision distributions of each region, which
this does not.  It reads the localizations, not merged
blinks, and only those the layer's filter keeps; a drawn ROI does not
narrow it, as it does not narrow a block's regions.  So on a MINFLUX file a
filter on `trace_index` (each localization's number in its trace, from 1)
leaves out the start of each trace, where a still emitter often creeps into
place over its first 5 to 30 localizations: `trace_index > 20`, say, before
a trace's centre and spread are measured.

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

**1. The regions.**  For an id attribute, every distinct value, 0
included, is one region; a localization with -1 (DBSCAN's noise, say) is in
none.  For the ROIs, each ROI with *use* ticked is one region, holding the
localizations its shape (and its ids and ranges) takes in; ROIs may
overlap, and a localization in two of them counts in both.  Only the
localizations the layer's filter keeps count, and a region with none kept
is left out.

**2. Count, centre and size.**  For each region, always: its number of
localizations, the mean of each coordinate -- its centre, where the region
is -- and its lateral spread `std_xy_nm`, the root mean square of the
standard deviations of x and y around the centre, as `xy_err_nm` is of the
two precisions.

**3. Position.**  For each axis ticked under *attributes*: its own standard
deviation (`std_z_nm`) and the chosen statistics -- the median, maximum,
percentiles (`z_median_nm`, `z_p95_nm`); the mean is the centre.

**4. Time.**  With `time_s` ticked (a clock, MINFLUX) the first and last
time and the duration; with `frame` ticked, the first and last frame and
the span.

**5. Attributes.**  For each other ticked attribute, the chosen statistics
of the values that were measured -- a localization whose `cfr` is missing
counts for neither the mean nor the median.

**6. Shape** (with *shape* ticked).  For each region: the
covariance of its x and y positions -- how they spread together, an
ellipse -- whose long and short axis give the standard deviations along and across the region
and the direction it is stretched in; the extents along both axes; the
radius of gyration; and the area of its [convex hull](plugin:Panels/Glossary),
the smallest convex outline around its localizations, with the density in
it.  All but the hull are computed for every region at once; the hulls are
computed one region after the other, which is most of the time *shape*
takes.

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

```figure Clusters drawn from Gaussians stretched 2.5 to 4 times along an angle chosen at random: the angle *shape* finds against the one they were drawn with (left) and the standard deviation along the long axis against the true one (right).  A point in a corner is an angle near 0 found near 180 degrees: the same axis.  Each cluster is several hundred localizations; with fewer the scatter grows, and a round cluster has no angle worth reading.
rng = np.random.default_rng(3)
k = 150
n = rng.integers(300, 600, k)
angle = rng.uniform(0, 180, k)
long = rng.uniform(30, 80, k)
short = long / rng.uniform(2.5, 4.0, k)
ids = np.repeat(np.arange(k), n)
u, w = rng.normal(size=len(ids)) * long[ids], rng.normal(size=len(ids)) * short[ids]
t = np.radians(angle[ids])
shaped = id_statistics(ids, {"x": u * np.cos(t) - w * np.sin(t),
                             "y": u * np.sin(t) + w * np.cos(t)}, shape=True).values
fig.set_size_inches(6.0, 2.9)
a1, a2 = fig.subplots(1, 2)
fig.subplots_adjust(wspace=0.45)
a1.plot(angle, shaped["angle_deg"], ".", color="0.35", markersize=4)
a1.plot([0, 180], [0, 180], "k:", linewidth=0.8)
a1.set_xlabel("true angle (degrees)")
a1.set_ylabel("angle_deg")
a2.plot(long, shaped["std_long_nm"], ".", color="0.35", markersize=4)
a2.plot([30, 80], [30, 80], "k:", linewidth=0.8)
a2.set_xlabel("true std along (nm)")
a2.set_ylabel("std_long_nm")
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
attribute's whole range can swap, which moves a median by no more than that.
A percentile $q$ is read from the same order, linear between the two values
around position $h = (k - 1)\,q/100$ among a region's $k$ finite values, as
`numpy.percentile` does by default; it is named `p` and the number, a dot
written as `_` (`photons_p2_5`), so the name stays one an expression can use.
Every other statistic needs no sort: the rows are put in region order once
(not at all for a MINFLUX file, whose traces follow each other), and the
regions are found by a lookup table on the ids, not by sorting them.  The
ROIs' localizations are found through the spatial index, ROI by ROI, as the
ROI manager finds them; their row lists are then joined -- a localization in
two ROIs listed twice -- and measured by the same pass, so an id attribute
and ROIs holding the same localizations give the same numbers.
2 × 10^7 localizations in 10^6 traces take about 12 s with three medians,
10^5 traces about a second.

**Shape.**  The shape uses the localizations whose x and y (and z, in 3D)
are finite; `n_locs` and the centres count every localization the filter
keeps, each coordinate's mean over its finite values, so the two differ only
where a position is missing.  With the sums of squared deviations $S_{xx}$,
$S_{yy}$, $S_{xy}$ about the centre and $n$ localizations, the covariance
entries are $v_x = S_{xx}/(n-1)$, $v_y$ and $c$ likewise, and its eigenvalues

$$\lambda_{\pm} = \frac{v_x + v_y}{2} \pm \sqrt{\left( \frac{v_x - v_y}{2} \right)^2 + c^2},$$

so `std_long_nm` $= \sqrt{\lambda_+}$ and `std_short_nm` $= \sqrt{\lambda_-}$;
a $\lambda_-$ below $10^{-12} \lambda_+$ (points on a line) is 0.  The long
axis' angle is $\theta = \frac{1}{2} \mathrm{atan2}(2c, v_x - v_y)$, given in
degrees from 0 to 180 from the x axis: a cluster has no direction, so 10 and
190 degrees are one axis.  (*Trace statistics* turns its axis the way a trace
moved, 0 to 360; this does not.)  `elongation` is `std_long_nm` /
`std_short_nm`, NaN where the short one is 0.  The extents are the range of
the positions turned onto the two axes.  `rg_nm` is
$\sqrt{(S_{xx} + S_{yy})/n}$.  The hull is qhull's (`scipy.spatial.ConvexHull`);
fewer than three points, or points on one line, have an area of 0, and
`density_per_um2` is $n$ over the area in µm², NaN where the area is 0.  In 3D,
`z_extent_nm` is the range of z and `volume_nm3` the volume of the 3D hull
(0 below four points).  A standard deviation of one localization is NaN.
*Cluster shape* computes its one region with the same function, so the two
agree to the last digit.

The hulls are the one loop over regions, about 35 µs each in qhull: 10^5
clusters of 5 to 60 localizations took 3.9 s in 2D and 7.9 s in 3D (two
hulls each) on an M-series Mac, against 0.15 s for every other number.
Regions too small for a hull are not passed to qhull.

A statistic is named before the unit, so that a name keeps the table's
`<quantity>_<unit>` form: `cfr_mean`, `xy_err_median_nm`.  The statistics of
one column are listed together in the table, so a reader can show
`cfr › median` as a level of its own.

The table is signed as a block's run would sign it: by the table, the id
attribute and every layer's filter; for the ROIs, each row by its ROI's own
inputs, as a per-region step's is, so moving one ROI makes its row alone
out of date.  Once any of those changes it is out of date, as a block's
would be, and *Summarize* no longer offers it until it is run again.

## Parameters

### regions
*auto* takes MINFLUX's `tid`, else a linked `track_id`, else `cluster_id`,
and the ROIs only when the table has none of them: on a MINFLUX file the
traces are what one measures, whatever ROIs were drawn.  Choose `group_id`
for blinks, or any other id attribute.

### attributes
The list opens from the button, which names what is ticked.  Its names are
in groups -- *position* (`x_nm`, `y_nm`, `z_nm`), *time* (`time_s`,
`frame`), *localization* (photons, background, precisions, PSF width,
likelihood), *MINFLUX* (`cfr`, `efo`, `dcr`) and *other*, the rest of the
table -- each under a header whose tick ticks or unticks all of them.  Id
attributes are not offered.

Ticked by default: the position, the time, `photons` and `xy_err_nm`.  The
centre and `std_xy_nm` are measured whatever is ticked; untick an axis to
leave out its own spread and statistics -- `z_nm` for a 2D measurement
whose z is only noise (a MINFLUX 2D file has z at ~10^-13 m).  Tick
*MINFLUX* on a MINFLUX file to judge its traces by `cfr` and `efo`.

A ticked name this table lacks is passed over with a warning, so one chain
runs over files of different kinds; a default name is passed over silently,
since no table has all of them.  The list hides a group member the table
lacks but keeps it ticked.  Every ticked attribute adds a column per
statistic, so the table grows wide: the results window's *select* picks
what to look at or save.

### shape
Off by default, since its hulls take about 35 µs per region, seconds for
10^5 regions.  Tick it for clusters (`cluster_id`); for a MINFLUX trace of a
still molecule the shape is that of its localization error, round, with an
angle that means nothing.  The 3D values come with `z_nm` ticked.

### statistics.median
Leave it on for `cfr` and `efo`, whose distributions within a trace have a
long tail; it is the slowest statistic, a sort per attribute.

### statistics.mean
The mean of an attribute with a long tail is pulled by its few large values;
the median is the safer summary there.

### statistics.std
The spread of an attribute within a region: a trace whose `efo` jumps between
two levels has a large one.

### statistics.min
Mostly for times and frames written as attributes of their own.

### statistics.max
A trace's worst `cfr`: one bad iteration in a trace shows here and not in
its mean.

### statistics.percentiles
Numbers from 0 to 100, separated by spaces or commas.  `5 95` gives the
range that holds a region's middle 90 %, less sensitive to one outlier than
the minimum and maximum; with a few localizations per region the extreme
percentiles are the minimum and maximum themselves.

## Output

The text gives the number of regions and localizations, the median count
per region, the median spread and duration, and the name the table is
kept under.  The figure shows the distribution over the regions of the
count, the spread, the duration (or the span in frames), and of the median
(else the mean) of each chosen attribute.

The table has, per region: `n_locs`; `centre_x_nm`, `centre_y_nm`
(`centre_z_nm`) and `std_xy_nm`; for each ticked axis `std_x_nm` and its
statistics, `x_median_nm`, `x_max_nm`, ...; with the time ticked `start_s`,
`end_s` and `duration_s`, with the frame `first_frame`, `last_frame` and
`frames`; and for each attribute its statistics, `cfr_mean`, `cfr_median`,
`cfr_max`, `cfr_p95`, ...; with *shape* also `area_nm2`, `density_per_um2`, `rg_nm`,
`std_long_nm`, `std_short_nm`, `angle_deg`, `elongation`, `extent_long_nm`,
`extent_short_nm`, and in 3D `z_extent_nm` and `volume_nm3`, which the text
summarises by their medians and the figure by the distribution of the
elongation.  A good MINFLUX measurement has a narrow peak of `std_xy_nm`
near the precision with a tail of a few wide traces; a second peak at twice
the first is two emitters measured as one.
