---
version: "1"
covers: [smappy.plugins.curves.summarise, smappy.plugins.curves.spread_of, smappy.plugins.curves.common_grid, smappy.plugins.curves.x_axis, smappy.plugins.curves.curve_names, smappy.plugins.curves.mean_values, smappy.plugins.curves.Curves.run, smappy.plugins.curves.Curves.regions, smappy.plugins.curves.Curves.evaluation, smappy.plugins.spatial_statistics.SpatialStatistics.pool]
---

## What it does

Some per-region steps measure a curve rather than a number:
[Spatial statistics](plugin:Analysis/Measure/Spatial statistics) gives the
[pair correlation](plugin:Panels/Glossary) g(r) and L(r) - r of every
[ROI](plugin:Panels/Glossary).  One ROI's curve is noisy; this draws them
together -- the **mean curve with its spread** over the regions -- and, with
*group by*, one curve per keyword the ROIs were annotated with ("control",
"treated"), so conditions are compared in one plot.

It reads the curves of one evaluation (*results from*), the arrays each
region kept beside its values.  Regions of an id column (clusters) are read
too, as one group: they carry no keywords.

**Counts are pooled, not averaged.**  Where the measurement is a ratio of
counts -- g is the pairs found over the pairs chance would give -- the curve
drawn is made from the counts summed over the group's ROIs, not the mean of
the ROIs' curves.  The mean weighs an ROI with 20 pairs in a ring as much as
one with 20,000, and its noise with it; the sum weighs each pair once.  For
Spatial statistics the pooled g is the g of all the group's ROIs taken
together.  The spread is still the ROIs' own: how much their curves differ.

Use it after running a curve-measuring step in every ROI (the per-region
steps, or a chain's block).  All the ROIs must have been measured with the
same settings: curves on different x grids are refused (*In detail*).

## How it works

```figure-setup
import tempfile, yaml
from pathlib import Path
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.curves import Region, draw, summarise
from smappy.plugins.spatial_statistics import (RadialSettings, SpatialStatistics,
                                               SpatialStatisticsSettings)
from smappy.regions import Region as Shape
from smappy.simulate import simulate
from smappy.simulate.settings import SimulationSettings, StructureSettings
# clusters of radius 60 nm on the left half, the same labels spread evenly on
# the right
disc = [[float(60 * np.cos(t)), float(60 * np.sin(t))]
        for t in np.linspace(0, 2 * np.pi, 33)[:-1]]
spec = {"name": "two halves", "parts": [
    {"elements": [{"polygon": disc, "density": 0.004}],
     "copies": {"n": 160, "field": [0, 0, 6000, 6000], "min_distance": 250}},
    {"elements": [{"polygon": [[6000, 0], [12000, 0], [12000, 6000], [6000, 6000]],
                   "density": 1.3e-4}]}]}
path = Path(tempfile.mkdtemp()) / "halves.yaml"
path.write_text(yaml.safe_dump(spec))
locs = simulate(SimulationSettings(n_frames=2000, seed=6,
                                   structure=StructureSettings(file=str(path))))
locs = locs[np.asarray(locs["photons"]) > 200]
x, y = np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])
plugin = SpatialStatistics()
settings = SpatialStatisticsSettings(radial=RadialSettings(r_max_nm=300, dr_nm=10,
                                                           null="none"))
regions = []
for i in range(6):                          # 2 x 2 um squares: 3 per half
    for j in range(3):
        x0, y0 = i * 2000.0, j * 2000.0
        shape = Shape.rect(x0, y0, x0 + 2000, y0 + 2000)
        inside = shape.mask(x, y)
        part = Localizations({"x_nm": x[inside], "y_nm": y[inside]}, {})
        result = plugin.run(Context(locs=part, region=shape, region_id=f"{i}{j}"),
                            settings)
        regions.append(Region(f"{i}{j}", dict(result.arrays), dict(result.values),
                              "clusters" if i < 3 else "uniform"))
x_name, grid, names, groups = summarise(regions, names=["g", "lr_nm"],
                                        pool=plugin.pool)
```

**1. The curves.**  For every included ROI, its curves under the chosen
evaluation, as the per-region step kept them; an ROI whose results are out
of date (it moved, its filter changed) is left out, as everywhere.

**2. One x axis.**  The curves are drawn against *x* -- on *auto*, `r_nm`
or another array named like an axis.  Every ROI must have the same x grid.

**3. Groups.**  With *group by* set to a keyword list, the ROIs with each
keyword are a group, and the ROIs with none are a group of their own,
"(none)", drawn last in grey.

**4. The centre.**  Per group and per x, the mean of the ROIs' curves -- or,
for a curve the measurement says how to pool, the curve made from the
counts summed over the group (for Spatial statistics: g, K and L - r).

**5. The spread.**  The standard deviation of the ROIs' curves at each x
(*spread*: SD), or the standard error of their mean (SEM: the SD over the
square root of the number of ROIs).  The SD says how different the ROIs
are; the SEM how well the mean is known, the one to compare two groups by.

```figure Eighteen square ROIs, nine in clustered and nine in uniform data, grouped by a keyword: the clusters' L - r peaks near 100 nm, with a band that says the squares agree; the uniform ones show only their blinking, a small peak at 20 nm. g and L - r are pooled from the pair counts; the bands are the ROIs' SD.
draw(fig, x_name, grid, names, groups, "sd")
fig.set_size_inches(7.0, 2.9)
```

**6. The numbers.**  The text gives, per group, the number of ROIs and the
mean of every value the step gave (`lr_max_r_nm`, `g_max`, ...).

## In detail

**Mean and spread.**  At each x the mean and the standard deviation
(with $n - 1$) are over the ROIs with a finite value there; the SEM is
$\mathrm{SD} / \sqrt{n}$; with fewer than two ROIs the spread is NaN.  A
region's NaN (a ring beyond its window, where nothing is expected) leaves
out that point of that region only.

**Pooling.**  A measurement whose curve is a ratio hands back its numerator
and denominator (`pairs` and `expected` for Spatial statistics), and says
how to make its curves from them (`Plugin.pool`).  The group's arrays are
summed point by point and handed to it; Spatial statistics returns

$$\hat{g}(r) = \frac{\sum_i H_i(r)}{\sum_i E_i(r)} ,$$

$H_i$ the pairs counted in ROI $i$ and $E_i$ those CSR expects there, and
K and L - r from it as in a single run.  This is g of the ROIs taken as one
measurement whose files are the ROIs: exactly what Spatial statistics gives
for several files, and the g of their union when the ROIs do not overlap
and hold equal numbers of localizations.  With very different numbers, a
single window over all of them would assume one density for all, which
they do not have; summing each ROI's own expectation does not.  Curves the
measurement does not pool -- and every curve of a measurement without
counts -- are averaged.

```figure ROIs of unequal size in the uniform half: two of 2 µm and 24 of 400 nm. The mean of their g (grey) counts each ROI once, so the small ROIs, with a few pairs per ring each, make it ragged; the pooled g (black) counts each pair once and is the g of all 26 measured together -- flat at 1 beyond the blinking.
ax = fig.subplots()
fig.set_size_inches(4.4, 3.0)
uneven = []
boxes = [(6000, 0, 2000), (8000, 0, 2000)] + \
    [(6000 + 500 * i, 4000 + 500 * j, 400) for i in range(12) for j in range(2)]
for k, (x0, y0, side) in enumerate(boxes):
    shape = Shape.rect(x0, y0, x0 + side, y0 + side)
    inside = shape.mask(x, y)
    part = Localizations({"x_nm": x[inside], "y_nm": y[inside]}, {})
    result = plugin.run(Context(locs=part, region=shape, region_id=str(k)), settings)
    uneven.append(Region(str(k), dict(result.arrays), dict(result.values)))
_, _, _, (plain,) = summarise(uneven, names=["g"])
_, _, _, (pooled,) = summarise(uneven, names=["g"], pool=plugin.pool)
ax.plot(grid, plain.mean["g"], color="0.6", label="mean of the ROIs' g")
ax.plot(grid, pooled.mean["g"], color="k", label="pooled g")
ax.axhline(1, color="k", linewidth=0.5)
ax.set_ylim(0.4, 2.5)
ax.set_xlabel("r (nm)")
ax.set_ylabel("g(r)")
ax.legend(fontsize=7, frameon=False)
```

**One grid, not interpolation.**  The x arrays of all regions must agree to
a millionth of their span.  A region on another grid is refused with its
number and both grids in the message, rather than interpolated onto the
first: a different grid means the step ran with other settings (another bin
width or range), and then the curves are not one measurement -- counts
summed across rings of different widths are wrong, and a mean of curves
binned differently is not a mean of anything.  Run the evaluation again with
one setting.

**Which curves.**  On *auto*: the pooled curves when the measurement pools,
else every array as long as the x axis, except the x axis and an
envelope's edges (`g_lo`, `lr_nm_hi_L2`).  *Curves* names them otherwise,
comma separated: `g_L2`, `lr_nm_L1xL2` for a second layer or a pair.

**Which results.**  *Results from* on *auto* takes, in a chain, the
chain's own evaluation; else the only one on the ROIs or per cluster that
kept curves; with several, it asks which.

## Parameters

### x
Auto finds `r_nm`; name another array only for a measurement whose x axis
is called something else.

### curves
Empty is right for Spatial statistics: g, K and L - r, pooled.  Name
`pairs` and `expected` to see the counts themselves, or a second layer's
curves (`g_L2`).

### spread
SD to see how much the ROIs differ -- a broad band is heterogeneity, worth
knowing before reading the mean; SEM to judge whether two groups' means
differ.

### group_by
Annotate the ROIs first (the ROI manager's *Annotate*); ROIs without a
keyword in the list form the group "(none)".

## Output

**The figure**: a panel per curve, a line per group with the number of
ROIs and "pooled" in the legend, the spread shaded.

**The text**: the evaluation, the number of regions, the curves and which
are pooled; per group the number of regions and the mean of each value.

**`Result.data`**: `x` and `x_name`, `curves`, and per group `n`, `ids`,
`mean` and `spread` (a curve each), `count` (the regions with a value at each
x), `pooled` and `values`.

A good result: bands narrow enough that the groups' curves separate where
they differ.  A band as wide as the curve's height: the ROIs disagree --
look at them one by one before trusting the mean.
