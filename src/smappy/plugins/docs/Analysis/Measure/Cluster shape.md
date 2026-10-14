---
version: "1"
covers: [smappy.plugins.cluster_shape.cluster_shape, smappy.plugins.cluster_shape.hull_measure, smappy.plugins.cluster_shape.principal_axes, smappy.plugins.cluster_shape.draw, smappy.plugins.cluster_shape.ClusterShape.run]
---

## What it does

Describes the shape of a group of localizations in a few numbers: how many
there are, the area they cover, how far they spread from their centre, how
elongated they are and in which direction, and how densely they are packed.

It is meant for [clusters](plugin:Panels/Glossary) (localizations that lie
closer to each other than to the rest).  After
[Find clusters](plugin:Analysis/Regions/Find clusters) has given every localization a
`cluster_id`, add this plugin to the
[per-region steps](plugin:Panels/Glossary) over `cluster_id` and it runs once
in every cluster, a row of numbers each: the size distribution of
the clusters, whether they are round or stretched, whether the big ones are
also the dense ones.  It runs the same way in every
[ROI](plugin:Panels/Glossary) of the ROI manager, and on the view, where it
measures everything that is shown as one shape.

Each visible [layer](plugin:Panels/Glossary) (one picture of the
localizations with its own filter and colours) is measured on its own, as
the layer shows it: its filter, and merged blinks if the layer merges them.
Merged blinks are usually what is wanted -- every blink once, at its better
position -- since a molecule that blinked in twenty frames would otherwise
weigh twenty times.

The numbers describe the localizations, not the structure under them: the
spread includes the [localization precision](plugin:Panels/Glossary), and
a cluster of a few localizations has a shape mostly by chance.  Compare
clusters of similar size, and leave out the small ones: a filter on
`n_in_cluster` before the per-region steps, or *Filter ROIs* on `n_locs`
for ROIs.

## How it works

```figure-setup
import os, tempfile
import yaml
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.plugins.cluster_shape import cluster_shape, draw

def outline(a, b, angle, cx, cy, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    u, v = a * np.cos(t), b * np.sin(t)
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    return [[float(cx + c * p - s * q), float(cy + s * p + c * q)] for p, q in zip(u, v)]

# a disc, an ellipse four times as long as wide and a thin line, as areas of labels
truth = [("disc", 150, 150, 0, 500), ("ellipse", 240, 60, 30, 1300), ("line", 300, 8, 110, 2100)]
structure = {"elements": [{"polygon": outline(a, b, angle, cx, 500), "density": 0.004}
                          for _, a, b, angle, cx in truth]}
path = os.path.join(tempfile.mkdtemp(), "shapes.yaml")
with open(path, "w") as f:
    yaml.safe_dump(structure, f)
locs = simulate(SimulationSettings(n_frames=3000, seed=2,
                                   structure=StructureSettings(file=path),
                                   labelling=LabellingSettings(efficiency=1.0)))
bright = np.asarray(locs["photons"]) > 200
x, y = np.asarray(locs["x_nm"])[bright], np.asarray(locs["y_nm"])[bright]
shapes = []
for name, a, b, angle, cx in truth:
    near = np.abs(x - cx) < 450
    shapes.append({"name": name, "colour": "#1f77b4", "x": x[near], "y": y[near],
                   "values": cluster_shape(x[near], y[near]), "truth": (a, b, angle, cx)})
```

**1. The centre and the spread.**  The centre is the mean position.  The
[radius of gyration](plugin:Panels/Glossary) is the typical distance of a
localization from it (the root of the mean squared distance): one number
for the size, whatever the shape.

**2. The axes.**  The spread is then split by direction.  The direction in
which the localizations spread most is the **long axis**; at right angles
to it is the short axis.  The plugin gives the standard deviation along
each, the angle of the long axis, and their ratio, the **elongation**: 1
for a round cluster, 4 for one four times as long as wide.  The positions'
full range along each axis is given too, as the extents.

**3. The area.**  The [convex hull](plugin:Panels/Glossary) is the
shape a rubber band would take if stretched around all the localizations:
the smallest outline without dents that contains them.  Its area, and the number of localizations per
µm² of it, the **density**, follow.

**4. In 3D**, if the table has `z_nm`, the hull is taken in three
dimensions for the volume, and the z range is given as the z extent.

```figure Three simulated structures -- a disc, an ellipse four times as long as wide at 30°, and a thin line at 110° -- with their localizations (dots), the convex hull (line), the ellipse at two standard deviations along the measured axes (dashed) with the long axis through it, and the true outline (grey).  The elongation measured comes out near the true one for the disc and the ellipse; for the line the localization precision sets the width, so the elongation is far below the true one of 38.  The hull reaches out to the stray localizations a poor fit put outside the disc.
fig.set_size_inches(7.5, 2.8)
axes = fig.subplots(1, 3)
for i, (ax, shape) in enumerate(zip(axes, shapes)):
    a, b, angle, cx = shape["truth"]
    true = np.array(outline(a, b, angle, cx, 500) + [outline(a, b, angle, cx, 500)[0]])
    ax.plot(true[:, 0], true[:, 1], color="0.6", linewidth=2.5, zorder=0)
    draw(ax, [shape])
    v = shape["values"]
    ax.set_title(f"{shape['name']}: elongation {v['elongation']:.2g} (true {a / b:.2g})\n"
                 f"angle {v['angle_deg']:.0f}°, Rg {v['rg_nm']:.0f} nm", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    if i:
        ax.set_ylabel("")
```

## In detail

For the $n$ positions $(x_i, y_i)$ of a region, with the centre
$(\bar{x}, \bar{y})$ their mean:

**Radius of gyration.**
$R_g = \sqrt{\frac{1}{n} \sum_i \left( (x_i - \bar{x})^2 + (y_i - \bar{y})^2 \right)}$,
in x and y only, also in 3D.  For a uniformly filled disc of radius $R$ it
is $R/\sqrt{2}$.

**Axes.**  The covariance matrix of $x$ and $y$ (with $n - 1$ in the
denominator, as MATLAB's `cov`) has eigenvalues
$\lambda_1 \geq \lambda_2$; the standard deviations are
$\sigma_{long} = \sqrt{\lambda_1}$ and $\sigma_{short} = \sqrt{\lambda_2}$,
and the angle is that of $\lambda_1$'s eigenvector, from the x axis towards
the y axis, between 0° and 180°.  The elongation is
$\sigma_{long} / \sigma_{short}$.  For a uniformly filled ellipse with
semi-axes $a$ and $b$ the standard deviations are $a/2$ and $b/2$, so the
elongation is $a/b$; for a line of length $L$ the long one is
$L/\sqrt{12}$.  The extents are the range of the positions after rotating
them onto the two axes.

**Hull.**  The convex hull is computed by qhull (`scipy.spatial.ConvexHull`)
in x and y for the area, and in x, y and z for the volume.  The density is
$n$ divided by the area, per µm².  The hull of uniformly scattered points
lies inside the true outline, so the area is too small and the density
too large: for points scattered evenly in a disc, the hull covers about
98 % of it with 3000 points, 85 % with 100 and half with a dozen.  Compare
densities of clusters of similar size.  A concave cluster (a crescent, a ring) is
covered by its hull including the empty part, and its density is
underestimated.  And the hull runs through the outermost
localizations, so a single stray one -- a poor fit, a neighbour that DBSCAN
counted in -- enlarges it; the standard deviations and $R_g$ move far less.

**Precision.**  The measured variances are the cluster's own plus the
localization precision's: $\sigma_{measured}^2 = \sigma_{true}^2 +
\sigma_{loc}^2$.  Nothing is subtracted.  A cluster no larger than the
precision is measured as a round blob of about the precision's size,
whatever its true shape.

**Few points.**  With no localizations only `n_locs` (0) is given, the
rest NaN.  With one, the centre and $R_g = 0$.  The standard deviations,
the angle and the extents need two.  The area needs three points that are
not on one line, the volume four not in one plane; otherwise it is 0, and
the density -- with nothing to divide by -- NaN.  An elongation with a
short standard deviation of 0 (points on one line: below a millionth of
the long one, to absorb rounding) is NaN too.

**Names.**  The values in the region table are `n_locs`, `centre_x_nm`,
`centre_y_nm`, `area_nm2`, `density_per_um2`, `rg_nm`, `std_long_nm`,
`std_short_nm`, `angle_deg`, `elongation`, `extent_long_nm`,
`extent_short_nm`, and in 3D `centre_z_nm`, `z_extent_nm` and
`volume_nm3`; for the second and further layers with `_L2`, `_L3`, ...
appended.

## Parameters

### use_z
Untick it for a 3D table whose clusters should be compared by their
footprint only, or when z is too imprecise for a volume to mean anything.

## Output

The figure shows each layer's localizations, the convex hull, and the
ellipse at two standard deviations along the measured axes with the long
axis drawn through it.  The text gives, per layer, the number of
localizations, the area, the density, $R_g$, the two standard deviations
and the angle, the elongation, and in 3D the z extent and the volume.  Per
region, the same numbers are a row of the region table.

A round cluster has an elongation near 1, but not exactly: localizations
scattered evenly in a disc give a median elongation of about 1.2 with 30 of
them, 1.1 with 100 and 1.03 with 1000.  An elongation that falls as a cluster gets bigger is
usually that noise; one that stays the same is a shape.

## Differences from SMAP

Based on the shape half of SMAP's `Analyze/cluster/ClusterStatistics`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), which, despite its
description, computed no pair correlation.  The main changes:

* SMAP loops over a `clusterindex`, `groupindex` or `tid` column itself;
  here the per-region steps run the plugin once per region, over any id
  column or the ROIs.
* SMAP wrote the standard deviations, angle and extents onto the
  localizations (`c_stds`, `c_stdl`, `c_angle`, `c_extl`, `c_exts`) and the
  area and volume into a separate table only with *advanced analysis*
  ticked; here every value goes into the region table, and *Regions to
  column* writes any of them onto the localizations.
* SMAP also gave the area and volume of MATLAB's `boundary` (an alpha shape
  with a shrink factor of 0.5); it is left out, since its area depends on
  a shrink factor with no physical meaning.
* The angle is in degrees rather than radians; the radius of gyration, the
  elongation and the density are new.
* SMAP's *min number of locs* and *skip first* are gone: a filter on
  `n_in_cluster` decides which clusters count, *Filter ROIs* on `n_locs`
  which ROIs.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
* Barber CB, Dobkin DP, Huhdanpaa H. The Quickhull algorithm for convex
  hulls. *ACM Trans Math Softw* 22, 469 (1996).
  [doi:10.1145/235815.235821](https://doi.org/10.1145/235815.235821) -- the
  convex hull (qhull).
