---
version: "2"
covers: [smappy.plugins.neighbourhood.count_neighbours, smappy.plugins.neighbourhood.neighbour_statistic, smappy.plugins.neighbourhood.voronoi_density, smappy.plugins.neighbourhood.voronoi_areas, smappy.plugins.neighbourhood.neighbourhood, smappy.plugins.neighbourhood.express, smappy.plugins.spatial_statistics.window_coverage, smappy.plugins.neighbourhood.Neighbourhood.run]
---

## What it does

Writes one number per localization that describes what is around it, as a
new column of the table.  The column can then be filtered on (keep the
localizations in dense places, drop the isolated ones), coloured by in the
Render tab, or measured like any other.  It segments nothing: finding
clusters is [Find clusters](plugin:Analysis/Regions/Find clusters)'s job.

Three *quantities*:

* **Neighbours within r** -- how many localizations lie within a radius
  *r*: the [local density](plugin:Panels/Glossary), in localizations per circle.  In 3D the circle is
  a cylinder, and the neighbours may be weighted by a Gaussian of their
  distance instead of counted in a sharp circle.  The count can be
  *expressed as* the count itself, as the **local L(r)**, or **relative to
  the mean density** (below).
* **A statistic of a column** -- the mean, median, minimum, maximum,
  standard deviation or a quantile of any column (photons, `z_nm`, the
  frame) over the neighbours within *r*: a smoothing of that column in
  space, for a picture coloured by it.
* **Voronoi density** -- one over the area of the localization's Voronoi
  cell, the region of the plane closer to it than to any other localization
  ([Voronoi tessellation](plugin:Panels/Glossary), as in SR-Tesseler,
  [Levet et al. 2015](https://doi.org/10.1038/nmeth.3579)).  It needs no
  radius: the cell is small where the localizations are dense and large
  where they are sparse.  It is of one layer's own localizations, in x
  and y only.  Since only the nearest neighbours shape a cell, one
  molecule's repeated localizations look dense: use it on a layer that
  merges blinks.

Two layers say what is counted: *around*, the layer whose localizations
get the value, and *neighbours from*, the layer whose localizations are
counted.  Each is read as that layer shows it, with its filter and with or
without [merged blinks](plugin:Panels/Glossary) -- so what is counted is
decided where it always is, in the layer.  The same layer twice is the
ordinary local density.  Two layers -- two colour channels -- give a
**cross-density**: how many of channel B lie around each localization of
channel A, which is a measure of colocalization.

**Local L and relative density.**  A count of 12 means little until it is
compared with what the mean density would put in the circle.  *Relative to
the mean density* divides by that: 1 in a uniform field, 5 where the
localizations are five times as dense as on average.  *Local L(r)* is the
radius a circle would need to hold that many neighbours at the mean
density ([Getis and Franklin 1987](https://doi.org/10.2307/1938452)): $r$
in a uniform field, more in a cluster -- the per-localization version of
[Spatial statistics](plugin:Analysis/Measure/Spatial statistics)' L(r),
whose mean over the localizations it is.  Both are corrected at the edge of
the window, so a localization at the edge of the cell is not called sparse
for having half a circle of data around it.

**DBSCAN thresholds the same quantity.**  A localization is a core point of
[DBSCAN](plugin:Analysis/Regions/Find clusters) when it has at least
*neighbours* within *radius* -- the count here, thresholded.  So a filter on
the count (or on local L, which is the count on a scale that does not
depend on the density of the whole data set) shows which localizations
DBSCAN would call core before it joins them into clusters: Getis and
Franklin's clustering -- threshold local L and join what is dense -- is
DBSCAN, and is not a separate algorithm here.

Use the count when the structures have a size you know (a radius about
the size of a cluster); use the Voronoi density when they do not, or as
the first step of a Voronoi clustering.  Neither says anything reliable
within a radius of the edge of the field, where part of the circle or the
cell is outside the data.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import SimulationSettings, StructureSettings
from smappy.plugins.neighbourhood import (count_neighbours, neighbour_statistic,
                                          voronoi_density)
locs = simulate(SimulationSettings(n_frames=3000, seed=5,
                                   structure=StructureSettings(preset="demo")))
keep = np.asarray(locs["photons"]) > 200
xy = np.column_stack((locs["x_nm"], locs["y_nm"]))[keep]
dye = np.asarray(locs["dye"])[keep]
n = len(xy)
density = count_neighbours(xy, xy, 50.0, same=np.arange(n))
voronoi = voronoi_density(xy)
ring, rest = dye == 1, dye == 2
cross = count_neighbours(xy[ring], xy[rest], 100.0)
```

**1. The two tables.**  The localizations of the *around* layer that pass
its filter, and those of the *neighbours from* layer that pass its own.  On
a layer that merges blinks these are the blinks.  Each file of the session
is done on its own: a localization of one file is never another file's
neighbour.

**2. The neighbours.**  For every localization around, the neighbours
within the *radius*, and in 3D also within the *z half-height* above or
below it.  A localization is never its own neighbour, so one with nobody
near has 0.

**3. The value.**  Their number; or, with *Gaussian weight*, the sum of
$e^{-d^2/2r^2}$ over them, where $d$ is the distance and $r$ the radius,
so a neighbour close by counts nearly 1 and one at $2r$ counts 0.14; or the
chosen statistic of the column over them, the localization's own value
included; or, for the Voronoi density, one over the area of its cell.

**4. The column.**  The value goes onto the localizations of the *around*
layer that passed its filter, and every other localization gets NaN (no
value), so a filter on the new column also drops them.  A blink's value
goes onto each of its localizations.

```figure The local density brings out the structures: the ring and the lines have tens of neighbours within 50 nm, the scattered molecules a few (left). The Voronoi density (right) sees only the nearest neighbours, so a lone molecule's own repeated localizations make its cells as small as the ring's: merge blinks before using it.
fig.set_size_inches(7.0, 3.3)
left, right = fig.subplots(1, 2)
order = np.argsort(density)
dots = left.scatter(xy[order, 0], xy[order, 1], c=density[order], s=0.5,
                    cmap="viridis", vmax=np.percentile(density, 99))
fig.colorbar(dots, ax=left, shrink=0.8, label="neighbours within 50 nm")
v = np.log10(np.where(np.isfinite(voronoi), voronoi, np.nan))
order = np.argsort(np.nan_to_num(v, nan=-1))
dots = right.scatter(xy[order, 0], xy[order, 1], c=v[order], s=0.5, cmap="viridis",
                     vmin=np.nanpercentile(v, 1), vmax=np.nanpercentile(v, 99))
fig.colorbar(dots, ax=right, shrink=0.8, label="log10 Voronoi density (/um$^2$)")
for ax in (left, right):
    ax.set_aspect("equal")
    ax.axis("off")
```

```figure A cross-density: each localization of the ring (dye 1) coloured by the dye-2 localizations within 100 nm. It is high only where the two lines cross the ring, where the two labels are close.
fig.set_size_inches(4.2, 3.6)
ax = fig.subplots()
ax.scatter(xy[rest, 0], xy[rest, 1], s=0.3, color="0.85")
order = np.argsort(cross)
dots = ax.scatter(xy[ring][order, 0], xy[ring][order, 1], c=cross[order], s=1.5,
                  cmap="magma_r")
fig.colorbar(dots, ax=ax, shrink=0.8, label="dye-2 neighbours within 100 nm")
ax.set_aspect("equal")
ax.axis("off")
```

## In detail

**Counting.**  For an around localization at $(x_i, y_i, z_i)$ the
count is the number of neighbours $j$ with

$$(x_j - x_i)^2 + (y_j - y_i)^2 \leq r^2 \quad \mathrm{and} \quad |z_j - z_i| \leq h$$

(the second only in 3D, with $h$ the *z half-height*), $j \neq i$.  In a
uniform field of $\rho$ localizations per nm² the count is Poisson
distributed with mean $\pi r^2 \rho$, and $2 \pi r^2 h \rho_3$ in a
cylinder.  With *Gaussian weight* each neighbour contributes

$$w_{ij} = \exp\left( -\frac{d_{ij}^2}{2 r^2} - \frac{(z_j - z_i)^2}{2 h^2} \right)$$

out to $3r$ laterally and $2.5h$ in z, as in SMAP; the mean in a uniform
field is $2 \pi r^2 \rho$ (times $(1 - e^{-4.5})$ for the cut).

```figure In a uniform field the counts follow the Poisson distribution of mean pi r^2 rho (line), here 20 neighbours at r = 80 nm.
fig.set_size_inches(4.2, 3.0)
ax = fig.subplots()
rng = np.random.default_rng(0)
side, m, r = 20000.0, 400000, 80.0
u = rng.uniform(0, side, (m, 2))
c = count_neighbours(u, u, r, same=np.arange(m))
inner = ((u > r) & (u < side - r)).all(axis=1)
mean = (m - 1) / side ** 2 * np.pi * r ** 2
k = np.arange(0, 45)
from math import lgamma
pmf = np.exp(k * np.log(mean) - mean - np.array([lgamma(i + 1) for i in k]))
ax.hist(c[inner], bins=np.arange(-0.5, 45), density=True, color="0.7")
ax.plot(k, pmf, color="C3")
ax.set_xlabel("neighbours within 80 nm")
ax.set_ylabel("fraction")
```

**Expressed.**  With $N$ localizations of the *neighbours* layer in the
file's window $W$ (area $|W|$; $N - 1$ others for a localization that is
itself among them), CSR expects $\frac{N - 1}{|W|} a_i$ neighbours around
localization $i$, where $a_i$ is the area of its circle inside the window
-- $\pi r^2$ away from the edge, half that at a straight edge.  The
relative density is $n_i$ over that, and the local L is

$$L_i(r) = r \sqrt{\frac{n_i \, |W|}{(N - 1) \, a_i}} = \sqrt{\frac{|W| \, n_i}{\pi (N - 1)}} \quad \mathrm{away\ from\ the\ edge} .$$

With *Gaussian weight* $a_i$ is the integral of the weight over the part of
its 3-sigma circle inside the window, and local L is $r$ times the square
root of the relative density.  The window is Spatial statistics': the drawn
ROI, or, without one, the occupied area (the localizations of both layers
binned at 200 nm, closed) or the bounding box, per file (*window*).  $a_i$
is the whole circle's for a localization whose circle lies inside the
window, and otherwise integrated over the window's pixels -- exactly across
the pixel rows, by Gauss-Legendre along x.  Expressed counts are in x and
y only; *in 3D* is not used.  The mean of local L is a little below $r$ in
a uniform field, since a square root of a count scatters below the square
root of its mean.

```figure Clusters in a uniform field: local L at r = 60 nm is about 60 nm for the scattered localizations -- at the window's edge, where half the circle is outside, as well as inside -- and several times that in the clusters.
fig.set_size_inches(4.4, 3.0)
ax = fig.subplots()
from smappy.plugins.neighbourhood import express
from smappy.plugins.spatial_statistics import window_coverage, window_of_points
rng = np.random.default_rng(1)
field = rng.uniform(0, 10000, (80000, 2))
centres = rng.uniform(500, 9500, (40, 2))
blobs = (centres[:, None, :] + rng.normal(0, 30, (40, 200, 2))).reshape(-1, 2)
pts = np.vstack([field, blobs])
counts = count_neighbours(pts, pts, 60.0, same=np.arange(len(pts)))
w = window_of_points(pts[:, 0], pts[:, 1], "box")
cover = window_coverage(w, pts[:, 0], pts[:, 1], 60.0)
local = express(counts, cover, len(pts) - 1, w.area, 60.0, "local_l")
edge = ((pts < 30) | (pts > 9970)).any(axis=1)
bins_ = np.linspace(0, 300, 61)
ax.hist(local[:80000][~edge[:80000]], bins_, color="0.7", density=True, label="scattered")
ax.hist(local[:80000][edge[:80000]], bins_, color="C0", alpha=0.6, density=True,
        label="scattered, within 30 nm of the edge")
ax.hist(local[80000:], bins_, color="C3", alpha=0.6, density=True, label="in clusters")
ax.axvline(60, color="k", linewidth=0.6)
ax.set_xlabel("local L(r = 60 nm) (nm)")
ax.set_ylabel("fraction per nm")
ax.legend(fontsize=7, frameon=False)
```

**The statistic** is over the neighbours with a finite value of the
column, the localization itself among them (this is a smoothing filter: a
localization alone in its circle keeps its own value).  A quantile $q$ is
linear between the order statistics, as numpy's: with $n$ values sorted,
the value at position $q(n - 1)$.

**Voronoi density.**  The cells come from the Delaunay triangulation (each
cell's corners are the centres of the circles through the triangles that
meet at its localization).  The density is $k / A$ per µm², with $A$ the
cell's area and $k$ the number of localizations at exactly that position
(they share one cell).  A cell at the edge of the field is open, with no
area, and its localization gets NaN.  In a uniform field the mean area of
a cell is $1/\rho$, so the harmonic mean of the densities is $\rho$; a
single density scatters widely about it (a cell's area varies by about
50 %), which is why it is mostly used through a threshold or after a
smoothing.

For millions of localizations the field is cut into tiles of about 50,000,
each triangulated with a margin around it.  A cell is kept only when every
circle through its triangles is empty of the field's other localizations,
which proves the triangles are those of the whole field, and is otherwise
done again with a wider margin.  The answer is the same as from one
triangulation of everything.

**Speed.**  The neighbours are found by sorting the localizations into
square cells one radius wide, so that each localization tests only the
nine cells around it, in compiled code on every core.  5 million
localizations over 40 µm at *radius* 50 nm take about 3 s (SMAP's sweep
along x alone tests the whole strip $2r$ wide and as tall as the field:
23 s for the same data); *Gaussian weight* about 10 s; the Voronoi density
about a minute.  A run expected to take more than 20 s asks first.  The
panel shows the percentage done, and Stop ends the run and leaves the
table as it was.

**Merged blinks.**  The column is kept as a recipe with the rule *mean*:
when blinks are merged again, a blink takes the mean of its localizations'
values (ignoring NaN), which is the blink's own value when it was computed
on blinks.

## Parameters

### around
The layer whose localizations get the value.  Set its filter first: only
the localizations it keeps get a value; the rest get NaN.

### neighbours_from
The same layer as *around* for the local density.  Another layer, set to
show another channel, gives the cross-density.

### radius_nm
About the size of the structure to tell apart from its surroundings: a
cluster's radius, or a few times the localization precision.  Too small
and most localizations have 0 or 1 neighbours, too large and every
cluster blurs into its surroundings.  50 nm suits clusters of proteins;
the counts grow with $r^2$, so compare counts only at the same radius.

### use_z
Off: the neighbours are counted in a circle whatever their z, as if the
data were 2D.  Only applies when both layers have `z_nm`.

### z_half_nm
Two to three times the axial precision, or the thickness of the structure;
the default 100 nm suits astigmatic 3D data.  SMAP's 12 nm laterally and
24 nm in z were too small for a default: at typical densities they leave
most localizations with no neighbour.

### gaussian
Smoother than a sharp circle -- a neighbour just inside and one just
outside the radius no longer differ by a whole count -- at about 9 times
the cost, since it looks three radii out.

### express_as
*Local L* to compare data sets of different densities on one scale, and
for a threshold that means the same in each; *relative* for "how many times
the average"; the plain count when the radius and density are fixed, or
in 3D.

### window
As in Spatial statistics: the occupied area for a cell, the bounding box
for data that fill a rectangle.  A drawn ROI is the window when there is
one.

### statistic
*Median* is robust to the few outliers a column like photons has; *std*
shows where a value varies, such as z across the edge of a structure.

### name
Left empty, the name says what was computed: `density_r50`,
`density_gauss_r50`, `photons_median_r50`, `voronoi_density`, with
`_from2` when the neighbours came from layer 2.  Running again with the
same name replaces the column; a column of the localizations' own (such as
`photons`) is never overwritten.

## Output

The new column on the localizations (`local_l_r50`, `density_rel_r50`
when expressed), and a figure: the *around*
localizations coloured by the value, and its histogram.  The result's line
says how many localizations got a value and their median.

A count of 0 on most localizations means the radius is small for the
density; a histogram with one broad peak and no tail is a uniform
distribution with no clusters to find.  Clustered data shows two
populations: a peak near 0 (the localizations between clusters) and a
broad one at the clusters' density.

## Differences from SMAP

Based on SMAP's `density_calculator`, `SpatialFilterLocs` and the density
of `ClusterVoronoi` ([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)),
in one plugin.  The main changes:

* A localization is not its own neighbour.  SMAP counts it, so an isolated
  localization has 1 there and 0 here.  The statistic keeps it, as SMAP's
  does.
* In 3D the region is a cylinder; SMAP's is an ellipsoid with the same two
  half-axes, which holds two thirds as many neighbours.
* SMAP never counted the localization with the largest x as anyone's
  neighbour (its sweep stops one short of the end).
* Voronoi cells open at the edge of the field get NaN; SMAP gave them 0,
  which reads as "empty" rather than "unknown".  SMAP also rounded the
  positions to whole nanometres first, which merged localizations closer
  than that into one cell; here only exactly equal positions share one.
  The density is per µm² rather than per nm².
* *Express as* local L or relative density is new; SMAP gave the count
  only.
* The two layers and every file are kept apart as they are in the layers;
  SMAP had its own choices of layers, channels and files in the dialog.
  SpatialFilterLocs read layer 1 only.

## References

* Levet F, Hosy E, Kechkar A, Butler C, Beghin A, Choquet D, Sibarita JB.
  SR-Tesseler: a method to segment and quantify localization-based
  super-resolution microscopy data. *Nat Methods* 12, 1065-1071 (2015).
  [doi:10.1038/nmeth.3579](https://doi.org/10.1038/nmeth.3579) -- the
  Voronoi density.
* Andronov L, Lutz Y, Vonesch JL, Klaholz BP. SharpViSu: integrated
  analysis and segmentation of super-resolution microscopy data.
  *Bioinformatics* 32, 2239-2241 (2016).
  [doi:10.1093/bioinformatics/btw123](https://doi.org/10.1093/bioinformatics/btw123)
  -- the Voronoi code SMAP used.
* Getis A, Franklin J. Second-order neighborhood analysis of mapped point
  patterns. *Ecology* 68, 473-477 (1987).
  [doi:10.2307/1938452](https://doi.org/10.2307/1938452) -- the local L.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
