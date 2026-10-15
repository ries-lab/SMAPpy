---
version: "1"
covers: [smappy.plugins.clusters.dbscan, smappy.plugins.clusters.auto_eps, smappy.plugins.clusters.voronoi_clusters, smappy.plugins.clusters.hdbscan_clusters, smappy.plugins.clusters.by_size, smappy.plugins.clusters.with_clusters, smappy.plugins.clusters.FindClusters.run, smappy.plugins.clusters.FindClusters.preflight]
---

## What it does

Finds clusters -- places where localizations lie closer together than
elsewhere -- and names them.  Nothing has to be said about how many clusters
there are; a cluster may have any shape, and what is not crowded belongs to
none.  Three *algorithms*, each with its own settings, of which the form
shows only the chosen one's:

* **DBSCAN** ([DBSCAN](plugin:Panels/Glossary), density-based spatial
  clustering of applications with noise, Ester et al. 1996): a cluster is a
  run of localizations each with enough neighbours within a *radius*.  The
  default, and the one to use when the size of what you look for is known.
* **Voronoi** (SR-Tesseler,
  [Levet et al. 2015](https://doi.org/10.1038/nmeth.3579)): a localization
  is dense when its [Voronoi](plugin:Panels/Glossary) cell -- the part of the
  plane closer to it than to any other localization -- is small, and dense
  localizations whose cells touch are one cluster.  It needs no radius,
  only how much denser than the field's average a cluster must be.
* **HDBSCAN** (hierarchical DBSCAN,
  [Campello et al. 2013](https://doi.org/10.1007/978-3-642-37456-2_14)):
  DBSCAN at every radius at once, keeping the clusters that persist over
  the widest range of radii.  It finds clusters of different densities in
  one field, which a single radius cannot.  It needs the optional package
  hdbscan (`pip install "smappy-smlm[clusters]"`), and cannot be stopped
  once it runs (see below).

What it gives is a **name for every cluster**, written onto the table:
`cluster_id` (1 for the largest cluster, 2 for the next, ..., 0 for a
localization in no cluster) and `n_in_cluster` (how many localizations its
cluster has).  With them

* a filter on `n_in_cluster` keeps the large clusters and drops the specks;
* a **block** of a [chain](plugin:Panels/Glossary) (plugins run one after
  another as one), or the [per-region steps](plugin:Panels/Regions tab)
  set to *for each* `cluster_id`, runs any measurement once in every cluster
  -- [Localization Statistics](plugin:Analysis/Measure/Localization Statistics)
  or [Cluster shape](plugin:Analysis/Measure/Cluster shape), say -- and
  keeps a row of numbers per cluster, which the block's *write values* or
  [Regions to column](plugin:Analysis/Regions/Regions to column) puts back
  onto the localizations, to filter or colour by;
* [Column to ROIs](plugin:Analysis/Regions/Column to ROIs) makes an
  [ROI](plugin:Panels/Glossary) (region of interest) of each cluster, to look at, review and annotate in the
  [ROI manager](plugin:Panels/ROI manager).

It clusters the localizations of the layer as it is shown: its
[filter](plugin:Panels/Glossary) (the lowest and highest values of columns a
localization must lie between), its ROI and, on a layer that merges blinks,
its blinks -- [merge blinks](plugin:Panels/Glossary), also called grouping or
linking, combines the localizations in consecutive frames that come from one
blinking event into one.  Use it on structures that are
separate from each other -- pores, clusters of a receptor, vesicles.  On a
continuous structure (a filament network) every localization is crowded
and it joins everything into one.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.plugins.clusters import dbscan, sizes_of, voronoi_clusters
locs = simulate(SimulationSettings(n_frames=3000, seed=3,
                                   structure=StructureSettings(preset="npc"),
                                   labelling=LabellingSettings(efficiency=0.6)))
bright = np.asarray(locs["photons"]) > 200
xy = np.column_stack((locs["x_nm"], locs["y_nm"]))[bright]
copy = np.asarray(locs["copy"])[bright]
labels = dbscan(xy, 50, 10)
```

### DBSCAN

**1. Neighbours.**  For every localization the others within the *radius*
(called *eps*) are counted.  One with at least *neighbours* (called *k*)
others that close is a **core** point: it sits inside a cluster.

**2. Clusters.**  Core points within the radius of each other belong to the
same cluster, and so do their core neighbours, and theirs: a cluster is
every core point that can be reached from another in steps no longer than
the radius.  Its shape does not matter.

**3. Edges and noise.**  A localization that is not a core point but lies
within the radius of one is on a cluster's edge (a **border** point) and
joins the cluster of the nearest core point.  The rest are **noise**:
`cluster_id` 0.

**4. Names.**  The clusters are numbered by size, the largest 1.
`n_in_cluster` is written beside `cluster_id`; both are kept when blinks
are merged, a blink taking its cluster.

```figure 141 simulated nuclear pores (left; a colour per pore) and the clusters DBSCAN finds with a 50 nm radius and 10 neighbours (right; a colour per cluster, noise grey): every cluster holds one pore, and one pore came out as two clusters.
fig.set_size_inches(8, 4)
left, right = fig.subplots(1, 2, sharex=True, sharey=True)
left.scatter(xy[:, 0], xy[:, 1], s=0.5, c=copy % 20, cmap="tab20", linewidths=0)
left.set_title(f"{len(np.unique(copy))} pores, the truth", fontsize=9)
noise = labels == 0
right.scatter(xy[noise, 0], xy[noise, 1], s=0.5, c="0.75", linewidths=0)
right.scatter(xy[~noise, 0], xy[~noise, 1], s=0.5, c=labels[~noise] % 20,
              cmap="tab20", linewidths=0)
right.set_title(f"{labels.max()} clusters", fontsize=9)
for ax in (left, right):
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
left.set_ylabel("y (nm)")
```

### Voronoi

**1. Cells.**  The plane is divided into one cell per localization, the
part of it nearer that localization than any other.  Where localizations
crowd, the cells are small.  Localizations at exactly the same position
share a cell.

**2. Dense cells.**  A cell's density is the localizations in it over its
area.  It is dense when that is above *density factor* times the field's
mean density: all the localizations over the area of the smallest convex
outline around them.  A cell at the edge of the field is open, has no area,
and is never dense.

**3. Clusters.**  Dense cells that share an edge are one cluster.  A
cluster with fewer than *minimum size* localizations is dropped.

```figure The same pores with 2000 scattered localizations added. Each localization coloured by the density of its cell over the field's mean (left): the pores' cells are far denser than the mean, the scattered ones below it. With a density factor of 2 and a minimum size of 10 (right), the pores come out as clusters without a radius having been chosen, and the scattered localizations as none.
from smappy.plugins.neighbourhood import voronoi_density
rng = np.random.default_rng(1)
xy = np.vstack([xy, rng.uniform(xy.min(axis=0), xy.max(axis=0), (2000, 2))])
density = voronoi_density(xy)
vlabels, mean = voronoi_clusters(xy, 2.0, 10)
fig.set_size_inches(8, 4)
left, right = fig.subplots(1, 2, sharex=True, sharey=True)
order = np.argsort(np.nan_to_num(density))
dots = left.scatter(xy[order, 0], xy[order, 1], s=0.5, linewidths=0,
                    c=np.log10(np.nan_to_num(density[order], nan=mean) / mean),
                    cmap="viridis", vmin=-1, vmax=2)
fig.colorbar(dots, ax=left, shrink=0.7, label="log10 density / mean")
noise = vlabels == 0
right.scatter(xy[noise, 0], xy[noise, 1], s=0.5, c="0.75", linewidths=0)
right.scatter(xy[~noise, 0], xy[~noise, 1], s=0.5, c=vlabels[~noise] % 20,
              cmap="tab20", linewidths=0)
right.set_title(f"{vlabels.max()} clusters", fontsize=9)
for ax in (left, right):
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
left.set_ylabel("y (nm)")
```

### HDBSCAN

**1. Core distances.**  For every localization, the distance to its
*neighbours*-th nearest neighbour: small inside a cluster, large in the
background.

**2. A tree of clusters.**  Two localizations are as far apart as the
larger of their distance and their core distances (the *mutual
reachability*).  Joining the localizations from the closest pair on gives
a tree: at every radius, the clusters DBSCAN would find with it.

**3. The lasting clusters.**  Of the clusters in that tree with at least
*minimum size* localizations, those that persist over the widest range of
radii are kept; the rest of the localizations are noise.  The hdbscan
package does all three.

## In detail

### DBSCAN

**Core.**  With $N_\varepsilon(i)$ the number of localizations within
$\varepsilon$ of localization $i$, itself included, $i$ is a core point when

$$N_\varepsilon(i) \geq k + 1 .$$

**The automatic radius.**  A radius of 0 asks for SMAP's: the radius of a
ball that would hold $k$ localizations at the mean density of the field,

$$\varepsilon = \left( \frac{V\, k\, \Gamma(d/2 + 1)}{m\, \pi^{d/2}} \right)^{1/d} ,$$

with $V$ the volume (area in 2D) of the localizations' bounding box, $m$
their number and $d$ the dimension.  It is fair for a field evenly covered;
for a cell in a mostly empty field it is far too large.

**3D.**  With *in 3D* ticked and `z_nm` in the table, distances are measured
in x, y and z.

**How it is computed.**  The localizations are sorted into square cells
$\varepsilon$ wide, so each one looks for its neighbours only in the nine
cells around it, on every processor core -- the search of
[Neighbourhood](plugin:Analysis/Process/Neighbourhood).  Core points within
$\varepsilon$ of each other are joined into clusters while the cells are
scanned again, without listing the pairs first.  Two million localizations
take about 2 s.  Each file of the session is clustered on its own, and the
numbers of the second file's clusters follow the first's.  The panel shows
the percentage done, and Stop ends the run and leaves the table as it
was.

### Voronoi

With $A_i$ the area of cell $i$, $n_i$ the localizations in it and $N$
the localizations of the file over the area $A$ of their convex hull, cell
$i$ is dense when

$$\frac{n_i}{A_i} > \delta \, \frac{N}{A} ,$$

$\delta$ the *density factor*.  Cells vary in size even where nothing
is clustered: in a uniform field a factor of 1 calls 56% of the cells
dense, 2 calls 16% and 3 calls 6%, and those cells, where they touch, make
small clusters of the background -- which is what *minimum size* is for.  Cells touch when they share an edge -- when their
localizations are neighbours in the Delaunay triangulation, which is how
the touching cells are found.  The triangulation is computed in tiles on
every core and checked to be the one of the whole field, as in
[Neighbourhood](plugin:Analysis/Process/Neighbourhood); it takes about 20
µs a localization, so a run of more than about a million asks first.

The mean density is that of the convex hull of the localizations, so a
field with large empty parts (outside a cell) has a low mean, and more of
it is dense than in a field evenly covered.  An ROI around the cell, or a
higher factor, corrects for it.  SR-Tesseler, where this criterion comes
from, uses the same multiple of the mean density; ClusterViSu (the
clustering of SharpViSu,
[Andronov et al. 2016](https://doi.org/10.1038/srep24084)) sets the
threshold instead where the data's distribution of cell areas crosses that
of a random one, from simulations, which is not done here.

### HDBSCAN

The clusters are the hdbscan package's
([McInnes et al. 2017](https://doi.org/10.21105/joss.00205)), with
*minimum size* as its `min_cluster_size` and *neighbours* as its
`min_samples` (auto: the same as the minimum size), on x and y, or x, y and
z with *in 3D*.  Its labels are renumbered by size like the others'.

The tree of step 2 is a minimum spanning tree of the mutual reachability.
The package builds it by Borůvka's algorithm on a k-d tree, which looks only
at nearby localizations: 2.5 s for 100,000 localizations and about a minute
for a million.  scikit-learn's HDBSCAN builds it by Prim's algorithm, every
localization against every other: 48 s for 100,000 and hours for a million,
which is why it is not used.  The exact tree is asked for
(`approx_min_span_tree=False`, no slower here).  Many mutual reachabilities
are equal, so several trees are equally minimal, and the two libraries
break the ties differently: on the same data their clusters agree almost
but not quite (an adjusted Rand index of 0.99).

It is one computation that Stop cannot end early, and a run expected to
take more than 20 s -- about 400,000 localizations -- asks first.  Merge
blinks, or run it in an ROI, for a shorter run.

**When to prefer HDBSCAN.**  When the clusters in one field have very
different densities -- a tight cluster beside a loose one -- no single
DBSCAN radius fits both, and HDBSCAN finds each.  It also needs no radius
when nothing is known about the clusters' size.  **When not.**  When a
cluster is defined physically -- localizations within a known distance,
such as one protein complex -- DBSCAN's radius says exactly that, and
HDBSCAN's answer depends on what else is in the field.  HDBSCAN also tends
to give scattered localizations near a cluster to it: in a test with two
clusters and 60 scattered localizations it put about a quarter of the
scattered ones into the clusters.

## Parameters

### dbscan.radius_nm
About the distance between neighbouring localizations inside a structure,
not its size: a few times the [localization precision](plugin:Panels/Glossary)
(also called the CRLB: how far a fitted position may lie from the true one),
30 to 80 nm for
typical data.  Too small and clusters break into pieces; too large and
neighbouring structures merge.

### dbscan.neighbours
Larger values ask for denser clusters and leave more of a cluster's edge as
noise.  Raise it with the number of localizations per structure; 5 to 20 is
usual.

### dbscan.use_z
Off clusters by x and y only, which joins structures that are above each
other.

### voronoi.factor
Lower finds fainter clusters and calls more of the background dense, where
random crowding of the background then forms clusters of a few
localizations; *minimum size* removes those.  Higher keeps only the cores
of the clusters.  2 to 3 suits clusters that stand well out of a
background.

### voronoi.min_size
Above the number of localizations random crowding of the background
gathers -- with merged blinks 5 to 10, without them higher, since one
molecule's repeated localizations are dense among themselves.

### hdbscan.min_cluster_size
The smallest structure worth calling a cluster, in localizations.  It is
also the main control of how finely HDBSCAN divides: larger merges
neighbouring clusters.

### hdbscan.min_samples
How dense a neighbourhood must be to start a cluster.  Larger values leave
more localizations as noise and keep only well-defined clusters.

### hdbscan.use_z
As for DBSCAN.

## Output

* **The table**, with `cluster_id` and `n_in_cluster`.
* **Text**: how many clusters, how many localizations are in one, and the
  algorithm's settings: DBSCAN's radius as used, Voronoi's mean density.
* **Figure**: the clusters in colour, the rest grey; and how large they
  are.  A good result is one colour per structure.  Many small clusters
  inside one structure mean the radius is too small; a few huge ones, that
  it is too large.

## Differences from SMAP

DBSCAN is based on SMAP's `Analyze/cluster/DBSCAN_cluster`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), which runs the
MIiSR version of DBSCAN ([Caetano et al. 2015](https://doi.org/10.1371/journal.pcbi.1004634)).

* **Border points** join the cluster of their nearest core point; in SMAP,
  that of whichever core point was expanded first, which depends on the
  order of the table.
* **The radius** defaults to 50 nm; SMAP's default is the automatic one,
  which in a sparse field merges everything.  A radius of 0 still gives it.
* **The output** is `cluster_id` 0 for noise (SMAP: `clusterindex` -1) and
  `n_in_cluster`; SMAP writes `clustertype` (core, border, noise) instead.
  Clusters are numbered by size.  ROIs are made of them by a plugin of
  their own, for any clusterer's ids.
* **Files** are clustered one at a time, and on a layer that merges blinks
  the blinks.
* **Voronoi and HDBSCAN** are new: SMAP's `ClusterVoronoi` computes only
  the Voronoi density (here [Neighbourhood](plugin:Analysis/Process/Neighbourhood)),
  and SMAP has no HDBSCAN.

## References

* Ester M, Kriegel H-P, Sander J, Xu X. A density-based algorithm for
  discovering clusters in large spatial databases with noise. *Proceedings
  of the Second International Conference on Knowledge Discovery and Data
  Mining (KDD-96)*, 226-231 (1996).
* Caetano FA, Dirk BS, Tam JHK, et al. MIiSR: Molecular Interactions in
  Super-Resolution Imaging Enables the Analysis of Protein Interactions,
  Dynamics and Formation of Multi-protein Structures. *PLoS Comput Biol* 11,
  e1004634 (2015).
  [doi:10.1371/journal.pcbi.1004634](https://doi.org/10.1371/journal.pcbi.1004634)
  -- the version SMAP runs, and its automatic radius.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
* Levet F, Hosy E, Kechkar A, Butler C, Beghin A, Choquet D, Sibarita JB.
  SR-Tesseler: a method to segment and quantify localization-based
  super-resolution microscopy data. *Nat Methods* 12, 1065-1071 (2015).
  [doi:10.1038/nmeth.3579](https://doi.org/10.1038/nmeth.3579) -- the
  Voronoi clustering and its density factor.
* Andronov L, Orlov I, Lutz Y, Vonesch JL, Klaholz BP. ClusterViSu, a method
  for clustering of protein complexes by Voronoi tessellation in
  super-resolution microscopy. *Sci Rep* 6, 24084 (2016).
  [doi:10.1038/srep24084](https://doi.org/10.1038/srep24084)
* Campello RJGB, Moulavi D, Sander J. Density-based clustering based on
  hierarchical density estimates. In *Advances in Knowledge Discovery and
  Data Mining (PAKDD 2013)*, Lecture Notes in Computer Science 7819,
  160-172 (2013).
  [doi:10.1007/978-3-642-37456-2_14](https://doi.org/10.1007/978-3-642-37456-2_14)
* McInnes L, Healy J, Astels S. hdbscan: Hierarchical density based
  clustering. *J Open Source Softw* 2, 205 (2017).
  [doi:10.21105/joss.00205](https://doi.org/10.21105/joss.00205) -- the
  implementation used.
