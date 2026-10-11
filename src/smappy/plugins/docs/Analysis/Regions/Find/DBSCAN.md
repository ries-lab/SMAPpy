---
version: "1"
covers: [smappy.plugins.clusters.dbscan, smappy.plugins.clusters.auto_eps, smappy.plugins.clusters.with_clusters, smappy.plugins.clusters.hull, smappy.plugins.clusters.DBSCAN.run]
---

## What it does

Finds clusters -- places where localizations lie closer together than
elsewhere -- and names them.  It uses DBSCAN
(Ester et al. 1996): a
cluster is a run of crowded localizations, however it is shaped, and what is
not crowded belongs to none.  Nothing has to be said about how many clusters
there are or how large they are.

What it gives is a **name for every cluster**, written onto the table:
`cluster_id` (1 for the largest cluster, 2 for the next, ..., 0 for a
localization in no cluster) and `n_in_cluster` (how many localizations its
cluster has).  With them

* a filter on `n_in_cluster` keeps the large clusters and drops the specks;
* a **block** of a chain, or the [region steps](plugin:Panels/Regions tab)
  set to *for each* `cluster_id`, runs any measurement once in every cluster
  -- [Localization Statistics](plugin:Analysis/Measure/Localization Statistics),
  say -- and keeps a row of numbers per cluster, which the block's *write
  values* or [Regions to column](plugin:Analysis/Regions/Regions to column)
  puts back onto the localizations, to filter or colour by;
* *output* set to ROIs makes an ROI of each cluster, to look at, review and
  annotate in the [ROI manager](plugin:Panels/ROI manager).

It clusters the localizations of the layer as it is shown: its filter, its
ROI and, on a grouped layer, its blinks.  Use it on structures that are
separate from each other -- pores, clusters of a receptor, vesicles.  On a
continuous structure (a filament network) every localization is crowded
and it joins everything into one.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.plugins.clusters import dbscan, sizes_of, hull
locs = simulate(SimulationSettings(n_frames=3000, seed=3,
                                   structure=StructureSettings(preset="npc"),
                                   labelling=LabellingSettings(efficiency=0.6)))
bright = np.asarray(locs["photons"]) > 200
xy = np.column_stack((locs["x_nm"], locs["y_nm"]))[bright]
copy = np.asarray(locs["copy"])[bright]
labels = dbscan(xy, 50, 10)
```

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
`n_in_cluster` is written beside `cluster_id`; both are kept when the
localizations are grouped, a blink taking its cluster.

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

## In detail

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

**How it is computed.**  The neighbours are found with a k-d tree, and the
clusters are the connected components of the graph of core points within
$\varepsilon$ of each other -- a hundred thousand localizations take about
a second.  Each file of the session is clustered on its own, and the
numbers of the second file's clusters follow the first's.

**ROIs.**  An ROI's outline is the convex hull of its cluster, its centre the
mean position, and it remembers the cluster's number and size (`origin`).
A cluster whose localizations lie on a line, or that has fewer than three,
gets no ROI.  ROIs cannot be made inside a block, where clusters are found
per region.

## Parameters

### radius_nm
About the distance between neighbouring localizations inside a structure,
not its size: a few times the localization precision, 30 to 80 nm for
typical data.  Too small and clusters break into pieces; too large and
neighbouring structures merge.

### neighbours
Larger values ask for denser clusters and leave more of a cluster's edge as
noise.  Raise it with the number of localizations per structure; 5 to 20 is
usual.

### use_z
Off clusters by x and y only, which joins structures that are above each
other.

### output
The column is what a block repeats over; the ROIs are for looking.  Both
write the column and make the ROIs.

### min_size
Keeps the ROI manager to the clusters worth looking at; the column holds all
of them either way.

### replace
Without it, running again adds a second set of ROIs over the first.

## Output

* **The table**, with `cluster_id` and `n_in_cluster` (unless *output* is
  ROIs alone).
* **ROIs** in the ROI manager, with *output* ROIs or both.
* **Text**: how many clusters, how many localizations are in one, the
  radius used.
* **Figure**: the clusters in colour, the rest grey; and how large they
  are.  A good result is one colour per structure.  Many small clusters
  inside one structure mean the radius is too small; a few huge ones, that
  it is too large.

## Differences from SMAP

Based on SMAP's `Analyze/cluster/DBSCAN_cluster`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), which runs the
MIiSR version of DBSCAN ([Caetano et al. 2015](https://doi.org/10.1371/journal.pcbi.1004634)).

* **Border points** join the cluster of their nearest core point; in SMAP,
  that of whichever core point was expanded first, which depends on the
  order of the table.
* **The radius** defaults to 50 nm; SMAP's default is the automatic one,
  which in a sparse field merges everything.  A radius of 0 still gives it.
* **The output** is `cluster_id` 0 for noise (SMAP: `clusterindex` -1) and
  `n_in_cluster`; SMAP writes `clustertype` (core, border, noise) instead.
  Clusters are numbered by size.  ROIs can be made of them.
* **Files** are clustered one at a time, and on a grouped layer the blinks.

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
