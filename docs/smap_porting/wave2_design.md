# Wave 2: local density, clusters, segmentation -- the design

Decided with Jonas on 2026-10-14.  Each step is one session: read this file,
`SMAP_porting_plan.md` (steps 9-12), `wave4_research.md` (what SMAP's
plugins compute) and CLAUDE.md, then implement the step with its tests and
its page, run `--slow`, update the plan's *Done* list, commit and push.

The decision behind all four: **segmentation and local density are two
different jobs.**  Segmentation says which localizations belong together
and gives an id or an ROI; local density describes each localization's
neighbourhood and writes a column.  Segmentation comes as two plugins by
what they are for -- *Find clusters* (many regions, an id column) and
*Segment regions* (few, well-defined regions, ROIs).

**Renaming is free.**  Nobody else uses smappy yet: there is no alias table
for old plugin paths.  Move a plugin, and update every chain, the shipped GUI
(`data/default.gui.yaml`), the tests and the docs that name it.

Order: 9, then 10 (its Voronoi clustering uses step 9's Voronoi density),
then 11, then 12 (independent; can go any time).

---

## Step 9 -- Neighbourhood

`Analysis/Process/Neighbourhood`, module `plugins/neighbourhood.py`.
Ports SMAP's `density_calculator`, `SpatialFilterLocs` and the density half
of `ClusterVoronoi`.

**Two layers are the reference, as in SMAP**: *around* -- the layer whose
localizations get the value -- and *neighbours from* -- the layer whose
localizations are counted or averaged.  Each is read as that layer shows
it: its own filter and its own *merge blinks*, so whether blinks are merged
and what is filtered is the user's choice, made where it is always made.
The same layer twice is the ordinary local density; two layers give a
**cross-density** (how many of channel B lie around each localization of
channel A: colocalization).  Use `ctx.table(i)` for each layer (CLAUDE.md,
"A plugin that measures usually wants one answer per layer").

**What it writes** (one *quantity* choice; only its settings shown, with
`FormState.shown`):

* **count** -- neighbours within a radius: a circle, or in 3D a cylinder
  (lateral radius and z half-height, SMAP's default 12 nm and 24 nm are too
  small to be a default here -- choose by the data, e.g. 50 nm / 100 nm and
  say why on the page); optionally Gaussian-weighted (SMAP's
  `countneighbours*Gauss`).  Self excluded when the two layers are one.
* **statistic of a column** -- mean, median, min, max, std or a quantile of
  a chosen column over the neighbours within the radius (SpatialFilterLocs).
* **Voronoi density** -- 1 / the area of the localization's Voronoi cell,
  2D (`scipy.spatial.Voronoi`; open cells at the hull get NaN), only for
  *around* = *neighbours from*.  SMAP's ClusterVoronoi multiplied by the
  number of duplicate points; say what is done with duplicates.

**Writing the column.**  Onto the localizations of the *around* layer that
pass its filter; NaN elsewhere.  When *around* shows merged blinks, the
blink's value goes onto its localizations through `group_id` (as DBSCAN
does, `clusters._per_localization`).  The column is a recipe
(`mathparse.remember`, rule `mean` or `max` -- decide and say why) so
merging blinks again keeps it.  Name: a default from the settings
(`density_r50`, `photons_median_r50`, `voronoi_density`), editable.

**Algorithm in module-level functions** on arrays (`scipy.spatial.cKDTree`,
`query_ball_point` / `count_neighbors`; chunk for millions).  Time it on
5 M localizations and add a `preflight` if it can take more than a few
seconds.  Per file: never count across files (`filenumber`), as DBSCAN.

Tests: a uniform field of known density gives the expected count
(Poisson mean pi r^2 rho) and Voronoi density (mean 1/area = rho); a
cross-density between two layers; a statistic of a column with a known
spatial gradient; grouped vs ungrouped; NaN outside the filter.  Page with a
figure: density coloured on simulated clusters.

---

## Step 10 -- Find clusters

`Analysis/Regions/Find clusters`, module `plugins/clusters.py` (DBSCAN is
there).  Replaces `Analysis/Regions/Find/DBSCAN`.

One plugin, an **algorithm** choice, a settings part per algorithm, only
the chosen part shown (`FormState.shown`, step 1):

* **DBSCAN** -- the existing code moved in unchanged: same numbers, same
  tests (rename their plugin path).  The default.
* **Voronoi** -- SharpViSu's (Andronov et al. 2016): a localization is dense
  when its Voronoi density (step 9's function) exceeds a multiple of the
  field's mean density; dense localizations whose cells touch (Delaunay
  neighbours) are one cluster; clusters below a minimum size are dropped.
* **HDBSCAN** (Campello et al. 2013; McInnes et al. 2017) -- clusters of
  different densities in one field without a radius: a minimum cluster size
  (and minimum samples).  Needs scikit-learn >= 1.3 (`sklearn.cluster.HDBSCAN`):
  an **optional** dependency (`pip install smappy[clusters]`, a new extra);
  the plugin's `describe` says how to install it when HDBSCAN is chosen and
  it is missing, and the run refuses with the same message.  Its page says
  when to prefer it (varying densities, no known radius) and when not
  (a physically defined cluster size; speed on millions).

All write `cluster_id` (1, 2, ... by size, 0 for none) and `n_in_cluster`
as `with_clusters` does now, so Shuffle ids, Column to ROIs (soon a method
of Segment regions), blocks over `cluster_id` and Cluster shape work on
every algorithm.  Per file, merged blinks or not as the layer shows, as now.

Update: the shipped GUI, `tests/test_roi_pipeline.py` (lists the region
plugins), `docs/regions.md` ("Clusters", "The plugin tree"), CLAUDE.md's
shape table (`clusters.py` row), the DBSCAN page becomes the Find clusters
page (a section per algorithm).

---

## Step 11 -- Segment regions

`Analysis/Regions/Segment regions`, a `SegmenterPlugin` (Preview,
*replace*, a step of the ROI undo -- as now).  Replaces
`Analysis/Regions/Find/NPC`, `Analysis/Regions/Find/Density Peaks` and
`Analysis/Regions/Column to ROIs`.

* A **method** is an algorithm, a settings part each, only the chosen one
  shown: **ring template** (today's NPC segmenter, `npc.segment_npcs`),
  **density peaks** (`roi.density_peaks`), **from an id column** (today's
  Column to ROIs: an ROI per id of `cluster_id` or any integer column, its
  hull or the project's shape).  Later: CME, beads, other templates.
* A **preset** is a set of values for one structure -- "NPC" (ring, 55 nm,
  8 corners, ...), later "CME" -- a YAML file chosen from a list that fills
  the settings; shipped ones in `src/smappy/data/segment_presets/`, the
  user's under the config directory, as `io/conversions.py` and
  `simulate/structure.py` do, so a lab adds its structure without code.
  Empty preset = the settings as they stand.
* Move the code, not rewrite it: the existing tests must pass with the new
  path.  NPC Corners and NPC labelling efficiency are untouched.

Update: `src/smappy/plugins/npc_analysis.chain.yaml` (names Find/NPC), the
shipped GUI, `tests/test_roi_pipeline.py`, `docs/regions.md` (the plugin
tree), `docs/batch.md` if it names them, CLAUDE.md's shape table (`roi.py`,
`npc.py`, `region_column.py` rows), the pages (one Segment regions page
with a section per method; the old pages go).  Grep for every old path.

---

## Step 12 -- Cluster shape

`Analysis/Measure/Cluster shape`, `per_region = True`, so it runs per
cluster (a block over `cluster_id`) and per ROI.  The shape half of SMAP's
`ClusterStatistics`: number of localizations, convex-hull area (and volume
in 3D), radius of gyration, the standard deviations along the long and the
short axis and the angle (covariance eigenvectors), elongation
(long / short), density (n / area), z extent.  Values named plainly
(`area_nm2`, `rg_nm`, ...).  Not in `DEFAULT_REGION_STEPS`; in the shipped
Analysis tab.  Tests on shapes with known numbers (a disc, an ellipse, a
line).
