# SMAP → smappy: work plan

From your annotations in `SMAP_porting_list.xlsx`, read loosely: I went by the
comments more than by the Decision column.  Row numbers (`#12`) are the `#`
column of the spreadsheet; its **Work package** column says where each row
went, **Update 2026-10-14** what the regions restructuring changed, and
**Plan step** which numbered step of the implementation plan takes it.

**To start, read [Implementation plan: where to start](#implementation-plan-where-to-start)
at the end**: 19 numbered steps in five waves.  The sections A–J before it
are the backlog by area.

**Updated 2026-10-14** (and the segmentation/density split decided the same day) against `origin/main` `c9f650a`, after the regions
restructuring (`docs/regions.md`).  What it settled, as far as this plan is
concerned:

* **A region is an ROI or the rows sharing an id** (`cluster_id`, `roi_id`,
  any integer column).  A finder and a clusterer are one kind of plugin,
  under `Analysis/Regions/Find/` (Density Peaks and NPC make ROIs, DBSCAN
  writes `cluster_id` / `n_in_cluster`).
* **There is no separate evaluator kind any more.**  A plugin with
  `per_region = True` returns `Result.values` and runs on the view, in each
  ROI, or in each cluster.  A **block** of a chain runs a list of steps in
  every region (blocks nest: clusters inside ROIs), and *write back* / *write
  values* bring results onto the localizations.  This was E4 of the old plan:
  **done**.
* **ROI management**: Sort ROIs, Filter ROIs, Regions to column (including
  `roi_id`), Column to ROIs (an ROI per cluster, or per value of any id
  column), and annotation with keyword lists and keys in the ROI manager.
  Most of E1 and E2: **done**.
* One loader (`File/Load/Auto`, with a *format* choice), and the layer
  filters as a step, `Analysis/Process/Layers`.

What follows is the plan with that taken out: what is left, and how the new
structure changes the way it should be built.

---

## How to build a measurement now

Most of what is left is a measurement: B1, B3, B4, circle fit, NPC
geometry, Line Profile per ROI, the spatial statistics.  Under the new
contract each is **one ordinary plugin** under `Analysis/Measure/`.  It runs on
the view, and with `per_region = True` plus `Result.values` it also runs per
ROI and per cluster.  Nothing is ported twice: SMAP's `X` (Analyze) and
`X` (ROIManager/Evaluate) collapse into one.  Summaries over many regions read
`region_results` and live under `Analysis/Regions/Summarize/`.

---

## A. Small plugins (P1); none of them depends on regions

| | what | rows | status |
|---|---|---|---|
| A1 | **Export**: one plugin with a format choice: CSV/TXT, Picasso HDF5, and SMAP `_sml.mat` as an optional, low-priority format; choose the fields; all or filtered | #53 #54 (#55) | open. There is still only *Save smappy HDF5* and *Export Image*. |
| A2 | **Loaders**: Picasso HDF5 and DECODE, as two more `Reader`s in `io/formats.py` (the single loader's *format* list picks them up) | #51 #52 | open. Smaller than before: a reader, not a plugin. |
| A3 | **Fit region** (`RoiAdder`): fit only inside, or exclude, drawn regions of the chip | #7 | open |
| A4 | **Histogram of any column**, per layer, with mean/median/std | #86 | open. *Regions/Summarize/Histograms* is the histogram over regions; the one over localizations is still missing. Make it `per_region` as well, and the same plugin gives a histogram per ROI for free. |
| A5 | **Time course**: any columns against frame, several in one graph, merged with Image Tags | #87 | open |
| A6 | **Vibrations**: spectrum of x, y, z, with zoom-ins and filters | #88 | open. Still needs: which filters, which zoom-ins. |
| A7 | **Export the region table** to CSV: the user picks the columns; ROIs or clusters through `region_results` | #130 | open. A summary plugin under `Regions/Summarize/`. Nothing writes the table to CSV yet. |

## B. Segmentation, local density, spatial statistics (P1)

**Decided 2026-10-14:** two different jobs, kept apart.

* **Segmentation** says which localizations belong together, and gives an id
  or an ROI.  It comes in two plugins, by what they are usually used for:
  * **Find clusters**: classical cluster analysis, *many* regions, written as
    an id column (`cluster_id`, `n_in_cluster`).  Algorithms: DBSCAN (there
    now), HDBSCAN, Voronoi threshold (density above a cut, then connected
    cells), maybe others.
  * **Segment regions**: *few but well-defined* regions, usually made into
    ROIs for review and measurement.  Algorithms: NPC (ring template, there
    now), Density Peaks (there now), other templates (CME, beads, ...), each
    with its own model and parameters, as presets; only the selected preset's settings are shown.
* **Local density** describes each localization's neighbourhood and writes a
  column; it segments nothing.  That is **Neighbourhood** (B1).
* **Shuffle colours** is a third, separate, small plugin, because it must
  work on any integer index: `cluster_id`, `roi_id`, `tid`, `track_id`,
  `frame`.

| | what | rows | status |
|---|---|---|---|
| B1 | **Neighbourhood** (the renamed `SpatialFilterLocs` / `density_calculator`): per localization, the number of neighbours within r (local density), the mean/median/min/max/quantile of a field around it, and the Voronoi cell density (area⁻¹, SharpViSu). Writes columns (recipes), which can be filtered and coloured, and which Find clusters' Voronoi threshold can use. | #70 #93 (density) #94 | done (step 9) |
| B2 | **Find clusters**: one plugin with an algorithm choice. Each algorithm's settings are a part; only the selected algorithm's part is **shown**, the others hidden (step 1 below adds that to the form). It replaces `Regions/Find/DBSCAN`. Writes `cluster_id` and `n_in_cluster` as now. | #91 #93 (clusters) | done (step 10): DBSCAN, Voronoi, HDBSCAN in `Regions/Find clusters` |
| B2s | **Shuffle colours**: for any integer column, writes a column with the ids randomly permuted (for example `<column>_shuffled`, a recipe), so colouring by it separates neighbouring ids. A new seed on each run. | #95 | done (step 7): `Analysis/Process/Shuffle ids` |
| B2b | Statistics on clusters, also inside ROIs | #96 #120 | done by the structure (a block over `cluster_id`). The cluster-shape measurement is **done** (`Measure/Cluster shape`, step 12). |
| B3 | **Spatial statistics**: Ripley K/L/H, pair correlation, 2D auto- and cross-correlation between layers, the 1D correlation along a line; `per_region`, so it runs in every ROI too | #90 #92 #119 | decided (`wave4_design.md`): one plugin with a list of approaches, steps 17a-c; the radial approaches **done** (`Measure/Spatial statistics`, step 17a), their summary over regions too (`Summarize/Curves`, step 17b), the directional ones (correlation image, correlation along a line, step 17c) |
| B4 | **Counting**: a `per_region` measurement plus a `Summarize` histogram fit | #98 #121 | decided (`wave4_design.md`): *Blinking*, a `per_region` measurement (step 18); the counting fit later, on top of the NPC model; DNA-PAINT / qPAINT later, after looking at Picasso |

What this changes in the code: `Regions/Find/` becomes the home of the two
segmenters, `Find/DBSCAN` folds into *Find clusters*, and `Find/NPC` and
`Find/Density Peaks` fold into *Segment regions*.  That is a change to the
tree that `docs/regions.md` ("The plugin tree", "Clusters") and CLAUDE.md's
shape table describe, so both are updated in the same change.  Saved chains
that name `Regions/Find/DBSCAN` or `Regions/Find/NPC` (the shipped
`npc_analysis.chain.yaml` among them) need their step paths mapped.

## C. Transformations and channels (P2), unchanged

| | what | rows |
|---|---|---|
| C1 | **Transform**: apply a saved transformation, a shift (typed in, or from C2), mirror, rotate, scale; to tables and image layers | #40 #69 #74 |
| C2 | **Register layers**: shift between layers, rendered or images, by cross-correlation; feeds C1 | #41 #119 |
| C3 | Channel alignment against frame (diagnostic) | #41 #42 |
| C4 | Assign colours for more than 2 colours (**done**: up to six) | #47 |
| later | single-channel fits → dual colour by pairing (#46); Calibrate transform from separate camera files (#39) | |

## D. Drift (P2), unchanged

D1 apply an existing drift curve from another file (#35) · D2 fiducial
drift (#34) · D3 time-dependent transformation (magnification) in COMET
(#38) · later: Picasso drift (#36).

## E. ROI manager and regions

| | what | rows | status |
|---|---|---|---|
| E1 | Filter ROIs | #127 | **done** (`Regions/Filter ROIs`) |
| E1 | Sort ROIs | #128 | **done** (`Regions/Sort ROIs`) |
| E1 | Site number / an evaluation value → localization column | #131 | **done** (`Regions/Regions to column`, `roi_id` included; the block's *write values*) |
| E1 | Export the site table | #130 | open; now A7 |
| E1 | Remove double sites | #129 | **done** (`Regions/Remove double ROIs`, step 14) |
| E1 | Remove localizations outside the ROIs | #107 | **done** (Remove Localizations, *region* = *all included ROIs*, step 14); for clusters, a filter on `cluster_id > 0` |
| E1 | Use only the ROIs inside the drawn ROI | #106 | **done** (*Filter ROIs*: `in_drawn_roi`, step 14) |
| E1 | Import site positions | #135 | open, P3 |
| E1 | ROI per file (`files2sites`) | #108 | **mostly there**: *Column to ROIs* on `filenumber`. You had marked it skip anyway. |
| E2 | **Annotate ROIs** | new | **done** (`gui/roi_annotate.py`; the named lines step 13): any number of keyword lists written in the panel and saved with the project; keys 1–9 pick a keyword, 0 clears it, Tab switches lists, Up/Down moves to the next ROI, Space toggles *use*; a comment per ROI; the keywords become columns of the results table, so Filter and Sort read them. Any number of named lines per ROI (step 13): drawn in the ROI manager, reversed, `<name>_angle_deg` and `<name>_length_nm` columns of the results table. |
| E3 | **Segment regions** with presets (NPC, Density Peaks, CME, beads, other templates); see B | #104 #105 #109 #111 | decided: one plugin, see B. NPC and Density Peaks exist as separate Find plugins, to be folded in. *Centre all* (#116): **done** (`Regions/Centre ROIs`, step 15). |
| E4 | No distinction between evaluator and analysis plugin | #118 | **done** (`per_region`; Line Profile too, step 15) |
| E5 | **Circle fit** as a `per_region` Measure plugin (#115, P2); **NPC geometry** (#114, P3; independent of LocMoFit, but better as one of its models: decide H first) | #114 #115 | Circle fit **done** (`Measure/Circle fit`, step 15); NPC geometry open |
| E6 | Figures of regions: montage, TIFF stack, single TIFFs, volumes, sorted by any column | #132 #133 | open, P3. Now a Summarize-style plugin that reads the ROI order (Sort ROIs) and annotations. |
| E7 | **Average ROIs**: centre plus an orientation from a measured value or a drawn line → a new table | #134 | **done** (step 16, `Regions/Average ROIs`) |

## F. Single-particle tracking (P1, own project)

Unchanged, with one simplification from the structure: a **track is a
region**.  Linking writes `track_id` (an id column, like `cluster_id`), so
every `per_region` plugin runs per track for free, and *Column to ROIs* makes
ROIs of tracks to review them.  What the project still needs: a linker (look
for an optimized algorithm), then the analyses (MSD, diffusion, jump
distances) as `per_region` measurements plus summaries over tracks, and tracks
drawn into a movie export (#100 #101 #141).

**Decided 2026-10-15** (`wave4_design.md`): inside smappy, steps 19a-c --
track simulation and a simple linker benchmarked against a global
assignment (tracking is sparse: the problem is bridging dark frames, not
dense linking), a *tracks* render mode, MSD and jump distances.  Step
finding and changes of motion within a track later.

## G. MINFLUX (P2, own project)

The same holds here: MINFLUX's trace id `tid` is already an id column, so
per-trace measurements run through the regions machinery (#103 #152).  Plan
from scratch what is needed now.  G shares the track model with F.

## H. LocMoFit (P2, own project)

Port it, or build on Weiqing's approach (#117 #138).  Under the new contract a
model fit is a `per_region` Measure plugin: its parameters are its values, so
Filter ROIs, Sort ROIs, Histograms and Regions to column work on them with no
extra code.

## I. Rendering and movies (P3), unchanged

z-stack export (#76) · Voronoi renderer (#77) · movie export (#139 #140 #141;
with the tracks as an option: step 19b's render mode with a *tail*, over the
raw frames)
· image processing for image layers (#81).

## J. Later / when needed, unchanged

GPU fitter (#10) · background before fitting (#6) · astigmatic z (#11) ·
4Pi (#13) · DECODE / DECODE-plex (#15) · sCMOS maps (#22) · uiPSF robustness
(#24) · depth-dependent aberrations (#26) · GainOffsetFFT (#27) · DHPSF (#29) ·
Register3Dstacks (#44) · combine files (#65) · grouped table as primary (#66) ·
rename/remove fields (#68) · alternating excitation (#71) · TIFF intensity at
sites (#124).

**GroupAny (#67)**: still no plugin needed.  The regions machinery now covers
the measuring half: a block over any id column.  What is left is collapsing
each region into one row (one localization per cluster or track).  If ever
wanted, that is `group.combine` keyed by the id column: an option of
Regions to column, or a small plugin of its own.

---

## Implementation plan: where to start

Each step is sized for one session: a module, its tests and its page, plus
any docs and exhaustive tests the change touches, as CLAUDE.md asks.  The
steps within a wave are independent and can run in parallel.  Sizes:
S ≤ a day, M a few days, L a week or more.

### Wave 1: foundations and small P1 plugins (all independent)

| step | what | where | size | done when |
|---|---|---|---|---|
| **1** ✅ | **Hide what the settings do not use.** `FormState.shown: Dict[str, bool]`, beside `active`: a part or field set False is **hidden** (`setVisible` on a part's section; `QFormLayout.setRowVisible` on a row, available from Qt 6.4 and smappy needs ≥ 6.6), not greyed. Like `active`, it is presentation only: a hidden field keeps its value, and a script ignores it. | `plugins/__init__.py` (`FormState`), `gui/params.py` (`set_shown`), `gui/plugin_panel.py`; the chain panel picks it up through the same `describe` | S | a test with a plugin whose `describe` hides a part, in the plugin panel and in a chain step; `docs/plugin-reference.md` and CLAUDE.md's `describe` bullet mention `shown` |
| **2** ✅ | **Export localizations**: one plugin, a format choice (CSV/TXT, Picasso HDF5; SMAP `_sml.mat` as an optional later format), the fields to write, all or filtered | `File/Export/Localizations` in `plugins/file.py`; writers beside the readers in `io/formats.py` | S–M | a round trip through each format with the loaders (Picasso through step 3) |
| **3** ✅ | **Readers**: Picasso HDF5, DECODE (h5/csv), registered as `Reader`s so *File/Load/Auto*'s *format* list offers them | `io/formats.py` | S | reading a file each format's own tool writes (a small fixture) |
| **4** ✅ | **Histogram of any column**, per layer, with mean/median/std; `per_region`, so it also gives one per ROI or per cluster | `Analysis/Measure/Histogram` | S | test on simulated data; page |
| **5** ✅ | **Time course**: any columns against frame (photons, locs per frame, precision, background), several in one graph, beside the image tags | extend `image_tags.py`, or `Analysis/Measure/Time course` sharing its drawing | S–M | test and page; Image Tags' page updated if it changes |
| **6** ✅ | **Export the region table** to CSV: the columns chosen by the user, ROIs or clusters through `region_results` | `Analysis/Regions/Summarize/Export table` | S | a CSV of an NPC Corners run, the ROI annotations and `roi_id` included |
| **7** ✅ | **Shuffle ids**: for any integer column (`cluster_id`, `roi_id`, `tid`, `frame`, ...), write `<column>_shuffled` with the ids permuted, so colouring by it separates neighbouring ids; a new seed per run | `Analysis/Process/Shuffle ids` | S | the permutation keeps id 0 / noise at 0; grouping keeps a blink's value (recipe with rule `max`) |
| **8** ✅ | **Fit region**: fit only inside / exclude regions drawn on the camera frame | the fitters' detection part (`plugins/fit.py`), a mask on the candidates | S–M | candidates outside the region are dropped; the Localize tutorial still runs |

### Wave 2 -- the design is in `wave2_design.md` (decided 2026-10-14)

Paths: `Analysis/Process/Neighbourhood` (9), `Analysis/Regions/Find clusters`
(10, DBSCAN moved in, Voronoi, HDBSCAN optional), `Analysis/Regions/Segment
regions` (11, NPC, density peaks and Column to ROIs moved in, presets as
files), `Analysis/Measure/Cluster shape` (12).  No aliases for the old
paths: chains, GUI, tests and docs are updated.  The table below is the
first sketch; where they differ, `wave2_design.md` wins.

| step | what | where | size | done when |
|---|---|---|---|---|
| **9** ✅ | **Neighbourhood**: per localization, neighbours within r (local density), mean/median/min/max/quantile of a field within r, Voronoi cell density; written as recipe columns | `Analysis/Process/Neighbourhood` (module-level functions on a `cKDTree` and `scipy.spatial.Voronoi`) | M | densities of a simulated uniform field match the expected count; page with figures |
| **10** ✅ | **Find clusters**: one plugin, an algorithm choice with one part per algorithm (only the selected one shown): DBSCAN (moved from `clusters.py`), Voronoi threshold (density from step 9, then connected cells), HDBSCAN if scikit-learn ≥ 1.3 is installed (an optional dependency, asked for at Run like uiPSF) | `Analysis/Regions/Find/Find clusters`; `Find/DBSCAN` removed | M | DBSCAN's tests pass through the new plugin; Voronoi finds simulated clusters; `docs/regions.md` ("Clusters", "The plugin tree") and CLAUDE.md's shape table updated; the old path `Regions/Find/DBSCAN` removed, with no alias (`wave2_design.md`) |
| **11** ✅ | **Segment regions**: one plugin with presets, one part each, only the selected shown: NPC (moved from `npc.py`), Density Peaks (moved from `roi.py`); a `SegmenterPlugin` as now (Preview, *replace*, ROI undo). Templates such as CME and beads come later as further presets. | `Analysis/Regions/Find/Segment regions` | M–L | NPC and Density Peaks tests pass through it; `npc_analysis.chain.yaml` and the NPC tutorial use it; the old paths removed, no aliases; docs as in step 10 |
| **12** ✅ | **Cluster shape**: area, convex hull, radius of gyration, density, elongation, as a `per_region` measurement (the statistics a block over `cluster_id` does not yet give) | `Analysis/Measure/Cluster shape` | S | known shapes return their numbers |

Steps 10 and 11 share the docs and the moves out of the old paths: do them
one after the other, in the same session if possible.

### Wave 3: the ROI manager's remaining pieces

| step | what | size | needs |
|---|---|---|---|
| **13** ✅ | **Named lines per ROI**: the one *direction* becomes any number of named lines, drawn in the ROI manager, each with angle and length as values in the results table, and a reverse action | M (GUI, ROI project file format) | – |
| **14** ✅ | **Small ROI tools**: remove double ROIs (closer than d: keep the larger); remove localizations outside the included ROIs (a third *region* choice of Remove Localizations); keep the ROIs inside the drawn ROI (a *Filter ROIs* condition on position) | S | – |
| **15** ✅ | **Line Profile `per_region`** (width, distance, AIC winner as values) and **Circle fit** (`Analysis/Measure/Circle fit`, `per_region`), with *centre ROIs* on the circle or median centre as a Regions action | S–M | – |
| **16** ✅ | **Average ROIs**: centre plus orientation from a measured value or a named line → a new table | M | 13 |

### Wave 4 -- the design is in `wave4_design.md` (decided 2026-10-15)

| step | what | size | needs |
|---|---|---|---|
| **17a** ✅ | **Spatial statistics, radial**: `Analysis/Measure/Spatial statistics` -- pair distances from the C++ `Index`, the window's set covariance as edge correction, g, K, L-r, H for each visible layer and each pair, CSR / *keep blinks* envelopes, `per_region` values; Neighbourhood's count *express as* local L(r) | M-L | - |
| **17b** ✅ | **Curves summary**: `Analysis/Regions/Summarize/Curves`, mean and spread of any evaluation's curves, grouped by an ROI keyword, pooled where counts are given | S-M | 17a |
| **17c** ✅ | **Spatial statistics, directional**: 2D correlation image, 1D correlation along a line with its period; axes x-y / x-z / y-z / z, a line or rectangle ROI's own directions | M | 17a |
| **18** | **Frame time** per file, and **Blinking** (`Analysis/Measure/Blinking`, `per_region`: counts, bursts, on- and dark times; per-file blinking histograms) | S-M | - |
| **19a** | **Tracks**: simulated tracks in `smappy.simulate`; `Analysis/Tracking/Link tracks` (simple and global assignment, benchmarked) writing `track_id` | M-L | - |
| **19b** | **The *tracks* render mode**: a fifth render mode, segments drawn in C++ | M | - |
| **19c** | **Tracking analysis**: MSD and jump distances, per track and over all, vectorised; an *id column* (`track_id`, `tid`) | M | 18, 19a |

17a, 18, 19a and 19b can run in parallel.

### Wave 5 -- designed a topic at a time in `wave5_design.md`

Topics: 1. coordinates (decided 2026-10-15) · 2. vibrations (A6) ·
3. rendering and movies (I; movie export with the tracks as an option:
step 19b's render mode with a *tail*, over the raw frames) · 4. figures of
regions (E6) · 5. LocMoFit (H) and NPC geometry · 6. MINFLUX (G), with the
channel *Check overlay* (C3).  AssignColors for more than 2 colours (C4):
**done** (`assign_colors.py`, up to six).

| step | what | size | needs |
|---|---|---|---|
| **5a** | **Transform** (C1, D1): typed, named, from a file, or another file's drift; localizations (default) or an image layer's placement; named transformations in the session, written to files; a drift *reference* (first / last frame) | M | - |
| **5b** | **Register layers** (C2): shift by cross-correlation, z optional; hands its result to Transform | S-M | 5a |
| **5c** | **Fiducial drift** (D2): bright long-lived tracks, one spline dx, dy (dz), a `Drift` | M | - |
| **5d** | **Magnification over time in COMET** (D3), discussed with Jonas first | M-L | - |

### Done

* **Step 1** (2026-10-14): `FormState.shown`.
* **Steps 2 and 3** (2026-10-14): *File/Export/Localizations* (csv with
  smappy's or ThunderSTORM's names, Picasso HDF5; visible or all, as the
  layer or localizations or merged blinks, the columns ticked in a list);
  readers for Picasso and DECODE; the csv reader keeps every column, reads
  lengths in nm, µm, mm or pixels and frames from 0 or 1, and saved
  conversions found by the headers (ThunderSTORM's and SMAP's shipped);
  *File/Load/Load*, the File tab's first line.  SMAP's `_sml.mat` export is
  left for later.
* **Step 4** (2026-10-14): *Analysis/Measure/Histogram*: any column, a
  histogram per visible layer on common bins, mean / median / std / mode
  of the values in the range, per region as well; pinned in the Analysis
  tab of the shipped GUI.
* **No more `favorite`** (2026-10-14): the shipped GUI is a GUI file,
  `data/default.gui.yaml`, and the per-region steps a project starts with
  are `DEFAULT_REGION_STEPS`; the unused favourite star went with it.
* **Step 5** (2026-10-14): *Analysis/Measure/Time Course*: the rate, any
  column's median per window and the image tags against frame, time or
  file; a panel per series or one graph scaled to the maxima; a line per
  layer.  Image Tags stays, for one tag in detail beside the rate.
* **Step 6** (2026-10-14): *Analysis/Regions/Summarize/Export table*: the
  results table (every evaluation on the ROIs, or one, or per cluster) as
  csv/tsv, the columns ticked, with each ROI's centre, file and comment.
* **Step 7** (2026-10-14): *Analysis/Process/Shuffle ids*: any integer
  column's ids permuted among themselves into `<column>_shuffled` (a
  recipe, `max` for merged blinks), 0 and below kept, the seed recorded.
* **Step 8** (2026-10-14): the fitters' *fit region*: rectangles and
  polygons drawn on the Preview's frame (fit in / leave out), kept in the
  settings in frame pixels, the candidates outside dropped before fitting.
  A figure's panel declares itself drawable with `Plot.edit` (`ShapeEdit`);
  the figure window adds the tools (`gui/shape_tools.py`).  Wave 1 done.
* **Step 9** (2026-10-14): *Analysis/Process/Neighbourhood*: per
  localization, the neighbours within r (circle, cylinder or Gaussian;
  itself not counted), a statistic of a column within r, or the Voronoi
  density (per um^2, NaN for open cells), between an *around* and a
  *neighbours from* layer as they show (cross-density for two), per file;
  a recipe column (`mean`), NaN outside the around layer's filter.  The
  search is SMAP's sorted sweep made 2D in C++ (`csrc/neighbours.cpp`,
  sorted as the grouping sorts; 5 M localizations in ~3 s); the Voronoi
  cells come from tiled Delaunay triangulations checked to equal one (5 M
  in ~1 min); a preflight asks above 20 s; the % done is reported and Stop
  writes nothing.
* **Step 10** (2026-10-14): *Analysis/Regions/Find clusters* replaces
  `Regions/Find/DBSCAN` (no alias): an *algorithm* choice, one part each,
  only the chosen one shown.  DBSCAN moved in with the same numbers;
  Voronoi (SR-Tesseler's: dense above a multiple of the mean density,
  dense cells that touch -- the whole field's Delaunay edges, from step 9's
  tiled triangulation -- one cluster, a minimum size); HDBSCAN from the
  hdbscan package (Boruvka spanning tree: 53 s at a million, where
  scikit-learn's Prim's takes hours), the new extra
  `smappy-smlm[clusters]`, named by the form and the run when missing,
  with a preflight question.  DBSCAN reports % and stops, and runs on
  Neighbourhood's C++ cell search with union-find (2 M in 2 s, was 28 s).
* **Step 11** (2026-10-15): *Analysis/Regions/Segment regions* replaces
  `Find/NPC`, `Find/Density Peaks` and `Column to ROIs` (no aliases): one
  `SegmenterPlugin` (`plugins/segment.py`), a *method* part each -- ring
  template, density peaks, from an id column -- only the chosen one shown;
  the code moved, the algorithms where they were, the tests on the new
  path.  A *preset* (YAML in `data/segment_presets`, `NPC` shipped, the
  user's under the config directory) fills the settings; each ROI's
  `origin["segment"]` names its method, so *replace* removes that method's
  only.  The id-column method gained a Preview, and is refused in a block
  when the chain is made, as every segmenter.  One page for the three.
* **Step 12** (2026-10-14): *Analysis/Measure/Cluster shape*, `per_region`:
  per layer the number of localizations, centre, convex-hull area and
  density, radius of gyration, standard deviations along the long and the
  short axis with the angle and the extents, elongation, and in 3D the z
  extent and the hull's volume (`n_locs`, `area_nm2`, `rg_nm`, ...);
  tested on a disc, ellipses, a line and a cylinder; in the Analysis tab of
  the shipped GUI, not in `DEFAULT_REGION_STEPS`.  SMAP's `boundary` area
  and *skip first* left out.

* **Step 14** (2026-10-15): small ROI tools.  *Analysis/Regions/Remove
  double ROIs*: of used ROIs of one file closer than a distance, the one
  with more localizations (as the ROI manager shows them) stays, the other
  is excluded, one ROI undo step -- a plugin of its own, not a mode of
  *Filter ROIs*, since a double is a pair and a filter tests one ROI's row.
  Remove Localizations' *region* gained *all included ROIs* (each ROI's
  shape in its file).  *Filter ROIs* (and *Sort ROIs*) read each ROI's
  centre, `roi_x_nm` / `roi_y_nm`, and `in_drawn_roi`: the condition
  `in_drawn_roi` is SMAP's UseSitesInROI.
* **Step 13** (2026-10-15): named lines per ROI.  The one *direction*
  became `ROI.lines`, ``{name: [start, end]}``, under names the project
  keeps (`ROIProject.line_names`, written under *line names* in the
  annotation panel, `direction` to start with).  The ROI manager draws the
  chosen line (*line*, *Draw line*, L), reverses it (*Reverse*, R) and
  clears it; Shift+L or a click on a line chooses it; each name has a
  colour and its label at the tip.  Each line is `<name>_angle_deg`
  (SMAP's `atan2(dy, dx)`) and `<name>_length_nm` in the results row, so
  Filter, Sort, the summaries and *Regions to column* (with or without an
  evaluation) read them.  An old file's `direction` opens as the line
  `direction`, and the signatures of its results still match.

* **Step 15** (2026-10-15): *Analysis/Measure/Line Profile* is
  `per_region` (version unchanged, the numbers are the same): in an ROI the
  line is one of its named lines (step 13; *ROI line*, the first it has by
  default), as wide as the ROI, else a horizontal line across it; the row keeps per layer each model's width (and distance,
  radius) with its error and AIC, and the AIC winner.
  *Analysis/Measure/Circle fit*, `per_region` and *follow ROI*: the NPC
  segmenter's soft-L1 `fit_circle`, with the covariance's errors, the rms
  distance and the reduced chi-square, per layer.
  *Analysis/Regions/Centre ROIs*: every used ROI onto its circle's centre
  or its localizations' median, iterated, one ROI undo step; a Regions
  plugin rather than an ROI-manager button, so a chain can centre between
  segmenting and measuring.  Circle fit in the Analysis tab, Centre ROIs in
  the Regions tab of the shipped GUI.
* **Step 16** (2026-10-15): *Analysis/Regions/Average ROIs*: every used
  ROI centred (its centre, measured x and y values, or the start, middle
  or end of a named line) and turned (none, a named line, a measured
  angle) by SMAP's `rotcoord`, superimposed into one table added as a
  file (or replacing the table), centred on (0, 0), with `roi_id`; each
  ROI's frames moved past the previous one's so merging blinks never
  joins two ROIs (`frame_in_file` keeps the file's).  ROIs lacking what
  the settings ask for are left out and counted.  SMAP's AverageSites and
  its kinetochore, window and NPC variants in one; LocMoFit's model not
  ported.
* **Step 17a** (2026-10-15): *Analysis/Measure/Spatial statistics*,
  `per_region`: the radial approaches (g(r), Ripley's K(r), L(r) - r),
  ticked on or off, of every visible layer as it shows and every pair.
  Pair distances from a new mode of the C++ `Index`
  (`Index.pair_histogram`, per file, no self-pairs; 10^6 localizations to
  500 nm in ~1.6 s on 4 cores), normalised by the window's set covariance
  (the FFT autocorrelation of its raster: Ohser's translation correction,
  Veatch's normalisation, any shape).  The window is the region's shape,
  the drawn ROI, or the occupied area (200 nm bins, closed) or bounding
  box.  Monte Carlo envelopes (99, pointwise band plus the MAD test on
  L - r), null CSR or *keep blinks* (whole blinks moved through
  `group_id`); a cross-correlation keeps layer A and moves B.  Values: the
  first maximum of g and L - r, both at chosen distances, the test's p and
  whether the curve leaves the global envelope; arrays: r, the curves, the
  pair and expected counts, the bands.  A preflight above 5 s offers the
  run without the envelope or with 19 simulations.  *Neighbourhood*
  (version 2): the count gains *express as* count, local L(r)
  (Getis-Franklin) or relative to the mean density, edge-corrected by the
  circle's area inside the window (`window_coverage`).  "H" is L - r, the
  name most papers give it, not a fourth curve.
* **Step 18** (2026-10-15): a frame time per file and
  *Analysis/Measure/Blinking*.  The frame time is `CameraMetadata.frame_time_ms`
  (so `metadata["camera"]["frame_time_ms"]`, beside the exposure, by the
  same three sources: the camera database, the file, the user), and per
  file in a session `FileInfo.frame_time_ms`, saved in `metadata["files"]`;
  `metadata.frame_time_ms` falls back to the exposure, and
  `file_frame_time` / `table_frame_time` are what every reader of seconds
  calls (tracking, 19c, too).  Read from Micro-Manager's
  `<camera>-ActualInterval-ms`, a time-lapse `Interval_ms` longer than the
  exposure, and SMAP's `timediff`; set in the camera editor (a *frame time*
  row; the database's `frame_interval_ms` is now `frame_time_ms`, the old
  name still read) and the fitters' camera part (*frame time*, advanced).
  Time Course (v2) and Localization Statistics' on-time (v5, `on_time_s`
  per region; its *exposure* setting became *frame time*) read it.
  *Blinking*, `per_region`: per region the localizations, the blinks as the
  layer merges them (else runs of consecutive frames), the bursts at a
  *dark time* T (SMAP's `locsdf`, a dark time over T frames splits), the
  mean on- and dark time in frames and seconds; on the view, per file,
  molecules linked by `group.connect` across T dark frames, and the
  histograms of on-times, dark times and blinks per molecule (SMAP's
  `dyeblinking`).  Tested on `simulate.kinetics` with known rates.  Also
  fixed: an ROI's context cut its localizations before the merging it
  triggered wrote `group_id`, so the first ROI measured on a merged layer
  could not trace its blinks.
* **Step 17b** (2026-10-15): *Analysis/Regions/Summarize/Curves*, a
  `SiteAnalysisPlugin` over the evaluations that kept arrays: per x the
  mean of the regions' curves with their SD or SEM, a line per keyword of
  an ROI keyword list (*group by*; an id column's regions are one group),
  and the per-group mean of every value in the text.  Where the evaluator
  hands back numerator and denominator it pools: a new hook,
  `Plugin.pool(sums, x)`, gets every array summed over the group and
  returns the curves made from them -- Spatial statistics' g, K and L - r
  from the summed pairs and expected pairs, the g of the regions measured
  together.  Regions on different x grids are refused, naming the region,
  not interpolated: another grid is another setting, and counts summed
  across rings of different widths are wrong.  In the Regions tab of the
  shipped GUI.
* **Step 19a** (2026-10-15): tracks.  `simulate.simulate_tracks`: Brownian
  particles in one or more diffusive states with switching, blinking (dark
  runs inside a track), a precision, a density and spurious localizations,
  the truth in `track_id_true` and `state_true`.  *Analysis/Tracking/Link
  tracks* (`smappy/tracking/link.py`): `track_id` (0 for unlinked and
  short tracks) and `track_length`, recipes with `max`; maximum jump,
  memory, minimum length, 3D, per channel (files always apart); the
  candidates from a new `Index.pairs` with x shifted per frame; *simple*
  (nearest first, one C++ pass; the default) and *global assignment*
  (Crocker-Grier by `linear_sum_assignment`, trackpy's links exactly).
  Benchmarked with trackpy (`studies/tracking/bench.py`; NOTES.md,
  "Tracking"): the same links below a jump of 0.1 of the spacing, global
  0.3-3% better from 0.1 to 0.4, simple 2-10x faster.  In the Analysis tab
  of the shipped GUI.  The simulation is also *File/Simulate/Tracks*, a
  plugin of its own (the structure simulator shares almost none of its
  settings), pinned in the File tab.
* **Step 17c** (2026-10-15): the directional approaches of *Spatial
  statistics* (version 2), the algorithm in `smappy/correlation.py`: a
  *correlation image* (x-y, x-z or y-z) and a correlation *along a line*
  (x, y or z: the profile, its correlation, the period nearest a guess or
  the first strong peak, the spectrum's period, the cross-correlation's
  shift), of every visible layer and pair, normalised by the window's own
  autocorrelation (17a's estimator not averaged over directions: g = 1 at
  every shift for random positions, where SMAP's `xcorr / mean^2` fell off
  and needed a detrend).  A line ROI, or an ROI's named line, turns x along
  it and y across; z over the data's 0.5th-99.5th percentiles clipped to
  -500 .. 500 nm.  Values `period_nm`, `period_g`, `period_fft_nm`,
  `period_fft_mag`, `shift_nm_L1xL2`; arrays `lag_nm` and `line_g` (Curves
  averages them).  SMAP's multi-correlation filter not ported.
* **Step 19c** (2026-10-15): *Analysis/Tracking/Tracking analysis*,
  `per_region`, of any id column (`track_id`, MINFLUX's `tid` on its
  `time_s` clock): the MSD per track and pooled, the offset from the
  precisions (or fitted), the number of lags Michalet's optimum computed
  exactly from the MSD's covariance (Isserlis) since his formula could not
  be checked; D and alpha; per-track `track_D_um2_per_s` / `track_alpha`
  on a tick.  Jump distances at a lag as 1-3 diffusive states with the
  localizations' errors, by EM (at most 200 000 jumps), BIC to compare.
  Vectorised: 1 M localizations in 54 k tracks in ~10 s.  Tested on
  `simulate_tracks` with known D, states and precision.  In the Analysis
  tab of the shipped GUI.  Frame time from step 18.

* **Step 5a** (2026-10-15): *Analysis/Process/Transform* replaces SMAP's
  ApplyTransform, TransformLocsManually, Mirror and ChangeCamPixelsize.
  The target is a file's localizations (one `channel`, or all; logged, one
  undo step) or an image layer, moved by a new affine
  `ImageData.placement` that the resampling looks pixels up through -- not
  resampled, saved with the file, undone exactly; images refuse the
  projective and polynomial maps and a z shift.  The source: typed (shift
  x, y, z; rotate, scale, mirror about the middle of the target, the origin
  or a point; one 4x4 matrix), a named transformation
  (`Session.transformations`, through `Result.transformations`, saved with
  the workspace and the file, recorded in `metadata["transformations"]`
  when applied), a file (`_2ct.h5` converted to nm with the file's pixel
  size, affine kept affine; the new `_T.h5`, written by *write to*), or
  another file's drift (`Drift.apply(rows=)`, refused for other frames).
  `Drift` gained a *reference* -- as estimated (unchanged: RCC zero mean,
  COMET close to it), first or last frame -- in COMET and RCC; no version
  moved.  `smappy/transformation.py`; Register layers (5b) opens it with
  `settings_for(name)` through the GUI's `"transform"` window.  Pinned in
  the Analysis tab of the shipped GUI.

### Start here

Wave 1 is eight independent sessions.  If only one runs at a time, start with
**step 1**, because waves 2 and 3 build on it.  Then go on with 2 and 3
(export and readers together), then 4, 6 and 7.
