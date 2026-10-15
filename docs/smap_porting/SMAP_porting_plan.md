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
| B2s | **Shuffle colours**: for any integer column, writes a column with the ids randomly permuted (for example `<column>_shuffled`, a recipe), so colouring by it separates neighbouring ids. A new seed on each run. | #95 | open, S |
| B2b | Statistics on clusters, also inside ROIs | #96 #120 | done by the structure (a block over `cluster_id`). The cluster-shape measurement is **done** (`Measure/Cluster shape`, step 12). |
| B3 | **Spatial statistics**: Ripley K/L/H, pair correlation, 2D auto- and cross-correlation between layers, the 1D correlation along a line; `per_region`, so it runs in every ROI too | #90 #92 #119 | open |
| B4 | **Counting**: a `per_region` measurement plus a `Summarize` histogram fit | #98 #121 | open |

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
| C4 | Assign colours for more than 2 colours | #47 |
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
| E1 | Remove double sites | #129 | open. A small Regions plugin (or a *Filter ROIs* mode: closer than d → keep the one with more localizations). |
| E1 | Remove localizations outside the ROIs | #107 | open. Simplest as a third *region* choice of Remove Localizations ("all included ROIs"); for clusters, a filter on `cluster_id > 0` already does it. |
| E1 | Use only the ROIs inside the drawn ROI | #106 | open, P3. Could be a *Filter ROIs* condition on position. |
| E1 | Import site positions | #135 | open, P3 |
| E1 | ROI per file (`files2sites`) | #108 | **mostly there**: *Column to ROIs* on `filenumber`. You had marked it skip anyway. |
| E2 | **Annotate ROIs** | new | **mostly done** (`gui/roi_annotate.py`): any number of keyword lists written in the panel and saved with the project; keys 1–9 pick a keyword, 0 clears it, Tab switches lists, Up/Down moves to the next ROI, Space toggles *use*; a comment per ROI; the keywords become columns of the results table, so Filter and Sort read them. **Still missing from SMAP's version: several named lines per ROI.** There is one *direction* line now. Extending it to named lines (angle and length as values) is what E7 needs for orientation. |
| E3 | **Segment regions** with presets (NPC, Density Peaks, CME, beads, other templates); see B | #104 #105 #109 #111 | decided: one plugin, see B. NPC and Density Peaks exist as separate Find plugins, to be folded in. *Centre all* (#116) fits as a Regions action: recentre the ROIs by a measured centre (median, circle fit), with the circle fit from E5. |
| E4 | No distinction between evaluator and analysis plugin | #118 | **done** (`per_region`). Left: Line Profile itself is not `per_region` yet (it is `follow_roi`). Giving it `Result.values` (width, distance, AIC winner) is a small change. |
| E5 | **Circle fit** as a `per_region` Measure plugin (#115, P2); **NPC geometry** (#114, P3; independent of LocMoFit, but better as one of its models: decide H first) | #114 #115 | open |
| E6 | Figures of regions: montage, TIFF stack, single TIFFs, volumes, sorted by any column | #132 #133 | open, P3. Now a Summarize-style plugin that reads the ROI order (Sort ROIs) and annotations. |
| E7 | **Average ROIs**: centre plus an orientation from a measured value or a drawn line → a new table | #134 | open, P3. Depends on the named lines in E2. |

## F. Single-particle tracking (P1, own project)

Unchanged, with one simplification from the structure: a **track is a
region**.  Linking writes `track_id` (an id column, like `cluster_id`), so
every `per_region` plugin runs per track for free, and *Column to ROIs* makes
ROIs of tracks to review them.  What the project still needs: a linker (look
for an optimized algorithm), then the analyses (MSD, diffusion, jump
distances) as `per_region` measurements plus summaries over tracks, and tracks
drawn into a movie export (#100 #101 #141).

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

z-stack export (#76) · Voronoi renderer (#77) · movie export (#139 #140 #141)
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
| **9** | **Neighbourhood**: per localization, neighbours within r (local density), mean/median/min/max/quantile of a field within r, Voronoi cell density; written as recipe columns | `Analysis/Process/Neighbourhood` (module-level functions on a `cKDTree` and `scipy.spatial.Voronoi`) | M | densities of a simulated uniform field match the expected count; page with figures |
| **10** ✅ | **Find clusters**: one plugin, an algorithm choice with one part per algorithm (only the selected one shown): DBSCAN (moved from `clusters.py`), Voronoi threshold (density from step 9, then connected cells), HDBSCAN if scikit-learn ≥ 1.3 is installed (an optional dependency, asked for at Run like uiPSF) | `Analysis/Regions/Find/Find clusters`; `Find/DBSCAN` removed | M | DBSCAN's tests pass through the new plugin; Voronoi finds simulated clusters; `docs/regions.md` ("Clusters", "The plugin tree") and CLAUDE.md's shape table updated; an old chain step `Regions/Find/DBSCAN` still loads (path alias) |
| **11** ✅ | **Segment regions**: one plugin with presets, one part each, only the selected shown: NPC (moved from `npc.py`), Density Peaks (moved from `roi.py`); a `SegmenterPlugin` as now (Preview, *replace*, ROI undo). Templates such as CME and beads come later as further presets. | `Analysis/Regions/Find/Segment regions` | M–L | NPC and Density Peaks tests pass through it; `npc_analysis.chain.yaml` and the NPC tutorial use it; old paths alias; docs as in step 10 |
| **12** ✅ | **Cluster shape**: area, convex hull, radius of gyration, density, elongation, as a `per_region` measurement (the statistics a block over `cluster_id` does not yet give) | `Analysis/Measure/Cluster shape` | S | known shapes return their numbers |

Steps 10 and 11 share the old-path aliasing and the docs: do them
one after the other, in the same session if possible.

### Wave 3: the ROI manager's remaining pieces

| step | what | size | needs |
|---|---|---|---|
| **13** | **Named lines per ROI**: the one *direction* becomes any number of named lines, drawn in the ROI manager, each with angle and length as values in the results table, and a reverse action | M (GUI, ROI project file format) | – |
| **14** | **Small ROI tools**: remove double ROIs (closer than d: keep the larger); remove localizations outside the included ROIs (a third *region* choice of Remove Localizations); keep the ROIs inside the drawn ROI (a *Filter ROIs* condition on position) | S | – |
| **15** | **Line Profile `per_region`** (width, distance, AIC winner as values) and **Circle fit** (`Analysis/Measure/Circle fit`, `per_region`), with *centre ROIs* on the circle or median centre as a Regions action | S–M | – |
| **16** | **Average ROIs**: centre plus orientation from a measured value or a named line → a new table | M | 13 |

### Wave 4: analysis projects (a short design note first, in `docs/`)

| step | what | size |
|---|---|---|
| **17** | **Spatial statistics** (B3): Ripley K/L/H, pair correlation, 2D auto- and cross-correlation between layers, 1D correlation along a line; `per_region` | M–L |
| **18** | **Counting** (B4): per-region blinks and dark times plus a Summarize histogram fit | M |
| **19** | **Single-particle tracking** (F): linker writing `track_id`; MSD, diffusion and jump distances as `per_region` measurements and summaries; tracks in a movie export | L |

### Wave 5: when the above is in

Transformations C1 + C2 (Transform, Register layers) · drift D1–D3 (apply
existing, fiducials, magnification in COMET) · Vibrations (A6, once its
filters and zoom-ins are settled) · AssignColors for more than 2 colours (C4)
· figures of regions (E6) · LocMoFit decision (H) and then the port · MINFLUX
(G), after F · rendering and movies (I).

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

### Start here

Wave 1 is eight independent sessions.  If only one runs at a time, start with
**step 1**, because waves 2 and 3 build on it.  Then go on with 2 and 3
(export and readers together), then 4, 6 and 7.
