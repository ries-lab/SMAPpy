# SMAP → smappy: work plan

From your annotations in `SMAP_porting_list.xlsx`, read loosely: I went by the
comments more than by the Decision column.  Row numbers (`#12`) are the `#`
column of the spreadsheet; its **Work package** column says where each row
went, and **Update 2026-10-14** what the regions restructuring changed.

**Updated 2026-10-14** against `origin/main` `c9f650a`, after the regions
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

## B. Neighbourhood, clusters, spatial statistics (P1)

| | what | rows | status |
|---|---|---|---|
| B1 | **Neighbourhood** (the renamed `SpatialFilterLocs`): per localization, the neighbours within r (density), the mean/median/quantile of a field, Voronoi cell density, written as a column | #70 #93 #94 | open. A column writer like Math Parser; not a region. |
| B2 | **Clusterers**: DBSCAN is **done** (`Regions/Find/DBSCAN`). What is left: Voronoi threshold, maybe HDBSCAN, template-based (the NPC ring filter), and a *shuffle colours* action | #91 #93 #95 | open. **Decision for you:** your comment asked for one cluster plugin with several algorithms. The tree now has one Find plugin per algorithm (Find/DBSCAN, Find/NPC, Find/Density Peaks), all writing an id column. Adding more Find plugins fits that; one plugin with a choice would go against it. I'd follow the tree. *Shuffle colours* then belongs to the render side (a LUT option for id columns) and works for every id column, not only clusters. |
| B2b | **Statistics on clusters**, also inside ROIs | #96 #120 | **done** by the structure: a block over `cluster_id` with Localization Statistics, nested in an ROI block if wanted. A dedicated cluster measurement (area, convex hull, radius of gyration, density) is still missing: one small `per_region` plugin. |
| B3 | **Spatial statistics**: Ripley K/L/H, pair correlation, 2D auto- and cross-correlation between layers, the 1D correlation along a line | #90 #92 #119 | open. Make it `per_region`: then it runs in every ROI (spectrin periodicity per ROI) with no extra work. |
| B4 | **Counting**: localizations and blinks per region, histogram fits for protein number, labelling efficiency, re-activations | #98 #121 | open, but now clear how. A `per_region` measure plugin (blinks and dark times per region) plus a `Summarize` plugin that fits the histogram, the same pair as NPC Corners and NPC labelling efficiency. It works on ROIs and on clusters alike. |

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
| E3 | **One segmenter with presets** | #104 #105 #109 #111 | **partly answered by the structure**: finders and clusterers are one kind (`Regions/Find/…`). Open: presets (NPC, CME, beads) and whether they are one plugin or several Find plugins (the same question as B2). *Centre all* (#116) fits as a Regions action: recentre the ROIs by a measured centre (median, circle fit), with the circle fit from E5. |
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

## Suggested order

1. **A1, A2, A4, A5, A7**: independent, small, and they can run in parallel.
2. **E2 named lines** and the small E1 leftovers (remove doubles, remove
   locs outside ROIs).
3. **B1**, then **B3** and **B4** as `per_region` measurements, and the
   remaining **B2** clusterers once the one-plugin-or-several question is
   answered.
4. **F** (design note first; tracks as an id column).
5. **C1 + C2**, **D1–D3**, **E3 presets**, **E5**, Line Profile `per_region`.
6. **H** decision → LocMoFit; G after F.
7. A3 and A6 whenever they are wanted.
