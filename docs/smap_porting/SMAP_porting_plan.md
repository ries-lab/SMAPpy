# SMAP → smappy: work plan

From your annotations in `SMAP_porting_list.xlsx`, read loosely: I went by the
comments more than by the Decision column, and treated "change" as "work on"
where the comment asks for work.  Row numbers (`#12`) are the `#` column of the
spreadsheet; its new **Work package** column says where each row went.

The work splits into **small plugins** you can hand out one by one (A), a few
**projects** that need a short design first because your comments merge several
SMAP plugins into one general tool (B–E), and **own projects** to plan from
scratch (F–H).

---

## A. Small plugins — P1, can start now

| | what | rows | notes |
|---|---|---|---|
| A1 | **Export** — one plugin with a format choice: CSV/TXT, Picasso HDF5 (and SMAP `_sml.mat` as one more format, low priority); choose the fields; all or filtered | #53 #54 (#55) | You marked `SMLMsaver` *skip* but wrote "SMAP can be another export" under #54 — I read that as: no separate plugin, an optional format of this one. |
| A2 | **Loaders**: Picasso HDF5, DECODE | #51 #52 | Small, symmetric with A1. |
| A3 | **Fit region** (`RoiAdder`): fit only inside / exclude drawn regions of the chip | #7 | A mask on the candidates. |
| A4 | **Histogram of any column**, per layer, with mean/median/std | #86 | |
| A5 | **Time course**: any columns against frame (photons, locs/frame, precision, background, …), several in one graph, merged with Image Tags | #87 | Extending `image_tags.py` is the obvious home — it already draws tags against frame with locs/frame on a second axis. |
| A6 | **Vibrations**: spectrum of x, y, z (beads), with zoom-ins and filters | #88 | You raised it to P1. Questions for when it starts: which filters (band-pass on the trace?), which zoom-ins (time window, frequency band)? |

## B. Neighbourhood, clusters, spatial statistics — P1, one project, three plugins

Your comments on #70 #90–#96 #119 fit together as:

| | plugin | does | rows |
|---|---|---|---|
| B1 | **Neighbourhood** (rename of `SpatialFilterLocs`) | Looks at the neighbourhood of every localization and writes a column: number of neighbours within r (= density, `density_calculator`), mean/median/min/max/quantile of a field, Voronoi cell density. | #70 #93 (density) #94 |
| B2 | **Cluster** | Several algorithms behind one form: DBSCAN, Voronoi threshold (SharpViSu), maybe HDBSCAN, and template-based (the ring filter the NPC segmenter uses). Writes `cluster_id`; a *shuffle colours* button that also works on an existing clustering; a per-cluster table and statistics. Runs on the selection or as an ROI evaluator. | #91 #93 (clusters) #95 #96 #120 |
| B3 | **Spatial statistics** | Ripley K/L/H, pair correlation, 2D image auto- and cross-correlation, the cross variants between two layers, and SMAP's 1D correlation along a line. | #90 #92 #119 (structure part) |
| B4 | **Counting / general statistics on ROIs and clusters** | Locs and blinks per cluster/site, histogram fits for protein number, labelling efficiency, re-activations. You called it a larger project: design after B2, because it runs on B2's clusters as well as on ROIs. | #98 #121 |

Answer to "not sure what `correlation` does" (#90): it has two modes.
**1D** along a line ROI: the profile, its autocorrelation and FFT, with a guessed
period, averaged over many ROIs. That is for periodic structures, such as the
~190 nm spectrin rings.
**2D**: pair correlation of the localizations between layers.
Both belong in B3.

B2 and the unified segmenter (E3) share algorithms (template/ring filter,
density peaks).  Write the algorithms once, as module-level functions, and let
both plugins call them.

## C. Transformations and channels — P2

| | what | rows | notes |
|---|---|---|---|
| C1 | **Transform**: one plugin for every geometric change of a table *or an image layer*: apply a saved transformation (from Calibrate transform), shift (typed in, or taken from C2), mirror/flip, rotate, scale/pixel size | #40 #69 #74 | On "plugin or a shift option on the layer?" (#74): I'd make it a plugin, plus an *apply* button on the registration result. A shift kept on a layer only moves the picture; every analysis plugin reads the table and would see the unshifted data. |
| C2 | **Register layers**: shift (x, y, z) between two layers — rendered images or loaded images — by cross-correlation; the result feeds C1 | #41 (second part) #119 (shift part) | Reuses `rcc.py`'s correlation. |
| C3 | **Channel alignment over time** (diagnostic): channel 1 vs channel 2 against frame, also x1/y1 vs x2/y2 from a dual-channel fit with unlinked positions | #41 (first part) #42 | P3; it becomes more relevant if channel drift is added to the dual fitter. |
| C4 | **AssignColors for more than 2 colours**: find more than one minimum to split at, inside the existing plugin | #47 | P2. |
| later | single-channel fits → dual colour by pairing localizations across channels (transformation → n1, n2) | #46 | P3, your comment. |
| later | Calibrate transform from separate camera files | #39 | When needed. |

## D. Drift — P2

| | what | rows |
|---|---|---|
| D1 | **Apply existing drift** from another file (the curve `drift_result` already stores) | #35 |
| D2 | **Fiducial drift** (beads) | #34 |
| D3 | **Time-dependent transformation in COMET**: magnification (and more) changing over time; RCC later if feasible | #38 |
| later | Picasso drift file, only when requested | #36 |

## E. ROI manager — P1/P2

**E1 Housekeeping**: small plugins, P1 first.

| what | rows | your priority |
|---|---|---|
| Filter sites by an expression over evaluation results (reuse `mathparse`) | #127 | P1 |
| Sort sites (sites only, no cells) | #128 | P1 |
| Export the site table: the user picks the columns | #130 | P1 |
| Remove double sites | #129 | P2 |
| Remove localizations outside sites | #107 | P2 |
| Site number / an evaluation value → localization column | #131 | P2 — to decide: is the site number written by default? |
| Use only the sites inside the drawn ROI | #106 | P3 |
| Import site positions | #135 | P3 |

**E2 Annotate ROIs**: new plugin, P2 (your addition).  What SMAP's
`SEAnnotation` does: four lists of labels, each loadable from a text file;
the selected entry is stored on the site; with several sites selected, the
label goes to all of them; a free comment per site; two lines drawn per site,
shown as angle and length, with a button that reverses a line's direction.  Keys:
←/→ go to the previous or next site, Shift+←/→ switch between lists.

A smappy version:
* label lists defined in the plugin's settings (name + entries) and saved with
  the project;
* keyboard: ←/→ for the site, number keys pick an entry of the active list, Tab
  or Shift+arrows switch lists;
* any number of named lines per ROI (generalising the one *Direction* the ROI
  manager already has), each with angle and length and a reverse action;
* labels, line angles and line lengths become columns of the site table.
  Filter (E1), Sort, Histograms, Export and Average (E7) then use them like any
  evaluation result.

This needs keyboard focus and drawing in the ROI manager window, so it is a GUI
change, not only a plugin: make it a panel of the ROI window (or `Plugin.window`,
the way Bead calibration is opened).

**E3 One segmenter with presets**: P2.  NPC (current), CME, beads, cluster-based
(B2), density peaks, each preset with its own model and parameters, plus
*centre all* (recentre the ROIs by whatever describes the centre: median, circle
fit, template).  Rows #104 #105 #109 #111 #116.

**E4 Drop the line between evaluator and analysis plugin**: P2.  Any analysis
plugin can run on the ROIs of the ROI manager; an evaluator is "the same plugin,
run per ROI".  Line Profile is the first case (#118), and B2 the second.  This is
a framework change: the plugin contract, the evaluation pipeline,
`docs/plugin-architecture.md`, and the CLAUDE.md lines on `scope = "site"`.  It is
worth doing **before** E5–E7 and B2's evaluator mode, so they are written once.

**E5 Circle fitter** as an evaluator (#115, P2).  **NPC geometry** (#114, P3).
Your question was whether NPC geometry is independent of LocMoFit, and it is.
SMAP's `NPCgeomtryQuantify` has its own ring-centre fit (`fitposring`), a 3D
template fit (`runNPC3DfittingJ`), a two-Gaussian fit of the z histogram for the
ring distance, and an angular cross-correlation of the two rings for the twist.
It can be ported alone, but LocMoFit's NPC model measures the same quantities
better.  My suggestion: decide H first; if LocMoFit comes, NPC geometry is one
of its models.

**E6 Figures of sites**: P3.  One plugin that exports selected sites as a
montage, a TIFF stack, individual TIFFs or 3D volumes, sorted by any column
(#132 #133).

**E7 Average sites**, simple: P3.  Centre plus an orientation from any
evaluation result or from a line drawn in E2 → a new table (#134).

## F. Single-particle tracking — P1, own project

One plugin, or two under the hood with the tracking callable from inside the
analysis (#100 #101): linking (look for an optimized algorithm, not just
nearest neighbour), then analysis options to pick and plot (MSD, diffusion
coefficients, jump distances, …).  Tracks drawn into a movie export (#141).
Design first: which analyses, and how users extend them.

## G. MINFLUX — P2, own project

Stepping, dual-track analysis, tracks to movie, and the MINFLUX analysis
plugins (#103 #152).  Plan from scratch what is needed now; take from SMAP only
what is used.  F and G should share the track model.

## H. LocMoFit — P2, own project

Port SMAP's LocMoFit, or build on Weiqing's newer approach (#117 #138).  That
decision comes first; NPC geometry (E5) and averaging by model (E7) depend on it.

## I. Rendering and movies — P3

| what | rows |
|---|---|
| Export a z-stack, from the 3D render | #76 |
| Voronoi renderer (comes nearly free with B1/B2) | #77 |
| Movie export: running window, raw + SR side by side (and tracks, F) | #139 #140 #141 |
| later: simple image processing for image layers (background subtraction, local normalisation) | #81 |

## J. Later / when needed

GPU fitter (#10) · background estimation before fitting (#6) · astigmatic
Gaussian z (#11) · 4Pi (#13) · DECODE and DECODE-plex integration, simplifying
from the DECODE side (#15) · sCMOS pixel maps, potentially (#22) · uiPSF more
robust, compare with Yiming's implementation, own project (#24) ·
depth-dependent aberrations (#26) · GainOffsetFFT (#27) · DHPSF (#29) ·
Register3Dstacks (#44) · combine files into one file number with continuous
frames (#65) · grouped table as the primary one, e.g. for smaller files (#66) ·
rename/remove fields (#68) · alternating excitation, flagged in your comment
(#71) · intensity of a TIFF at sites, likely needed (#124).

**GroupAny (#67): what is it for?**  Nothing else in SMAP calls it.  It merges
all localizations that share a value of a chosen field (per file and channel)
into one, through SMAP's normal grouping.  That gives one row per track, per
cluster or per site.  In smappy that is `group.combine` with a different key.
smappy's grouping already does exactly this for MINFLUX's trace id `tid` (`group.py`).  So it needs no
plugin of its own: it becomes a "one row per cluster / track" option in B2 and F.

---

## Suggested order

1. **A1–A5**: independent, a day or two each, and they can run in parallel.
2. **E1**, the P1 part (filter, sort, export).
3. **E4**, the evaluator/analysis unification, before the ROI-heavy work.
4. **B1 → B2 → B3**, then B4.
5. **F**: design note, then implementation.
6. **C1 + C2**, **D1–D3**, **E2**, **E3**.
7. **H** decision → LocMoFit; G after F.
8. A6 can slot in anywhere once its filters and zoom-ins are settled.

Before a project (B, E3, E4, F, G, H) starts, it gets a short design note in
NOTES.md or `docs/`.  That is where the open questions above get settled.
