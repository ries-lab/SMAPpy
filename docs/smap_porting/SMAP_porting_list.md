# SMAP → smappy: what is there, what is missing, what to port

> **Snapshot of 2026-10-11, before the regions restructuring.**  The current
> state is in `SMAP_porting_list.xlsx` (column *Update 2026-10-14*) and
> `SMAP_porting_plan.md`.

A list to make choices from, not a plan.  Compiled 2026-10-11 from `jries/SMAP`
(`plugins/`, 345 plugin files, plus `fit3Dcspline`, `LocMoFit` and the GUI) and
smappy at `origin/main` (`eeae43b`).

**Status:** ✅ in smappy · ◐ partly, or done another way · ✗ missing ·
⊘ left out on purpose (NOTES.md says why)
**Suggestion:** **P1** port soon · **P2** worth porting · **P3** only if someone
asks · **skip** project-specific, superseded or dead
**Effort:** S = a day or less · M = a few days · L = a week or more (with page and tests)

Plugins that are one-off project code (CME, yeast, fibrils, platelets,
kinetochore, `summarize_*`, `*_old`, `*_forVilma`, `TestTabPlugin`) are grouped
at the end and not rated one by one.

---

## Summary: what I would do first

The bulk of SMAP's day-to-day use is already there: fitting (Gaussian, spline,
2-channel global), bead calibration, uiPSF v2 (its window in progress), drift (COMET, RCC), rendering
with layers/filters/grouping and any-column axes, 3D view, ROI manager with
NPC, line profiles, statistics, FRC, simulation and ground truth, history,
batch.  The large gaps are **whole areas**, not single plugins:

| # | area | what to port | why first |
|---|------|--------------|-----------|
| 1 | **Clustering & spatial statistics** | DBSCAN, Ripley's K/L and pair correlation, Voronoi density, per-localization neighbour density | Nothing at all exists, and it is the most common analysis people outside the lab do. Algorithms are small (scipy/sklearn). |
| 2 | **Export for other tools** | CSV/TXT export with column choice, Picasso HDF5, SMAP `_sml.mat` save | Users need to get data *out*; also eases the transition from SMAP. All S. |
| 3 | **Particle averaging + LocMoFit** | `AverageSites` (simple) now; LocMoFit (model fitting of sites) as a larger project | LocMoFit is a lab method and a reason to use the ROI manager at all. |
| 4 | **ROI-manager housekeeping** | filter sites by expression, sort, remove duplicates, gallery/montage, export site table, site id → localization column | Small each; together they make the ROI manager usable for real projects. |
| 5 | **Generic histograms / stats vs time** | histogram of any column; photons, precision, locs/frame, … vs frame | Basic QC people expect. S each. |
| 6 | **Drift extras** | fiducial (bead) drift, apply drift from another file | Fiducials are still common; "apply existing" is trivial given `drift_result`. |
| 7 | **Single-particle tracking** | linking into tracks (reuse `group.py`), MSD / diffusion analysis | Whole area missing; moderate effort; skip uTrack. |
| 8 | **Depth-dependent aberration correction** | `calibrate3Daberrations` / `correct3Daberrations` (Li et al. 2018) | Needed for quantitative 3D in thick samples; lab method. |

Simplifications that make sense across the board are listed per row; the
recurring ones are: one plugin where SMAP had several near-duplicates
(`FilterFields1–4`, `Get2CInt*`, `CME2CSide*`), express things through the math
parser and `group.combine` instead of dedicated plugins (rename/remove fields,
group by any field, site id columns), and run external deep-learning tools the
in their own Python (a worker script reporting on stdout, as the uiPSF v1 bridge did,
retired October 2026: `git show 0164d42:src/smappy/uipsf/runner.py`) instead of porting them.

---

## 1. Localization (fitting workflow)

SMAP's `WorkflowModules` are replaced by `pipeline.py` + `plugins/fit.py`; the
module chain itself is ⊘ (NOTES.md, "No module chain").

| SMAP | does | smappy | suggestion |
|---|---|---|---|
| `TifLoaderParallel`, `TifLoader`, `CameraConverter` | read TIFF / MM stacks, ADU→photons | ✅ `io/tiff`, `ndtiff`, camera db + editor | – |
| `fileFeeder` | fit many stacks in a row | ✅ batch | – |
| `cutoutImages`, `WFsplitter*` / `WFcombiner*` | several channels on one chip | ◐ 2-channel split handled by the 2C fitters | skip (3-channel only if needed) |
| `ImageFilter`, `PeakFinder` | DoG/Gauss filter, local max, cutoffs | ✅ `detect.py` (DoG, dynamic cutoff) | NMS and probability cutoff ⊘ – fine |
| `ImageNormalize*` (Anscombe), probability map | | ⊘ | – |
| `BG_wavelet`, `MedianBGcalculator`, `fitOnBackground`, `Roi_bg` | background estimate before fitting | ⊘ | **P3**, M – only for strong, structured background (autofluorescent tissue). Wavelet à-trous is ~50 lines if ever needed. |
| `RoiAdder` | fit only inside / exclude drawn regions of the chip | ✗ | **P2**, S – a region mask on candidates; cheap and saves bad fits on dirt/edges. |
| `RoiCutterWF` | cut ROIs | ✅ `roi.py` | – |
| `RoiCutterWF_group(Ext)` | sum frames of one blink before fitting | ✗ | **P3**, M – grouping after fitting covers most of the gain. |
| `MLE_GPU_Yiming` | Gaussian / spline MLE (GPU) | ✅ C++ CPU; ◐ no GPU fitter | GPU fitter **P3** – only if CPU speed is a real limit. |
| astigmatic Gaussian z (sx/sy → z), `Sxsy2z*` | 3D without a spline calibration | ⊘ (Gaussian free fits sx, sy) | **P3**, S–M – spline covers it; useful only for users who have only an sx/sy calibration. |
| `MLE_global_spline` | multi-channel global fit | ✅ 2C (Gaussian and spline) | – |
| `MLE_4Pi`, `Loader4PiMat`, `Channeldrift4Pi`, `RegisterLocs4Pi`, `Phase2z4Pi` | 4Pi | ✗ | skip unless 4Pi data comes back into use (L). |
| `iterativeMDfitter` | multi-emitter-aware single fitter | ✗ | **P3**, M – for medium density; DECODE is the better answer. |
| `DECODE_fitting`, `Loader_decode`, `DECODE_training_estimates` | deep-learning localization | ✗ | **P2**, M – run DECODE in its own Python (external worker + import of its output). Don't port the training glue as is. |
| `deepSMLM`, `TrainDeepSMLM`, `FalconHD2D` | other high-density localizers | ✗ | skip |
| `RadialSymmetry2D/3D` | fast non-MLE localizer | ✗ | **P3**, S – nice for beads/tracking; not for SMLM precision. |
| `usePeakfinder`, `pushLocs` | pass positions through | n/a | skip |
| `LocFilter` | filter before saving | ◐ display filter / Chain Layers | skip |
| `OnlineReconstruction`, `PlotLocsPreview`, `autoLocSaver` | live view, preview | ✅ live fit, fit preview | – |
| `GrabFijiStacks`, `Arrayfeeder` | | ✗ | skip |
| sCMOS per-pixel variance/offset maps (`calibrateCMOS`) | per-pixel noise model | ⊘ (one variance per chip) | **P3**, M – revisit only if a camera with bad pixels shows bias. |

## 2. Calibration (PSF, camera, 3D)

| SMAP | does | smappy | suggestion |
|---|---|---|---|
| `calibrater3D_so` / `fit3Dcspline` | cspline PSF from beads | ✅ Bead calibration window | – |
| `learnPSF_invmodeling` | PSF by inverse modelling | 🔶 uipsf (uiPSF v2, `src/uipsf/`); its *PSF calibration* window is being built (`docs/uipsf/plan.md` §7.5) | – |
| `CheckCalibration` | fitted vs true bead z | ✅ `calibrate/validation.py` | – |
| `calibrate3Daberrations`, `correct3Daberrations`, `CorrectDepthDependentOffset` | depth-dependent PSF / z-scaling correction (Li et al. 2018) | ✗ | **P2**, M–L – needed for quantitative z deep in the sample. Simplify to: calibrate a z-correction map from beads in gel → apply as a table operation. |
| `GainOffsetFFT` | camera gain/offset from data (Heintzmann 2016) | ✗ | **P2**, S–M – feeds the camera database when the conversion factor is unknown. |
| `intensity_ratios_channels`, `ratio_field` | channel intensity ratios / field dependence | ◐ dual-colour calibration | **P3** |
| `calibrateDHPSF`, `fitDHPSF`, `getThresholdDHPSF` | double-helix PSF | ✗ | skip |
| `calibrateSALM`, `SALM*`, `zSALM`, `GetIntensitiesSALM` | supercritical-angle z | ✗ | skip (unless SALM is used again) |
| `Biplane_getPSFxy`, `gradientEllipticity3D2z` | | ✗ | skip |

## 3. Drift correction

| SMAP | does | smappy | suggestion |
|---|---|---|---|
| `driftcorrectionXYZ` | cross-correlation drift, x/y/z | ✅ RCC; COMET as default | – |
| `showdriftinfo` | show saved drift | ✅ drift result redrawn | – |
| `driftcorrection_beads`, `Analyse_Drift` | fiducial-based drift | ✗ | **P2**, M – segment bright persistent spots, average their tracks, smooth (reuse COMET's spline). |
| `applyexistingdriftcorrection` | apply drift of file A to file B (e.g. second channel) | ✗ | **P1/P2**, S – `drift_result` already stores the curve; needed for sequential 2-colour. |
| `applyPicassoDriftcorrection` | apply Picasso drift file | ✗ | **P3**, S |
| `driftcorrection_dme` | entropy-based drift (DME) | ✗ | skip – COMET is the same class of method. |
| `driftcorrectionM` | drift + magnification change | ✗ | skip |

## 4. Channels: registration and colour assignment

| SMAP | does | smappy | suggestion |
|---|---|---|---|
| `RegisterLocs2` | transformation from localizations (beads, ratiometric) | ✅ Calibrate transform (split chip) | ◐ two *cameras*/two files not covered – see next row |
| `ApplyTransform`, `CombineChannels`, `AttachTransformation` | apply a transform to the localizations of one channel/file, merge paired locs | ✗ (the transform is used inside the 2C fit only) | **P2**, S–M – one plugin "Apply transform" (choose file/channel, optionally average paired locs). Needed for two-camera setups and sequential imaging. |
| `RegisterImages`, `XYZdisplacement`, `Zdisplacement`, `correlationAnalysis`, `shiftCrossCorrelation` | shift between two layers by cross-correlation | ✗ | **P2**, S – reuse `rcc.py`'s correlation; one plugin with x/y(/z). Same function as a site evaluator later. |
| `QuantifyChannelDrift` | relative drift between channels vs frame | ✗ | **P3**, S |
| `FrameDependentTransformation`, `Transform2C_interleaved` | | ✗ | skip |
| `Register3Dstacks` | stitch z-stacks of slabs into whole-cell 3D | ✗ | **P3**, M – only if whole-cell 3D by slab-stepping is done. |
| `Intensity2Channel` | colour from intensity ratio | ✅ AssignColors (from global fit) | – |
| `Get2CIntImages2cam`, `Get2CIntImagesWF`, `Get2CIntLoc`, `roi2int_*`, `EvaluateIntensity_s`, `IntLoc2posN`, `Loc2pos` | read intensities at transformed positions (salvaged fluorescence) | ⊘ replaced by the global 2C fit | skip – keep the global-fit route. |
| `Intensity2ManyChannels` | >2 dyes from ratios | ✗ | **P3**, S – generalise AssignColors to k clusters if ≥3-colour ratiometric is used. |

## 5. Files: load, save, export

| SMAP | does | smappy | suggestion |
|---|---|---|---|
| `Loader_auto`, `Loader_sml`, `Loader_csvAndMore`, `Loader_minflux(_json)` | | ✅ Auto, SMAP, csv, MINFLUX, smappy HDF5 | – |
| `Loader_tif`, `Loader_sml_Tiff`, `Rescale_tiff` | associate a widefield TIFF | ✅ image layer (pixel size, x0, y0) | – |
| `Loader_settings`, `saver_settings`, `Loader_workflow` | GUI settings | ✅ workspace / chains | – |
| `Loader_decode` | DECODE output | ✗ | with DECODE (§1), or the csv loader with a preset mapping, S |
| (Picasso HDF5 *loading*) | not in SMAP | ✗ | **P2**, S – symmetric with the export below; many users have Picasso data. |
| `export_coordinates` | CSV/TXT/XLS, chosen columns, all or filtered | ✗ | **P1**, S |
| `export_picasso_hdf5` | Picasso format | ✗ | **P1/P2**, S |
| `SMLMsaver` | save `_sml.mat` | ✗ | **P2**, S–M – lets people go back to SMAP for what is not ported yet. |
| `TifSaver` | rendered image as TIFF | ✅ Export/Image | – |
| `VISPexport`, `sendDataToVR` (Genuage) | 3D viewers | ✗ | **P3**, S – CSV export with the right columns probably covers it. |
| `PushToFiji` | | ✗ | skip – export TIFF |
| `seSaver`, `generalSeSaver` | site-explorer data | ✅ ROI project storage | – |
| `Loader_ioPlugin` | | ✗ | skip |

## 6. Table operations (`Process/Modify`)

| SMAP | does | smappy | suggestion |
|---|---|---|---|
| `MathParser`, `Mathematics` | new field from an expression | ✅ Math Parser | – |
| `RemoveLocs`, `keepfiltered` | | ✅ Remove Localizations (ROI or selection, remove or hide) | – |
| `FilterFields`, `FilterFields2/3/4` | apply a filter as a step | ✅ Chain/Layers | – |
| `DefineMainCoordinates` | use other fields as x/y/z | ✅ render axes | – |
| `CombineFiles` | merge loaded files into one table | ◐ files appended with `filenumber` | **P3**, S – check what is missing (saving one merged file?). |
| `Connect2Unconnect`, `CorrectUnconnectbyConnect` | grouped ↔ ungrouped coordinates | ◐ grouped table + `group_id` on both | skip |
| `GroupAny` | combine locs with the same value of any field | ✗ | **P2**, S – expose `group.combine` with a chosen key (track id, cluster id, site id); falls out of clustering/tracking. |
| `RenameFields`, `RemoveFields` | | ✗ | **P3**, S – one "Edit columns" plugin. |
| `Mirror`, `ReverseAxis`, `TransformLocsManually`, `ChangeCamPixelsize` | geometric edits | ✗ | **P3**, S – one "Transform table" plugin (flip, rotate, shift, scale/pixel size). |
| `SpatialFilterLocs` | local mean/median/quantile of a field around each loc | ✗ | **P2**, S – a neighbourhood column for colouring (KD-tree). Pair with neighbour density (§9). |
| `AlternatingExcitationCorrect`, `ImposeSymmetry`, `rotSymCMERec` | | ✗ | skip |

## 7. Rendering and viewing

| SMAP | does | smappy | suggestion |
|---|---|---|---|
| render modes hist / Gauss / precision Gauss / DL, colour by field, LUTs, layers, contrast, scale bar | | ✅ Render tab | – |
| `VersatileRenderer` | any field vs any other | ✅ render axes | – |
| per-layer `shiftxy` (Channel tab) | shift a layer for display | ✗ | **P3**, S – becomes unnecessary once "Apply transform"/"layer shift" (§4) exists. |
| `Viewer3DV01`, `sideview`, `Phase2z4Pi` | 3D view, side view | ✅ 3D view (slab, rotation) | – |
| `volume3D`, `volume3D_hist` | voxel rendering, export as stack | ✗ | **P3**, M – a "render z-stack" export (slices of the 2D renderer) is the useful 80%. |
| `VoronoiRenderer` | Voronoi-density image | ✗ | **P3**, S – comes almost free with Voronoi clustering (§9). |
| `Show_Raw_Data`, `DisplayImageTags(Tiff)` | raw frames, image tags | ✅ raw frames, Image Tags | – |
| `SetTiffLocation` | re-point to moved raw data | ◐ | **P3**, S |
| `ImageIntensities2Locs`, `intensityTiff` | TIFF intensity at each loc / site | ✗ | **P3**, S – a column from an image layer; trivial once asked for. |
| `Tiff_remove_BG` | | ✅ Process image (wave 5, step 6d) | – |

## 8. Measurement and quality control

| SMAP | does | smappy | suggestion |
|---|---|---|---|
| `Locstatistics` | photons, precision, on-time | ✅ Localization Statistics | – |
| `FRCresolution` | FRC | ✅ (in Localization Precision) | – |
| `lineprofile` (+ROI evaluator) | line profile with models | ✅ Line Profile (unbinned, AIC) | ◐ as a site evaluator: **P3** |
| `CompareToGroundTruth` | | ✅ Ground Truth | `CompareToGroundTruthChallenge`: skip |
| `Histograms` | histogram + stats of any field | ◐ only the four fixed ones, and ROI Histograms | **P1**, S – any column, per layer, with mean/median/std. |
| `StatsVsTime` | photons, locs, precision, bg … vs frame or file | ◐ Image Tags shows locs/frame | **P1**, S – the first thing to look at when an acquisition degrades. |
| `Vibrations` | spectrum of x/y/z of beads | ✗ | **P3**, S |
| `ImageDecorrResolution` | decorrelation resolution | ✗ | skip – SMAP's own help says it is unsuitable for SMLM. |
| `BackgroundEstimation`, `correlation`, `storerois` | | ✗ | skip |

## 9. Clustering and spatial statistics — nothing exists yet

| SMAP | does | smappy | suggestion |
|---|---|---|---|
| `DBSCAN_cluster` | DBSCAN, cluster id per loc + cluster table | ✗ | **P1**, S–M – `sklearn`/`scipy` KD-tree; write `cluster_id`, return a per-cluster table (n, area, radius, density). |
| `ClusterStatistics`, `SpatialPointPattern` (and the MINFLUX copy) | Ripley's K/L/H, pair correlation (Sengupta 2011) | ✗ | `ClusterStatistics` computes only cluster shapes despite its description: that is ✅ `Measure/Cluster shape`. The spatial statistics: **P1/P2**, M – one plugin; edge correction and a CSR envelope by simulation; in a ROI. Pair correlation needs the blinking correction to be honest – say so on the page. |
| `ClusterVoronoi` | Voronoi-cell density, clusters by threshold (SharpViSu) | ✗ | **P2**, M |
| `density_calculator` | neighbours within r per loc | ✗ | **P2**, S – a column; filterable, colourable. |
| `ShuffelClusterIndex` | random colour per cluster | ✗ | S – part of DBSCAN (write a shuffled id). |
| `clusterAnalysis` (ROI evaluator) | DBSCAN/k-means/GMM per site | ✗ | **P2**, S once DBSCAN exists – one evaluator, DBSCAN and GMM only. |
| `dyeblinking` | on-times, off-times, blinks per molecule, per file | ✅ | `Measure/Blinking` (step 18): the three histograms per file; the photons and on-time against a series of dark times, the on-fraction and the molecule's duration left out. |

## 10. Counting (molecules per cluster)

| SMAP | does | smappy | suggestion |
|---|---|---|---|
| `cmakehist`, `cloadhist`, `cfithist`, `csegmentcluster`, `locsfromSE`, `countingStatistics(_kinetochore)` | histogram of locs/blinks per cluster, fit with a model of protein number, maturation efficiency and re-activations | ✗ | **P2**, M – merge into one ROI analysis: evaluator counts blinks per site (needs grouping), analysis fits the histogram. Same pattern as NPC labelling efficiency. |
| `NPCLabelingQuantify_s`, `NPCLabelingEfficiency`, `GetExposureTime` | | ✅ NPC Corners, NPC Labeling Efficiency | – |

## 11. Single-particle tracking — nothing exists yet

| SMAP | does | smappy | suggestion |
|---|---|---|---|
| `SimpleTracking` | link locs into tracks | ✗ | **P2**, S–M – reuse `group.py`'s linking (it is the same nearest-neighbour link with larger radius/gap); write `track_id`. |
| `AnalyzeSPT`, `DisplaySingleTrack`, `Single_tracking_analysis` | MSD, diffusion coefficients, jump distances, track viewer | ✗ | **P2**, M – MSD and jump-distance fit per track and pooled; track overlay in the render view. |
| `uTrack` | external uTrack | ✗ | skip |
| `Dual_tracking_analysis(_old)`, `Tracks_to_movie`, MINFLUX `Segment_*tracks`, `MotorPAINT`, `StepsMINFLUX*` | co-tracking, MINFLUX steps | ✗ | skip (project code) – revisit if MINFLUX tracking is a focus. |

## 12. ROI manager

### Segment

| SMAP | smappy | suggestion |
|---|---|---|
| `segmentNPC` (+`NPCsegmentCleanup`) | ✅ Segment/NPC | – |
| `SiteAutoDetect`, generic density segmenters | ✅ Density Peaks | – |
| `UseSitesInROI` | ✗ | **P2**, S – keep sites inside the drawn ROI. |
| `RemoveLocsOutsideSites` | ✗ | **P2**, S – shrink a file to its sites. |
| `files2sites`, `makeCellGrid`, cells as a level | ✗ (no cells) | **P3**, M – cells are SMAP's middle level; probably not needed if files ~ cells. |
| `segmentBeads` | ◐ bead calibration has its own | skip |
| `SimulateSites` | ✅ Simulate structure | – |
| `segmentCME*`, `segmentCells*`, `segmentSites*`, `fibrilFinder`, `CMESiteFinder_eq`, `Make3Dshape`, `proteinES_ctrl` | ✗ | skip (project) |

### Evaluate

| SMAP | smappy | suggestion |
|---|---|---|
| `generalStatistics`, `clusterStatistics` | ✅ Evaluate/Statistics | – |
| `NPCLabelingQuantify_s` | ✅ NPC Corners | – |
| `NPCgeomtryQuantify` (+ Analyze `NPCGeometry`) | ✗ | **P2**, M – radius, ring distance, twist; NPC as a ruler (3D calibration check). |
| `circleFitter` | ◐ circle fit inside the NPC segmenter | **P2**, S – expose as an evaluator. |
| `CenterSites`, `CenterPoints` | ✗ | **P2**, S – recentre ROIs (median / circle fit / correlation). |
| `LocMoFitGUI`, `SMLMModelFit*`, `modelFitter`, `WeiMoFit`, `fit_displayer*` | ✗ | **P1/P2**, L – see §13. |
| `lineprofile` | ◐ Line Profile is not a site evaluator | **P3**, S |
| `shiftCrossCorrelation`, `correlationAnalysis` | ✗ | with §4 registration, S |
| `clusterAnalysis` | ✗ | with §9 |
| `countingStatistics` | ✗ | with §10 |
| `LocIndices` | n/a (performance hack) | skip |
| `beadanalysis`, `beadCalibrationParameters` | ◐ bead calibration window | skip |
| `selectCoordinates2Color`, `annotatePeaks`, `plot3D_annotate`, `movieframes`, `assignDist`, `gtLLExp`, `intensityTiff`, CME/fibril/MINFLUX evaluators | ✗ | skip |

### Analyze

| SMAP | smappy | suggestion |
|---|---|---|
| `BatchAnalyze`, `EvaluateSites` | ✅ chains/batch, NPC chain | – |
| ROI histograms | ✅ Analyze/Histograms | – |
| `FilterSites` | ✗ | **P1**, S – expression over evaluation results; reuse `mathparse`. |
| `SortROIs`, `SortCells` | ✗ | **P2**, S |
| `RemoveDoubleSites` | ✗ | **P2**, S |
| `SaveSiteParameters`, `ExportEvaluationsTable`, `SaveEval` | ◐ batch writes `summary.csv`; no site-table export from the window | **P1**, S – export the site table (CSV). |
| `Sitenumbers2loc`, `Eval2loc` | ✗ | **P2**, S – site id / an evaluation value as a localization column (then filter, colour, `GroupAny`). |
| `roiMontage`, `SMLMModelFit_gallery`, `siteGallery_mCME` | ✗ | **P2**, S–M – one "Gallery" figure of selected sites, sorted by a value. |
| `export_tiffs`, `export_3Dvolumes`, `exportrotatedsites` | ✗ | **P3**, S |
| `AverageSites` (+`_NPC`, `_window`, `ByDistance`, `_ModleFit`, `_kinetochore`) | ✗ | **P2**, M – one plugin: align by centre (+ optional rotation by cross-correlation), sum into a new table; model-based alignment comes with LocMoFit. |
| `ImportSitePositions` | ✗ | **P3**, S |
| `RunAnalysisScript` | ✗ | skip – a Python plugin file is the script |
| `Fisheyeview`, `mkMovieByFilter`, `splitAvrageByRank`, `analyzeCME*`, `analyzeNPC*`, `summarize_*`, `yeastCME_QC`, `cellLevelQC`, `CME_DC`, `PlotRingCME`, `AnalyzeRingCME`, `updatePosRot_temp`, `clearFailAnnotatedPeaks`, `beforeAvg_NPC3Dfit_multiC`, `LocMoFit_*` export/gallery | ✗ | skip, or with LocMoFit |

## 13. Model fitting of sites (LocMoFit) and averaging

`SMAP/LocMoFit` (Wu et al. 2023) fits geometric/image models to the
localizations of each site, with model selection, and drives averaging and
dynamic reconstruction.  It is the largest single thing not ported.

**Suggestion: P1/P2, L.**  Port the *core* only: a model is a density
function with parameters (ring, double ring, sphere/cap, line, point; plus a
user-defined Python function), fitted to the site's localizations by unbinned
likelihood with the localization precision folded in — the same machinery
`line_profile.py` already uses in 1D. Then: an evaluator (`scope = "site"`)
with the model as a setting, results per site, and `AverageSites` aligned by
the fitted pose.  Leave out the GUI model builder, the image-model branch and
the multi-step pipelines until someone needs them.

## 14. Movies and presentation

| SMAP | does | suggestion |
|---|---|---|
| `MovieRunningWindow`, `mkMovieByFilter` | sliding-window reconstruction movie (live cell) | **P3**, S–M – one "Export movie" plugin (render frame windows → mp4/TIFF stack). |
| `BlinkingMoviePresentation` | raw frames beside a building-up SR image | **P3**, S – same plugin, a "raw + SR" layout option; uses kept raw frames. |
| `makeMovieTiff`, `Tracks_to_movie` | | skip |

## 15. Infrastructure and GUI

| SMAP | smappy |
|---|---|
| `ShowHistory` | ✅ History |
| `BatchAnalysis`, `Batchprocessor` | ✅ chains + `smappy-batch` |
| Undo | ✅ |
| SearchHelp / plugin help | ✅ pages, Help window |
| Camera settings (`cameras.mat`) | ✅ camera database + editor, reads SMAP's `*_cameras.mat` |
| Tutorials | ✅ (SMAP has none in the GUI) |

---

## Project-specific / not rated (skip unless the project returns)

* **CME / yeast:** `CME2CSide*`, `CME2D*`, `CME3DDSpherefit*`, `CMEeqViewCellBound`,
  `segmentCME*`, `segmentCells*`, `segmentSites*`, `analyzeCME*`, `summarize_yCME`,
  `yeastCME_QC`, `rotSymCMERec`, `SMLMModelFit_dynamicRec_mCME`, `siteGallery_mCME`,
  `PlotRingCME`, `AnalyzeRingCME`, `AnalyzeCME2Cside`
* **Fibrils / BALM:** `fibril*`, `BALM_fibril_growth`, `boundaryFinder`
* **Platelets, polarization, misc. projects:** `Platelets_*`, `PlotPolarization`, `Y_fit`
* **Kinetochore:** `AverageSites_kinetochore`, `countingStatistics_kinetochore`
* **MINFLUX analysis:** `ClusterStatistics_MINFLUX`, `DCoffset_Beads_ROIs`,
  `Localizationnumber`, `MINFLUX_timings_scouting`, `Tails_analyze`,
  `Cluster_MINFLUX(_Roi)`, `StepsMINFLUX*`, `MotorPAINT`, `Segment_*tracks`
  (loading MINFLUX is ✅; the generic parts — clustering, tracking — come with §9 and §11)
* **Model-fit summaries:** `summarizeModFit*`, `summarize_NPCModelSelection*`,
  `summarize_MT3D`, `summarizeLocprecTitration`, `NPC3Dfitting*`
* **Duplicates / dead:** `*_old`, `*_forVilma`, `SimulateSitesYL`, `TestTabPlugin`,
  `FilterFields2–4`, `sendDataToVR` (×2)
