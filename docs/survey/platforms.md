## The large general platforms: PYME, GDSC SMLM, locan / napari-locan, viewers and commercial tools

Surveyed: python-microscopy (PYME: recipes, cluster, PYMEVisualise), GDSC SMLM (Fiji), locan and
napari-locan (the only SMLM napari plugin with substance; no `napari-storm` or `napari-smlm` exists under
those names -- napari-moltrack is a thin Picasso/trackpy/GPUfit wrapper), with ViSP, Genuage and the
commercial tools from their documentation only.  Main conclusion: on the core algorithms SMAPpy is
already level or ahead (its FRC, MSD, spatial statistics, 3D view and batch fingerprinting are better
than the equivalents read here).  The worthwhile gaps are a handful of **quantitative-counting and
cluster-validation methods** (the PC-PALM model fit, Annibale's blink estimator with Coltharp's
threshold choice, coordinate-based colocalisation, label-density/accumulation and convex-hull cluster
checks, residence times) and three **UX ideas from PYME**: the interactive session *is* a re-runnable
recipe, multi-input / parallel batches, and analysis chained onto a live acquisition when it completes.

### The software

| tool | what | language | licence | port route | repo (commit read) |
| --- | --- | --- | --- | --- | --- |
| PYME (python-microscopy) | acquisition + fitting + recipes (DAG) + distributed cluster + PYMEVisualise (VisGUI, OpenGL) + dh5view | Python, C, Cython | GPL-3 overall (`COPYING`); a few files BSD by header (`contrib/cpmath`, `util/shmarray`, `misc/sqlite_ns.py`) -- none of the analysis below | re-implement from paper/description | python-microscopy/python-microscopy `7adbde6` (2026-09-17) |
| GDSC SMLM | ~120 Fiji plugins: fitting, tracing, blinking, PC-PALM, OPTICS, TC-PALM, diffusion, benchmarking, camera models | Java | GPL-3 (`LICENSE.txt`, every file header) | re-implement from paper/description | aherbert/gdsc-smlm `a817fbc` (2026-10-13) |
| locan | Python library: LocData, analysis classes (CBC, Ripley, accumulation, convex-hull expectation, subpixel bias, FRC/FSC...), pipelines | Python | BSD-3 (`LICENSE.md`) | port with attribution | super-resolution/locan `8703cad` (2026-10-17) |
| napari-locan | napari widgets over locan (render, ROI, cluster, collections, project save, script runner) | Python | BSD-3 | port with attribution | super-resolution/napari-locan `e77faba` (2025-05-13) |
| ViSP, Genuage | 3D / VR point-cloud viewers | Java / Unity C# | not read | ideas only | -- |
| ONI CODI, Bruker Vutara SRX, Nikon N-STORM, Abberior | commercial | -- | closed | ideas only, from documentation *(unverified; ONI's help pages were unreachable, only search snippets)* | -- |

### Worth porting  (ranked, most valuable first)

#### PC-PALM model fit of g(r) -- GDSC SMLM  (S-M, GPL: re-implement from Sengupta 2011 / Veatch 2012)
What it does (`docs/pc_palm_plugins.rst:420-526`, `ij/plugins/pcpalm/PcPalmFitting.java`): g(r) of the
*molecules* (blinks, not merged across dark frames) is fitted as g = g_stoch + g_protein, with
g_stoch = exp(-r²/4σ²)/(4πσ²ρ) (the self-correlation of one molecule's repeated localisations, σ the
precision, ρ the protein density) and three nested models: *random* (g_stoch + 1), *random clusters*
((A exp(-r/ξ) + 1) ⊗ g_PSF, with an occupancy N_cluster) and *emulsion* ((A exp(-r/α) cos(πr/2r₀) + 1) ⊗ g_PSF).
Fitting is least squares by bounded CMA-ES with restarts then a gradient polish; r below k·σ is
ignored; the random model is rejected when g - g_stoch exceeds a threshold, and a clustered model is
accepted only if its fitted σ and ρ stay within a percentage of the estimates and adjusted R² improves.
SMAPpy instead: `plugins/spatial_statistics.py` computes g(r), K, L with exact window covariance and
Monte Carlo envelopes, including a *keep blinks* null, and says explicitly "Sengupta's PC-PALM fit of
the blinking is not ported" (`spatial_statistics.py:61`).  Why: the envelopes say *whether* the data are
clustered; the fit says *how* (domain size, molecules per cluster, fraction clustered) with the
blinking self-term removed -- the number biologists report.  Fit: a `fit_pcpalm(r, g, sigma, rho)`
module function and a *model* part in Spatial statistics; the per-region `Result.arrays` already give
pooled curves to fit.  Test data: `smappy.simulate` clusters with known radius and blinking.

#### Blink estimator and grouping-threshold optimisation -- GDSC SMLM  (S, GPL: from Annibale 2011, Coltharp 2012)
What it does (`ij/plugins/BlinkEstimator.java`): traces the localisations with distance 2.5 × mean
precision (`searchDistance = 2.5`, `relativeDistance`, lines 108-118) and a dark-time threshold t_d for
every t_d = 1..80 frames (`calculateCounts`, ~line 400: one `traceMolecules` per t_d), then fits
N(t_d) = N (1 + n_blink exp((1 - t_d)/t_off)) (line 782) by Levenberg-Marquardt on the first 5 points
(`numberOfFittedPoints = 5`), sweeping the number of fitted points to show how stable N, n_blink and
t_off are, and warns when the curve does not asymptote above the data.  *Trace Molecules → Optimise*
(`docs/analysis_plugins.rst:424-503`) then scans distance × time thresholds on a geometric grid and
scores (traces - expected)/expected with expected = pulses / blinking rate, picking the zero-score point
nearest the origin (Coltharp 2012).  SMAPpy: `plugins/blinking.py` counts blinks/bursts at one chosen
*dark time* and states the counting fit is "not here"; `group.connect` (`group.py:355`) is the linker
needed.  Why: it turns "what dark time do I merge with?" into a measured answer and gives the copy
number without a reference sample.  Fit: a function `blink_curve(x, y, frame, distance, max_dark)`
that calls `group.connect` per t_d (80 C++ passes), a fit function, and an *approach* of Blinking.
Test: simulated blinking with known n_blink and t_off (`simulate/kinetics.py`).

#### Coordinate-based colocalisation (CBC) -- locan  (S, BSD: port with attribution)
What it does (`locan/analysis/cbc.py:45-116`): for each localisation of A, the neighbours of A and of B
within R are counted on n_steps = 10 radii, normalised as cumsum · R²/r²/N_R, Spearman-correlated, and
the coefficient multiplied by exp(-d/R).  Note: locan takes d as the *largest* neighbour distance
(`cbc.py:110-113`); Malkusch et al. weight with the distance to the *nearest* B neighbour -- port the
paper's version and say so.  SMAPpy has layer cross-correlation g_AB(r) (`spatial_statistics.py`) but no
per-localisation colocalisation score.  Fit: a *quantity* of Neighbourhood (`plugins/neighbourhood.py`,
which already has *around* / *neighbours from* layers and C++ neighbour search), written as a recipe
column, so the histogram and the per-region median come for free.  Test: two simulated layers, one
offset copy of the other (CBC → 1) and independent (→ 0).

#### Label-density / temporal accumulation cluster check -- locan  (S-M, BSD)
What it does (`locan/analysis/accumulation_analysis.py`): on subsets of increasing size (random
subsets, or the first k frames for Spahn's temporal version) it clusters (HDBSCAN by default), and
plots η (area fraction covered by clusters, bounding-box or convex hull) against ρ (density inside
clusters) versus the overall density; a random distribution with overcounting and a truly clustered one
give different curves (Baumgart 2016; Spahn 2016).  locan marks it "exploratory".  SMAPpy: nothing
equivalent; *Find clusters* (`plugins/clusters.py`) gives the clusters, *time windows* exist.  Why: the
standard artefact-free answer to "are these nanoclusters real or blinking?" and complements the keep-
blinks null.  Fit: a `per_region` plugin under Analysis/Regions that loops `clusters` over frame
prefixes.  Test: simulate CSR-with-blinking vs. clusters.

#### Convex-hull expectation diagnostic -- locan  (M, BSD)
What it does (`locan/analysis/convex_hull_expectation.py`): for clusters, plots hull area (2D) or
volume/area (3D) against localisation count n and compares with the expectation for n points drawn
from a Gaussian of σ = precision -- an analytic integral in 2D (`compute_convex_hull_region_measure_2d`,
line ~73) and shipped lookup tables otherwise (`resources/convex_hull_expectation/*.npy`) -- to separate
resolution-limited single molecules from extended structures and to estimate the precision (Ebert
2022).  SMAPpy: `plugins/cluster_shape.py` measures shapes but has no expectation.  Fit: a curve
overlay on the cluster-shape scatter; regenerate the tables by simulation rather than copy.

#### Residence-time analysis -- GDSC SMLM  (S, GPL: from the description)
What it does (`docs/analysis_plugins.rst:1964-2010`, `ResidenceTimeAnalysis.java`,
`smlm/fitting/ResidenceTimeFitting.java`): tracks of bound molecules (jump filter), track durations fitted
by MLE to P(T ≥ t) = Σ w_i exp(-k_i t), 1-3 populations, each observation weighted by the survival
difference over its frame (discrete time), first/last-frame censoring, and k_obs = k_b + k_a with a
user-given photobleaching rate k_a, τ = 1/(k_obs - k_a).  SMAPpy: `tracking_analysis.py` has MSD and jump
mixtures but no dwell times.  Fit: an *approach* of Tracking analysis.  (The SPT survey may cover this
better; included because it is small and complete here.)

#### Jittered-triangulation and quad-tree renderers -- PYME  (M, GPL: from Baddeley 2010)
What it does (`PYME/LMVis/renderers.py:447-559`, `visHelpers.py:457-540`): n renderings in which every
point is jittered by a per-point σ (the precision, the nearest-neighbour distance, or their minimum:
`_genJitVals`, `renderers.py:174-187`), Delaunay-triangulated, each triangle filled with 1/area,
averaged; optional Monte Carlo point dropping (`MCProbability`); seeds stored in the output metadata so
the image is exactly reproducible.  Quad-tree: adaptive histogram, leaves split until ≤ N points.
SMAPpy's render modes are hist, gauss, precision, dl, tracks, voronoi (`render.py:584`).  Why: the
paper shows Gaussian rendering loses resolution and invents structure in sparse regions; triangulation
suppresses detail where sampling is poor.  Fit: two more `render.py` modes; triangulation via
`scipy.spatial.Delaunay`, per-triangle rasterisation in C++ beside the existing kernels.

#### OPTICS with instant re-cutting and LoOP -- GDSC SMLM  (M, GPL; or scikit-learn's BSD OPTICS)
What it does (`ij/plugins/Optics.java`, `docs/analysis_plugins.rst:2211-2226`): computes the OPTICS
ordering once, then re-extracts clusters at any DBSCAN distance ≤ the generating radius or by the ξ
steepness method without recomputing (lines 1473-1519), shows the reachability plot linked to the image,
offers FastOPTICS (random projections, no radius) and per-point Local Outlier Probability (LoOP) as a
colour.  SMAPpy: DBSCAN, Voronoi, HDBSCAN (`plugins/clusters.py`).  Why: the interactive part --
drag a threshold on the reachability plot and see the clusters change -- is the value; HDBSCAN already
covers the algorithm.  Fit: an *algorithm* part of Find clusters with a reachability `Plot` whose
threshold is a `ShapeEdit` line; preview cached.

#### TC-PALM cumulative counts per region -- GDSC SMLM  (S, GPL: from Cisse 2013)
`TcPalmAnalysis.java:505-553`: for a drawn ROI, cumulative localisations vs time; bursts found by time
linkage with a *dark time tolerance* and *min cluster size*; selecting a time range on the plot shows
those localisations on the image (and vice versa).  SMAPpy's Blinking already finds bursts per region;
add the cumulative plot and the plot→view time selection (the view's time filter).

#### Subpixel-bias check -- locan  (S, BSD)
`locan/analysis/subpixel_bias.py`: histogram of coordinate modulo pixel size, the pixel-locking
diagnostic (Gould 2009).  One more figure in Statistics.

#### Spherical-harmonic shells and point-cloud meshes -- PYME  (L, GPL)
`recipes/surface_fitting.py:422-560`, `Analysis/points/spherical_harmonics.py:53-160`: fit a nucleus-like
shell as spherical harmonics up to order 3 with outlier rejection (tolerance 0.3, 2 iterations), add
`inside_shell` and `distance_to_shell` columns; isosurfaces from an octree (dual marching cubes) or Delaunay
tetrahedra, meshes as layers, STL export (Marin 2021).  Niche for SMAPpy's users (nuclear envelope,
whole-cell 3D) and large; listed for completeness.

### Better than SMAPpy's current implementation

* **Motion blur in MSD and jump fits -- GDSC SMLM** (S, GPL: from Michalet 2010 / Backlund 2015).
  GDSC fits MSD(n) = 4DΔt(n - 1/3) + 4s² (`TrackPopulationAnalysis.java:4142-4158`,
  `TraceDiffusion.java:1876-1956`, derivation and simulation in `docs/msd_correction.rst`): a position
  averaged over a continuously illuminated exposure lowers every MSD point by 4DΔt/3 (2D).  SMAPpy's
  `tracking/analysis.py:27-29` says plainly that motion blur is not modelled, so with the offset taken
  *from the precisions* (its default) D is biased low by up to a third at lag 1 for fast molecules.  Add
  R = 1/6 (continuous) or 0 (stroboscopic) as a setting in the model, `2dDΔt(k - 2R) + 2Σσ²`, in both the
  MSD and the jump-distance likelihood; the optimal-lag covariance code already takes the model.
* Compared and SMAPpy is as good or better: FRC (GDSC `Fire.java` corrects spurious correlation with a
  fitted Q; SMAPpy's time-block split, `frc.py:18-33`, avoids it at the source and reports the blur
  envelope); drift (PYME autocorrelation and fiducial modules vs RCC/COMET/fiducials); MSD lag choice
  (SMAPpy's exact optimal lags beat GDSC's fixed windows); DBSCAN (PYME, GDSC); Ripley (PYME
  `pointcloud.py:259`, locan -- SMAPpy's window covariance and envelopes are more correct); qPAINT dark-time
  CDF (PYME `Analysis/points/qpaint.py`: a 1- or 2-exponential fit to the upper envelope of the gap CDF --
  simple; see the Picasso section for qPAINT).

### User experience and automation

* **The interactive session is the recipe** (PYME).  In PYMEVisualise the pipeline *is* a `Recipe`
  (`LMVis/pipeline.py:303`), shown in a *Pipeline Recipe* tab, editable, saved as YAML and re-run on other
  files (`VisGUI.py:193-216`, `-r recipe.yaml`); it is also written into every output's metadata
  (`visCore.py:747-750`).  SMAPpy logs the same information (`plugins/history.py`) but cannot turn it back
  into something runnable.  Proposal: *History → Save as chain*: the logged runs that changed the table,
  with their settings, become a `.chain.yaml` (measurements that ran too can be offered as ticks), and the
  batch window opens on it.  This is the shortest road from "I did it by hand once" to a batch.
* **Lazy, incremental re-execution with rollback** (PYME `recipe.py:180-285, 494-528`): a DAG over a
  namespace; only modules downstream of a changed input re-run, and a module that fails is removed with
  its outputs.  SMAPpy's chain is deliberately linear (`docs/batch.md`), but *re-running a chain step-by-
  step from the first changed step*, keeping earlier steps' tables, would make tuning a late step in a
  long chain cheap in the GUI; the fingerprint machinery in `batch.py` already says which steps changed.
* **Several inputs per job and parallel files** (PYME bakeshop, `recipes/batchProcess.py:35-60`): named
  inputs, each a glob list, zipped by position (`{'ch1': [...], 'ch2': [...]}`) and run on
  `cpu_count()` processes with an SVG backend.  Several inputs per dataset is in SMAPpy's own *Not
  yet* list (`docs/batch.md`), as a question for every plugin that reads files, not only the batch;
  PYME's zip-by-index form is the simplest schema for "two stacks per cell".  Parallel files were
  decided against (2026-10-28): the analysis is parallel already.
* **Analysis chained onto acquisition** (PYME `cluster/rules.py:1-35, 376-382, 786-811`): a localisation
  rule releases frames as the spooler writes them, and `on_completion` chains recipe rules (render, then
  measure) with input patterns like `{{spool_dir}}/analysis/{{series_stub}}.h5r`; Barentine 2023 reports
  10,000 cells/day this way, including queueing new acquisitions from results.  SMAPpy's `LiveSession`
  (`gui/live_session.py`) fits live and hands back the file; adding an optional *chain to run when the
  writer finishes* (a `.chain.yaml` path in the MicroClaw call, run as a one-file batch in a subprocess
  with `results.json`) would give MicroClaw a measured answer per field without a person.
* **Portable session files** (PYME `visCore.py:865-910`, `sessionpaths.py`): `.pvs` YAML of data sources,
  recipe, layers and view, with paths rewritten relative to a `SESSIONDIR` token when they lie under the
  session file's folder.  SMAPpy saves the GUI (`workspace.py`) and the table's history, not "this
  analysis state of these files"; a session file referencing the `.h5` files plus ROI project, layers and
  view would let a figure be reopened months later.
* **Reports per file** (PYME `recipes/output.py:397-537`, `ReportForEachOutput` with Jinja templates in
  `PYME/reports/templates`): an HTML page per input with its numbers and figures.  SMAPpy writes
  `results.json` and PNGs per file; one generated `index.html` in the batch output folder (a row per file:
  thumbnails of each figure, the summary scalars, status) would make a 50-cell batch browsable.
  *Built 2026-10-28* (`smappy/batch_page.py`).
* **Recipes look good in notebooks** (PYME `recipe.py:960`, `_repr_mimebundle_` → SVG of the graph) and
  rules show live status tables in Jupyter (`rules.py:80-120`).  A `_repr_html_` for `Chain`,
  `Localizations` and `Result` (columns, history, the figures) is cheap and helps the scripting users.
* **An embedded Python console** with the session in its namespace (PYME `VisGUI.py:120-130`,
  napari-locan's *Run script* widget with shipped example scripts).  SMAPpy has none; a qtconsole dock
  exposing `session`, `ctx` and `smappy` is a common power-user request.
* **Configuration templates with example data** (GDSC *Template Manager*, `docs/fitting_plugins.rst:1675`):
  named fit configurations, some shipped with an example image to try them on.  SMAPpy has YAML presets
  for *Segment regions*; the same preset mechanism for the fitters (per camera / PSF / dye) and a
  *File > Open example* (napari-locan's sample-data menu: NPC and tubulin) would shorten onboarding.
* **Recordable dialogs** (GDSC through ImageJ's macro recorder; every plugin's settings are macro
  options): SMAPpy's equivalent is the history→chain idea above, plus a *Copy as Python* on a plugin panel
  (`plugins.get(path)().run(Context(locs=...), Settings(...))`).
* **Browse a collection** (napari-locan `widgets/widget_render_collection_2d.py`): a cluster/ROI
  collection rendered as a stack with napari's slider.  This is SMAPpy's open E6 (figures of regions,
  `SMAP_porting_plan.md:134`); a slider-through-ROIs view in the ROI manager sorted by *Sort ROIs* is the
  interactive half of it.
* **Brushing between plots and the image** (GDSC TC-PALM and OPTICS: select in the plot, see the
  localisations; select in the image, see the plot range).  SMAPpy's figures can take shapes
  (`Plot.edit`); letting a range drawn on a histogram set a filter on the view would generalise it.
* **Live preview of filters with immediate counts** (ONI CODI, *(unverified, from search snippets)*):
  HDBSCAN as the default clustering, apps per workflow (clustering & counting, EV profiling) applied in
  batch to datasets grouped by condition, with condition-level comparison plots.  SMAPpy's `summary.csv`
  has no notion of *condition*; a `condition:` key per input in `.batch.yaml` and a pooled per-condition
  summary (box plots) would cover the "Pooling across files" item of `docs/batch.md`.
* **VR / immersive 3D** (Genuage, Blanc 2020; ViSP, El Beheiry 2013): SMAPpy's wgpu engine
  (`gpu.py`) already does sprites and shaded spheres; an export to a standard point-cloud format (PLY with
  colour) lets Genuage/ParaView/Blender users take it further at no cost.

### Licence-restricted (re-implement only)

* PC-PALM fitting (GDSC, GPL-3): Sengupta 2011 gives the three models; Veatch 2012 the over-counting
  self-term and the cross-correlation argument.  Model selection by adjusted R² is GDSC's own.
* Blink estimator and threshold optimisation (GDSC, GPL-3): Annibale 2011 (the N(t_d) model), Coltharp
  2012 (score and grid).
* Residence-time MLE, Track Population Analysis (sliding-window features + Gaussian mixture), MTT-style
  dynamic tracing (GDSC, GPL-3): the GDSC docs give the equations; MTT from Sergé 2008.
* MSD motion-blur term (GDSC, GPL-3): from Michalet 2010 / Backlund 2015, standard.
* OPTICS with ξ extraction, FastOPTICS, LoOP (GDSC, GPL-3): Ankerst 1999 and Kriegel 2009; scikit-learn's
  BSD `OPTICS` is a usable base.
* TC-PALM (GDSC, GPL-3): Cisse 2013.
* Jittered triangulation, quad-tree rendering (PYME, GPL-3): Baddeley 2010 has both algorithms.
* Spherical-harmonic shells, octree meshing (PYME, GPL-3): Marin 2021 and the module docstrings.
* Recipe DAG, cluster rules, bakeshop (PYME, GPL-3): ideas only -- nothing to copy is needed.

### Not worth it / already in SMAPpy

* PYME/GDSC/locan DBSCAN, Ripley, nearest-neighbour, local density, Voronoi -- SMAPpy's `clusters.py`,
  `spatial_statistics.py`, `neighbourhood.py` are equal or better.
* PYME autocorrelation/fiducial drift, GDSC *Drift Calculator* -- RCC, COMET, fiducials in SMAPpy.
* PYME vibration spectrogram (`LMVis/Extras/vibration.py`) -- SMAPpy's `stability.py` is fuller.
* Rendering provenance in image metadata (PYME `Rendering.*` keys) -- SMAPpy's `picture.describe_view`
  writes the view into TIFF/PNG/JPEG already.
* Events-based frame flagging (PYME `flagZMotionArtifacts.py`) -- SMAPpy's frame tags (`image_tags.py`).
* GDSC Results Manager / in-memory named result sets -- SMAPpy's session files and layers.
* GDSC camera tools (mean-variance, EM-gain PMF, sCMOS maps) and the fitting benchmark/filter-optimisation
  workflow -- belong to the camera/fitting survey; the idea of tuning fit and filter settings against
  `smappy.simulate` with `ground_truth.py` as the score is worth keeping in mind.
* PYME multiview/ratiometric colour, DeClump -- SMAPpy's dual-colour assignment and grouping cover them.
* napari as a second GUI -- SMAPpy's Qt GUI is far ahead of napari-locan; interoperability (export a layer
  as a napari Points/Image) is the most that would pay.
* locan FRC/FSC, localization precision, localisation property correlation -- SMAPpy has FRC and precision.
* Commercial tools (Vutara SRX, N-STORM, Abberior Imspector, NimOS): no documented algorithm beyond
  DBSCAN/HDBSCAN-type clustering and standard rendering could be verified.

### References

* Annibale P et al. (2011) Quantitative photo activated localization microscopy: unraveling the effects of
  photoblinking. PLoS One 6:e22678. https://doi.org/10.1371/journal.pone.0022678 (blink estimator)
* Coltharp C, Kessler RP, Xiao J (2012) Accurate construction of PALM images for quantitative
  measurements. PLoS One 7:e51725. https://doi.org/10.1371/journal.pone.0051725 (threshold optimisation)
* Sengupta P et al. (2011) Probing protein heterogeneity in the plasma membrane using PALM and pair
  correlation analysis. Nat Methods 8:969-975. https://doi.org/10.1038/nmeth.1704
* Veatch SL et al. (2012) Correlation functions quantify super-resolution images and estimate apparent
  clustering due to over-counting. PLoS One 7:e31457. https://doi.org/10.1371/journal.pone.0031457
* Malkusch S et al. (2012) Coordinate-based colocalization analysis of single-molecule localization
  microscopy data. Histochem Cell Biol 137:1-10. https://doi.org/10.1007/s00418-011-0880-5
* Baumgart F et al. (2016) Varying label density allows artifact-free analysis of membrane-protein
  nanoclusters. Nat Methods 13:661-664. https://doi.org/10.1038/nmeth.3897
* Spahn C et al. (2016) Temporal accumulation analysis provides simplified artifact-free analysis of
  membrane-protein nanoclusters. Nat Methods 13:963-964. https://doi.org/10.1038/nmeth.4065
* Ebert V et al. (2022) Convex hull as diagnostic tool in single-molecule localization microscopy.
  Bioinformatics 38:5421-5429. https://doi.org/10.1093/bioinformatics/btac700
* Doose S (2022) LOCAN: a python library for analyzing single-molecule localization microscopy data.
  Bioinformatics 38:2670-2672. https://doi.org/10.1093/bioinformatics/btac160
* Baddeley D, Cannell MB, Soeller C (2010) Visualization of localization microscopy data. Microsc
  Microanal 16:64-72. https://doi.org/10.1017/S143192760999122X (jittered triangulation, quad-tree)
* Marin Z et al. (2021) PYMEVisualize: an open-source tool for exploring 3D super-resolution data. Nat
  Methods 18:582-584. https://doi.org/10.1038/s41592-021-01165-9
* Barentine AES et al. (2023) An integrated platform for high-throughput nanoscopy. Nat Biotechnol
  41:1549-1556. https://doi.org/10.1038/s41587-023-01702-1 (PYME cluster, analysis during acquisition)
* Cisse II et al. (2013) Real-time dynamics of RNA polymerase II clustering in live human cells. Science
  341:664-667. https://doi.org/10.1126/science.1239053 (TC-PALM)
* Sergé A et al. (2008) Dynamic multiple-target tracing to probe spatiotemporal cartography of cell
  membranes. Nat Methods 5:687-694. https://doi.org/10.1038/nmeth.1233
* Blanc T et al. (2020) Genuage: visualize and analyze multidimensional single-molecule point cloud data
  in virtual reality. Nat Methods 17:1100-1102. https://doi.org/10.1038/s41592-020-0946-1
* El Beheiry M, Dahan M (2013) ViSP: representing single-particle localizations in three dimensions.
  Nat Methods 10:689-690. https://doi.org/10.1038/nmeth.2566
* Not verified here (cited by the code; DOIs not checked): Ankerst et al. 1999 (OPTICS, SIGMOD), Kriegel et
  al. 2009 (LoOP, CIKM), Backlund et al. 2015 (MSD with blur, cited in GDSC docs), Michalet 2010 (cited in
  SMAPpy's `tracking/analysis.py`), Gould et al. 2009 Nat Protoc (subpixel bias, cited by locan).
* DOIs above were checked in PubMed.
