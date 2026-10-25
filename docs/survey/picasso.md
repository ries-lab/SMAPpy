## Picasso and its satellites (Jungmann lab)

Surveyed: Picasso 0.11.3 (`jungmannlab/picasso`, the whole Python package:
`aim.py`, `g5m.py`, `spinna.py`, `clusterer.py`, `postprocess.py`, `average.py`,
`scmos.py`, `localize.py`, `plugins.py`, `server/`, the Render GUI's menus), against
SMAPpy at `b0a6722`.  Picasso is **MIT throughout** (Gpufit/Gpuspline ports
carry their own MIT licences in `LICENSES/`), so everything below can be ported
with attribution.  The main conclusion: Picasso's value for SMAPpy is in the
**DNA-PAINT-specific quantification layer** that SMAPpy does not have at all --
qPAINT counting, G5M molecular mapping, SPINNA oligomer-state fitting, the SMLM/RESI
clusterer with its "frame analysis" sticking filter -- plus AIM as a fast drift
estimator.  On drift (RCC, fiducials), NeNA, pair correlation and file exchange,
SMAPpy is already equal or better.

### The software

| tool | what | language | licence | port route | repo (commit read) |
| --- | --- | --- | --- | --- | --- |
| Picasso (Localize, Render, Filter, Average, Average3, Design, Simulate) | DNA-PAINT pipeline: fit, render, picks, drift, clustering | Python (numba, numba-CUDA) | MIT | port with attribution | jungmannlab/picasso `a1e639a` (0.11.3, 2026-10-01) |
| AIM (in Picasso, `aim.py`) | fiducial-free drift by intersection maximisation | Python/numba | MIT (Picasso's implementation; authors incl. Ma, Liu) | port with attribution | same |
| G5M (`g5m.py`) | modified Gaussian-mixture molecular mapping | Python/numba, sklearn-derived | MIT (BSD-3 sklearn heritage) | port with attribution | same |
| SPINNA (`spinna.py`, GUI `gui/spinna.py`) | oligomer stoichiometry from NND vs simulation | Python | MIT | port with attribution | same |
| RESI / SMLM clusterer (`clusterer.py`, `postprocess.resi`) | local-maximum clusterer, sequential-imaging cluster centres | Python | MIT | port with attribution | same |
| Picasso Server (`server/`) | Streamlit dashboard, SQLite file database, folder watcher | Python | MIT | port with attribution | same |
| Nanotron (`nanotron.py`) | MLP classifier of rendered picks | Python/sklearn | MIT | port with attribution | same |

### Worth porting  (ranked, most valuable first)

#### qPAINT: binding-site counts from dark times -- Picasso Render  (S, MIT)
What it does.  Per pick (any ROI), `postprocess.pick_kinetics`
(`postprocess.py:3042`) links *all* localizations of the pick into binding events
(`link(..., r_max=999999, max_dark_time=3)`, `:3013-3040`), computes each event's
bright time `len = last - first + 1` and its dark time to the *closest preceding
event end in the same pick*, `first - last_prev - 1` (`dark_times`,
`:3233`; the numba double loop `_dark_times`, `:3291`; this convention changed in
0.11.3, see changelog, and is the same as SMAPpy's `blinking.py`).  Mean bright
and dark times come from fitting the ECDF with `a(1 - exp(-x/t)) + c` by
`scipy.optimize.curve_fit` with bounds `t in [min, max]` (`lib.fit_cum_exp`,
`lib.py:1024`; `estimate_kinetic_rate`, `:1056`, falls back to the mean for
<= 2 events or constant data).  The count is `n_units = 1 / (influx * tau_dark)`
(`pick_properties`, `postprocess.py:3191`), influx default 0.03 /frame.  The GUI
calibrates influx from picks of known site number: `influx = 1 / pooled_dark /
units_per_pick` (`gui/render.py:5210`).  `pick_properties` also writes
`qpaint_idx_cdf = 1/tau_dark`, per-pick means/stds of every column, and the pick area.
SMAPpy instead: `plugins/blinking.py` already has the blinks, dark frames and
bursts per region (same dark-time convention) but explicitly says "Not here: a
counting fit ... and qPAINT"; the porting plan (B4) defers qPAINT "after looking
at Picasso".  This is that look: it is small.
How it fits: a `per_region` value in `blinking.py` (`tau_dark_frames`,
`tau_bright_frames`, `qpaint_index`, `n_sites`) plus a `SiteAnalysisPlugin`
(`Analysis/Regions/Summarize`) that calibrates the influx from ROIs tagged with a
known count (an ROI keyword) and writes `n_sites` for the rest.  Improvements
over Picasso worth making: fit the dark-time distribution by maximum likelihood
(the mean of an exponential, censoring the last open interval) instead of a
least-squares fit to an unnormalised ECDF with a free offset `c`; report the
Poisson/Gamma error of 1/tau (n events -> relative error 1/sqrt(n)); and allow
the dark times to be pooled across sites before fitting (Jungmann 2016 uses the
mean dark time).  Test data: `smappy.simulate.kinetics` already simulates on/off
kinetics with known rates; the test is n sites in, n sites out.

#### G5M: molecular mapping by a constrained Gaussian mixture -- Picasso  (M, MIT)
What it does (`g5m.py`).  Runs inside each prior cluster (`group`, typically from
DBSCAN or the SMLM clusterer) a Gaussian mixture whose number of components is
chosen by BIC: try K = 1, 2, ... until `MAX_ROUNDS_WITHOUT_BEST_BIC = 3` rounds
without improvement, K <= min(100, n/min_locs) (`_find_optimal_G5M_2D`,
`:1059`; constants `:37-55`).  The modifications against plain GMM (sklearn-derived EM,
k-means++ init, numba):
* **sigma bounds from the localization precision**: in each M-step every
  component's variance is clipped to `[0.8^2, 1.5^2] * (responsibility-weighted
  mean lp)^2` (`_m_step_2D`, `:1014`), i.e. a component must look like one
  molecule's localization cloud -- this is what stops overfitting;
* **Sparrow check**: a model is rejected if, along the line between any two
  component means, the mixture pdf has no local minimum (40 samples,
  `_check_G5M_resolution_2D`, `:873`);
* **min_locs** (default 10) per component, components below are invalid;
* 3D: diagonal covariance with the astigmatic x/y width ratio taken from the z
  calibration at the component's z (`_m_step_3D`, `:1510`), or a rotated xy block;
* per component: SEM `sigma/sqrt(n)` or a 20-sample parametric bootstrap
  (`_approximate_sem`, `:2377`; `_bootstrap_sem`, `:2413`), `n_events` (binding
  events, split at gaps > 3 frames, assigned by `predict`), weighted mean and
  std of frame, and a **p-value** that the component's mean log-likelihood
  matches its expectation `log(w/((2pi)^{D/2} sqrt(det C))) - D/2` with std
  `sqrt(D/2/(n w))` (`:2540-2557`);
* default post-filter (`:3536`): `std_frame > 0.1 * n_frames`, `p_val > 0.015`,
  `n_events > 3` -- removes sticky events and bad fits.
* `clusterer.test_subclustering` (`clusterer.py:1196`): compares `n_events` of
  molecules with a neighbour < 25 nm against isolated ones (> 80 nm) to detect
  over-splitting.
SMAPpy instead: nothing comparable.  `clusters.py` gives clusters,
`cluster_shape.py`/`circle_fit.py` measure them, `npc.py` counts corners by a
model.  G5M is the step "cluster -> molecules" for DNA-PAINT/RESI at 2-5 nm
precision, and the published benchmark (from the paper: 27-fold higher recovery
of Rayleigh-limit dimers) makes it the strongest single algorithm in Picasso.
How it fits: a `per_region` plugin under `Analysis/Clusters` (or a *method* of a
new "Molecules" plugin) that hands back a new table of molecule centres through
`Result.files` (like `average_rois.py`), with `n_events`, `p_val`, `std_frame`,
`sigma_rel` as columns, and writes `molecule_id` onto the localizations as a
recipe column; it runs in a block over `cluster_id`.  Port the EM core as
module-level numpy functions (n per cluster is small; numba not needed, or
C++ in `csrc/` if speed matters).  Needs `xy_err_nm` (SMAPpy has it).  Test data:
`simulate` DNA-PAINT dimers at known separation, with `ground_truth.py`'s
matching to score recovery.

#### SMLM clusterer with frame analysis (RESI) -- Picasso  (S, MIT)
What it does (`clusterer._cluster`, `clusterer.py:198`; Schichthaerle 2021,
Reinhardt 2023).  Count neighbours within `radius` for each localization (point
itself included); a **local maximum** is a point whose count equals the maximum
over its neighbourhood and exceeds `min_locs`; each local maximum in order labels
its unlabelled neighbours (two maxima within the radius merge); clusters below
`min_locs` are dropped.  Optional anisotropic 3D radius (`cluster_3D`, `:340`).
Then **frame analysis** (`_frame_analysis`, `:36`): a cluster is rejected if its
mean frame is outside 20-80 % of the acquisition, or if any of 20 equal time bins
holds > 80 % of its localizations -- the standard DNA-PAINT filter for sticky
imager and non-specific binding.  `find_cluster_centers` (`:979`) writes centre
(mean), `std/sqrt(n)` as the centre's precision, `n_events`, convex hull area,
and for 3D an lp-weighted z mean.  `postprocess.resi` (`:5173`) just runs this per
imaging round and concatenates the centres with `resi_channel_id` -- RESI is
"cluster each sparse round, keep the centres".
SMAPpy instead: `clusters.py` has DBSCAN/Voronoi/HDBSCAN with `cluster_id`, and
`cluster_shape.py` measures each.  Missing: (1) the local-maximum clusterer,
which is designed for small dense DNA-PAINT sites (one cluster per binding site,
not per density blob, and no chaining as in DBSCAN); (2) **frame analysis**,
which is useful for any clusterer.  Fit: a fourth *algorithm* `"smlm"` in
`clusters.py` on `csrc/neighbours.cpp`'s cell search (counts per point are
`Index.pairs`-style); frame analysis as a tick shared by all algorithms
(`kinetic filter`) setting rejected clusters to 0; the centres as a `Result.files`
table ("cluster centres") -- which, with the channel kept, is RESI.  Test data:
simulated sites plus a few simulated sticky events (all locs in one burst).

#### SPINNA: oligomeric states from nearest-neighbour distances -- Picasso  (M-L, MIT)
What it does (`spinna.py`; Masullo, Kowalewski et al. 2025).  The user defines
structures (monomer, dimer at d nm, trimer, ..., possibly of several targets);
`StructureSimulator` (`:1537`) places N structures uniformly (CSR) inside a
**mask** of the observed density (`MaskGenerator`, `:927`, from
`masking.py`'s thresholds: Otsu, Li, Yen, triangle, ...) or a rectangle,
rotates them randomly in 2D or 3D, adds Gaussian label uncertainty and drops each
target with probability 1 - LE (labelling efficiency).  The experimental and
simulated **k-th nearest-neighbour distances** for each target pair (k = 1..n,
`get_NN_dist`, `:695`, KDTree) are compared by the mean two-sample KS statistic
(`NND_score`, `:845`).  The proportions of structures are searched over a grid
(granularity) by brute force, by a two-pass coarse-to-fine search with
farthest-point sampling (`fit_coarse_to_fine`, `:3377`), or by Bayesian
optimisation with a GP surrogate (Matern) and expected improvement
(`fit_bayesian`, `:3507`); bootstrap gives errors (`_run_bootstrap`, `:3809`);
`compare_models` (`:4202`) also fits label uncertainty and chooses among models;
`fit_le` (`:4562`) fits LE from a reference-tag pair (Hellmeier 2024).
SMAPpy instead: `spatial_statistics.py` has g(r)/K/L with envelopes and
cross-correlation, `neighbourhood.py` local densities; nothing infers
stoichiometry.  SPINNA is the analysis that pairs with G5M/RESI output (molecule
positions at single-protein resolution) and with SMAPpy's existing simulator.
How it fits: an `Analysis/Spatial/SPINNA` plugin on a table of molecule
positions (layers = targets), structures as a small YAML (like segment presets),
the window from SMAPpy's own ROI/occupied-area window (already used in
`spatial_statistics.py`) rather than a separate mask generator; brute force and
coarse-to-fine first, the GP search later.  Algorithm in `smappy/spinna.py` with
module-level functions.  The KS-on-NND score is cheap; the cost is simulation
repeats.  Test data: simulate a known monomer/dimer mixture, recover the
proportions (the module's own tests do exactly this).

#### AIM drift correction -- Picasso (Ma et al. 2024)  (S, MIT)
What it does (`aim.py:772`).  Cut the acquisition into segments of
`segmentation` frames (default 100).  Encode each localization as one integer
key on a grid of `intersect_d` (default 20 nm): `round(x/d) + round(y/d) * W`.
For each segment, count for every integer shift in a `(2 roi_r/d + 1)^2` box
(default roi_r 60 nm -> 7x7) how many target keys coincide with reference keys
(`sum min`-style count via binary search, numba, `_count_intersections`,
`:28`).  The sub-bin peak is the **phase of the first Fourier coefficient** of
that small box along each axis (`_get_fft_peak`, `:310`).  "Adaptive": the
search of segment s is centred on the drift accumulated so far
(`rel_drift_x += px`, `aim.py:597`), so total drift is unbounded as long as the change
per segment is < roi_r.  Round 1 uses the first segment as reference; round 2
uses the whole once-corrected dataset as reference, **leaving the segment being
aligned out of the reference counts** (`exclude_self`, `:515-525`, `:566-590`; fixed in
0.11.2 -- without it round 2 is biased to zero shift).  Per-frame drift is a
cubic spline through segment midpoints with one linearly extrapolated knot
added at each end (`_interpolate_drift`, `:383`).  z follows after xy, as a 1D
shift search on 3D keys (`intersection_max_z`, `:615`).  Drift is mean-centred.
SMAPpy instead: COMET's spline fit (`drift.py`, the accurate default, which can
take long -- it has a cost preflight) and RCC (`rcc.py`, the coarse pass that
costs seconds).  AIM sits between them: linear in N, no images, 100-frame
segments so it tracks high-frequency drift that 20-window RCC cannot, CPU only.
Worth porting as a third estimator in the `KeepsDrift` shape
(`plugins/drift_aim.py`, algorithm in `smappy/aim.py`), and to benchmark as
COMET's prepass in place of RCC (`drift._rcc_prepass`).  Port caveats found in the
code: the keys are `np.int32` (`:243`, `:296`, `:512`, `:662`); in 3D `W*H*z` overflows int32 for
large fields (2048^2 px at 130 nm with d = 20 nm overflows beyond about +-240 nm
in z) -- use int64; a negative x after drift aliases onto the previous y row
(edge-only effect; offset the coordinates).  The phase estimator is a centroid
on a small box and is biased if the peak is near the box edge; SMAPpy's
`rcc._quadratic_peak` could be compared.  Test data: SMAPpy's drift tests and
the simulated drift of `drift_comet`'s page.

#### Pick similar: segment by example -- Picasso Render  (S, MIT)
What it does (`postprocess.pick_similar`, `:510`; circles `:638`).  From the
user's picks take mean and std of the number of localizations and of the RMSD
about the centre of mass; scan a hexagonal grid of candidate picks of the same
size over the field (spacing `sqrt(3) d/2` in x, d in y, offset odd columns),
move each candidate to the centre of mass of its localizations, accept it if both
numbers lie within `mean +- 2 std` (`std_range`), and discard candidates
overlapping accepted ones.  Rectangles are additionally rotated onto the
principal axis of the enclosed localizations and get the median length
(`_pick_similar_rectangular`, `:1602`).
SMAPpy instead: `segment.py` (circle template, beads, id column, positions from a file).  Pick
similar is the method with no parameters to choose: draw five ROIs on good
structures, get all the others.  Fit: a fourth *method* of *Segment regions*
("like the selected ROIs"), reading the statistics from the ROI manager's selected
ROIs, writing `origin["segment"]`.  Could use more features than n and RMSD
(any measured column of those ROIs, e.g. a circle-fit radius), which Picasso
cannot.

#### Binding-event cores: drop partial frames -- Picasso (Steen 2024)  (S, MIT)
`postprocess.select_binding_event_cores` (`:3385`): link localizations into
events (`r_max` 0.1 px, no dark frames), discard events with < 3 localizations,
keep only those not in the first or last frame of their event -- the frames where
the imager binds or unbinds during the exposure, with a biased photon count and
worse precision.  SMAPpy's `group.py` merges blinks but has no such option.  Fit:
a tick in `remove_locs.py` or in merging ("drop first and last frame of each
blink"), or a flag column for filtering.  Useful for photon/precision statistics
and for `precision.py`.

#### sCMOS per-pixel calibration -- Picasso  (M, MIT)
`scmos.py` (`calibrate_scmos`, `:283`): offset = temporal mean and readout
variance = temporal variance of a dark movie (>= 100 frames required, warned
below a larger threshold), streaming moments in chunks; optional gain map from a
bright series by a photon-transfer fit through the origin; `validate_calibration`
(`:470`) checks a stored map against a fresh dark movie with a chi-square test;
maps feed the Huang 2013 likelihood.  SMAPpy (NOTES.md "Scope") deliberately uses
one variance for the chip; the porting plan lists "sCMOS maps (#22)" for later.
When that is taken up, Picasso's calibration and validation module is the
template (the fit side is SMAPpy's `pipeline.with_readout` made per-pixel).

#### Temporal median background for detection -- Picasso  (S, MIT)
`localize.TemporalMedianMovie` (`:638`): detection on `max(frame -
median(window of 51 frames), 0)`, the median evaluated only every `stride`
frames (default = window) with `np.partition` in row stripes; fitting always on
raw frames.  For DNA-PAINT with a high, structured imager background (TIRF/HILO
inhomogeneity, autofluorescence) this cleans detection.  SMAPpy's `detect.py`
has DoG and Gaussian filters only.  Fit: a third `ImageFilter` in `detect.py`
that needs frame blocks (the pipeline already reads blocks).

### Better than SMAPpy's current implementation

* **Fiducial drift: weighting beads** -- Picasso `_undrift_from_picked_coordinate`
  (`postprocess.py:4490`): each bead's mean-subtracted trace, an average over beads
  weighted by 1/MSD of each bead to the unweighted mean, NaN frames linearly
  interpolated.  SMAPpy's `fiducials.py` is better in its model (offsets fitted
  jointly with a spline, late-starting beads correct) but says "every bead that
  passes is used, all with equal weight; rejecting a bead ... comes later".
  Picasso's 1/MSD weight is the cheap version of that later step: weight each
  bead's rows by 1/(its residual variance) and refit once or twice (IRLS).  (S)
* Compared and **not better**: Picasso RCC (`postprocess.undrift`, `:4271`;
  `imageprocess.rcc`, `:160`: Gaussian-blurred renders, integer-pixel-plus-fit
  peak in a 5-px box, max shift 32 px, plain least squares, cubic spline,
  xy only) -- SMAPpy's `rcc.py` (IRLS with Cauchy weights, PCHIP, tiled z) is
  better.  NeNA (`postprocess.nena`, three starting points of a least-squares fit
  to a histogram) -- SMAPpy's `precision.py` (likelihood fit of displacement
  pairs) is better.  DBSCAN/HDBSCAN -- same algorithms; Picasso's anisotropic
  3D DBSCAN (Lörzing 2024) is just scaled z, which SMAPpy can do by scaling.
  Average (`average.py`: iterative alignment to the sum image over rotation angles,
  integer-pixel FFT xcorr peak, `align_group_core`, `:49`) is the classic
  reference-free averaging; SMAPpy's `average_rois.py` aligns by measured centre
  and angle instead -- different, see UX below.

### User experience and automation

* **Test clustering on one pick** (Render `TestClustererDialog`,
  `gui/render.py:3483`): pick one region, choose DBSCAN/HDBSCAN/SMLM/G5M, see the
  result rendered live with unclustered localizations as a second channel, then
  "apply to whole dataset".  In SMAPpy this is a Preview of *Find clusters* with
  `follow_roi = True` on the drawn ROI; `clusters.py` does not set `follow_roi` today.
* **Show trace / Select picks (trace)**: for each pick, x, y and photons against
  frame, accept/reject with keys, one pick after another.  The time trace is the
  fastest way to judge a DNA-PAINT site (regular blinks vs one sticky burst).  In
  SMAPpy: a figure in the ROI manager's site view (`gui/roi_site.py`) and a key
  for *use*.
* **qPAINT in the pick info** (Render *Show info > Calculate info below*): per-pick
  number of localizations, events, RMSD, bright/dark time, units, means and
  histograms; "Calibrate influx" button.  The natural home in SMAPpy is the
  region table.
* **Picasso Server** (`server/`, Streamlit + SQLite; Strauss 2022): a folder
  *watcher* localizes every new movie automatically, every localized file gets a
  database row (`localize.get_file_summary`, `:8578`: means/stds of every
  column, NeNA, drift, mean bright time, locs per frame, settings), and the
  *history*/*compare* pages plot these over all experiments -- instrument QC over
  months.  SMAPpy has `smappy-batch` and the batch window; a per-file summary row
  appended to a project-wide table (CSV/SQLite) by each batch run, plus a
  plot-over-time page, would give the same.  A watcher is a small addition to
  `batch.py` (the live fit already exists).
* **Plugin store** (`plugins.py`): plugins can add GUI menus, a Python API
  (`PICASSO_API`) and CLI subcommands (`register_cli`); an online registry
  (`index.json` in a GitHub repo) with **sha256-pinned** downloads, nothing loaded
  until explicitly enabled, a diff shown before an update, `picasso plugins
  list/enable/install/update` headless.  SMAPpy's drop-in folder plus AST
  discovery is already safer than imports; the registry with pinned hashes and an
  explicit enable is the part to copy if SMAPpy ever distributes third-party
  plugins.
* **Every result carries its provenance in the file**: each step appends a dict
  to the metadata list (`"Generated by": "Picasso v0.11.3 AIM"`, parameters),
  embedded in the HDF5 since 0.11 (`/metadata`, JSON) and optionally the picks
  used.  SMAPpy's recipe/history is richer; check that the Picasso writer
  (`io/picasso.py`) appends SMAPpy's steps the same way so a file that goes to
  Picasso and back keeps its history.
* **`localize_frames`** (`localize.py:7866`): a GUI-free fit of an in-memory
  frame block with absolute frame indices, for live acquisition.  SMAPpy has its
  live fit; nothing to take.
* **Error log** (`diagnostics.py`): every uncaught exception (main, worker,
  unraisable) to `~/.picasso/logs/picasso.log` with rotation and a message box
  with the traceback.  Cheap to copy if SMAPpy does not log to a file.
* Small Render conveniences: right-click a channel's tick to show only that
  channel; log-scale contrast sliders; colour bar saved beside every exported
  image coloured by a property; brush and box ROI shapes.

### Licence-restricted (re-implement only)

* None: Picasso and all its satellites read here are MIT (Gpufit/Gpuspline
  ports MIT too).  The original AIM reference implementation (MATLAB, Ma et al.)
  was not read; Picasso's port is MIT and is what to port from.

### Not worth it / already in SMAPpy

* Picasso HDF5 read/write -- done (`io/picasso.py`, checked against 0.11.3); Picasso also reads/writes SMAP `_sml.mat` (`picasso smap2hdf/hdf2smap`).
* Localize fitting (LQ/MLE Gaussians, spline, multichannel global "globLoc", Numba-CUDA Gpufit port) -- SMAPpy has MLE Gaussian, cspline, dual-channel global fit, uiPSF, C++ LM; nothing algorithmically new.  Picasso's identification (net gradient in a box) is no better than DoG + local maxima.
* Lateral (affine/projective/polynomial) bead transforms -- SMAPpy has `transformation.py` and *Register layers*.
* RCC, NeNA, FRC, pair correlation, local density, k-NN distances (`nn_analysis`) -- equal or better in SMAPpy (see above); k-NN histograms come with SPINNA if ported.
* Average (2D) and Average3 -- reference-free iterative xcorr averaging; useful only for structures without a measurable orientation; low priority behind LocMoFit in SMAPpy's plan.
* Nanotron (Auer 2020) -- an sklearn MLP on rendered pick images; superseded by modern classifiers, little demand.
* Design / Simulate (DNA origami design, DNA-PAINT movie simulation) -- SMAPpy's simulator covers the analysis needs.
* `cluster_combine(_dist)`, `calculate_fret`, k-means in a pick, masking thresholds -- minor; masking only as part of SPINNA.

### References

* Schnitzbauer J, et al. Super-resolution microscopy with DNA-PAINT. Nat Protoc 2017 -- Picasso. [10.1038/nprot.2017.024](https://doi.org/10.1038/nprot.2017.024) (DOI resolves in the PubMed ID converter; no PMID returned)
* Ma H, Chen M, Nguyen P, Liu Y. Toward drift-free high-throughput nanoscopy through adaptive intersection maximization. Sci Adv 2024;10:eadm7765 -- AIM. [10.1126/sciadv.adm7765](https://doi.org/10.1126/sciadv.adm7765) (Picasso's readme prints it as "adm776", link correct)
* Kowalewski R, Reinhardt SCM, et al. Molecular mapping in DNA-PAINT via modified Gaussian Mixture Modeling. Nat Commun 2026;17 -- G5M. [10.1038/s41467-026-70198-5](https://doi.org/10.1038/s41467-026-70198-5)
* Masullo LA, Kowalewski R, et al. Spatial and stoichiometric in situ analysis of biomolecular oligomerization at single-protein resolution. Nat Commun 2025;16:4202 -- SPINNA. [10.1038/s41467-025-59500-z](https://doi.org/10.1038/s41467-025-59500-z)
* Hellmeier J, et al. Quantification of absolute labeling efficiency at the single-protein level. Nat Methods 2024;21:1702 -- SPINNA's LE fit. [10.1038/s41592-024-02242-5](https://doi.org/10.1038/s41592-024-02242-5)
* Reinhardt SCM, Masullo LA, Baudrexel I, Steen PR, et al. Ångström-resolution fluorescence microscopy. Nature 2023 -- RESI, SMLM clusterer. [10.1038/s41586-023-05925-9](https://doi.org/10.1038/s41586-023-05925-9)
* Schichthaerle T, et al. Nat Commun 2021 -- the local-maximum clusterer. [10.1038/s41467-021-22606-1](https://doi.org/10.1038/s41467-021-22606-1)
* Jungmann R, et al. Quantitative super-resolution imaging with qPAINT. Nat Methods 2016 -- qPAINT. [10.1038/nmeth.3804](https://doi.org/10.1038/nmeth.3804)
* Steen PR, et al. Nat Methods 2024 (Ext. Data Fig. 1f) -- binding-event cores. [10.1038/s41592-024-02374-8](https://doi.org/10.1038/s41592-024-02374-8)
* Huang F, et al. Video-rate nanoscopy using sCMOS camera-specific single-molecule localization algorithms. Nat Methods 2013 -- sCMOS maps. [10.1038/nmeth.2488](https://doi.org/10.1038/nmeth.2488)
* Martens KJA, Turkowyd B, Endesfelder U. Front Bioinform 2022 -- temporal median filter. [10.3389/fbinf.2021.817254](https://doi.org/10.3389/fbinf.2021.817254)
* Strauss MT. Picasso-server. Commun Biol 2022;5:930. [10.1038/s42003-022-03909-5](https://doi.org/10.1038/s42003-022-03909-5)
* Wang Y, et al. Opt Express 2014 -- RCC (for comparison). [10.1364/OE.22.015982](https://doi.org/10.1364/OE.22.015982)
* Przybylski A, et al. Gpufit. Sci Rep 2017. [10.1038/s41598-017-15313-9](https://doi.org/10.1038/s41598-017-15313-9)
* Lörzing P, Schake P, Schlierf M. Anisotropic DBSCAN for 3D SMLM data clustering. J Phys Chem B 2024. [10.1021/acs.jpcb.4c02030](https://doi.org/10.1021/acs.jpcb.4c02030)
* Auer A, Strauss MT, et al. Nanotron. Bioinformatics 2020. [10.1093/bioinformatics/btaa154](https://doi.org/10.1093/bioinformatics/btaa154)

DOIs checked against PubMed (ID converter and article metadata), 2026-10-17.
