## Drift, registration, resolution metrics, particle averaging / model fitting, and fitting alternatives

Surveyed: DME (qnano/drift-estimation and photonpy), Picasso's AIM, NanoJ-SQUIRREL/-Core, ImDecorr (decorrelation), the TU Delft particle-fusion family (smlm_datafusion2d/3d, smlm_classification2d), PERPL, SMAP's LocMoFit (the baseline for H), ZOLA-3D, Gpufit/Cpufit, fit3Dcspline's depth correction, uiPSF's model types, storm-analysis (3D-DAOSTORM multi-fit, multiplane, micrometry, sa_utilities) and Picasso's sCMOS maps. SMAPpy's drift work is ahead of most of these on everything except one point: it uses one isotropic sigma and no per-localization precision. SMAPpy's FRC, NeNA, RCC, fiducials and channel registration are at least as good as anything here. The real gaps are all on the structure side: **model fitting (LocMoFit, H)**, **template-free particle fusion and classification**, **depth-dependent z correction**, and a group of cheap fitter additions: a **GLRT detection p-value**, **per-pixel sCMOS variance** and **multi-emitter fitting**.

### The software

| tool | what | language | licence | port route | repo (commit read) |
|---|---|---|---|---|---|
| DME | drift by minimising an entropy bound, per-spot CRLB, Catmull-Rom spline | C++/CUDA + Python | MIT (`setup.py:32`; no LICENSE file) | port with attribution | qnano/drift-estimation (2ee39f7) |
| photonpy | Cnossen's SMLM library: DME, GLRT detection, RCC, FRC, SIMFLUX | C++/CUDA + Python | MIT (`LICENSE.txt`) | port with attribution | qnano/photonpy (9c03a1c) |
| Picasso AIM | drift by intersection counting on a grid | Python/numba | MIT | port with attribution | jungmannlab/picasso (a1e639a) |
| Picasso scmos | per-pixel offset/variance/gain maps (Huang 2013) | Python | MIT | port with attribution | same |
| NanoJ-SQUIRREL | error map vs a widefield reference (RSE/RSP), FRC map | Java (ImageJ) | GPL-3 | re-implement from paper | HenriquesLab/NanoJ-SQUIRREL (a1511a3) |
| NanoJ-Core | cross-correlation drift and registration of images | Java | GPL-3 | re-implement | HenriquesLab/NanoJ-Core (c454d5a) |
| ImDecorr | decorrelation resolution from a single image | MATLAB, ImageJ | GPL-3 | re-implement from paper | Ades91/ImDecorr (db536a3) |
| smlm_datafusion2d | template-free 2D particle fusion (all-to-all + Lie averaging) | MATLAB + C++/CUDA | Apache-2.0 | port with attribution | imphys/smlm_datafusion2d (58cfc76) |
| smlm_datafusion3d | the same in 3D, anisotropic sigma, symmetry detection | MATLAB + CUDA | Apache-2.0 | port with attribution | imphys/smlm_datafusion3d (2e08161) |
| smlm_classification2d | classifying heterogeneous particles (Bhattacharya + MDS) | MATLAB | Apache-2.0 | port with attribution | imphys/smlm_classification2d (df49d2e) |
| PERPL | model fits to relative-position distributions | Python | Apache-2.0 | port with attribution | AlistairCurd/PERPL-Python3 (0ae4922) |
| LocMoFit | ML fit of geometric models to localizations in ROIs | MATLAB (in SMAP) | GPL-3 (SMAP); Ries lab's own code | Jonas's call (own code) | jries/SMAP (d066594) |
| ZOLA-3D | Zernike PSF with index mismatch, multi-emitter, 3D drift | Java + CUDA | GPL-3 (`COPYING`) | re-implement from paper | imodpasteur/ZOLA-3D (d1f76d9) |
| Gpufit / Cpufit | GPU LM fitter, Gaussian and spline models, MLE | CUDA/C++ | MIT | port with attribution | gpufit/Gpufit (a0bd66c) |
| fit3Dcspline | spline calibration and **depth-dependent z correction** | MATLAB | GPL-3 (Ries lab) | own code | jries/fit3Dcspline (c542f01) |
| uiPSF | PSF inverse modelling, incl. field-dependent (FD) | Python/TF | GPL-3 (Ries lab) | already wrapped | ries-lab/uiPSF (7677072) |
| storm-analysis | 3D-DAOSTORM multi-fit, multiplane, micrometry, sa_utilities | Python + C | mixed: no root licence (`pyproject.toml:30` "License :: Mixed"); `spliner/` and `L1H/` academic non-commercial; `rolling_ball_bgr/`, `dbscan/`, `voronoi/`, `wavelet_bgr/` MIT | re-implement from paper (except the MIT parts) | ZhuangLab/storm-analysis (dd1a00f) |

### Worth porting  (ranked, most valuable first)

#### 1. LocMoFit: ML model fitting of localizations per ROI -- SMAP  (L; own code, GPL in SMAP)
*What it does (from the code).*
* **Likelihood.** The objective is `sum_i w_layer * log p(x_i | theta)` (`LocMoFit/@LocMoFit/LocMoFit.m:1652-1700`, `fit.m:161-166`). The per-localization density `p` (`@LocMoFit/intensityCal.m`) mixes three things:
  * each model's density, evaluated at the localization;
  * a weight per model, normalised within a layer;
  * a uniform background over the ROI, a cylinder of the ROI size: `offsetUnit = 1/(pi r^2)`, or `1/(pi r^2 h)` in 3D.

  The mixture is `(1 - bg_w) * model + bg_w * offsetUnit`.
* **Discrete ("point") models.** The model is a set of points. Each localization's density is a sum of Gaussians whose sigma is the localization's own precision times `sigFactor`. Distances beyond 3.5 sigma are cut off (`helperFcn/gaussDist.m`, `distCutoff` 3.5).
* **Continuous and image models.** These are sampled into an image, which is blurred and read at each localization (`modelType` 'continuous' / 'image').
* **Optimisation.** Bounds are relative to the start (`fit.m:130-140`) and clipped to each parameter's meaningful range. The solver is `fmincon`, `fminsearchbnd` or `particleswarm`.
* **Sigma cascade.** The fit is repeated with inflated sigma factors and bounds, each step started from the previous best (`fit.m:118-125`). This is the same coarse-to-fine idea as COMET's sigma schedule.
* **Outputs.** AIC and AICc per fit (`fit.m:255-257`). Optionally a z-score of the fit's log-likelihood against the expected log-likelihood from 1000 simulated draws (`fit.m:266-274`, `getLLExpDist`): a goodness-of-fit for each ROI.
* **Model library.** `LocMoFit/models/*.m`, about 40 current models: `ring2D/3D`, `dualRing3D_discrete` (NPC), `dualEllipse3D_*`, `sphericalCap3D_surfaceArea` (CME), `cylinder3D`, `csplineTube3D_*`, `arc2D`, `gaussianCluster2D`, `locsBG2D/3D`, `simplex2D`, `triangle`, `vertexOfSquare`.
* **Fusion.** LocMoFit also does template-free particle fusion (`LocMoFit/script/template_free_particle_fusion.m`):
  * pre-registration by a dual-ring fit;
  * all-to-all LL registration on 50 sites;
  * an initial template from the best-ranked site;
  * iterative registration to the growing average: up to 100 iterations, stop after 10 stalls, minimum improvement 0.1 LL.

*SMAPpy has* the pieces around it, not the fitter:
* `plugins/circle_fit.py` (soft-L1 circle);
* `plugins/line_profile.py` (unbinned ML profiles with AIC model choice: the right statistical shape, in 1D);
* `npc.py` (a labelling-efficiency count model, not geometry);
* `plugins/average_rois.py` (superimposes ROIs by a centre and an angle, with no registration).

*Why.* It is the largest open item (H, E5's NPC geometry, and the "LocMoFit's model" left out of Average ROIs). `per_region` already makes every fitted parameter a column of the region table, so Filter/Sort/Histograms come free (porting plan, H).

*How it would fit.*
* A `per_region` Measure plugin, `Analysis/Regions/Model fit`, with the fitter in `smappy/modelfit.py` as module-level functions over arrays.
* The fitter is `p(x | theta)` as a mixture of point-model Gaussians with per-localization sigma, plus a uniform background; minimise `-sum log p` with `scipy.optimize.minimize` (L-BFGS-B with bounds); run the sigma cascade.
* Port first the models that cover the main uses: ring 2D/3D, dual ring (NPC: radius, ring distance, twist, tilt), spherical cap (CME), line/arc, Gaussian cluster. A model is a small Python function returning points (or a density) from parameters.
* The density evaluation is a neighbour sum, the shape `csrc/neighbours.cpp` already has.
* Line Profile's AIC/verdict machinery can be reused for the model comparison.
* Fusion (item 2) reuses the same likelihood as its registration cost.

*Test data.* `smappy.simulate` structures (NPC, rings) with known parameters. SMAP's `LocMoFit/docs/applicationNote/simulations.md` describes the published simulations.

#### 2. Template-free particle fusion and classification -- smlm_datafusion2d/3d, smlm_classification2d  (L; Apache-2.0, port with attribution)
*What it does.*
* **Pair registration.** `MATLAB/all2all/pairFitting.m`:
  1. Rigid GMM-L2 registration (`gmmreg_L2`), started from `nAngles` initial rotations (6 in `all2all.m:48`), on clouds subsampled by density (`resampleCloud2D`).
  2. Of the candidates, keep the one with the highest Bhattacharya cost (`expdist/expdist_ref.c`): `sum_ij exp(-|a_i - b_j|^2 / 2(s_i^2 + s_j^2)) / (s_i^2 + s_j^2)`. This is a precision-weighted Gaussian overlap, COMET's cost with per-point sigma.
* **Consistency.** All N(N-1)/2 relative transforms go into Lie-algebraic averaging on SE(2)/SE(3) (`avg/outlier_removal.m`, `MeanSE3Graph`). Pairs whose residual rotation exceeds 5 degrees are dropped as outliers, a second averaging follows, and graph connectivity is checked (`isConnected`).
* **Bootstrapping.** `one2all/one2all.m`: each particle is registered to the average of all the others (`delParticle`), for a few iterations.
* **3D** (`smlm_datafusion3d/new_demo.m`):
  * anisotropic sigma, xy and z;
  * a `scale_sweep` picks the GMM scale from the cost landscape on 6 pairs;
  * localizations with sigma above 1.5 times the mean are discarded;
  * 8 initial angles per degree of freedom;
  * optional symmetry prior (`USE_SYMMETRY`). The paper detects symmetry order from the average *(from the paper)*.
* **Classification** (`smlm_classification2d/demo_classification.m`):
  1. The all-to-all Bhattacharya matrix, normalised (`MakeMatrix`), becomes a dissimilarity.
  2. Metric MDS to 30 dimensions, then k-means (1000 replicates).
  3. Each class is fused separately; an eigen-image analysis (`eigenApproach`) merges classes.

*SMAPpy has* `average_rois.py`: centre and angle from the ROI, a line or a measured value. There is no alignment by the data.

*Why.* This is the standard way to get a structure from many under-labelled copies (NPC, origami, centrioles). The all-to-all-plus-averaging design is what makes it robust to mis-registration. Classification gives the heterogeneity analysis that a plain average hides.

*How it would fit.* An `Analysis/Regions/Fuse ROIs` plugin that hands back a new table through `Result.files`, like Average ROIs, plus the per-ROI transform as values.
* The Bhattacharya cost and its gradient over two point sets is a neighbour sum; it belongs in C++ beside `csrc/drift.hpp`, which already evaluates the same Gaussian-pair kernel.
* The GMM-L2 step can be replaced by direct optimisation of the Bhattacharya cost from several starting angles (simpler, one kernel).
* Lie averaging in 2D is averaging angles and translations on a graph, a small linear least-squares problem, like RCC's redundant solve (`rcc._solve`, IRLS). The 3D rotation averaging needs quaternions/SO(3) (`avg/R2q.m`, `q2R.m`).
* Cost grows as N^2 pair registrations; restrict all-to-all to a subset (LocMoFit uses 50) and register the rest one-to-all.
* Classification: MDS plus k-means from sklearn/scipy.

*Test data.* Each repo ships simulated particles (`data/simulation/tud_dol100.mat`, `smlm_datafusion3d/data/example_data.mat`, `smlm_classification2d/data/200x_simulated_TUD_flame`); `smappy.simulate` NPCs as well.

#### 3. Precision-weighted, anisotropic drift cost -- DME  (M; MIT, port with attribution)
*What it does.*
* **Cost.** DME maximises `-entropy = (1/N) sum_i log(1 / (1 + sum_j exp(-KL_ij)))` (`dme/DME/DriftEstimation.cu:228-256` constant CRLB, `:290-316` per spot). `KL_ij` is the Kullback-Leibler divergence between the two localizations' Gaussians, built from each one's **own CRLB per axis** (`:28-35`).
* **Neighbours.** Within 3 x the mean CRLB (`:150-154`), pairs from the same frame excluded (`:121-122`), capped at `maxNeighbors` = 1000 (`dme.py:19`). The list is rebuilt only when the drift has moved more than half the search range (`:107-123`).
* **Representation.** Drift is a Catmull-Rom spline, `framesPerBin` frames per knot (`:420-520`).
* **Optimiser.** Plain gradient ascent with an adaptive step: x1.2 on an accepted step, x0.5 on a rejected one, stop after 10 rejections (`:180-205`).
* **Start and precision.** Initialised by RCC with 10 bins (`dme.py:79-90`); optional coarse pass with an inflated sigma (`dme.py:121-133`). It always reports a **split-half precision**: two halves split at the median of the axis with the larger variance, both re-estimated, and the RMSD between them printed and plotted (`dme.py:100-170`).

*SMAPpy has* COMET's overlap `sum_ij exp(-|d_ij|^2 / sigma^2)`. sigma is one scalar, shared by x, y and z, and stepped down a schedule to `target_sigma_nm` = 30 nm (`drift.py:133`; the kernel at `csrc/drift.hpp:78`, `inv_sigma_sq` scalar).

*Why it would be better.*
* **(a) Per-localization weighting.** In COMET a 30 nm-precision localization counts as much as a 5 nm one. SMAPpy compensates by filtering (docs/drift.md: "Filter before estimating, and include z"); DME weights instead of cutting.
* **(b) Anisotropy.** A z precision 2-4x the lateral one gets its own sigma. With one sigma, the axial term is either too sharp for z or too blurred for xy. docs/drift.md records that the axial estimate was carried away by the out-of-focus tail and needed a hard `|z| < 300` cut. A per-axis sigma, at least `sigma_z = k * sigma_xy`, addresses exactly that.
* **(c) The log.** It saturates dense clusters, so a bright, repeatedly blinking cluster no longer dominates. This is the same over-weighting that docs/drift.md found grouping fixes.
* DME's paper reports 5x better precision than RCC *(from the paper)*. Against COMET it is unmeasured.
* The split-half RMSD is the drift error bar that docs/drift.md says "nothing currently does".

*How it would fit.*
* Not a new estimator: two options in SMAPpy's own C++ kernel (`csrc/drift.hpp`). (i) A per-axis sigma vector, cheap. (ii) A per-pair variance `s_i^2 + s_j^2` taken from `xy_err_nm` / `z_err_nm`, one extra load per pair.
* The log-of-local-density cost as a third option needs a per-localization normalisation pass. The pair list is already there.
* Measure with the split-by-molecule and out-of-sample protocol in docs/drift.md before making anything the default.
* Separately, make the split-half noise a reported number (on a tick) in the COMET/RCC plugin output: two extra grouped runs, a few seconds.

*Test data.* `drift-estimation/example_data/gattaquant 80nm RY.hdf5` (Picasso HDF5, DNA-PAINT nanoruler); SMAPpy's clathrin set and simulated drift.

#### 4. Depth-dependent z correction from beads in gel -- fit3Dcspline  (M; Ries lab's own code, GPL)
*What it does.* `calibrate3Daberrations.m` (Li et al. 2018):
1. Beads in gel are scanned in z and fitted with the coverslip calibration. Each bead's in-focus frame `f0` is found (`getf0Z_so`), and the glass frame is detected automatically.
2. Per localization, `dzcorr = (f0 - frame) * dz - z_fit`, against the objective position `frame * dz`.
3. A 2D smoothing spline `dz(objective position, fitted z)` is fitted: smoothing `[0.05 0.002]`, a second pass with a 500 nm outlier cut, then iterative bead rejection at 3x the mean error, within +-800 nm (`maxrange`).
4. `correct_3Daberrations.m`: `z_out = z_in + zcorr(objectivepos, z_in)`. A separate refractive-index factor (0.75 in `example_depth_aberration.m`) scales to true depth.

*SMAPpy has* nothing for it (J #26). There is no RI-mismatch factor except the MINFLUX/MSR loaders' *z scale* (`io/minflux.py:37`, `gui/msr_import.py:95`). uiPSF's in situ model handles depth, but only by relearning the PSF.

*Why.* Any 3D data taken more than ~1 um into a sample has z compressed and distorted by spherical aberration when it is fitted with a coverslip calibration.

*How it would fit.*
* A calibration product, `_zcorr.h5`, built from a bead-in-gel localization file.
* Applied by a recipe column, or by *Transform* as a new non-affine kind keyed by the objective position, which comes from metadata or a setting.
* A small plugin under `Localize/` or `Analysis/Transform`, plus `calibrate/depth.py`.

*Test data.* The example names `data/beads_in_gel_2.csv`, but that file is not in the repository (there is no `data/` folder). Simulate with uiPSF's vector model at depth instead, or ask the Ries lab for the original files.

#### 5. GLRT detection significance per localization -- photonpy  (S; MIT)
*What it does.* `photonpy/cpp/glrt.py:26-36`, `cpp/SMLMLib/Estimators/GLRT_PSF.cu`:
* After the fit, compute the log-likelihood of the fitted PSF model (`ll_on`) and of a flat background equal to the ROI mean (`ll_off`).
* Test statistic `Tg = 2 (ll_on - ll_off)`; false-positive probability `pfa = 2 Phi(-sqrt(Tg))`, kept as a log.

This is Smith et al. 2015's threshold-free detection.

*SMAPpy has* `logl_rel`, a per-pixel goodness of fit against the saturated model (`csrc/lm.hpp:80-108`, `locs.py:146`). It says how well a spot fits, not whether there is a spot at all. Detection is DoG with a dynamic cutoff (`detect.py`).

*Why.*
* A p-value gives a filter with a meaning ("keep pfa < 1e-6") that transfers between datasets and cameras, unlike photon or `logl_rel` cuts.
* It is what lets a low detection threshold be used without flooding the table with noise.

*How it would fit.*
* A few lines in `crlb_and_logl`: also accumulate `d log(b0) - b0` with `b0` the ROI mean plus the readout variance (the variance is already added; `pipeline.with_readout`).
* Output a `log_pfa` column and register it in `columns.py`.
* The two-channel global fit sums both channels.

*Test data.* `smappy.simulate` camera frames with known spots; check the pfa calibration on pure-background ROIs.

#### 6. Per-pixel sCMOS variance and gain -- Picasso `scmos.py`, storm-analysis `multi_fit.c`  (M; Picasso MIT, storm-analysis re-implement)
*What it does.*
* **Picasso** (`picasso/scmos.py`) builds per-pixel offset and variance from dark frames, streamed in one pass (`_streaming_moments`). The gain comes from a photon-transfer fit through the origin over bright series (`_gain_map`, Huang 2013 Eqs. 2.3-2.5). It refuses calibrations with too few frames: the relative SE of a variance is sqrt(2/(M-1)). It flags hot pixels (variance above a multiple of the median) and validates the calibration against a movie.
* **storm-analysis** adds `variance/gain^2` per pixel to data and model (`sa_library/multi_fit.c:24-55`, `:1386-1390`), with a per-pixel RQE as well. This is the per-pixel version of what SMAPpy does with one number.

*SMAPpy has* one variance for the chip (NOTES "Read noise"; J #22 "sCMOS maps").

*Why.* NOTES.md measures 0.7-1.4 e- across pixels. Hot pixels of tens of e- pull localizations onto themselves, and a scalar cannot represent that.

*How it would fit.*
* A `camera_db` entry pointing to a map file.
* `pipeline.with_readout` cuts the variance ROI beside each data ROI; the C++ models take a `(n, sz, sz)` variance stack instead of a scalar. Detection weights by variance as well (SMAP's sCMOS filter).
* The calibration tool is a small `Localize/sCMOS calibration` plugin.

*Test data.* `smappy.simulate` with a synthetic variance map containing hot pixels.

#### 7. SQUIRREL error map and RSE/RSP against a widefield image layer -- NanoJ-SQUIRREL  (M; GPL-3, re-implement from paper)
*What it does.* `ErrorMap_.java:225-480`:
1. Register the SR image to the reference by cross-correlation and crop the borders.
2. Estimate a linear intensity map (alpha, beta) and the RSF sigma: a particle-swarm meta-optimisation (`Kernel_SquirrelSwarmOptimizer`), then joint optimisation with bounds `alpha in [0.001, 10 alpha0]`, `sigma in [0.5, 10]` reference pixels.
3. Blur the SR image by sigma x magnification, scale it, and report:
   * RSE, the RMS error;
   * RSP, the Pearson correlation;
   * the pixel-wise error map, which shows artefacts such as missing or extra structure.

`FRCMap_.java` adds block-wise FRC maps.

*SMAPpy has* image layers with placements, `layer_registration.py` (cross-correlation shift on a common grid, sub-pixel peak) and FRC/FPC (`frc.py`). There is no reconstruction-vs-reference comparison.

*Why.* It is the one quality check that catches artefacts FRC cannot: structure missing from the SR image but present in widefield. The registration and rendering it needs are already in SMAPpy.

*How it would fit.* A `Analysis/Measure/Error map` plugin:
* the reference is an image layer;
* the SR image is the localization layer rendered on the same grid (`layer_registration.common_grid`, `image_on`);
* alpha, beta and sigma by `scipy.optimize.least_squares`, linear in alpha and beta for a given sigma, so a 1D search over sigma;
* the error map goes back as a new image layer via `Result.image_layers`.

*Test data.* `NanoJ-SQUIRREL/test_data` (Fig. 3 crop: widefield plus three reconstructions); check its licence before reusing it in tests.

#### 8. A local FRC map from the tiles SMAPpy already computes -- NanoJ-SQUIRREL `FRCMap_`  (S; re-implement)
`frc_resolution` already transforms the field in square tiles and sums the ring statistics over them (`frc.py:467-560`, `_tile_buckets`, `_over_tiles`). Keeping each tile's ring sums and taking each tile's crossing gives a resolution map at no extra transform cost: the shape of NanoJ's FRC map (and of rFRC, Zhao et al. 2023 *(unverified)*). It returns an image layer, or values per ROI. Small tiles have noisy curves, so require a minimum count and mark tiles with no crossing.

#### 9. Multi-emitter fitting -- 3D-DAOSTORM (storm-analysis), ZOLA-3D  (L; re-implement)
*What it does.* `sa_library/fitting.py:984-1019`, `:819-908`:
* **Whole image as the model.** The model is the whole image: the sum of all current peaks plus background. Each peak is LM-fitted against the image minus the others (`f_data`/`bg_data` accumulation in `multi_fit.c:24-55`). Overlapping peaks share the per-pixel background as the average of their estimates.
* **Iteration.** Peaks are found again in the residual over several iterations, with the threshold lowered by 1 per iteration down to the minimum (`findPeaks`, `:320-348`).
* **Pruning.** After each round, the dimmer of two peaks within 1 sigma is removed (`markDimmerPeaks`), as are peaks below a significance (`markLowSignificancePeaks`). Peaks near a newly added one are set running again (`runningIfHasNeighbors`).
* **Other variants.** The same machinery runs multiplane fits with CRLB plane weighting (`multi_plane/plane_weighting.py`). ZOLA-3D has `process/gpu/LocalizationMultiEmitter_.java`.

*SMAPpy has* single-emitter ROI fits only. NOTES "Crowding compresses z" measured that a neighbour 2-4 px away halves the z slope, and `logl_rel` filtering is the only remedy.

*Why.* High-density data and crowded 3D.

*How it would fit.* A residual-and-refit second pass in `pipeline.py`:
1. After the first fit, render the fitted models into a per-frame residual (the C++ models can evaluate themselves).
2. Re-detect in the residual.
3. Refit each ROI with its neighbours' models subtracted, one Gauss-Seidel round.

Simpler than full DAOSTORM, and it attacks the measured bias directly. Gate it behind a setting; measure with the `neighbour_nm` truth the simulator already writes.

#### 10. PERPL relative-position model fits -- PERPL  (M; Apache-2.0)
*What it does.*
* Builds the distribution of pairwise relative positions (in 1D along an axis, in 2D, or 3D Euclidean) within a limit, as a histogram or KDE.
* Fits parametric RPD models by least squares (`modelling/modelling_general.py:466`). An example is a polygon of n vertices (`rot_2d_symm_fit.py:122-165`), with Gaussian broadening, repeated-localization and substructure terms at zero, and a linear isotropic background (`:168-217`).
* Selects among models over a configured sweep by AIC/AICc (`:548-552`; `perpl_models_config.yaml`).

The approach works for structures too sparse to see in any one copy.

*SMAPpy has* pair-distance curves (`spatial_statistics.py`, `Index.pair_histogram`) and Line Profile's AIC verdicts, but no structural models for g(r).

*How it would fit.* A model-fit part in Spatial statistics that fits a chosen model to its own pooled pair-distance curve. Prefer unbinned likelihood, as Line Profile does, over PERPL's least squares on a KDE.

#### 11. Bead-free registration of separate camera files without an initial guess -- storm-analysis micrometry  (S-M; re-implement, no licence file)
*What it does.* `micrometry/micrometry.py`, `quads.py` (Lang et al. 2010, astrometry.net):
* Build "quads" from 4 nearby points, with A and B defining a frame and the positions of C and D in that frame as a scale- and rotation-invariant hash.
* Match quads between the two lists; each match proposes an affine transform.
* Score each proposal by `bg_p + (1 - bg_p) * mean(exp(-d^2/2))` over nearest neighbours (`fgProbability`).
* Refine with control points (`refine_map.py`).

*SMAPpy has* `calibrate/transform.py`: pair-vector voting finds the offset for split-frame and same-camera data, then projective/polynomial refinement. The vote assumes no large rotation, scale or mirror.

*Why.* Porting-plan item C "Calibrate transform from separate camera files (#39)": two cameras have unknown mirror, rotation and magnification.

*How it would fit.* As the global step of `register_channels` when the layout is "separate files". Fiducial beads, or the first frames' bright spots, as the point sets.

### Better than SMAPpy's current implementation
* **Drift weighting:** DME's per-localization, per-axis precision in the cost (item 3) is the one concrete algorithmic improvement over SMAPpy's COMET kernel, which uses one isotropic sigma (`csrc/drift.hpp:78`). Everything else SMAPpy's drift does (spline in time, two-stage, grouped, preflight cost estimate, RCC with tiled z) matches or beats DME, AIM, NanoJ-Core and storm-analysis's `xyz_drift_correction.py`.
* **Channel registration:** none better for split-frame data. SMAPpy's pair voting plus projective/polynomial map is a cross-correlation on points; Picasso `registration.py` (from signal or beads) does the same in spirit. Only the no-initial-guess case (item 11) is missing.
* **FRC:** none better. `frc.py` (split by time blocks, repeats as the error bar, tiling, FPC for planes, envelope against NeNA) is more careful than storm-analysis/frc, photonpy `smlm/frc.py` or the SQUIRREL FRC.
* **Fitter vs Gpufit:** Gpufit's MLE (`Gpufit/estimators/mle.cuh:61`) is plain Poisson, with no read-noise term and the observed-information Hessian. SMAPpy's expected-information Hessian and readout variance (NOTES "A fit that starts at no background") are better. Gpufit is useful only as a template for the GPU port (J #10): one block per fit, pixels on threads, Gauss-Jordan per fit (`cuda_gaussjordan.cu`), and box constraints (`gpufit.h:49-50`). Its `spline_3d_multichannel` and `spline_3d_phase_multichannel` (4Pi) models show the layout for SMAPpy's `global.hpp` on GPU.

### User experience and automation
* **Report the drift's own noise.** DME prints and plots the split-half RMSD on every run (`dme.py:150-195`). SMAPpy's COMET/RCC panels could show "noise x/y/z ~ 0.7/0.6/0.7 nm (split-half)" with both half-curves, dashed, in the drift figure.
* **AIM's `exclude_self`.** It takes each segment out of the reference while that segment is aligned, removing the pull towards zero (`picasso/aim.py:423-470`). The same bias argument applies to SMAPpy's RCC against a global reference, if one is ever added.
* **Model-fit ROI viewer.** LocMoFit's `@fitViewer` shows data, fitted model and residual side by side per site, with parameter tables (`plugins/+ROIManager/+Analyze/LocMoFit_siteGallery.m`). SMAPpy's ROI manager plus the E6 gallery would host this: the model drawn over each ROI.
* **Expected-LL z-score as a quality column.** It filters failed model fits per ROI without hand-tuned thresholds (`fit.m:266-274`).
* **Scale sweep.** smlm_datafusion3d chooses its scale parameter automatically from the cost landscape on a few pairs (`MATLAB/scale_sweep.m`); a fusion plugin should do the same instead of exposing "scale".
* **sCMOS calibration validation.** Picasso's `validate_calibration` checks a map against a movie's chip size and binning before it is used, a check the camera database could make.
* **uiPSF field-dependent models.** SMAPpy offers only `zernike`, `voxel`, `insitu` and `insitu_pupil` (`uipsf/__init__.py:34-37`); uiPSF also has `FD` and `insitu_FD` (`uiPSF/config/psftype/`). Using an FD model needs a spatially varying spline in the fitter (one calibration per tile, chosen by ROI position). That is the part to plan before offering FD.

### Licence-restricted (re-implement only)
* **SQUIRREL error map / RSE / RSP and FRC map** -- NanoJ-SQUIRREL, GPL-3. The paper (Culley 2018) gives the model: reference ≈ alpha * (SR ⊗ G_sigma) + beta, with alpha, beta and sigma minimised in squared error after registration. RSE = RMS(ref - model), RSP = Pearson(ref, model).
* **Decorrelation resolution** -- ImDecorr, GPL-3. From the paper (Descloux 2019) and readable in `funcs/getDcorr.m`:
  1. Normalise the spectrum `In = F/|F|` inside r < 1 and compute the decorrelation function `d(r) = corr(F, In * mask(r))` over 50 radii.
  2. Repeat after `Ng` = 10 Gaussian high-pass filters with log-spaced widths; refine twice.
  3. The resolution is `2 * pixel / k_c` at the highest local maximum.

  Single-image, so it works on image layers (widefield, SIM, imported reconstructions), where FRC cannot. For localizations it cannot see blink-correlated errors, the very reason `frc.py` splits by time, so offer it for images only. S-M.
* **ZOLA-3D** -- GPL-3:
  * Zernike phase retrieval from beads with an explicit refractive-index-mismatch, depth-dependent PSF (`process/cpu/ZernikePhaseRetrieval.java`);
  * multi-emitter fitting;
  * lateral "wobble" calibration from beads (`postprocess/WobbleCorrection.java`), relevant only to Gaussian fits; a spline/uiPSF model already contains the wobble;
  * 3D cross-correlation drift in FFT tiles.

  The depth-dependent PSF is covered for SMAPpy by uiPSF in situ and by item 4. Paper: Aristov 2018.
* **3D-DAOSTORM multi-fit, multiplane plane weighting, micrometry** -- storm-analysis, no licence on those directories. The papers (Babcock 2012; Lang 2010) describe the algorithms well enough (items 9, 11). `spliner` and `L1H` are under an academic non-commercial licence: do not copy. `rolling_ball_bgr`, `wavelet_bgr`, `dbscan` and `voronoi` are MIT.
* **LocMoFit and fit3Dcspline** are GPL-3 in their repositories but Ries-lab code, so the choice is Jonas's. Code by Yu-Le Wu (LocMoFit) may need his agreement for a BSD relicence.

### Not worth it / already in SMAPpy
* AIM (Picasso, Ma 2024): intersection counting on a grid; very fast for DNA-PAINT, but SMAPpy's grouped COMET spline (2-5 s) and RCC already cover speed and robustness.
* NanoJ-Core drift and registration (image cross-correlation): RCC and `layer_registration.py` do this.
* storm-analysis `rcc/`, `frc/`, `sa_utilities/xyz_drift_correction.py`, `fiducials.py`: SMAPpy's `rcc.py`, `frc.py` and `fiducials.py` are more thorough.
* storm-analysis `spliner` / fit3Dcspline spline calibration: already ported (`calibrate/core.py`).
* Gaussian astigmatic z (`zfit.py` in Picasso, `sxsy2z.m` in fit3Dcspline): deliberately not ported (NOTES); spline/uiPSF supersede it.
* Picasso `average.py`: iterative rotational cross-correlation of rendered particles. Superseded by item 2, and Picasso is surveyed elsewhere.
* photonpy RCC/FRC/phasor localization: nothing beyond SMAPpy.
* storm-analysis `mortensen.py` (an analytic CRLB): SMAPpy returns the fitter's own CRLB.
* Gpufit as a dependency: its MLE is weaker than SMAPpy's; use it only as a design reference for a CUDA port.

### References
* Cnossen J, Cui TJ, Joo C, Smith C. Drift correction in localization microscopy using entropy minimization. *Opt Express* 29, 27961 (2021). https://doi.org/10.1364/OE.426620
* Ma H, et al. Toward drift-free high-throughput nanoscopy through adaptive intersection maximization. *Sci Adv* 10, eadm7765 (2024). https://doi.org/10.1126/sciadv.adm7765
* Culley S, et al. Quantitative mapping and minimization of super-resolution optical imaging artifacts (SQUIRREL). *Nat Methods* 15, 263 (2018). https://doi.org/10.1038/nmeth.4605 (DOI from the repository README)
* Descloux A, Grußmayer KS, Radenovic A. Parameter-free image resolution estimation based on decorrelation analysis. *Nat Methods* 16, 918 (2019). https://doi.org/10.1038/s41592-019-0515-7 (DOI not re-checked online)
* Heydarian H, et al. Template-free 2D particle fusion in localization microscopy. *Nat Methods* 15, 781 (2018). https://doi.org/10.1038/s41592-018-0136-6 (DOI not re-checked online)
* Heydarian H, et al. 3D particle averaging and detection of macromolecular symmetry in localization microscopy. *Nat Commun* 12, 2847 (2021). https://doi.org/10.1038/s41467-021-22006-5
* Huijben TAPM, et al. Detecting structural heterogeneity in single-molecule localization microscopy data. *Nat Commun* 12, 3791 (2021). https://doi.org/10.1038/s41467-021-24106-8
* Wu Y-L, Hoess P, Tschanz A, Matti U, Mund M, Ries J. Maximum-likelihood model fitting for quantitative analysis of SMLM data (LocMoFit). *Nat Methods* 20, 139 (2023). https://doi.org/10.1038/s41592-022-01676-z
* Curd AP, et al. Nanoscale pattern extraction from relative positions of sparse 3D localizations (PERPL). *Nano Lett* 21, 1213 (2021). https://doi.org/10.1021/acs.nanolett.0c03332
* Li Y, et al. Real-time 3D single-molecule localization using experimental point spread functions (fit3Dcspline; depth-dependent correction). *Nat Methods* 15, 367 (2018). https://doi.org/10.1038/nmeth.4661 (DOI not re-checked online)
* Liu S, et al. Universal inverse modeling of point spread functions for SMLM localization and microscope characterization (uiPSF). *Nat Methods* 21, 1082 (2024). https://doi.org/10.1038/s41592-024-02282-x
* Aristov A, Lelandais B, Rensen E, Zimmer C. ZOLA-3D allows flexible 3D localization microscopy over an adjustable axial range. *Nat Commun* 9, 2409 (2018). https://doi.org/10.1038/s41467-018-04709-4 (DOI not re-checked online)
* Przybylski A, Thiel B, Keller-Findeisen J, Stock B, Bates M. Gpufit: an open-source toolkit for GPU-accelerated curve fitting. *Sci Rep* 7, 15722 (2017). https://doi.org/10.1038/s41598-017-15313-9 (DOI not re-checked online)
* Babcock H, Sigal YM, Zhuang X. A high-density 3D localization algorithm for stochastic optical reconstruction microscopy (3D-DAOSTORM). *Opt Nanoscopy* 1, 6 (2012). https://doi.org/10.1186/2192-2853-1-6
* Huang F, et al. Video-rate nanoscopy using sCMOS camera-specific single-molecule localization algorithms. *Nat Methods* 10, 653 (2013). https://doi.org/10.1038/nmeth.2488 (DOI not re-checked online)
* Smith CS, Stallinga S, Lidke KA, Rieger B, Grunwald D. Probability-based particle detection that enables threshold-free and robust in vivo single-molecule tracking (GLRT). *Mol Biol Cell* 26, 4057 (2015). https://doi.org/10.1091/mbc.E15-06-0448
* Lang D, Hogg DW, Mierle K, Blanton M, Roweis S. Astrometry.net: blind astrometric calibration of arbitrary astronomical images (the quad hashing behind micrometry). *Astron J* 139, 1782 (2010). https://doi.org/10.1088/0004-6256/139/5/1782 (DOI not re-checked online)
