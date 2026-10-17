## Cluster analysis, spatial statistics, molecule counting and colocalisation

Surveyed: Locan (read its `analysis/` and `process/cluster/` modules), PoCA (the successor of SR-Tesseler and Coloc-Tesseler, with its detection cleaner), ClusterViSu (SharpViSu), Clus-DoC, RSMLM (ToMATo, Ripley-L, Voronoi, DBSCAN), BaClAva (the Rubin-Delanchy / Griffié Bayesian engine), FOCAL3D, SMoLR, pointpats, AuToMATo, and the data and scoring code of the Nieves et al. 2023 benchmark. For StormGraph, CAML, SuperResNET, LAMA, DDC and the counting papers I found no public repository, so they are covered from the papers only. **Main conclusion:** SMAPpy's radial statistics are already better than any open tool I read. Its pair counts are exact and come from C++, with set-covariance edge correction for any window shape, Monte Carlo envelopes, a global MAD test and a *keep blinks* null. Locan, Clus-DoC and RSMLM correct edges only crudely or not at all. What SMAPpy lacks is everything **above the blink**: a correction for the repeated blinks of one molecule (DDC-style time-separated pairs, the Sengupta fit), per-localisation colocalisation (CBC/DoC, Coloc-Tesseler), a single-molecule diagnostic per cluster (convex-hull expectation), and the counting fit (B4). The counting fit can be built almost entirely from the probability-generating-function (pgf) machinery already in `smappy/npc.py`.

### The software

| tool | what | language | licence | port route | repo (commit read) |
|---|---|---|---|---|---|
| Locan | analysis library: CBC, Ripley, g(r), NN, blinking, convex-hull/variance expectation, accumulation check, HDBSCAN/DBSCAN | Python | BSD-3 | port with attribution | super-resolution-microscopy/locan (8703cad) |
| PoCA | SR-Tesseler + Coloc-Tesseler successor: Voronoi objects, DBSCAN, Ripley, object coloc, detection cleaner, macros/Python | C++/Qt/OpenGL | LGPL-3 (file headers) | re-implement from paper/description | flevet/PoCA (d384e9f) |
| ClusterViSu (SharpViSu) | Voronoi clustering with a Monte Carlo threshold | MATLAB | GPL-3 | re-implement | andronovl/SharpViSu (8a528f4) |
| Clus-DoC | DoC (CBC) colocalisation + DBSCAN + Ripley per ROI | MATLAB | none found | re-implement from paper | PRNicovich/ClusDoC (df56c24) |
| RSMLM | ToMATo, persistence-Ripley, Voronoi, DBSCAN, simulation | R + C++ | GPL (>= 2) | re-implement | JeremyPike/RSMLM (37cc428) |
| BaClAva | Bayesian cluster analysis GUI (Rubin-Delanchy model, Griffié code) | Python + R | MIT (code says "adapted from Griffié et al.") | port with attribution, but check Griffié's original licence | saskiakutz/BaClAva (a887e3a) |
| FOCAL3D | grid-based density clustering + parameter scan | Python | GPL-3 | re-implement | MilsteinLab/FOCAL3D (b4b552a) |
| SMoLR | R package: KDE threshold clusters, DBSCAN, features, spatstat bridge | R | LGPL-3 | re-implement | ErasmusOIC/SMoLR (0c7f1ef) |
| pointpats | F, G, J, K, L with edge corrections and simulation tests; Knox space-time | Python | BSD-3 | port with attribution | pysal/pointpats (95ca252) |
| AuToMATo | parameter-free ToMATo | Python | AGPL-3 | re-implement | m-a-huber/AuToMATo (985cea3) |
| Nieves benchmark | ARI/IoU scoring; 10 simulated scenarios with and without multiple blinking | R + CSV | GPL-3 | data usable as an external test set (not shipped) | DJ-Nieves/ARI-and-IoU-cluster-analysis-evaluation (3b0580b) |
| StormGraph, CAML, SuperResNET, LAMA, DDC | graph clustering / ML / network analysis / CBC GUI / blink correction | MATLAB / Python | no repo found (LAMA: GPL-3 per its homepage, *unverified*) | re-implement from paper | -- |

### Worth porting  (ranked, most valuable first)

#### 1. Counting fit (B4) -- Lee 2012 / Puchner 2013 / Fricke 2015 / Hummer 2016; SMAP `brightnessdistribution.m`  (M, re-implement from papers; SMAP is Jonas's own)
**What the field does.** All of these methods share one idea (from the papers). A fluorophore is functional with probability q: maturation, labelling, activation. If functional, it produces a random number of detected events (blinks or bursts), geometric in the simplest kinetic model (Lee 2012) and negative binomial in SMAP and Puchner. A complex of N copies therefore yields the N-fold convolution of that distribution, and the oligomer state is fitted to the histogram of events per cluster. Hummer 2016 shows that the per-molecule distribution takes a universal form with two or three constants, independent of the detailed photophysics. Those constants transfer between experiments, so they can be calibrated on a monomer reference and then held fixed (Fricke 2015 does the same with a monomer/dimer reference). *(From the abstracts; take the exact functional form from Hummer 2016 itself.)* Rollins 2015 (aggregated Markov model on each cluster's whole time trace) is more accurate when molecules overlap in time, but it is much heavier (L): leave it for later.

**What SMAPpy has.** *Blinking* (`plugins/blinking.py:162`, `region_blinking`) already writes `n_localizations`, `n_blinks` and `n_bursts` per region. Its `molecule_blinking` (`:208`) gives blinks per isolated molecule on the view, which is exactly the monomer calibration histogram. `smappy/npc.py` already has the per-copy pgf `_copy_pgf(u, le, p) = 1 - le + le p u / (1 - (1 - p) u)` (`npc.py:92`), the evaluation on roots of unity with an FFT back to a pmf (`joint_pmf`, `:121`), a likelihood conditioned on a lower cut (`fit`, `:145`, `k >= k_min`) and Hessian errors (`_errors`, `:187`). SMAP's fit (wave4_research.md) used least squares on the cumulative histogram, scanned N0 as an integer, and had a bug (`p_ref` undefined).

**What it should look like.**
* `smappy/counting.py`, functions on arrays:
  * `molecule_pgf(u, q, p, model="geometric"|"hummer")`, the one-copy pgf.
  * `cluster_pmf(weights_N, q, p, n_max)`, which computes Σ_N w_N G(u)^N on n_max roots of unity followed by one inverse FFT. Optionally the number of complexes per cluster is a zero-truncated Poisson with mean m (the composition of the two pgfs, P(G(u)), as in SMAP), which models two complexes that fall into one cluster.
  * `fit_counts(counts, n_min, model, fixed={...})`, a maximum-likelihood fit conditioned on `n >= n_min`. Clusters with fewer events are never found, which is the same reasoning as `k_min` in the NPC fit.
* **Three modes.**
  * (a) *Calibrate*: fit q and p on the monomer histogram, either *Blinking*'s blinks per molecule or regions known to be monomers.
  * (b) *Stoichiometry*: q and p fixed, fit the fractions w_1..w_Nmax (softmax-parametrised), with Nmax chosen by AIC as in `line_profile.py`.
  * (c) *Copy number*: a single N0 with binomial labelling (SMAP's model). Profile the likelihood over integer N0 instead of scanning it with least squares.
* As a `SiteAnalysisPlugin` (`roi.py:311`), `Analysis/Regions/Summarize/Counting`, reading `n_bursts` (or `n_blinks`) from the *Blinking* evaluation. Allow q, p to come from the NPC labelling-efficiency evaluation, which fits the same LE and p with the same pgf: the natural in-cell calibration that Thevathasan 2019 makes possible.
* Output the histogram with the model's pmf, the fractions with errors, and the AIC table.
* Tests: `simulate.kinetics` with known N, LE and blink number; check that the fractions come back. A page with the burst-splitting figure (dark time T versus overcounting).

**Why.** B4 is the open item in the plan. The heavy part is already written and tested in `npc.py`, so this is a few days of work.

#### 2. Time-separated pair correlation (DDC's key idea) -- Bohrer et al. 2021  (S-M, re-implement from paper)
**What it does** *(from the paper)*. Pairs of localisations separated in time by much more than a fluorophore's survival time cannot come from the same molecule. Their distance distribution is therefore the true distance distribution between different molecules. DDC then removes repeats by maximising a likelihood against that reference. The full removal is L and needs the authors' code, which I did not find.

**What SMAPpy has.** The *keep blinks* null (`spatial_statistics.py:323`) moves whole `group_id` blinks, which takes care of consecutive frames only. A dSTORM molecule that blinks again after the merge's dark time is two blinks, and still clusters with itself; the module docstring and the page say so (page line 60: "Blinking is clustering"). The Sengupta correction is explicitly not ported (`spatial_statistics.py:61`).

**The port, as SMAPpy-native.**
* Give `Index.pair_histogram` (`csrc/neighbours.cpp:470`) an optional frame array per side and a minimum |Δframe| T, and count only those pairs.
* The CSR expectation changes only in the pair total: replace n(n-1) by N_T = Σ_{i≠j} 1[|t_i - t_j| > T]. That is computable exactly in O(F) from the frame histogram, with no simulation. `expected_counts` (`:281`) already takes `pairs` as an argument, so g_T = H_T / E_T drops into `curves_of` unchanged.
* g_T(r) is then the pair correlation of *molecules*. It assumes that which molecules are on at which time is unrelated to where they are, which holds for SMLM's stochastic activation; bleaching lowers the density uniformly, and N_T accounts for it.
* Offer it as a *null / estimator* choice next to *keep blinks*: "pairs further apart than T frames". The envelopes stay valid, because simulated points get the data's frames.
* The same mask gives a time-separated cross-correlation between two channels, and a time-separated DBSCAN core count. A *Neighbourhood* count of "neighbours more than T frames away" is a blink-free local density.

**Why.** It is the only blink correction that needs no model of the photophysics and no calibration. It answers the question the page currently can only warn about. Test: simulated random molecules with many blinks per molecule (`simulate`'s `blinks=`); g_T must stay inside the CSR band where g leaves it.

#### 3. Sengupta PC-PALM fit of g(r) -- Sengupta 2011, Veatch 2012  (M, re-implement from paper)
**What it does** *(from the papers)*. It fits g(r) = g_repeat(r) + g_molecules(r) ⊛ g_PSF:
* g_repeat ∝ (1/ρ) exp(-r²/4σ²)/(4πσ²) is the term of one molecule seen several times, averaged over the measured precision distribution;
* g_molecules = 1 + A exp(-r/ξ) is the model of the molecules' own clustering.

It returns the cluster size ξ, the amplitude A, the number of molecules per cluster and the average number of repeats per molecule. SMAPpy already stores pair and expected counts per region (`per_region` arrays) and has `xy_err_nm`, so this is a fit on top of `curves_of`, with the σ distribution taken from the table. It would be an approach of *Spatial statistics*, giving ξ and A as `Result.values` so that *Curves* and the region table can use them. It complements item 2: item 2 removes the repeats; this models them, and works when the frame column is gone (merged or exported data).

#### 4. Per-localisation colocalisation: CBC / DoC -- Malkusch 2012 (Locan, Clus-DoC); Coloc-Tesseler -- Levet 2019 (PoCA)  (S, Locan BSD: port with attribution)
**CBC** (`locan/analysis/cbc.py:44-116`). For each localisation of A:
* the neighbour counts of A and of B within radii r_k = R·k/n_steps (default R = 100, n_steps = 10), each normalised as N(r)/N(R)·R²/r² (`:96-101`);
* the Spearman rank correlation of the two curves (`:104`);
* times exp(-E/R) (`:110-114`).

E should be the distance to the nearest B, as in Clus-DoC (`DoCCalc.m:120-121`, `knnsearch`) and *(from the paper)* Malkusch. **Locan uses the farthest neighbour within R** (`np.max` over `distances_2`, `:110`), which looks like a bug; follow Clus-DoC there. Clus-DoC first keeps only points whose Getis-Franklin local L is above the CSR expectation (`DoCCalc.m`, `Lr_Threshold`). Its normalisations of D1max and D2max are inconsistent, as its own comments say.

**SMAPpy** has nothing per localisation. *Neighbourhood* has exactly the two-layer convention needed (*around* / *neighbours from*) and the C++ counts (`neighbourhood.py:151`), so CBC is one more *quantity*: n_steps radii counted in one pass, Spearman per row, a recipe column `cbc_<layer>`. Default R = 100 nm: Malkusch's examples, *unverified*. The summary is the histogram of the column, or the fraction of localisations with CBC > 0.4 (Clus-DoC's convention *(from the paper; Pageon 2016)*); per region, it becomes a `per_region` value.

**Coloc-Tesseler** (`poca_coloctesselerplugin/ColocTesseler.cpp:51-113`, LGPL: re-implement):
* each localisation gets its own channel's Voronoi density over the channel mean, and the other channel's normalised density there (PoCA takes the nearest other-channel localisation's density, `:99-108`);
* the log-log scattergram, Spearman per ROI, and Manders-like fractions above density thresholds (`ColocTesselerBasicCommands.cpp:376-420`).

SMAPpy's `voronoi_density` (`neighbourhood.py:458`) gives the first half. A two-layer variant gives the rest; the scattergram with two threshold lines drawn on it is a natural figure.

#### 5. Single-molecule test per cluster: convex-hull and variance expectation -- Ebert et al. 2022 (Locan)  (S, BSD)
**What it does.** It compares each cluster's hull area (and perimeter; volume in 3D) and its coordinate variance with what n points drawn from one Gaussian of the cluster's own precision would give. The 2D expected area is analytic: 3π·C(n,3)·∫Φ(x)^(n-3) φ(x)³ dx · σ² (`convex_hull_expectation.py:122-160`). Perimeter and 3D come from lookup tables simulated for n = 3..200 (`:74-110`, `resources/*.npy`). The variance expectation carries the small-sample bias (`position_variance_expectation.py:164-171`). A cluster with ratio ≈ 1 is resolution-limited, i.e. one molecule's blinks.

**SMAPpy.** *Cluster shape* (`cluster_shape.py`) measures hull area, rg and covariance, but deliberately subtracts no precision (docstring, last paragraph). Adding `area_expected_nm2` and `area_ratio` (and `rg_ratio`), computed from the region's median `xy_err_nm`, would let *Filter ROIs* drop single-molecule "clusters". That is the most common artefact DBSCAN produces on dSTORM data. Port the formula; regenerate the tables with SMAPpy's own simulator rather than copying the .npy files.

#### 6. ToMATo persistence clustering -- Chazal 2013, Pike 2020 (RSMLM)  (S-M, GPL: re-implement)
**What it does** (`RSMLM/R/clusterUtilities.R:20-33`, `src/Core.h:82-210`).
* The density is the neighbour count within r (`pointdensity`, frequency).
* Points are taken in decreasing density over the r-graph. A point with no denser neighbour starts a mode; otherwise it joins its densest neighbour's cluster (union-find), and two clusters merge only when the lower peak's prominence (peak density minus current density) is below the persistence threshold τ.
* The persistence diagram (birth/death of every mode, `tomatoDiagram`, `:284`) is the tool for choosing τ: a gap separates the real modes from the noise.

**SMAPpy.** DBSCAN with the cell search, HDBSCAN optional. ToMATo would be a fourth *algorithm* part in *Find clusters*: the density is *Neighbourhood*'s count, the graph comes from `Index`, and union-find is already in `dbscan_join`. About 100 lines plus the diagram figure, which belongs in a Preview. HDBSCAN already covers similar ground, but ToMATo is faster, has two intuitive parameters (r, τ), and the diagram is a better UX than HDBSCAN's condensed tree. It was among the seven methods scored by Nieves 2023.

#### 7. Bayesian cluster analysis -- Rubin-Delanchy 2015 / Griffié 2016 (BaClAva)  (M, MIT per BaClAva)
**What it does** (`BaClAva/pythonr/internal_bayesian.R`).
* *Proposals* (`Kclust_parallel`, `:112-200`): over a grid of r (10..200 step 5) and L thresholds (5..500 step 5), keep the points with Getis-Franklin L(r) ≥ th (toroidal edge, `:51`) and join them in the graph of distances < 2r ("Ripley' K based"), or < r with border points (DBSCAN-like).
* *Score* (`scorewprec`, `:485-523`):
  * each cluster's marginal likelihood is that of a Gaussian cluster, with a per-point precision weight 1/(s² + σ_i²), its centre integrated over the ROI and its size s integrated on a grid (`mcgaussprec`, `:6-49`);
  * background points are uniform;
  * plus a Dirichlet-process prior on the labels (`plabel`, `:468-483`; α = 20, p_b = 0.5).
* The best-scoring proposal is the answer, and the score surface over (r, th) is the figure.

`plabel` writes `B*log(pb) + (1 - B)*log(1 - pb)` (`:474`), where (N - B) is expected. That looks like a bug; check it against the original supplementary code.

**SMAPpy.** It has the local L (*Neighbourhood* *express as* local L) and the joining (DBSCAN's union-find), so proposals are cheap. The O(N²) `dist` matrix of R becomes the C++ index. It fits as a `per_region` plugin run in a block over ROIs (the method assumes ROIs of a few µm), writing `cluster_id` per ROI and the posterior-best (r, th) as values. Value: a parameter-free answer users trust in the immunology community. Caveat: the model has no blinking either; it over-splits or over-clusters multi-blink data unless it is fed merged blinks or a time-separated density (item 2).

#### 8. Label-density variation / temporal accumulation -- Baumgart 2016, Spahn 2016 (Locan `AccumulationClusterCheck`)  (S, BSD)
**What it does** (`accumulation_analysis.py:49-90`). It clusters a growing subset of the data (Locan: random subsets; Spahn: the first f·T frames). For each subset it records the relative area covered by clusters η and the mean density inside clusters ρ_in, plotted against the overall density. For blinking-only pseudo-clusters, η and ρ_in grow in step with the density; real clusters reach a plateau. Locan marks the module as exploratory. SMAPpy has every piece: subsets by frame, then DBSCAN/HDBSCAN on arrays, then hull area. It would be a small *Measure* plugin ("Accumulation check") with one figure. It is an independent diagnostic, especially useful because it needs no blinking model.

#### 9. Nearest-neighbour functions G, F, J with envelopes -- pointpats  (S, BSD)
pointpats has G (event to nearest event), F (empty space) and J = (1-G)/(1-F), with Kaplan-Meier/Hanisch/border corrections (`distance_statistics.py:256-856`) and simulation tests (`:1362-1680`). SMAPpy has no NN-distance analysis at all. Adding G and F as approaches of *Spatial statistics* reuses the window, `Window.uniform` and the envelope code. The cross-G (nearest B to each A) is the simplest colocalisation curve, and is what PoCA's *nearest locs multicolor* plugin shows. A border correction against the raster window (distance to the mask edge by a distance transform) is enough.

#### 10. Inhomogeneous K / g -- Baddeley, Møller & Waagepetersen 2000 (spatstat idea)  (S-M, from paper)
It weights each pair by 1/(λ(x_i)λ(x_j)), with λ a heavily smoothed intensity, and so separates clustering at 50-200 nm from cell-scale density gradients, which the occupied-area window only partly handles. It needs a per-point weight in `pair_histogram` (C++), and λ from the existing Gaussian *Neighbourhood* count at a large σ. The envelopes have to use inhomogeneous Poisson simulations, by thinning `Window.uniform`.

### Better than SMAPpy's current implementation

* **Blinking in pair correlation: SMAPpy is good, but not the best available.** Its *keep blinks* null is better than anything in Locan, Clus-DoC, RSMLM or PoCA, which test against plain CSR. The *merge blinks* layer convention is consistently respected by DBSCAN, Neighbourhood and Spatial statistics. But it handles only consecutive-frame repeats (`group_id`). The best alternatives handle **repeat blinks of one molecule after long dark times**: DDC's time-separated pairs (item 2, the simplest and most general) and Sengupta's fitted repeat term (item 3). A cheap intermediate step is to let *keep blinks* move any id column (`groups` currently comes from `group_id` only), so that whole *molecules* move together: *Blinking*'s `molecules` (`blinking.py:150`), linked with the long dark time. That is S, and it makes the null blink the way the data do over the whole acquisition.
* **Blinking in clustering: also below the best.** DBSCAN/HDBSCAN on merged blinks still find each multi-blink molecule as a cluster when k is small. No surveyed tool solves this inside the clusterer: Nieves 2023 shows multiple blinking degrades all seven algorithms *(from the paper)*. The practical best is (a) a time-separated core count (item 2) and (b) a post-hoc single-molecule test (item 5).
* **Automatic merge dark time** -- PoCA's cleaner (`poca_detectionsetplugin/CleanerCommand.cpp:378-463`):
  * it links in a radius (default 50 nm) with up to `maxDarkTime` (10) frames;
  * it histograms the dark times and fits y = a + b·exp(-t/τ);
  * it sets the dark time to 3τ + 0.5 frames (`:463`).

  It also fits the blinks-per-molecule histogram with Lee's geometric (`EquationFit.cpp:106-112`). SMAPpy's *Blinking* shows the same histograms (`draw_histograms`) but fits nothing, and the merge's *dark frames* is set by hand. Adding a geometric fit of the dark times (discrete; Locan fits a continuous exponential to frame counts, `analysis/blinking.py:414-450`, which is slightly wrong) and a suggested dark frames shown as a `FormState.hint` is S.
* **Voronoi threshold** -- ClusterViSu (`VoronoiMonteCarlo.m`): the threshold is where the data's histogram of cell areas first drops below the mean of CSR simulations (default bins 2·n^(1/3), limited to 3 × the median CSR area; linear interpolation). SMAPpy uses SR-Tesseler's fixed multiple (`clusters.py:188`), as its docstring says. The Monte Carlo version is easy with `Window.uniform`, but it inherits the blinking problem: it should simulate with the *keep blinks* placements, which ClusterViSu cannot do. Worth adding as an *auto* value of the factor, with that twist.
* **Ripley edge correction**: none better found. Locan's K/L/H uses the bounding-box area and no correction (`ripley.py:78-118`). Its g(r) normalises by the ring area and mean density only (`radial_distribution.py:101-150`). Clus-DoC uses a polygon area with no correction. pointpats offers isotropic and translation corrections for a hull polygon, which is equivalent to SMAPpy's set covariance for convex windows, while SMAPpy's handles any raster. SMAPpy's is as good or better throughout.

### User experience and automation

* **Persistence diagram as the parameter chooser** (RSMLM `plotTomatoDiagram`): in SMAPpy, a Preview figure of *Find clusters* where τ is a horizontal line one drags; the same idea works for HDBSCAN's cluster stability.
* **Parameter scans with the cluster count against the parameter** (FOCAL3D `Scan_Parameters_FOCAL_DBSCAN.py:191-353`; Locan `serial_clustering`, `process/cluster/utils.py:19`): a *scan* button in *Find clusters* that runs the module-level `dbscan` over a range of eps/k and plots the number of clusters and the fraction of points in clusters. The plateau is the objective choice FOCAL argues for.
* **Coloc scattergram with draggable thresholds** (Coloc-Tesseler): fits SMAPpy's `ShapeEdit` figure panels.
* **Ground-truth scoring of clusterings** (Nieves 2023: ARI over localisations and IoU of cluster hulls, `ARI_IoU_source.R:12-115`): *Ground truth* could score `cluster_id` against a simulated `cluster_id_true`. Their 10 scenarios with and without multiple blinking (CSV x, y, index; 382 MB, GPL repository) are a ready external benchmark for a study script. Note that they carry no frame column, so they cannot test items 2/8.
* **Macro recorder and Python console** (PoCA): SMAPpy's history and chains already cover this; nothing to add.
* **Simulated ground-truth generator in the same GUI** (BaClAva module 1; Locan's Thomas/Matern/Neyman-Scott makers, `simulation/simulate_locdata.py:537-1106`): SMAPpy's *Blinking Structure* covers it. A "Thomas process" preset (Gaussian clusters at Poisson centres) would match the literature's standard test case.

### Licence-restricted (re-implement only)

* **ToMATo** -- RSMLM (GPL >= 2), AuToMATo (AGPL-3). Chazal et al. 2013 gives the full algorithm (density, graph, union-find with prominence threshold) in a few lines. AuToMATo picks τ automatically by bootstrapping the persistence diagram *(from its README; unverified)*.
* **Coloc-Tesseler, SR-Tesseler objects, detection cleaner** -- PoCA (LGPL-3). Levet 2015 and 2019 give the densities, the thresholds and the Spearman/Manders definitions. The cleaner's constants are listed above.
* **ClusterViSu Monte Carlo threshold** -- GPL-3; the algorithm is fully stated above.
* **DoC / CBC as Clus-DoC implements it** -- no licence. Use Locan's BSD version (with the nearest-neighbour weight fixed) instead.
* **FOCAL3D** -- GPL-3. It is grid DBSCAN: count per voxel, a 3×3(×3) box sum, core if ≥ minL, cores grown by one 4- or 6-connected step, connected components, clusters smaller than minC removed (`main_FOCAL_3D.py:60-160`). SMAPpy's exact DBSCAN is as fast, so only the scan UX is worth taking.
* **StormGraph** (Scurll et al., bioRxiv 10.1101/515627; no code found): a graph weighted by the localisation uncertainties, then community detection with multi-level clusters and a two-colour overlap. Only from the paper.
* **CAML** (Williamson 2020; no code found): each localisation's distances to its 1..k nearest neighbours fed to a network trained on simulations, giving clustered or not. It could be built on `Index` and a small classifier, but it needs a training-data policy; low priority.
* **SuperResNET** (Khater 2018; no code found, *unverified* whether released): merging, a proximity network, per-blob graph features (degree, modularity, hollowness) and machine-learning classes. The graph features could become extra values of *Cluster shape*.
* **LAMA** (Malkusch & Heilemann 2016; GPL-3 per the paper and its homepage, *unverified*): CBC, Ripley, cluster statistics. Nothing that Locan does not have.

### Not worth it / already in SMAPpy

* DBSCAN (Locan, SMoLR, PoCA, RSMLM): SMAPpy's C++ version with nearest-core border assignment is faster and order-independent.
* HDBSCAN (Locan wraps the same `hdbscan` package): equal.
* Voronoi density and SR-Tesseler-style clusters: equal (`neighbourhood.voronoi_cells`, `clusters.voronoi_clusters`).
* Ripley K/L/H and g(r) from Locan, RSMLM, Clus-DoC and PoCA: SMAPpy's is better (exact edge correction, envelopes, MAD test, pooling over regions).
* KDE-threshold clusters (SMoLR, one of Nieves' seven methods): *Segment regions*' density-peak method and a Gaussian *Neighbourhood* count with a filter cover it.
* Cluster shape (hull area, covariance axes, rg; SMoLR features, RSMLM `clusterStats`): equal. Only the expectation ratio (item 5) is missing.
* Blink statistics (Locan `BlinkStatistics`): equal, except for the fit (above).
* Locan's localisation-property correlation matrix: trivial; *Statistics* is enough.
* pointpats' space-time Knox/Mantel tests: of theoretical interest for blinking, but item 2 answers the same question directly.

### References

* Malkusch S et al. (2012) Coordinate-based colocalization analysis of single-molecule localization microscopy data. Histochem Cell Biol. https://doi.org/10.1007/s00418-011-0880-5 (DOI from Locan's docstring)
* Pageon SV et al. (2016) Functional role of T-cell receptor nanoclusters... (Clus-DoC). PNAS. https://doi.org/10.1073/pnas.1607436113
* Levet F et al. (2015) SR-Tesseler. Nat Methods. https://doi.org/10.1038/nmeth.3579 (from the PoCA README)
* Levet F et al. (2019) A tessellation-based colocalization analysis approach for SMLM (Coloc-Tesseler). Nat Commun. https://doi.org/10.1038/s41467-019-10007-4
* Levet F, Sibarita J-B (2023) PoCA. Nat Methods. https://doi.org/10.1038/s41592-023-01811-4 (from the README)
* Andronov L et al. (2016) ClusterViSu. Sci Rep. https://doi.org/10.1038/srep24084
* Rubin-Delanchy P et al. (2015) Bayesian cluster identification in SMLM data. Nat Methods. https://doi.org/10.1038/nmeth.3612
* Griffié J et al. (2016) A Bayesian cluster analysis method for SMLM data. Nat Protoc. https://doi.org/10.1038/nprot.2016.149 (from a search listing, not opened)
* Pike JA et al. (2020) Topological data analysis quantifies biological nano-structure from SMLM (ToMATo). Bioinformatics. https://doi.org/10.1093/bioinformatics/btz788
* Chazal F, Guibas LJ, Oudot SY, Skraba P (2013) Persistence-based clustering in Riemannian manifolds. J ACM. DOI 10.1145/2535927 (from memory, not verified)
* Nino DF, Djayakarsana D, Milstein JN (2020) FOCAL3D. PLoS Comput Biol. https://doi.org/10.1371/journal.pcbi.1008479
* Nieves DJ et al. (2023) A framework for evaluating the performance of SMLM cluster analysis algorithms. Nat Methods. https://doi.org/10.1038/s41592-022-01750-6
* Williamson DJ et al. (2020) Machine learning for cluster analysis of localization microscopy data (CAML). Nat Commun. https://doi.org/10.1038/s41467-020-15293-x (from a search listing)
* Khater IM et al. (2018) Super resolution network analysis defines the molecular architecture of caveolae and caveolin-1 scaffolds (SuperResNET). Sci Rep. https://doi.org/10.1038/s41598-018-27216-4
* Malkusch S, Heilemann M (2016) LAMA. Sci Rep. https://doi.org/10.1038/srep34486 (from a search listing)
* Ebert V et al. (2022) Convex hull as diagnostic tool in SMLM. Bioinformatics. https://doi.org/10.1093/bioinformatics/btac700 (from Locan's docstring)
* Baumgart F et al. (2016) Varying label density allows artifact-free analysis of membrane-protein nanoclusters. Nat Methods. https://doi.org/10.1038/nmeth.3897
* Spahn C et al. (2016) Temporal accumulation analysis... Nat Methods. https://doi.org/10.1038/nmeth.4065
* Sengupta P et al. (2011) Probing protein heterogeneity in the plasma membrane using PALM and pair correlation analysis. Nat Methods. https://doi.org/10.1038/nmeth.1704
* Veatch SL et al. (2012) Correlation functions quantify super-resolution images and estimate apparent clustering due to over-counting. PLoS One. https://doi.org/10.1371/journal.pone.0031457
* Bohrer CH et al. (2021) A pairwise distance distribution correction (DDC) algorithm to eliminate blinking-caused artifacts in SMLM. Nat Methods. https://doi.org/10.1038/s41592-021-01154-y
* Lee SH et al. (2012) Counting single photoactivatable fluorescent molecules by PALM. PNAS. https://doi.org/10.1073/pnas.1215175109
* Puchner EM et al. (2013) Counting molecules in single organelles with superresolution microscopy... PNAS. https://doi.org/10.1073/pnas.1309676110
* Rollins GC, Shin JY, Bustamante C, Pressé S (2015) Stochastic approach to the molecular counting problem in superresolution microscopy. PNAS. https://doi.org/10.1073/pnas.1408071112
* Hummer G, Fricke F, Heilemann M (2016) Model-independent counting of molecules in SMLM. Mol Biol Cell. https://doi.org/10.1091/mbc.E16-07-0525
* Fricke F et al. (2015) One, two or three? Probing the stoichiometry of membrane proteins by SMLM. Sci Rep. https://doi.org/10.1038/srep14072
* Baddeley AJ, Møller J, Waagepetersen R (2000) Non- and semi-parametric estimation of interaction in inhomogeneous point patterns. Stat Neerl. DOI 10.1111/1467-9574.00144 (from memory, not verified)

All the DOIs above without a note were checked in PubMed.
