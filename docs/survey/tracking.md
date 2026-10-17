## Single-particle tracking and track analysis

Surveyed: the linkers of TrackMate (LAP, Kalman, overlap), laptrack, trackpy and u-track; the diffusion-state analyses of Spot-On, saSPT, ExTrack and vbSPT; step finders (AutoStepFinder, BNP-Step); and trajectory classification (u-track's MSS, TraJClassifier, the AnDi simulators). **The linker is not where SMAPpy falls short.** For sparse single-molecule tracking its *simple* and *global* linkers are as good as a LAP tracker with gap closing; a LAP linker would only matter in crowded data. **The analysis is where the gap is.** *Tracking analysis* fits a one-lag jump mixture that treats jumps as independent. It has no correction for molecules leaving the focus, no per-track state assignment, no state transitions and no step finding. Spot-On, saSPT and ExTrack each supply one of these, and two of the three are MIT-licensed.

### The software

| tool | what | language | licence | port route | repo (commit read) |
| --- | --- | --- | --- | --- | --- |
| TrackMate | Fiji tracking GUI: LAP (Jaqaman), Kalman, nearest neighbour, overlap; track features, filters, TrackScheme, exporters | Java | GPL-3 (`LICENSE.txt`) | re-implement from the paper / description | fiji/TrackMate (14c933d) |
| laptrack | Python LAP tracker: frame-to-frame LAP, then gap closing, splitting and merging as one segment LAP | Python | BSD-3 (`LICENSE.rst`) | port with attribution | yfukai/laptrack (13acf99) |
| trackpy | Crocker–Grier linker with subnet resolution, adaptive search, predictors; MSD, drift from tracks | Python | BSD-3 (`LICENSE`) | port with attribution | soft-matter/trackpy (9a52b00) |
| u-track | LAP with motion models (Brownian, directed, switching) and Kalman filtering; gap closing, merge/split; MSS classification | MATLAB | GPL-3 (`GPL-License.txt`, `LICENSE`) | re-implement from the paper | DanuserLab/u-track (204b819) |
| Spot-On (fastspt) | kinetic model of jump-length histograms at several lags, 2–3 states, z-escape correction | Python | GPL-3 (`LICENSE`) | re-implement from the paper | gitlab tjian-darzacq-lab/Spot-On-cli (3c155b3) |
| saSPT | state arrays: a variational Bayes mixture over a grid of (D, σ), per track, with a defocalisation correction | Python | MIT | port with attribution | alecheckert/saspt (ad0fa30) |
| ExTrack | HMM of diffusive states with localisation error marginalised, transition rates, per-localisation state probabilities | Python | MIT (`LICENSE`; `Licenses/cc-by-sa-4.0.txt` presumably the docs) | port with attribution | vanTeeffelenLab/ExTrack (24c79cd) |
| vbSPT | variational Bayes HMM: number of states, D, transition rates | MATLAB | not read: no GitHub repository found | *(unverified)*, re-implement from the paper | – |
| AutoStepFinder | iterative binary-split step fitting with an S-score | Python + MATLAB | **no licence file** | re-implement from the paper | jacobkers/Loeff-Kerssemakers-et-al-AutoStepFinder (f22ea07) |
| BNP-Step | Bayesian non-parametric step finder (Gibbs sampling) | Python | **no licence file** | re-implement from the paper | LabPresse/BNP-Step (c8c6621) |
| TraJClassifier | random-forest classifier of sub-trajectories (normal, anomalous, confined, directed) | Java (+R via Renjin) | MIT | port with attribution | thorstenwagner/ij-trajectory-classifier (fc4513a) |
| andi_datasets | simulators for anomalous diffusion, including models with state changes (dimerisation, traps, confinement) | Python | MIT | port with attribution | AnDiChallenge/andi_datasets (11ba748) |
| Swift | Bayesian linker for sptPALM (Endesfelder lab) | – | closed, beta on request | re-implement from the paper (none published yet) | – |

### What SMAPpy has now

These are the parts the comparisons below refer to.

* **Link tracks** (`tracking/link.py`).
  * *Candidates*: one C++ cell index (`Index.pairs`), queried once per lag 1..memory+1, with one maximum jump R for every lag.
  * *simple*: visits the candidates ordered by (frame, d², lag) and takes a link when both ends are free (`greedy_links`).
  * *global*: per frame, a `linear_sum_assignment` on each connected group of competing candidates. The gain is R² − d², so leaving a point unlinked costs R².
  * The NOTES.md benchmark ("Tracking"): the two linkers agree to 0.3% below a jump/spacing ratio of 0.1. *global* gives exactly trackpy's links.
* **Tracking analysis** (`tracking/analysis.py`).
  * *MSD*: Michalet's optimal number of lags, computed exactly from the MSD covariance (`msd_covariance`, Isserlis' theorem). The offset comes from `xy_err_nm`. Alpha is a log-log slope.
  * *Jump fit* (`fit_jumps`): an EM fit of 1–3 states to the squared jumps at **one** lag. Each jump uses its own error variance σa² + σb². The jumps are treated as independent, so no errors are given.
* **Simulation** (`simulate/tracks.py`): states that switch, blinking, a localisation precision and spurious localisations. There is **no axial escape** (a track's length is geometric, independent of D) and no motion blur.
* **Rendering**: a tracks render mode, and a tail in the movie export. Per-track `track_D_um2_per_s` and `track_alpha` can be written as columns.

### Worth porting  (ranked, most valuable first)

#### 1. Multi-lag jump-distance fit with z-escape (defocalisation) correction -- Spot-On's model, saSPT's correction  (M; saSPT's code: port with attribution; Spot-On: re-implement)

**What it does.**
* **Spot-On** fits 2 or 3 states jointly to the jump-length distributions at lags 1..TimePoints−1 (default 7 lags: `fastspt.py:33,81,146`). It allows `GapsAllowed` dark frames and uses only the first `JumpsToConsider`=4 jumps of each track (`fastspt.py:116-131`). That last choice stops long-lived, slow or bound tracks from dominating the fit.
* **The model** (`_compute_2states`, `fastspt.py:357-375`): each state is a 2D Rayleigh of variance 2(D·Δt + σ²). The *free* state is multiplied by Z_corr(Δt, D), the probability of staying inside a slab of thickness ΔZ_use. Z_corr comes from an absorbing-boundary series summed until a term is below 1e-10 (`C_AbsorBoundAUTO`, `fastspt.py:559-590`). The slab is widened empirically: ΔZ_use = dZ + a·√D + b (`fastspt.py:446`; a = 0.15716 and b = 0.20811 are in a comment at `:442`).
* **The fit** is least squares on the PDF or the CDF histogram.
* **saSPT's correction** is cleaner and needs no empirical a and b (`defoc.py:108-154`, `f_remain_rbm`). A uniform profile inside |z| ≤ dz/2 is convolved, frame by frame, with the Gaussian kernel of 4DΔt by FFT, the outside zeroed each frame. This gives the fraction still in focus after n frames. The occupations are then divided by it (`defoc_corr_rbme`, `defoc.py:273`).

**What SMAPpy has instead.** `fit_jumps` fits one lag. It uses all jumps of every track, so long tracks weigh more. It has no focus correction, so the fast fraction is underestimated in 2D sptPALM.

**Why it is worth it.** This is the standard bias in 2D SPT of nuclear factors (Hansen 2018). A fast state leaves the ~0.7 µm focal slab within a few frames, so its fraction is underestimated by tens of percent. Several lags pin down D and the fractions far better than one, and the per-lag focus loss is itself information.

**How it would fit SMAPpy.**
* Extend `fit_jumps` to a list of lags. The EM keeps its form: the membership is per (jump, lag). Each state's fraction at lag k is multiplied by the in-focus fraction f(D, k), renormalised, so the M step for D is no longer closed-form. A 1D Brent step per state inside each EM round would do.
* Keep the per-jump σ² that SMAPpy already has: Spot-On uses one LocError, so SMAPpy's handling of localisation error is the better part to keep.
* Add a *first jumps per track* setting (Spot-On's 4) and an *axial range* in µm (0 or empty = no correction). In 3D tables the correction is the same, with the axial range of the fit.
* Plugin shape: `JumpFitSettings` in `plugins/tracking_analysis.py`. The page `Analysis/Tracking/Tracking analysis.md` needs a new section and a version bump.

**Test data.** `simulate_tracks` needs an axial escape. Give each particle a z inside a slab and end the track (or start a dark run) when |z| > dz/2; the 3D path (`use_z`, `z_range_nm`) already diffuses z. Spot-On-cli also ships sample datasets (`datasets/datasets.py`).

#### 2. Per-track state posterior and state arrays -- saSPT  (M, port with attribution)

**What it does.**
* For every track it evaluates the exact Gaussian likelihood of its whole jump vector on a grid of D and σ (`lik.py:146-202`). The default grid is 100 values of D, log-spaced from 0.01 to 100 µm²/s, and σ from 0 to 70 nm in 2 nm steps (`constants.py`).
* The covariance is tridiagonal: 2(DΔt + σ²) on the diagonal, −σ² off it, because consecutive jumps share a localisation. Tracks are split into pieces of at most `splitsize`=10 jumps.
* A variational Bayes loop finds the Dirichlet posterior of the state occupations over the grid (`sa.py:325-379`, 200 iterations, concentration parameter 1). Each round: n = Σ R·jumps; then R ∝ L·exp(ψ(n + α)).
* The result is a smooth D spectrum with no number of states chosen, plus each track's posterior over D (`posterior_assignment_probabilities`). The occupations can then be focus-corrected.
* There are also fBM (`FBMELikelihood`) and gamma-approximation likelihoods.

**What SMAPpy has instead.** A mixture of 1–3 states with a fixed number of states chosen by BIC. Jumps are treated as independent, which the docstring admits. Nothing is assigned per track.

**Why it is worth it.**
* No number of states has to be chosen: the D spectrum shows how many populations there are.
* The likelihood is correct for the correlated jumps.
* Each track gets a posterior mean D or most likely state, which SMAPpy can write as a column (`track_state`, `track_D_post_um2_per_s`). That column then serves everything SMAPpy already does with columns: colour the tracks render by state, filter, make ROIs, run a block per state.

**How it would fit SMAPpy.**
* A third approach in *Tracking analysis* (*D spectrum*), or a plugin of its own, `Analysis/Tracking/State arrays`.
* Use SMAPpy's own σ per localisation rather than gridding σ. The covariance becomes tridiagonal with per-localisation σ², which is still cheap with a banded Cholesky. That drops the σ axis and fits SMAPpy's precision-first approach (NOTES, "offset from the precisions").
* The grid evaluation vectorises over tracks of equal length, as saSPT does.
* The output is `Result.arrays` (the spectrum) plus a recipe column, `per_region`.

**Test data.** `simulate_tracks` with 2–3 states, `switch_probability`=0.

#### 3. HMM with transitions, localisation error marginalised -- ExTrack (vbSPT as the Bayesian variant)  (M–L, ExTrack: port with attribution; vbSPT: re-implement from the paper)

**What it does.** ExTrack's `P_Cs_inter_bound_stats` (`tracking.py:160-390`) and its thresholded and "ages" variants (`:618`, `:1116`) compute each track's likelihood under a hidden Markov model.
* The true positions are integrated out analytically, the Gaussians fused step by step.
* State sequences that differ only more than `frame_len` steps ago are merged (moment matching, `fuse_gaussians`, `:420`), which keeps long tracks tractable.
* Inputs are the per-state D, the fractions, a transition matrix, the localisation error (one value or one per localisation: the error message at `:192`), and a bleaching or leaving probability `pBL`.
* `auto_fitting.fit_2states` / `fit_3states` fit D, the fractions and the transition rates. `segmentation.py` returns per-localisation state probabilities.

vbSPT (Persson 2013) is the variational-Bayes HMM with model selection over the number of states. Its code was not read.

**What SMAPpy has instead.** Nothing models state switching. `simulate_tracks` already simulates it (`switch_probability`), so the truth for a test exists.

**Why it is worth it.** Binding and unbinding rates are the biological answer most sptPALM experiments want. A static mixture cannot give them. Per-localisation states also open dwell-time histograms of the bound state and maps of where binding happens.

**How it would fit SMAPpy.**
* A plugin `Analysis/Tracking/Diffusive states (HMM)`, with the algorithm in `smappy/tracking/hmm.py`.
* Output: a per-localisation `state_prob_k` / `state` recipe column; D, fractions and the rate matrix (in s⁻¹ from the frame time) in `Result.values`.
* The simplest faithful port is ExTrack's forward recursion in numpy (or C++ if slow), keeping `frame_len` of about 4–6.
* It should run after item 2, and could use item 2's spectrum to start the fit.

**Test data.** `simulate_tracks(d_um2_s=(0.02, 1.0), fractions=(0.5, 0.5), switch_probability=0.05)`. Its `state_true` column is per localisation.

#### 4. Step finding for MINFLUX and processive tracks -- AutoStepFinder (BNP-Step as the alternative)  (S–M, re-implement from the paper: no licence)

**What it does.** AutoStepFinder (`python/stepfindCore.py`):
* **Split** (`A121_split_until_ready`, `:47-143`): greedily insert the step that most reduces the residual of a piecewise-constant fit. Track a *counter-fit* (steps placed midway between the fit's steps). Record the S-score, mean((x − counterfit)²) / mean((x − fit)²), for each step count.
* **Evaluate** (`A122_eval_Scurve`, `:251-267`): subtract the baseline from 1 to the last S value. The best count is at the maximum, accepted if it exceeds `tresH`=0.15.
* **Two passes** (`AutoSteppyfinder.py:71-110`, `multiPass`): the first fit is subtracted and the residual fitted again, which catches small steps that big ones hid.

BNP-Step instead samples the number of steps, their heights and the noise by Gibbs sampling. It ships a BIC comparison (`bic_tools.py`).

**What SMAPpy has instead.** Nothing. NOTES.md ("Which column names the tracks") and `wave4_design.md:47` list "step finding to come". SMAP's StepsMINFLUX, the most maintained of SMAP's MINFLUX plugins, uses AutoStepFinder (`wave4_research.md:55-67`).

**How it would fit SMAPpy.**
* A `per_region` plugin `Analysis/Tracking/Find steps`, with a *track id* field through `tracking.columns.track_column` and a projection on the track's main axis (PCA) or a chosen axis.
* Output: per-localisation `step_id` / `step_level_nm`, and dwell times and step sizes as `Result.arrays` for *Curves*-style pooling.
* The split loop is O(N·steps) with cumulative sums, so it takes a day.

**Test data.** Simulate a staircase track (8 nm steps, exponential dwells, MINFLUX noise) in `simulate/tracks.py`. AutoStepFinder ships test data (`python/test_data`, `matlab/test_data`), but with no licence it is better to simulate your own.

#### 5. Lag- and track-aware search radius and a likelihood linking cost -- u-track (re-implement), Swift (paper not yet out), laptrack (cost framework, BSD)  (S–M)

**What it does.** u-track's gap-closing cost (`costMatRandomDirectedSwitchingMotionCloseGaps.m:37-114, 217-358`):
* Each track end gets its own search radius: `brownStdMult` × that track's own displacement std (`:336`) × the gap raised to `brownScaling` (√gap by default, `timeReachConfB`), bounded by `minSearchRadius` and `maxSearchRadius`.
* The radius can be scaled by the local nearest-neighbour distance (`useLocalDensity`, `closestDistScale`).
* A gap of n frames is penalised by `gapPenalty`ⁿ.
* An amplitude-ratio window applies for merging and splitting (`ampRatioLimit`).

TrackMate adds a feature penalty to d²: cost = d²·(1 + Σ 1.5·w·|f₁−f₂|/(f₁+f₂))² (`FeaturePenaltyCostFunction.java:35-83`).

**What SMAPpy has instead.** One R for every lag (`link.py` docstring: "the same maximum jump holds for a gap as for one frame"). There is no gap penalty: a 3-frame gap and a 1-frame link cost the same d². The precision-scaled cost was "left out" (NOTES).

**Why it is (moderately) worth it.** In the mixed-mobility sptPALM case, one R tuned for the fast state lets a slow track's gap closures reach R far away, into its neighbours. Tuned for the slow state, R cuts the fast tracks. A cost of −log p(jump | D_track, σa, σb, gap) = d²/(2(2·D·k·Δt + σa² + σb²)) + log terms, with D from the track so far (or two passes: link, estimate D per track, link again), handles both. It gives a natural gap penalty and would use the precisions SMAPpy already has. The benchmark harness (`studies/tracking/bench.py`) can measure the gain at once: simulate 2 states with D a factor of 50 apart.

**How it would fit SMAPpy.** Optional per-candidate weights in `link_global`'s `gain`. Candidates are generated at R_max as now, and a candidate is accepted only when d² ≤ (radius of its end)². Size S for the per-track radius, M for the likelihood cost.

#### 6. Drift from the tracks themselves -- trackpy  (S, port with attribution)

**What it does.** `motion.compute_drift` (`motion.py:271`): the mean displacement of all particles between consecutive frames, accumulated (optionally smoothed). `subtract_drift` (`:318`) removes it.

**What SMAPpy has instead.** Drift from beads (`fiducials.py`), RCC and COMET. SPT movies often have no beads and are too sparse for RCC.

**How it would fit SMAPpy.** A `Drift` from tracks, the shape of `drift_fiducials.py` (*Fiducials*), but with every track as a weak fiducial. The robust version: the median jump per frame, then COMET's spline (`smappy/fiducials.py`) for smoothing. **Caveat for the page**: collective flow is indistinguishable from drift.

#### 7. Trajectory classification: MSS and a confinement or directed test -- u-track (re-implement), TraJClassifier (port)  (S for MSS; L for a classifier, low priority)

**What they do.**
* **u-track MSS** (`trackMSSAnalysis.m`): moments of order 0–6 of the displacement, their scaling exponents, and the slope of exponent against order. The track is called immobile, confined, Brownian or directed against thresholds tabulated from 35 500 simulations as functions of track length (`threshMSS2D_p10` etc., `:138-190`). There are no 3D optimal thresholds (`:164`).
* **TraJClassifier**: windowed random-forest classification of sub-trajectories. Its defaults are a minimum track length of 160 and a window of 60 positions (`TraJClassifier_.java:101-102`): too long for sptPALM tracks of 5–30 localisations.

**What SMAPpy has.** Alpha per track. For SMAP's processive-motor analysis, `Single_tracking_analysis` (a robust velocity fit) is still on SMAP's side (`wave4_research.md`).

**Worth it?** MSS is a small, vectorisable addition to `tracking/analysis.py` (same bincount machinery over |Δr|^ν). It needs its own threshold tables from SMAPpy's simulator, not u-track's. A confinement radius per track (the MSD plateau, MSD(∞) = R²·const) is equally cheap. A classifier is not worth it at SMLM track lengths.

#### 8. Better test data for the analysis -- andi_datasets  (S, port with attribution)

`models_phenom` (MIT) simulates
* fBM (`disp_fbm`, `:33`)
* multi-state switching with α and D per state (`multi_state`, `:437`)
* dimerisation (`:697`)
* immobile traps (`:937`)
* confinement compartments and condensates (`_make_condensates`, `:605`).

These are the ground truths items 2, 3 and 7 need beyond pure Brownian motion. SMAPpy's `simulate_tracks` could gain *confined* and *immobile-trap* states in a few dozen lines, ported or written from the AnDi 2 description (Muñoz-Gil 2021).

### Better than SMAPpy's current implementation

**The linker: is SMAPpy's global assignment as good as LAP with gap closing?** For the data SMAPpy targets, yes; it is structurally different only in ways that matter when the data are crowded.

* **The LAP (laptrack, TrackMate) works in two stages.**
  * *Stage 1* (`_tracking.py:282-367`) links consecutive frames only. Its cost is d², and its "no link" alternative costs 1.05 × the 90th percentile of all accepted costs (`_cost_matrix.py:76-170`; TrackMate `TrackerKeys.java:243,252`, `JaqamanSegmentCostMatrixCreator.java:60-67`). SMAPpy's alternative is R².
  * *Stage 2* (`:369-457`, `:549-604`) solves **one global LAP over all segment ends and starts** across the whole movie (`min_weight_full_bipartite_matching`, `_optimization.py`), optionally with splitting and merging.
* **SMAPpy's *global*** decides gap links frame by frame, in the same assignment as the 1-frame links into that frame (`link_global`). A gap closure therefore competes with links of other tracks into the same frame, but cannot be revised by what happens later.
* **Where LAP is better.** Stage 2 sees every end and start at once. Two particles that both blink in the same window can be untangled. And no 1-frame link is ever chosen over a better gap closure merely because it was decided first: frame-to-frame links are made without knowledge of gaps either, but the gap stage cannot then be stolen by a greedy earlier choice.
* **Where SMAPpy is equal or better.** On sparse data the problem is bridging dark frames, and SMAPpy's benchmark shows 98.6–99.6% of true links below a jump/spacing ratio of 0.1, with *global* finding 0.3–3% more links up to 0.4 (NOTES.md "Tracking"). That regime is exactly where the two-stage LAP and the per-frame assignment give the same answer. The LAP's percentile alternative cost is arguably *worse* for SPT: it is data-dependent and shrinks when most jumps are short, which cuts fast molecules.
* **What would actually improve linking** is the cost, not the solver (item 5): a lag- and D-dependent radius or likelihood, which u-track has and both SMAPpy and laptrack lack (laptrack's default `gap_closing_cutoff` is the same squared distance as for one frame, `_tracking.py:132`).
* **A LAP gap-closing pass would be an S-size addition if wanted.** Run *simple*/*global* with memory 0, then one `linear_sum_assignment` over segment ends→starts within memory+1 frames on the same `Candidates`. Benchmark it in `studies/tracking/bench.py`; laptrack (BSD) is a reference to compare against. Expect no gain below a ratio of 0.1.
* **Merging and splitting** (u-track, laptrack, TrackMate) do not apply to single fluorophores and are not worth it.

**Jump-distance analysis.** SMAPpy's MLE on the raw jumps with per-jump σ² is better than Spot-On's least-squares histogram fit with one LocError. Its weaknesses are the single lag, the missing focus correction, the independence assumption and no per-track output (items 1–3).

**MSD.** SMAPpy's exact optimal number of lags (Isserlis covariance) is more careful than trackpy (`motion.imsd`/`emsd`, a fixed max lag) or TrackMate (no MSD fit). Nothing better found.

### User experience and automation

* **Track filters on per-track features, with histograms** (TrackMate `gui/components/FilterGuiPanel.java`, `InitFilterPanel.java`; features in `features/track/*`: duration, mean speed, `CONFINEMENT_RATIO`, `LINEARITY_OF_FORWARD_PROGRESSION`, `MEAN_DIRECTIONAL_CHANGE_RATE` in `TrackMotilityAnalyzer.java:55-65`). In SMAPpy the cheap per-track features (length, span, radius of gyration, confinement ratio, mean jump, straightness) would be recipe columns written by *Tracking analysis*. The layer filter then *is* the track filter. Colour the tracks render by any of them (TrackMate's `FeatureDisplaySelector`).
* **Interoperability: TrackMate XML / CSV readers and writers** (`io/TmXmlReader.java`, `TmXmlWriter.java`, `CSVExporter.java`). Many labs move tracks between TrackMate, trackpy, saSPT and Spot-On. SMAPpy has no tracks importer or exporter beyond its own table. Two `Reader`/`Writer`s in `io/formats.py` (TrackMate XML: spots plus edges → `track_id`; a saSPT/Spot-On CSV of `trajectory, frame, x, y`) are an S-size task. Re-implement from the published XML schema, not the GPL code.
* **Manual track editing in the view** (TrackMate: link or unlink selected spots; TrackScheme for lineages). For SMLM SPT, a light version: select two tracks in the view → *join* or *split at frame*, stored as an edit of the `track_id` recipe. TrackScheme itself (cell lineages) is not needed.
* **Close gaps by interpolation** (TrackMate `action/closegaps/CloseGapsByLinearInterpolation.java`) inserts invented localizations; not appropriate for SMLM.
* **The tracks render already exists**; Spot-On and saSPT add figures SMAPpy could copy: the jump histograms per lag with the model drawn over them (Spot-On's `fastSPT_plot.py`), the D spectrum, and per-track posterior heat maps over time and space (`saspt/plot.py`: `plot_temporal_assignment_probabilities`, `plot_spatial_assignment_probabilities`). The spatial map of the state posterior is the *diffusion map* SMAP's AnalyzeSPT had (`wave4_research.md`). A render mode coloured by `track_state` gives it for free.
* **Batch**: saSPT runs a whole condition's files as one `StateArrayDataset` (`dataset.py`) and reports the occupations per file and pooled. In SMAPpy this is a chain with *Link tracks* → *Tracking analysis* over files (`smappy-batch`), with the per-file `Result.values` collected. It is already possible; a *Summarize* across files for the D spectrum would complete it.

### Licence-restricted (re-implement only)

* **u-track** (GPL-3). What the paper and code comments give to re-implement from:
  * the motion-model search radius (√gap scaling, per-track std, local-density scaling) and the gap penalty
  * the MSS classification and its thresholds, regenerated by simulation (Jaqaman 2008; Ewers 2005).
* **TrackMate** (GPL-3): the feature-penalty cost (the formula is documented in the Javadoc), the Kalman tracker (constant-velocity model, initial and maximum search radius: `KalmanTracker.java:102-153`), the XML schema for interoperability. For Python code of the same LAP, use laptrack (BSD) instead.
* **Spot-On** (GPL-3): the multi-lag kinetic model and the absorbing-boundary Z_corr series (Hansen 2018, equations in the methods). Prefer saSPT's MIT numerical defocalisation (`defoc.py:108-154`), which needs no empirical a and b.
* **AutoStepFinder** (no licence file): the split, counter-fit and S-score algorithm is fully described in Loeff 2021 (DOI in the code, `AutoSteppyfinder.py:46`).
* **BNP-Step** (no licence file; bioRxiv 2023): the Gibbs sampler for a non-parametric step count.
* **vbSPT** (code not found on GitHub, licence unknown): the VB-HMM with automatic number of states (Persson 2013). ExTrack (MIT) covers the HMM part portably.
* **Swift** (closed, unpublished as of the search: the Abbelight page cites it as "in preparation"): nothing to re-implement from yet.

### Not worth it / already in SMAPpy

* Nearest-neighbour or greedy linker (TrackMate kdtree): SMAPpy's *simple* is the same, in C++.
* trackpy's subnet linker and adaptive search (`linking.py:290-315`): SMAPpy's *global* solves every subnet exactly with `linear_sum_assignment` and never overflows (trackpy raised `SubnetOversizeException` from a ratio of 0.4 in SMAPpy's benchmark).
* trackpy predictors (`predict.py`: `NearestVelocityPredict`, `DriftPredict`, `ChannelPredict`) and TrackMate's Kalman tracker: for directed motion (flow, motors). Low value for SMLM SPT; drift is corrected beforehand.
* Overlap and label-image tracking (laptrack `OverLapTrack`, TrackMate `overlap/`): for segmented cells, not localizations.
* Merging and splitting of tracks: single fluorophores do not merge.
* trackpy's MSD and ensemble MSD: SMAPpy's is more careful (optimal lags, offset from the precisions).
* TraJClassifier's random forest: needs tracks of more than 160 positions.
* Track scores against the truth: SMAPpy has `tracking.link.score` (links found and wrong, whole tracks); laptrack's `scores.py` adds split and merge scores that SMLM does not need.

### References

Verified (journal, PubMed or the repository):

* Jaqaman K et al. (2008) Robust single-particle tracking in live-cell time-lapse sequences. *Nat Methods* 5:695–702. https://doi.org/10.1038/nmeth.1237 — LAP with gap closing (u-track).
* Fukai YT, Kawaguchi K (2023) LapTrack: linear assignment particle tracking with tunable metrics. *Bioinformatics* 39:btac799. https://doi.org/10.1093/bioinformatics/btac799
* Tinevez J-Y et al. (2017) TrackMate: an open and extensible platform for single-particle tracking. *Methods* 115:80–90. https://doi.org/10.1016/j.ymeth.2016.09.016 (DOI from the TrackMate README).
* Hansen AS et al. (2018) Robust model-based analysis of single-particle tracking experiments with Spot-On. *eLife* 7:e33125. https://doi.org/10.7554/eLife.33125 — multi-lag kinetic model, z-escape correction.
* Heckert A, Dahal L, Tjian R, Darzacq X (2022) Recovering mixtures of fast-diffusing states from short single-particle trajectories. *eLife* 11:e70169. https://doi.org/10.7554/eLife.70169 — state arrays (saSPT).
* Simon F, Tinevez J-Y, van Teeffelen S (2023) ExTrack characterizes transition kinetics and diffusion in noisy single-particle tracks. *J Cell Biol* 222:e202208059. https://doi.org/10.1083/jcb.202208059
* Persson F, Lindén M, Unoson C, Elf J (2013) Extracting intracellular diffusive states and transition rates from single-molecule tracking data. *Nat Methods* 10:265–269. https://doi.org/10.1038/nmeth.2367 — vbSPT.
* Loeff L, Kerssemakers JWJ, Joo C, Dekker C (2021) AutoStepfinder: a fast and automated step detection method for single-molecule analysis. *Patterns* 2:100256. https://doi.org/10.1016/j.patter.2021.100256 (DOI in the code).
* Rojewski A et al. (2023) An accurate probabilistic step finder for time-series analysis. *bioRxiv*. https://doi.org/10.1101/2023.09.19.558535 — BNP-Step (from the README; authors not checked).
* Muñoz-Gil G et al. (2021) Objective comparison of methods to decode anomalous diffusion. *Nat Commun* 12:6253. https://doi.org/10.1038/s41467-021-26320-w — AnDi (DOI from the andi_datasets README).

DOI from memory, not verified in this session:

* Crocker JC, Grier DG (1996) *J Colloid Interface Sci* 179:298–310, 10.1006/jcis.1996.0217 — the linker of trackpy and SMAP.
* Ewers H et al. (2005) *PNAS* 102:15110–15115, 10.1073/pnas.0504407102 — moment scaling spectrum.
* Wagner T et al. (2017) *PLoS One* 12:e0170165, 10.1371/journal.pone.0170165 — TraJClassifier.
* Ershov D et al. (2022) TrackMate 7, *Nat Methods* 19:829–832, 10.1038/s41592-022-01507-1.
