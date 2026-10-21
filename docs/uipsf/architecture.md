# Architecture, as built

What exists in the code now, module by module, with the APIs as they are. The plan (`docs/uipsf/plan.md`) says where
this is going; this file says where it is. Update it with every milestone.

## Milestone M1: the forward model

```
src/uipsf/
├── device.py         default_device(), resolve(device, dtype), complex_dtype(), to_cpu64()
├── params.py         ParamSpec, Params; transforms Identity, Softplus, Exp, Bounded; prior Gaussian
├── optics/           no parameters, no fitting
│   ├── system.py     Optics(NA, wavelength_nm, n_imm, n_med, pixel_size_nm=(x, y), n_cov=None)
│   │                 choose_pupil_size(optics, z_max_nm, max_phase_step=π/2)
│   ├── grid.py       PupilGrid.build(optics, size, device, dtype): k-space, aperture, k_z, Fresnel, apodisation
│   ├── zernike.py    noll_to_nm, nm_to_noll, zernike_modes(n_max, exclude), zernike_basis(grid, modes, high_na)
│   ├── vectorial.py  dipole_fields(grid, analyser=None) -> [6|3, P, P]; scalar_field(grid) -> [1, P, P]
│   ├── mdft.py       MatrixDFT(grid, (Y, X), spacing_nm)(field, x0, y0) -> [..., Y, X]
│   └── kernels.py    gaussian_otf, sphere_otf, bin_pixels
└── model/
    ├── pupil.py      ZernikePupil(grid, n_max, …)(p) -> (magnitude, phase); PixelPupil
    └── psf.py        VectorPSF(grid, pupil, roi, upsample, kind, analyser, bead_radius_nm)
                      (p, xyz [E, K, 3], depth_nm=None, z_step_nm=None) -> [E, K, Y, X]
```

### How a PSF is computed

```python
optics = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
grid = uipsf.PupilGrid.build(optics, size=64)                 # default device, float32
pupil = uipsf.ZernikePupil(grid, n_max=6)                     # phase: Noll 5–28, magnitude: Noll 2–28
psf = uipsf.VectorPSF(grid, pupil, roi=(21, 21), bead_radius_nm=20)
p = uipsf.Params(psf.specs()).physical()                      # {"pupil.phase", "pupil.magnitude", "psf.sigma"}
I = psf(p, xyz, z_step_nm=50)                                 # expected photons per pixel per photon
```

1. `pupil_field(p)`: magnitude × exp(i·phase), with the magnitude rescaled to the reference energy.
2. `intensity(p, xyz)`:
   - multiply by the defocus wave `exp(2πi k_z,imm z)` (and `exp(2πi k_z,med d)` for depth);
   - multiply by the six dipole fields;
   - take the matrix DFT, which also places the emitter laterally;
   - sum `|E|²` over the fields.
   Shape `[E, K, F, P, P]` before the DFT.
3. `__call__`: Gaussian blur in 2-D, or blur × bead kernel in 3-D along a padded stack; then sum the upsampled samples of each pixel.

Pupil coefficients with a leading emitter axis give one pupil per emitter. That is the slot field dependence will use.

### Contracts the tests hold

- Emitter independence: a tangent on one emitter's parameters changes no other emitter's pixels (`test_psf.py`). The solver's block structure (M2) relies on it.
- Exact gradients for every parameter, including xyz, depth and σ, in float64 (`gradcheck`).
- v1 and vecfit agreement (`test_oracles.py`); conventions in `docs/uipsf/conventions.md`.

### Parameters

- `ParamSpec(name, shape, scope="shared"|"emitter", init, transform, prior, fixed)`.
- `Params(specs, n_emitters)` keeps unconstrained values:
  - `physical()` gives what components see;
  - `shared_vector()` and `emitter_matrix()` pack the free ones for the solver;
  - `replace(shared=, emitter=)` unpacks them;
  - `fix()` and `free()` change which parameters are free;
  - `prior_penalty()` adds the Gaussian priors.
- Components declare specs. In M1 that is `pupil.phase`, `pupil.magnitude` and `psf.sigma`; emitters, photons and background come with the forward model of M2.

### Measured (benchmarks/uipsf/speed.py, float32, torch 2.14)

| Beads × 41 slices | Mac GPU (MPS) | Mac CPU, 8 threads | Linux CPU, 4 cores (cloud) |
|---|---|---|---|
| 6 | 0.015 s | 0.052 s | 0.090 s |
| 20 | 0.044 s | 0.160 s | 0.31 s |
| 40 | 0.091 s | 0.31 s | 0.70 s |

- On the Mac the GPU is 3.5× faster than its own CPU, more than the review's 2.5× for its prototype on the M1 Pro.
- For comparison, the review measured 0.73 s for 6 beads with v1 (TF, eager, 4 CPU cores).
- No CUDA numbers yet.
- The gpu tier (`tests/uipsf_tests/test_gpu.py --gpu`) passes on an Apple-silicon Mac (MPS): the same PSF and gradient as on the CPU. It first found one float64 tensor built on the device, which MPS refuses; the bead kernel is now built on the CPU. Not yet run on CUDA.

## Milestone M2: likelihood and solver

```
src/uipsf/
├── data/dataset.py   Batch(data [E, C, K, Y, X] photons, z_nm [K] | [E, K], s, mask); subset(), to()
│                     Batch.z_from_objective(objective_nm): SMAPpy's z sign, written once
├── model/
│   ├── geometry.py   BeadStackGeometry(drift=False, z_scale=False): xyz[e, k] = xyz_e + (δx_k, δy_k, s·z_k + δz_k)
│   ├── intensity.py  Bleaching(order=1) (default), PerSliceFactor(order, prior_sigma), PerObservation()
│   ├── background.py Constant() (default), Planar(pixel_size_nm)
│   ├── start.py      rough_start(batch, pixel_size_nm): position, photons, background from the data
│   └── forward.py    ForwardModel(psf, geometry, intensity, background)
│                     .specs(batch, start=None) -> [ParamSpec];  (p, batch) -> μ [E, 1, K, Y, X]
├── noise.py          ShiftedPoisson(excess=1), Poisson(), EMCCD(); Camera(conversion, offset, em_gain, read_noise_e)
│                     .deviance(μ, d, s, mask), .derivatives(μ, d, s, mask) -> (∂(½D)/∂μ, Fisher weight)
├── solve/
│   ├── problem.py    Problem(model, noise, batch, chunk_size=None, tangents=None, device=None)
│   │                 .objective(params), .normal_equations(params) -> Normal, .deviance(params), .expected(params)
│   ├── lm.py         LevenbergMarquardt(max_iter=100, tol=1e-3, xtol=1e-3, damping=1e-3).solve(problem, params,
│   │                 progress, cancel) -> FitResult;  schur_solve(normal, λ, D_s, D_e)
│   └── result.py     FitResult: params, history, stop_reason, converged, objective, covariance_shared,
│                     std (CRLB per parameter, physical), condition, remaining, clamped
├── workflows/beads.py fit_beads(model, batch, noise=None, start=None, first_order=4, …) -> FitResult
└── simulate.py       template(E, z_nm, roi), parameters(model, batch, values), simulate(model, params, batch, seed)
```

### How a bead fit runs

```python
model = uipsf.ForwardModel(uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=6), (21, 21), upsample=2,
                                           sigma_nm=(30, 30)))
batch = uipsf.Batch(data=photons[:, None], z_nm=uipsf.Batch.z_from_objective(objective_nm))
result = uipsf.workflows.fit_beads(model, batch)          # Poisson; the start comes from the data
result.params.physical()["pupil.phase"], result.std["pupil.phase"]
```

1. `ForwardModel.specs(batch)` collects every component's `ParamSpec`s, emitter ones started by
   `rough_start`.  Defaults: `emitter.xyz`, `emitter.photons` (softplus), `emitter.bleaching`,
   `emitter.background` (softplus) per bead; `pupil.phase`, `pupil.magnitude`, `psf.sigma` shared;
   `geometry.drift` and `geometry.z_scale` declared but fixed.
2. `Problem.normal_equations` takes each chunk of emitters, computes μ, one JVP per emitter parameter
   (a unit tangent on that parameter of every emitter at once) and one per shared parameter (vmapped),
   and adds JᵀWJ and Jᵀg per emitter in float64 on the CPU.  Priors add their exact least-squares terms.
3. `LevenbergMarquardt` solves (H + λD)δ = −g by the Schur complement of the emitter blocks,
   Jacobi-scaled, with D Moré's running maximum of diag(H); accepts only decreases; adapts λ by the gain
   ratio; stops on decrease, step, gradient, no decrease, max_iter or cancel.  At the end the block
   inverse gives the CRLBs.
4. `fit_beads` runs it twice: Zernikes up to n = 4, then all of them, starting from the first stage.

### Contracts the tests hold

- The deviance is 0 at μ = d, its gradient is that of ½D, its weight the Fisher information (exact sum
  over Poisson counts), its mean ≈ 1 on Poisson data (`test_noise.py`).
- LM solves linear least squares in one undamped step; the Schur solve equals the dense one; priors add
  exactly (`test_solve.py`).
- Rescaling parameters by 10⁻³–10³ leaves every iterate unchanged to 10⁻⁸ (float64); chunk sizes 1–5
  change no sum beyond 10⁻¹¹ and no float32 fit beyond 1 % of a standard error.
- The CRLB matches the spread of 200 simulated localisations within 10 %; a matched fit's deviance is
  1 per degree of freedom.
- Emitter independence through the whole forward model (`test_forward.py`).
- The review's bright, dim and very dim beads are learned to their CRLB without settings
  (`test_benchmark.py`, slow).

### Measured

The review's beads (`benchmarks/uipsf/synthetic/beads.py`; 6 beads + 2 held out, 41 slices, 21 × 21 px, bin 2,
n_max 6; Linux CPU, 4 cores, float32), default settings, one run each:

| Scenario | Iterations | Time | Phase RMS error (CRLB) | z spread (CRLB) | Deviance (held out) |
|---|---|---|---|---|---|
| bright | 13 | 71 s | 0.0008 rad (0.0008) | 0.8 nm (0.7) | 1.004 (1.011) |
| dim | 30 | 80 s | 0.0085 rad (0.0084) | 3.8 nm (5.2) | 1.002 (1.010) |
| very dim | 25 | 122 s | 0.021 rad (0.024) | 15 nm (15) | 1.003 (1.013) |

- For comparison, v1 on the dim beads: 0.059 rad and 61 nm by default, 0.0146 rad and 5.3 nm with the
  best hand-set weights (review, 8 beads, 200 iterations, ~170 s).
- Rewriting photons, background and the pupil in other units (×0.1 … ×10) gives the same dim fit to
  < 0.05 standard errors.
- One iteration of the 6-bead problem costs ~8 s on 4 CPU cores: 59 JVPs at ~0.12 s each, two to three
  forwards' worth.  Each JVP propagates only its own block; a zero tangent on the other block cost as
  much as a real one.  Speed is M7.
- Without the two stages, the dim fit stopped in a local minimum 50 nats up (`workflows/beads.py`).


### First look at real data: Zenodo `1ch_40nm_bead`

The ROIs v1 fitted (`rois/psf_data`, 19 beads × 39 slices × 25 × 25), with v1's optics and n_max 8, fitted
by `fit_beads` (`benchmarks/uipsf/real/zenodo_1ch_beads.py`; rows in `benchmarks/uipsf/results/zenodo_1ch_beads.jsonl`):

| Photons | Iterations | Time (4 CPU cores) | Mean deviance, v2 / v1 | Blur σ, v2 / v1 (nm) |
|---|---|---|---|---|
| bleaching (default) | 23 | 9 min | 0.303 / 0.281 | (75.1, 66.3) / (74.8, 66.6) |
| per observation (v1's) | 23 | 13 min | 0.2813 / 0.2813 | (75.2, 66.3) / (74.8, 66.6) |

- **Zernikes:** with v1's photon model, 40 of 41 phase modes agree with v1 within 2 CRLB (median 0.2).
  The exception is spherical, Noll 11 (v2 −0.001, v1 −0.012 rad): v1 also fits defocus, Z4 = −0.10 rad,
  which v2 leaves out, and Z11 trades against it.  The magnitude agrees to ~0.002.
- **Photons per slice matter here:** bleaching leaves 0.022 more deviance per pixel than a free count per
  slice, so these beads' brightness varies from slice to slice beyond a smooth curve.  Which option, and
  whether a shared `PerSliceFactor` suffices, is for the residual atlas (M3).
- **v1's camera conversion is wrong.** In v1's photons ((ADU − 400)·0.22) the residual variance is only
  0.13–0.36 of the model, hence the mean deviance of 0.28 for both programs.  SMAPpy's camera database
  identifies the camera from the stacks' tags as the Evolve 512 (normal port): 4.7 e⁻/ADU, offset 398.6.
  v1's 0.22 is about 1/4.7, so its photons are ~21× too few, and its likelihood weights and CRLBs are off
  by that factor; the fitted shape is not (a common count scale does not move the maximum-likelihood
  point).  In the right units the variance of the spots' pixels is still 3–5× Poisson, model mismatch or extra noise
  for the residual atlas (M3) to show.  M3 reruns this in SMAPpy's photons (plan §5.7).

## Milestone M3: data, camera, rejection, diagnostics

*Since October 2026 (plan §7.5) the reader, the finder, the pairing and `noise.Camera` below are gone: smappy
reads, converts, finds and pairs (`smappy.calibrate.pupil`, `tests/uipsf_tests/test_bead_rois.py`), and uipsf
takes the `Batch`.  What follows records them as they were built; the measurements stand.*

```
src/uipsf/data/
├── io.py        read_stack(path, camera: noise.Camera, dz_nm=None, mirror=None) -> BeadStack (Micro-Manager TIFF)
│                BeadStack(photons, z_objective_nm, read_noise_var, chip_origin, saturated, source, camera)
│                BeadStack.from_adu(adu, z, Camera): photons and saturation; .mirrored(axis)
├── rois.py      find_beads(stacks, roi_size, detection_sigma_px=2, detection_threshold_sigma=6,
│                min_distance_px=25, padding=5) -> Beads: SMAPpy's detect_beads convention
│                extract_rois(stacks, beads, roi_size, slices=slice(1, None)) -> Batch: the first slice dropped
└── dataset.py   Batch gains field_xy [E, 2] (chip pixels) and stack_id [E]
src/uipsf/diagnostics/camera.py  noise_transfer(data, mu) -> NoiseTransfer(slope, intercept, …): camera check
```

- **Camera:** given by the caller (`noise.Camera`); inside SMAPpy its camera database supplies it.
  `test_camera.py` holds the conversion and read noise to SMAPpy's `CameraMetadata` on the same ADU
  (skips without SMAPpy).  An earlier version read SMAPpy's camera database format itself; it was
  dropped for simplicity (plan §5.7).
- **Bead finding:** SMAPpy's convention; `test_rois.py` checks it against SMAPpy's `detect_beads`.
- **Zenodo `1ch_40nm_bead`:** SMAPpy's database gives 4.7 e⁻/ADU and offset 398.6 for its Evolve 512;
  v1's ROIs are exactly transpose((ADU − 389)·0.22) (docs/uipsf/data.md).  The finder of the time kept 72
  isolated beads in the 31 stacks (the runs below used it); v1 kept 19 of its 20.  The photon-model test is closed too: with the right option
  (`PerSliceFactor` for a laser, `PerObservation` for speckle) the pupil comes back within 4 CRLB and the
  atlas is quiet; with the default the error is larger (`test_atlas.py`).
- **The same 19 beads from the raw stacks**, camera frame, SMAPpy's photons, defaults otherwise
  (`zenodo_1ch_beads.py --source raw`; rows in `benchmarks/uipsf/results/zenodo_1ch_beads.jsonl`):

  | Pixels x × y (nm) | Read noise | Mean deviance | Iterations / time | Blur σ (nm) |
  |---|---|---|---|---|
  | 127 × 116 | 1 e⁻ (SMAPpy's default) | 12.71 | 41 / 15 min | (86.9, 55.2) |
  | 116 × 127 | 1 e⁻ | 12.65 | 47 / 11 min | (66.4, 75.1) |
  | 116 × 127 | 21 e⁻ | **1.390** | 42 / 13 min | (65.9, 74.1) |
  | 127 × 116 | 21 e⁻ | 1.416 | 36 / 13 min | (85.8, 54.0) |

  - **Read noise is what was wrong.** The residuals' variance at zero signal is 20 ADU² (4.5 ADU, ≈ 21 e⁻
    at 4.7 e⁻/ADU), the normal-port read noise of this EMCCD; SMAPpy's default of 1 e⁻ is an sCMOS value
    and its camera database carries none.  With 21 e⁻ the deviance falls from 12.7 to 1.39.
    `diagnostics.noise_transfer` makes this check after every fit.
  - **The pixels are 116 nm in x and 127 nm in y** in the camera's frame: v1's 127 × 116 through its
    transpose.  The deviance prefers it by 0.026 per pixel, ~6000 nats over the 19 beads, and the blur
    then matches v1's.
  - **Zernikes are v1's, transposed:** Noll 6 changes sign (−0.558, v1 +0.558), x and y coma swap
    (0.035 / −0.099, v1 −0.099 / 0.036).
  - The remaining excess, 0.39 per pixel, is for the residual atlas; photons per slice are the first
    suspect (with v1's ROIs, a free count per slice removed 0.022 of 0.30).

### Bead rejection and the bead table

```
src/uipsf/workflows/
├── beads.py      learn_beads(model, batch, noise=None, decisions=None, warm_start=None, rejection=None,
│                 heldout=0.2, first_order=4, …) -> Calibration(model, result, beads, scores, heldout, first)
│                 first_stage(…): stage 1 alone; fit_beads: both stages, every bead
├── scoring.py    score_beads(model, shared, batch, noise) -> BeadScores(deviance, z_bias_nm, photons,
│                 xy_offset_px, masked, fit); localise(…), slice_z(…)
└── rejection.py  RejectionSettings(k=3, floors, max_xy_offset_px=3, max_masked=0.01, min_beads=5)
                  decide(scores, settings, heldout, decisions) -> BeadTable(rows, thresholds); held_out(…)
```

1. Stage 1 on every bead not held out or excluded by hand (skipped after a warm start).
2. Every bead localised on that pupil, shared parameters fixed, and scored: deviance per pixel, z bias
   (each slice localised alone against the stack fit, the median over beads removed: v1's `msez` in nm),
   photons against the median, offset from the ROI centre, masked fraction.
3. `decide`: median + 3·1.4826·MAD per relative score, with floors (deviance 5 % of the median, z bias
   5 nm, photons a factor 1.5) so a tight set is not thinned by its noise; absolute limits for the
   offset and the masked fraction.  Manual decisions win.
4. Stage 2 on the used beads, from stage 1 and their localisations.
5. Every bead scored again on the final pupil (`BeadRow.final`), so the table shows whether leaving a
   bead out was right.

**For the GUI:** `Calibration.beads.as_dicts()` is one row per bead (state, reason, manual decision, every
score and its final value); `learn_beads(..., decisions={bead: "include" | "exclude"}, warm_start=cal)`
is the window's *Calibrate* pressed again with only the marks changed.  The window applies no rule of its own.

Tested (`test_rejection.py`): each planted outlier (a double bead, a 300 nm jump in z, 3× the photons,
3 % saturated pixels) is rejected for its own reason and the fit then reaches its CRLB; 20 matched beads
lose at most one; a warm run with manual decisions equals a cold fit of the same beads.

**On Zenodo `1ch_40nm_bead`** (v1's 20 candidates, camera frame, 21 e⁻, 116 × 127 nm, nothing held out;
`benchmarks/uipsf/real/zenodo_1ch_rejection.py`, 18 min, rows in `benchmarks/uipsf/results/zenodo_1ch_rejection.jsonl`):
- v2 rejects one bead, in Pos10 (stage-1 deviance 2.96 > 2.53).  On the final pupil it is still the worst
  of the 20 (2.18; the others 1.14–1.66).  v1 kept it.
- v1's one rejection, in Pos13 at (178, 32), looks ordinary to v2: final deviance 1.23, z bias 9.6 nm,
  photons within the spread.  v1's reason is not recorded; its photon ratio or a non-positive photon count
  in one slice (`Fitter.relearn`) are the candidates.
- Stage-1 deviances (1.2–3.0) are higher than the final ones (1.1–1.7): stage 1 has modes up to n = 4
  only.  The thresholds are relative, so this shifts them, not the decisions.

### Residual atlas, likelihood-ratio probes, held-out report

```
src/uipsf/diagnostics/
├── atlas.py     residual_atlas(model, params, batch, noise) -> Atlas(mean_deviance, groups, checks, findings)
├── probes.py    probe_terms(model, params, batch, noise, p=1e-6) -> [Probe]; select_terms(…) -> (model, params, added)
└── heldout.py   heldout_report(model, shared, batch, noise) -> HeldOutReport (z-bias curve, CRLB vs spread)
```

- **Atlas:** deviance and signed residuals grouped by slice, |z|, radius, bead and field, each with a
  heterogeneity chi²/dof; and three specific checks, each ≈ 1 on matched data: photons per slice shared
  by all beads (a laser; measured against the per-observation scatter so speckle does not raise it),
  photons per bead and slice (speckle), a plane in the signed residuals (a background ramp).  Thresholds:
  4 null standard errors for the specific checks, 8 for the generic groups (their pixels share fitted
  parameters; matched fits reached 3.8 where 4 would allow 3.3).
- **Probes:** some physics leaves no pattern at all.  A 5 % piezo scale over ±1 µm left the deviance at
  matched level and no group flagged, yet freeing the z scale recovered 1.052 ± 0.002 and gained 355 nats
  (matched: 0.5).  Each optional term (z scale, planar background, per-slice photons) is refitted from the
  fit and flagged when its gain exceeds the χ² quantile at p = 10⁻⁶; `select_terms` adds the largest and
  probes again, since a wrong term can borrow a little of a missing one (a ramp lent the z scale 44 nats,
  its own probe 6876).
- **Held out:** the held-out beads localised stack by stack and slice by slice on the learned pupil: the
  z-bias curve per slice, lateral bias, spread over beads against the CRLB, held-out deviance.
  `learn_beads` attaches the atlas and this report to every `Calibration`.

Tested (`test_atlas.py`, slow): a background ramp, a fluctuating laser and speckle are each named by their
own check and no other specific one; forward selection adds the z scale (recovering 1.05), the planar
background and the per-slice factor first for their own mismatch; 16 matched beads raise no flag, select
no term, and their 4 held-out beads have deviance ≈ 1, no z bias beyond the noise and spread ≈ CRLB.
Not yet covered: polychromatic emission, which needs a model the code does not have (M8).

### The whole pipeline on Zenodo `1ch_40nm_bead` (first run)

`benchmarks/uipsf/real/zenodo_1ch_full.py`: raw stacks, SMAPpy's camera with 21 e⁻, 116 × 127 nm pixels, all 41
slices, 72 beads found, `learn_beads` defaults (n_max 8, 20 % held out), 4 CPU cores:
- **Beads:** 46 used, 14 held out, 12 rejected.  Nine of the rejected are 12–60× dimmer than the median
  bead with a deviance ≈ 1 (faint spots the finder took); two fit badly (deviance 4–5).
- **Time:** stage 1 in 29 min (53 iterations), scoring, stage 2 in 8 iterations; 90 min in all.
- **Held out** (re-judged from the saved fit with the current rules, `zenodo_1ch_resume.py`): two held-out
  "beads" fail them (one with no photons, one faint) and are dropped; the 12 left give a z bias of
  3.9 nm RMS over the central slices (the curve within ±7 nm), lateral 1.5 nm, and a spread over beads
  2.5× the CRLB: on these beads the CRLB is optimistic by that factor, as the deviance of 1.47 implies.
- **Atlas:** mean deviance 1.48; the largest excess is in the spot's shape (radius chi² 4155, |z| 1054,
  slice 608) and over the field (345), then the photons per observation (93) and a background ramp
  (28); a shared per-slice photon pattern is marginal (2.5 against 1.9).
- **Probes** (16 of the used beads, refits capped at 8 iterations, so the gains are lower bounds; the
  full run's own probes were lost to a container restart):

  | Term | Gain (nats) | Threshold | Fitted |
  |---|---|---|---|
  | per-slice photons | 10 410 | 50 | 0.96–1.22 |
  | z scale | 2 193 | 12 | 0.968 |
  | planar background | 1 594 | 43 | slopes ≤ 0.002 photons/px/nm |

  The per-slice factors are not a laser: **the first slice is 22 % brighter** (an artefact of the first
  frame; v1's settings drop the first and last slice for this reason), and the rest follow a smooth
  U of ±4 % over z, shared by all beads, with ~1 % scatter.  Together with a 3 % z scale, the model's z
  dependence is off: the stage, or the depth and index part of the optics.  Field dependence (the
  field chi² of 345) is the other large term, and arrives later as a whole.
- **Defaults changed from this** (decided October 2026): `extract_rois` drops the first slice of every
  stack.  The z scale stays fixed until its cause is understood (plan, "To investigate").

### Bead finders compared (`compare_finders.py`, removed with uipsf's finder)

Since this comparison uipsf's finder follows SMAPpy's convention, and on these stacks both find the same
84 beads.  What the first finder did differently, on the 31 Zenodo `1ch_40nm_bead` stacks, ROI 25 px:
- uipsf's `find_beads` keeps 72; SMAPpy's `detect_beads` finds 84 and keeps 54 after its brightness range
  of 6; 67 are common.  Of v1's 20 beads uipsf keeps all 20; SMAPpy detects 19 and accepts 16.
- SMAPpy alone (17): all below uipsf's threshold (10 robust σ of the background-free projection;
  SMAPpy's is 6 σ after a 2-px smooth): faint spots.
- uipsf alone (5): 4 have another spot within 25 px, which SMAPpy drops in pairs whatever the
  neighbour's brightness (uipsf drops a bead only for a neighbour at least 10 % as bright); 1 lies within
  SMAPpy's 5-px padding of the edge.
- Of the beads SMAPpy's brightness range drops, uipsf's rejection also rejects the faint ones (z bias and
  photons) and two that fit badly, but uses 5 and holds out 4 that fit well.

## Next in M3

- Camera tests against SMAPpy, with the camera from SMAPpy's database; data readers, ROI extraction,
  held-out split, residual atlas.
- `learn_beads`: bead scoring and rejection between the two stages, the bead table `Calibration.beads`,
  manual decisions and warm starts (plan §5.7).
- The Zenodo `1ch_40nm_bead` comparison with v1 in full.

## Milestone M7, in progress: speed

```
src/uipsf/
├── solve/problem.py   Problem(..., linearise=True); .value_and_grad(params) -> (F, g_s, g_e): one backward pass
├── solve/lbfgs.py     GaussNewtonLBFGS(max_iter=300, refresh=10, memory=10, tol, xtol, damping).solve(problem,
│                      params) -> FitResult: L-BFGS whose initial matrix is the damped Gauss–Newton matrix
├── model/psf.py       VectorPSF.pupil_linearisation(p, xyz, …) -> apply(dA [T, P, P]) -> [T, E, K, Y, X]
├── model/forward.py   ForwardModel.pupil_linearisation(p, batch) -> (names, field, apply)
├── optics/mdft.py     MatrixDFT.matrices(x0, y0), .apply(field, Ax, Ay); .real_matrices(x0, y0) -> (By, Bx),
│                      .apply_wide(f [..., 2P, N, P], By, Bx) -> [..., N, Y, 2, X]; wide_product(f, By, Bx)
├── optics/kernels.py  blur_bin_matrices(shape, spacing, sigma, factor) -> (M_y, M_x): blur + binning, no FFT
├── solve/problem.py   TANGENT_BUDGET[device]: output values per batch of Jacobian columns
└── device.py          compiled(fn, device): torch.compile on MPS only (UIPSF_COMPILE=0/1 overrides)
benchmarks/uipsf/profile_lm.py          where an LM iteration's time goes, in forward + backward passes
benchmarks/uipsf/synthetic/solvers.py   the bead benchmark per solver (rows in benchmarks/uipsf/results/solvers.jsonl)
scripts/uipsf/time_v1_zenodo.py         uiPSF v1 timed on Zenodo 1ch_40nm_bead (TensorFlow environment)
```

**Profile** (dim beads, 6 × 41 slices, n_max 6, 4 CPU cores): one LM iteration costs 61 forward + backward
passes, 57 of them the shared Jacobian (53 JVPs at 2.7 forward passes each), 9 the emitter Jacobian; the
sums and the Schur solve are negligible.  In a forward pass: the matrix DFT 31 %, elementwise products
25 %, the complex exp of the defocus waves 18 %, copies 10 %.

**v1 on the same data** (`scripts/uipsf/time_v1_zenodo.py`, the demo's settings, 4 CPU cores, TF 2.20): 17
beads, 7.6 min of learning: L-BFGS-B ran 126 iterations (159 evaluations), then 55 (95) after its
rejection, both stopping on their relative-reduction test; ~1.9 s per evaluation.  v2 took 11–15 min on
v1's 19 ROIs (M3).  So v1 was faster, about 1.7× per bead.

**The pupil's Jacobian, linearised (candidate A).**  The PSF depends on the pupil field A only through
|DFT(A·waves·fields)|², so a column of the Jacobian for a pupil coefficient is 2·Re(E*·DFT(dA·W)) with E
and W computed once; dA comes from a JVP of the small pupil function.  Same normal equations to 10⁻¹⁶
in float64, with and without a bead kernel (`test_solve.py`); 2.5× cheaper than the JVPs for six beads.
Used by default wherever the model offers it (`Problem(linearise=True)`).

**Gauss–Newton L-BFGS (candidates B and D).**  L-BFGS with the damped Gauss–Newton matrix as its initial
matrix, applied by LM's Schur solve; one `value_and_grad` per iteration, the matrix rebuilt every 10
iterations or when the line search fails.  Stopping is decided only at a rebuild, so the CRLBs are LM's.
It stays unit-free: rescaled parameters give the same iterates to 10⁻⁸ (`test_solve.py`).

The dim benchmark (`benchmarks/uipsf/synthetic/solvers.py`, 6 beads, default two-stage `fit_beads`, 4 CPU cores):

| Seed | LM, JVPs | LM, linearised | GN-L-BFGS, linearised | memory 0 (chord) |
|---|---|---|---|---|
| 0 | 117 s, 12 Jacobians | 62 s | 36 s, 6 Jacobians | 36 s |
| 1 | 102 s, 15 | 59 s | 38 s, 6 | 35 s |
| 2 | 562 s, 63: **local minimum**, +100 nats, phase error 0.10 rad | 326 s, same | 37 s, 6: 0.012 rad | 45 s |

On seeds 0 and 1 all solvers end within 0.003 nats of each other, with the same phase error and z spread.
On seed 2 LM wanders for 80 iterations into a minimum 100 nats worse; the L-BFGS solvers do not.

**On Apple's GPU (MPS)** (M1 Pro, 32 GB, torch 2.14.1; `benchmarks/uipsf/results/profile.jsonl`).  The first
profile found MPS only 1.8× faster than the same Mac's CPU: the linearised columns cost as much per
tangent as a whole forward pass, because the matrix DFT ran as 1 476 small complex products per tangent
(6 beads), a few per cent of the GPU's peak, and the blur's FFTs of 42-sample regions were as slow as
the DFT.  Changed, all with the same numbers (the tests unchanged, the normal equations of the compiled
and the eager path identical to the bit):
- *Real, wide matrix DFT* (`MatrixDFT.apply_wide`): real and imaginary parts stacked as [Re; Im], one
  real product instead of four, and every dipole field (and tangent) of an emitter and slice side by
  side in one wide product.  Forward pass 2× faster on both devices; the pupil columns 3.4× on MPS.
- *Blur and binning as two small real matrices* (`blur_bin_matrices`) when there is no bead kernel: the
  Gaussian's transfer function is separable, so the circular blur is a circulant matrix per axis, exact
  to 10⁻¹⁴.  With a bead kernel the 3-D FFT stays.
- *MPS pitfalls*, each a comment where it was avoided: strided `.real`/`.imag` views of a complex tensor
  made a product 3.5× slower; a sum over two non-adjacent axes 150× slower than two sums; and
  `.to("cpu", torch.float64)` from MPS returns **wrong values** (every element, off by up to the data's
  range), which corrupted every simulation on MPS until `simulate` copied first and cast after
  (`test_gpu.py`).
- *Tangent batches per device* (`TANGENT_BUDGET`): 2²¹ output values per batch, at most 16 tangents.
  The old 2¹⁹ gave one tangent at 20 beads; twice the new budget swaps on MPS (4.5 s instead of 1.6 s).
  The same budget is best on both devices; smaller chunks than all beads were never faster.
- *`torch.compile` on MPS* for the per-tangent part (`device.compiled`; batches padded to the first one's
  width, so one shape compiles): 1.7× on that part after 4 s of compiling (1.7 s warm), 12 % off the
  normal equations at 20 beads.  On the CPU only 1.2× after 12 s and a C++ compiler, so off there.
- *Tried and dropped:* a shared z grid of pupil fields (the defocus waves are 4 % of a forward pass on
  MPS, and interpolation would change the numbers); caching the DFT matrices across calls (≈ 8 ms of a
  1.2 s normal-equations call, and hidden state).  Building them from cos and sin directly, without
  complex numbers, took 20 % off their cost.

One `normal_equations` (dim beads, 41 slices, n_max 6), before → after:

| Device | 6 beads | 20 beads | forward + backward, 20 beads |
|---|---|---|---|
| MPS | 1.59 s → 0.41 s | 5.7 s → 1.21 s | 0.118 s → 0.081 s |
| CPU, 8 cores | 2.87 s → 1.02 s | 10.2 s → 3.38 s | 0.279 s → 0.118 s |

The dim benchmark on MPS (`solvers.py --device mps`, 6 beads; the first fit includes compiling):

| Seed | LM | GN-L-BFGS | chord |
|---|---|---|---|
| 0 | 6.3 s, 13 Jacobians | 4.7 s, 6 | 3.8 s, 6 |
| 1 | 4.8 s, 13 | 4.3 s, 6 | 3.9 s, 6 |
| 2 | 25.7 s, 62: the same local minimum, phase error 0.10 rad | 3.8 s, 6: 0.012 rad | 5.4 s, 8 |

Objectives, phase errors and z spreads as on the CPU (within 0.003 nats).  So six dim beads fit in
4–5 s on a laptop GPU, 8× faster than the cloud CPU's 36 s.

With 20 beads (`--beads 20`), every solver reaches the same optimum (within 0.003 nats, phase error at
the CRLB, 0.004–0.005 rad):

| Seed | LM | GN-L-BFGS | chord |
|---|---|---|---|
| 0 | 15.7 s, 14 Jacobians | 12.6 s, 6 | 276 s, 6: **323 iterations**, stopped at max_iter |
| 1 | 13.7 s, 13 | 10.9 s, 6 | 259 s, 7: 305 iterations |
| 2 | 15.4 s, 15 | 10.7 s, 6 | 10.2 s, 6 |

So a 20-bead calibration takes ~11 s with GN-L-BFGS on the Mac's GPU.  Without its memory (chord) the
fit crawls on two seeds of three: the curvature the Gauss–Newton matrix misses between rebuilds grows
with the beads.

**The CPU comparison, continued** (cloud, 4 x86 cores; `benchmarks/uipsf/results/solvers.jsonl`,
`zenodo_1ch_solvers.jsonl`, `z_spacing.jsonl`).

*Two fixes to GN-L-BFGS from Zenodo (41bb693).* Its first Zenodo fit ended 1852 nats above LM's.  Not a
minimum of the physics: one step drove two beads' softplus backgrounds (11 and 8 photons) deep into the
flat tail, where d(background)/du ≈ 0, so no solver could bring them back (their emitter blocks turned
singular).  The line search now refuses a step that shrinks any transform's derivative by more than
100× (`lbfgs._saturates`).  It then reached LM's optimum but kept "accepting" steps of α ≈ 0 at the
float32 floor, each followed by a rebuild; LM's step test now stops it there.

*Zenodo `1ch_40nm_bead`*, v1's 19 beads, raw stacks, 21 e⁻, 116 × 127 nm, n_max 8 (`zenodo_1ch_beads.py
--solver`):

| | v1 (17 beads) | LM, M3 | LM, linearised | GN-L-BFGS | GN-L-BFGS, after the MPS merge |
|---|---|---|---|---|---|
| Time | 7.6 min | 13 min | 343 s | 98 s | **88 s** |
| Objective | | | 321971.3 | 321971.3 | 321971.3 |

Both v2 solvers end at the same Zernikes to the 10⁻⁴ rad printed, mean deviance 1.3904.

*The synthetic matrix* (6 beads unless said, two-stage `fit_beads`, before the MPS merge):

| Scenario | Seeds | LM | GN-L-BFGS | Note |
|---|---|---|---|---|
| bright | 0, 1, 2 | 51–54 s | 17–24 s | same optimum |
| dim | 0, 1 | 61–62 s | 33–37 s | same optimum |
| dim | 2 | 338 s | 29 s | LM: +100 nats, phase error 0.10 rad |
| very dim | 0, 2 | 86–92 s | 29–32 s | same optimum |
| very dim | 1 | 395 s | 56 s | LM: +14 nats, phase error 0.15 rad (CRLB 0.03) |
| dim, rescaled units | 0 | 56 s | 34 s | same optimum as unscaled |
| dim, n_max 8 | 0 | 76 s | 43 s | same optimum |
| dim, 20 beads | 0 | 251 s | 116 s | same optimum |
| dim, 20 beads | 1, 2 | | 98 s; chord 107–117 s | chord converges in 26 iterations |

GN-L-BFGS reached the best optimum in every case, 1.7–12× faster; LM fell into a worse minimum in 2 of
9 seeded fits.  The chord's crawl at 20 beads on MPS was with the solver before 41bb693; with the fixes
it converges, a little slower than GN-L-BFGS.

*Tangents per batch on x86.* The merged default (up to 16) made a Jacobian build 20 % slower on these
cores (6.4–7.2 s against 5.7 s before the merge, alternating runs).  Best of 3: 2 tangents 5.0 s, 4:
5.6 s, 8: 5.9 s, 16: 6.5 s (6 beads); flat at 20 beads.  `TANGENT_MAX` is 2 on x86 CPUs, 16 elsewhere;
with it the merged code takes 4.7 s.

*z sampling* (`benchmarks/uipsf/synthetic/z_spacing.py`: dim beads acquired at 10 nm over ±1 µm, fitted thinned
by `Batch.thinned`, 4 held-out beads localised with every slice):

| Max spacing | Slices | Time | Phase error (CRLB), rad | z spread (CRLB), nm | Held-out deviance |
|---|---|---|---|---|---|
| 10 nm | 201 | 143 s | 0.0047 (0.0044) | 2.3 (2.7) | 1.0019 |
| 20 nm | 101 | 67 s | 0.0062 (0.0062) | 5.4 (3.8) | 1.0022 |
| 50 nm | 41 | 24 s | 0.0077 (0.0097) | 4.0 (6.0) | 1.0025 |
| 100 nm | 21 | 29 s | 0.0101 (0.0135) | 6.3 (8.3) | 1.0030 |
| 200 nm | 11 | 7 s | 0.0168 (0.0185) | 9.0 (11.5) | 1.0039 |

Thinning loses precision as the CRLB says, for the photons left out, and adds no bias: errors stay at
their CRLB and the held-out beads localise within theirs at every spacing.  Time falls with the slices.
The default is not decided yet (plan §9 M7).

**Defaults since M7 (decided October 2026).** `fit_beads` and `learn_beads` use `GaussNewtonLBFGS` and
thin the slices to at most 50 nm apart (`max_dz_nm=50`; None keeps every slice); LM stays available as
``solver=LevenbergMarquardt()`` and is still what localises single beads in `scoring`.  The blur σ is
reported as a magnitude (`params.Magnitude`): the model sees only σ², and a warm and a cold fit gave
±40.8 nm at the same objective.  `--slow -n auto`: 122 passed, 4 skipped.

**After the merge: where a fit's time goes, and candidate C.**  At n_max 8 (87 shared parameters, 4
x86 cores) one Jacobian build costs 8.4 s for 6 beads and 30 s for 20, of which the pupil's columns
are 76–81 % and the beads' 19 %; a gradient costs 0.19 / 0.79 s.  In the Zenodo fit about 75 % of the
time is its 6 builds.
- *A smaller pupil grid* would cut every cost by ~(P/64)², but 56 saves only 14 %, and the model is not
  converged in P anyway: against 128 samples, 64 differs by χ² ≈ 60 per bright bead stack (50 000
  photons per slice) and 80 and 96 do no better (41, 52), so the error is the pixelated aperture edge,
  not the sampling density.  For M8, not M7.
- *Matrix-free Gauss–Newton* (`solve/cg.py`, `GaussNewtonCG`; `Problem.gauss_newton_operator`: H·v
  for 0.8 of a gradient after a 1.8 s `torch.func.linearize` trace, or 2.3 untraced): each step solves
  for the current H by CG, preconditioned by the last built one.  The preconditioner is so good that a
  step needs about one product, so each step costs ~3 gradients against ~1 for GN-L-BFGS: dim beads 41 s
  untraced, 62 s traced, against 33 s, same optimum.  Kept, tested and not the default: it is the start
  for models whose shared block is too large to build, which will need a preconditioner that does not
  need H either.

**The emitter block linearised, a looser stage 1, and localising first.**
- *Emitter columns* (`ForwardModel.emitter_linearisation`): a bead's position enters the pupil only as a
  factor (−2πi k_x, −2πi k_y through the DFT, 2πi k_z through the defocus wave), so ∂I/∂(x, y, z) are
  three tangents of `pupil_linearisation`; photons, bleaching and background act through μ = N·I + b,
  JVPs of the small components.  Exact against the JVPs for every intensity and background model
  (`test_solve.py`).  Emitter block, n_max 8: 2.2 → 0.52 s (6 beads), 8.5 → 1.6 s (20); a build at 20
  beads 29.9 → 25.1 s; on Zenodo's 19 beads with the bead kernel 23.7 → 19.8 s.
- *Stage 1 stops at 1 nat or 0.1 standard error* (`workflows.beads.STAGE1_TOL`): it is a start for
  stage 2.  On Zenodo the end of stage 1 had spent ~26 s confirming an unchanged objective.  Zenodo
  111 → 96 s at the same optimum (+0.2 nats); dim seeds 0–2 25–33 s, same optima; 20 beads 107 s.
- *Localising every bead on the starting pupil before stage 1* (`localise_first`, off): fewer
  iterations but no faster on dim beads (29–33 s against 28–29 s), 11 % on Zenodo.  Kept as an option.
- *Very dim, seed 1* now ends in the worse of two minima (54125.7, phase error 0.080 rad, against
  54114.9 and 0.023 before the MPS merge).  Bisected to the merge: the faster DFT's float32 rounding
  alone picks the basin; stage 1's tolerance does not.  For the local-minima item (plan §10).

**The full calibration, and scoring without a fit.**  The 84-bead Zenodo calibration (`learn_beads`
defaults, `zenodo_1ch_full.py`, rows in `benchmarks/uipsf/results/zenodo_1ch_full_m7.jsonl`) took 64 min, of
which 9 were the two fitting stages; ~55 min were bead scoring, twice: every bead localised again on the
fixed pupil (829 s per scoring; one shared damping kept all 84 iterating to max_iter) and every slice
localised alone for the z bias (419 s).
- `BlockLM` (`solve/blocks.py`): LM with one damping, acceptance and stop per emitter, for problems whose
  shared parameters are fixed; the same localisations to 1 % of a standard error; 829 → 180 s, 419 → 86 s.
- `SplineFrames` (`workflows/spline.py`): the slice fits with SMAPpy's cubic-spline fitter on a voxel PSF
  sampled from the whole model; agrees with the pupil-model fits to 0.5 CRLB; optional, needs SMAPpy.
- Then decided (October 2026): **scoring takes no fit.**  A bead in a fit is scored from it (deviance,
  photons, xy offset, saturation); the planted 300 nm z jump shows in its deviance (1.42 against
  0.99–1.08).  The z bias is gone (on Zenodo it rejected 2 of 27 beads on its own); per-slice
  consistency is SMAPpy's to show.  Excluded and held-out beads are not scored; no beads are held out by
  default; the held-out report, which localises, is a diagnostic.  `BlockLM`, `slice_fits` and
  `SplineFrames` stay for it and for later tests of fit-based rejection (plan §10).

**On the Mac after the cloud's M7 work** (M1 Pro, 32 GB, torch 2.14.1; merged at 003ac89).  Tests:
`--slow -n auto` 132 passed (SMAPpy installed; without it the three SMAPpy tests skip), `test_gpu.py
--gpu` 2 passed.  Fixed for MPS, no test changed: the residual atlas, bead scoring and `SplineFrames`
cast device tensors to float64 for NumPy (MPS has none), and `zenodo_1ch_full.py` ran only on the CPU
(now `--device`).

One `normal_equations` at n_max 8 (87 shared parameters; `profile.jsonl`):

| | MPS, 6 beads | MPS, 20 | CPU, 6 | CPU, 20 |
|---|---|---|---|---|
| Emitter block, JVPs | 0.104 s | 0.381 s | 0.207 s | 0.781 s |
| Emitter block, linearised | **0.034 s** | **0.097 s** | 0.081 s | 0.248 s |
| Pupil block, linearised | 0.332 s | 1.038 s | 1.169 s | 3.638 s |
| Whole build | 0.40 s | 1.16 s | 1.26 s | 3.95 s |

The synthetic matrix (`solvers.py dim "very dim" --seeds 0,1,2 --solvers lm,gnl`, 6 beads), on both of
the Mac's devices:

| Scenario | Seed | LM, MPS / CPU | GN-L-BFGS, MPS / CPU | Optimum |
|---|---|---|---|---|
| dim | 0, 1 | 3.4–4.7 s / 8.7–9.1 s | 3.1 s / 5.2–6.1 s | the same on both devices |
| dim | 2 | 18.0 s / 47.4 s | 2.9 s / 5.2 s | LM: the +100 nats minimum on both |
| very dim | 0, 2 | 4.3 s / 10.7–11.0 s | 2.5–3.4 s / 4.3–4.9 s | the same on both devices |
| very dim | 1 | 8.8 s / 23.1 s | 13.4 s / 16.5 s | GN-L-BFGS: the worse basin on both, 54124.5 (MPS), 54125.1 (CPU) |

GN-L-BFGS reaches the CPU's optimum on MPS to 0.01 nats in five cases of six; in the sixth, very dim
seed 1, both devices end in the worse basin noted above, 0.6 nats apart within it (the cloud: 54125.7).

*The full Zenodo calibration* (`zenodo_1ch_full.py --no-probes`, 84 beads, `learn_beads` defaults;
`zenodo_1ch_full_mps.jsonl`):

| | MPS | CPU, 8 cores |
|---|---|---|
| Wall time, reading the stacks included | **45 s** | 113 s |
| Stage 1 / stage 2 | 29 s (35 iterations) / 13 s (15) | 52 s (29) / 58 s (25) |
| Beads used | 59 | 60 (bead 30 kept) |

The two agree on the Zernikes to 1.2·10⁻⁴ rad (half a CRLB), on σ to 0.003 nm, atlas deviance 1.377 and
1.371; bead 30 lies on the rejection threshold, where float32 rounding decides.  So the plan's target, a
bead calibration in about a minute on a laptop GPU, is met for 84 beads at n_max 8.

**The cloud's 4 x86 cores, after the merge** (`zenodo_1ch_full_cpu.jsonl`): the same calibration in
7.5 min (stage 1 starts after 42 s of reading, stage 2 after 57 s), 59 beads used and 25 rejected as on
MPS, atlas deviance 1.377, σ (66.96, 74.81) nm.  `--slow -n auto`: 132 passed, 2 skipped.

### M7 in one table: the full Zenodo `1ch_40nm_bead` calibration (84 beads, n_max 8, rejection included)

| | Time |
|---|---|
| uiPSF v1 (17 beads, its own finder; 4 cloud cores) | 7.6 min |
| v2 at the end of M3 (4 cloud cores) | 90 min |
| v2 during M7, before scoring took no fit (4 cloud cores) | 64 min |
| v2 now, 4 cloud x86 cores | 7.5 min |
| v2 now, M1 Pro CPU (8 cores) | 113 s |
| v2 now, M1 Pro GPU (MPS) | **45 s** |


## The model file (`export/model_file.py`, plan §5.8)

- `save_model(cal, path)` writes one HDF5 file, `format = "uipsf-model"`, `version = 1`:
  - `/model`: the components as JSON (`describe`; `build` makes the same `ForwardModel` again), the
    optics, every shared parameter in physical units (`/model/params/<name>`) and its CRLB
    (`/model/crlb/<name>`); the Noll indices are in the description, in the coefficients' order;
  - `/fit`: objective, stop reason, iterations, condition, history, the shared covariance and its
    layout, the used beads' own parameters and CRLBs;
  - `/beads`: the bead table, its thresholds, the rejection settings, the held-out beads;
  - `/diagnostics`: the residual atlas and the held-out report.
  No data: the stacks are the user's files, and the file stays small.
- `load_model(path)` returns a `LearnedPSF` (model, shared values, CRLBs and the record), not a
  `Calibration`: what SMAPpy needs is the PSF, and a `Calibration` would need the stacks back.
  `LearnedPSF.zernikes()` lists (Noll, coefficient, CRLB).
- `LearnedPSF.psf(xyz_nm, roi_px, depth_nm, include_bead, upsample)` and `Calibration.psf` sample the
  learned PSF, [N, Y, X] photons per pixel for one photon: the learned blur included; the bead's
  size left out unless asked (a point emitter has none); the stage's z scale not applied (z is the
  emitter's height above focus). Pixels are integrated over `upsample`² sub-samples, the model's by
  default; a spline should be sampled with more. Tests: `tests/uipsf_tests/test_model_file.py` (round trip, the
  same model from the description for every component, the sampled PSF is the model's, one photon
  over a wide region, x along columns, sub-sampling converges, the bead only on request).

## What a free pupil magnitude absorbs (`benchmarks/uipsf/synthetic/magnitude.py`, plan §10, 2)

Each candidate error is the truth; the fit is always uipsf's default model (NA 1.43, apodisation as in
`optics/grid.py`, beads in water, n_med 1.335), once with the magnitude free and once flat. 12 beads,
31 slices over ±750 nm, 5000 photons per slice, background 20, blur 35 nm. Zernikes to n = 6, three
seeds (ranges below), and one seed with n = 8 (`benchmarks/uipsf/results/magnitude.jsonl`). *Gain* is what the
free magnitude gains in likelihood over the flat one; by chance it exceeds 39 nats (n = 6, 24 magnitude
modes) or 52 (n = 8) with p = 10⁻⁶. *Radial* is the share of the magnitude's power in m = 0 modes.
*Phase bias* is the largest |fit − truth| / CRLB over the phase modes.

| truth | gain (nats) | radial | largest modes | phase bias, free / flat | σ (nm), free / flat | n = 8: gain, bias free / flat |
|---|---|---|---|---|---|---|
| matched | 9–17 | 0.02–0.27 | — | 2–3 / 2–3 | (35, 34) / (35, 34) | 22, 3 / 3 |
| NA 1.40 | 1160–1220 | 0.98 | Z4, Z11, Z22 < 0 | 14–16 / 15–16 | (37, 37) / (31, 34) | 1211, 10 / 39 |
| soft edge, outer 10 % | 3510–3610 | 1.00 | Z4, Z11, Z22 < 0 | 2–4 / 31–34 | (35, 35) / (26, 33) | 2530, 2 / 102 |
| apodisation × √cos θ | 3200–3370 | 0.99 | Z4 < 0 | 2 / 28–36 | (34, 35) / (25, 33) | 2778, 2 / 87 |
| dichroic, \|t_y\|² = 0.8 | 16–19 | 0.25–0.43 | — | 3–5 / 3–4 | **(43, 24)** / (43, 24) | 27, 4 / 4 |
| phase modes above n_max | 155–190 | 0.05–0.11 | Z24, Z8, Z12 | 15–18 / 11–16 | (38, 35) / (38, 35) | 113, 13 / 12 |
| astigmatism ±0.15 rad over the field | 9–22 | 0.20–0.46 | — | 2 / 2 | (36, 37) / (36, 38) | 33, 2 / 2 |
| n_med 1.40 (mounting medium) | 187–207 | 0.88–0.90 | Z4, Z22, Z6 | 17–19 / 25–29 | **(24, 15)** / (20, 16) | 227, 13 / 26 |
| n_med 1.00 (dried beads) | 1080–1130 | 0.85–0.87 | Z11, Z22, Z12 | 46–49 / 56–60 | **(83, 83)** / (84, 82) | 1067, 35 / 59 |

What it says (the same for every seed, and at n = 8):

- **Amplitude errors make a radial magnitude, and a free one is then right to be there.** A lower NA,
  a soft edge or a different aplanatic factor give an almost purely rotationally symmetric magnitude
  falling to the edge (Z4, Z11, Z22 negative), thousands of nats better than flat. Freed, the magnitude
  takes the error and the phase stays unbiased; kept flat, the phase's spherical modes are biased by
  30 CRLBs at n = 6 and by about 100 at n = 8 (more modes, more to absorb with), and σ shrinks to
  narrow the spot. The NA is the exception: it also changes the defocus (k_z), which no magnitude
  repairs, and the phase stays biased (less at n = 8).
- **The only strongly asymmetric magnitude comes from phase the model lacks.** Modes above n_max give
  a magnitude with only 5–13 % radial power, 110–190 nats over flat, and the phase biased by 13–18
  CRLBs either way. A strong asymmetric magnitude is first a sign to raise n_max, not a property of
  the optics.
- **A wrong sample medium is the costliest error, and σ takes much of it.** n_med enters the Fresnel
  transmission and the apodisation above NA = n_med, where most of the light of a high-NA objective
  is. Fitted as water, dried beads (n = 1.00) bias the phase by ~50 CRLBs whatever the magnitude does,
  raise σ from 35 to 83 nm and leave a deviance far above the matched fit (≈ 59 000 against 54 000
  nats). Beads in a mounting medium (1.40) narrow σ to (24, 15) nm. A radial magnitude appears in both,
  but repairs little. So the medium must be right before the magnitude or σ can be read.
- **A dichroic does not show in the magnitude; it shows in σ.** A scalar magnitude multiplies Ex and
  Ey alike; the free magnitude gains nothing. The fit widens the blur along x and narrows it along y
  instead: (43, 24) nm from (35, 35). σx ≠ σy is a candidate signature of t_s ≠ t_p (or of n_med), worth
  a probe before reading it as motion or pixel crosstalk.
- **Field-dependent aberrations show in neither:** the gain stays under threshold and the mean phase
  unbiased; the cost is in the deviance (≈ 2000 nats over matched), where the atlas's field group sees it.

**For reading a real fit:** a radial magnitude falling to the edge is expected (NA, edge, apodisation)
and should be kept, or replaced by a physical edge model; an asymmetric one points at n_max; a large
or anisotropic σ points at n_med or polarisation before anything else. Next: t_s/t_p and n_med as
likelihood-ratio probes (`diagnostics/probes.py`), then on Zenodo.

## How finely SMAPpy's spline must sample a uipsf model (`benchmarks/uipsf/synthetic/spline_step.py`, plan §6)

The truth is a uipsf model (astigmatic, 0.8 rad Noll 6, with coma and spherical; 110 nm pixels, σ 30
nm) at 400 random positions (±½ px, z ±600 nm). Splines are made as SMAPpy makes them: knots at pixel
centres over 27 px, each the PSF integrated over its pixel (8 × 8 sub-samples), z steps of 10–100 nm,
normalised to a maximum plane sum of 1. SMAPpy's Spline 3D fits the noise-free images (the spline's
systematic error alone) and Poisson ones (`benchmarks/uipsf/results/spline_step.jsonl`).

| z step | bias rms x, y, z (nm) | bias max x, y, z (nm) | bias / CRLB rms at 10⁴ photons | noisy RMSE / CRLB |
|---|---|---|---|---|
| 10 nm | 1.0, 1.1, 0.4 | 4.4, 3.5, 1.5 | 0.10, 0.19, 0.05 | 1.00, 1.03, 0.99 |
| 50 nm | the same | the same | the same | the same |
| 100 nm | 1.0, 1.1, 0.4 | 4.4, 3.5, 1.5 | 0.10, 0.19, 0.05 | 1.00, 1.03, 0.99 |

- **The z step does not matter from 10 to 100 nm:** the PSF changes slowly along z, and a cubic
  interpolates it far below the CRLB. The stacks' own step will do; a finer one only makes the file
  larger.
- **The lateral knots set the error.** Knots at pixel centres hold the right quantity, the PSF
  convolved with the pixel, as a function of the emitter's position; between them the spline
  interpolates it, at one pixel's spacing. The error pulls positions towards pixel centres, up to 4.4 nm
  at the pixel edges, and is largest where the PSF is narrowest (x below focus, y above: the astigmatic
  focal lines). At 10⁴ photons it is 0.1–0.2 of the CRLB (rms); at 10³, 0.02–0.03. Photons are right to
  0.05 %. A finer lateral grid would need a change to SMAPpy's spline format and fitter.

## Every fit on the fastest device (October 2026)

The model is built on `default_device()` (CUDA, then MPS, then the CPU; `UIPSF_DEVICE` overrides), but
readers cut the stacks on the CPU and the solver works where the batch is: a `learn_beads` on
`extract_rois`' batch ran on the CPU unless the caller moved it (`zenodo_1ch_full.py --device` did).
Now `fit_beads`, `learn_beads`, scoring, the probes, the atlas and the held-out report move the batch to
the model's device first (`device.on_model_device`); `tests/uipsf_tests/test_gpu.py` checks it with `--gpu`.
Held-out beads still failed on MPS, found by the model-choice study (October 2026): the held-out report met
the CPU's deviance with the device's mask, and the slice fits gave their frames a z on the CPU.  Both fixed;
`test_held_out_beads_with_masked_pixels_are_reported_on_the_gpu`.

## Milestone M4: several channels

```
src/uipsf/
├── model/channels.py  Channel(psf, transform=None, share=None, dz_nm=None, name=None, fit_transform=True)
│                      prefix(c), view(p, c), unpack_transform, map_points, inverse_map (adjugate, no inverse)
├── model/forward.py   ForwardModel(psf | [Channel, …], geometry, intensity, background) -> μ [E, C, K, Y, X]
│                      .positions(p, batch, xyz, c), .share(p, c), .psf_names(); the linearisations per channel
├── model/background.py emitter.background [E, C] with several channels; Planar per channel
├── model/start.py     rough_start: photons summed, shares and biplane offset from the data
├── model/psf.py       VectorPSF(analyser="unpolarised" | "x" | "y"), unpolarised by default (review F1)
├── data/dataset.py    Batch.roi_xy [E, C, 2]: each channel's ROI centre in chip pixels
├── data/pairing.py    pair_beads(stacks, roi_size, layout="auto") -> Pairs (SMAPpy's convention);
│                      detect_layout, fit_projective, extract_pair_rois(stacks, pairs, roi) -> Batch [E, 2, …]
├── export/model_file  version 2: a "channels" list; reads version 1.  sample_psf(…, channel=c);
│                      LearnedPSF.psf(…, channel), .shares(), .transform(c), .dz_nm(c), .zernikes(which, channel)
└── workflows          learn_beads / fit_beads / scoring / atlas: a bead is a pair, every channel staged and scored
benchmarks/uipsf/real/zenodo_2ch_beads.py     Zenodo 2ch_40nm_bead against SMAPpy's dual calibration and v1
benchmarks/uipsf/synthetic/channels_speed.py  two channels against one
```

**The model.**  μ_c = f_c·N·I_c(xyz_c) + b_c.  Each channel has its own PSF (optics, pupil, blur, analyser);
the emitter (x, y, z, photons, bleaching) is shared; each channel has its own background per emitter.
Channel c ≥ 1 adds a projective transform to the main channel's chip pixels (SMAPpy's
`DualColorCalibration.transformation`, 8 parameters with H₂₂ = 1), a photon share f_c (Σ f_c = 1, so N is
the emitter's total, SMAPpy's convention) and a defocus offset δz_c.  The emitter's position in channel c
is H⁻¹ of its main-channel chip position, exactly.  Conventions in `docs/uipsf/conventions.md`, "Channels".
Emitter independence holds per emitter across channels (`test_channels.py`).

**The speed paths per channel** (M7).  The pupil columns of channel c come from that channel's
linearisation and fill only its pixels; the emitter columns take three PSF tangents per channel.  New:
shared parameters that act only through where the emitter is and how bright (the transform, share and
offset; also a per-slice factor) take their columns from the emitter linearisation the emitter block
builds anyway, JVPs of the small components instead of the whole model.  With a bead kernel a slice
depends on its neighbours' positions, so only columns that move every slice alike qualify there (the
drift and z scale fall back to JVPs, checked per column).  All equal the JVPs' normal equations to 10⁻¹⁰
in float64, mirrored, with a bead kernel and drift, and planar background (`test_channels.py`).

**Pairing** (`data/pairing.py`, since replaced by SMAPpy's own through `smappy.calibrate.pupil.dual_rois`) followed SMAPpy's `collect_dual_beads`: the layout by the vote over every
within-stack bead pair (DoG-scored histograms of differences and of sums for a mirror), the cut at the
middle, beads per half by `find_beads`, the offset by a histogram vote after the mirror, mutual nearest
neighbours within 12 px, then a projective fit to the centroids as the start.  On Zenodo `2ch_40nm_bead`
it finds SMAPpy's layout (up–down mirrored), split (256) and the same 84 pairs, each within 0.18 px of
SMAPpy's position (`test_pairing.py`, with SMAPpy and the data).

**Synthetic recovery** (`test_multichannel.py`, slow; 12 beads, 21 slices over ±1 µm, 17 × 17 px,
20 000 photons): biplane (350 nm, half the light each) at n_max 4 and a mirrored ratiometric pair (30 nm
redder, 40 nm focal shift, 30 % of the light) at n_max 6 come back within 3.3 CRLB for every parameter
(the transform's 8, share, offset, both pupils' phase and magnitude, both blurs; the largest of ~60
values), atlas quiet, 14–19 s each on 4 cores.  The transform starts from a projective fit to the
centroids (0.37 px off).  A start 2.2 px off by a translation, or 1.4 px at the field's edge by a 0.4°
rotation, reaches the same optimum; a translation alone (rotation, scale and perspective left out, ~3 px
at the edge) stopped in a wrong minimum, the secondary pupil bent to take the misplacement.  A better
start of the transform is in plan §10 ("To investigate", 5).

**Zenodo `2ch_40nm_bead`** (`zenodo_2ch_beads.py`; rows in `benchmarks/uipsf/results/zenodo_2ch_beads.jsonl`):
14 stacks, SMAPpy's camera (4.7 e⁻/ADU, offset 398.6, 21 e⁻), 116 × 127 nm, v1's optics (680 nm in both
channels, n_med 1.35), 20 nm bead, n_max 8, 25 × 25 px, `learn_beads` defaults; 28 min on 4 cloud cores.
- **Beads:** 84 pairs, 70 used, 14 rejected.  Atlas deviance 1.36 in the main channel (the 1ch data: 1.38)
  and 1.89 in the secondary, which has five times the light; the atlas flags the spot's shape, z, field
  and photons per slice as on 1ch, and the channel group (the secondary fits worse).
- **The transform** agrees with SMAPpy's dual calibration of the same stacks to 0.025 px RMS (0.032 max)
  where the used beads are, and with v1's (`res/T`, converted to chip pixels) to 0.025 px (0.048 max).  On
  SMAPpy's own bead positions the three leave residuals of 0.036 (uipsf), 0.027 (SMAPpy, its own fit to
  those points) and 0.037 px (v1).
- **v1's frame**, found from its 74 ROI centres (all on beads uipsf detects): arrays transposed, and v1's
  channel 0, its reference, is the camera's lower half with y reversed: v1's reference is SMAPpy's
  secondary.  Zernikes compared through that frame (a signed permutation found on the pupil grid):
  secondary channel correlation 0.9998 with v1's, RMS difference 0.0017 rad over Noll 5–45; main channel
  0.997 and 0.007 rad, most of it spherical (Noll 11: −0.023 against v1's −0.003), the trade with v1's
  fitted defocus seen on the 1ch data too.
- **Photons:** the shares are 0.164 / 0.836 (± 0.0001); SMAPpy's dual calibration of the same stacks
  stores `photon_normalization` 0.160 / 0.840, the same split to 0.004.  On the data, the light
  summed over the stack in SMAPpy's 37-px box after its background gives a ratio of 5.20 (quartiles
  4.9–5.5 over beads), uipsf 5.10.  SMAPpy's *brightness ratio* (4.2), the median of each pair's
  brightest planes, is a different statistic and not what its splines are normalised by.  Inside the
  fitted ROI the model's light equals the data's to 0.1 % in each channel; beyond it the model's wings
  are heavier than the data's by ~2 % of each channel's light, so the ratio is unaffected.  v1's
  intensity ratio (2.3) is not a photon split: v1 normalises each channel's model in its own ROI.
  The background is fitted per bead and channel (μ = f·N·I + b), as SMAPpy subtracts one per channel.
- **Defocus offset** 35.5 ± 0.15 nm (a chromatic focal shift); blur (75.6, 61.0) and (57.9, 75.7) nm.

**Into SMAPpy's global fitter** (`tests/uipsf_tests/test_smappy_dual.py`, with SMAPpy): a two-channel model sampled
per channel on one z grid (`sample_psf(…, channel=c)`), normalised by SMAPpy's `positive_pair_models`
(its `photon_normalization` comes back as the share, 0.30), with the model's transform and the split's
geometry (`Pairs.geometry`, stored in the model file: `LearnedPSF.geometry`), saved and reloaded as a
`DualColorCalibration`.  Split frames drawn from uipsf's forward model (mirrored layout, 0.5° turn,
ratiometric, 20 000 photons) fitted by `DualChannelEngine`: x and y within 0.005 px, z within 3 nm,
ratio 0.295–0.298; photons are the light in the spline's box around focus (82 %), as with SMAPpy's own
bead calibrations.  On Zenodo the stored geometry equals SMAPpy's field for field (`test_pairing.py`).

**Cost** (`channels_speed.py`, 6 dim beads, 41 slices of 21², n_max 6, 4 cloud cores;
`benchmarks/uipsf/results/channels_speed.jsonl`): one Jacobian build 2.9 s for one channel, 6.5 s for two (116
shared parameters against 53); a gradient 0.18 / 0.32 s; the default fit 10 / 32 s (15 / 22 iterations).
So a channel costs what a single-channel model costs.

## The calibration's entry point (`workflows/calibrate.py`, `diagnostics/figures.py`, plan §7.5)

What SMAPpy's *PSF calibration* window will call; step 1 of §7.5's order of work.

- **`BeadSettings`**: the choices shown in the form, as SMAPpy `param` fields (`smappy.plugins` imports no
  Qt), so the window builds its form as it does a plugin's: *Zernike order* (auto: 6), *pupil magnitude*
  (free; symmetric, the m = 0 modes, Noll 4, 11, 22 …; fixed, the apodisation only), *blur* (free or
  fixed) and *blur σ* (its start, or its value), *bead diameter*, and under *more* the rejection's k
  and the held-out fraction.  Choices, not flags, so an option added while plan §10, 2–3 is
  investigated, or taken out once it is settled, is one line.  The instrument is not a setting: it
  arrives as `Optics`.
- **`bead_model(optics, settings, roi_shape, transforms, z_max_nm)`**: one channel per `Optics`, the pupil
  sampled by `choose_pupil_size` for the stacks' z range.  A fixed blur is `VectorPSF(fit_sigma=False)`,
  a fixed `psf.sigma` spec; the model file records it (`fit_sigma`, read as True from older files).
- **`calibrate_beads(batch, optics, settings, transforms, geometry, start, …)`** builds that model and
  runs `learn_beads`; the `Calibration` keeps the settings (`options`), and the model file writes them
  (`/model` attribute `options`, `LearnedPSF.options`).
- **A start from a previous model** (`start=`, a `LearnedPSF` or `Calibration`): `start_values` places its
  coefficients by Noll index, so a model of another order or magnitude choice starts the new one; a free
  blur, the channel transforms, shares and offsets carry over, a fixed blur keeps the settings' value.
  `calibrate_beads` takes it into stage 1 (`learn_beads(start_stage1=True)`, since October 2026: "How a
  bead fit starts" below); `learn_beads(start=)` alone still replaces stage 1, every bead localised on
  that pupil (`scoring.localise`), which is what a richer model started from a simpler one's solution
  wants.  On simulated beads (10 beads, 21 slices, n_max 4 → 6) replacing stage 1 reached the truth to
  0.002 rad in 1.9 s against 3.1 s from the data, on the CPU.
- **A pair's transform** starts from the beads' Gaussian positions (`gaussian_transforms`,
  `model.astig.gaussian_start`) when at least `GAUSS_MIN_PAIRS` beads give one in both channels and they
  fit it to `GAUSS_MAX_RMS_PX`, else from the pairing's; `Calibration.transform_start` says which.
- **Figures** (`diagnostics.figures`): `draw_pupil`, `draw_zernikes` (with CRLBs, beside a reference
  model mode by mode), `draw_atlas`, `draw_beads`, each drawing on a figure it is given; `figures(cal,
  reference)` gives them as {title: draw}, for tabs that draw lazily.  They read a `Calibration` or a
  `LearnedPSF` alike through `zernikes` and `pupil_maps`.  The phase is drawn in rad, waves or nm
  (`PHASE_UNITS`, `phase_scale`), and with a `HalfSplits` the half-split uncertainty beside the CRLB.
- **Half-split uncertainty** (`diagnostics.halves`): `half_splits` learns the calibration again on K
  complementary random halves of its used beads, each cold from the run's own start with every bead of
  the half pinned in by a decision, and estimates the full fit's standard deviation as the root mean
  square of (A − B)/2 (`half_spread`); at least `MIN_BEADS` = 8 used beads.  It sees bead-to-bead
  differences and an optimiser that does not land in one place, which the CRLB cannot; on simulated
  beads with an exact model it agrees with the CRLB (median ratio 0.8–0.9 at 3–5 splits).  Kept on the
  `Calibration` (``halves``) and in the model file (``/model/halves``).
- Tests: `tests/uipsf_tests/test_calibrate.py`.
- **smappy makes the batch** (`smappy.calibrate.pupil`, step 2 of §7.5): `read_bead_stacks`, the camera as
  a fit resolves it (`camera_for`), `camera.to_photons`, the read noise and EM excess from `CameraMetadata`
  (`BeadROIs.noise`), beads by `detect_beads`, pairs, layout and the start transform by
  `collect_dual_beads` and `fit_dual_transform`.  Raw ROIs around the detection's whole pixel, saturated
  pixels masked, the first slice dropped, stacks cut to the shortest about their middles; `rois_at` cuts
  around centres given by hand.  uipsf's own reader, finder, pairing and `noise.Camera` are removed.  End to
  end, simulated ADU stacks read this way give back the aberrations to 0.02 rad
  (`tests/uipsf_tests/test_bead_rois.py`).
- **One *find* for both calibrations** (step 3): `core.find_stack_beads` (the beads of one stack and their
  records: whole pixel, border background, brightness, peak, saturation) and `core.classify_brightness`
  (*saturated*, *too dim*, *too bright*, *accepted*) are what `collect_beads` used inline; it calls them now,
  and so does `pupil.single_rois`, so both calibrations list the same beads with the same verdicts.  The
  bead average drops what is not *accepted*; the pupil model shows the verdict and judges by its fit.
- **The spline from the model** (`pupil.spline_calibration`, `dual_calibration`, `save`): `sample` gives
  the learnt PSF of a point emitter on the central pixel, each pixel as the model was learnt (M6, below), on
  `z_grid(dz, range)` (z = −(index − z0)·dz, plane 0 the highest emitter), the learnt blur kept and the
  bead's size left out.  One channel: clipped at zero, the brightest plane summing to one, no smoothing,
  as `_single_model`.  Two: `positive_pair_models` with smoothing off, the model's transform and split,
  so `photon_normalization` is each channel's learnt share.  `parameters` records `method = "uipsf2_pupil"`,
  every channel's Zernikes with CRLBs, the settings, the sampling and, once saved, `model_file`.  `save`
  writes `x_3dcal.h5` and the model file `x_3dcal.uipsf.h5` beside it; a loaded model file re-samples
  without learning.  On 12 simulated astigmatic beads (`tests/uipsf_tests/test_pupil_spline.py`, the
  pixels integrated 8 × 8 as a camera records them), Spline 3D with the learnt model's spline fits frames
  of the true PSF with z bias 0.1 nm and spread 11.1 nm (the bead average of the same stack: −4.2 and
  13.7 nm, the same way up), photons within 0.2 % of the bead average's.
- **Two channels on Apple's GPU** failed in the solver, not the data: on MPS a Python number meeting a
  0-d tensor under `vmap(jvp)` becomes a float64 tensor, which MPS refuses, and the photon share is a 0-d
  parameter.  `params.Bounded` and `Softplus` now make their constants tensors of the parameter's dtype
  (`test_device.py`).
- **Zenodo through smappy's adapter** (19 October 2026, a Mac's GPU, `--no-probes` for 1ch): the results
  stand as recorded before the port.  `1ch_40nm_bead`: 84 beads, 59 used and 25 rejected, 43 s with reading.
  `2ch_40nm_bead`: 84 pairs, up–down mirrored, split 256; 70 used, 14 rejected; share 0.836, defocus offset
  35.4 nm, blur (75.6, 61.0) and (57.9, 75.8) nm; the transform 0.025 px RMS from SMAPpy's; the Zernikes
  against v1's, correlation 0.997 (main) and 0.9998 (secondary).  1.4 min on MPS, where it had failed
  (above), against 28 min on 4 cloud cores.

## M6: the chain's acceptance (October 2026, plan §6.5 and §9 M6)

- **Simulated beads → uipsf → spline → Spline 3D** (`test_pupil_spline.py`): 12 astigmatic beads drawn into
  an ADU stack with their pixels integrated 8 × 8, cut by `single_rois`, learnt by `calibrate_beads`,
  sampled by `spline_calibration`; 1000 frames over ±600 nm at 5000 photons.  The fitted z against the true
  z: slope 0.9986 (three seeds 0.998–1.000), intercept −0.1 nm; the photons 0.2 % above the bead average's
  of the same stack.
- **Spline 3D and uipsf's localisation on the same ROIs**: every slice of every bead fitted by
  `scoring.slice_fits` (the pupil model, from the beads' stack fits) and by Spline 3D on the exported spline
  (4 px larger than the ROI, so both see the same pixels).  Over ±600 nm: z 0.07 nm apart in the median and
  2.8 nm rms, against a CRLB of 3.4 nm at 40 000 photons; the photons in a constant ratio (the spline counts
  the light its box holds) to 0.6 %.  On beads drawn with the model's own pixel centres the two agree to
  1.6 nm rms and 0.4 nm in x, with no spline step from 10 to 50 nm making a difference: what is left is the
  lateral interpolation of the knots, a pixel apart (above, "How finely").
- **The sampling was wrong, and this found it.**  The spline sampled each pixel integrated over 4 × 4.  But
  the learning samples pixel centres (`VectorPSF(upsample=1)`, v1's default), so on a camera's integrated
  pixels the learnt blur takes the pixel's width: σ 35 nm learnt as 47.3, and √(35² + 110²/12) = 47.4.
  Integrating that model again counted the pixel twice: z slope 0.992, photons 1.2 % high, and 6.1 nm rms
  from uipsf's localisation with a ±7 nm bend in z.  `pupil.UPSAMPLE` is now the model's own sampling;
  "How finely" above still holds for a model whose blur does not hold the pixel.  On Zenodo `1ch_40nm_bead`
  (56 beads, 52 used, a Mac's GPU, σ learnt as 68 and 75 nm), the same comparison: at the model's own
  sampling Spline 3D and `slice_fits` agree to 0.6 nm rms (CRLB 2.5) with slope 0.9999 and the photon ratio
  constant to 0.02 %; integrated 4 × 4, 5.0 nm rms, a ±7 nm bend and slope 0.989.  Against the stage both
  localisers find a slope of 1.011, uipsf's own too: that is the model or the stage, not the spline (the
  4 × 4 spline's 0.999 was two errors cancelling).  The fit quality's median |centred error| there is 6.5 nm.  Learning with the pixels
  integrated (upsample ≥ 2) would make the blur the optics' own, at the cost of 4× the pixels per fit: not
  done.
- **Dual colour: the fit quality is the global fit** (`pupil.fit_quality`): each slice of a pair fitted
  by `GlobalSplinePSF` with x, y and z shared (`LINK_XYZ`), the photons free per channel, the link built
  by `dualfit.build_link` from each ROI's centre and the learnt transform, as Spline 3D 2C builds it.
  Before, each channel was fitted alone with its own half of the calibration, which never tried the
  transform or whether the two splines agree on z.  On 8 simulated ratiometric pairs (mirrored, ±600 nm,
  `test_smappy_dual.py`): z −0.2 nm from the truth in the median, 6.6 nm rms after each bead's offset, the
  ratio 0.297 for a share of 0.3, flat in z.  A biplane pair (350 nm apart) gives a ratio of 0.471 for a
  learnt share of 0.501, also flat in z: `positive_pair_models` takes each channel's share as the light
  its box holds at the main channel's focus, where the defocused secondary holds less.  That is the dual
  calibration's convention, the bead average's too, and not changed here.  On Zenodo `2ch_40nm_bead` (88
  pairs, 71 used, learnt in 70 s on a Mac's GPU, fit quality 3 s): median |centred error| 4.5 nm, 6.5 nm rms
  over ±600 nm, the ratio 0.837 for a learnt share of 0.838.  The global z is 1.0 nm rms from the bright
  secondary channel's alone (84 % of the light); the dim main channel alone wandered by 32 nm rms, which
  the former per-channel page showed as a second row of curves the data never meet.

## Which model choices can be fixed (`benchmarks/uipsf/real/model_choices.py`, plan §7.5 and §10)

Every variant is one bead calibration of the same ROIs (SMAPpy's adapter, `pupil.bead_rois`, 25 px),
with one choice changed from **base**: the form's defaults (`BeadSettings()`: order 6, magnitude free,
blur free), the z scale fixed at 1, bleaching of order 1 per bead, stage 1 to order 4, slices thinned to
50 nm, 20 % held out (`rejection.held_out`, the same beads for every variant).  Base runs the rejection;
every other variant gets its bead states as decisions, so all fit the same beads.  On a Mac's GPU (M-series,
MPS, torch 2.14), October 2026; `benchmarks/uipsf/results/model_choices.jsonl`, and
`model_choices.py --summary` prints the full table.  Base reproduces to the nat between invocations.

- *Δ train*: the training objective (nats) against base, the same beads.  A nested model can only lower
  it at its optimum: a positive value for a model that contains base means the fit ended in a worse basin.
- *Δ held out*: the held-out beads localised on the learnt pupil with every slice (`heldout_report`), the
  deviance in nats over all their pixels.  It does not reward parameters, only a better pupil.
- *z bias*: the held-out report's, the pupil model's slice fits against its stack fit (RMS over the central
  60 % of the range, nm).  *Spline*: Spline 3D on the exported spline (`pupil.fit_quality`), per slice the
  median over beads of the centred z error; RMS over the central slices for the held-out beads / the used
  ones.  The spline leaves the bead's size out, so on 40 nm beads a narrower model scores better here
  than it should (on the simulation the true model scores 5–7 nm, a blur fixed at 0 2.5–4).
- Data: Zenodo `1ch_40nm_bead` (56 beads after the dim specks: 43 used, 2 rejected, 11 held out; ~70 000
  photons per slice) and `2ch_40nm_bead` (88 pairs: 57, 13, 18), NA 1.43, n_imm 1.516, beads in water
  1.33, 600 nm (1ch) and 680 nm (2ch) as v1 ran them, 40 nm beads, `zenodo_camera.py`.  *sim*: 36
  astigmatic beads in three stacks drawn by uipsf's model through the adapter (Zenodo's camera, pixels
  and 50 nm slices; phase to n = 6, a radial magnitude Noll 4 = −0.15, σ 30 nm; 20 000 photons per
  slice, background 40), two seeds; the phase is compared with the truth (0.001 rad RMS for base).

| variant | 1ch Δ train / Δ held out | 1ch z bias, spline held/used | 2ch Δ train / Δ held out | 2ch z bias, spline held/used | sim Δ held out (2 seeds), z bias | σ 1ch (nm) |
|---|---|---|---|---|---|---|
| base | 0 / 0 | 5.4, 3.6 / 3.1 | 0 / 0 | 3.3, 6.6 / 5.1 | 0, 0; 2.8, 3.2 | (66, 73) |
| order 8 | −35 300 / −10 400 | **3.4**, 3.5 / 2.6 | −132 500 / −68 900 | **2.6**, 7.1 / 4.9 | +5.5, +3.8; same | (65, 73) |
| magnitude symmetric | +46 400 / +16 200 | 5.2, 4.6 / 4.3 | +297 800 / +162 200 | 4.0, 7.1 / 5.7 | 0, −2; same | (64, 75) |
| magnitude fixed | +243 200 / +92 100 | **21.3**, 23.7 / 23.2 | +2.05 M / +1.10 M | **30.2**, 20.4 / 20.3 | +6200, +5400; 22 | (62, 78) |
| order 8, symmetric | −6 100 / −900 | 3.5, 3.6 / 2.7 | +83 600 / +53 700 | 3.5, 7.0 / 5.2 | +3, −1; same | (64, 75) |
| blur fixed 0 nm | +429 700 / +157 000 | 20.5, 13.1 / 13.6 | — | — | +420, +380; 5.0 | (0, 0) |
| blur fixed 20 nm | +345 500 / +125 900 | 18.7, 11.7 / 12.2 | +1.03 M / +659 300 | 16.3, 13.6 / 13.6 | +130, +120; 3.5 | (20, 20) |
| blur fixed 40 nm | +162 900 / +58 700 | 14.1, 7.9 / 8.4 | +465 400 / +332 000 | 12.5, 11.0 / 11.0 | +190, +160; 4.7–5.3 | (40, 40) |
| z scale free | −5 300 / −3 300, s = 0.962 | 6.3, 13.4 / 13.2 (slope −0.037) | **+623 500** / +246 600, s = 0.957 | 9.8, 9.1 / 9.0 | 0, 0; s = 1.000, 1.001 | (65, 72) |
| λ +10 / +20 / +30 nm | −400 / +500 / +2700; held −900 / −1300 / −1200 | 5.6–6.0, 3.5 / 3.0 | — | — | +32…+400; σ falls to 0–9 nm at +30 | (64…59, 71…68) |
| NA −0.007 / −0.013 / −0.020 | −1900 / −2700 / −2300; held −700 / −900 / −600 | 5.5–6.0, 3.2–3.3 / 3.0–3.6 | — | — | +13…+220 | (66, 73) |
| bleaching order 2 | −38 100 / −26 700 | **3.9**, 4.4 / 4.2 | −123 700 / −29 000 | **2.2**, 5.8 / 3.7 | −4, −6; same | (65, 72) |
| per-slice photon factor | −10 800 / **+6 600** | 4.3, 4.2 / 4.1 | **+27.0 M** / +8.7 M | **80.5**, 80.0 / 79.1 | +1, +8; same | (65, 72) |
| photons free per slice | fails: `score_beads` reads `emitter.photons` | | fails | | fails | |
| order 8, bleaching 2 | −68 100 / −38 600 | **2.8**, 4.0 / 3.4 | **+170 000** / +84 900 | 7.9, 5.8 / 5.8 | +2, −2; same | (64, 73) |
| no stage 1 | 0 / 0 | the same | 0 / −8 | the same | 0; same | |
| stage 1 to order 2 | 0 / 0 | the same | **+27.8 M** / +9.1 M | 81, 430 / 431 | 0; same | |
| `max_dz` 100 nm | (fewer pixels) / −7 | 5.3, 3.5 / 3.1 | / +340 | 3.0, 6.4 / 4.7 | +3, −5; same | |
| `max_dz` 200 nm | / +120 | 5.2, 3.5 / 3.2 | / +1550 | 2.8, 6.4 / 4.7 | +6, +3; same | |

Times (MPS, after the first fit of a process, which compiles): base 12–14 s for 1ch and 57 s for 2ch;
order 8 about the same; `max_dz` 100 / 200 nm 8 / 9 s and 36 / 22 s.

What it says:

- **Order 8 is better on both real datasets, by far**: 10 000 and 69 000 nats on held-out beads, the
  pupil's z bias 5.4 → 3.4 and 3.3 → 2.6 nm, at no cost in time.  On the simulation, whose truth stops at
  n = 6, the extra modes cost 4–6 nats on the held-out beads and nothing in z.  The default 6 leaves real
  aberrations out.
- **The magnitude must be free, and asymmetric.**  *Fixed* is the worst choice everywhere: the z bias goes
  to 21–30 nm (22 nm on the simulation, where the truth has a radial magnitude and the phase is then off by
  0.026 rad).  *Symmetric* loses 16 000 / 162 000 nats on held-out beads at order 6, and at order 8 still
  29 000 / 216 000 against the free magnitude: the Zenodo data want the non-symmetric modes, which the
  dictionary (above) ties to phase above n_max or to the field.  Freeing them moves the phase by 0.014 rad
  RMS (2ch).  On the simulation free and symmetric agree, as they should.
- **No fixed blur reproduces the data.**  σ fits to (65, 73) nm on 1ch and (73, 59) on 2ch's main channel,
  and fixing it at 0, 20 or 40 nm costs 59 000–157 000 nats (1ch) and 0.33–0.66 M (2ch) on held-out beads
  and raises the z bias to 12–20 nm.  On the simulation the free σ comes back at the true 30 nm.  Whatever
  the 70 nm stand for (plan §10, 3), the model needs them.
- **The z scale: neither the NA nor the wavelength explains it.**  Freed on 1ch it goes to 0.962, gaining
  5300 nats (3300 held out); then the spline's z is 3.7 % shorter than the stage's (the slope).  A lower
  NA recovers at most 2700 of the 5300 (at −0.013, with s fixed), a redder emission 400 (+10 nm) and less
  beyond; on the held-out beads both give ~900–1300.  So neither is the cause alone.  On 2ch the free z
  scale also goes to 0.957, but the fit ends 624 000 nats *worse* than base, a model that contains it:
  the solver did not find its optimum (plan §10, 4).  On the simulation (true s = 1) the free s stays at
  1.000 and gains nothing: no false alarm.
- **Bleaching of order 2 helps real beads and costs simulated ones nothing**: 27 000 / 29 000 nats held
  out, z bias 3.9 and 2.2 nm.  The per-slice photon factor overfits on 1ch (better in training, 6600
  nats worse held out) and wrecks 2ch (σ 33 nm, z bias 80 nm: a trap).  A free count per bead and slice
  (`PerObservation`) cannot run in `learn_beads` at all: the bead scores read `emitter.photons`.
- **Combinations are where 2ch breaks.**  Order 8 with bleaching 2 adds up on 1ch (−68 000 / −38 600, the
  best z bias, 2.8 nm) but on 2ch ends 170 000 nats worse than base, though each alone gains over
  120 000.  Three of the richer 2ch models (this one, the free z scale, the slice factor) landed in worse
  basins; every 1ch and simulated fit was consistent.
- **The first stage is not needed on these data, and order 2 is a trap.**  No stage 1, or stage 1 to
  order 4, give the same fit to the nat on all four datasets; to order 2 on 2ch it ends with z off by 430 nm.
- **Thinning to 100 nm costs nothing measurable**: on the held-out beads (judged with every slice) −7 and
  +340 nats of 0.23 M and 0.92 M, the z bias the same or lower, 1.6× faster on 2ch; 200 nm costs 120 and
  1550 nats, still the same z bias, 2.6× faster on 2ch.
- Not tested here: rejection's k, the held-out fraction, the bead diameter, upsampling, and n_med (the dictionary
  above covers it on simulations).

### The z scale and the magnitude, drift and the blur, and fits started from a simpler one

Three questions the table above left (October 2026, the same script, the same beads; the fits marked *from*
start at another variant's solution through `start_values`, past stage 1).

| variant | 1ch Δ train / Δ held out | 1ch s, σ (nm), z bias | 2ch Δ train / Δ held out | 2ch s, σ, z bias | sim (true s = 1): s, Δ held out |
|---|---|---|---|---|---|
| s free, magnitude fixed | +68 400 / +22 800 (−174 800 against fixed s) | **0.920**, (61, 72), 5.4 | +579 900 / +272 900 | **0.897**, (70, 64), 4.8 | **0.929**, +3400 |
| s free, magnitude symmetric | +44 000 / +13 800 | 0.968, (64, 75), 6.0 | +293 400 / +154 600 | 0.964, (72, 64), 4.5 | 1.000, 0 |
| s free, magnitude free, from base | −5 300 / −3 300 | 0.962, (65, 72), 6.3 | −43 100 / −27 400 | 0.948, (72, 57), 4.3 | 1.000, 0 |
| order 8, s free, from order 8 | −36 300 / −11 100 (−1 000 against order 8) | 0.977, (64, 73), 3.9 | −143 100 / −75 000 (−10 600) | 0.966, (72, 58), 2.7 | 0.999, +5 |
| order 8 + bleaching 2, from base | −68 100 / −38 600 | 1, (64, 73), 2.8 | **−243 500 / −91 600** | 1, (72, 59), **1.9** | 1, +2 |
| drift shared, from base | −17 900 / −4 800 | 1, **(62, 71)**, 2.9; drift (10, 15) nm | −99 200 / −53 300 | 1, **(71, 57)**, 1.9; drift (7, 17) | 1, +13; drift (1.4, 1.1) |
| drift per bead | −7 600 / −4 000 | 1, (65, 73), 5.3; drift (5.6, 4.5) | −50 300 / −24 800 | 1, (73, 59), 3.2; drift (8.7, 8.5) | 1, −197; drift (2.1, 2.1) |
| drift per bead, σ fixed 20 nm | +335 400 / +120 400 | 1, (20, 20), 18.6 | +966 000 / +627 200 | 1, (20, 20), 16.1 | 1, −71 |

*Drift shared*: uipsf's `geometry.drift`, one lateral and axial shift per slice for every bead.  *Drift per
bead*: `BeadDrift` in the script, a lateral shift per bead and slice with the same 5 nm prior, which bounds
what any drift between frames can do (one per file, the physical one, is 1440 shared parameters for 1ch,
more than MPS takes in one Jacobian; plan §10, 7).  Fresh, the shared drift ended 70 000 (1ch) and 428 000
(2ch) nats *worse* than base, as did 2ch's free s (+624 000) and order 8 with s free (+669 000); from the
simpler fit each gains.

- **A magnitude the model lacks makes a z scale.**  On simulated beads whose truth has s = 1 and a radial
  magnitude, a fit with the magnitude fixed asks for s = 0.929; with it free, 1.000.  On Zenodo s follows
  the magnitude the same way: 0.920 / 0.897 fixed, 0.968 / 0.964 symmetric, 0.962 / 0.948 free at order
  6, 0.977 / 0.966 free at order 8, where 1ch gains only 1000 nats from it.  So most of the z scale is the
  pupil's amplitude, not the piezo; what 2ch still asks for at order 8 is 10 600 nats.
- **Drift between frames is not the blur.**  A drift per bead and slice fits 5–9 nm and leaves σ where it
  was; with σ held at 20 nm the drift cannot take its place (16–19 nm z bias).  The shared drift, started
  from base, fits 10–17 nm per slice, takes σ down by 2–4 nm and gives the lowest z bias of any single
  change (2.9 and 1.9 nm): a real stage motion common to the beads of a frame, but not 70 nm of blur.  On
  the simulation (no drift) the drifts stay at 1–2 nm.  What is left is blur within the exposure or a
  mismatch of the model (plan §10, 3).
- **Richer models need the simpler one's fit as their start.**  Every fresh 2ch fit that ended worse than
  a model it contains reached the expected gain when started from that model's solution; on 1ch and the
  simulation the start changes nothing.  Order 8 with bleaching of order 2, from base, is the best fit
  measured on 2ch: −91 600 nats held out, z bias 1.9 nm.

## Is the blur the microscope's?  The lab's own beads, M2 and M5 (October 2026, plan §10, 2 and 3)

On Zenodo the model learns an extra blur σ ≈ (67, 75) nm and a free pupil magnitude; whether they belong
to the microscope or soak up model mismatch was open.  `benchmarks/uipsf/real/lab_beads.py` learns nine of
the lab's 2026 bead calibrations, five from M2 and four from M5 (no M6 stacks exist), through SMAPpy's
headless path: `pupil.run_calibration`, the camera as `camera_for` resolves it, the beads `single_rois`
finds, `BeadSettings(zernike_order=8, magnitude="free", blur="free")` with 100 nm beads; then
`pupil.fit_quality` and the z-scale probe on the used beads.  Optics from the user: M2 NA 1.43, oil 1.516,
116 × 127 nm; M5 NA 1.50, oil 1.518, 105.6 nm; TetraSpeck in water at 676 nm.  The stacks and their
metadata: `docs/uipsf/microscopes.md`.  Results: `benchmarks/uipsf/results/lab_beads.jsonl`; the figure:
`lab_beads_figure.py`.  Each run took 7–15 s on a Mac's GPU for M2 and 0.3–1.1 min for M5, whose pupil
(NA 1.50 under oil of 1.518) needs 128 samples where M2's needs 64 and only fits 10 GB with the solver
chunked one bead at a time (`--chunk 1`; the headless path does not chunk).

**The pixel first.**  M2's astigmatic path has 116 × 127 nm pixels, and the camera changed in July from the
Evolve to an iXon mounted 90° turned: with the same beads the likelihood picks 116 × 127 for the Evolve and
127 × 116 for the iXon, by 28 000 and 38 000 nats, and the square 127 nm SMAPpy's database gives moves σ by
21 nm along the wrong axis (`lab_beads_pixel.jsonl`).  The runs below use each camera's own orientation.

σ as learnt (± CRLB), with the pixel's share taken out (√(σ² − p²/12): uipsf learns at pixel centres, so the
blur holds the pixel's width; 0 where σ is below it), and the same with the z scale free; the magnitude's
RMS over the radial (m = 0) and the other modes, with the z scale fixed / free; phase in rad (CRLBs ≤ 0.001);
the atlas's mean deviance; the probe's z scale and its gain in nats (threshold 12); the fit quality's median
| RMS centred error (nm) over ±600 nm.

| | acquisition | camera | beads | σx, σy learnt (nm) | pixel out | z free | magnitude radial | other | Z(2,2) | Z(2,−2) | Z(4,0) | Z(6,0) | deviance | z scale (gain) | fit quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2 | 260622_AA_BeadsCalZ_638 | Evolve | 14/15 | 75.8 ± 0.1, 74.0 ± 0.1 | 68, 64 | 67, 63 | 0.18 / 0.12 | 0.03 / 0.03 | −0.67 | +0.02 | −0.08 | −0.06 | 2.05 | 0.958 (4325) | 3.9 / 7.2 |
| M2 | 260829_AA_BeadsCalZ_638_SingleMode | iXon | 11/11 | 67.2 ± 0.1, 53.5 ± 0.1 | 56, 42 | 56, 41 | 0.16 / 0.11 | 0.04 / 0.05 | +0.55 | −0.04 | +0.19 | +0.03 | 2.43 | 0.967 (3211) | 4.2 / 6.7 |
| M2 | 260904_AA_BeadsCalZ_638_SingleMode | iXon | 11/12 | 64.4 ± 0.1, 52.4 ± 0.1 | 53, 40 | 52, 39 | 0.18 / 0.12 | 0.06 / 0.07 | +0.54 | −0.02 | +0.15 | +0.01 | 2.23 | 0.957 (4152) | 4.4 / 7.8 |
| M2 | 260915_AASJ_BeadsCalZ_638_665LP | iXon | 18/24 | 85.3 ± 0.1, 59.6 ± 0.1 | 77, 49 | 76, 48 | 0.15 / 0.09 | 0.05 / 0.05 | +0.56 | −0.04 | +0.20 | +0.04 | 2.29 | 0.966 (5481) | 6.4 / 9.8 |
| M2 | 261012_FF_beadscalib | iXon | 22/23 | 91.2 ± 0.1, 65.7 ± 0.1 | 83, 57 | 83, 56 | 0.10 / 0.07 | 0.04 / 0.04 | +0.54 | −0.05 | +0.26 | +0.06 | 2.28 | 0.984 (2249) | 4.4 / 8.7 |
| M5 | 260205_HL_beatcalibration_z_640 | Fusion | 8/9 | 30.5 ± 0.2, 26.1 ± 0.2 | 1, 0 | 9, 0 | 0.26 / 0.20 | 0.05 / 0.05 | +0.37 | −0.03 | +0.41 | +0.25 | 2.60 | 0.943 (6121) | 8.6 / 13.6 |
| M5 | 260818_PM_beadsCal | Fusion | 19/20 | 42.9 ± 0.1, 30.9 ± 0.1 | 30, 5 | 31, 6 | 0.28 / 0.15 | 0.09 / 0.09 | +0.55 | −0.12 | +0.19 | +0.16 | 3.60 | 0.893 (39300) | 8.5 / 15.6 |
| M5 | 260826_AA_beadsCalZ_640_new | Fusion | 16/17 | 44.4 ± 0.1, 30.9 ± 0.1 | 32, 5 | 32, 4 | 0.32 / 0.31 | 0.12 / 0.15 | +0.71 | −0.17 | +0.43 | +0.32 | 5.43 | 0.975 (6740) | 12.1 / 22.4 |
| M5 | 260901_MA_BeadsCalZ_20nm_-1to1 | Fusion | 22/22 | 38.0 ± 0.1, 29.6 ± 0.1 | 23, 0 | 24, 0 | 0.22 / 0.16 | 0.04 / 0.04 | +0.64 | −0.23 | +0.15 | +0.15 | 4.22 | 0.945 (19910) | 8.9 / 14.9 |

**σ is the microscope's.**  With the pixel taken out, M2 blurs by 40–83 nm, M5 by
0–32: the two never overlap, while on one microscope the acquisitions spread by up to 30 nm.  The Evolve on
M2 learns (76, 74) nm, as on Zenodo's (68, 75) from the same microscope and camera.  The iXon's blur is
anisotropic, 53–83 nm along x against 40–57 along y; the Evolve's on the same microscope is not, but it was
recorded before July's realignment (below), so camera and optics changed together and these stacks cannot
tell which made the difference.  M5's sCMOS is anisotropic along x too: 23–32 nm (1 in February, 8 beads), against y at or below
the pixel's own share, so an EMCCD's charge spread cannot be the whole story; a dichroic's σx ≠ σy (the
simulations of §10, 2) can.  The CRLBs (≤ 0.2 nm) are far below the spread between acquisitions, so
that spread is not noise: what varies is the sample or the day.  Freeing the z scale leaves σ within 1 nm (one M5 run, 8 nm): σ does
not trade with it.

**The magnitude is larger on M5 and mostly radial on both.**  Radial RMS 0.10–0.18 on M2, 0.22–0.32 on M5;
the non-symmetric part 0.03–0.06 on M2, 0.04–0.12 on M5.  The maps (the figure) are bright in the centre
and fall to the rim, M5's more strongly: the pattern the dictionary of §10, 2 gives to an amplitude error
(an NA lower in effect than nominal, a soft edge, the aplanatic factor), not to missing phase.  It does
trade with the z scale: freed, the radial RMS drops by 23–46 % (M5 260826: 3 %), so part of the radial
magnitude stands in for the z scale, as on Zenodo.

**Every stack asks for a z scale, and the model does not fit at the noise.**  The probe's z scale is
0.89–0.98 with gains of 2 000–39 000 nats (threshold 12), and the atlas flags every group, at a mean
deviance of 2.0–2.4 on M2 and 2.6–5.4 on M5 (Zenodo: 1.4).  An effective NA below nominal acts as a z
scale (§10, To investigate, 1) and makes the magnitude fall off radially; on M5, NA 1.50 within 0.018 of
the oil's index is the likeliest candidate, and it would also explain M5's worse fit quality (8.5–12 nm
median against 3.9–6.4).  Not tested here: an NA scan is the next run.

**Phase.**  Astigmatism Z(2, 2) is +0.54 to +0.56 rad on M2's iXon (−0.67 on the Evolve, turned 90°) and
+0.37 to +0.71 on M5, with oblique astigmatism Z(2, −2) up to −0.23 on M5 only.  Spherical Z(4, 0) is +0.15
to +0.26 on M2 since July and −0.08 before; M2 was realigned then (folders `260720_FR_spherical_M2`,
`260727_FR_sphericalAdj_M2`, with the camera change).  M5 has the most spherical, Z(4, 0) up to +0.43 and
Z(6, 0) up to +0.32, as a high NA into water would give.

**Answer.**  σ differs between M2 and M5 by more than between acquisitions on either; it is specific to
the microscope (on M2, to its state since July's camera change and realignment), and not a sink for
the z scale.  Its anisotropy, along x on both, is the next thing to explain.  The magnitude differs too (M5's
radial part is about twice M2's), but its radial part partly absorbs the z scale and the effective NA,
which the model does not yet separate: a physical magnitude needs the NA probe of §10, 2 first.

## How a bead fit starts (October 2026, plan §10, 5; `benchmarks/uipsf/real/starts.py`)

Today's start is the pupil at zero, each bead at its brightest slice and the transform from the pairing's
centroids.  The study first gave fits pieces of their own answer, to bound what a better start could buy,
then tried the two candidates: a previous model of the same microscope and elliptical Gaussians per slice
(`model/astig.py`).  Every start ran on the cold fit's bead states, so objectives compare; rows in
`benchmarks/uipsf/results/starts.jsonl` (sim, Zenodo, M2) and `starts_sim2.jsonl` (two simulated channels).

**One channel never needed a better start.**  The cold fit reached the best optimum on 5 simulated
datasets (`model_choices.simulated`), Zenodo 1ch and all five M2 acquisitions.  Its own final values as the
start (the ideal previous model) take 35–50 % off the time, the stage-1 part of it; the astigmatism alone
(the cold fit's or the truth's Noll 5 and 6) and the Gaussian start (`gauss`: z, x, y and Noll 5/6) take
nothing off: stage 1 still needs 12–14 iterations to place the beads and the low modes together.

**A previous model: into stage 1, not instead of it.**  Each M2 acquisition from each of the other four
(the 665LP one among the 676/37: the filter change made no difference):

| previous model | same camera (12) | across July's camera change (8, 0.26 rad apart) |
| --- | --- | --- |
| replacing stage 1 (`learn_beads(start=)`, today's *start from*) | 0.77 of cold's time, all at the optimum | 1.8 of cold's time; one +159 000 nats |
| into stage 1 (`learn_beads(start=, start_stage1=True)`) | 0.69 (0.49–0.81), all at the optimum | 0.59 (0.51–1.0), all at the optimum |

Skipping stage 1 frees every mode at once from a pupil that may be wrong (another camera turns the
astigmatism by 90°), the trap stage 1 exists for (`workflows/beads.py`); started inside it, the previous
model's low modes move with the beads and the higher ones wait.  The 665LP acquisition started the
676/37 ones as well as they started each other, so no wavelength rescaling was needed there.

**Two channels: the transform is the start that matters.**  Simulated pairs (12 beads, biplane and
mirrored ratiometric, seeds 0–4): the centroid pairing starts 0.37–0.59 px off and every cold fit reaches
the optimum; moved 1.5 px more (2.4–2.6 px off) the ratiometric pairs failed 3 times in 5, at 4.5 px every
fit did (10 of 10, +86 000 to +208 000 nats, the secondary pupil bent to take the misplacement).  A true
transform saves no time.  The Gaussians' positions (each bead's mean over the slices near its focus, in
each channel, a projective fit: `gaussian_start`'s ``transform``) start 0.11–0.25 px off, 2–3 × closer than
the centroids, and every fit from them reached the optimum.

Zenodo 2ch (71 pairs, n_med 1.33), before the fixes of "Why Zenodo 2ch stalled" below, which every start
now reaches the same optimum after (2.98239 M, below this paragraph's best): the cold fit ends 22–26 M nats
above the best, stage 1 stopping after 4 iterations (the stall of roadmap item 4).  From the transform the cold fit itself ends with, or from the
Gaussians', it reaches the best (to 70–110 nats over three runs, in 52–80 s); the astigmatism into stage 1
as well stalled again (+13 M), the angled Gaussian's start ended +1 000.  Which basin the stalling stage 1
falls into depends on small differences of path (the same start gave +0 and +113 nats in two runs):
the start shifts the odds, the stall itself is the solver's or the model's to fix -- it was both the start's
backgrounds and the solver's (below).

**The estimator** (`model/astig.py`).  Per slice the spot's covariance; through focus D = C_xx − C_yy and
2C_xy are linear in the defocus (σ²_u − σ²_v = 4σ₀²c·z/d² in the astigmatic width model), so each bead's z
is where they cross zero and their slopes are the astigmatism, matched to the slopes the same estimator
reads from the model's own noise-free PSFs (so its biases cancel).  Weighted by each slice's precision
(N_eff / (C_xx² + C_yy²)), Huber per bead, the window started at the brightest slice: started at the
smallest spot it failed with 0.4 rad of spherical, where a fit far from focus collapses onto one lobe.
- *Estimators:* SMAPpy's `GaussXY` (C++, axis-aligned, < 1 s for 70 beads × 41 slices), an angled Gaussian
  (torch, 7 parameters, 20–40 s) and adaptive moments (0.2 s).  On pure astigmatism all three recover it
  (bright beads: 0.93, 0.91, 0.93 for 0.9 rad); moments put the bead z 75–150 nm off as soon as coma is
  present; only the angled fit sees Noll 5.  The lab's and Zenodo's astigmatism is Noll 6 (Noll 5 below
  0.02 rad: the cylindrical lens sits along the camera's axes), so **an angle in SMAPpy's fitter is not worth adding**.
- *Amplitude:* 84–86 % of the learnt Noll 6 on Zenodo (−0.48 for −0.56; 2ch −0.39, −0.42 for −0.46, −0.49),
  0.5–0.8 on simulations with 0.4 rad of spherical and a radial magnitude (Noll 4 = −0.15), which change
  how fast the spot widens and are not in the start.  Below about 1 000 photons per slice over 80–150
  background the slopes are noise.
- *Bead z:* within 5–15 nm (bright) and 15–30 nm (dim) of the truth, with a common offset of up to 60 nm
  where spherical shifts the focus.

**Decided from it and built** (plan §10, 5): `calibrate_beads` takes a previous model into stage 1 and
starts a pair's transform from the Gaussians' positions (above, "The calibration's entry point"); the
Gaussian astigmatism and z are not used as a start.

## Why Zenodo 2ch stalled (October 2026, roadmap item 4)

The cold fit of `2ch_40nm_bead` (71 pairs, n_med 1.33) stopped stage 1 after 4 iterations, 70.06 M nats
from a start of 70.13 M, and ended 22–26 M above the best.  Three things combined, each found in stage 1's
history (objective, α, evaluations, damping, step per iteration) and fixed:

1. **The start put backgrounds at softplus' floor** (`model/start.py`).  Channel 0 is dim: its border
   median is around 11 photons and moves in steps of the camera's gain, 4.7 photons.  Its minimum over 40
   slices, the start's background, was 0 or below for 50 of 88 beads and was clamped to 10⁻³, where the
   softplus's d(value)/du is 10⁻³.  The start now floors it at one photon (slope 0.63), `BACKGROUND_FLOOR`.
2. **The saturation guard shortened every parameter's step for one** (`solve/lbfgs.py`).  The first
   Gauss–Newton step drove three of those backgrounds further down; `_saturates` refused any step that
   shrinks a derivative more than 100-fold, and the line search quartered α for the whole step until it
   did not: 6 quarterings without an evaluation, α = 2·10⁻⁴ for all 71 beads and the pupil, the next
   iteration 9 quarterings and no step at all.  Then a step at α ≈ 4·10⁻⁶ moved no parameter by 0.1 of
   its standard error and stage 1's step test (`STAGE1_XTOL`) stopped it as converged.  Now only the
   components that would saturate are quartered (`_held_back`, zero after 8 tries); the rest keep the
   full step.
3. **The gradient was wrong where the pupil magnitude is negative** (`model/psf.py`).  1 + Σ c_j Z_j dips
   below zero at the rim once magnitude coefficients reach −0.1 to −0.2, and `torch.polar`'s backward
   (torch 2.14) flips the gradient's sign there while its forward mode (the Jacobian) and finite differences
   do not.  `value_and_grad` then disagreed with `normal_equations` by up to 3 000 nats per unit in every
   magnitude coefficient: each L-BFGS step after a rebuild failed its line search and the damping
   ratcheted ×10 per refresh, until a small simulated fit stopped 1 400 nats above the optimum, "converged".
   The pupil is now `polar(1, φ) · magnitude`.

| code (cold start) | stage 1 | final objective | time |
| --- | --- | --- | --- |
| before | 4 iterations, *step* | 25 992 485 | 62 s |
| the polar fix alone | 4, *step* | 25 762 666 | 53 s |
| the floor alone | 22 | 2 982 389 | 62 s |
| the polar fix and the guard, no floor | 35 | 2 982 389 | 135 s |
| all three | 22 | 2 982 390 | 66 s |

The floor removes the cause on this data, the guard keeps a parameter that reaches its bound mid-fit (the
M7 backgrounds, 11 photons to 0) from holding back the rest, and the polar fix matters wherever the
magnitude goes negative, Zenodo or not.  The step test is unchanged: with the guard fixed, a short step
again means the line search found nothing better, and requiring a gain below `tol` as well cost stage 1
17 more iterations here.

**What it changes.**  Every start now reaches the same optimum, 2 982 389 ± 1 nats (`starts.jsonl`, run
2026-10-21): cold in 66 s, the Gaussians' start (`gauss`) in 35 s, its positions in 37 s, its transform
alone or the cold fit's final transform in 52–69 s.  The "best" of the start study above, 3.0346 M, was
52 000 nats short: those fits started from the same floored backgrounds and kept many of them there.  So
the transform start was never what prevented the stall; it changed the first steps' direction enough to
miss the three saturating backgrounds.  n_med 1.35 (v1's) stalled in stage 1 the same way (70.08 M) and
escaped in stage 2 by chance (3.065 M); with the fixes it ends at 3.013 M, 31 000 nats worse than 1.33:
the data prefer water.

Found on the way: the Zenodo 2ch folders `Pos0_..._Z_1` and `Pos1_..._Z_1` hold the `..._Z_2` file, the
same bytes as the `_Z_2` folders', so 12 of the 88 beads are copies of others (stacks 0 and 1, 4 and 5;
beads 3 and 9 have the same data).  Every number on 2ch above includes them; left as they are (decided
2026-10-21), a later study may leave one copy out.

## M5, step 1: the in situ fit on cut regions (October 2026, plan §9 M5)

The core of in situ learning, before detection: each emitter's region in, the pupil, the focus and every
emitter's height out, with the sign of the even aberrations checked.

- **`model.InSituGeometry`**: `emitter.xyz` holds x, y and the height h above the coverslip; the shared
  `geometry.focus` (prior N(f₀, 500 nm)) is the focal plane's height.  It returns four coordinates per
  observation, (x, y, z_k − f, h); `ForwardModel` passes the fourth to the PSF as `depth_nm` (per emitter
  and observation now), and `emitter_linearisation` adds its tangent factor 2πi k_z,med.  So both the
  emitters' heights and the focus keep the linearised Jacobian columns: the in situ normal equations
  equal the JVPs' to 1e-10 (`test_insitu.py`).
- **`workflows.insitu.learn_insitu`**: every emitter localised on the starting pupil from heights every
  150 nm over ±900 nm about the height in focus, f·n_med/n_imm, the best kept (`localise_from_heights`);
  then `learn_beads` with `start_stage1` (low orders first, the beads' rejection); then the twin test
  (`twin_fit`): the learnt pupil's twin, every emitter localised on it again, fitted, and the optima
  compared.  ±10 nats (`TWIN_DECISIVE_NATS`) decides; the sign the start gave is kept and a preference
  for the twin is a warning (`InSituCalibration.warnings`).

**Measured** (simulated, 40 emitters 0–1000 nm deep, one frame each, 3000–6000 photons, NA 1.43 oil into
water, astigmatism 0.7 rad with coma and spherical, focus 720 nm started at 600; 4 CPU cores, ~9 s):

| start | phase | focus | heights | twin test |
| --- | --- | --- | --- | --- |
| astigmatism +0.4 (right sign) | every mode within 1.5σ of the truth | 610 nm, σ 82 | −80 nm in common, 19 nm rms about it | agree, +62 nats |
| astigmatism −0.4 (wrong sign) | the twin: astigmatism −0.69, spherical (Noll 11) +0.28 to make up for the mismatch | 664 nm, σ 114 | mirrored | prefer twin, −54 nats: warns |

The focus and all heights move together, nearly degenerate (the focal shift n_med/n_imm): the heights
relative to each other are well determined, their common offset only to the focus's CRLB.  That is what
the coverslip edge (plan §9 M5) is to pin, and part of the depth, n_med and spherical study.  Held-out
emitters are not in this step: the beads' held-out report fits slices of a stack, which a single frame
does not have; the in situ one comes with selection (step 4).

## M5, step 2: linked blinks (October 2026, plan §9 M5; `benchmarks/uipsf/synthetic/linked_blinks.py`)

A blink seen in K frames is one emitter: `InSituGeometry` gives every frame its one position, and
`PerObservation` its photons per frame.  The start needs nothing new (`rough_start` already places an
emitter once over its frames); `localise_from_heights` now starts every intensity model through the
batch's start, `PerObservation` takes a start per frame ([E, K]), and the emitters' photon score is their
mean per frame (`score_beads`).  Four linked frames halve the height's CRLB, as four times the light
should (`test_insitu.py`).

**Localisation on the true pupil**: 60 emitters 0–1500 nm deep, focus 600 nm (in focus at ~530), 3 frames
each, localised linked and frame by frame; the fraction within 100 nm of the true height and the median
error (`linked_blinks.jsonl`, 2026-10-21):

| photons per frame | distance from focus | linked: within 100 nm | median | single frames: within 100 nm | median |
| --- | --- | --- | --- | --- | --- |
| 500 | 0–300 nm | 94 % | 19 nm | 81 % | 45 nm |
| 500 | 300–600 nm | 57 % | 36 nm | 44 % | 240 nm |
| 500 | 600–1000 nm | 14 % | 650 nm | 5 % | 920 nm |
| 1500 | 0–300 nm | 100 % | 11 nm | 96 % | 15 nm |
| 1500 | 300–600 nm | 93 % | 21 nm | 77 % | 34 nm |
| 1500 | 600–1000 nm | 50 % | 110 nm | 19 % | 380 nm |

Linking matters where the decision was made for it: 300–1000 nm from focus it places 1.2–3 times as many
emitters.  What it cannot rescue, the dim ones far from focus, is mislocated by hundreds of nanometres,
far beyond the CRLB (77–170 nm): a localisation that ends on the wrong side or in a local minimum, from
heights 150 nm apart, in a 13-pixel region the spot outgrows.  Selection (step 4) has to keep those out.

**Learning** (`learn_insitu`, 30 blinks × 3 frames at 1500 photons per frame, start astigmatism +0.4,
focus 720 nm started at 600): linked and the same 90 frames alone learn the pupil equally well, phase error
0.016 rad rms against a CRLB of 0.016 for both; 20 s against 25 s, and 4 of the 90 single frames rejected.
The focus came back 590 ± 114 and 553 ± 111 nm (the degeneracy of step 1), the twin test agreed (+56, +52
nats).  So linking pays in localisation, the start, selection and the rounds, not in the joint fit, where
the shared pupil already holds each frame in place.

## M5, step 3: simulated blinking movies (October 2026, plan §9 M5; `uipsf/movie.py`)

`movie.simulate_movie(model, shared, MovieSettings)` makes the test data of the steps to come: whole
frames of molecules blinking at random places and heights, with a truth table (`MovieTruth`: frame,
molecule, blink, x, y, height, photons).

- *Molecules* uniform over the field, heights uniform in a range, a fraction exactly on the coverslip
  (`on_coverslip`, for the edge that is to pin the focus).
- *Blinking* is SMAPpy's model (`smappy.simulate.kinetics.blink` and `emission`), not a second one:
  continuous-time on, off and bleaching, a frame's photons by the time on in it.
- *Illumination*: a Gaussian profile on the molecules' photons, not on the background.
- *Spots*: the model's own PSF, pupil, blur and focus through `InSituGeometry`, rendered in a region of
  `spot_px` (25) pixels about each spot's pixel; flat background, Poisson noise, optional read noise.
  Frames are photons.

**A spot cut from the movie is the fit's μ** for that emitter (`test_movie.py`), up to the fit region's
circular blur: an 11-pixel region is 0.2–0.6 % of the peak off at its edges, under 10⁻⁴ in its centre;
renders in 25 and 41 pixels agree to 10⁻⁶.  An emitter in water imaged at NA 1.43 puts only ~60 % of its
"photons" in the frame: the normalisation counts the supercritical light of an emitter at the coverslip,
which one in the sample never emits.  Photons in the truth are therefore the model's, and a detection's
photons will be ~0.6 of them away from the coverslip.

**Size and density** (128 × 128 pixels, 1000 frames, 5000 ± 2500 photons per blink over 1.5 frames,
3 blinks each, P = 64, 4 CPU cores): 2000 molecules give 15 spots per frame (0.06 µm⁻²), 32 % of them
with another within 1.5 µm, in 2.6 s; 6000 give 45 per frame (0.19 µm⁻²), 70 % with a neighbour, in
5.8 s.

## M5, step 4: detection and selection (October 2026, plan §9 M5; `smappy/calibrate/insitu.py`)

A blinking movie in photons to the blinks `learn_insitu` learns from, in SMAPpy (detection and the
fitters are SMAPpy's, src/uipsf/CLAUDE.md), torch-free; `InSituROIs.batch()` makes the `uipsf.Batch`.
The steps as the plan decided them, with what the simulated movies changed:

- **Background**: a sliding median over 21 frames per pixel.
- **Detection**: DoG at σ 1.2, 2 and 3 px; a candidate is kept on its absolute significance (photons
  above the background in a disc of radius 2σ against the background's Poisson noise, ≥ 6).  An in-focus
  and a defocused molecule of the same photons are both found where the defocused one's DoG peak is
  under 0.2 of the other's, below v1's relative threshold (`test_insitu_rois.py`).
- **The first fit**, SMAPpy's elliptical Gaussian in a 15-pixel region.  *Found on the movies:* a
  defocused, astigmatic spot has several local maxima, and a secondary one sums the molecule's own
  photons, so it passes the significance test; half the detections were such, each rejecting its own
  molecule as an overlap.  Detections whose Gaussians meet within 2 px are one molecule.  And a quarter
  of what remained were Gaussians spread over the background (σ ~11 px against 2.7 for molecules), each
  rejecting everything within 3σ: a fit wider than a third of the region is no detection.  With both,
  96 % of the detections are molecules.
- **Blinks**: SMAPpy's `group.connect`, no dark frame, the precision test with a floor of 0.5 px (the
  Gaussian's CRLB, ~0.09 px, understates a fitted Gaussian's scatter on a PSF that is not one, ~0.2 px,
  and split blinks).  A blink's frames are **summed** into one region about their mean position: Poisson
  with mean (ΣN_k)·I + K·b, the read noise K times; K = 1 for every emitter, no padding.
- **Overlaps** by 3σ of each spot, masks, photon steps (inner frames by more than 2×), as decided.
- **Selection**: z proxy σx² − σy² (or σx² + σy²), 21 bins over its 1–99 % range, shape outliers out
  (3 robust sds of σx², σy² in the bin), a random draw from the factor-3 brightness window below the
  bin's 90th percentile, 10 % of the draw held out.  Every blink keeps its state and reason.

**On a simulated movie** (128 × 128 px, 600 frames, 600 molecules 0–1400 nm deep, 10 % on the
coverslip, focus 700 nm, Gaussian illumination of FWHM 160 px; astigmatism 0.7 rad with coma and
spherical): 5725 detections, ~1400 blinks; ~510 rejected as overlaps, ~515 outside the brightness window,
~86 for their shape; 230–280 used.  `learn_insitu` on them, started from astigmatism +0.4: every phase
mode within ~2σ (σ 0.003–0.008 rad), the focus 717 ± 30 nm, the twin test +832 nats; 6 s for the
selection, ~60 s for learning (4 cores).  The used blinks are molecules, 95 % within 3 px of one; spots
600–700 nm from focus (σ 4–5 px) are placed 2–4 px off by the Gaussian, truncated by the region.  Their
region is still about the molecule and the pupil fit places it; a larger region (17–21 px) would serve
those spots better, to measure with real data.

**Frames, not summed blinks, are fitted** (decided October 2026, after the above): each selected frame
is its own region about its own position (`InSituSettings.unit = "frame"`, v1's way); linking stays, so
that overlaps and photon steps are judged on a blink's frames and a molecule's frames are used or held
out together.  Summing (`unit = "blink"`) is the option, to be compared before it is the default.
Learning on the summed blinks and on the frames gave the same pupil in step 2's benchmark, and the
brightness window keeps a blink's bright central frames, drops its partial first and last.

## M5, step 5: the rounds and the coverslip edge (October 2026, plan §9 M5; `workflows/insitu_rounds.py`)

`calibrate_insitu(model, rois, start)`: round 1 learns from the Gaussian's selection; from round 2 every
unit the detection kept is localised on the model, chosen again by the model's height (z bins), photons
(brightness window) and deviance (shape, one-sided), and learnt again from the previous model with the
emitters started where they were found.  It stops when the Zernikes and the focus moved by no more than
their noise (rms of change / √(σ_new² + σ_old²) below 1: the largest of twenty changes is ~3 from noise
alone, since each round learns from other molecules) and the edge sits at 0, or after 4 rounds.  The
twin test runs once, at the end.

**The coverslip edge** (`coverslip_edge`): the lowest 30 % of the localised heights, fitted by maximum
likelihood as a flat density above an edge s, optionally a spike on it (kept if it gains > 6 nats), each
height blurred by κ ≥ 1 times its CRLB, plus ≤ 5 % outliers uniform over the window.  Each part was
found necessary on the simulated movies: at the true focus the molecules on the coverslip are localised
at 1 nm (median, IQR 27), but a few lie hundreds of nanometres low and the scatter (~20 nm) exceeds the
CRLB (17 nm), and without κ and the outliers the fit moved the edge 75 nm down to explain them; with a
spike's weight free and no molecules on the coverslip, its bound made the edge 10–28 nm high; with 20 %
outliers and a 2-nat threshold, a false spike 80 nm above the edge once in five.  On simulated heights
(1500, uniform 0–1400 nm, 1.2× the CRLB's scatter, 1 % outliers):

| on the coverslip | CRLB | bias | scatter | reported sd |
| --- | --- | --- | --- | --- |
| 20 % | 30 nm | −0.8 nm | 2.2 | 2.5 |
| 20 % | 80 nm | −12 nm | 31 | 13 |
| none | 30 nm | +1.7 nm | 10 | 8.6 |
| none | 80 nm | +3.4 nm | 15 | 16 |
| 5 % | 30 nm | −22 nm | 18 | 7.4 |
| 5 % | 80 nm | −59 nm | 15 | (none) |

A sparse pile-up is the weak case: neither model fits it, and the edge is pulled low.

**From the edge to the focus.**  The edge's shift s says how far all heights moved together; the next
round's focus prior is N(f − s·r, σ_s·r) with r the focus per height of the coverslip's molecules.  The
paraxial r = n_imm/n_med = 1.14 is far off at NA 1.43 into water: a focus 120 nm high raised those
molecules by 65 nm, not 106, so r ≈ 1.6–1.85 and a paraxial correction made only ~60 % of the way.  So r
is measured (`_measured_ratio`: the window's molecules localised again with the focus 50 nm higher) and
then refined by a secant on the rounds' (focus, edge) pairs.

**On simulated movies** (as step 4's, the focus started at 450 nm against 700):

| movie | round 1 | round 2 | then | twin |
| --- | --- | --- | --- | --- |
| 600 molecules, 20 % on the coverslip | 690 ± 29 | 699 ± 5 | 697, 702 (± 5) | +732 nats |
| 300 molecules, 20 % on the coverslip | 655 ± 41 | 697 ± 5 (r measured) | stopped | +312 |
| 600 molecules, none on the coverslip | 668 ± 34 | 675 ± 7 | stopped | +568 |

The Zernikes stay within ~2σ of the truth in every round.  With a pile-up the edge pins the focus to a
few nanometres; without one only to the flat edge's ~10–15 nm, and its reported error is optimistic
(675 ± 7 against 700).  A round costs one localisation of every unit (~2000 here, ~10 heights each) and
one learning: ~110 s on 4 cores at P = 32.

## M5, step 6: in situ in the PSF calibration window (October 2026, plan §9 M5)

A third method, *in situ (blinking molecules)* (`smappy.calibrate.psf_methods.InSitu`), beside the bead
average and the pupil model; the window shows its section and knows nothing else of it.

- **Files are movies** (`insitu_run.movie_rois`): *frames per movie* (4000) read in blocks of 200 spread
  over each movie, through the camera as a fit resolves it; detection per block, the choice over all.  The
  Stacks page reads one frame at a time in movie mode (`StackBrowser.set_movie`) and rings each unit on its
  own frame, coloured by state.
- **The section *in situ*** (`InSituModelSettings`): *focus* (a value, or *auto*: estimated from the
  molecules on the coverslip before learning, `focus_from_data`), *astigmatism* (the start's Noll 6 in nm,
  when there is no *start from*), *on the coverslip*, *rounds*, *spline z*; the detection's settings
  (`InSituSettings`) under *more*.  One channel: *mode* is disabled.
- **The table counts** frames and molecules by state; no marks (thousands of molecules are chosen by rule).
- **Pages**: *z bins*, *Rounds and edge*, uipsf's *Pupil and Zernikes*, *Zernike table* and *Residual
  atlas*, and *Model vs data* per height bin (`insitu_views`).
- **Saved** as the pupil model is: the spline beside the movies under the bead calibration's name, the model
  file beside it (its geometry *in situ*, the focus with it; `export.model_file`).  The spline samples the
  model at the learnt focus through the used heights ± 100 nm, every 20 nm (`insitu_spline`); its z is the
  height above the plane in focus (z = 0 at the sharpest plane) or above the coverslip (the grid then
  reaching 0, since z0 must lie on it), recorded in its parameters with the focus and the offset.

**Measured** (simulated movie, 300 frames, 300 molecules, a fifth on the coverslip, true focus 700 nm):
a TIFF movie through the camera to a saved calibration, two rounds, focus 702 ± 5 nm from 450 given, every
phase mode within 1.6σ; 3 min on 4 cores and Apple's GPU.  With the focus on *auto*, the estimate before
learning was 774–780 nm (from a guess of 1000) and the rounds took it to 703 ± 5.  The estimate alone is a
start, 75–80 nm off: the window's rounds and the edge refine it.  Bounded to 200 molecules started from
heights every 250 nm over 0–3 µm, it takes ~100 s on one CPU core in float32.

**Tests and benchmark** (October 2026): every in situ test takes seconds on one core -- learning on 8–16
molecules with a pupil of 16 samples, a 64 × 64 movie of 150 frames, the window's run on a small model --
and the full-size runs (movies of 600 molecules, the rounds, the focus from the data) are a benchmark run by
hand, `benchmarks/uipsf/synthetic/insitu_movie.py`, whose numbers are the ones above.  The twin fit must
converge to the solver's own tolerance: stopped at stage 1's looser one, its optimum stayed too high and a
start of the wrong sign passed as agreeing.

**Speed** (October 2026).  A window-sized run -- the window's model, ~300 molecules learnt, ~1500 kept,
two rounds -- ran for over 40 minutes on Apple's GPU, for three reasons, each fixed:

- *the pupil's sampling*: `insitu_model` asked for defocus up to 2 µm without aliasing, 96 samples; what
  must not alias is the molecules' distance from the plane in focus (the depth's and the focus's waves
  largely cancel there), ±1 µm: 64 samples, 2.25 times less work;
- *recompiling*: on Apple's GPU the tangents' kernel is compiled for fixed shapes, and in situ calls it with
  a new number of emitters at every localisation, round and twin; after 8 shapes torch stopped compiling.
  The emitters and tangents are now padded to powers of two (`model.psf._bucket`);
- *the starts*: every unit was localised from 10–20 heights, the whole pool every round.  Each now starts
  where the model's own Gaussian widths match its own (`insitu.width_starts`: the two best matches and the
  height in focus) and where it was found last: 3–4 starts.

After: the same run in 69 s (two rounds, focus 707 nm from 450, truth 700): localising the pool 23 s, the
first round's localisation 10 s, learning 6 s.  The benchmark (P = 32, CPU) went from 558 s to 96 s and
from 193 s to 60 s with the same answers -- focus 702 ± 5 and 676 ± 9 nm, the largest Zernike error 3.3σ
over twenty-odd modes.

**The benchmark's run** (`insitu_movie.jsonl`, 2026-10-21, 4 CPU cores, P = 32): a fifth on the coverslip,
the focus from 450 nm: 690 ± 29, 704, 700, 705 (± 5) over four rounds, phase within 2.6σ, twin +663 nats,
558 s; none on the coverslip: 668 ± 34, then 678 ± 9, stopped, twin +568, 193 s.  The focus from the data
before learning, from a guess of 1000 nm: 735 nm with molecules on the coverslip, 628 without.
