# Field-dependent aberrations: how to learn them, and how to fit with them

A study, November 2026: the literature, the options for uipsf's coefficient field (`plan.md` §3, the slot
`NodalPolynomial` + `BSplineResidual` + `FocalSurface`) and for SMAPpy's fitter (`plan.md` §7.3), and
measurements on synthetic data that decide between them.  What v1 built of it is §8, and the first real field §9.

**In short.**
- *Representation in uipsf:* the Zernike coefficients as a nodal-aberration-theory polynomial of the field
  position, plus a residual on a grid of nodes with a smoothness prior, only on the modes that vary
  (astigmatism, coma).  Tilt and defocus over the field are coordinate maps (distortion, focal surface),
  never PSF shapes.
- *Fitting in SMAPpy:* a spline per node of a 5 × 5 to 7 × 7 grid over the field, the four neighbouring
  splines' coefficients blended bilinearly per spot.  That is exactly the spline of the bilinearly blended
  PSFs, it reaches the bias of the exact spline at 7 × 7, and costs about 0.25–0.5 GB of RAM.
- *Today's one spline per field* is biased by up to 40 nm in z at the corners of a mild field (85 nm at
  twice that field); the nearest of 20 × 20 tiles (what Droste et al. do) is close but not quite exact.

---

## 1. What varies over the field, and how: nodal aberration theory

Nodal aberration theory (NAT; Thompson, Shack) gives the field dependence of each Seidel aberration in a
nominally rotationally symmetric system whose elements are slightly decentred or tilted.  In complex notation
H = x + iy for the field position relative to the optical axis:

| aberration | over the field (NAT) | real parameters | where it belongs |
|---|---|---|---|
| tilt (Noll 2, 3) | distortion, ∝ H³ (with linear terms: magnification) | – | a lateral coordinate map, not a PSF |
| defocus (Noll 4) | field curvature ∝ \|H\|², plus linear (a tilted focal plane) and constant | 3–4 | the `FocalSurface`: a z offset, more exact at high NA than a Zernike defocus |
| astigmatism (Noll 5, 6) | a·H² + b·H + c, a real, b and c complex: the binodal field | 5 | a PSF shape term |
| coma (Noll 7, 8) | a·H + c, a real, c complex | 3 | a PSF shape term |
| spherical, trefoil, higher | essentially constant | 0 | the base pupil |

So the field's shape changes come from four coefficients, astigmatism and coma; the rest of the field
dependence moves the spot and changes nothing a spline holds.  Caveats: the axis centre trades with the
linear terms (fix one); the dichroic, a cylindrical lens or a deformable mirror add constants, which the
constant terms take; a tilted sample gives coma that is constant over the field (§2) -- the layered model's
coverslip tilt (`optics/layers.py`) separates it from the objective's.

## 2. Literature

**Droste et al. 2024** (Delft and Albuquerque; bioRxiv, [DOI](https://doi.org/10.1101/2024.12.11.627909);
`roadmap.md` names it as the reference) is the closest work.
- *Model:* a fully vectorial pupil with Zernike phase; no amplitude aberrations and, in the main text, no
  extra blur.  The Zernike coefficients are NAT polynomials written in Legendre polynomials of the field
  coordinates (for a square field): **13 coefficients at first order** (defocus, astigmatism, coma,
  spherical), 43 more at second order (trefoil, secondary astigmatism, coma, spherical).  Tilt is not
  modelled.  No residual beyond the polynomials.
- *Learning, without beads:* a few thousand spots (17 × 17 px in 3D), an equal number from each cell of a
  10 × 10 grid over the field; alternating local updates (x, y, z, photons, background, the NAT
  coefficients fixed) and global ones (the NAT coefficients, on a random 20 % of the spots); 6–10 rounds,
  30–45 s on a 12-core CPU.  In 2D, 10 random restarts.  The ±z mirror solution is fixed by the sign of the
  constant astigmatism.
- *Fitting:* "Vectorfit", maximum likelihood with the vectorial model on an NVIDIA GPU: a phasor start for
  x, y (and z from astigmatism), linear regression for photons and background, and a precomputed OTF that
  cuts the Fourier transforms per iteration from 24 to 4.  **The OTFs are precomputed on 20 × 20 patches
  of the field**: a fit uses its patch's.  3D: about 7·10³ fits/s for 17 × 17 px on an RTX A4000, which
  they put at 30× slower than a GPU spline fitter.  (The exponents are lost in the PMC text; these are
  consistent with the ratio they state.)
- *Results:* χ² still 16–20 % above shot noise, which they attribute to higher orders and amplitude; against
  constant aberrations, up to 1.5× better χ² at the edges, z shifts up to 150 nm and lateral shifts of tens
  of nm on Fu et al.'s 180 µm NPC data; precision at the CRLB.  Bead calibrations on two days differed by a
  constant coma, traced to sample tilt.

**Others.**
- uiPSF v1 (Liu et al., [DOI](https://doi.org/10.1101/2023.10.26.564064)): a free map per Zernike coefficient
  over the field, with smoothing.  Droste calls the bead-derived maps of FD-DeepLoc "large and erratic" at the
  edges -- the argument for a constrained model.
- FD-DeepLoc (Fu et al. 2023, [DOI](https://doi.org/10.1038/s41592-023-01775-5)): interpolated Zernike maps
  from beads, a network that takes the field position.
- PPG3D (Xiao et al. 2024, [DOI](https://doi.org/10.1126/sciadv.adj3656)): PCA of measured bead PSFs (151–424
  field positions), interpolated locally between 3 lateral × 2 axial neighbours; three times more accurate than
  interpolating Zernikes between the calibrated positions; a generator of training data for a network, not a
  maximum-likelihood fit, and no physics (a vectorial model is needed for the index mismatch).
- Lu et al. 2025 ([DOI](https://doi.org/10.1364/BOE.545534)): discrete field-dependent PSFs (tiles) on a GPU.
- von Diezmann et al. 2015 ([DOI](https://doi.org/10.1364/OPTICA.2.000985)): field-dependent errors of
  50–100 nm, corrected after the fit from a nano-hole array.
- Astronomy has modelled field-dependent PSFs as interpolated principal components for decades (Jee et al.
  2007, Hubble ACS, about 20 eigen-PSFs).

We found no work that fits SMLM data with splines interpolated over the field from a physical model.

## 3. Speed and memory today

Measured in a 4-core cloud container unless said otherwise.
- *SMAPpy's Spline 3D:* 3.6·10³ fits/s on one thread, 1.0·10⁴ on four; on an M1 Pro, about 850 000 spots in
  22 s (`docs/performance.md`), about 4·10⁴ fits/s.  The fitter is CPU-only C++; a spline lives in RAM.
- *One spline:* 64 × 160 × 26 × 26 float32 = 28 MB for a 27 px box and 10 nm planes over ±800 nm; about 10 MB
  for a 17 px box, half that at 20 nm planes.
- *Making one from a uipsf model* (`calibrate.pupil.spline_calibration`, 161 planes of 27 px): 0.3 s.
- *The pupil model itself on the CPU:* 3.5·10³ PSFs/s for 13 × 13 px without derivatives, so roughly
  10² fits/s: a fit on the model is out of question on the CPU, and the reason splines stay.
- uipsf runs on CUDA, MPS and the CPU (§5.1 of the plan); Droste's GPU code is NVIDIA only.

## 4. How to store the PSF between field positions

`benchmarks/uipsf/synthetic/field_interpolation.py`, `results/field_interpolation.jsonl`.  The truth is the
vector model (NA 1.43, 660 nm, 110 nm pixels, 0.8 rad of Noll 6) with a nodal field over u, v ∈ [−1, 1]:
quadratic astigmatism 0.15 rad and linear 0.03 rad, coma 0.10 rad at the edge, and twice that.  The metric is
the expected excess χ² of a spot at 5000 photons and a background of 20, the parameters held at the truth
(about 1 is what one spot just notices; it grows with the photons), mean / max over 120 random positions.

| representation | stored / evaluated per spot | field × 1 | field × 2 |
|---|---|---|---|
| one PSF for the field | 1 / 1 | 29 / 197 | 114 / 716 |
| nearest of 9 × 9 tiles | 81 / 1 | 0.69 / 4.0 | 2.7 / 18 |
| nearest of 20 × 20 tiles (Droste) | 400 / 1 | 0.17 / 1.2 | 0.66 / 4.1 |
| 3 × 3 grid, PSFs bilinear | 9 / 4 | 1.0 / 6.4 | 10 / 38 |
| 5 × 5 grid, PSFs bilinear | 25 / 4 | 0.059 / 0.39 | 0.65 / 4.4 |
| 7 × 7 grid, PSFs bilinear | 49 / 4 | 0.012 / 0.069 | 0.13 / 0.85 |
| 9 × 9 grid, PSFs bilinear | 81 / 4 | 0.004 / 0.028 | 0.045 / 0.37 |
| PCA of the 9 × 9 grid's stacks, K = 6 | 7 / 7 | 0.18 / 1.2 | 2.7 / 16 |
| PCA, K = 8 | 9 / 9 | 0.053 / 0.40 | 0.98 / 7.0 |
| 5 × 5 grid, **Zernikes** bilinear | – | 0.022 / 0.21 | 0.088 / 0.85 |
| 7 × 7 grid, **Zernikes** bilinear | – | 0.004 / 0.045 | 0.017 / 0.17 |

- **Interpolation beats tiles.**  A 5 × 5 grid blended bilinearly is better than 400 tiles.
- **A global PCA basis is not the economy it looks.**  It needs 6–8 components for what a 5 × 5 grid does, and
  falls behind as the field grows: a basis of whole z-stacks fits the PSF's change with z badly.  It saves
  memory, not error.
- **Interpolating Zernikes is 3–8× better than interpolating PSFs at the same grid**: the PSF is nonlinear in
  the aberrations, so a blend of PSFs is worse than the PSF of blended aberrations.  What is left is a bilinear
  surface approximating a quadratic one: 4–8 mrad at 7 × 7.

## 5. Fitting with blended splines, end to end

A spline's coefficients are linear in the samples it is made from, and so is the OTF.  So blending
coefficients bilinearly, blending OTFs bilinearly, and blending the sampled PSFs bilinearly are one and the same
model (checked: 4·10⁻⁹ against coefficients of 4·10⁻²).  It is not interpolating the aberrations (§4's
last rows).  Two conditions: every node's spline keeps the model's own normalisation (photons per photon),
not one of its own -- the bead calibration's "brightest plane sums to one" would make the blend wrong -- and
the field's tilt and defocus stay out of the node PSFs, or the blend averages spots at different places.

`benchmarks/uipsf/synthetic/field_spline_fit.py`, `results/field_spline_fit.jsonl`: spots simulated with the
exact model at 6 positions near the edges and corners, 3000 each, 5000 photons, background 20, z ±500 nm,
13 × 13 px, fitted by SMAPpy's `fit_cspline` with each spline.  Bias: the median error per 100 nm z bin, RMS
over the bins, mean over the positions; "worst z" is the worst position.  In nm; the "exact" row is the noise
floor.

| spline | field × 1: bias x / y / z | worst z | field × 2: bias x / y / z | worst z |
|---|---|---|---|---|
| exact, at that position | 0.5 / 0.6 / 0.8 | 1.1 | 0.6 / 0.6 / 1.2 | 1.9 |
| **one for the field (today)** | 3.1 / 8.1 / 15.6 | **38** | 10 / 20 / 37 | **85** |
| nearest of 20 × 20 tiles | 0.7 / 0.6 / 1.7 | 3.5 | 1.3 / 1.5 / 3.3 | 5.9 |
| 3 × 3 grid, blended | 0.8 / 1.0 / 2.7 | 4.5 | 2.6 / 4.1 / 6.6 | 9.5 |
| 5 × 5 grid, blended | 0.6 / 0.6 / 1.2 | 1.7 | 1.1 / 1.3 / 2.8 | 4.2 |
| **7 × 7 grid, blended** | **0.6 / 0.6 / 0.8** | **1.2** | **0.8 / 1.0 / 1.6** | **2.4** |

The one spline also loses z precision at twice the field (SD 22 nm against 14).  No fit failed in any
variant.  Not yet measured: the cost of blending per spot in C++.  The four cells' 64 coefficients are mixed
before the usual tricubic evaluation, so an estimate is 1.5–2× today's fit, not 4×; `CSpline4Pi`
(`csrc/models.hpp`) already evaluates weighted sums of splines per spot.

## 6. Learning: what to do

1. **`NodalPolynomial`** for the Zernike coefficients, first order by default, second-order terms only where a
   likelihood ratio on held-out data asks for them (`select_terms`, `diagnostics/probes.py`).
2. **Tilt and defocus outside the pupil:** the field's defocus is the `FocalSurface`, its tilt a distortion map.
   Both are the coordinate corrections the fitter needs anyway.
3. **A residual on a grid of nodes** -- the export's nodes -- only on astigmatism and coma, interpolated
   bilinearly, with a second-difference prior.  Where data are thin the prior returns the nodal polynomial;
   where they are rich it takes what NAT misses.  4 modes × 49 nodes ≈ 200 parameters: fine in situ with the
   spots chosen evenly over the field, not by 20–50 beads alone.  It is the plan's `BSplineResidual` with linear
   B-splines on the export's knots.
4. **Magnitude and blur σ shared over the field first; then watch σ.**  Field dependence goes into σ
   (`plan.md`, M8 study); σ should shrink once the field is modelled.  If it does not, the blur has other causes.
5. **Data and solver:** beads over the field (Zenodo `1ch_FD_bead`) and in situ (`1ch_FD_smlm_nup96`), the
   spots stratified over the field.  The joint Levenberg–Marquardt solve with its Schur complement takes the
   place of Droste's alternation.  Each emitter already has its own pupil (the emitter axis), so the field adds
   shared parameters and almost no forward cost.
6. **Degeneracies:** the axis centre against the linear terms; in situ, the ±z mirror solution and a focal
   surface against a tilted or curved sample.

**Export:** sample the learnt model at each node of the grid (49 stacks, seconds), with one normalisation and
without the field's tilt and defocus; the spline calibration holds them with the grid's geometry, the focal
surface and the distortion map.

## 7. Our data: M5 beads

`benchmarks/uipsf/real/m5_field_map.py`, from results already kept (`results/m5_field.jsonl`, each bead's own
phase to order 6; `results/m5_modes/*__zernike_free.npz`, every slice localised alone with the shared model).
The beads span about 240 px, 25 µm, of the field.

**Aberrations.**  One field term is clear and the same in all three acquisitions: *linear astigmatism*,
Noll 6 growing along x and Noll 5 along y, 6–9 nm of path per half-field -- about 0.09 rad at 600 nm, 0.13 rad
in a corner.  That is NAT's misalignment term (b·H), not the quadratic one.  Coma and spherical aberration show
no consistent gradient (≤ 2.5 nm per half-field where the acquisition is clean).  What the plane leaves, 1–5 nm
from bead to bead (the CRLBs are 0.1–0.2 nm), is scatter, not a field.

**Localisation with one model.**  The part of each bead's slice error that is a plane over the field is
1–4 nm laterally and 3–10 nm in z (rms over the beads within ±600 nm); a plane fitted to the same errors with the
beads' positions shuffled gives 1–3 and 2–12 nm.  The field is detectable (p < 0.05 in about half the
acquisitions and axes), not large: the slope at the edge of the half-field is 1–6 nm laterally and 4–13 nm in z,
while the beads scatter by 7–35 nm in z for reasons that have nothing to do with where they are.  One outlier,
`40nm_match_3D_0.05um`, scatters by 57 nm in z, a property of that acquisition, not of the field.

**A grid for this field.**  Linear astigmatism of 0.09 rad per half-field, `field_interpolation.py`'s measure:
one PSF 7.4 / 26 (mean / max excess χ²), 2 × 2 nodes blended 0.11 / 0.33, 3 × 3 0.007 / 0.022, 5 × 5 below
0.001.  A 3 × 3 grid is the exact model for M5's present fields.

**Defocus over the field** -- field curvature and a tilted focal plane -- cannot be read from these results:
each bead's z is fitted, so it absorbs both.  Which is the point for learning: with beads, a focal surface costs
nothing and needs no model.  In the fitted localisations a tilt of the focal plane is a shear of z over the
field, z' = z + a·x + b·y: harmless for structure (distances along the coverslip change by 1 − cos of the tilt),
visible only where absolute z across the field matters.  Two places where it does: a depth-dependent PSF
(in situ with an index mismatch), whose depth is then off by the tilt times the distance -- 0.1° over 25 µm is
44 nm, well below what changes the PSF; a degree is 440 nm and starts to -- and a z range near its end.  A finer
grid cannot take a focal surface: grid PSFs shifted in z do not blend into a shifted PSF; a z offset map,
applied to the fitted z, is exact and costs nothing.  So it is left out until fields grow or in situ needs it.

## 8. What is built (v1, November 2026)

Decided with the user: beads only, one channel, no gridded residual and no focal surface yet; *field
dependence* is a tick in the PSF calibration window, and the run reports whether the field is real.

- **Learning:** `FieldPolynomial` (`src/uipsf/model/field.py`) -- not NAT's coupled form but a polynomial of
  each mode: astigmatism to second order, coma to first, 14 coefficients.  NAT's form is exact only in the
  frame where the pupil's axes and the field's agree in handedness; a mirror in the detection path
  conjugates one against the other, and the polynomial holds NAT's terms in any frame.
- **The report:** `diagnostics.field_report`, a Wald test of the coefficients and each mode's largest change
  over the beads, in the window's summary.
- **Export and fitting:** node splines on the smallest grid that keeps the blend within 0.5 % of the PSF's
  peak (one node when the field is flat), blended per spot in C++, 1.7× the time of one spline.

`docs/uipsf/architecture.md`, "Field dependence", has the details and the measurements.

## 9. Zenodo `1ch_FD_bead` (November 2026)

`benchmarks/uipsf/real/zenodo_fd_beads.py`, `results/zenodo_fd.json`.  Two stacks of astigmatic beads over a
1608 × 1608 px Prime 95B chip (177 µm), 10 nm steps over 2 µm; 239 beads found.  v1's optics (NA 1.5, 680 nm,
immersion 1.515, medium 1.406, 110 nm pixels) and conversion, 0.47 e⁻/ADU: the files do not record it; the
residuals' photon transfer gave 1.08 with the field on a first set of beads and 0.62 on the second, so it is
not well pinned.  60 beads spread over the chip were learnt from, in chunks of 8 (60 at once took more than
15 GB: the solver's chunks were sized for speed, not memory -- since capped by the memory available on
every system, `solve.problem._host_chunk_values`).

| | without field | with field |
|---|---|---|
| beads used | 52 of 60 | 51 of 60 |
| deviance per pixel | 2.30 | 1.36 |
| blur σ (x, y) | 93, 117 nm | 75, 102 nm |
| time (4 cores) | 12.6 min | 30.9 min |

The report: significant, 935 000 nats over 14 coefficients (threshold 27); over the beads' 170 µm astigmatism
changes by up to 1.4 rad (Noll 5) and 0.7 rad (Noll 6), coma by 0.3 rad.  **The learnt field is NAT's,
unconstrained:** centred on the chip (808, 809 px), the linear terms vanish, Noll 6 = 0.27 u² − 0.22 v² and
Noll 5 = 0.51 uv, astigmatism ≈ 0.26 H² with a real coefficient; Noll 8 = 0.18 u and Noll 7 = 0.16 v, coma
≈ 0.17 H (rad, u and v in units of 50 µm).  v1's per-bead maps of the same data show the same pattern.  The
export made a 9 × 9 grid, the largest of `GRIDS`.

**Localisation**: every slice within ±600 nm of every bead, the 188 not learnt from included; z error rms per
bead after its own height, median, nm:

| spline | all | 0–40 µm from the centre | 40–70 µm | 70–120 µm |
|---|---|---|---|---|
| field (node splines) | **36** | **25** | **27** | **51** |
| the field model's centre spline | 92 | 29 | 66 | 160 |
| learnt without the field | 97 | 35 | 76 | 137 |

What is left at the edges is the model, not the grid: in a first run (whose beads missed the chip's right side)
a spline drawn exactly at each far bead did no better than the 9 × 9 blend (86 against 85 nm), though the blend
misses the PSF by 2.7 % of its peak at 9 × 9 (1.3 % at 13 × 13), above `TOLERANCE`; the export now says so when it
stops at the largest grid (`field_splines.limit_warning`, kept as the calibration's ``field_warning``).
Candidates: second-order terms (Droste's), a field-dependent magnitude
(vignetting), the conversion.

## 10. Open

- Whether the residual's prior keeps a bead calibration of 20–50 beads stable (the residual is not built).
- What limits the edges of a 177 µm field (§9): second-order terms, a field-dependent magnitude.
