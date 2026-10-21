# uipsf roadmap: what is done, what is next, what could come

October 2026.  The short version of `docs/uipsf/plan.md`: the plan holds the reasoning, the decisions and the
milestones' detail, `docs/uipsf/architecture.md` what is built and measured, and
`docs/uipsf/report-2026-10.md` the study of what the learnt blur and pupil magnitude are.  Update this page
when a milestone lands or an item moves.

## Status at a glance

| milestone | what | state |
| --- | --- | --- |
| M0 | scaffold | done |
| M1 | forward model (vectorial, matrix DFT) | done: matches uiPSF v1 and the TU Delft model |
| M2 | likelihood and solver | done: synthetic beads to their CRLB without settings |
| M3 | data, camera, rejection, residual atlas | done |
| M4 | several channels | done: transform to 0.025 px of SMAPpy's and v1's |
| M7 | speed | done: Zenodo 1ch (84 beads, order 8) in 45 s on a Mac GPU, 2ch in 1.4 min |
| M6 | SMAPpy integration and release | mostly done: window, adapter, spline, acceptance tests; open below |
| M5 | in situ (blinking molecules) | not started |
| M8 | physics where the atlas asks for it | ongoing: measured, not yet modelled |

## Done

- **In SMAPpy's repository** (plan §7.4): `src/uipsf/`, torch as the extra `smappy-smlm[uipsf]`; the v1
  bridge (TensorFlow, subprocess) is gone.
- **smappy reads, uipsf fits** (`smappy.calibrate.pupil`): stacks, camera from the database, photons,
  bead detection and pairing are SMAPpy's; uipsf takes a `Batch`.  Bead picking ignores faint specks and
  judges a pair by its light in both channels.
- **The *PSF calibration* window** (`Localize/PSF calibration`): a *method* switch, bead average or pupil
  model; *start from* a previous model; *save model*; the spline (or dual calibration) and the model file
  saved beside the stacks; fit quality by Spline 3D (2C for pairs) with z-against-frame curves that select
  a bead; *Install PyTorch* when torch is missing.  The headless path is `pupil.run_calibration`.
- **The window in two steps** (October 2026): the stacks shown on load (file and plane sliders); *Detect
  beads* rings them and estimates the learning time (`pupil.estimate_seconds`, one Gauss–Newton matrix
  timed); *Calibrate* with a progress bar and the time left; the fit quality and *Model vs data*
  (`calibrate.model_views`: the beads aligned onto the model by their fitted geometry, data, model and
  residual on one scale) after every run, for both methods; Zernikes named by short abbreviations.
- **How a bead fit starts** (October 2026, plan §10, 5): measured; a previous model starts stage 1, a
  pair's transform starts from the beads' Gaussian positions (`model/astig.py`).
- **M6 acceptance:** z slope 0.999 through the whole chain, photons 0.2 % from the bead average's; Spline
  3D and uipsf's own localisation 0.6 nm rms apart on Zenodo.
- **Studies** (report-2026-10.md): the model choices measured; σ and the magnitude on Zenodo and on the
  lab's M2 and M5; pixel sampling against integration; a comparison with Chromatix.

## To do, in order

1. **Decide the model choices** (measured in plan §10, 7): Zernike order 8, magnitude free only, no blur
   choice, bleaching of order 2 after the first fit, slice thinning 100 nm.  Then apply them to
   `BeadSettings` and the form.
2. **Microscope database** (plan §7.5): optics per microscope chosen by the file's metadata; the pixel
   size following the 3D lens (`camera_db` can map a tag already); read noise per readout mode; M2's
   iXon filed under M2; M5's pixel settled.  Needs the lab's metadata (`docs/uipsf/microscopes.md`).
3. **Retire *Bead calibration*:** remove the old window and plugin, move the default GUI and the tutorial
   storyboard to *PSF calibration* (M6).
4. **Fix: 2ch stalls** with the window's default indices (1.518 / 1.33) and converges with v1's 1.516 / 1.35.
   Stage 1 stops after 4 iterations, 22 M nats up; a transform started from the beads' Gaussian positions
   avoids it in these runs (plan §10, 5), the stall itself remains.
5. **Drift per stack and file** in the bead model (plan §10, 7): a block per file.
6. **Explain the blur σ** (report: 40–83 nm on M2, 0–32 on M5, cause open): where a blur-free model misses
   (core, wings, rings); a one-sided transfer trail on EMCCDs; at M2, beads at several clock settings.
7. **Model the magnitude physically:** an effective NA plus supercritical transmission instead of free
   Zernikes, after a back-focal-plane image or beads in index-matched medium separate the two.
8. **The z slope of 1.011** against the stage on Zenodo 1ch: model or stage.
9. **In situ (M5):** learning from blinking molecules, depth and stage tilt (`FocalSurface`); the
   window's *source* switch.  Zenodo `1ch_smlm_TP`, `1ch_smlm_tubulin`.
10. **Release (M6):** the real benchmark against tuned v1 with no per-dataset settings, CUDA and Windows GPU
    runs, uipsf documentation, 2.0.

## Ideas: what could come

### Excitation PSFs (uiPSF review §C)

The pupil, Zernikes, Fresnel terms, bead kernel, Poisson likelihood, nuisance parameters and the
multi-channel machinery carry over; the forward model and the data's geometry change.

- **Confocal and ISM:** the focused excitation field (reversed apodisation; the input beam's fill,
  decentration and polarisation; a known phase mask such as a vortex, a top-hat or the SLM pattern; an
  incoherent floor ε), times detection convolved with the pinhole.  For SPAD-array ISM and ISM-FLUX each
  detector element has its own pinhole offset: uipsf's multi-channel model with a shared excitation.
- **MINFLUX and STED beams from bead scans:** 20–40 nm beads scanned through focus give an (x, y, z)
  raster, the bead pipeline's own data.  New: scan pixels (no pixel integration, motion blur along the fast
  axis), scanner distortion, SPAD dead time.  The bead size matters far more: a 100 nm bead fills the
  donut's zero.  Gold beads in scattering are the alignment standard; with a pinhole they need a coherent
  model.
- **MINFLUX in situ:** the beam's Zernikes, ε, polarisation, fill and beam-steering calibration fitted
  jointly with the molecules from the photon counts at the targeted beam positions.  No published in-situ
  phase retrieval for MINFLUX or STED beams was found: the most novel item here.
- **What to design for:** fit raw counts (the zero carries the information); take a z-scan (Φ and −Φ(−k)
  look alike without it); top-hat donuts are very sensitive to depth and spherical aberration; saturation at
  the beam's crest.
- Related work in the lab: `MINFLUXexcitation`, `SimuFLUX` (Marin & Ries 2025, the ε model).

### Other modalities

- **4Pi:** a phase axis inside the channels (Zenodo `4pi_40nm_bead`, `4pi_smlm_nup96`).
- **Lattice light-sheet:** a voxel model as the baseline (Zenodo `1ch_LLS_bead`).
- **Polarisation:** a Jones pupil (2 × 2) for birefringence and a dichroic's s/p transmission.

### Phase masks in the pupil (engineered PSFs, donuts)

Masks with phase jumps cannot be written as Zernikes, so they are a pupil component of their own: the pupil
phase is the mask's, placed on the pupil by a few fitted parameters, plus the Zernikes learnt on top of it as
the system's aberrations (noted October 2026).  The same component serves detection (engineered PSFs for 3D)
and excitation (the donuts of MINFLUX and STED, above).

- **Shipped, pre-calculated:** the double helix, the 2D vortex (e^{iφ}, charge 1) and the 3D top-hat vortex
  (a π step at a radius r_th); the tetrapod and saddle point are smooth enough for Zernikes but come as masks
  too, since they need high orders.
- **The user's own:** a phase image (PNG, TIFF or NumPy, with its scale relative to the pupil) or a formula
  Φ(ρ, φ), stored with the model file and with the microscope's entry in the database, so every calibration
  of that microscope starts from it.
- **Fitted parameters:**
  - its centre (x, y) on the pupil: the mask's misalignment;
  - its rotation, which turns a double helix;
  - a phase factor α (Φ → αΦ): a mask made for another wavelength does not delay by exactly π;
  - a scale: the mask's size against the pupil's, when the relay's magnification is not known exactly or the
    mask does not fill the pupil;
  - for the top-hat, its step radius r_th.
- **To work out:**
  - *Gradients through a jump.* The derivative of a step with respect to its position is a line, not a
    number on a grid: the mask's edges are drawn anti-aliased (each pupil sample takes the area-weighted
    mean of exp(iΦ)), so centre, scale and rotation get finite derivatives.  The pupil grid may need more
    than 64 samples to place an edge to a fraction of the aperture.
  - *Phase wraps.* Written as a transmission exp(iαΦ), a vortex is smooth except at its centre; with α ≠ 1
    its 2π wrap line becomes a real step, which is what an SLM's wrap does.
  - *What trades with what.* A decentred mask looks like coma, a scale error like defocus and spherical
    aberration, α like a share of the light left in the zero.  Fit the mask's parameters first with the
    low Zernikes, then free the rest (the two-stage fit of beads, generalised); the covariance shows what is
    left entangled.
  - *A mask off the pupil plane* (a relay not quite conjugate) is a propagation, not a phase: out of scope
    until data ask for it.
  - *SMAPpy:* the double helix's large, rotating spot needs a wider bead box and spline; the window's
    *model* section gets a *phase mask* choice, from the microscope database once it exists.

### Data and scope

- **In situ with several channels**, then field dependent (Zenodo `2ch_smlm_nup96_WGA`,
  `1ch_FD_smlm_nup96`).
- **Field dependence:** `NodalPolynomial` (the NAT term table, Droste et al. 2025), a B-spline residual, the
  axis centre and the focal surface; Zenodo `1ch_FD_bead`.  (uiPSF v1's map export to FD-DeepLoc is
  misregistered by n/(n−1), review §E.)

### Physics (M8 and this session's studies)

- Depth-integrated bead; the polystyrene bead's own index near the coverslip.
- Full Gibson–Lanni with a coverslip offset.
- Polychromatic emission (5–11 nm of σ measured).
- The aperture edge done properly: anti-aliased, or a fitted soft cut-off.
- Saturation I/(1 + I/I_sat); a per-slice background; exact EMCCD noise.
- A one-sided charge-transfer kernel for EMCCDs.

### SMAPpy uses of a model

- Fitting with the pupil model directly, or per-tile splines chosen by field position (plan §7.3), for field
  dependence and in situ.
- sCMOS per-pixel noise maps, in uipsf's likelihood and SMAPpy's fitter.
- A microscope's history: each calibration's model kept in the microscope database, the Zernikes drawn
  against earlier ones, for diagnosis over time.

### Infrastructure

- A standalone uipsf: a `smappy-core` package without PySide6.

### Considered and set aside

- FD-DeepLoc export (dropped).
- Learning without Zernikes: the real-space voxel PSF and the pixel ("image") pupil (to revisit).
