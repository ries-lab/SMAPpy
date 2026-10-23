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
| M5 | in situ (blinking molecules) | designed (plan §9, M5); steps 1–6 done: the fit on cut regions, the twin test, linked blinks, simulated movies, detection and selection, the rounds and the coverslip edge, the window's *in situ* method |
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
   `docs/microscope-database.md`: built and wired in (`microscope_db`, the EMU importer, M2 and M5
   shipped, cameras named for themselves in the same file; the lens-tied pixel over the file's,
   *PSF calibration*'s optics on *auto*, a calibration's microscope state and the fit's warning);
   next, the found-settings line where files are opened, manual elements (the splitting dichroic),
   then the manager from the menu.
3. **Retire *Bead calibration*:** the default GUI and the tutorials (`fitting_3d`, `two_colour_3d`) use *PSF
   calibration* (October 2026); the plugin and its window stay in the tree for now, to be removed (M6).
4. **Done: the 2ch stall** (architecture.md, "Why Zenodo 2ch stalled").  Not the indices: the start put
   50 of 88 dim channel-0 backgrounds at softplus' 10⁻³ floor, the saturation guard then shortened every
   parameter's step for three of them (α = 2·10⁻⁴) until the step test stopped stage 1, and `torch.polar`'s
   backward gave a wrong gradient wherever the pupil magnitude is negative.  Fixed in the start (a floor
   of one photon), the guard (only the saturating components are shortened) and the pupil; every start
   now reaches 2.98239 M, 52 000 nats below the old "best", cold in 66 s.  n_med 1.33 fits 31 000 nats
   better than 1.35.
5. **Drift per stack and file** in the bead model (plan §10, 7): a block per file.
6. **Explain the blur σ** (report: 40–83 nm on M2, 0–32 on M5, cause open): where a blur-free model misses
   (core, wings, rings); a one-sided transfer trail on EMCCDs; at M2, beads at several clock settings.
7. **Model the magnitude physically:** an effective NA plus supercritical transmission instead of free
   Zernikes, after a back-focal-plane image or beads in index-matched medium separate the two.
8. **The z slope of 1.011** against the stage on Zenodo 1ch: model or stage.
9. **In situ (M5), designed** (plan §9, M5, with a review of v1's): inclusive detection at several widths
   on absolute significance; an elliptical Gaussian on every candidate for photons and a z proxy; overlaps
   by the spot's size, rejected or masked; blinks linked; per z bin a random draw from a brightness window
   a factor 3 wide below the pairs' tail; ~10 % held out; v2's fit with the focus depth under a loose
   prior and the coverslip's edge; rounds from the whole pool until stable; the sign explicit > data (twin
   test) > database.  Astigmatic and 2D first; the window's *source* switch.  **Step 1 done**
   (`InSituGeometry`, `workflows/insitu.py`, architecture.md): the fit on cut regions and the twin test.
   **Step 2 done:** linked blinks, 1.2–3× as many emitters placed 300–1000 nm from focus.  Next simulated
   movies, detection and selection, the rounds, the window.  **Step 3 done:** blinking movies imaged with
   the model (`uipsf/movie.py`), SMAPpy's blinking, a truth table; next detection and selection.  To
   revisit: the molecules' places from the simulation plugin's ground truth (`smappy.simulate.ground_truth`:
   structures, labelling, linkage, drift) and its camera, instead of the movie's uniform placement.
   **Step 4 done:** detection and selection (`smappy/calibrate/insitu.py`): from a simulated movie to a
   pupil within its CRLB.  Frames are fitted, summing a blink's frames an option.  **Step 5 done:** the rounds
   and the coverslip edge (`workflows/insitu_rounds.py`): a focus started 250 nm off back within a few nm
   with molecules on the coverslip, ~25 nm without.  **Step 6 done:** *in situ (blinking molecules)*, the PSF
   calibration window's third method (movies, the section *in situ*, the focus typed or from the data, its
   pages, the spline above focus or above the coverslip).  **Zenodo `1ch_smlm_tubulin` against v1:** every
   mode within 0.05 rad, 0.018 rad rms, in 4 min on Apple's GPU (v1: 39 min on the CPU); the blur σ twice
   v1's, open: measured to be in the in-focus spots, not aliasing, the sample index, our learning or the
   wings; next an NA scan (architecture.md, "Why in situ's blur σ is twice v1's").  Detection on dense
   real data: false detections on background still count as neighbours, and a faint patch splits into
   several detections that reject each other.  A neighbour now counts only with a fifth of the molecule's
   light: 9 % fewer overlap rejections on tubulin, a third more candidates, the pupil unchanged.  **M2 with the
   same day's beads** (architecture.md, "In situ on M2"), a mismatched pair: the movie was taken without the
   cylindrical lens (its metadata: 3DA out, the beads' in), its bright spots round (log₂(σx/σy) −0.08 to +0.13,
   beads −0.57 to +0.70); to be repeated on 260821's 3D movie and beads, both with the lens in.  As measured: the beads' whole magnitude held costs 1–2 nats per
   molecule, its symmetric part nothing, neither brings σ to the beads'; a free magnitude trades its
   astigmatism with the phase's; the focus is barely determined.  **Matched M2 pairs** (architecture.md, "In situ on matched M2
   pairs"): on clean astigmatic data in situ is robust (starts within ~0.5 nats per molecule, the focus
   settles) and beats the bead model by 2–7 nats per molecule; the beads' symmetric magnitude (stable across
   days and colours, what keeps z straight on beads) held in situ is the steadiest choice; phase order 6 by
   default, order 8 only where disjoint molecule sets agree (cross-validated); DNA-PAINT's free-imager haze
   breaks the flat background, a stricter shape test after round 1 steadies it.  Next: a sloped background,
   the symmetric bead magnitude and the order choice in the PSF calibration window.  Later: summed blinks against frames; a sparse pile-up on the coverslip, where the
   edge is weak; speed (a round ~2 min).  Zenodo `1ch_smlm_tubulin`
   and simulations.  Open, optimised once it runs: what separates depth, n_med and spherical
   aberration.
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
  - a scale: the mask's size against the pupil's.  Not for the vortex, which looks the same at any size.  For
    the others it can be calibrated carefully, so it is fixed by default and freed as a diagnostic: a fitted
    scale far from the calibrated one says the relay or the mask is not what it should be;
  - for the top-hat, its step radius r_th.
- **To work out:**
  - *Gradients through a jump.* The derivative of a step with respect to its position is a line, not a
    number on a grid: the mask's edges are drawn anti-aliased (each pupil sample takes the area-weighted
    mean of exp(iΦ)), so centre, scale and rotation get finite derivatives.  The pupil grid may need more
    than 64 samples to place an edge to a fraction of the aperture.
  - *Phase wraps.* Written as a transmission exp(iαΦ), a vortex is smooth except at its centre; with α ≠ 1
    its 2π wrap line becomes a real step, which is what an SLM's wrap does.
  - *What trades with what.* A decentred mask looks like coma, a scale error like defocus and spherical
    aberration, α like a share of the light left in the zero.  **Separate them by taking two calibrations:**
    one with the mask out, which gives the microscope's own aberrations, then one with the mask in, started
    from the first with its Zernikes held fixed, fitting only the mask's parameters (centre, rotation, α,
    and the scale as a diagnostic).  The window's *start from* gets a *keep its aberrations* tick for this.
    Without a mask-free calibration: fit the mask's parameters with the low Zernikes first, then free the
    rest (the two-stage fit of beads, generalised); the covariance shows what is left entangled.
  - *A mask off the pupil plane* (a relay not quite conjugate) is a propagation, not a phase: out of scope
    until data ask for it.
  - *SMAPpy:* the double helix's large, rotating spot needs a wider bead box and spline; the window's
    *model* section gets a *phase mask* choice, from the microscope database once it exists.

### Data and scope

- **In situ with several channels**, then field dependent (Zenodo `2ch_smlm_nup96_WGA`,
  `1ch_FD_smlm_nup96`).
- **In situ, after M5:** an initial pupil model as the first fit, for PSFs a Gaussian cannot sort by z
  (double helix, tetrapod: Zenodo `1ch_smlm_TP`); the number of emitters per bin from the CRLB; selection
  stratified in the field too; splines at other depths; stage tilt (`FocalSurface`).
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
