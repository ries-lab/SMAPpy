# 4Pi in SMAPpy: what exists, what is wrong with it, and a plan

Status, 2026-11-02: M0-M5 are done (§6-§8, §11, §14): the fitter, the channel
layer (§9), real DCIMG data (§10), learning from beads in uiPSF v2 and the
bead average, both in the PSF calibration window, live DCIMG and simulated 4Pi
cameras (§16).  Of M7 the vectorial model is in and the window's default
(§17, §19), with a start that reaches one optimum (§20), and the window learns one objective's minimal pupil (§23); the locked refit, multicolour and two cameras are not started.  M6
(in situ) waits on the in-situ learning; dual colour on the layer (M8) is
planned, not started.  What is still missing, and a two-colour tubulin data set
for it, is §25 (2026-11-24); the work packages it planned are §26 -- 28.

This page is written from four sources:
- a read of SMAP's 4Pi code (`../SMAP`);
- a read of uiPSF v1's 4Pi learning (`../uiPSF`);
- a read of SMAPpy's own multi-channel code;
- the papers listed under [References](#references).

The point is a 4Pi path that works, not a port.  Where SMAP or v1 did something
fragile, the plan says what replaces it and why.

## 1. The method in one paragraph

**Two objectives, four images.**  Two opposing objectives collect the light of
one molecule.  The two beams interfere.  The microscope splits the result into
four images, p1, s1, p2 and s2, whose interference phases differ by roughly
0, π/2, π and 3π/2.

**What the phase gives.**  The relative intensity of the four images gives the
interference phase Φ, which changes by 2π every fringe period z_T.  z_T is
about 260 nm at NA 1.35 and λ 600 nm, longer than λ/2n because of the high NA.
So Φ gives z with a precision several times better than the PSF shape can,
but only modulo z_T.

**What the envelope gives.**  The PSF envelope (its width, or its astigmatism)
gives a coarse z that has no such ambiguity.  Unwrapping picks the fringe that
agrees with it.  The cavity phase drifts with time (0.04 rad/s has been
reported), so the offset between Φ and the coarse z has to be followed through
the measurement.

**The fitter's model.**  The model that makes a global fit of the four
channels possible is the IAB model (Li et al. 2020).  It is a cubic spline
for each of three volumes I, A and B:

    μ_c(x, y, z, Φ) = N · [ I_c(x,y,z) + A_c(x,y,z)·cos Φ + B_c(x,y,z)·sin Φ ] + b_c

- I_c is the incoherent part of channel c.
- A_c and B_c are the interference part, demodulated by the fringe, so they
  vary slowly in z and can be splined.
- z and Φ are separate fit parameters.  The fit returns an envelope z and a
  wrapped phase, and unwrapping is a separate step.

## 2. What exists

### SMAP

SMAP has two 4Pi routes that share almost no code.

**The native route** (`calibrate_4pi_v2` → `MLE_global_spline` 4pi mode →
`CPUmleFit_LM_MultiChannel_4pi`) **cannot run to the end**:
- `calibrate_4pi_v2` uses `shiftedstack`, `allPSFs` and `out.transformation`
  after the code that defines them was commented out (v2:65–67, 100–101,
  179–206).
- The multi-bead `IABfrom4PiPSFfitmany` crashes on an undefined `zshift0h`
  (L70) and indexes a 4-D array with three subscripts, so it always takes
  bead 1 (L158).
- `RegisterLocs4Pi` passes no `Tmode`, so it errors.
- The legacy fitter path adds the stack centre to `zstart` twice, appends a
  seventh entry to a six-entry link vector, and divides the data by `normf`
  before a Poisson fit.

**The uiPSF route** is the one that works: `learnPSF_invmodeling` → uiPSF v1
h5 → `MLE_4Pi` → the closed-source `CPU/GPUmleFit_LM_4Pi_v1`, then
`Channeldrift4Pi`, then `Phase2z4Pi`.  Its weaknesses:
- **The fitter is binary-only.**  Its source is not public; GlobLoc's public
  repository has only the non-4Pi multi-channel source.  The GPU build is
  Windows-only.
- **Phase starts cover only part of the circle.**  They are
  `[0, 1/3, 2/3]·π`, a third of it, where `2π·[0, 1/3, 2/3]` was surely meant
  (`resetparam_4Pi_fit.m:28`).  The GUI's z start is ignored.  The sCMOS
  variance is computed and never passed to the fitter.
- **The peak combiner adds channel 1's drift twice** (L226, L271).  It applies
  `inv(T)` in both directions, weights by `phot` here but by `sqrt(phot)` in
  the other branch, and its result depends on the order the channels are
  merged in.
- **The combined columns are wrong.**  They weight by 1/σ but compute the
  error as if the weights were 1/σ², and the error uses channels 1 and 2 only
  (`MLE_4Pi.m:151`).  `phot` comes out as a weighted mean of the channels, not
  a sum.
- **`Phase2z4Pi` makes a hard decision.**
  - It rounds to the nearest fringe and ignores the precision of the coarse z.
    It gives no probability of having picked the wrong fringe, and the error
    it reports ignores that possibility.
  - It flips signs in several places: the fitter flips z but not Φ, and the
    result is negated at the end.
  - It applies the first file's calibration to every file, needs a single
    `phase` column, and drops the frames on window boundaries.
- **The spline z-centre is defined five different ways** across the files:
  `nz/2`, `ceil((nz+2)/2)`, `round(nz/2)`, `ceil(nz/2+0.5)`, and a magic
  `−0.7`.

### uiPSF v1

**The model** is a scalar pupil model:
- one pupil per objective;
- the two fields are summed coherently with a per-bead phase, a shared phase
  for each deformable-mirror step, and a channel offset fixed at a multiple
  of π/2;
- a coherence fraction α for each channel.

**Export** evaluates the model at three phases, demodulates the result into
I, A and B by a fringe period z_T, splines each, and writes them into
`locres/coeff`, shaped `[C, 3 (A, B, I), 64, z, y, x]`.  The channel offsets
and channel brightness go into each channel's A, B and I, so the fitter gets
φ_c = 0.  This is the convention to keep.

**Weaknesses:**
- **z_T is typed in, not derived.**  It is a user constant (0.26 µm), not
  measured from the model's own phase slope.  A wrong z_T biases both the
  demodulation and z.
- **The model is over-parametrised.**  Each channel has its own relative tip,
  tilt and defocus between the objectives, although that misalignment is
  physically common to all channels.  The per-bead phase, the mirror phase and
  the per-channel piston all trade against each other, and no gauge is fixed.
- **The objective misalignment is learned but never used.**  `pos_d` is a
  learned variable that never enters the forward model, and it is exported
  anyway.
- **Channel positions ignore mirroring.**  The subpixel position in channel c
  is the reference position plus a shift of the ROI centre.  It never goes
  through the linear part of the transform, so it is wrong for a mirrored or
  rotated channel.
- **`coeff_reverse` is not a true reversal.**  A is flipped in z without being
  conjugated.
- **The photon scale is ambiguous.**  Each channel's I sums to about 0.5 per
  plane, so N is roughly half the detected photons.

### SMAPpy

**The C++ global fitter already takes four channels.**  `csrc/global.hpp`
(`MAX_CHANNELS = 4`) is a GlobLoc port.  Its parameter link
`θ_c = global·factor + offset` can share Φ across the channels and can hold a
per-channel phase offset, should we need one.

**Everything around the fitter assumes two halves of one chip:**
- `dualfit.py`: `combine_peaks`, `build_link` and `cut_paired_rois`;
- `detect.PeakFinder.split`;
- `DualColorCalibration`, with `main` and `secondary`;
- `psf._GlobalPSF`, which fixes five parameters;
- `_TwoChannelFit` and `simulate/camera.render`;
- `microscope_db.SIDES`;
- `io/tiff.ImageSource`, which is one 2-D frame series.

**uiPSF v2 can learn many channels** (`ForwardModel`, `Channel` with a 3×3
transform, a photon share and a defocus).  It has no interference and no
phase axis.

### The data

**Ries-lab 4Pi, `Fig2_4pi_dataset`.**  This is zenodo 10823308; the same
beads are in uiPSF's 8267520.
- NA 1.35 silicone objectives, n 1.406 matched, λ 600 nm, 129 nm pixels,
  sCMOS.
- **`40nm_600/`** holds 26 bead files.  Each `.mat` (HDF5) contains `qd1..qd4`,
  shaped (z 21 × phase 3 × 200 × 200): 50 nm steps and three mirror phase
  steps.
- **`cell3_nup96_mMaple_045/`** holds 69 × 234 MB of Nup96 4Pi SMLM.
- uiPSF's record also has v1's result files, among them
  `results/4pi_40nm_bead/psfmodel_zernike_4pi.h5`, with I, A and B splines in
  `locres/coeff`.  These are a ready-made model for a first fitter, and a
  regression target.

**Bewersdorf-design astigmatic 4Pi, zenodo 14287640.**  It holds 61 DCIMG
frames, 2048 × 168 pixels, with four images side by side on the strip and
20 nm steps.  It has no phase steps and no metadata.  It needs a DCIMG reader,
which SMAPpy does not have.  This data is a target, not only a test case: the
Ries-lab 4Pi writes the same kind of file (§5).

### The 4Pi fitter's source

The C/CUDA source of the 4Pi fitter is not available.  Everything that was
found is binary:
- SMAP's `CPUmleFit_LM_4Pi_v1` / `_nIAB` mex files;
- GlobLoc's `CPU/GPUmleFit_LM_4Pi.dll`;
- uiPSF v1's `libCPUmleFit_LM_4Pi.dylib`.

GlobLoc's public C++ and CUDA hold only the non-4Pi fitters.

**The fitter is written from the IAB model itself** (Li et al. 2020) and from
v1's export convention, not from any of the binaries.  With θ = (x, y, N, bg,
z, Φ), each channel evaluates its own I, A and B splines, and Φ is shared:

    μ_c = bg_c + N·(I_c + A_c·cos Φ + B_c·sin Φ)

The derivatives follow from the spline, and ∂μ/∂Φ = N(−A_c sin Φ +
B_c cos Φ).

**A black-box reference runs on a Mac.**  uiPSF v1's dylib is a universal
binary that includes arm64 and exports `CPUmleFit_LM_4Pi`.  A benchmark can
fit the same spots with it and with SMAPpy's fitter and compare the two.  This
is a script, not a test the suite depends on.

## 3. Design decisions

### D1. One model in the fitter: per-channel I, A and B, the phase offsets inside them

The fitter's model is exactly the one in §1.  It has no φ_c, no channel ratio
and no Dz of its own, because all of those live in each channel's splines:
- **It is general.**  A channel offset that is not exactly π/2, an arm
  imbalance, or an s-channel with less modulation (Bates et al. 2022 show that
  an unequal channel amplitude costs 5–10 nm of z bias) are all just
  different splines.
- **It matches v1's export,** so v1's models load unchanged.

The link in `global.hpp` still allows a per-channel phase offset δφ_c, kept
for a later per-channel phase drift.  It is zero by default.

### D2. Parameters, photon scale, columns

The fit parameters are x, y, z, N, Φ and b_c:
- x, y, z, N and Φ are shared by the channels;
- b_c is free per channel;
- any of them can be unlinked, as in Gaussian 2D 2C and Spline 3D 2C.  Linking
  N gains up to 1.5× in z (Li et al. 2022).

**The calibration is normalised** so that Σ_c Σ_xy I_c = 1 in the focal plane,
which makes N the total detected photons.  This fixes v1's factor of 2.  v1
files are renormalised when they are loaded.

**Columns:**
- `x_nm`, `y_nm`, `photons`;
- `bg_ch0..3`;
- `z_env_nm`: the envelope z;
- `phase_rad` and `phase_err_rad`: wrapped to (−π, π];
- `z_nm`: the envelope z until unwrapping replaces it;
- precisions as usual;
- with x and y unlinked, `x_nm_ch1..3`, and so on.

**The spline's z origin has one definition,** that of `SplineCalibration.z0`,
used everywhere.  The direction of z and the sign of Φ are fixed by a test on
a simulation: a molecule moved by +δz must give Φ moved by +2π·δz/z_T.

### D3. Starting values without a blind multi-start

The model is **linear in N, N·cos Φ and N·sin Φ** once x, y and z are fixed.
So for each z on a coarse grid we can do the following:
- At the peak's centre, solve the 3 × (4 channels × pixels) linear least
  squares for those three numbers.  This needs a weighted fit, the
  "phasor" start.
- Φ = atan2 of the two interference terms, and N comes out directly.
- Pick the z with the smallest residual.
- Start the LM from the best one or two z values.

Every Φ is covered with no phase grid, and the starts fall on the right fringe
far more often than a ±150 nm z grid does.  This replaces SMAP's 0–120° phase
grid and v1's {0, π} × {±150 nm}.

A fallback multi-start over Φ ∈ {0, 2π/3, 4π/3} remains as a setting, so the
two can be compared on simulations.

### D4. Unwrapping by likelihood

The fit gives z_env (with σ_env) and Φ (with σ_Φ).  With the phase offset
Φ0(t), the candidate positions are

    z_n = z_T·(Φ + 2πn − Φ0(t))/2π

**Each candidate gets a probability** from the Gaussian likelihood of
z_env − z_n under σ_env:
- the most probable candidate is the answer, `z_nm`;
- its probability is a column, `fringe_prob`, so a user can filter out
  ambiguous localizations;
- `z_err_nm` comes from σ_Φ.

**Φ0(t) is estimated in time windows** by a weighted circular mean of
Φ − 2π·z_env/z_T, which is a von Mises maximum-likelihood estimate.  It uses
only well-determined localizations and is smoothed over time.

**z_T comes from the calibration.**  It can optionally be refined from the
data, as the slope of Φ against z_env over all localizations.  A
refractive-index or wavelength scale factor is a setting; it is 1 for
matched indices, as in the Ries-lab data.

**Diagnostics:**
- a 2-D histogram of Φ − Φ0 against z_env, with the fringes drawn on it;
- Φ0(t);
- the expected ghost fraction.

**Later, a "locked" refit:** fix Φ = 2π·z/z_T + Φ0(t), so the fit has five
parameters, and fit from one start per fringe, keeping the best likelihood.
This is how PR-4Pi and Bates et al. choose the fringe.  It uses all of the
PSF's shape information and gives fewer ghosts at large |z|, so it is the
upgrade once Φ0(t) is known.

### D5. Channel geometry: N channels, of which dual colour is the case N = 2

One `ChannelLayout` describes, for each channel c:
- where its pixels come from: a rectangle on one camera's chip, a separate
  TIFF stack, or a plane of a multi-D TIFF;
- a mirror flag;
- a 3×3 projective map onto channel 0.

**`dualfit`'s halves become a two-channel layout,** and the engine becomes a
`MultiChannelEngine`:
- peak combining loops over C − 1 maps;
- ROI cutting loops over C;
- the link has shape (n, 2, C, P).

**Dual colour must give identical numbers before and after.**  This refactor
is the riskiest step for regressions, so it is done behind the existing
two-channel tests, with a bit-for-bit comparison on a saved run.

**Detection for 4Pi runs on the sum of the four channels mapped onto
channel 0.**  The interference cancels in the sum, the signal-to-noise ratio
is about twice that of one channel, and nothing has to be paired.  This is
an option of the layout, so a two-channel fit can use it too.

**`PeakFinder.split`** becomes a per-channel label rather than an axis and a
position.

**The microscope database** gets channel regions: a rectangle and a mirror
flag in place of a side.  A `4Pi` microscope entry describes the quadrants.

**Input goes through one preprocessing front end** that turns whatever the
microscope wrote into a common multi-channel source.  Downstream code
(detection, the fitters, the bead calibration, uiPSF) sees only that source:
- **The source:** frames of shape (n, C, h, w), the per-channel camera
  description, and the layout's rectangles and mirrors, so a localization can
  be put back on the chip.
- **The readers behind it:**
  - Hamamatsu DCIMG with all four quadrants in one frame, including the
    newer `0x1000000` version and its 4-pixel correction.  This is what the
    Ries-lab and Bewersdorf-design 4Pi microscopes write.
  - Rectangles of a TIFF.
  - Four separate stacks.
  - A multi-D TIFF with a channel axis.
  - The already-cut `qd1..qd4` `.mat` of the example data.
- **Reading is lazy.**  Frames are cut on the fly, never written out as a new
  file, so a live fit can follow a growing acquisition.

**One camera** is implemented first.  `CameraMetadata` per channel is designed
in from the start, because some 4Pi set-ups use two cameras.

### D6. Registration and channel drift

**Bead registration** is the existing robust projective fit
(`calibrate/dual.fit_dual_transform`), run for each channel onto channel 0.
The beads are found on the z-projection, summed over the phase steps,
because the interference makes a bead look different in each channel at any
one z.  uiPSF v2 refines the maps jointly with the model.

**Channel drift** (SMAP's `Channeldrift4Pi`, done properly):
- fit in frame blocks with x and y unlinked;
- take each channel's weighted offset from channel 0, smoothed in time;
- fit again with the drift folded into the link offsets.

This is a step of the 4Pi fitter, not a three-pass routine that drives the
GUI.

### D6a. Coarse z comes from astigmatism

Both target microscopes are astigmatic.  The envelope I_c therefore carries
the astigmatic z information.  z_env is precise, typically tens of nm, and it
is what unwrapping relies on.

The fringe probabilities of D4 use σ_env from the fit's CRLB.  As a check, a
setting can fall back to Huang et al.'s shape metric
m = σx³/σy − σy³/σx from an elliptical Gaussian fit.

### D7. Learning: uiPSF v2 gets a `FourPiPSF`

**The forward model** is

    E_c = t_c^up·P_up·e^{−ikz·z} + t_c^lo·P_lo·e^{+ikz·z}·e^{i(Φ_e + δ_c)}

The intensity is α_c·|E_c|² + (1 − α_c)·(|E_up|² + |E_lo|²), followed by the
bead OTF and the blur v2 already has.

**Parameters:**
- **Aberrations:** one pupil per objective, with Zernikes shared by all
  channels.
- **Misalignment between the objectives:** the relative piston, tip, tilt
  and defocus between the arms, as a few shared physical parameters, not a
  set per channel as in v1.
- **Per channel:** the arm transmissions t_c, the phase offset δ_c (free,
  with δ_0 = 0 as the gauge) and the coherence α_c.
- **Polarisation:** the first version is scalar, as v1 is.  Making it
  vectorial is a later step (M7; the forward model is in, §17).  v2's `VectorPSF` already has the six dipole
  fields and an analyser per channel, so the s and p images could be modelled
  as what they are: different polarisations, with their own PSFs and
  modulation.  With the vector model, part of what α_c and δ_c now absorb
  would become physics instead of free parameters.

**A phase axis:**
- each bead has a phase Φ_e;
- the mirror phase steps of a stack are shared and the first is fixed;
- in situ, each molecule has a free Φ_e.

**Export:**
- sample at three phases and demodulate to I, A and B;
- z_T comes from the model's own phase slope (the unwrapped arg A_c against z,
  fitted), never typed;
- write the per-channel splines, the maps, z_T and α into the 4Pi
  calibration file (D8).

**In situ** uses the SMAPpy 4Pi fitter for its starting values, as v1 used
its C fitter, and detects on the summed channels.  It follows the bead path.

### D8. The calibration file

The file is `format = "smappy-4pi-calibration"`, HDF5, and holds:
- `channel_<c>/{I, A, B}`: coefficients of shape (64, nz, ny, nx), on one
  shared z grid;
- dz and z0;
- the C maps onto channel 0;
- z_T, λ, and the refractive indices;
- the per-channel α and δ_c, kept for display;
- the layout.

**The format generalises.**  The same container with only I is an N-channel
spline calibration, so a later three- or four-colour fit needs no new format.
`DualColorCalibration` stays readable, and one-channel spline files are
unchanged.

**The loader also reads v1's `psfmodel_*_4pi.h5`.**  It renormalises, and
reorders A, B, I into I, A, B.

### D9. The sample between the coverslips: what to hold and what to free (proposed, 2026-10-30)

**The problem.**  In a 4Pi the sample is a sandwich: coverslip 1, a layer of sample (index n_s, gap D,
often about 10 µm) and coverslip 2.  An emitter at height h above coverslip 1 is seen by objective 1
through h of sample and by objective 2 through D − h.  Beads lie on one coverslip, so in a bead
calibration objective 2 alone looks through the whole layer.  Where the sample's index differs from the
immersion's, the layer adds an aberration to that arm.  How far today's Zernikes can take it up (NA 1.35,
silicone 1.406, 600 nm, 10 µm, after refocusing; radial Zernikes fitted to the rest):

| sample index | aberration | left beyond n ≤ 6 | beyond n ≤ 8 | beyond n ≤ 12 |
|---|---|---|---|---|
| 1.406 (matched) | 0 | 0 | 0 | 0 |
| 1.39 | 0.35 rad rms | 0.12 | 0.08 | 0.05 |
| 1.38 | 0.64 | 0.23 | 0.15 | 0.09 |
| 1.37 | 1.00 | 0.39 | 0.26 | 0.15 |
| 1.36 | 1.49 | 0.64 | 0.43 | 0.25 |

The aberration is steepest at the pupil's edge, where low orders fail; bead fits resolve a few
hundredths of a radian.  With water (1.33) the light above NA 1.33 does not cross 10 µm at all
(3 % of the aperture's area), so the far arm works at a lower NA than the near one and the two
interfere only inside it; Zernike magnitudes cannot make that edge.  A deformable mirror correcting the
spherical removes the smooth part, not the edge and not the lost light, and it is set for one depth.
For molecules the layers change with h, in opposite directions in the two arms.  So the layers are
modelled, and the Zernikes keep the instrument and the DM.

**How much each uncertainty matters**, as the wavefront moves after refocusing (the rule: above about
0.05 rad it is known or fitted, below it is held):

| uncertainty | 2 µm of sample | 10 µm of sample |
|---|---|---|
| index +0.005, n_s 1.36 | 0.05 rad | 0.27 rad |
| index +0.005, n_s 1.38 | 0.03 | 0.15 |
| index +0.005, n_s 1.406 | 0.02 | 0.09 |
| thickness +1 µm, n_s 1.36 / 1.38 | 0.15 / 0.06 | the same |
| thickness +100 nm, n_s 1.36 | 0.015 | the same |

The index matters mostly in the far arm; the gap must be known to about 0.3 µm when the index is
mismatched; 100 nm never matters.  The index also sets the phase's z scale: 0.005 is 0.4 % of z.

**The model.**  Each arm's wave is the in situ one (`InSituGeometry`: the height through the sample's
k_z, the focus through the immersion's):

    arm 1:  exp 2πi [k_z,med·h − k_z,imm·f]
    arm 2:  exp 2πi [k_z,med·(D − h) − k_z,imm·(D_imm − f)],  mirrored as now

f is the common focus, what a stage movement changes; the arms' foci stay Δz apart (`fourpi.offset`).
For beads (h = 0, the stage moving the whole sandwich) arm 1 is today's model and arm 2 carries one
constant layer D_bead; with a matched index it vanishes and today's `FourPiModel` is the special case,
which is the regression test.  uiPSF has the pieces: `VectorPSF`'s depth term, `InSituGeometry`'s focus,
`optics/layers.py` for a coverslip's index and thickness.

**The parameters, and where each is held or free:**

| parameter | what it is | beads | sample, in situ | SMLM fit |
|---|---|---|---|---|
| n_s (`Optics.n_med`) | sample index | held, measured | held, measured; checked by the period | held |
| `fourpi.gap` D | sample between the coverslips | held, measured | free, prior from measurement (±1 µm), per field of view | held |
| `geometry.focus` f | focus above coverslip 1 | none (h = 0, stage z) | free, from the lowest heights | held; its drift is z drift |
| `fourpi.cover` (2) | each arm's coverslip against the beads' sample | held at 0 | arm 1 free with a prior (±5 µm); arm 2 inside D | held |
| `fourpi.offset` Δ | arms' misalignment | free | Δz free with a prior (realigned per sample); Δx, Δy held | held |
| coherence α_c | per channel | free | free | in the spline |
| pupils, blur, channel offsets | the instrument | free | held | in the spline |

**Why this split.**
- **Beads learn the instrument only.**  With the layers held, the pupils are the objectives' and the
  DM's.  An index-matched bead sample makes D_bead drop out.  Mismatched, n_bead and D_bead are held at
  measured values: fitted, they trade with the lower pupil's spherical, D and n acting almost as one
  scalar, D·(k_med − k_imm).  Beads on both coverslips (or a second sample, flipped) separate them
  cleanly: each arm then sees the gap in turn.
- **Per sample, a layer is one scalar of known shape, not free Zernikes.**  Arm 2's constant term is
  k_med·D: one number times an exact shape, freed on top of held pupils.  Freeing Z11 or Z22 instead
  would let the pupil absorb anything spherical.  A coverslip off the beads' thickness has a similar
  shape, (k_cov − k_imm); in arm 2 it cannot be told from D and is part of an effective D, in arm 1 it is
  the only constant term and is freed there.  For silicone objectives that is the term their correction
  collars exist for (±5 µm of glass), and a collar setting falls into it.
- **The heights make the sample identifiable.**  Molecules at many h see k_med·h, which pins n_s and f
  together; the lowest give f, as `calibrate.depth.auto_focus` does for Spline 3D.  Molecules on
  coverslip 2 too (an upper membrane) would give D from their edge.
- **The DM is part of the instrument.**  A setting that differs between beads and sample enters as the
  commanded difference, held (`ZernikePupil`'s held modes), the commands kept in the metadata.  Learnt
  from the sample it would trade with D.
- **Δ is realigned with every sample.**  Its lateral direction was not stable even from beads (§8;
  the single-step models put it along three directions), so Δx and Δy are held; Δz, which realigning
  for the best interference moves, is free with a prior.
- **The coherence stays free in situ**: it is what changes most with the sample (scattering, depth,
  the cavity drifting during an exposure).

**Fitting molecules.**  The fitter keeps x, y, z, N, Φ and the backgrounds.  The geometry enters through
the spline: the export samples I, A and B as functions of h for given (n_s, D, f, Δz, cover), z the
height above coverslip 1 -- a 4Pi depth spline, as `calibrate.depth.depth_spline` is for Spline 3D.  The
fringe follows k_med·(2h − D), so its period, about λ/(2 n_s cos θ), is the same at every h; the export
measures it from the model at that n_s.  Another field of view in the same sample keeps n_s and the
coverslips, finds the focus again and refits D with the last value as its prior (the gap varies over
millimetres); another sample repeats the in situ step with its own measured n_s and D.

**n_s and D from beads on both coverslips.**  Beads on coverslip 1 alone cannot give n_s: the stage
moves the whole sandwich, so a stack changes only the immersion's paths.  With beads on both coverslips,
a stack around each and the stage offset Δs between the two stacks recorded, they can:
- **The aberration measures one number.**  After refocusing, the gap's and the index's aberrations in
  the far arm have nearly the same shape (correlation −0.985 at n_s 1.36, −0.998 at 1.38, −0.9999 at
  1.40): +0.001 of index looks like −390 nm of gap at 1.36 (−490 nm at 1.38).  So the stacks measure the
  optical mismatch, about D·(n_s − n_imm).
- **The stage travel measures the rest.**  Moving from one coverslip's beads into focus to the other's
  takes Δs = D + c(D, n_s), c the focal shift through the sample: about 11.1 µm for D = 10 µm at n_s
  1.36.  Along the direction the aberration cannot see (+0.001 of index with +390 nm of gap) Δs moves by
  about 400 nm.
- **Together** (n_s 1.36, D = 10 µm, the far arm's aberration to ±0.01 rad, Δs to ±20 nm): n_s to about
  ±0.0002 and D to about ±20 nm.  At 1.38–1.40 the aberration weakens but Δs keeps its lever.  At a
  matched index nothing separates them, and nothing in the PSF needs to; only the phase's z scale
  (∝ n_s) does, which the refined period or a refractometer gives.

The joint fit, in this model: coverslip-1 beads at h = 0 (arm 1 sees no sample, arm 2 sees D),
coverslip-2 beads at h = D (the other way round), so each pupil is seen once clean and once behind the
layer, which separates the instrument from the layer too.  Free and shared: both pupils, the blur, the
channels, Δ, the coherences, D and n_s (k_z,med then computed from a parameter, not read from the grid).
Held: n_imm, NA, the beads' λ and Δs.  Each coverslip's coverslip thickness acts alike on both bead sets
in its arm and stays in that arm's pupil.

- **The focus is the model's, not a criterion's.**  With 1–1.5 rad of spherical in the far arm, peak
  intensity, the sharpest image and the least-squares focus differ by tens to a hundred nm, the size of
  the lever.  So Δs is the recorded offset between the two stacks, each bead's z coming from the joint
  fit; a focus found beforehand by eye or autofocus enters only as a prior of about ±100 nm.
- **Each coverslip's beads lie on one plane**, a height and a tilt per coverslip, not a free z per bead:
  free z would take up Δs and lose the lever.
- **The stage's scale over the offset** is the other systematic (0.1 % of 10 µm is 10 nm).  Each stack's
  fringe calibrates the piezo locally (2 k_z,imm per nm of stage, λ and n_imm known); its linearity
  across the offset needs the stage's specification or a check.
- The cavity phase between the two sets would give the optical thickness to a fraction of a fringe, but
  across about 75 fringes and with the cavity drifting between the stacks; left out.
- **What it measures** is the index of the medium the beads sit in, and D at that position.  Inside a
  cell the index differs (1.36–1.39, unevenly), so this value is the in situ prior, not the answer.

**A cheaper scheme: the stage travel from dirt, beads on one coverslip.**  The travel between the two
coverslips' inner surfaces is focused on dirt, and beads lie on coverslip 1 only.  Two things change: Δs
comes from a focus criterion instead of the fit, and arm 2's pupil is never seen without the layer, so
its Zernikes take up the layer's low orders and only the steep remainder at the pupil's edge still
measures the index.  Idealised (the layer's shape the only unknown, 10 µm gap, the far arm's aberration
to the rms given):

| scheme | n_s 1.36: σ(n_s), σ(D) | n_s 1.38: σ(n_s), σ(D) |
|---|---|---|
| beads on both coverslips, Δs from the fit (±20 nm) | 0.0002, 20 nm | 0.0003, 21 nm |
| dirt focus ±50 nm, beads on one coverslip | 0.0003, 48 nm | 0.0008, 53 nm |
| dirt focus ±100 nm, beads on one coverslip | 0.0004, 95 nm | 0.0009, 100 nm |
| dirt focus ±200 nm, beads on one coverslip | 0.0005, 187 nm | 0.0009, 198 nm |
| dirt ±100 nm, one coverslip, pupil to n ≤ 8 | 0.0005, 95 nm | 0.0014, 103 nm |
| dirt ±100 nm, one coverslip, aberration to ±0.03 rad | 0.0010, 99 nm | 0.0025, 114 nm |
| dirt focus alone, no aberration | not determined | not determined |

On paper 2–3× worse in the index, and the gap as good as the dirt focus; the index stays well below the
0.005 that matters.  In practice it is worse than the table:
- **The index rests on the pupil's edge**, where sampling, apodisation and the magnitude model are least
  sure.  At ±0.03 rad there the index is 0.001–0.0025; more Zernikes make it worse, absorbing more of
  the edge.
- **The dirt focus is biased, not only scattered**: through 1–1.5 rad of spherical the sharpest image sits
  tens to a hundred nm from the model's focus, differently for reflected, scattered or fluorescent dirt.
  At about 400 nm of travel per 0.001 of index, 100 nm of bias is 0.00025.  The criterion could be
  modelled, at the cost of an assumption about how the dirt is imaged.  The travel enters as a prior of
  honest width, never held, and the dirt must be on the sample side of each coverslip.
- **Instrument and layer stay mixed in arm 2**: its pupil holds what the gap and index got wrong, carried
  to a sample of another gap at the wrong scale.

So: beads on both coverslips once, as the instrument's reference (arm 2's pupil clean, and a check of the
dirt focus's bias); dirt and beads on one coverslip for routine samples after that.  These numbers come
from the radial model of the layer alone; the joint fit of simulated stacks is the check, once the model
exists.

**n_s, checked by the data.**  In situ, *4Pi z*'s `refine_period` measures the phase's slope against the
envelope's z.  If the model's index is off by a fraction ε, the envelope reads h·(1 + ε) and the phase
h/(1 + ε) (paraxially: the envelope's defocus goes as 1/(n_s λ), the fringe as n_s/λ), so the slope is
off by about 2ε while λ cancels; no known height is needed.  0.005 of index at 1.36 is 0.74 % of slope,
and the refined period came within 0.3 % on M3's 5000 simulated molecules: about ±0.002 in n_s from the
statistics.  The limit is the bead model's own consistency: on the M5 beads, in a matched medium, the
envelope's z runs 1.016 and the phase's 1.007 per nm of stage, a 0.9 % gap that would read as about 0.006
of index.  So the bead model's two scales are put on one ruler first (the fringe, through a free stage
scale, `geometry.z_scale`), and what bends the envelope is found (§8's open shared pattern).  The refined
period is reported as an implied n_s beside the measured one, not fitted, unless the sample spans several
µm in h.

**Not covered:** the pupils changing over the field.

## 4. Plan

The milestones are ordered so that a fitter exists early, on v1's ready-made
models, and the risky refactor happens behind tests.

**M0. Data and formats**
- Fetch `4pi_40nm_bead/`, `results/4pi_40nm_bead/` and the two
  `4pi_smlm_nup96` files: about 0.8 GB, through
  `scripts/uipsf/fetch_zenodo.py`.
- Fetch the astigmatic DCIMG beads from zenodo 14287640 (25 MB).
- Write the DCIMG reader.
- Write a reader for `qd1..qd4` `.mat` files.
- Write a loader for v1's 4Pi h5 → `FourPiCalibration`, renormalised.

**M1. The fitter**
- `CSpline4Pi` in `csrc/models.hpp`: six parameters, three coefficient
  planes, the Φ derivative N(−A sin Φ + B cos Φ), and Φ wrapped in `clamp`.
- `fit_cspline4pi_global` in `fit.cpp`.
- The phasor start (D3).
- `_GlobalPSF` generalised from five parameters to P.
- `Global4PiPSF`.
- Tests:
  - simulated 4Pi spots from v1's model reach the CRLB in x, y, z and Φ;
  - Φ is recovered over the whole circle;
  - the fitter agrees with v1's `locres` on the training beads.
- A benchmark script fits the same spots with v1's dylib and compares.

**M2. Simulation**
- An N-channel camera layout.
- A 4Pi draw from a `FourPiCalibration`, with Φ = 2πz/z_T + Φ0(t) and a
  drift in Φ0.
- A 4Pi preset in *Simulate*.
- The truth includes `phase_true`.
- This is what every later test stands on.

**M3. Phase to z**
- `smappy/fourpi.py`, array functions: Φ0(t), fringe probabilities, z_T
  refinement.
- A plugin, *Analysis/After fit/4Pi z from phase*, which is also an *After
  the fit* step.
- Tests on M2's simulations:
  - Φ0(t) drift is recovered;
  - the ghost fraction matches `fringe_prob`;
  - the sign convention holds.
- Its page.

**M4. N channels end to end**
- The preprocessing front end (D5) and `ChannelLayout`.
- `MultiChannelEngine`, keeping dual colour's numbers.
- Detection on the summed channels.
- The 4-channel bead registration and channel drift.
- The readers.
- The 4Pi microscope entry.
- The plugin *Localize/Spline 4Pi*, with its page, tests and line in
  `default.gui.yaml`.
- First real result: Nup96 4Pi with v1's model.  The two rings, about 50 nm
  apart, are the test of z.

**M5. uiPSF v2 4Pi from beads** (done: §8, §11, §14)
- `FourPiPSF` (built as `uipsf.model.fourpi.FourPiModel`), the phase axis, the gauges, and the export
  with a measured z_T.
- Comparison with v1's model on `4pi_40nm_bead`:
  - the fitted bead positions;
  - the CRLB;
  - z_T.
- *4Pi beads* as a method of the PSF calibration window.

**M6. uiPSF v2 4Pi in situ** on the Nup96 data, compared with v1's in-situ
result files.  This comes after the bead model, and after the in-situ learning
work in progress elsewhere, which it builds on.  It holds and frees what D9
says, in this order:
- each arm its own depth in `FourPiModel` (h, D − h), with `fourpi.gap`,
  `fourpi.cover` and `geometry.focus`; a matched index gives today's model;
- a simulation study of what can be told apart: n_s against D, `cover`
  against D, Δz; and the joint fit of beads on both coverslips with the
  stage offset between the stacks;
- the bead model's envelope and phase z scales made to agree (§8: 1.016
  against 1.007), before any n_s is read from the refined period;
- the 4Pi depth spline for a given geometry, and the fitter's *sample index*,
  *gap* and *focus* (auto), as Spline 3D's *focus*;
- the implied n_s from the refined period, in *4Pi z*.

**M7. Later, as needed**
- The uiPSF 4Pi model made vectorial: dipole fields and the s and p
  polarisations of the channels (D7).  Done (§17): it fits better at no
  more cost, and is the window's default since §19.
- The locked refit (D4).
- A model-free bead average of I, A and B in SMAPpy.  For each voxel this is
  a linear least-squares fit over beads × phases, with the bead phases taken
  from the fitter.  It is an independent check of the learned model.
- Multicolour 4Pi, from the s/p ratio.
- Two cameras.

**What can run in parallel:**
- M2 can start alongside M1.
- M5 needs only M0's data and the calibration format, so it can start with
  M1.
- M4 is the long one; it needs M1.

## 5. Answered questions (2026-10-27)

1. **What the Ries-lab 4Pi writes:** Hamamatsu DCIMG, with all four quadrants
   in one frame.  This led to the preprocessing front end in D5.  Still to be
   settled from a real file: the channel order and the mirrors.
2. **Astigmatism:** yes.  It is the main source of the coarse z (D6a).
3. **Order:** beads first (M5).  In situ comes later (M6), after the in-situ
   learning work in progress elsewhere.
4. **The DCIMG data is a target,** not only a test case.  The DCIMG reader
   moves into M0.
5. **The fitter's source:** none was found; the 4Pi fitters exist only as
   binaries (`.mex*`, `.dll`, `.dylib`, `.so`).  The fitter is written from
   the IAB model, and v1's dylib is the black-box comparison (§2).

## References

- Li Y, Buglakova E, Zhang Y, Thevathasan JV, Bewersdorf J, Ries J (2020).
  Accurate 4Pi single-molecule localization using an experimental PSF model.
  *Opt Lett* 45:3765. [doi:10.1364/OL.397754](https://doi.org/10.1364/OL.397754).
  The IAB model.
- Li Y, Shi W, Liu S, … Ries J (2022). Global fitting for high-accuracy
  multi-channel single-molecule localization. *Nat Commun* 13:3133.
  [doi:10.1038/s41467-022-30719-4](https://doi.org/10.1038/s41467-022-30719-4).
  globLoc; linking photons across channels.
- Liu S, Chen J, Hellgoth J, … Ries J (2024). Universal inverse modeling of
  point spread functions for SMLM localization and microscope
  characterization. *Nat Methods* 21:1082.
  [doi:10.1038/s41592-024-02282-x](https://doi.org/10.1038/s41592-024-02282-x).
  uiPSF; the 4Pi pupil and voxel models.
- Liu S, Huang F (2020). Enhanced 4Pi single-molecule localization microscopy
  with coherent pupil based localization. *Commun Biol* 3:220.
  [doi:10.1038/s42003-020-0908-2](https://doi.org/10.1038/s42003-020-0908-2).
  Partial coherence; cavity phase drift; ghosts.
- Huang F, Sirinakis G, Allgeyer ES, … Bewersdorf J (2016). Ultra-high
  resolution 3D imaging of whole cells. *Cell* 166:1028.
  [doi:10.1016/j.cell.2016.06.016](https://doi.org/10.1016/j.cell.2016.06.016).
  Astigmatic unwrapping; drift.
- Bates M, Keller-Findeisen J, Przybylski A, … Jakobs S (2022). Optimal
  precision and accuracy in 4Pi-STORM using dynamic spline PSF models.
  *Nat Methods* 19:603.
  [doi:10.1038/s41592-022-01465-8](https://doi.org/10.1038/s41592-022-01465-8).
  Phase-shifted models; phase drift; the bias from channel imbalance.
- Wang J, Allgeyer ES, Sirinakis G, … Bewersdorf J (2021). Implementation of a
  4Pi-SMS super-resolution microscope. *Nat Protoc* 16:677.
  [doi:10.1038/s41596-020-00428-7](https://doi.org/10.1038/s41596-020-00428-7).
- Hao X, Antonello J, Allgeyer ES, Bewersdorf J, Booth MJ (2017). Aberrations
  in 4Pi microscopy. *Opt Express* 25:14049.
  [doi:10.1364/OE.25.014049](https://doi.org/10.1364/OE.25.014049).
  The covariant and contra-variant modes of the two pupils (§21).

## 6. What M0 and M1 found (2026-10-27)

**What exists now.**
- `smappy/fourpi.py`:
  - `FourPiCalibration`;
  - `from_volumes`, which normalises the channels together and splines each of
    I, A and B;
  - `load_uipsf_v1`;
  - the native file, which keeps the knots and splines them again on load
    (64 times smaller than the coefficients);
  - `model_calibration`, an analytic astigmatic 4Pi model for tests;
  - `simulate_rois` and `link`.
- `csrc`: `CSpline4Pi` and `fit_cspline4pi_global`, with the phasor start.
- `psf.Global4PiPSF`, and `_GlobalPSF` generalised beyond five parameters.
- `io/dcimg.py`: both versions, with the 4 pixels put back.
- `io/quadrants.py`: the `qd1..qd4` `.mat` files.
- `tests/test_fourpi.py`.

**The fitter reproduces v1's closed-source one.**
- The test fits v1's 18 training beads (1134 localizations) with v1's model
  and v1's channel offsets.
- Per localization, it agrees with v1's own results to 0.0003 px in x and y,
  0.001 rad in phase and 0.02 slices in z.
- Apart from that there are only constant offsets, v1's half-pixel
  convention, and v1's photon number, which is half ours.
- On the analytic model the fit reaches the CRLB in x, y, z and phase.  It
  finds the phase all round the circle from the phasor start, with no phase
  grid.  300 fits take 50 ms.

**v1's beads measure the fringe period.**
- The phase moves 0.0243 rad per nm of stage, a period of 258.5 nm.  v1 had
  been given 260 nm.
- The deformable mirrors' steps come out at +2.06 and −2.12 rad; they were set
  to [2, 0, −2].
- The model's modulation depth |(A, B)|/I has a median of 0.30 in the core and
  a maximum of 0.73.

**Photons must be linked.**  The phase is the photons' split between the
channels.  A photon number free in each channel absorbs the interference and
leaves the phase undetermined: exactly so for the analytic model, nearly so
for a real one.  SMAP's `MLE_4Pi` defaulted to that.  `Global4PiPSF` shares
the photons by default and says why.

**v1's `.mat` models are transposed.**  v1 read MATLAB files as h5py gives
them, with rows and columns swapped.  `io/quadrants` reads them the right way
round, and `load_uipsf_v1` transposes such a model to match.  The channel
transforms it keeps stay in v1's frame.

**The product form I·(1 + a cos Φ + b sin Φ) was considered and not used.**
- At the knots it is the same function as the sum.
- Its gain would have been positivity.  v1's additive splines evaluated
  between knots never go below zero; the worst is +7·10⁻⁵ of the peak.
- The ratio a = A/I is ill-defined in the wings.
- The sum keeps building I, A and B from beads a linear problem.
- a and b are worth showing as a diagnostic: the modulation map.

**DCIMG.**  The Bewersdorf-design file of zenodo 14287640 is version
`0x1000000` and leaves the crop record empty.  The reader then takes the ROI
to be centred on the sensor's rows, which puts the bad 4 pixels in row 83 of
168.  The four images sit at x = 151, 389, 1474 and 1713, each 168 wide, with
the last two mirrored: SMAP's `Yale4Pi_settings_3D.txt`.

## 7. What M2 and M3 found (2026-10-27)

**What exists now.**
- **Simulation:** `fourpi.spots` evaluates the model for thousands of
  emitters at once, so 3000 ROIs take 0.65 s.  `simulate_rois` takes photons
  and the cavity's phase per emitter, and `cavity_drift` makes a cavity
  phase that drifts steadily with a random walk on top.
- **Phase to z:** `fourpi.cavity_phase`, `unwrap` and `refine_period`.
- **The plugin** *Analysis/After fit/4Pi z* (`plugins/fourpi_z.py`), with its
  page.

**Still to do for M2.**
- **Frames:** simulated frames in the common multi-channel layout.
- **The *Simulate* preset:** both wait for M4's front end, which is what
  would read the frames.

**Measured on a simulation:** 5000 molecules over 5000 frames, 1500 photons
(lognormal), the cavity drifting 1.5 rad per 1000 frames with a random walk.
- **The cavity's phase** is recovered to 0.04 rad rms with windows of 500
  frames.  The linear extrapolation past the end windows matters: holding
  the end values left 0.18 rad of error there.
- **Wrong fringes:** 1.3 %, against the 1.1 % the probabilities predicted.
  The envelope's errors needed scaling by only 1.03.
- **z on the right fringe** scatters by 3.8 nm, against 45 nm for the
  envelope alone.
- **Refining the period** without a correction lands at 0.99 of the true
  slope.  This is errors-in-variables attenuation.  Dividing each term by
  exp(−k²s²/2) brings it within 0.3 %.

**For M4.**
- Grouping averages `phase_rad` linearly, which is wrong across ±π.  It
  needs a circular, weighted rule, or the phase has to be taken off before
  grouping.  *4Pi z* reads the ungrouped localizations.
- The 4Pi fit has to keep `phase_per_nm` in the table's metadata
  (`metadata["fourpi"]`), which *4Pi z* reads by default.
- The fitter plugin's *after the fit* gets a *4Pi z* step.

## 8. What M4 found (2026-10-27)

**What exists now.**
- **The front end.** Every 4Pi input is an ordinary `ImageSource` of 2D
  frames, so the camera, the frame view and the raw-frame keeper keep
  working:
  - `io/dcimg.py`'s `DcimgSource` reads one file of many frames, or a z stack
    written one file per step;
  - `io/quadrants.py`'s `QuadrantSource` reads a series of `qd1..qd4` `.mat`
    files with the channels side by side, and gives its own layout;
  - both are reached through `open_stack`.
- **`channels.ChannelLayout`:** each channel's rectangle, its mirror, and a
  projective map onto channel 0.  `maps_from_uipsf_v1` converts v1's
  affines, and `YALE_4PI` is the Bewersdorf-design strip.
- **`multichannel.MultiChannelEngine`:**
  - it finds molecules on the channels' sum, resampled onto channel 0 for
    detection only;
  - it cuts every channel's ROI at its own pixels, with the remainder of the
    map in the link;
  - it fits with `Global4PiPSF`;
  - it keeps `metadata["fourpi"]` and the layout with the table.
- **The plugin** *Localize/Spline 4Pi* (`plugins/fit_4pi.py`), with its
  page and a line in `default.gui.yaml`.
- **After the fit:** a *4Pi z* step, on by default, shown only for a phase.
- **Fits clamped at the end of the z grid** get the grid's range as their
  error.
- **`fourpi.render_frames`:** simulated 4Pi camera frames, the frame part of
  M2.

**Checked on real data, the Nup96 example with v1's in-situ model.**
- **Registration.** Fitting each channel's x and y freely gave back v1's
  affines to within 0.06 px.  So the converted maps, and the transpose of
  v1's `.mat` frame, are right.
- **v1's own in-situ ROIs.** The fitter agrees with v1's per molecule: a
  median of 0.11 z slices and 0.014 rad.
- **Speed.** 4000 frames of 4 × 200 × 200 take about 25 s.  Half of that is
  resampling for detection, which can be made faster.
- **The sign of the in-situ model.** v1 builds its in-situ 4Pi models with
  the opposite phase sign to its bead models.  Read with the beads' sign, z
  from the phase came out as −z_env.  The loader now takes the sign from the
  model's `datatype`, and a test guards it.  With the right sign, z from the
  phase follows the envelope's at a slope of +1.09, with 3.3 % expected wrong
  fringes for molecules above 800 photons.
- **What is left is the model.**
  - The envelope z of v1's in-situ model bends by up to ~40 nm against the
    phase's, and scatters 3.3 times its CRLB.
  - mMaple is dim: a median of 530 photons over all four channels.
  - A better in-situ model (M6) is what would improve this.

**Done after this section was first written (2026-10-27).**
- **Registration from beads.**  Beads are summed over the stack, so the
  interference averages out.  The steps are Gaussian centroids, a
  cross-correlation, pairing, and the dual calibration's robust projective
  fit (`channels.register_channels`).
  - On uiPSF's 26 bead scans: 111 pairs per channel, 0.03 px rms, and within
    0.06 px of v1's learned maps where the beads are.
- **Channel drift.**  Blocks of frames are fitted with x and y free per
  channel, and the median offsets are interpolated in time
  (`multichannel.ChannelDrift`).
  - On the Nup96 data, against the bead registration from another day: up
    to 1 px off, steady to 0.1 px.
- **Grouping:** `phase_rad` and `cavity_phase_rad` are combined as angles,
  weighted by their errors.

**Not done yet in M4.**
- **Dual colour** ran on `dualfit`; it is on the layer since M8's step 1 (§9).
- **A *Simulate* preset** for 4Pi.
- **The Nup96 ring-spacing check** on z, which needs the ROI segmentation of
  the pores.
- **A drawn fit region** is not offered by *Spline 4Pi*.  Molecules are found
  in channel 0's pixels, not on the frame a region is drawn on.

## 9. M8, planned: one channel layer for every multi-channel fit

Decided 2026-10-27.  Port dual colour later, not as a copy of `dualfit`
onto the new engine.  Make the geometry one layer, right after the loader,
so that everything after it is agnostic: it sees only corresponding
channel images, and every transformation is defined on those.

**The layer.**
- **Loader:** frames as the camera wrote them.  That is a chip with
  quadrants or halves, separate files, or a multi-D TIFF.
- **Channel layer:** a `ChannelLayout` cuts each frame into C channel images,
  (n, C, h, w), each mirrored channel flipped back.  It is the only code
  that knows rectangles, mirrors, the camera ROI and chip offsets.
- **Everything after it** sees channels:
  - maps from channel c to channel 0 in channel pixels, nearly
    translations once the flips are undone;
  - calibrations (dual-colour, 4Pi, registration) that store the layout
    with their maps;
  - detection per channel then paired (dual colour, different dyes), or on
    the mapped sum (4Pi), as a setting;
  - a fit region drawn on channel 0 and mapped through the layer;
  - a preview and frame view with the channels side by side;
  - bead calibration, movie registration and channel drift for any C.

**What it retires.**
- `dualfit`'s split geometry (`which_channel`, `split_position`, the
  halves, `region_counterparts`);
- `PeakFinder.split`;
- `DualColorCalibration`'s `main`/`secondary`;
- the simulator's two-halves loop.

**What it keeps.**  A dual-colour splitter becomes a two-rectangle layout.
Its existing calibration files load as a layout plus their transformation
re-expressed in channel coordinates.

**4Pi already runs on the layer (2026-10-27).**
- **The source.** `channels.ChannelSource` wraps any source.  `_FitPlugin`
  has two hooks: `open_source`, and `view_camera`, which takes the camera ROI
  off.
- **The engine** takes channel blocks only, and reports positions in
  channel 0's own pixels.
- **The geometry** comes from the microscope database
  (`Channel.region` / `mirror`, `Microscope.channel_layout`), from the
  file, or typed, on the chip.  It is put on the camera ROI by
  `ChannelLayout.on_camera`.
- **The registration** comes from the model, from beads (measured once,
  kept beside them as JSON, `channels.save_registration`) or from a file.
  `describe_map` reads a map about its channel's centre.
- **The model file** records the layout its maps were measured on.
- **The fit region** is drawn on the preview's sum, in channel 0's pixels,
  so `RegionFinder` works unchanged.

So 4Pi is the layer's first user and its test, and M8 is what is left: dual
colour on the same layer.

**The bar.**  Dual colour's numbers stay identical on the existing
two-channel tests and on a saved run.  The bead calibration and the
movie registration give the same maps.  It is worth doing when the next
multi-channel need comes: more than two colours, channel drift for dual
colour, biplane plus colour.  It is not required for 4Pi.

**Step 1, dual colour on the layer (begun 2026-10-29).**  M8 has three goals, taken one at a time:
dual colour on the layer; channel drift for dual colour; and a source of two files, one camera per
channel, with frames paired between them.  Step 1 is the first, and was decided as follows.

- **The bar.**  Not bit-identical: the layer cuts the channels before the peak finder filters them,
  where `dualfit` filtered the whole frame across the seam; a flipped channel rounds its ties the other
  way; and a mirrored secondary is fitted flipped back, with its spline flipped too, where `dualfit` cut it
  as the camera saw it with a link factor of −1.  The bar is the same localizations, positions within
  0.001 px and photons within 1e-4, for every molecule more than a box from the seam and the borders; the
  few near the seam may differ, and are counted.  Measured on six scenarios end to end through the
  plugins -- up-down, mirrored in y and in x, right-left with the main channel second, unequal halves,
  spline and Gaussian -- whose tables before the port are kept as the reference.
- **Positions in channel 0's own pixels** (until 2026-11; since then on the camera chip, where channel 0
  saw them -- `docs/microscope-database.md`, "the coordinates"), as 4Pi's: `x_nm` of dual-colour data moves by the main
  half's corner against files fitted before.  They are counted from channel 0's rectangle *on the chip*,
  so a calibration from one camera ROI applies to a movie from another: a ROI that cuts into a channel
  leaves that channel smaller, and its pixels keep their place in the rectangle (`ChannelLayout.origins`).
- **Channels of different sizes**: a measured or typed seam need not be in the middle.  A block keeps
  one array, (n, C, h, w) with the smaller channels padded, and every channel's own size; the summed
  detection of 4Pi still needs one size.
- **Detection per channel, then paired**, as `dualfit` did: a dye can be dark in one channel, so each
  channel is searched with its own cutoff, the second channel's peaks are put on channel 0 by the map,
  mutual nearest neighbours within 4 px are merged, and a lone peak keeps its partner's box where the
  map puts it.
- **The calibrations stay as they are on disk.**  A dual-colour calibration or a `_2ct.h5` is read as a
  two-rectangle layout on the chip -- the main half channel 0 -- with the map re-expressed between the
  channels' own pixels; the secondary spline is flipped to the layer's orientation when there is a mirror.

**Step 1, done (2026-10-29).**  Both two-colour fitters run on the layer (`_TwoChannelFit.open_source`,
`MultiChannelEngine(detection="pairs")`).  Against the reference tables, end to end through the plugins:
- up-down, main half first, spline or Gaussian: the same tables, to the bit;
- unequal halves, main half second (spline right-left, Gaussian lower): every molecule more than a box
  from an edge the same to 1e-5 px and the photons to the bit;
- mirrored (spline in y, Gaussian in x): of about 400 localizations the median difference is 0 and the
  99th percentile 4e-4 px; 2-3 differ by up to 0.4 px, each with another molecule within 7 px, where the
  likelihood is flat and the flipped summation moves the fit;
- lost: 0 in up-down, 24-52 in the others, every one with itself or its partner within half a box of its
  channel's edge -- the boxes `dualfit` let straddle the seam.  None was gained.

A movie from a camera ROI that cuts into both halves gives the whole frame's positions
(`test_a_movie_from_a_smaller_camera_roi_is_fitted_where_the_calibrations_halves_have_it`).

**The old split path, retired (2026-10-29).**  `dualfit.DualChannelEngine`, `combine_peaks`,
`cut_paired_rois`, `build_link` and `region_counterparts` are gone, with the single-channel fitters'
hooks for them (`back_projected`, `region_counterparts`, a split cutoff in the preview).  Their tests
run on `MultiChannelEngine` now, with the same tolerances: z, the colour ratio, linked photons, the
unlinked partner's offsets, a mirrored splitter.  The pupil calibration's fit quality (`_pair_link`)
builds its link itself, identical to the old one (checked plain and mirrored).  `dualfit` keeps what
both paths shared: `LINK_XYZ`, `MATCH_PIXELS` and `_match`, the geometry (`which_channel`,
`split_axis`), `PairedROIs` and `paired_to_localizations`.

Two things stay on the chip by design, not as leftovers.  **The movie registration**
(`_TwoChannelFit._calibrate`, *calibrate from this movie*) fits the whole frame with a plain Gaussian
and registers the halves from that: it measures the layout, so there is none yet to cut the frame by,
and it thresholds each half on its own (`PeakFinder.split`) at a provisional seam.  **The registration's
geometry** (`which_channel`, `split_position`) is how a `_2ct.h5` records which half is which, read when
the layout is built (`chip_halves`).
*Since 2026-11-04* the pass is a two-channel fit cut by the microscope's arrangement
(`multichannel.SplitEngine`) and the registration pairs its channels; the geometry records the
channels' rectangles (`geometry["channels"]`, `dual.geometry_layout`), and `which_channel`,
`split_position` and `chip_halves` are gone (`docs/microscope-database.md`).

## 10. Real data: 4Pi-SMS Nup96 in DCIMG (2026-10-27)

**The data.**  The 4Pi-SMS consortium's Example Data II, zenodo 4022827.
- Nup96-SNAP with Alexa 647, in the lower nuclear envelope of a U2OS cell,
  on a Bewersdorf-design astigmatic 4Pi.  It was taken on 22 October 2018.
- The zip also holds a bead z stack (101 one-frame DCIMG files, 20 nm
  steps, no phase steps), a SMAP 4Pi calibration (`cal4pi`), and the
  authors' alignment and phase calibrations.
- Fetched with `scripts/uipsf/fetch_zenodo.py --url` into
  `~/.cache/uipsf/zenodo/4pi_sms_nup96`: the beads, the calibrations and the
  first of the four 2 GB movie files (3000 frames).

**SMAP's model of these beads is not usable as it was.**  Read by
`fourpi.load_smap_4pi`, it had three problems:
- `normf` makes channels 2 and 3 twice as bright as 1 and 4.  On the beads,
  the four channels are nearly equal (1 : 0.88 : 0.92 : 0.96).
- Its incoherent part I oscillates with the fringe at its centre.  The
  pairwise construction (`make4Pimodel`) left the fringe in it.
- So a fit with it traded z for phase: the envelope z swung ±150 nm with the
  phase, and followed the stage at a slope of only 0.3.
- Its phase offsets, on the other hand, agree with what the beads show to
  0.1 rad.

**A model from the bead scan itself** (`fourpi.iab_from_beads`):
- The beads' centre brightness gives the phase per slice, each channel's
  gain and offset, and each bead's phase (`scan_phases`).
- Then I, A and B are solved by least squares per voxel over beads and
  channels.  That is linear because the channels share one shape up to
  gain and offset.
- On 3 usable beads it gives a period of 273.6 nm (SMAP: 278), and an I
  free of the fringe.
- The beads then fit with z following the stage at a slope of 0.83–0.92,
  within ~40 nm, and the phase's z following it to 5–26 nm.

**The registration** from the stack's 5 beads is an affine map
(`register_channels` now fits one from 3 to 7 pairs).
- Residuals 0.06–0.10 px, with rotations of −10, −4 and +7 mrad, as the
  authors' own alignment found.
- The movie's channel drift against it was 0.10–0.17 px, steady over the
  file.

**The movie** (*Spline 4Pi*: typed SMAP corners, the model's registration,
channel drift, *4Pi z* after the fit):
- 77,818 localizations from 3000 frames, at 13 frames/s.
- Median 18,900 photons; for the sure, precise ones a z error of 2.9 nm
  from the phase, xy 4.5 nm.
- *4Pi z* expects 11 % wrong fringes over all localizations.  The envelope
  scatters 6.7 times its CRLB, the 3-bead model's limit.

**The ring check.**
- Pores were found by DBSCAN on the localizations with fringe probability
  above 0.95, a z error under 8 nm and more than 1500 photons.
- z was taken relative to each pore's mean, with the envelope's local tilt
  (from the 12 nearest pores) removed.
- The pooled profile is bimodal, the two rings about equally occupied
  (0.83–0.86) and each ~17 nm wide.
- Their separation by a two-Gaussian fit is 42.7, 44.2 and 46.6 nm
  (± 0.7–1.1) for pores of at least 8, 20 and 35 localizations, rising as
  each pore's mean gets less noisy.
- This is a consistency check, not a calibration.  The rings' spacing in a
  given cell is not a known standard, and nothing is tuned towards a value.
  What it shows is two clearly separated, equally occupied rings, the same
  with two different models.

**With uiPSF v2's pupil model of the same beads (after merging M5).**
- **Learning.** The model was learned from the bead stack, one phase per
  slice (M5's single-phase start), all 5 beads at 21 px, in 3.5 min on the
  Mac's GPU.  I and Q were rendered from it at three phases and
  demodulated by its own measured period (`uipsf.workflows.fourpi`).
  - Its channel offsets: 0, −2.06, 3.02, 1.20 rad.
  - Coherence 0.44–0.47; a modulation of 0.44, constant through z.
  - Its learned maps agree with the bead registration to 0.06 px and
    0.6 mrad.
- **On the beads:** likelihood per pixel −1.03 (the real-space model −2.66).
  The envelope z follows the stage at a slope of 0.89–0.98 and scatters
  12–24 nm, near its CRLB of 7–16 nm.  The real-space model scattered
  60–96 nm.
- **Photons.** The pupil model fits a bead's 7,477 photons where its four
  channels hold 7,882 above background in 21 × 21 px.  The real-space model
  fitted six times too many.  The cause was `normalise`'s pedestal: it
  raised I to cover a noisy |A, B| > I, as v1 does.  `normalise` now shrinks
  A and B there instead, and takes off only a positive pedestal.
- **On the movie** (same settings as above):
  - 78,984 localizations at 27 frames/s;
  - the envelope scatters 1.93 times its error, against 6.7 times before;
  - median z error 5.1 nm, where the real-space model's overestimated
    modulation claimed 3.7;
  - 14 % expected wrong fringes over all localizations.
- **The rings:** 41.7, 43.1 and 44.7 nm (± 1.1–1.8) for pores of at least
  8, 20 and 35 localizations.  That is the real-space model's answer within
  2 nm, so the spacing is set by the pores' centring and tilt, not by the
  model.

**What would improve the ring check.**
- All four movie files, which gives more localizations per pore.
- Centring each pore by a fit of two rings, rather than on its mean.
- Ghost removal before the pores are found.

## 11. What M5 found (2026-10-27)

**What exists now.**
- `uipsf.model.fourpi.FourPiModel`: the coherent two-objective model of D7, scalar.  One Zernike pupil per
  objective, shared by the channels; the two arms' image fields summed with a per-bead phase, the mirror
  steps and a phase offset per channel; a coherence and an arm balance per channel; one shared
  misalignment of the objectives, (Δx, Δy, Δz).  v2's blur and bead kernel act on the four images
  |h_up|², |h_lo|², Re and Im of h_up* h_lo, of which every channel and step is a weighted sum.
- `uipsf.model.fourpi_start`: the start from the data alone.
- `uipsf.workflows.fourpi`: `learn_fourpi` (the start, each bead's phase from a circle of twelve, Zernikes
  up to n = 4, then all of them) and `calibration`, the export to `FourPiCalibration` with the period
  measured.
- `smappy.calibrate.fourpi_beads`: the quadrant `.mat` scans read, the beads found on all channels summed,
  each channel cut on its own centroid, learnt and saved.  No window method yet (below).
- `benchmarks/uipsf/real/zenodo_4pi_beads.py`; tests in `tests/uipsf_tests/test_fourpi.py`.

**The gauges, and one that was not foreseen.**
- δ_0 = 0 and the first mirror step 0, as planned.  A relative piston is the beads' phases, so it is no
  parameter; Δz moves the arms' foci apart but cancels from the interference's phase.
- **Which objective is "upper" cannot be told from the data.**  The lower pupil's twin A*(−k) as the
  upper one and the other way round, with Δ → −Δ and each channel's balance r → 1 − r, gives the same
  images exactly: the twin's field is the conjugate of the other arm's, and h_up* h_lo is unchanged.
  The first simulated fit landed on the swapped labelling.  `fourpi.swapped` gives the other one; the
  export does not depend on it.
- **The z direction is not fixed by the 4Pi model either**: z → −z with the arms swapped (without twins)
  and every phase reversed is the same model.  What fixes it is the mirror steps' sign.  The steps are
  known as set ([2, 0, −2]), not which way round they enter the phase, so the start measures the fringe's
  direction in the data and turns them round if needed.  On the example beads it did, with z taken in
  v1's direction.

**Simulated beads** (`test_fourpi.py`: 6 beads, 9 slices × 3 steps, four channels, astigmatism of
opposite sign in the two objectives, coma, spherical, misaligned arms, unequal channels, 8000 photons):
both pupils, Δ, the steps, the offsets, the shares, the balances, the coherences, every bead's phase and
position come back within 4 CRLB, in 12 s on the CPU.  The start finds the channels' offsets to 0.15 rad
and turns reversed steps round.

**The export is exact and its period is the model's own.**
- I and Q from three phases reproduce the model at any phase to 10⁻⁶.
- Spots drawn from a model and fitted by `Global4PiPSF` with its export: z to 0.5 nm over ±300 nm, z
  slope 1.0005, and the phase moves with z by the exported `phase_per_nm` to 1 %.
- On the learnt real model, noise-free spots at random sub-pixel positions: 0.76 nm rms of z bias (2.5
  at worst) against a CRLB of 3 nm at 30 000 photons, 0.46 nm in x.  The knots one pixel apart are the
  limit, as for plain splines (`docs/uipsf/architecture.md`).

**The period is a convention of the IAB model, not an observable.**  The fitter's phase Φ and the
demodulated A, B can turn against each other along z: Φ → Φ + δk z with A + iB turned by −δk z is the
same model.  So the phase a model's fit gives against the stage follows that model's own k: v1's
beads give a slope of 258 nm with v1's model (it was told 260) and 265.5 nm with v2's (it measured
267.4).  The observable is z from the phase, Φ/k: 1.007 nm per nm of stage with both models.  The
export therefore measures k where the interference is strong (the energy-weighted turn of Q between
planes), which keeps A and B as slow as the envelope; a typed k only makes them turn.  The start's own
estimate from the ROIs' summed light, 253 nm, weighs the pupil differently and is only a start.
`test_fourpi.py::test_v1s_beads_measure_the_fringe_period_and_the_mirror_steps` (258.5 nm) measures v1's
convention.

**v1's 18 beads, learnt by v2 and fitted back** (`zenodo_4pi_beads.py`: v1's own ROIs, so both models see
the same pixels; v1's optics; each bead, mirror step and slice fitted alone by `Global4PiPSF`, 15 × 15;
scatter about a line through the stage over the middle 15 slices, median over beads and steps; Mac GPU):

| model | photons | time | mean deviance | phase scatter (rad) | z scatter (nm) |
|---|---|---|---|---|---|
| v1 | per observation | | | 0.0506 | 6.76 |
| v2, n ≤ 6 | bleaching curve | 2.7 min | 1.162 | 0.0507 | 7.74 |
| v2, n ≤ 8 | bleaching curve | 14 min | 1.158 | 0.0542 | 6.94 |
| v2, n ≤ 6 | per observation | 6.4 min | 1.060 | 0.0543 | 7.12 |
| v2, n ≤ 6, 20 nm bead | bleaching curve | 3.6 min | 1.163 | 0.0749 | 8.18 |
| v2, n ≤ 8 | per observation | 5.3 min | 1.046 | 0.0505 | 8.23 |

The z scatter taken apart into the pattern all beads share against the stage (a bias of the model along
z) and each bead's scatter about that pattern:

| model | shared pattern (nm rms) | about the pattern (nm) |
|---|---|---|
| v1 | 2.96 | 6.39 |
| v2, n ≤ 6, bleaching | 5.08 | 6.12 |
| v2, n ≤ 8, bleaching | 2.78 | 6.72 |
| v2, n ≤ 6, per observation | 2.93 | 6.45 |
| v2, n ≤ 8, per observation | 5.84 | 6.06 |

- **The default is n ≤ 6 with a free photon count per observation** (`fourpi_beads.learn`, and the long
  test that holds it to v1's straightness within 10 %): v1's choices, within 1 % of v1 on both parts of
  the z scatter, 7 % more phase scatter, 6.4 min.  n ≤ 8 with free photons fits best (deviance 1.046) and
  matches v1's phase, but bends z more: a better likelihood does not by itself give a straighter z.
  Why the shared pattern changes between variants is open; it is the one place v2 is not yet ahead.
- **Both models measure the mirror steps alike:** +2.06 and −2.12 rad.
- **The CRLBs are the same:** median 0.0170 rad and 2.60 nm with v2 (n ≤ 6, per observation), 0.0169 rad
  and 2.50 nm with v1, at 34 000 photons.
- **The beads' light varies per frame:** a free photon count per observation (v1's choice) lowers the
  deviance from 1.16 to 1.06, the largest single term found.
- **Most of the z scatter is not the model's export.**  The phase is straight to 2 nm, so the stage is
  linear.  Each bead's scatter about the shared pattern is about 6 nm with every model; noise-free spots
  from the learnt model, fitted with its export, are off by 0.8 nm rms.  Localised by the uiPSF model
  itself, each slice's three steps together, the same beads give 5.1 nm: one step at a time the
  envelope's z is what the fitter has, and the phase does not help it.
- **What the model measures:** the arms are misaligned by about 100 nm in x (Δx −101 to −117 nm in every
  variant; Δy −7 to −58 nm, less stable; Δz 13–33 nm); each channel's light is split about evenly
  between the arms (r 0.47–0.54); coherence 0.71–0.77; blur (60, 68) nm; channel offsets 0, 1.82,
  −3.05, −1.42 rad, not quarter turns.  v1 learnt a misalignment (`pos_d`) but never used it, and its
  per-channel tip, tilt and defocus took it up channel by channel.
- **The bead kernel and the coherence trade** (open here, settled since): the kernel took the
  observations' z step, a third of the slices' with three mirror steps, so it damped the fringe by 28 %
  instead of 3.5 % and the coherence rose to make up for it.  With the slices' step the window learns
  with the beads' size (40 nm by default) and exports a point emitter (§15, §17).
- **Speed** (open here, done since): every shared column was a JVP, 14 min for n ≤ 8 on the Mac's GPU.
  The 4Pi model is now linearised, 9-12× less per iteration (§19).

**Done since (2026-10-27).**
- **The method in the window.** *4Pi beads* in the PSF calibration window
  (`psf_methods.FourPiBeads`, `fourpi_beads.scan_beads`).
  - Any bead scan comes in through the channel layer: DCIMG or quadrant
    files, one phase per slice or with mirror steps.  The geometry comes
    from the file, the microscope database, typed corners, or 4Pi-SMS.
  - Beads are found on channel 0, and the channels registered from the
    same scans.
  - The model is saved with its layout.  Then the beads are fitted back:
    the envelope's z and z from the phase against the stage.
  - On the 4Pi-SMS example beads: 3.6 min, z slopes 0.90–0.99, scatter
    9–22 nm.  z from the phase ran 7 % steep, the sign of a wrong emission
    wavelength (680 nm typed).
- **The example entry.** The microscope database ships *4Pi-SMS example*
  and its camera: the example data's layout, NA 1.35, n 1.406, an emission
  placeholder of 660 nm.  With them chosen, *Spline 4Pi* finds the layout
  and the camera with nothing typed.  It stands in until the Ries-lab 4Pi
  has its own entry.  Beside it, *uiPSF 4Pi example* and its camera (2026-10-28): uiPSF's
  example data, as uiPSF v1's settings give them -- 129 nm pixels, 0.44
  photons per ADU, offset 100, NA 1.35, n 1.406, 600 nm -- and no regions,
  the quadrant files being already cut.  The PSF calibration window reads
  folders of 4Pi scans (DCIMG or quadrant files, one per numbered series,
  `input.discover_acquisitions`); before, a folder of them read as no stack
  at all.

**Still to do for M5.**
- The beads' fitted positions against v1's own (`locres`), bead by bead.

## 12. Detection's resampling (2026-10-27)

**The step.**  To find the molecules, each channel is put onto channel 0's
pixels and the four are summed.  Detection only places the boxes; each
channel's box is cut at its own pixel from the exact map.  So this
resampling never reaches a fitted position.

**The change.**  Bilinear interpolation, frame by frame, gave way to the
nearest pixel, gathered for a whole block at once
(`channels.detection_lookup`, `onto_reference`).  The peak finder's filter
smooths the sum anyway.  The pixels a rounded map skips or doubles cost at
most half a pixel of alignment in one channel of four.  *shift*, a whole-pixel
slice by the map's offset at the centre, remains for the engine: it ignores
rotation and scale.

**Measured** with `benchmarks/fourpi/detection.py`, on the 4Pi-SMS example's
registration (rotations of −10 to +7 mrad, scale within 0.3 %).

| | bilinear | nearest | shift |
|---|---|---|---|
| Putting the channels together (ms/frame) | 1.75 | 0.23 | 0.07 |
| Simulation, 1800 molecules: found | 88.7 % | 88.7 % | 88.7 % |
| Simulation: x, robust spread | 5.75 nm | 5.73 nm | 5.71 nm |
| Simulation: envelope z, robust spread | 24.2 nm | 24.3 nm | 24.2 nm |

- **The movie's candidates.**  About 7 % of nearest's differ from
  bilinear's, all near the threshold: bilinear smooths the noise a little,
  which moves the dynamic cutoff.
- **The whole engine** (detection and fit, 1500 frames):
  - 4Pi-SMS, 68.5 → 72.0 frames/s: the fit dominates there.
  - v1's Nup96 (4 × 200², few molecules a frame), 180 → 293 frames/s.
  - The plugin's 13–27 frames/s on the 4Pi-SMS movie went mostly into
    reading the 2 GB DCIMG, the drift pass and writing.  Those are the next
    places to look.

## 13. How many starts the 4Pi fit needs (2026-10-27)

**The setup.**  `benchmarks/fourpi/starts.py` (real data, molecules matched
by their candidate) and `starts_sim.py` (simulated with the experimental
model at the data's photons and background).

**With the right model, one start converges.**
- 20,000 molecules were simulated with the 4Pi-SMS beads' pupil model:
  median 4,200 photons, background 29 per pixel and channel.
- Only 0.01 % ended more than five CRLBs off, with one start or with five.
- z and the phase decouple, as the IAB model says.

**On real data a fit can stop in a worse optimum.**  These are the fits more
than 0.5 in log-likelihood below the best of all settings.

| | 4Pi-SMS (Alexa 647) | uiPSF Nup96 (mMaple) |
|---|---|---|
| Stuck with 1 phase start | 5.0 % | 2.5 % |
| Their phase off by (median) | 1.5 rad | 1.8 rad |
| Their z off by (median) | nothing | 139 nm |
| Median photons, stuck vs all | 6,600 vs 4,700 | 650 vs 540 |
| Stuck fits that pass a typical filter | 57 % | 13 % |

- On 4Pi-SMS the stuck fits are brighter than average, and fit the model
  worse: `logl_rel` −1.28 against −0.81.  Only about a quarter are half a
  turn off.
- **So it is not one second valley.**  It is a phase landscape with
  hollows where the model misses the data.

**z starts.**  The setting is a ceiling: only the local minima of the
phasor's residual along z are fitted.  So 2 cost about 10 % more than 1.
- 1 against 3 starts: 0.4–0.9 % of the molecules differ by more than 50 nm.
- 2 against 3: 0.02 %.
- So 2 stays the default.

**Phase starts.**  Each one is a whole fit.  A second phase start rescues
every stuck fit, at twice the time.  Two cheaper rules were tried and
dropped:
- **A phase check:** the deviance a quarter, a half and three quarters of a
  turn away, everything else held, refitting from the best if it beats the
  fit's.  It caught 8 % of the stuck fits for +25 % of the time.  The
  better optimum is only reached when the other parameters adapt on the way.
- **A second start only for poor fits** by `logl_rel`: it caught them about
  in proportion to its cost (50 % of the stuck for +20 %), barely better
  than chance.

**Decision.**  *Spline 4Pi* has *phase starts* under *more*, 1 by default:
the speed of exploration.  2 is for the final localizations of a
measurement.

## 14. 4Pi as a mode of the PSF calibration window (2026-10-27)

**The structure.**  The window's *mode* is the channel geometry: single
channel, dual colour, or 4Pi.  The *method* is how the PSF is made.
- In the 4Pi mode, the pupil model is uiPSF's 4Pi pupil model
  (`psf_methods.FourPiPupil`).
- The bead average there is the model-free `fourpi.iab_from_beads`
  (`FourPiAverage`, `fourpi_beads.average_model`): sub-pixel-centred boxes,
  backgrounds off.
- In situ (M6) will be the third.
- `psf_methods.for_mode` resolves a method in a mode.  A combination that
  does not exist switches the other box.

**The two 4Pi models compared on the 4Pi-SMS example** (5 beads; the Nup96
movie, 3000 frames, *4Pi z* windows of 500):

| | bead average | pupil model |
|---|---|---|
| Time to build | 5 s | 3.6 min |
| Its own beads, fitted back: z slope | 1.00–1.01 | 0.90–0.99 |
| Its own beads, fitted back: scatter | 2–4 nm | 9–22 nm |
| Movie: likelihood per pixel | −0.668 | −0.750 |
| Movie: envelope scatter / its CRLB | 4.9 | 1.09 |
| Movie: median z error (phase) | 3.6 nm | 5.1 nm |

- The bead average is built from its own beads, so it describes them
  closely.
- On new data it fits the pixels better, but claims too much precision:
  a model of five beads carries their noise in its slopes.
- The pupil model's errors are honest.
- The expected wrong-fringe rates differ for the same reason (calibrated
  envelope errors), and need a ring or simulation check before they can
  be compared.

## 15. The bead average per channel (2026-10-27)

**The question.**  The 4Pi bead average (§14) took the four channels to
share one PSF, differing only in brightness and phase offset.  That hides
any difference between the arms, which the fitter's per-channel models are
there to keep.  Was it the cause of the over-confidence on the movie?

**Per channel without mirror steps.**  Per voxel and channel the light is
linear in I_c, A_c, B_c.  There is one equation per bead (and per mirror
step), at the bead's own phase at that slice.  Beads at different phases
determine it; `iab_from_beads(channels="each")`.

**Simulated scans** (`model_calibration`, 5 beads at 40,000 photons, 20 nm
slices; z of 300 simulated molecules of 4000 photons fitted back):

| channels | model | smoothing | z spread | claimed error |
|---|---|---|---|---|
| matched | truth | | 15.8 | 15.3 |
| matched | per channel | 10 nm | 24.9 | 7.1 |
| matched | shared | 10 nm | 19.3 | 11.3 |
| matched | per channel | 60 nm | 15.9 | 15.9 |
| matched | shared | 60 nm | 15.9 | 16.0 |
| 2 of 4 at half the astigmatism | truth | | 16.4 | 16.3 |
| 2 of 4 at half the astigmatism | per channel | 10 nm | 32.0 | 6.6 |
| 2 of 4 at half the astigmatism | shared | 10 nm | 28.4 | 11.3 |
| 2 of 4 at half the astigmatism | per channel | 60 nm | 16.8 | 17.0 |
| 2 of 4 at half the astigmatism | shared | 60 nm | 20.9 | 17.6 |

What this shows:
- **The over-confidence was noise.**  A model of a few beads is noisy, and
  its noise reads as information: the claimed error shrinks while the
  scatter grows.  This held for both solves, and twenty beads only halved
  it.
- **Smoothing along z by 60 nm takes it out.**  Both solves then match the
  truth.
- **Per channel then wins where the arms differ.**  So `iab_from_beads`
  smooths by 60 nm (`smooth_nm`) and solves per channel where every
  slice's least squares has a condition number below 10 (*auto*).
  Otherwise it solves shared.
- **Each bead is scaled to the beads' mean brightness** first.
- **A bead whose focus lies more than a quarter of the scan from the
  median is dropped.**

**Mirror steps** (`steps`) enter the angle.  Which way round they act is
found as for the pupil model, by trying both: the smaller residual wins.
They make every slice determined from one bead.

**On real data:**
- **4Pi-SMS beads** (5, no steps): the worst slice's condition number is
  23, so *auto* solves shared.  Forced per channel, the movie's likelihood
  per pixel fell from −0.66 to −1.11.
- **Shared with 60 nm smoothing** on the movie:
  - envelope scatter / error 2.5, from 4.9 with 20 nm smoothing (the pupil
    model: 1.1);
  - median z error 3.4 nm;
  - expected wrong fringes 7.2 %.
  - It is still over-confident: five beads are few.
- **uiPSF's 4Pi example beads** (29 found in 6 scans, 3 mirror steps, 50 nm
  slices):
  - condition 1.45, so per channel; 106 s;
  - fringe period 250 nm (uiPSF v1's model: 260 nm);
  - fitted back to their own beads: z slope median 1.01 (0.85–1.11),
    scatter 2–7 nm, phase slope 0.99–1.02.

**The z range: the union of the beads, not their overlap** (2026-10-28).
The models were compared on uiPSF's example movie (`4pi_smlm_nup96`,
4000 frames; the scratchpad comparison, `uipsf_cmp/`).
- **The problem.**  The first version kept only the slices every bead
  covers once their foci are lined up.  With 110 beads whose foci spread
  over 7 slices, that left −267 to +232 nm, against a cell of about
  −500 to +450 nm.  Its fits piled up at the model's ends: 36.8 % expected
  wrong fringes, against 11–12 % with the pupil models.
- **The fix.**  Each slice is now solved with the beads scanned through it,
  as far as a quarter of them reach.  *auto* decides per slice.  The
  brightness is measured over the slices all beads share.
- **The result** (26 scans): −416 to +432 nm, 21.3 % expected wrong
  fringes, likelihood per pixel −0.5590 (pupil models −0.5567).
- **What is left is the scan's own length.**  21 slices of 50 nm reach
  ±500 nm, and a model-free average cannot reach past what was scanned; a
  pupil model extrapolates to ±600 nm, where the cell still has molecules.
  So for whole cells the pupil model is the one to use, and the bead
  average wants scans longer than the sample is deep.
- **Within ±200 nm, where every model is defined,** the averages match the
  pupil models: 10.2–11.5 % wrong fringes against 10.3 % (v2 pupil) and
  10.25 % (v1 bead), and the same likelihood.
- **Also seen:** uiPSF v1's bead and in-situ models put z the opposite way
  round (their envelope z anticorrelates at −0.96; the bead model's
  parameters say `stage_mov_dir: reverse`), while their phases agree.

## 16. Simulated 4Pi cameras and live DCIMG (2026-10-27)

**Simulate's 4Pi camera** (*camera frames* > *4Pi*; the preset *4Pi nuclear
pores (Nup96)*):
- Four phase images side by side.  Each molecule is drawn in all four with
  the model's I, A, B at phase = phase_per_nm · z + the cavity's phase,
  drifting at *cavity drift* (0.04 rad/s), through `fourpi.add_frames`
  (`render_frames` without its noise).
- The model is the *PSF calibration* if it is a 4Pi model, with its layout
  and maps.  Otherwise it is the closed-form `model_calibration`, at 21 px,
  40 nm slices and ±800 nm (50 MB of spline), written beside the recipe as
  `<name>_4pi.h5`.
- The source carries its `layout`, as the quadrant files do, so Spline 4Pi's
  *auto* takes it.  `recipe_calibration` gives `("fourpi", model)`, and the
  fitter's `react` fills the model and the registration "model"
  (`_FitPlugin.calibration_kind`).
- `ChannelSource` reads through to the wrapped source for what it lacks: the
  simulation's camera, settings and truth.
- **On the preset**, 1500 frames: 4289 localizations, the rings at −30 and
  +29 nm (10th and 90th percentiles; truth ±25), envelope scatter 1.06× its
  error, 2.3 % expected wrong fringes.  With 300 frames the cavity windows
  are too thin, and over 10 % land one fringe off.

**Live DCIMG.**
- In a new-format file each frame's 32-byte footer follows its pixels and
  holds its frame number and time.  `DcimgFile` counts the frames whose
  numbers run on from the first (and have a time, where the camera records
  any).  It does not trust the header's count, which a file being written
  may not have yet.  A file grown ahead of its frames reads zeros there,
  which fail.
- `DcimgSource.watch` is `watch_stack`'s stream (poll, idle timeout, the
  writer's *finished*, idle blocks), with the series listed again each time.
- Old-format files keep the stamps at the end, and are refused live.
- The fit reads a live stream through `_FitPlugin.watch_blocks`.  Spline 4Pi
  takes it from its `ChannelSource`, so the channels are cut as the file
  grows.
- Channel drift is refused live: it needs the whole acquisition.
- **Assumed, not checked on a camera:** that DCAM writes a frame's footer
  after its pixels and leaves the header's count until the end.  The test
  writes a file that way (`_growing_dcimg`).

## 17. The vectorial forward model (M7, 2026-10-27)

**The light path** (Wang et al. 2021, Nat Protoc 16:677, "Optical
overview" and "QWP setup"):
1. A quarter-wave plate right after each objective, its axes at 45° to the
   setup plane, both plates aligned with each other.
2. The deformable mirror, then a wedge pair.  In one arm it is quartz with
   its axes along s and p, a variable retarder between s and p tuned to
   about π/2; in the other it is BK7, for dispersion.
3. The non-polarising 50:50 cube: s interferes with s, p with p, and its two
   outputs are π apart.
4. A polarising cube: four images, s and p of each output.

**The model** (`uipsf.model.fourpi`, `kind="vector"`):
- Each arm's six dipole fields (`VectorPSF`, x and y components of the x, y
  and z dipoles) are turned into the channel's polarisation by the plates'
  Jones rows.  Aligned plates give (E_x ∓ i E_y)/√2 in both arms: the
  circular components.  `"crossed"` gives the other handedness in the lower
  arm; `"none"` gives E_x and E_y.
- In the sample's frame the lower objective sees the axial dipole's field
  reversed, besides z → −z: the wave travels down, and its field's
  projection of μ_z changes sign.
- A freely rotating emitter is the incoherent sum over the three dipoles,
  so the cross term is Σ_d u_d* l_d.
- One polarisation's image is taken as the whole of a channel's light (×2),
  so the shares f_c mean what they mean in the scalar model.
- The retarder and the cube's π stay inside the free δ_c; the coherence α_c
  is now only what the optics do not explain.
- `fourpi_beads.learn(kind="vector", plates=...)`.

**What the optics do to the modulation.**  The figure below is the
photon-weighted per-pixel visibility, Σ|cross| / Σ ½(|h_up|² + |h_lo|²),
for an aberration-free emitter on the axis (NA 1.35, n_imm 1.406, 670 nm,
100 nm pixels).  The z values are in nm:

| model | −600 | −300 | −150 | 0 | 150 | 300 | 600 |
|---|---|---|---|---|---|---|---|
| scalar | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| vector, no plates (n_med 1.406) | 0.97 | 0.89 | 0.76 | 0.65 | 0.76 | 0.89 | 0.97 |
| vector, aligned plates (n_med 1.406) | 0.97 | 0.88 | 0.74 | 0.61 | 0.74 | 0.88 | 0.97 |
| vector, crossed plates (n_med 1.406) | 0.16 | 0.21 | 0.22 | 0.23 | 0.22 | 0.21 | 0.16 |
| vector, aligned plates (n_med 1.33) | 0.95 | 0.85 | 0.69 | 0.53 | 0.69 | 0.85 | 0.95 |

What this shows:
- **On the axis every model interferes fully.**  The loss is off the
  axis: at focus, 200–400 nm out, the axial dipole's ring overlaps the
  in-plane dipoles' spot, and its interference has the opposite sign.  So
  the modulation is lowest at focus and recovers with defocus, where the
  spot spreads.
- **The plates do not set the contrast.**  Aligned plates and no plates
  modulate alike.  The plates are there for equal s and p light whatever the
  dipole's orientation.  Behind aligned plates, s and p of a freely rotating
  emitter give the same image (to 0.14 % of the peak, the pupil grid's
  sampling).
- **Crossed plates would kill the interference on the axis.**  The data
  interfere strongly, so the plates are aligned, as the protocol says.
- **A sample of lower index lowers the modulation further** (0.53 at
  focus in water): the supercritical angles are lost.

**Against the bead averages** (model-free I, A, B, the same weighted
visibility within 4 px):
- **uiPSF's example, 26 scans, per channel:**
  - z = −300: 0.45–0.48
  - focus: 0.42–0.45
  - z = +150: 0.42–0.47
- **4Pi-SMS, shared solve:** 0.59 at focus; 0.53–0.64 elsewhere, with no
  clear minimum at focus.

Reading these:
- The direction is the predicted one on uiPSF's example, but the dip is far
  shallower.
- The aberration-free prediction is not the right comparison.  The
  deformable mirrors add astigmatism, of opposite sign in the two
  objectives (as v1 found).  Away from focus the two arms' images then
  differ in shape, which lowers the modulation there in every model and can
  cancel the vectorial rise.
- The test is to learn both models on the same beads, each with a free
  coherence, and compare the objective, the coherence each needs, and the
  fits of a movie.

**Learnt on the 4Pi-SMS beads** (5 beads, no mirror steps, n ≤ 6, 680 nm;
the objective is the deviance, lower is better):

| model | objective | coherence | movie: likelihood per pixel | wrong fringes |
|---|---|---|---|---|
| scalar | 789,394 | 0.44–0.47 | −0.750 | 26.2 % |
| vector, aligned plates, axial reversed | 796,771 | 0.47–0.49 | −0.763 | 30.8 % |
| vector, channels s, s, p, p | 797,143 | 0.47–0.50 | | |
| vector, axial not reversed | 798,318 | 0.49–0.51 | | |
| vector, no plates | 804,634 | 0.49–0.52 | | |

What this shows:
- **Among the vectorial models the physical one wins**: aligned plates,
  the axial field reversed, s and p alternating.  So the Jones calculus is
  likely right.
- **The scalar model still fits these beads better**, by 7,400 in about
  890,000 pixels.
- **Its coherence is higher**, as it should be: in the vectorial model α is
  only what the optics do not explain, the modulation being α times the
  geometric visibility.
- **The likeliest reason it fits worse** is the predicted dip of the modulation at focus,
  which the bead average of these beads does not show (0.59 at focus, flat
  within ±0.05).  Five beads without mirror steps pin the modulation poorly,
  so the mirror-step beads of uiPSF's example are the test that decides.

**Learnt on uiPSF's example beads** (6 scans, 29 beads, 3 mirror steps per
slice, 50 nm slices; NA 1.35, 600 nm, n ≤ 6; on Apple's GPU):

| model | objective | coherence | mirror steps | time |
|---|---|---|---|---|
| scalar | 7,091,872 | 0.71–0.77 | −2.06, −4.17 | 10.6 min |
| vector, aligned plates | 6,815,737 | 0.75–0.81 | −2.05, −4.17 | 9.8 min |

What this shows:
- **Here the vectorial model fits clearly better**: 3.9 % less deviance
  (276,000).  Mirror steps measure every slice's modulation, so the
  z dependence the optics predict can show, and it does.
- **It explains part of the lost modulation as geometry**, and needs more
  coherence for the rest.
- **It costs no more time to learn.**  Six fields instead of one: the
  matrix DFTs are batched, and the solver's other costs dominate.
- **The 4Pi-SMS beads disagree**, but they are five, without steps.  The
  vectorial model is the better description where the data can tell.

**Both models, exported, on uiPSF's example movie** (`4pi_smlm_nup96`,
4000 frames, Spline 4Pi with the models' maps, *4Pi z* windows of 500):

| model | localizations | likelihood per pixel | median z error | expected wrong fringes | envelope scatter / error | period |
|---|---|---|---|---|---|---|
| scalar | 34,654 | −0.5567 | 9.2 nm | 11.96 % | 1.24 | 265.9 nm |
| vector | 34,443 | −0.5562 | 8.9 nm | 11.08 % | 1.23 | 264.1 nm |

The vectorial model is better on every count, by a little.  The fitter
reads the IAB export, sampled from the model, so the gain is in the
model's I, A and B: the modulation's shape and its change with z.

**Cost, and a warm start** (2026-10-28; uiPSF's example, 6 scans, 29 beads, the 40 nm bead taken out,
n ≤ 6, Apple's GPU):

| model | time | iterations (first stage) | objective | coherence |
|---|---|---|---|---|
| scalar | 13.2 min | 76 (36) | 7,089,164 | 0.72–0.79 |
| vector, from scratch | 10.7 min | 56 (35) | 6,812,561 | 0.77–0.82 |
| vector, from the scalar fit | 7.8 min | 34 (–) | 6,827,283 | 0.76–0.81 |

What this shows:
- **The vectorial model costs no more.**  One forward pass over the 29 beads takes 0.16 s against the
  scalar's 0.17 s: the six fields go through one batched matrix DFT.  It also needs fewer iterations.
- **A warm start from the scalar fit does not pay.**  It is quicker on its own, but scalar and warm
  vector together take 21 min against 10.7, and it stops 14,700 worse: the scalar optimum is not near the
  vectorial one.  `learn_fourpi(values=...)` keeps it possible.
- **The Zernikes agree within about 0.05 rad rms**, except the split of the astigmatism (Noll 6)
  between the objectives:

  | model | upper | lower | difference | mean |
  |---|---|---|---|---|
  | scalar | +0.39 | −0.41 | 0.80 | 0.00 |
  | vector, from scratch | +0.59 | −0.23 | 0.82 | +0.18 |
  | vector, warm | +0.50 | −0.32 | 0.82 | +0.09 |

  The difference between the arms is pinned to 0.80–0.82 rad; their mean is not.  In the vectorial
  model it trades with something else, most likely the circular components' azimuthal winding, which
  astigmatism common to both arms resembles.  A common astigmatism of the order of 0.1–0.2 rad is
  therefore not to be read from a single fit.

## 18. The bead scan's z step and range (2026-10-28)

**The question.**  How coarse can the z step of a 4Pi bead scan be, with 3 mirror steps per slice,
before the learnt model gets worse?  50 nm is the plan; 100 nm if it costs nothing.  Then the
scan's range, at that step.

**The simulation** (`benchmarks/fourpi/zstep.py`):
- Bead stacks are drawn from a known scalar 4Pi model: uiPSF's example optics, opposite astigmatism
  in the two arms, coherence about 0.8, 40 nm beads, 12 beads.
- Each bead's photons and background are the same whatever the step: 3·10⁶ photons per bead in
  all, and 20 photons of background per pixel and frame at 10 nm steps.  Fewer slices get brighter
  frames.
- A model is learnt from each stack and exported.  It is judged on 3000 simulated molecules (3000
  photons, background 10, z within ±400 nm) drawn with the true model and fitted with the learnt one:
  - z from the phase, as *4Pi z* makes it (rms and slope);
  - the envelope's z;
  - wrong fringes.
- **Caveat:** the truth is the same kind of Zernike model that is learnt, so this measures what the
  noise costs, not unmodelled structure.  `benchmarks/fourpi/zstep_real.py` covers that on uiPSF's
  real 50 nm scans: every second slice against all, judged on held-out beads.

**The step** (±500 nm, seeds 0–2 at 3·10⁶ photons per bead, 0–1 at 3·10⁵):

| step | slices | learning | z from phase, rms | envelope z, rms | wrong fringes | period |
|---|---|---|---|---|---|---|
| true model | | | 2.392 nm | 9.02 nm | 0 % | 287.16 nm |
| 20 nm | 51 | 11.5 min | 2.392–2.400 nm | 9.02–9.03 nm | 0 % | 287.15–287.29 nm |
| 50 nm | 21 | 3.8–5.0 min | 2.391–2.392 nm | 9.02–9.03 nm | 0 % | 287.13–287.14 nm |
| 75 nm | 14 | 2.0–2.8 min | 2.392 nm | 9.02 nm | 0 % | 287.15–287.16 nm |
| 100 nm | 11 | 1.4–1.9 min | 2.392–2.393 nm | 9.02 nm | 0 % | 287.18–287.21 nm |
| 250 nm | 5 | 0.8 min | 71 nm | 54 nm | 13.6 % | 256 nm |
| *3·10⁵ photons per bead:* | | | | | | |
| 20 nm | 51 | 12–17 min | 2.392–2.394 nm | 9.02–9.07 nm | 0 % | 287.05–287.08 nm |
| 50 nm | 21 | 4.6 min | 2.392 nm | 9.03–9.04 nm | 0 % | 287.06–287.08 nm |
| 75 nm | 14 | 2.0–2.8 min | 2.394–2.395 nm | 9.02–9.05 nm | 0 % | 287.11–287.23 nm |
| 100 nm | 11 | 1.3–1.9 min | 2.395–2.400 nm | 9.02–9.05 nm | 0 % | 287.22–287.33 nm |

- With bright or ten times dimmer beads, 20 to 100 nm steps all give the true model's performance;
  at 3·10⁵ photons the scatter between seeds (0.03 nm in envelope z) is as large as any trend.
- The model breaks down between 200 nm (6 slices, still exact in a first trial) and 250 nm.
- The pupils' coefficients differ by up to 0.06 rad between fits with no pattern by step: equivalent
  descriptions of the same images.

**Real beads** (`zstep_real.py`, uiPSF's 50 nm scans, scalar model, 20 beads to learn, 10 held out):

| step | slices | held-out deviance per pixel | coherence per channel | learning |
|---|---|---|---|---|
| 50 nm (all) | 21 | 1.963 | 0.79 / 0.75 / 0.74 / 0.72 | 11.2 min |
| 100 nm (every second) | 11 | 1.941 | 0.79 / 0.75 / 0.75 / 0.73 | 7.1 min |

The model from every second slice predicts the held-out beads at least as well: real beads show no
structure along z that 100 nm steps miss.

**The range**, at 100 nm steps, with the same photons and background per bead whatever the range (a
scan of the same duration), seeds 0–2; molecules judged within ±400 nm and, beyond the shortest scan,
within ±600 nm:

| scan | slices | learning | z from phase, ±400 / ±600 | envelope z, ±400 / ±600 | wrong fringes |
|---|---|---|---|---|---|
| true model | | | 2.392 / 3.011 nm | 9.02 / 9.70 nm | 0 % |
| ±300 nm | 7 | 1.3–1.7 min | 2.392–2.400 / 3.011–3.024 nm | 9.02 / 9.70–9.74 nm | 0 % |
| ±500 nm | 11 | 1.5–2.0 min | 2.392–2.393 / 3.011–3.012 nm | 9.02 / 9.70–9.71 nm | 0 % |
| ±750 nm | 16 | 4.0–7.2 min | 2.391–2.394 / 3.011 nm | 9.01–9.03 / 9.70–9.71 nm | 0 % |
| ±1000 nm | 21 | 5.9–6.5 min | 2.392–2.393 / 3.009–3.012 nm | 9.01–9.03 / 9.70–9.72 nm | 0 % |

- Every range gives the true model's performance, even ±300 nm judged out to ±600 nm: a Zernike
  pupil extrapolates in z, because the defocus it adds is fixed by the optics, not learnt.
- ±300 nm is the only one with a visible spread between seeds beyond its range (0.04 nm in envelope
  z at ±600), and the only one whose period scatters by 0.13 nm.
- This again assumes the true PSF is a pupil model.  Real aberrations that a pupil does not describe
  (a bead's halo, scattered light) are learnt only where the scan reaches, so a scan should cover the
  depth one images.

**Conclusion:** 100 nm steps over the depth to be imaged lose nothing
measurable against 20 nm in simulation, and against 50 nm on uiPSF's real beads; they learn 6–9× faster.  50 nm
buys nothing; above 200 nm the model breaks.

## 19. Speed, the two objectives' Zernikes, and the vectorial default (2026-10-28)

**The linearisation.**  `Problem` builds the Gauss–Newton matrix column by column.  The 4Pi model had no
linearisation, so each of its ~220 columns was a JVP through the whole model, 2.9 forward passes each.
It now has the two the one-objective model has (`FourPiModel.pupil_linearisation`,
`emitter_linearisation`):
- **A pupil column** is one matrix DFT per channel of the tangent field, then the product rule on the
  four images: d|u|² = 2 Re(u*·du), d(u*·l) = du*·l + u*·dl.  The channels share z, so the pupil-plane
  product, the bulk, is formed once per arm and only the DFT's x and y differ
  (`VectorPSF.field_linearisation`).
- **Everything else but the blur** goes through nine coordinates of each observation in each channel:
  x, y and z, the interference phase ψ, the balance r_c, the coherence α_c and the misalignment Δ.  Their
  derivatives are PSF tangents (x, y, z and Δ: the position's tangent pupils, the lower arm's z and Δ
  reversed) or exact products after the blur (ψ, r, α).  So the emitter's parameters, the photons per
  frame, the shares, mirror steps, channel offsets and maps, the balance, the coherence and the
  misalignment all cost a product each; only the blur's two columns remain JVPs.
- The channels stacked, real arithmetic instead of complex, the four images mixed into one per mirror
  step before the bead kernel, a single field taken rather than summed over a length-1 axis (on Apple's
  GPU that sum took 0.3 s for eight tangents, the rest 0.01 s), and 8 tangents per batch (`tangents`).
- The Jacobian equals the JVPs' to 10⁻⁴ of each column's largest entry, scalar and vectorial, with and
  without a bead kernel, at random points and at the learnt optimum, on the CPU and Apple's GPU
  (`test_the_linearised_jacobian_is_the_jvps_one`).

One normal-equations pass, 29 simulated beads × 41 slices × 3 mirror steps, n ≤ 6, a 20 nm bead,
Apple's GPU:

| model | JVPs | linearised |
|---|---|---|
| scalar | 109 s | 9.1 s |
| vectorial | 110 s | 11.7 s |

On uiPSF v1's 18 training beads (31 slices × 3 steps, the long test's set) the whole learning took 0.5
min against 5.0 min, to the same optimum (objective 1 078 007, 64-69 iterations either way).

**Two of the changes help one objective too** (`VectorPSF`): the bead kernel's transform is real
(`rfftn`, exact, the image being real and the kernel even: 1.5× faster on the CPU, the same on Apple's
GPU), and the scalar model's single field is taken rather than summed (0.6 against 58 ms per batch of
tangents there).  The rest is particular to 4Pi: one objective's channels have pupils and foci of their
own, so there is no pupil-plane product to share.

**Two optima.**  The same learning through JVPs with 8 tangents per batch, instead of 1, reached a better
optimum, 1 063 226 (1.4 % lower), and a phase scatter of 0.053 rad, within the test's 10 % of v1's 0.051,
against 0.062.  Only the rounding differs between the runs -- their normal equations agree to 10⁻⁷ -- so
the optimisation has two basins, and which it reaches is chance.  The worse one puts the misalignment along y (Δ = (−28, 109, −10) nm); fits in §11
found it along x.  So `test_the_learned_model_measures_v1s_beads_as_straight_as_v1s_model` (long tier)
fails as the code stood before the linearisation too.  §20 settles it.

**How the two objectives' Zernikes trade.**  The covariance of the pupils, from the Gauss–Newton matrix
with the beads' own parameters taken out (Schur complement), at the truth of 12 simulated beads (3·10⁶
photons each, ±500 nm at 50 nm, 3 mirror steps, a 20 nm bead):
- **Piston, tip, tilt and defocus are not parameters of either pupil:** a relative piston is the beads'
  phase, a relative tip, tilt or defocus the misalignment Δ, and the common ones the bead's position.
  These trade exactly.
- **Every other mode is correlated between the arms, but not exactly:** |r| = 0.90-0.95 (scalar), 0.63-
  0.89 (vectorial).  One combination per mode is pinned by the interference (±0.3-0.7 mrad here), the
  other only by each arm's own, incoherent image (±0.8-2.6 mrad): 2.5-7× less well.
- **Which combination is which follows the mode's symmetry.**  For the even modes (astigmatism,
  spherical, quadrafoil, hexafoil) the difference up − lo is pinned and the mean loose (r > 0); for
  the odd ones (coma, trefoil, pentafoil) the mean is pinned and the difference loose (r < 0).  The lower
  objective looks from the other side, so the same aberration in both appears in its own coordinates
  as the twin of the upper's.
- **The vectorial model separates the arms better:** its correlations are lower, and most loose
  combinations are better determined than the scalar's, up to 2.4× (pentafoil), though secondary
  spherical is a little worse.  The weakest directions mix the arms' phase and magnitude of the same
  high mode (Noll 22-28).
- This is why §17 found the astigmatism difference pinned at 0.80-0.82 rad and its mean loose: the mean
  is the loose combination for an even mode, and with real data a mismatch of the model goes there first.
  Learning each objective on its own (one arm blocked) measures exactly the loose combination, so a
  joint fit with single-objective scans beside the 4Pi ones should pin both.

**The window.**  The pupil model is vectorial by default (*vectorial*, under *4Pi*; off: uiPSF v1's
scalar model).  The *Pupils* page shows each objective's magnitude beside its phase, and a third column,
what the two arms leave together: W_up(k) + W_lo(−k) (the magnitude's difference), zero where the lower
pupil is the upper one's twin and the arms make the same PSF at every height -- the even modes' sum, the
odd ones' difference, so the mirrors' opposite astigmatism cancels there and equal comas do not
(`fourpi_beads.arms_mismatch`).  4Pi scans added in the single-channel mode turn the window to *4Pi*.
Read as one channel's bead stack they had failed with "not a TIFF file". Interference phases -- the
channels' offsets, the mirror steps, the cavity's phase in *4Pi z* -- are shown in multiples of π
(`fourpi.in_pi`, `pi_axis`), aberrations in nm (`uipsf.diagnostics.figures.PHASE_UNIT`); columns and
files keep radians.


## 20. One optimum: the misalignment's start and the magnitude (2026-10-28)

The learning on uiPSF v1's 18 beads (31 slices × 3 steps, a free photon count per observation, n ≤ 6) from
several starts of the misalignment Δ, each a whole learning (stage 1 to n = 4, then all), with the
long test's phase and z scatter of the beads fitted back (v1's model: 0.0506 rad, 6.76 nm).

**The start.**  Δ was held at 0 through stage 1 -- the symmetric point of the arms' gauge, where it has no
curvature -- and freed in stage 2, where it broke away in whichever direction the rounding tipped it.
Freed in stage 1 from an offset start, every start reached one stage-1 optimum or its swap (Δ → −Δ),
1 080 227 against 1 215 094 held:

| scalar model | objective | Δ (nm) | phase (rad) | z (nm) |
|---|---|---|---|---|
| Δ held at 0 in stage 1 (before) | 1 078 007 or 1 063 226, by rounding | (−28, 109, −10) | 0.062 | |
| stage 2 from Δ = (100, 0, 0) | 1 056 993 | (116, 5, −3) | 0.0512 | 7.93 |
| Δ free in stage 1 from (±60, 0, 0) or (0, ±60, 0) | 1 060 569 | ±(100, 49, −33) | 0.0543 | 7.12 |

The last is now `learn_fourpi`'s (`OFFSET_START_NM` = (60, 0, 0), only the balance held in stage 1): the
same optimum by JVPs and linearised, with 8 tangents or 1, and the long test passes again.  The lowest
objective found, 1 056 993, came only from one start in stage 2 and has z 17 % worse than v1's.

**The vectorial model still had several, with the magnitude free in every mode.**  Its magnitude was the
cause, as the optics suggest (an objective's magnitude is its apodisation and transmission, with little
structure).  Final objectives from Δ free in stage 1 from 0, (60, 0, 0) and (0, 60, 0), for the pupils'
magnitude in

| magnitude | objectives | phase (rad) | z (nm) |
|---|---|---|---|
| every mode to n = 6 (27 each) | 914 133, 922 288, 907 711 | 0.052-0.054 | 5.8-6.5 |
| to n = 4 | 922 326, 931 317, 922 938 | 0.052-0.054 | 6.2-6.3 |
| to n = 2 | 940 373, 963 980, 940 373 | 0.053 | 6.4-6.8 |
| rotationally symmetric, Noll 4, 11, 22 | 914 132 from all three | 0.0527 | 6.58 |
| uniform | 1 039 166 from all three | 0.0561 | 6.37 |

The rotationally symmetric magnitude (`uipsf.optics.zernike.symmetric_modes`, 3 coefficients per
objective against 27) reaches one optimum, 0.7 % above the best of the free ones and within the free ones'
phase and z scatter; a uniform magnitude loses 14 %.  So the window's vectorial model (`fourpi_beads.learn`)
has the symmetric magnitude, and so has `FourPiModel`, now vectorial by default (scripts and benchmarks
that want v1's model say ``kind="scalar"``).  The scalar model keeps every mode: it
uses them, presumably for what its scalar fields leave out -- symmetric 1 107 108 (phase 0.0563, z 7.71),
uniform 1 344 285 -- and with Δ's new start it reaches one optimum anyway.

The vectorial learning with the symmetric magnitude took 0.4 min.  Not looked at: whether the high
modes' phase has optima of its own on other beads, and the 4Pi-SMS's DCIMG beads.


## 21. The pupils in Hao et al.'s 4Pi modes (2026-10-29)

**The idea.**  Hao et al. (2017) describe the two pupils by modes Q±_j: Z_j in the upper pupil and ±Z_j in
the lower, in coordinates that mirror x and y between the objectives and flip the phase with the direction
of travel.  It is a rotation of the two arms' coefficients by 45°, mode by mode.  In `FourPiModel`'s frames,
each objective's own, it is

    q+_j = (c_up − s_j c_lo)/√2,   q−_j = (c_up + s_j c_lo)/√2,   s_j = (−1)^|m_j|.

For Noll 1-4 the model already parametrises this way: q+ is the bead's phase (piston) and position (tip,
tilt, defocus), q− the misalignment Δ, and q−_1 does nothing.  §19's split of the higher modes -- the
difference pinned for even m, the mean for odd m -- is the same one, so q+ are Hao's covariant modes and
the ones the interference pins.  The paper simulates the modes' foci only; its claim that they help phase
retrieval is not tested there.

**The test** (`benchmarks/uipsf/real/zenodo_4pi_mode_basis.py`): `ModeBasisFourPi` reads ``pupil.phase``
as q+ and ``lo.pupil.phase`` as q− (the magnitudes stay per objective), so `learn_fourpi`'s truncation
and the linearised Jacobian run unchanged.  Its images and Jacobian equal the per-objective model's to
10⁻⁷.  §20's setting: v1's 18 beads, a free photon count per observation, vectorial, the symmetric
magnitude, n ≤ 6, Apple's GPU.

| basis | Δ start (nm) | iterations (stage 1 + 2) | time | objective | phase (rad) | z (nm) |
|---|---|---|---|---|---|---|
| per objective | (60, 0, 0) | 29 + 15 | 28 s | 914 133 | 0.0527 | 6.59 |
| q± | (60, 0, 0) | 29 + 14 | 28 s | 914 133 | 0.0527 | 6.58 |
| per objective | (0, 60, 0) | 26 + 13 | 25 s | 914 132 | 0.0527 | 6.58 |
| q± | (0, 60, 0) | 20 + 13 | 25 s | 914 133 | 0.0527 | 6.59 |
| per objective | 0 | 26 + 16 | 34 s | 914 132 | 0.0527 | 6.58 |
| q± | 0 | 26 + 15 | 32 s | 914 133 | 0.0527 | 6.58 |

Every run reaches the same optimum: Δ = (−17, −14, 143) nm and the same coefficients to 0.01 rad.

**What this shows:**
- **The basis does not make the learning more robust or faster.**  The optimum was already unique after
  §20, and the damping by diag H, which the rotation does change, costs at most a few iterations.
- **On real beads, q± is the basis in which the pupils decouple.**  At the optimum, with the beads' own
  parameters taken out (Schur complement), the arms correlate at |r| = 0.48-0.85: positively for even m
  and negatively for odd m, as §19 found on simulated beads.  In q± no pair correlates by more than
  |r| = 0.19.
- **q+ is 1.6-3× better determined than q−:** ±0.3-0.6 mrad against ±0.7-1.1 mrad.  So the interference
  pins q+, and q− is fixed only by each arm's incoherent image.
- **The physics reads off directly.**  The astigmatism that sets z's range is almost all q+:
  q+_6 = 0.58 rad and q−_6 = 0.11 rad, §17's difference of 0.82 rad.  The largest q− terms are
  spherical, q−_11 = 0.12 rad, and astigmatism; everything else is below 0.07 rad.

**So:** keep the model per objective.  Use q± where it says more than the two pupils: to report the
pupils with their errors (the *Pupils* page, the export's parameters), and as the place for a prior or a
truncation of q− if a data set ever has the loose directions run off.  Nothing here needed one.

**The first stage's orders split** (2026-10-29, the same script's ``--stage1 P,M``: q+ to radial order P
and q− to M in stage 1, ``none`` holding q− at 0, then everything; 4,4 is `learn_fourpi`'s own and gives
its numbers exactly).  Three Δ starts each, (60, 0, 0), (0, 60, 0) and 0.

*v1's 18 beads* (3 mirror steps):

| stage 1 | objectives | iterations | time |
|---|---|---|---|
| per objective, n ≤ 4 | 914 132-133 | 44, 39, 42 | 25-34 s |
| q± 4, 4 | 914 133 | 43, 33, 41 | 25-32 s |
| q± 6, 4 | 914 133 | 31, 30, 34 | 22-33 s |
| q± 4, none | 914 132-133 | 40, 31, 39 | 23-33 s |

The same optimum from every start, and the same phase and z scatter (0.0527 rad, 6.58 nm).  q+ to 6 in
stage 1 saves a quarter of the iterations; stage 1 then ends at 918 704 instead of 940 071.

*The 4Pi-SMS example's 5 beads* (`4pi_sms_nup96/Beads_stack`, 101 slices 20 nm apart, no mirror steps;
camera 0.5 e⁻/ADU, offset 100, 100 nm pixels, NA 1.35, 680 nm, as §10 and §17).  Two things had to be
found first:
- **The window's *4Pi-SMS (Bewersdorf lab)* channels do not fit these data.**  `channels.YALE_4PI` puts
  the channels at x = 151, 389, 1474, 1713; SMAP's calibration of the same beads
  (`Cell01_642_000_000_3dcal.mat`) at 93, 469, 1409, 1769.  With the preset, `scan_beads` registered the
  channels wrongly and kept 1 bead of 5; with SMAP's corners typed it keeps all 5 and registers them as §10
  did (affine, 0.06-0.10 px).
- **The blur σ falls to 0 on these beads, and the solver stalls there.**  σ enters squared, so at 0 its
  Jacobian column vanishes and Gauss–Newton's curvature of it with it, and every step's σ component is
  enormous (why, and the fix, at the end of this section).  Levenberg–Marquardt from the stalled point predicted 4 753 nats of gain and got 2.4 million
  more; L-BFGS's line search rejected every step.  7 of the 12 runs stopped stage 2 within 7 iterations,
  at 801 354-805 189, the others went on to 794 835-799 976.  Held at 0, σ stalls nothing.

With σ held at 0:

| stage 1 | from (60, 0, 0) | from (0, 60, 0) | from 0 |
|---|---|---|---|
| per objective, n ≤ 4 | 796 797 | 796 797 | 793 783 |
| q± 4, 4 | 796 797 | 796 797 | 793 783 |
| q± 4, none | 796 797 | 793 783 | 796 797 |
| q± 6, 4 | 798 982 | 798 982 | 798 982 |

- **Three optima.**  793 783: Δ = (−44, −93, 413) nm, coherence 0.63-0.68.  796 797: Δ = (51, 88, −437) nm,
  coherence 0.58-0.64.  798 982: Δ = (63, 41, −133) nm, coherence 0.45.  The first two are near-twins,
  Δ reversed: the arms' labels, which the vectorial model does not swap exactly.
- **The basis changes nothing:** per objective and q± 4, 4 reach the same optimum from each start.
- **No split of the orders gives one optimum.**  Holding q− at 0 only changes which twin a start reaches.
  q+ to 6 reaches one optimum from every start, but the worst of the three, 5 200 nats above the best.
- **Five beads without mirror steps** leave Δz free to run to ±400 nm, three times v1's 143 nm.  Mirror
  steps measure the modulation in every slice, and only then did §20's start reach one optimum.

**So, for the orders:** splitting them is not worth adding.  It saves a quarter of the iterations on
beads that already have one optimum, and on beads that don't it gives none.  What the 4Pi-SMS beads need is
a σ that cannot stall the solver, and either both of Δz's signs tried, keeping the better one, or mirror
steps.

**The σ stall, fixed (2026-10-29).**  The model sees the blur only as σ², so the objective is
F ≈ F₀ + a·σ² near σ = 0, with curvature 2a.  Gauss–Newton's curvature of σ is 4σ²·(JᵀWJ in σ²), which
vanishes there, while the gradient 2aσ vanishes only as fast as σ: the Gauss–Newton step is
−a/(2σ·JᵀWJ), ∝ 1/σ.  The missing term is exact and costs nothing: ∂²F/∂σ² = 4σ²·∂²F/∂(σ²)² + 2·∂F/∂(σ²),
and 2·∂F/∂(σ²) = g_σ/σ.  `Problem.normal_equations` (and the matrix-free operator) now add it, where
positive, for every parameter whose transform says the model sees it squared (`Transform.squared`,
`Magnitude`).  On simulated beads near σ = 0 H's σ entry then matches the objective's finite-difference
curvature (0.01198 against 0.01198 at σ = 1 nm; 0.00024 without), `test_solve.py::
test_the_blurs_curvature_is_the_objectives_where_gauss_newton_alone_has_none`.

With σ free again, per objective, n ≤ 4 in stage 1:

| beads | from (60, 0, 0) | from (0, 60, 0) | from 0 | before the fix |
|---|---|---|---|---|
| 4Pi-SMS | 796 797, 47 it. | 796 797, 52 | 796 797, 56 | 805 189, 799 976, 805 188 |
| v1 | 914 132.6, 41 | 914 132.4, 41 | 914 132.7, 38 | 914 132-133, 44, 39, 42 |

On the 4Pi-SMS beads every start reaches one optimum now, σ = 0, the 796 797 twin (Δz = −437 nm); held at 0
before, the start from 0 had found the better one, 793 783, so the twins remain, and trying both Δz signs
is still to do.  v1's beads, whose σ is 57 nm, are unchanged.  The term is in uipsf's solver, so it serves
every model with a blur: on simulated dual-colour beads whose σ is 0, LM without it ran to its 100
iterations (biplane and ratiometric) and L-BFGS stopped on "step", 0.3 nats short in ratiometric; with it
both converge, LM in 10 iterations.  SMAPpy's own fitters are not concerned: the spline models fit no
width, and the Gaussian ones fit σ itself, not σ².



## 22. The window's estimate, its bead finding and a round blur (2026-10-29)

**The time estimate.**  The window said 3.5 min per 505 bead frames (slice × mirror step), measured before
§19's speed-ups, and counted every bead found, excluded or not.  On the Zenodo 4Pi beads (21 slices × 3
steps, Apple's GPU, n ≤ 6, vectorial) the learning now takes 21, 31, 30, 73 and 90 s for 4, 8, 16, 32 and
41 beads (252 to 2583 frames): about 15 s and 0.03 s per frame (`psf_methods.FourPiPupil`).  Ten beads of
101 slices and 3 steps are about 2 min, where the window said 21.  It counts the beads not marked out.

**Finding the beads** has the settings the bead calibration has: a *peak threshold* (0.2 of the brightest,
as before), an *isolation* (both of a pair closer than the box left out; before, only maxima within 6 px
were merged) and an *outlier factor* (2.5: a bead that far from the median brightness is marked excluded,
not dropped).  With the isolation, the first ten Zenodo scans give 41 beads where they gave 47.

**A round blur** (`FourPiModel(round_sigma=True)`, the window's default): one σ for x and y.  On 8 Zenodo
beads it learnt 56 nm.  An x and a y σ apart are astigmatism, which belongs to the pupils.

**The magnitude modes** can be set (*magnitude modes*; empty or *none* is a flat magnitude, *auto* the model's
default), and both objectives can share one magnitude (*same magnitude*, `FourPiModel(same_magnitude=True)`):
the lower pupil has no ``lo.pupil.magnitude`` and reads the upper one's, each in its own frame -- for the
rotationally symmetric modes the frames do not matter.  Its linearised Jacobian column is the sum of the two
arms' (`Problem._shared_columns` adds the groups of a pupil every channel sees).  On 8 Zenodo beads, vectorial,
n ≤ 6:

| magnitude | objective | time |
|---|---|---|
| per objective, Noll 4, 11, 22 | 661 647 | 21 s |
| shared, 4, 11, 22 | 661 731 | 14 s |
| shared, 4, 11 | 668 384 | 17 s |
| flat | 735 380 | 37 s |

Left free, the two objectives learnt nearly the same magnitude (−0.166 / −0.163, −0.072 / −0.065,
−0.081 / −0.053), and sharing it costs 84 in the objective for three coefficients fewer.  Noll 22 is worth
6 700, a flat magnitude 74 000.

**Rotating A and B by each channel's phase offset** would make B about 0 and A about the modulated part of
I, and would read better, but would not interpolate better: a cubic spline is linear in its knots, so the
spline of the rotated knots is the rotated spline, with the same error.  What is steep is the PSF across
the pixels, which only finer knots in x and y would follow.


## 23. The minimal pupil, and the objective's effective NA (2026-11-02)

The form's PSF model is now one objective's (`PSFModelSettings`, `docs/uipsf/architecture.md` "The minimal
model"): cos θ^γ, one γ for both objectives under *same magnitude* (on by default), σ round over the pixel's
blur.  The sample is index-matched, so there is no supercritical ring and no SAF damping; the *magnitude
modes* are shown only for a Zernike magnitude (`fourpi_beads.fourpi_shown`).  The export keeps both arms'
described pupils (`parameters["pupils"]`, `VectorPSF.describe`).

**A bug on the way.**  `FourPiModel` took ``pixel_blur`` and blurred by the learnt σ alone (`blur(p)`, not
`image_sigma(p)`), so "σ over the pixel's blur" was not true for 4Pi; fixed, with a test that the image with σ
over the pixel equals the image of their sum without it.  Models learnt without the pixel blur -- the old
default -- were not affected.

**Simulated** (`test_minimal_4pi_beads_give_back_their_gamma_sigma_pupils_and_phases`, long tier): six beads
drawn with the minimal model (γ 0.35, σ 35 nm over the pixel, astigmatism of opposite sign, coma, spherical, a
misalignment), learnt from the window's start.  Three noise draws: γ 0.343-0.352 (CRLB 0.007), σ 34.6-35.6
nm (0.3), every phase mode of both pupils, Δ and each bead's phase within 2 CRLB, the z-scores spread as a
unit normal's.

**Real** (`benchmarks/uipsf/real/zenodo_4pi_minimal.py`, `results/fourpi_minimal.jsonl`): uiPSF v1's 18
beads of Zenodo `4pi_40nm_bead`, through `fourpi_beads.learn`, vectorial, order 6, a photon count per
observation, Apple's GPU; each model exported and fitted back with `Global4PiPSF` -- z from the phase and from
the envelope against the stage, over the scan's middle 15 slices, and the envelope's bias against the
phase's z (median over beads per slice).  The beads have ~35 000 photons: *4Pi z* chose the right fringe
for every frame with every model, so its behaviour on beads says nothing between them; the envelope's bias is
what moves it for dim molecules, and at 2-4 nm against a 265 nm fringe it does not.

| model | NA (aperture) | objective | shared parameters | γ | σ (nm) | Δ (nm) | phase z rms / CRLB (nm) | envelope z rms / CRLB (nm) | envelope bias rms (max) (nm) |
|---|---|---|---|---|---|---|---|---|---|
| symmetric Zernike per objective, σ 60, no pixel blur (old default) | 1.35 | 916 143 | 98 | | 55.4 | (−18, −13, 142) | 2.21 / 0.67 | 6.77 / 2.63 | 1.97 (4.6) |
| symmetric Zernike shared, pixel blur | 1.35 | 916 509 | 95 | | 41.0 | (−17, −13, 141) | 2.20 / 0.67 | 6.76 / 2.63 | 2.00 (4.7) |
| minimal, γ per objective | 1.35 | 925 889 | 94 | 0.400, 0.405 | 41.1 | (−19, −15, 157) | 2.20 / 0.67 | 7.24 / 2.56 | 3.43 (8.9) |
| minimal, one γ (new default) | 1.35 | 925 889 | 93 | 0.403 | 41.1 | (−19, −15, 157) | 2.20 / 0.67 | 7.25 / 2.56 | 3.43 (8.9) |
| minimal, one γ | 1.33 | 913 057 | 93 | 0.304 | 42.1 | (−23, −13, 127) | 2.20 / 0.67 | 6.84 / 2.62 | 2.27 (5.4) |
| minimal, one γ | 1.32 | 911 215 | 93 | 0.240 | 42.8 | (−24, −13, 121) | 2.20 / 0.67 | 6.66 / 2.65 | 1.75 (3.7) |
| minimal, one γ | 1.31 | 913 129 | 93 | 0.173 | 43.4 | (−26, −14, 113) | 2.21 / 0.67 | 6.66 / 2.69 | 1.93 (4.3) |
| minimal, one γ | 1.29 | 929 199 | 93 | 0.031 | 44.7 | (−32, −16, 95) | 2.23 / 0.67 | 6.56 / 2.77 | 4.43 (9.8) |
| minimal, one γ | 1.26 | 1 012 371 | 93 | 0 (bound) | 45.9 | (3, 0, −67) | 2.22 / 0.66 | 6.34 / 2.95 | 12.6 (23.6) |
| symmetric Zernike shared, pixel blur | 1.32 | 910 379 | 95 | | 41.7 | (−20, −13, 141) | 2.19 / 0.67 | 6.80 / 2.64 | 1.85 (4.1) |
| minimal, one γ, aperture learnt (prototype) | 1.311 | 912 642 | 94 | 0.230 | 42.7 | (−25, −14, 107) | 2.21 / 0.67 | 6.62 / 2.65 | 1.77 (3.9) |

The fringe period of every export is 264.6-265.1 nm (the beads' own, §14, 251 nm), the phase's z runs at 0.99 of
the stage, and the coherence is 0.76-0.82 in every row.

- **One γ costs nothing**: 925 889 with one γ or two (0.400 and 0.405).  The pixel's blur and a shared
  magnitude cost the symmetric Zernikes 366 nats.
- **At the nominal NA the minimal pupil is 9 700 nats behind** the old default, and its envelope bends twice
  as much (3.4 nm rms, 8.9 nm at most).  The old fit's magnitude says why: normalised to the centre it falls
  to 0.79 by ρ 0.5, stays there to 0.8, then drops to 0.67 at 0.9 and 0.2-0.3 at the rim -- an edge inside
  the nominal aperture, which cos θ^γ (0.6 at the rim for γ 0.40) cannot make.  γ rises to imitate it.
- **With the aperture at an effective NA of ~1.32 the minimal pupil is 4 900 nats ahead of the old default**
  with five shared parameters fewer, within 840 nats of the symmetric Zernikes at the same aperture, and bends z
  as little (1.75 nm rms).  γ falls to 0.24.  The symmetric Zernikes gain from the lower aperture too
  (5 800 nats), their coefficients shrinking to a third: the nominal aperture was wrong for both models.
- **The effective NA is 0.97-0.98 of the nominal one**, as on one objective's beads (1.40 against 1.43,
  `docs/uipsf/architecture.md`, "The bead's physics with an effective NA").  A parabola through 1.31-1.33 puts
  it at 1.318; learnt (*learn NA*, `MinimalPupil.learn_na`, one for both objectives) it came out at 1.311.  Neighbouring fixed apertures differ by ~2 000 nats where a parabola would
  have them nearly equal, so the solver's stopping point is worth ~1-2 000 nats on these beads.
- **Phase z does not care**: 2.20-2.23 nm rms everywhere (3.3× its CRLB: the beads' own motion and the stage),
  the envelope's z 6.3-7.3 nm (2.4-2.8×).

**Decided (the user, 2026-11-02)**: the NA stays typed by default everywhere, 4Pi included, and the minimal
pupil stays 4Pi's default at the typed NA; learning it is an option, *learn NA* (`PSFModelSettings.learn_na`,
`pupil.na`, started at the typed NA, bounded to 0.8-1.05 of it), offered in every method. Plainly: **on the
Zenodo 4Pi beads the typed 1.35 fits 9 700 nats worse than a learnt 1.31.** The remedy is to learn the NA once
with *learn NA* and then type the learnt value under *microscope*. It is not a universal ratio: on M5's
one-objective beads the learnt NA ran to the grid's edge, 1.51 behind a typed 1.50 (ries-lab/SMAPpy#135). The
symmetric Zernikes remain a choice of *pupil magnitude*. Why the aperture is smaller is open; the buffer's index
does not explain it (ries-lab/SMAPpy#135, learnt beside the NA it moved to 1.397 for 106 nats). The sample's
index n_med stays typed by default too.

## 24. The buffer's index learnt: it does not explain the effective NA (2026-11-02)

§23 left the minimal pupil's effective NA, 1.31-1.32 behind objectives of 1.35, unexplained, and named the
buffer's index as the suspect: the 4Pi model took the sample as index-matched, the buffer is measured only when it
is matched, and a buffer of lower index cuts the light at its own index.  So the sample's index is now a parameter
of every uipsf model (`docs/uipsf/architecture.md`, "The sample's index learnt"), and the 4Pi form reads it.

**The form.**  The microscope's *sample index* is *auto* by default -- water's 1.33 for one objective and, for 4Pi,
the immersion's (index-matched), each mode's assumption as before -- or typed, which 4Pi now takes for the buffer
between the coverslips (`fourpi_beads.learn(n_med=)`, `OpticsSettings.sample_index`).  *learn sample index*
(`PSFModelSettings.learn_n_med`, off) learns it from there, one index for both objectives (the same buffer).  A
typed index below the NA brings the minimal pupil's ring and its h (one for both objectives under *same
magnitude*), and the form shows *SAF damping* then (`fourpi_shown(ring=)`); the immersion's index stays typed.
4Pi's first stage holds the index at the typed value (`workflows.fourpi.HELD_FIRST`): free from the start, it
stopped 17 500 nats behind the typed index's optimum on these beads (943 370, Δz −50 nm against +157).

**What the index does to z and the fringe.**  A bead scan moves the stage: coverslips, buffer and beads together,
so each bead keeps its distance to both coverslips, and only the two immersion layers change, one thicker, one
thinner.  The phase per nm of stage is k_z,imm in each arm whatever the buffer -- `VectorPSF._waves` is right
to use it -- and the buffer enters only through fixed terms: the far arm's (k_z,med − k_z,imm)·D through the
sample's thickness D, a defocus (part of the misalignment Δz) and spherical aberration (the lower pupil's
Zernikes), the Fresnel factors, and the supercritical light.  So the scan's fringe hardly moves with the buffer:
with a scalar model at NA 1.35, n_imm 1.406, 600 nm, the period at the image's centre is 289 nm matched, 292 nm at
n_med 1.37 and 298 nm at 1.33 -- longer, not shorter, and §23's 265 nm model period against the beads' 251 nm
(`beads_period_nm`, a different weighting of the pupil, §11) is not the index's.  A molecule moving *in* the
buffer, the stage still, changes its phase by k_z,med instead: the same model gives 309 nm at 1.37 and 341 nm
at 1.33, so a height from a bead model (in nm of stage) is the molecule's true height × 1.057 at 1.37 and × 1.145
at 1.33 (paraxially n_imm/n_med would say 1.026 and 1.057: the high angles weigh more).  That is *4Pi z*'s *z
scale*; it is 1 for a matched buffer, and should be set from the buffer's index where it is not.

**Real beads** (`benchmarks/uipsf/real/zenodo_4pi_minimal.py`, `results/fourpi_nmed.jsonl`; uiPSF v1's 18 beads as in
§23, the minimal pupil, one γ, at the nominal NA 1.35, on the CPU -- the typed matched row is §23's to the nat).
"+warm" starts from the typed row's converged model of the same beads.

| model | n_med | nats | γ | h (nm) | σ (nm) | Δ (nm) | period (nm) | envelope bias rms (max) (nm) |
|---|---|---|---|---|---|---|---|---|
| typed, matched (1.406) | 1.406 | 925 889 | 0.403 | | 41.1 | (−19, −15, 157) | 264.8 | 3.43 (8.9) |
| learnt from 1.406 | 1.4266 ± 0.0007 | 925 265 | 0.369 | | 41.6 | (−20, −16, 157) | 265.1 | 3.38 (8.7) |
| learnt from 1.406, +warm | 1.4266 ± 0.0006 | 925 265 | 0.369 | | 41.6 | (−20, −16, 157) | 265.1 | 3.37 (8.7) |
| typed, water (1.33), ring damped | 1.33 | 948 358 | 0.573 | 400 (bound) | 38.6 | (−1, −3, −41) | 264.4 | 6.90 (13.0) |
| typed, water, a point's ring | 1.33 | 958 475 | 0.627 | 0 | 37.8 | (0, −3, −42) | 263.7 | 4.82 (9.2) |
| learnt from 1.33 | 1.3319 ± 0.0004 | 948 095 | 0.569 | 400 (bound) | 38.7 | (0, −3, −40) | 264.4 | 6.90 (13.0) |
| learnt from 1.33, +warm | 1.4142 ± 0.0007 | 943 370 | 0.429 | 357 | 41.0 | (2, −2, −50) | 265.4 | 4.46 (8.6) |
| *learn NA* (§23, NA 1.311) | 1.406 | 912 642 | 0.230 | | 42.7 | (−25, −14, 107) | 265 | 1.77 (3.9) |
| *learn NA* and *learn sample index* (NA 1.310) | 1.3968 ± 0.0006 | 912 536 | 0.239 | | 42.6 | (−24, −14, 107) | 264.9 | 1.73 (3.9) |

- **The beads want no buffer of lower index.**  Learnt from the matched start, n_med rises to 1.427 -- above the
  immersion's -- and gains 624 nats; learning the aperture gained 13 200.  Typed as water the model is 22 000
  nats worse, and the beads refuse the ring the index brings: h runs to its 400 nm bound, damping the light
  beyond 1.33 away, and a point's undamped ring costs another 10 000.  The envelope bends twice as much.
- **From water the index climbs back**: warm, it rises to 1.414 (h 357 nm), 5 000 nats better than typed water
  but still 18 000 behind the matched optimum, in the other basin of Δz (−50 nm against +157); cold it stays
  at 1.332, held by the ring at its bound.  Both basins say the same: the index wants to be at least the
  immersion's.
- **With the aperture free the index stays put**: learnt together, the NA comes out at 1.310 as alone, and n_med at
  1.397, 0.009 below the immersion's, for 106 nats.  The aperture explains the beads; the index does not.
- **The fringe period does not move** (263.7-265.4 nm), as the physics above says for a stage scan.
- **An index above the immersion's** does not say the buffer is denser than the oil.  Above the NA, n_med acts
  only through the Fresnel factors, the apodisation and the dipoles' field directions, so a free index is one
  more knob on the aperture's radial weight, and it moves in the direction of the effective NA's correction,
  far less effectively.  Not checked further: its CRLB (± 0.0007) is that of a parameter taking up a model error,
  not of a measured buffer.

So the buffer is not the cause of the effective NA (§23's decision: learn the NA once, then type it); the cause stays open (the objective's high-angle
transmission, its coatings, or a field stop).  The default remains: index-matched on *auto*, the NA typed, *learn
NA* and *learn sample index* offered as diagnostics.

**One objective** (`benchmarks/uipsf/real/m5_nmed.py`, `results/m5_nmed.jsonl`): M5's 100 nm beads in PBS at 600
nm (`20261029_600_100nm_mismatch_50ms_2D_1`, 29 central beads), the shipped bead defaults at order 8, NA 1.50
typed, n_med from PBS's 1.334:

| model | nats | n_med | NA | γ | h (nm) | σ (nm) |
|---|---|---|---|---|---|---|
| typed | 573 504 | 1.334 | 1.50 | 0.363 | 62 | 14.5 |
| *learn sample index* | 572 905 | 1.3447 ± 0.0003 | 1.50 | 0.318 | 85 | 16.4 |
| *learn NA* | 572 530 | 1.334 | 1.510 (bound) | 0.364 | 65 | 14.7 |

Here the index moves by 0.011 for 600 nats and the NA wants to rise to the grid's edge for 975: on M5's objective
at its own NA there is no effective-NA deficit to explain, and PBS's index holds within 0.01.  (The 676 nm stack
was not on this machine; M2's 1.40 behind 1.43 is another objective's, not checked here.)

## 25. What is still missing, and the 2017-11-28 tubulin data (2026-11-24)

### The folder `gabin/4Pi_frames/2017-11-28`

On the VBC share (`/Volumes/VBCShare/RIES/Users/gabin/4Pi_frames/2017-11-28`), about 245 GB.  A
Bewersdorf-design astigmatic 4Pi-SMS, as §10's Nup96: an sCMOS writing DCIMG strips of 2048 × 168 px with
the four phase images side by side (the authors' channel centres x = 235, 474, 1557, 1797), plus **a second
camera, an EMCCD (128 × 128 px)** -- the salvaged-fluorescence path for colour (Zhang et al. 2020).  The
analysis files are those of the authors' MATLAB pipeline (`4Pi-SMS/fit_ast.m`), redone in 2020-09/10.
`note.txt`: tubulin, AF647 + CF660C, 0.05 % (two dyes); Cell01, Cell02 4Pi at 100 fps, 642 nm at 600 mW,
gain 200.  Read with `io/dcimg.py` as it is (a series opens as one source).

| what | frames | what it is |
| --- | --- | --- |
| `Cell01/` (231 GB) | 120 DCIMG × 3000 = 360 000, 10 ms | **the tubulin movie, two colours**; `Cell01_642_000_000.tif` is a 3000-frame 168 × 168 crop of one channel |
| `EMCCD/Cell01/` (11 GB) | 120 TIFF × 3001, 128 × 128 | the same movie's salvaged-fluorescence camera; 3001 pages per file against 3000 frames, so the first or last page has to be matched |
| `Cell02` | -- | in `note.txt`, not in the folder |
| `Cell03/`, `Cell04/`, `Cell05/` (41 MB each) | 61 one-frame DCIMG, 20 nm steps (1.2 µm), 0.3 / 0.5 / 0.5 s | **bead z scans, one bead each**, at the centre of each channel; no phase steps.  Cell04 is what the authors' phase calibration (`bead_642_dphi_cali_20200911.mat`) and astigmatism fit (`bead_642_Astfit_20200911.mat`) were made from.  A bright vertical structure sits at the channels' edges, away from the bead |
| `Cell06/` -- `Cell12/` (7 MB each) | 10 frames, 100 ms; `EMCCD/CellNN/` 11 frames | **many beads in focus on both cameras**: channel alignment (`align_642_FMTtransform_20200914.mat`, from Cell12) and the sCMOS -> EMCCD registration (`two_camera_reg_20200911.mat`, a second-order polynomial per axis) |
| `Cell01_642_tmpresult_20200912.mat` | 11.8 M rows | the authors' raw fits: x, y, t, astigmatic z, phase z, LL, CRLB, I, bg, σx, σy |
| `Cell01_ratio.mat` | 11.8 M rows | per-localization photons on the two cameras (`RT`, `PT`) -- the colour ratio |
| `Cell01_642v20_60.mat`, `…_two_color.mat` | 5.1 M rows | the filtered, unwrapped, drift-corrected result in Vutara's columns, and split into the two colours: **a reference to compare against** |
| `Cell01_DCresult.mat`, `Cell01_1_drift.mat` | 119 cycles | the authors' drift correction (x, y, z per cycle) |
| `Cell01_ll/`, `Cell01_ll_/` | -- | Vutara SRX `particles.csv` exports of the result |
| `old/` | -- | the 2017 analysis, superseded by the 2020 one |

### What it adds to the data we have

| set | where | sample | colours | beads |
| --- | --- | --- | --- | --- |
| Ries-lab 4Pi, zenodo 10823308 | `~/.cache/uipsf/zenodo/4pi_40nm_bead`, `4pi_smlm_nup96` | Nup96 | 1 | 26, with 3 mirror phase steps |
| 4Pi-SMS Example Data II, zenodo 4022827 | `…/4pi_sms_nup96` (one of four movie files) | Nup96 | 1 | 5 in one z stack |
| zenodo 14287640 | `…/4pi_dcimg_astig` (Cell09, 2024) | -- | -- | one z stack |
| **this folder** | VBC share | **tubulin** | **2, second camera** | 3 one-bead stacks; 7 two-camera bead fields |

It is the only one with **two colours** and **two cameras**, the only **continuous structure** (filaments,
where Nup96 is rings), and the only **whole long acquisition** (an hour, 360 000 frames) with the authors' own
analysis of every step to compare with.  It is weaker in beads: one per stack, three stacks.

**Test data, copied** to `~/.cache/uipsf/4pi_tubulin_2017-11-28/` (5.7 GB, 2026-11-24): the three bead stacks,
Cell06 -- 12 on both cameras, the first two movie files (`Cell01_642_000_000` and `_001_000.dcimg`, 6000 frames)
with their EMCCD files, `note.txt`, every analysis `.mat` of the top folder (the raw fits, the ratio, the
result, the drift, the calibrations) and `Cell01_dmap.tif`.  The whole movie stays on the share, which reads
at 100 MB/s (≈ 145 frames/s; the local copy reads at 700 frames/s), for the one run that needs it.  Nup96
(4022827, 10823308) stays the test of z (the rings); this set is single colour on the sCMOS for now, the
EMCCD and colour later.

**The Ries-lab Nup96 in full.**  zenodo 10823308 (uiPSF, Fig. 2) has `Fig2_4pi_dataset.zip`, 16.35 GB: the 26
phase-stepped bead stacks and all the movie files of `cell3_nup96_mMaple_045`.  The cache and the share
(`Users/shah/example data for uiPSF`, `Projects/DECODE_data`) held only 2 of them; all 70 movie files and
`parameters.txt` are in `~/.cache/uipsf/zenodo/4pi_smlm_nup96` since 2026-11-24 (15 GB).  4022827's zip (3.25 GB)
is the whole 4Pi-SMS example, of which the cache holds the beads and the first movie file.

### What is missing

Ordered by what a user of a 4Pi-SMS would hit first.

1. **Two colours by salvaged fluorescence (two cameras)** -- later, by decision (2026-11-24); the analysis is
   single colour on the sCMOS for now.  M7's "multicolour" and "two cameras", not started.
   What this data needs: the EMCCD's frames read beside the strip and matched frame by frame (the extra
   page); a registration between the cameras from beads seen on both (Cell06 -- 12; the authors use a
   second-order polynomial, `calibrate.dual` has affine and higher maps); per localization the photons on
   the EMCCD at the registered position, position held (a fixed-position fit or an aperture sum); the ratio
   against the 4Pi photons, and colours assigned from it as *Assign colours* does with MINFLUX's `dcr`.
   The authors' `Cell01_ratio.mat` and `…_two_color.mat` are the check.  Colour from the s/p ratio inside
   the 4Pi cavity (the other route M7 names) needs no second camera, but this data cannot test it.
2. **DECODE for 4Pi** (`docs/decode.md` §10 step 6): the torch IAB renderer and the phase convention are done
   (P5, §30); the network is designed there, not started.  A 4Pi network
   outputs the phase as well (one mixture with a von Mises factor, not Hao's two), so it is not netfit's C = 2
   plus two channels.  Dense tubulin, where the fitter's ROIs overlap, is where it should gain most.
3. **Speed.**  The fitter is C++ on the CPU, 4× a 2C fit per spot (measured below); the whole movie is an
   hour or two, reading included.  A GPU fitter or DECODE (2) would help; reading 231 GB over the share is
   its own limit.
4. **Ghosts** -- later, by decision (2026-11-24).  §10 expects 11 -- 14 % wrong fringes; *4Pi z* gives the probability but nothing removes or
   re-fits them: the locked refit (D4) is not started.  Filaments are continuous in z, so a structure-aware
   unwrapping (a localization's fringe judged against its neighbours') is an idea this data can test and the
   rings could not.
5. **Drift over an hour in z.**  *4Pi z* follows the cavity's phase; the sample's 3D drift needs RCC / COMET on
   the phase z, with ghosts in it.  Never run on a 4Pi movie this long; `Cell01_DCresult.mat` is the reference.
6. **In situ (M6)**, and from dye stacks -- neither exists (below), after the in-situ learning: with one bead per stack here, a model learned from the
   movie's own 11.8 M localizations matters more than for the Nup96 sets.  D9's sample geometry comes with it.
7. **A model from three one-bead stacks.**  `iab_from_beads` and the pupil learning both took stacks of
   several beads; here the three stacks have to be learned together, with a phase per stack (the cavity moves
   between them).  The first thing to try on this data, before anything above.

### Asked, and answered from the code (2026-11-24)

- **4Pi from dye stacks: no.**  The PSF calibration window's 4Pi mode has two methods, the bead average and the
  pupil model, both on bead scans (`psf_methods.FOURPI`, `fourpi_beads.scan_beads`); the molecule-stack methods
  have no 4Pi variant, so `for_mode` offers none.
- **4Pi in situ: no.**  Nothing in `calibrate/insitu*.py` or uipsf's in-situ workflows knows 4Pi; the 4Pi
  learning is `uipsf/workflows/fourpi.py`'s `learn_fourpi`, from bead stacks.  M6 and D9 are plans.
- **Channel drift: a prefit, yes.**  With *channel drift* on (off by default), Spline 4Pi first fits
  *drift blocks* (10) blocks of *frames per block* (500) spread over the movie, with x and y free per channel
  (`Global4PiPSF(shared=(False, False, True, False, True, True))`); `multichannel.measure_channel_drift`
  keeps the better half by `x_err_nm` and fits each channel's offset against channel 0 as a robust,
  error-weighted spline over time (*over time*), or a constant -- the channel move's machinery
  (`channel_move.over_time`).  It is measured again on every run, not cached, and refused for a live fit.
- **The fitter's speed.**  Yes, it is the IAB model: per pixel and channel three splines (I, A, B) where
  Spline 3D 2C evaluates one, in four channels instead of two -- 6× the arithmetic per iteration, partly
  repaid by fewer iterations (4.1 against 7.0).  Measured (`fit_cspline4pi_global` against
  `fit_cspline_global` on the I splines of two channels, 2000 simulated spots, v1's bead model, all cores of
  the Mac):

  | ROI | 4Pi, 1 start | 4Pi, 2 z starts (default) | 4Pi, 2 × 2 phase starts | 2C |
  | --- | --- | --- | --- | --- |
  | 13 px | 5 840 fits/s | 5 400 | 1 850 | 24 100 |
  | 17 px | 3 700 | 3 600 | 1 150 | 14 900 |

  So a 4Pi spot costs 4× a 2C one, 13× with two phase starts.  The whole plugin on v1's Nup96 (1000 frames,
  7 localizations a frame): 251 frames/s, 1 840 localizations/s, 60 % of it in the C++ fit.  §10's 13 -- 27
  frames/s on 4Pi-SMS went mostly into reading, the drift pass and writing (§12, where the engine alone did
  72 frames/s).  For Cell01: 11.8 M candidates at ~5 000 fits/s is about 40 min of fitting, and reading
  231 GB from the share about as long -- an hour or two for the whole movie.

### The plan: work packages (agreed 2026-11-24)

Single colour first; two colours and ghosts later.  Each package is a session of its own, in its own
worktree; P1, P2 and P4 started and landed on 2026-11-24 (§26, §27, §28).

- **P1. Gabin's tubulin, single colour, end to end.**  A model from the three one-bead stacks (Cell03 -- 05)
  learned together with a phase per stack (item 7); the layout and the channel registration from the
  many-bead fields (Cell06 -- 12); Spline 4Pi and *4Pi z* on the first two DCIMG files of Cell01 (6000
  frames); compared with the authors' `Cell01_tmpresult.mat` / `v20_60` on the same frames.  Data in
  `~/.cache/uipsf/4pi_tubulin_2017-11-28`.  Where: `fourpi.py`, `calibrate/fourpi_beads.py`, uipsf's
  `workflows/fourpi.py`, `benchmarks/fourpi/`.
- **P2. 4Pi in situ (M6).**  A 4Pi model learned from a movie's own localizations, started from a bead
  model; first on the full Ries-lab Nup96 (`~/.cache/uipsf/zenodo/4pi_smlm_nup96`, 70 files) against v1's
  in-situ models, then as a method of the PSF window's 4Pi mode.  Where: uipsf's `workflows/fourpi.py` and
  in-situ workflows, `calibrate/insitu_run.py`, `psf_methods.FOURPI`.
- **P3. 4Pi from dye stacks** -- not now: no such data (2026-11-24).
- **P4. Spline 4Pi throughput, on the CPU.**  Profile the whole plugin on a 4Pi-SMS DCIMG; the channel drift
  cached beside the movie as the channel move is; the spline coefficients read contiguously per voxel in
  `CSpline4Pi`; reading overlapped with fitting.  A GPU fitter is decided later, with the other fitters and
  after benchmarking.  Where: `csrc/`, `plugins/fit_4pi.py`, `multichannel.py`, `pipeline.py`.
- **P5. DECODE for 4Pi** -- open: DECODE-Plex and netfit's C = 2 are on main (0.7.6).  The torch IAB
  renderer equal to `CSpline4Pi` with the phase convention settled, then a C = 4 network with the phase
  as one more factor of each molecule's mixture, a von Mises distribution (the Gaussian of an angle).
- **P6. 3D drift over a long 4Pi movie** -- done (§31): RCC / COMET on the phase z, sure fringes, z on knots of its
  own, checked against `Cell01_DCresult.mat` on the authors' fits of the whole movie and on our fit of 45 000
  frames.

P1 and P2 both touch uipsf's `workflows/fourpi.py`; P1 adds a per-stack phase to the bead learning, P2 a
learning from localizations, so whichever lands second rebases.

Checked for this note: the folder's files, the frame sizes and counts (read with `io/dcimg.py`), the
authors' calibration files' contents, and the result files' columns.  Not checked: whether Spline 4Pi fits
these frames with the default layout.

- Zhang Y, Schroeder LK, Lessard MD, … Ries J, … Bewersdorf J (2020).  Nanoscale subcellular architecture
  revealed by multicolor three-dimensional salvaged fluorescence imaging.  *Nat Methods* 17:225 (checked on
  PubMed, 31907447).
  [doi:10.1038/s41592-019-0676-4](https://doi.org/10.1038/s41592-019-0676-4).  The second camera and the
  colour ratio.

## 26. The 2017-11-28 tubulin data, single colour, end to end (P1, 2026-11-24)

Package P1 of the plan in §25 (on `claude/4pi-gaps` while this is written): Gabin's Bewersdorf-design
4Pi-SMS data in `~/.cache/uipsf/4pi_tubulin_2017-11-28`, the sCMOS strip only.  The scripts are
`benchmarks/fourpi/tubulin_*.py`, their outputs in `smappy_p1/` beside the data, the numbers in
`benchmarks/fourpi/tubulin_results.md`.

**The chip.**  The four images are the 4Pi-SMS design's rectangles (`channels.YALE_4PI`, x 151, 389, 1474, 1713,
168 px, the last two mirrored): their centres are the authors' (235, 474, 1557, 1797).  The files say nothing
of the camera.  The strip outside the channels gives an offset of 102 ADU and a read variance of 9.9 ADU²; the
photon transfer of the movie's background, per pixel against its own median, gives 0.40 e⁻/ADU at the median
and 0.50 at the 20th percentile -- blinking only raises a pixel's variance, so the conversion is nearer the upper
-- so the 4Pi-SMS example camera's 0.5 e⁻/ADU and offset 100, with the authors' 128 nm pixels.

**The registration**, from the many-bead fields Cell06 -- 12 (`register_channels`, 86 pairs per channel, 0.06 --
0.08 px rms), is the authors' (their FMT alignment, from Cell12) to 0.0005 in scale and 1.1 mrad in rotation:

| channel | ours: scale x, y | rotation (mrad) | authors: zoom x, y | angle (mrad) |
|---|---|---|---|---|
| 2 | 1.0098, 1.0099 | 19.9 | 1.0102, 1.0098 | 19.3 |
| 3 | 1.0101, 1.0110 | 13.6 | 1.0102, 1.0106 | 12.5 |
| 4 | 0.9999, 1.0013 | −7.1 | 0.9999, 1.0013 | −7.0 |

The bead scans cannot give it: Cell03, 04 and 05 hold one bead each, the same place in all three (within 0.8 px;
most likely one bead scanned three times, at 0.3, 0.5, 0.5 s), and `register_channels` made an exact affine map of
the three nearly equal points -- any turn at all a few pixels away.  It now gives a shift where the beads reach
less than 3 px rms in some direction, and the PSF window's 4Pi beads have a *registration* part: from the scans
(as before), from a field of beads (Spline 4Pi's `registration_from_beads`, kept beside them), or from a file
(`fourpi_beads.ScanRegistrationSettings`, `given_maps`).  The pupil model then holds the maps as given
(`fourpi_beads.learn(maps=)`, `FourPiModel(fit_transform=False)`): one place fixes no turn or scale.

**A phase per scan, and a rate per scan.**  Each bead of `iab_from_beads` and of uiPSF's learning already had its
own phase (`ScanPhases.bead_phases`, `emitter.phase`), so three scans learnt together need nothing new for the
cavity's offset between them: their phases are 0.95, −1.49 and 2.73 rad, well spread, and the per-voxel solve is
determined per channel (condition 3.4).  What is new is the cavity's drift *through* a scan: `scan_phases`'s own
rate per bead was 0.431, 0.425 and 0.453 rad per 20 nm slice, each to about 0.002, against the median the bead
average took for all.  So `iab_from_beads(scan_of=)` gives each scan its beads' median rate, and holds each bead's
phase where its fit fixes it, at the middle of the fit's window (`ScanPhases.centre`): a phase at slice 0 taken at
another rate was off by the difference times 30 slices, 0.6 rad here.  `fourpi_beads.average_model` passes each
bead's scan.  On simulated scans drifting by 0, +0.03 and −0.03 rad per slice, one rate averaged the interference
away towards the scan's ends (modulation 0.66 against the truth's 0.80 at ±500 nm), a rate per scan kept it
(0.80) (`test_scans_whose_cavity_drifted_each_its_own_way_keep_their_interference_at_a_rate_per_scan`).  On the
real three, which the model fits back to the scale of noise either way, leaving one scan out did not decide
between them (the third fitted with a model of two scattered 5 -- 33 nm in z either way): with three beads the
average's ends are poorly fixed whichever rate.  The pupil model, which has the physics to fill them, is the one
to use.  uiPSF's learning keeps one phase per bead and the optics' rate; a drift rate per emitter was not needed
(below), so `uipsf` is unchanged.

**The pupil model** (`tubulin_pupil.py`; NA 1.35, silicone 1.406, index-matched, the PSF window's minimal
vectorial pupil, maps held, 45 -- 95 s on the Mac's GPU).  The emission had to be found: at 620 nm the model's
fringe period was 265.8 nm against the beads' 291.5 nm (`period_mismatch` warned) and the coherence fell to
0.52 -- 0.69 to make up for it; at **680 nm** (AF647's emission) it is 294.0 nm and the likelihood 1.6 M nats
better.  The beads' size then matters: taken as 40 nm, the coherence is 0.74 -- 0.89; as **100 nm**, 0.82 --
0.99 and 16 000 nats better (851 590 against 867 614) -- coherences near their bound, so the beads are about
100 nm, and a smaller size hands their own blur of the fringe to the coherence.  Fitted back to the scans, the
envelope's z follows the stage at a slope of 0.98 -- 0.99 with 2 -- 3 nm scatter, and the phase at 0.95, 0.95
and 1.02 of the model's rate: the third scan's drift, now in the data and not in the model.  The channels'
phase offsets, 0, 1.28, 3.05, −1.72 rad, are the bead average's (0, 1.23, 3.04, −1.73) to 0.05 rad.

**The authors' calibration** is of another kind: an astigmatic Gaussian's widths against z
(`bead_642_Astfit_20200911.mat`) and a linear phase against z with the s and p offsets
(`bead_642_dphi_cali_20200911.mat`: 0.0208 rad/nm, a period of 302 nm, from Cell04 alone).  Their period is
3.5 % longer than the beads' rate here (291.5 nm over all three; Cell04's own 0.425 rad per slice, 296 nm), which
is the 0.965 between their z and ours below.

**The movie** (`tubulin_fit.py`, `tubulin_compare.py`): Spline 4Pi with the defaults, the registration file,
the model, *4Pi z* after it, on Cell01's first 6000 frames; the authors' raw fits of the same frames
(`Cell01_642_tmpresult_20200912.mat`, 320 285) and their final table (`Cell01_642v20_60.mat`, filtered,
unwrapped and drift-corrected, 90 850, tied to the raw fits by `vutarapr`); pairs within a frame closer than
0.5 px.

| model | one fit per molecule | ours | expected wrong fringes | envelope scatter / error | theirs paired | their final paired | z vs their final: spread | on another fringe | x, y vs theirs (nm) |
|---|---|---|---|---|---|---|---|---|---|
| bead average, rate per scan | no | 404 154 | 11.0 % | 1.93 | 89 % | 99 % | 4.6 nm | 0.2 % | −0.6, −6.0 |
| pupil, 680 nm, 40 nm bead | no | 404 233 | 8.4 % | 2.14 | 89 % | 99 % | 4.6 nm | 0.1 % | −1.8, +0.5 |
| **pupil, 680 nm, 100 nm bead** | **yes** | **369 582** | **7.2 %** | 2.03 | 89 % | 99 % | **4.4 nm** | **0.1 %** | −1.9, +0.5 |

- **z agrees with theirs where they kept it.**  On the 90 178 localizations of their final table, our z less
  theirs scatters 4.4 nm about one offset, and 0.1 % are on another fringe; none of those we call 90 % sure (95 %
  of them).  The fringe is the same molecule's in both.  Two conventions differ: **their z runs the other way**
  (their final and astigmatic z fall where ours rise, correlation −0.96; the phases agree), and their 302 nm
  period makes their z 3.5 -- 7 % longer (slope 0.93 -- 0.97 of ours against minus theirs).  Their z is drift-
  corrected and ours not, which the 4.4 nm includes.
- **Their final table is a quarter of their fits** (90 850 of 320 285).  Over all of ours *4Pi z* expects 7.2 %
  wrong fringes (§10's Nup96: 11 -- 14 %), 16 % below 90 % sure; ghosts are P1's limit, as §25 said.
- **Lateral**: within 2 nm of theirs, 7 -- 8 nm robust spread between the two fits of a molecule; our x error
  5.7 nm median against their CRLB of 7.1 nm (a Gaussian on a 4Pi spot).  The bead average sat 6 nm off in y:
  its boxes are moved onto each bead by an interpolation of a three-bead average, the pupil model's are a
  physical PSF's.
- **Photons**: ours 1.60 × theirs per molecule.  Their background is the four channels' sum per pixel, 119.7
  against our 4 × 34.3 = 137, so the camera's scale is nearly the same; the photon difference is the models' --
  an astigmatic Gaussian on the summed image takes the spot's core, the 4Pi model all of its light.
- **One fit per molecule.**  With the bead average and the 40 nm model 9 % of our fits (36 715) were a second
  fit within a pixel of another in the same frame, 30 353 of them ones the authors have once: two candidates on
  one spot in the channels' sum, both fits ending on it.  The single-channel engine has dropped such a second fit
  since the DECODE benchmark (`pipeline.one_per_molecule`); the multi-channel engine did so only for the
  two-colour fitters' paired detection.  It now does for the sum too (Spline 4Pi version 5).  What is left
  unpaired (85 780, 23 % of ours) is brighter than the paired (5 650 against 4 180 photons), half within 5 px of
  another of ours with a worse likelihood (`logl_rel` −1.9 against −0.8): overlapping molecules on the dense
  filaments, which their pipeline fits once or not, and the last 8 rows of the chip (9 % of them), where theirs
  stop.  Theirs unpaired, 36 483 (11 %), are a median 2.8 px from one of ours.
- **Speed**: 10 -- 17 frames/s on the Mac (the three runs shared it with other work, not comparable); the whole
  360 000-frame movie is 6 -- 10 hours at that rate -- P4's.

**What a user of this data hits in the GUI, and what changed.**
- *Spline 4Pi* on *auto* stopped: no microscope entry gives this chip's rectangles.  *auto* now takes the model's
  (`fit_4pi.model_geometry`): a model made in the PSF window keeps its scans' layout, the movie's chip, and is
  used where it fits the frames, unmoved by a camera ROI.  With the camera typed and the model chosen, Cell01 fits
  with nothing else set.  The camera still has to be typed (the file names none; the notice opens the editor).
- *4Pi beads* on a scan folder stopped on `Cell03_642_000_000.tif`, a 168 × 168 crop saved beside the DCIMG
  series: a file the channels do not fit is now left out, and the progress says which.
- *4Pi beads* registered the channels from the one-bead scans: now a shift, and the *registration* part takes a
  field of beads or a file (above).
- The emission has to be right for a pupil model: `period_mismatch` said so at 620 nm, as it should.

**Not done, for later.**  The authors' drift (`Cell01_DCresult.mat`) against ours (P6); ghosts (out of scope);
the EMCCD and the second colour (out of scope); a drift rate per emitter in uiPSF's learning (not needed: the
fitted-back phase rates 0.95 -- 1.02 are each scan's drift, held by the beads' own phases); the bead average's
ends from three beads.

## 27. In situ: a 4Pi model learned from the movie's own molecules (P2, M6; design 2026-11-24)

§25's P2.  A bead model describes the instrument on the day of the beads, through a bead sample; the movie
is another day, another sample and another alignment.  In situ learns the model from the movie's own
molecules, starting from the bead model, as uiPSF v1 did for this Nup96 movie
(`results/4pi_smlm_nup96/psfmodel_iter1_insitu_zernike_4pi.h5`, 2538 molecules) and as
`calibrate/insitu_run.py` does for one channel.  Single colour; the Nup96 sample is index-matched
(silicone, 1.406 both), so D9's layers vanish and today's `FourPiModel` is the model: D9's gap, cover
and focus come with the first mismatched sample, not here.

**The unit is one molecule in one frame**: four boxes, one per channel, one observation each (K = 1, no
mirror steps).  `FourPiModel` takes it as it is: ``n_phases = 1``, ``Batch.z_nm = 0``, every molecule its
own x, y, z, cavity phase φ, photon count (`Bleaching(order=0)`) and background per channel; no bead
kernel.  Linking a blink's frames into one unit (as single-channel in situ does) is left for later:
mMaple's blinks are short, and the cavity phase moves between frames.

**The cavity phase needs no model while learning**: each molecule has its own φ, so the drift over the
movie is absorbed molecule by molecule, and so is the fringe: the interference's z dependence is in the
PSF, φ is what is left.  That φ should then follow one smooth curve in time -- a check of the learnt
model, and later a prior that would decide wrong fringes (not in P2).

**Held and free** (D9's table, its in-situ column, for a matched index):

| parameter | in situ | why |
|---|---|---|
| both pupils' phase Zernikes | free | the aberrations are the sample's and the day's: depth, DM, realignment |
| pupils' magnitude (γ, apodisation) | held at the beads' | the objectives' own; molecules too dim to tell |
| coherence α_c, balance r_c, shares f_c | free | what changes most with the sample (D9) |
| channel phase offsets δ_c (δ_0 = 0) | free | the wedges and the cavity, retuned per sample |
| misalignment Δ | Δz free, Δx, Δy held | D9: Δz is what realigning moves; the lateral direction was not stable even from beads |
| channel maps ``ch<c>.transform`` | free | the movie's registration is up to 1 px off the beads' (§8); the molecules span the field |
| blur σ | free | takes up what moves during an exposure |
| n_med, NA, wavelength | held | typed, as for beads |

**The molecules** (`smappy.calibrate.fourpi_insitu`, no torch, no Qt):
1. Spline 4Pi on the movie with the bead model: a table of x, y (channel 0's chip), z (the envelope's),
   phase, photons, backgrounds, CRLBs and log-likelihood per molecule, and the layout it used.
2. Chosen: converged fits, bright (above a photon floor), inside the model's z range less a margin,
   alone (no other localization within a box in that frame), not an outlier in log-likelihood; then
   spread over z by quotas in bins, up to a number (a few thousand), and spread over the movie.
3. Cut: each molecule's position mapped into each channel by the layout (`ChannelLayout.to_channel`),
   a box centred on the nearest pixel in every channel's image, in photons; each box's centre in its
   channel is `Batch.roi_xy`, so the model's maps start from the layout's.

**The learning** (`uipsf.workflows.fourpi_insitu`):
1. The molecule model: the bead model's pupils, optics and per-channel numbers, ``n_phases = 1``.
2. Each molecule localised on the start, the shared parameters held: x, y, z from the fitter; φ the best
   of twelve round the circle (`best_phases`); z also from a fringe period above and below, the best kept
   -- the same molecule has optima a period apart, which only the envelope tells apart.
3. Outliers out by their deviance per pixel (a neighbour in the box, a bad cut), robust z-score.
4. Every free parameter together, molecules and shared, with the beads' solver.
5. Localised again from the three z starts on the new model, outliers out, learnt again (two rounds).
6. Exported as the beads' are (`workflows.fourpi.calibration`): the same convention, so no sign to undo
   (v1's in-situ export had the opposite phase sign to its bead export, §8).

**The test of the method** is a simulation with a known truth: molecules drawn with a model whose
pupils, coherence and maps differ from the start's by what a day can move, the cavity phase drifting;
learnt from the start, the truth must come back closer than the start was, and the refit's z bias must
fall.  **The test on Nup96** (the whole movie, 70 files, 140 000 frames): learnt on molecules from the
odd files, judged on the even ones, against the v2 bead model and v1's bead and in-situ models -- the
pupils, the fits' log-likelihood on the same molecules, the CRLB at matched photons, z from the phase
against the envelope's (slope and scatter), the share of uncertain fringes, and the rings' separation.

### What it found (2026-11-24)

**Simulated** (`tests/uipsf_tests/test_fourpi_insitu.py`, and 200 molecules in a script first): molecules drawn with
a sample's pupils, coherence, channel offsets, Δz, maps and blur, learnt from a bead model a day off in each, a
tenth of them started a fringe off.  Everything comes back: the pupils within 0.01 rad (from 0.15 off), the
coherence within 0.02 (from 0.1), the channel offsets within 0.04 rad, Δz 63 nm for 60 (from 20), the maps within
0.02 px (from 0.5), σ 50 for 50 (from 40); z within 5.6 nm median and every molecule on its fringe.

**Nup96** (`benchmarks/fourpi/insitu/`; the whole movie, 140 000 frames).  The bead model is v2's, learnt by the
window's defaults from all 102 beads of `4pi_40nm_bead` (9 min on a Mac's GPU; fringe period 264.9 nm).  The movie
fitted with it (Spline 4Pi, *4Pi z*; 17 min, 1.15 M localizations); 3000 molecules chosen from the odd files
(photons above the median, 841 median of those chosen), learnt in two rounds (21 min on a Mac's GPU; 16 left out
as poor fits).  Every model then judged on the **even files** only:

| model | held-out molecules: objective (nats) | deviance / px | logl_rel | CRLB x / y (nm) | CRLB z, envelope / phase (nm) | uncertain fringes | p < 0.9 | z_phase vs z_env: slope, scatter (× CRLB) | ring separation (nm, median, IQR) |
|---|---|---|---|---|---|---|---|---|---|
| v2 beads | 781 626 | 1.156 | −0.561 | 9.26 / 8.73 | 26.7 / 5.44 | 10.1 % | 29.8 % | 1.031, 1.37 | 61.9 (51.8 -- 72.7) |
| **v2 in situ** | **773 213** | **1.144** | **−0.557** | 8.41 / 8.06 | 24.4 / 4.55 | 10.4 % | 30.7 % | 1.046, 1.46 | 62.1 (52.4 -- 72.1) |
| v1 in situ | -- | -- | −0.559 | 9.54 / 8.78 | 19.7 / 5.17 | 14.7 % | 39.9 % | 1.076, 2.03 | 66.5 (54.2 -- 79.3) |
| v1 beads | -- | -- | −0.561 | 9.65 / 8.91 | 26.2 / 5.63 | 10.7 % | 31.0 % | 1.015, 1.42 | 96.6 (74.4 -- 115.6) |

The CRLBs are at 700 -- 1400 photons; the held-out objective is 2000 molecules of the even files, each localised
on every model by uipsf from three z starts; the rings are each pore's z fitted by two Gaussians of one width,
after one 3D drift (RCC on the beads' fit, sure fringes) taken off every table alike, over 4100 pores found by
*Segment regions*' ring template.  The uncertain fringes are *4Pi z*'s own estimate, from each model's CRLBs.

**What the in-situ model changed**: the coherence rose from 0.79 -- 0.84 to 0.93 -- 0.98 in every channel (v1 saw the
same direction: modulation 0.76 from beads, 0.82 in situ), Δz from −155 to −93 nm, Δx, Δy by 6 -- 11 nm against
a prior of 1 nm (the data pull hard; a run whose prior followed each round's result landed within 3 nm of
this one), σ 40 → 36 nm, the period 264.9 → 267.4 nm.  The pupils moved by 0.19 (upper) and 0.15 rad rms (lower):
the upper astigmatism 0.48 → 0.41 rad, its coma Z7 0.02 → −0.09, and the spherical Z11 −0.07 → −0.13 (upper)
and −0.07 → −0.18 (lower) -- what a sample a few µm deep adds.

**So, plainly: the in-situ model describes the molecules better, but measures the Nup96 no better.**
- Better: 8 400 nats on 2000 held-out molecules (4 per molecule, the deviance per pixel 1.156 → 1.144), and the
  spline fit's log-likelihood on every held-out localization.
- Not better: the rings' separation is the same (62.1 against 61.9 nm, the same spread); the share of uncertain
  fringes is not lower (10.4 against 10.1 %); the envelope's and the phase's z agree no better in nm (1.46 ×
  24.4 = 35.6 nm against 1.37 × 26.7 = 36.6 nm).  The smaller CRLBs come from the higher coherence, and the
  measurements do not show them: the scatter in units of the CRLB rose by as much as the CRLB fell.
- Both v2 models beat v1's in-situ model clearly (14.7 % uncertain fringes, z scatter 2.0 × its CRLB).

The Nup96 sample is index-matched and imaged near the coverslip, with the beads from the same instrument a short
time before: the bead model was already right, and what in situ adds is within what these measures can see.
In situ should matter where the bead model is wrong -- a mismatched buffer (D9), a deep sample, or beads only
from another day or as few as Gabin's three one-bead stacks (§25, P1; §26).

**In the window** (decided 2026-11-24, the user: in situ an option beside the beads): the in-situ sample has a
4Pi mode, *pupil model (in situ), 4Pi* (`psf_methods.FourPiInSitu`, `calibrate/fourpi_insitu.run_insitu`), with
a section of its own, *4Pi in situ*: the bead model file to start from (a pupil model this window saved, which
now keeps its recipe and learnt values, `fourpi_beads.model_from_calibration`), the movie's Spline 4Pi fit with
it or the frames to fit here, and the molecules.  A fifth of the molecules is held out and localised on both
models: the summary says by how many nats per molecule the in-situ model explains them better, and a warning
says to keep the bead model where it does not -- the evidence a user needs, since the CRLBs a higher coherence
brings are not it.  The beads stay the default.  What is next, in order: the tubulin movie with P1's three-stack bead model; ring separations measured per time window (the
62 nm against Nup96's ~50 nm is mostly the measurement's: drift within a window, ghosts, the two-Gaussian fit);
the cavity phase as a prior over time (above), which would decide fringes rather than only describe them.

## 28. Spline 4Pi's speed on the CPU (P4, 2026-11-24)

Work package P4 of the plan (§25, on branch `claude/4pi-gaps` when this was written; P1's results are §26, P2's in situ §27): the whole plugin profiled
on two real data sets, and what dominated made faster.  The profile is `benchmarks/fourpi/speed.py`; it also
compares a run's localizations with a saved run's, which is how every change below was checked.

**The data.**  *tubulin*: Gabin's 4Pi-SMS DCIMG (`Cell01_642_000_000.dcimg`, strips of 2048 × 168, the 4Pi-SMS
corners), 73 candidates a frame.  There is no model of that microscope yet (P1 makes one), so the Ries-lab bead
model stands in, without a registration: the work per molecule is a real fit's (a 60-plane grid of 20 × 20,
13-px ROIs), but the fits are poor and take 24 iterations where Nup96's take 10 -- the fit is slower than it will
be with the right model.  *nup96*: the Ries-lab Nup96-mMaple `.mat` series with v1's in-situ model and maps,
7.7 candidates a frame.  1500 frames each, the Mac's ten cores, niced.

**Where the time went (before).**

| | tubulin | Nup96 |
| --- | --- | --- |
| reading alone (the channel layer) | 700 frames/s | 780 frames/s |
| detection (one thread) | 0.6 ms/frame | 1.2 ms/frame |
| the C++ fit | 1 690 fits/s, 23 frames/s | 2 430 fits/s, 315 frames/s |
| *channel drift* prefit (10 × 500 frames) | 280 s | 31 s |
| writing, *4Pi z* | < 0.3 s | < 0.1 s |

Reading runs in its own thread (`pipeline.prefetch`) and was never the limit; writing and *4Pi z* (46 ms for
100 000 localizations) are nothing.  What was: the C++ fit, then detection, which ran in turn with it on one
thread while nine cores waited, and the drift prefit, which is a second fit of 5000 frames on every run.

**1. The splines read voxel by voxel, summed four at a time** (`csrc/models.hpp`, `CSpline4Pi`).  Each pixel and
channel sums 64 products for each of 12 quantities -- the value and three derivatives of I, A and B.  The
coefficients were read in the calibration's layout, (3, 64, nz, ny, nx), each a stride of nz ny nx floats from the
last: 192 cache lines per pixel and channel.  They are now a copy kept on the calibration,
`FourPiCalibration.fit_coeff`, (C, nz, ny, nx, 3, 64): one run of 768 bytes.  That alone gained 15-25 % per core;
the bigger cost was the arithmetic, every sum one dependent chain of 64 multiply-adds, scalar, waiting on the
latency of the one before.  The four sums of a spline are now one 4-wide vector (GCC and Clang's vector type;
the same sums in plain loops for MSVC), each split into two halves: six independent chains.  `value` alone went
from 450 to 50-100 ns.  The halves change the order of the sums, so the numbers move in the last bits of a float:
against the old fit on the same ROIs, the 99th percentile of the differences is 0.1 nm in x and y and under 1 nm
in z; 0.07 % of fits (tubulin's poor ones) end in another optimum, as often better as worse.  On Nup96 none do.
Per core, measured in CPU time with the old and new build alternating: **2.4× (tubulin), 2.8× (Nup96)**.  On all
ten cores the old layout lost more -- ten cores' working sets of 8 MB each do not share a cache -- 2.6× and
up to 5×.  `CSpline` (Spline 3D, 2C) has the same scalar chain and would gain from the same vectors; not done
here, since it moves every 3D fit's last bits -- done since, 2.0x per core (`docs/performance.md`, "The
spline's sums").

**2. Each block fitted while the next is found** (`MultiChannelEngine.overlap`).  The fit of a block goes to a
worker thread, and `flush` hands back the block before; `pipeline.drive` takes the last from `end`.  The same
table in the same order (`test_fitting_while_the_next_block_is_found_changes_nothing`).  On for a whole file and
the drift prefit; off for a live fit, which shows each block when it is fitted, and the preview.

**3. The channel drift kept beside the movie** (`multichannel.cached_drift`, `<movie>.channel_drift.json`), keyed by
the movie's path, size and time, the model's bytes, the layout with its maps, and every setting the prefit reads
(camera, detection, fit region, fit, the drift's own) -- as the two-colour fitters' channel move is.  The second
run of a movie fits nothing for it.  The curve is kept, not its fits: a run that reads it says so and draws the
curve alone.  Measured again, the curve moved by at most 0.0016 px.

**Before and after** (the best of the runs: other sessions' work loaded the Mac to 40-160 at times, so these
vary by 20-40 % between runs; the per-core ratios above are the robust ones):

| | tubulin before | tubulin after | Nup96 before | Nup96 after |
| --- | --- | --- | --- | --- |
| C++ fit, ten cores | 1 690 fits/s | 4 310 fits/s | 2 430 fits/s | 11 960 fits/s |
| engine (detection + fit) | 23 frames/s | 58 frames/s | 230 frames/s | 720 frames/s |
| whole plugin, no drift | 19 frames/s | 51 frames/s | 190 frames/s | 565 frames/s |
| whole plugin, drift measured | 4 frames/s | 11 frames/s | 38 frames/s | 112 frames/s |
| whole plugin, drift kept | (measured again) | 51 frames/s | (measured again) | 565 frames/s |

("no drift" before is the fit loop of the run with the drift, its prefit taken out; after, a run reading the
kept drift.)  The drift prefit's share is large here because 1500 frames are a short movie: on the whole Cell01
(360 000 frames) its 5000 frames are 1.4 %.  At 51 frames/s Cell01 is two hours, with the stand-in model's 24
iterations; with its own model (Nup96's 10) about one.  Reading from the share (145 frames/s) is no longer slower
than fitting a dense movie, and on a sparse one it would be: copy the movie locally first.

**What a GPU fitter would gain**, for the later decision.  A fit is about 50 evaluations of 4 × 169 pixels of
800 multiply-adds -- 30 MFLOP, the same for every molecule, no branches: what a GPU does well, 20-100× the ten
cores.  But after these changes the Nup96 movie is limited by reading (900 frames/s) and detection (one thread,
1.2 ms a frame), not the fit: a GPU would gain less than 2× there.  Dense data is where it would pay: tubulin's 73
molecules a frame keep the CPU at 51-58 frames/s, and a GPU fit would take it to reading's 700-800, 10-15×.
So the GPU question is one of density: a DECODE-like network for dense 4Pi data (P5) answers the same need.

**Measured again on a quiet Mac (2026-11-25)**, as the reference for the next speed work and the GPU decision.
M1 Max (8 performance and 2 efficiency cores, 32 GB), main at `b605a3a9`, `benchmarks/fourpi/speed.py --frames
2000 --kept --repeat 3`, niced, nothing else of ours running: the 1-min load was below 3 before each step (2.5-2.9);
Word and macOS's `ANECompilerService` (one core) ran throughout, Zoom only before these steps.  Tubulin is now
fitted with P1's model (§26: `smappy_p1/pupil_680_100_held_4pi.h5`, the channels registered from the bead fields,
0.5 e-/ADU, 128 nm) instead of the stand-in -- 77 fits a frame after P1's duplicate check; Nup96 as above, 7.8.
Each number the median of three runs (range):

| | tubulin (P1's model) | Nup96 |
| --- | --- | --- |
| reading alone (the channel layer) | 4 735 (903 cold - 4 774) frames/s | 997 (964-1 000) frames/s |
| detection, in the engine | 1.23 (1.21-1.28) ms/frame | 0.96 (0.96-0.97) ms/frame |
| C++ fit, one thread | 1 102 (1 099-1 113) fits/s | 1 707 (1 702-1 708) fits/s |
| C++ fit, ten cores | 6 874 (6 847-6 875) fits/s, 6.2× | 12 396 (12 333-12 533) fits/s, 7.3× |
| engine (detection + fit, overlapped) | 86 (85-88) frames/s | 815 (808-823) frames/s |
| whole plugin, drift kept | **84 (83-85) frames/s** | **602 (602-604) frames/s** |
| whole plugin, drift measured (one run) | 21 frames/s (prefit 71 s) | 142 frames/s (prefit 11 s) |

Tubulin's first read is from the disk (900 frames/s), the next from the page cache; Nup96's `.mat` series decodes
at 1 000 frames/s however often it is read.  P1's model fits tubulin 1.6× faster on all cores than the stand-in
did (6 900 against 4 310 fits/s), as §28 expected: its fits converge.  **Where the time goes now**: tubulin is the
fit's -- 77 fits at 6 874 a second are 11.2 ms a frame against 1.2 ms of detection, so the engine runs at the
fit's rate (89 frames/s) and the whole plugin at 84; Cell01's 360 000 frames are 72 minutes.  Nup96 is detection's
and reading's: 0.63 ms of fit a frame beside 0.96 ms of detection (overlapped, 1.23 ms a frame in all), and the
whole plugin at 602 is bounded by the `.mat` reader's 1 000.  Detection on the tubulin's channel sum is 0.69 ms a
frame on one thread and 0.47 on all, of which 0.32 ms is turning the four channels into photons and summing them,
on one thread (`docs/performance.md`, "The fitters' speed on a quiet machine").  So a GPU fit would take tubulin
from 84 towards detection's ~800 frames/s (about 10×) and gain Nup96 nothing measurable.

**After the photons and the channel sum were threaded (2026-11-25, `d60301c8`)**: the four channels' conversion to
photons and their sum onto channel 0 are C++ on every core (`csrc/photons.hpp`, `multichannel.detection_image`),
the same image to the last bit; every other fitter now overlaps its fit with detection as this engine already did
(`docs/performance.md`, "After").  The channel sum alone (`benchmarks/fitters/speed.py detect`, quiet): 0.46 →
**0.17 ms a frame** on all cores, 0.70 → 0.48 on one; its photons-and-sum part 0.33 → 0.03 ms.  In the engine
(`benchmarks/fourpi/speed.py --kept --repeat 3`, before at `3363cece` and after one right after the other): both
data sets' localizations are **identical** to before (all 129 440 and 13 575, every column, max |d| 0); tubulin's
whole plugin is 84 (84-84) frames/s before and after, as it must be -- the fit's 11 ms a frame was never waiting
on detection; Nup96's detection in the engine is 0.96 → **0.41 ms a frame** and the engine alone 813 → **1 118
frames/s**, past the `.mat` reader's 1 000, which is now all that bounds its whole plugin.  These two Nup96
numbers were measured with another session's work on the machine (1-min load 15-20, alike before and after), so
their ratio holds better than their values; the tubulin engine-alone timing after could not be taken quietly
(the load rose to 150) and is left out.

## 29. In situ on the 2017-11-28 tubulin: no better than the beads, and why (2026-11-25)

§27's in situ run on Gabin's tubulin (Cell01's first 6000 frames), started from P1's bead model (§26: the pupil
model of the three one-bead scans, 680 nm, 100 nm beads, maps held), and judged as P2 judged it on the Nup96,
with the authors' fits beside.  The scripts are `benchmarks/fourpi/insitu/tubulin_*.py` (their order in that
folder's README), their outputs in `smappy_insitu/` beside the data.

**What it tripped on first.**  P1's model was saved the day before pupil models kept their recipe with them, so
in situ could not start from it (`model_from_calibration` is None) and said only that a bead average will not
do.  It now says to learn the model again from the same beads (`bead_model_of`); `tubulin_pupil.py 680 100 held`
gives the same model (objective 851 590, as P1's), recipe and all, in a minute.

**The learning.**  2853 molecules from the odd blocks of 500 frames (photons above the median: 8300 median; z
−420 to +450 nm), two rounds, 22 min on the Mac's GPU; 1900 from the even blocks held out.  The movie then fitted
with each model (Spline 4Pi, *4Pi z*, the registration file) and judged on the even blocks; the authors' raw
fits and final table paired as in §26 (`tubulin_compare.py`).

| model | held out: objective (nats) | deviance / px | uncertain fringes | p < 0.9 | z_phase vs z_env: scatter (× CRLB, nm) | z vs their final: spread, other fringe | envelope z vs their final | x, y vs theirs (nm) | photons ours / theirs | period (nm) |
|---|---|---|---|---|---|---|---|---|---|---|
| beads (P1's) | 1 819 070 | 2.83 | 7.2 % | 16.2 % | 2.10, 44.5 | 4.4 nm, 0.1 % | 26.9 nm | −1.9, +0.5 | 1.60 | 294.2 |
| in situ | **1 634 684** | **2.55** | 13.6 % | 30.0 % | 2.19, 58.2 | 7.7 nm, 1.7 % | 58.8 nm | +14.8, +12.5 | 1.86 | 282.5 |
| in situ, cavity tied | 1 642 350 | 2.56 | 7.8 % | 16.3 % | 1.71, 43.7 | 6.3 nm, 0.1 % | 30.3 nm | +13.7, +15.2 | 1.89 | 282.8 |

The held-out objective is the 1900 even-block molecules, each localised on each model by uipsf from three z
starts; the deviance per pixel its mean.  On them the two in-situ models put 15 % (untied) and 5.6 % (tied) on
another fringe than the beads do, their z otherwise within 89 and 39 nm (robust sd) of the beads'.  The rest are each model's own fit of the movie (the uncertain fringes
over all of it, the scatter on the even blocks' sure ones), against the authors' as in §26.

**So, plainly: in situ does not help here, and it does not explain the 1.6× photons.**
- **In situ explains the molecules much better** -- 97 nats per held-out molecule (Nup96: 4) -- **and measures
  z worse.**  It doubled the uncertain fringes, put 1.7 % of the authors' final localizations on another fringe
  (beads: 0.1 %), and its envelope z against the authors' phase-precise z spread twice as wide (59 against 27 nm),
  with a slope of 1.48 where the beads' was 0.86 and a plateau in the middle: the envelope's z axis warped.  The
  window's check, the held-out likelihood alone, would have recommended it.
- **Why the z axis can warp.**  Every molecule has its own cavity phase φ, and for the interference a molecule's
  phase and its z are one number, φ + kz: nothing ties the model's phase against z, or the envelope's scale, to
  anything.  On simulated molecules (§27's test) the PSF's shape fixed both; on these it did not.  The cavity's
  phase is the instrument's and common to the molecules of a frame -- here it drifted 0.25 rad over the 60 s and
  tilts by about 0.05 rad/µm in y (*4Pi z*'s cavity, and the residual of its phases across the field) -- so in situ can tie each molecule's phase to it: a prior
  of σ 0.2 rad about the cavity's phase, found from the molecules in windows of 1000 frames with a plane over the
  field (`InSituSettings.cavity_sigma_rad`, `cavity_phases`).  Tied, the envelope stays put (30 nm against the
  authors', slope 1.10) and the uncertain fringes are the beads' (7.8 against 7.2 %).  The period still moved
  (294.2 → 282.8 nm, as untied): the molecules' phases scatter 1.5 rad about the cavity's even after learning
  (the bead model's envelope z error, 2× its CRLB, times k), so the tie holds the envelope, not the period, which
  the bead scans measure (291.5 nm by the stage) and these molecules cannot.
- **Tied, in situ is the beads' equal, not better**, though it still explains the held-out molecules by 93 nats each
  better.  The same fringes and envelope scatter in nm; z against the
  authors' 6.3 nm against 4.4 (the phase's z less linear in theirs: 5.7 against 3.8 nm about a line); x and y
  moved 14 nm from theirs (the misalignment Δx, Δy, held by a 1 nm prior, moved 18 -- 24 nm: the data pulled).
- **What the molecules show that no model fits.**  Their boxes are wider than the bead model's spots (rms width
  2.3 -- 2.5 against 1.7 -- 1.9 px, at every z), and the deviance is 2 per pixel even at a box's edge, where only
  background is modelled, rising with the neighbours a molecule has in its box (1.55, 2.23 and 4.17 for 1 -- 2,
  3 -- 5 and 6 or more localizations within a box in the frame and the two beside it, the
  molecule's own there counted too): these are dense filaments, and the boxes hold
  their neighbours' light and their out-of-focus light.  In situ takes it up as PSF: the blur 84 → 106 nm, the
  coherence 0.82 -- 0.99 → 0.60 -- 0.95 (light that does not interfere with the molecule's), Δ moved.  That
  describes the boxes better and the molecules worse.  The Nup96, sparse and near the coverslip, had none of it
  (deviance 1.15).
- **The photons.**  A Gaussian with a flat background fitted to the model's own spots summed over the channels --
  what the authors fit, astigmatic, to the sum (`tubulin_gauss.py`, noise-free, at the movie's median photons and
  background) -- takes 0.76 of the bead model's photons (0.62 at −400 nm, 0.83 at +400 nm; 16-px box) and 0.67 of
  the in-situ model's, whose spot is wider.  1.60 / 1.31 = 1.22 and 1.86 / 1.50 = 1.24: the rest is the same for
  both models, so the camera's, and the backgrounds say the same -- ours over theirs 137 / 119.7 = 1.14, 1.18 with
  the Gaussian's own background bias (+3.5 -- 5 photons a pixel here) taken off theirs.  **So the 1.6× is ~1.3
  from the Gaussian taking only the spot's core and ~1.2 from the two pipelines' photon scales** (their camera
  conversion is not in the files; P1 measured 0.40 -- 0.50 e⁻/ADU and took 0.5).  In situ does not explain it: it
  raises our photons further, to 1.86 -- 1.89.

**What changed.**
- `fourpi_insitu.bead_model_of` says what to do with a pupil model saved without its recipe.
- In situ ties the molecules to the cavity's phase, in the window by default (*cavity tie*, 0.2 rad, an advanced
  setting of the *4Pi in situ* section; 0 learns as §27 did): `uipsf.workflows.fourpi_insitu.cavity_phases`, and
  `Params.subset` now carries a prior's mean per emitter into a solver's chunk.  §27's Nup96 numbers are untied;
  that movie was not learnt again tied.
- The window warns when the in-situ model puts more than 5 % of the held-out molecules on another fringe than the
  bead model, or moves the period by more than 2 % (`fourpi_insitu.warnings_of`, `FourPiInSituRun.other_fringe`):
  here 15 % untied and 5.6 % tied, and 4 % for both.  The summary says the first.

**Not done.**  A background that is not flat in a molecule's box, or the neighbours fitted with it, which is what
this sample needs before in situ can learn from it; molecules chosen sparser (alone over several boxes and
frames); the period held at the beads' while learning; the tied model on the Nup96.

## 30. DECODE for 4Pi: the design, and the renderer (P5, 2026-11-25)

Work package P5 (§25), step 6 of `docs/decode.md` §10: netfit at C = 4, with the interference phase as one more
quantity of each molecule.  Three stages, each landed on its own: (1) a torch IAB renderer equal to `CSpline4Pi`,
with the phase convention settled; (2) the network; (3) judged on simulations against the CRLB and Spline 4Pi, then
on Nup96 and Gabin's tubulin.  Stage 1 is done (below); Jonas agreed the design on 2026-11-25.

### Stage 1: the renderer (done)

**`netfit/render_4pi.py`, `IABRenderer`.**  Draws ``N (I_c + A_c cos phi + B_c sin phi)`` in every channel in torch,
on any device, from a `FourPiCalibration` as the fitter reads it (`fit_coeff`, (C, nz, ny, nx, 3, 64)): the
geometry of `render.SplineRenderer` and of `CSpline4Pi::prepare` -- a square of ``min(nx, ny) - 3`` pixels around the
nearest pixel, the voxel clamped to the grid and the fraction not, z by `z_nm_to_index`, negative values set to 0
after I, A and B are summed (as `fourpi.add_frames`).  Each spot is drawn in each channel at that channel's own
position, ``x``, ``y`` of shape (m, C), so the registration is the caller's.  Checked (`tests/test_netfit_render_4pi.py`):

| check | result |
| --- | --- |
| against `fourpi.spots` (numpy), z beyond the grid at both ends, phases beyond pi | < 10^-6 of the peak |
| noise-free torch spots of 10^5 photons, channels shifted by up to 0.4 px, fitted by Spline 4Pi's C++ | x, y < 0.002 px, z < 0.1 nm, phase < 0.001 rad, N < 0.1 % |
| frames on the channel images against `fourpi.add_frames` cut by a layout (mirrored and stretched channels) | < 10^-4 of the peak |
| float32 on MPS against the CPU | < 10^-5 of the peak |

(The C++ fit's own float32 precision is the limit of the second row: the models are one.)  Speed, a 64 × 64
four-channel frame with 40 spots, 23-px squares, on the Mac while other sessions loaded it: 7 500 spots/s on the CPU,
9 500 on MPS, against 34 000 and 40 -- 60 000 for one channel's spline (`render.SplineRenderer`).  A 4Pi spot is
twelve times a one-channel spot's data per pixel (four channels, three splines): the gather of 192 coefficients per
pixel is the cost, the arithmetic one 64-term product per spot (a spot's fractions are the same for all its
pixels).  On the CPU 256 spots per gather stay in the caches and run three times as fast as 1024; a GPU takes 1024.
A training batch of 32 three-frame samples with ~50 molecules each is ~5 000 spots: half a second on the Mac,
milliseconds on the 4090 or the H100, where the trainings run.

**The phase convention, settled on beads** (`benchmarks/fourpi/phase_convention.py`).  SMAPpy keeps each channel's
phase offset inside its own A_c and B_c, and the channels share one phase with nothing added (`smappy.fourpi`).
DECODE_Internal's branch `4pi-simulation` (Hao Sha) reads uiPSF v1's files the same way (A = 2 Re, B = -2 Im) and
then adds o_c = 0, pi/2, pi, 3 pi/2 per channel, cos(phi + o_c) and sin(phi + o_c) (`psf_kernel.py:1812`, `:2183`).
Both fitted by the C++ fitter to the 1134 bead ROIs uiPSF v1 learned its model from (18 beads × 3 mirror phases ×
21 planes), the branch's convention as A and B turned by o_c:

| convention | Σ log L | median log L / ROI | period from the phase | the mirrors' steps (set +2, -2 rad) |
| --- | --- | --- | --- | --- |
| SMAPpy's | -2 132 646 | -1 580 | 258.5 nm | +2.06, -2.12 |
| the branch's | -4 637 977 | -3 713 | 594.8 nm | -0.02, +0.05 |

SMAPpy's fits every one of the ROIs better, and only it gives back the steps and the period; the branch counted the
channels' offsets twice (`docs/decode.md` §2, "4Pi", item 2).  The renderer, the fitter and the training share
SMAPpy's.

### Stage 2: the network (built 2026-11-25)

**How Hao's network outputs the phase, and why ours differs.**  The branch's `loss.py` (`_vmmm_core_loss`) has a
phase head with ``tanh``, read as ``phase = (pi t + pi) mod 2 pi``, and a sigma head read as the concentration
``kappa = 1 / (sigma + 10^-4)``; the loss is ``3 (GMM(N, x, y, z), bg, 10 VMM(phi))``, the von Mises mixture over the
pixels with the detection map's weights, separate from the position's mixture.  Three things differ here, each for
a reason:

1. **One mixture.**  Each component is Normal(N, x, y, z) × von Mises(phi) -- one molecule, one pixel, its phase.
   In two mixtures a true molecule's phase may be explained by another pixel than its position, and the two
   likelihoods need weights set by hand (the 10 and the 3).  The joint likelihood needs none: the phase is one more
   factor of the molecule's density, as z is.
2. **The mean as a point on the circle.**  The head gives two numbers, (c, s), and the mean is ``atan2(s, c)``
   (normalised, so only the direction is learnt).  A scalar through ``tanh`` has an edge: phases just below +pi and
   just above -pi are neighbours on the circle and opposite ends of the output, where ``tanh`` is flattest -- the
   network has to jump across its whole range between two nearly equal spots, and is least able to there.  (c, s)
   is continuous all round.
3. **The concentration as a log.**  ``kappa = exp(k)`` (or softplus), with the von Mises' normaliser
   ``log I_0(kappa)`` computed stably as ``log i0e(kappa) + kappa`` (torch's ``i0e``).  The reported
   ``phase_err_rad`` is ``1 / sqrt(kappa)``, the Gaussian width the fitter's CRLB is, to which it tends at the
   concentrations a molecule has (kappa of a few hundred to thousands).

**What the network outputs is the fitter's table.**  The phase as the fitter reports it -- ``phase_per_nm z +
phase0``, wrapped -- and z the envelope's (`z_env_nm`), not z from the phase: that needs the cavity's phase0, which
drifts and is found from many molecules over time (*4Pi z*, `fourpi.cavity_phase`), which no single window can
know.  So netfit's table has `z_nm` = `z_env_nm`, `phase_rad`, `phase_err_rad`, and *4Pi z* runs on it unchanged.
One photon count, the total over the channels, as the fitter's shared N (the channels' shares are in their
splines); a background per channel (Hao's branch had one for all four, though the channels' backgrounds differ).

**Inputs.**  Netfit's two-channel grid (`netfit.channels.TwoChannels`, plex-plan §16) generalised to C channels: the
network's grid is channel 0's pixels, every other channel moved onto it by a whole-pixel shift, the registration's
remainder ``d_c(p)`` as two offset maps per channel (six for C = 4), read only where all four channels hold the
camera's pixels.  The layout and the maps from the 4Pi model or the microscope (`fit_4pi.layout_of`), as Spline 4Pi
takes them.  Early fusion, which won C = 2's architecture study (plex-plan §16): the four channels and their offset
maps stacked as one input of the per-frame U-Net; netfit's other parts as they are (ConvGRU, 3 frames; the mixture
with the reported sigma; `mixture_peak`).

**Training sets** drawn with `IABRenderer`: molecules uniform in the crop at z uniform over the model's range, each
channel's spot where the registration puts it; the cavity's phase0 per sample uniform on the circle, drifting within
the window as `fourpi.cavity_drift` draws it (a rate and a random walk per frame, from the prefit); a background
per channel and frame; the camera on each channel.  On-time 1 -- 3 frames (Jonas's rule).  The models: v2's bead
model of the Ries-lab Nup96 (`~/.cache/uipsf/p2/bead_v2_4pi.h5`, with its maps) and P1's pupil model of the tubulin
microscope (`smappy_p1/pupil_680_100_held_4pi.h5`).  The prefit -- backgrounds, photons, the phase's drift rate --
by Spline 4Pi on the movie, as the two-channel job is prefitted by Spline 3D 2C (`netfit/prefit_two.py`).

**Where, as built.**  `netfit/fourpi.py`: `FourChannels` (the C-channel grid, `TwoChannels`' generalisation, its
offsets read by `channels.Grid` unchanged), `draw_batch` (frames (b, C, window, s, s), the targets' wrapped phase in
`Batch.phase`, each target's remainders in `Batch.shift`), `cavity`, `crlb` (the IAB bound, x, y, N, z and the
phase shared and a background per channel free, by central differences in float64 -- equal to the C++ fitter's
CRLB to 0.15 %), `unwrap_known` and `phase_scores`.  `settings.ChannelSettings` kind "4pi" (``count``, the cavity's
``cavity_rate_rad`` and ``cavity_walk_rad``); `network.TwoStageUNet`'s ``phase`` and `Layout.phase` (three
channels at a slot's end; early fusion now resamples every channel beyond the first by its own offset maps);
`loss.terms`' von Mises factor; `extract`'s ``phase_rad`` and ``phase_err_rad``; `train.Drawer` and the test
set's bound and scores.  `tests/test_netfit_fourpi.py`: the grid puts a spot at p + d_c; a batch's targets are what
Spline 4Pi fits from its frames (isolated molecules: mean error 0.07 of the CRLB, median |error| 0.68, a
Gaussian's); the bound is the C++ fitter's; the mixture prefers the right phase and, at it, a sharper one; a
training runs and scores the phase.  Not yet: the prefit, the predictor on a real movie and `fit_decode.py`
reading a 4Pi model -- stage 3.

### Stage 2: the first trainings (2026-11-25)

**Two networks**, one per microscope, each trained on its own 4Pi model with netfit's default parts and early fusion
(2.25 M parameters, 3-frame window, on-time 1 -- 3 frames log-uniform, 25 molecules per 40 × 40 px, 6 M samples, bf16;
settings in `benchmarks/decode/netfit_4pi/`):

| | model | photons (all channels) | background / px / channel | camera | trained on |
| --- | --- | --- | --- | --- | --- |
| A, Nup96 | v2 bead model (§27) | 1000 ± 800, from 100 | 1.5 -- 6 | sCMOS, 129 nm | gpuvm, 1.2 h |
| B, tubulin | P1's pupil model (§26) | 6000 ± 5000, from 300 | 20 -- 60 | sCMOS 0.5 e/ADU, 128 nm | decodebox, 1.0 h |

The ranges are those of the data's Spline 4Pi fits: Nup96's photons 150 -- 2200 (5th -- 99th percentile, median 480),
backgrounds about 3; tubulin's 1100 -- 28 000 (median 4400), backgrounds about 35.  (The two times are from two machines
and are not a comparison.)

**Scored** by `benchmarks/decode/netfit_4pi/evaluate.py` on fixed sets drawn as the training draws, 1024 samples
each, at 5 and 25 molecules per 40 × 40 px and mean on-times of 1 and 3 frames (3-frame window); near focus,
|z| < 300 nm; the widths are robust widths of the error over the single-frame CRLB of the IAB model, which is Spline
4Pi's (`results/*_scores.json`):

| net | set | Jaccard | lat. RMSE / width | z RMSE / width | phase RMSE / width | claimed phase width | wrong fringe | z from phase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | d5, on 1 | 0.885 | 25.2 nm / 0.86 | 65.9 nm / 0.93 | 0.275 rad / 0.81 | 1.05 | 6.6 % | 9.2 nm |
| A | d25, on 1 | 0.852 | 28.2 nm / 0.97 | 75.2 nm / 1.08 | 0.305 rad / 0.90 | 1.07 | 7.8 % | 10.3 nm |
| A | d5, on 3 | 0.914 | 20.5 nm / 0.77 | 52.5 nm / 0.79 | 0.223 rad / 0.80 | 1.07 | 3.8 % | 7.9 nm |
| A | d25, on 3 | 0.882 | 23.8 nm / 0.90 | 63.6 nm / 0.98 | 0.255 rad / 0.84 | 1.06 | 5.5 % | 8.4 nm |
| B | d5, on 1 | 0.813 | 28.9 nm / 0.76 | 59.6 nm / 0.74 | 0.283 rad / 0.72 | 1.03 | 4.6 % | 11.4 nm |
| B | d25, on 1 | 0.775 | 30.1 nm / 0.92 | 66.7 nm / 0.97 | 0.306 rad / 0.87 | 1.06 | 5.1 % | 10.6 nm |
| B | d5, on 3 | 0.874 | 26.5 nm / 0.75 | 51.4 nm / 0.74 | 0.265 rad / 0.74 | 1.01 | 3.4 % | 9.5 nm |
| B | d25, on 3 | 0.826 | 25.4 nm / 0.92 | 51.4 nm / 0.96 | 0.234 rad / 0.85 | 1.04 | 2.7 % | 8.2 nm |

"Wrong fringe" and "z from phase" unwrap the network's phase at its envelope z with the cavity's phase known
(`fourpi.unwrap_known`): the share more than half a fringe from the truth, and the RMSE of the rest -- what *4Pi z*
can do at best with these localizations.  Over the whole z range the numbers are within a few hundredths of these
(the JSON files: widths up to 0.06 higher, wrong fringes up to 1 point lower).

**The phase is learnt as precisely as x, y and z.**  Its error is at its CRLB or below it, at both densities, and
the network's own sigma for it is right: the error over the claimed sigma has a width of 1.01 -- 1.07.  Widths
below 1 are not a breach of the bound: the bound is one frame's, and the network reads three (a molecule on in its
neighbours is pooled, which on-time 3 shows most), and it knows the ranges it was trained on.

**Against Spline 4Pi** (`benchmarks/decode/netfit_4pi/fitter_reference.py`, `results/fitter_reference.txt`): the
fitter on sets drawn the same way (on the CPU, so other samples), each molecule of the middle frame fitted *at its
true position* -- perfect detection, which favours it -- in 13-px ROIs in every channel, linked as the plugin links
them.  Near focus:

| model | set | failed fits | lat. width | z width | phase width | wrong fringe | alone: lat / z / phase width, wrong fringe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Nup96 | d5, on 1 | 28 % | 1.41 | 1.63 | 1.42 | 15.7 % | 1.04 / 1.36 / 1.00, 6.2 % |
| Nup96 | d25, on 1 | 34 % | 3.10 | 3.51 | 1.96 | 35.6 % | (1 % alone) |
| tubulin | d5, on 1 | 15 % | 1.65 | 1.46 | 1.41 | 22.8 % | 1.06 / 1.07 / 1.22, 10.4 % |
| tubulin | d25, on 1 | 27 % | 7.18 | 5.28 | 2.99 | 45.6 % | (1 % alone) |

("Alone": no other molecule on in the middle frame within 10 px.)  On molecules alone the fitter is at its CRLB, as
it must be; the rest is neighbours inside its ROIs, which at 5 per 40 × 40 px -- the tubulin movie's density --
already concern most molecules, and at 25 nearly all.  Its failed fits are Nup96's dimmest molecules (median 140
photons, which return NaN) and, for tubulin, neighbours.  The network, on the same kind of sets, is at 0.7 -- 1.1 of
the bound at both densities and puts 3 -- 8 % on a wrong fringe where the fitter puts 16 -- 46 %: what P5 set out to
show on simulations.  Not shown yet: the network on real movies (a predictor for the 4Pi grid, the prefit), and a
comparison on the *same* frames with the fitter's own detection -- stage 3.

### Stage 3: on movies, against Spline 4Pi with its own detection (2026-11-25)

**What was added.**  `predict.Predictor` reads a 4Pi model: C channels on its grid, their offset maps, the phase
out.  `fourpi.localize` runs it over a movie's channels in blocks, each with the window's context, and puts the
positions where channel 0 saw the molecule on the camera frame, as Spline 4Pi reports them.  `fourpi.prefit` and
`settings_from_prefit` take a training's photons (uniform over the fits' range, `decode_smlm.prefit.photon_range`),
each channel's background, z and the cavity's drift from a Spline 4Pi table after *4Pi z* -- the cavity drifts by
about 3e-5 rad a frame on both data sets, a hundredth of what the trainings drew, so the drift a window sees is
negligible.  The benchmarks in `benchmarks/decode/netfit_4pi/`: `movie_compare.py` (simulated movies, both
fitters each with its own detection on the same frames), `real_movie.py` (a model on the real movie, read through
Spline 4Pi's own source, then *4Pi z*).  SMAPpy's extensions were built on gpuvm (pybind11 in a folder of its
own), so that both fitters run there: the Mac was short of memory throughout.  Not yet: the DECODE fitter
(`fit_decode.py`) reading a 4Pi model in the GUI, and a 4Pi job in the *Train DECODE* window.

**The same simulated frames.**  `movie_compare.py`: ten movies of 100 frames each per row, drawn by
`smappy.fourpi.render_frames` in the model's own layout with its maps (the simulation Spline 4Pi's tests use, not
the training's), molecules blinking with a mean on-time of 1 or 3 frames, the cavity drifting; Spline 4Pi with the
plugin's default detection, fit and starts; netfit with the stage-2 networks.  Scored where both read (the grid's
box less 8 px), near focus: Jaccard / the robust width of the error over the CRLB in x, z and the phase / the
wrong-fringe share with the cavity known (`results/movie_*.json`):

| model | density / 40² px | on-time | netfit | Spline 4Pi |
| --- | --- | --- | --- | --- |
| Nup96 | 5 | 1 | 0.935 / 0.72 0.73 0.75 / 3.9 % | 0.712 / 1.14 1.23 1.08 / 7.0 % |
| Nup96 | 5 | 3 | 0.946 / 0.67 0.69 0.68 / 3.2 % | 0.712 / 1.19 1.35 1.08 / 6.4 % |
| Nup96 | 25 | 1 | 0.884 / 0.84 0.89 0.82 / 5.6 % | 0.230 / 1.99 2.81 1.27 / 13.7 % |
| Nup96 | 25 | 3 | 0.899 / 0.79 0.85 0.76 / 4.0 % | 0.123 / 2.74 4.15 1.43 / 17.2 % |
| tubulin | 5 | 1 | 0.826 / 0.69 0.69 0.69 / 3.7 % | 0.568 / 1.35 1.39 1.09 / 4.5 % |
| tubulin | 5 | 3 | 0.879 / 0.65 0.65 0.66 / 2.8 % | 0.624 / 1.48 1.52 1.11 / 4.5 % |
| tubulin | 25 | 1 | 0.785 / 0.86 0.88 0.81 / 4.3 % | 0.129 / 4.54 4.60 1.61 / 12.3 % |
| tubulin | 25 | 3 | 0.834 / 0.86 0.88 0.80 / 2.7 % | 0.063 / 7.81 7.46 2.11 / 16.4 % |

Both are precise (0.95 -- 0.99); the difference is recall and, with neighbours in its ROIs, Spline 4Pi's errors.
At 25 per 40 × 40 px its detection keeps a sixth of the molecules: far beyond the density it is used at (the
Nup96 movie has about 0.2 per 40 × 40 px, the tubulin movie about 4), so the rows at 5 are the ones that bear on
the data.  There netfit finds 30 -- 40 % more molecules and puts half as many on a wrong fringe.  Each fitter took
3 -- 8 s for a row's 1000 frames on gpuvm.

**Real data: the coordinates agree.**  On the first 200 frames, the molecules both find: x, y within 0.1 -- 2.3 nm of
each other at the median (robust sd 9 -- 14 nm), the envelope's z within 3 nm, the phase within 0.01 rad, the
photons within 3 %.  netfit finds 85 -- 95 % of Spline 4Pi's molecules and many more of its own: on Nup96 most of
them dim (median 87 photons, below the 100 the training started at), on tubulin bright ones too.

**Nup96, the whole movie** (138 687 frames, 300 frames/s on the H100), judged by P2's `insitu/evaluate.py` (the even
files, P2's drift, sure fringes for the rings), netfit's localizations from 150 photons -- Spline 4Pi's 5th
percentile -- beside Spline 4Pi with the same v2 bead model (`results/nup96_eval.txt`):

| | rows | pores | ring separation (IQR) | uncertain fringes | z_env error claimed (700 -- 1400 ph) | (z - z_env) / z_env_err |
| --- | --- | --- | --- | --- | --- | --- |
| Spline 4Pi | 599 449 | 4 182 | 61.9 nm (51.8 -- 72.7) | 10.1 % | 26.7 nm | 1.18 |
| netfit | 875 741 | 5 636 | 62.5 nm (50.9 -- 76.4) | 9.9 % | 20.5 nm | 1.06 |

A third more pores, the two rings as far apart, as few uncertain fringes; netfit's envelope z is claimed more
precise and its claim is nearer the truth (the robust width of z_env's error against the phase's z, 1.06 against
1.18).  Below 150 photons netfit reports 4.1 M more, which a user filters as Spline 4Pi's own detection does.

**Tubulin, the first 6000 frames**, judged by P1's `tubulin_compare.py` against the authors (Zhang et al. 2020):

| | localizations (a frame) | paired with the authors' final, filtered table | z against their final: spread, other fringe | envelope z spread |
| --- | --- | --- | --- | --- |
| Spline 4Pi | 369 582 (61.6) | 99 % | 4.4 nm, 0.1 % | 26.9 nm |
| netfit, from 1000 photons | 655 959 (109.3) | 97 % | 6.4 nm, 0.0 % | 24.4 nm |
| netfit, all | 1 024 280 (170.7) | 98 % | 6.4 nm, 0.0 % | 24.5 nm |

netfit finds nearly every molecule the authors kept, its phase z agrees with theirs as Spline 4Pi's does (one
fringe on 0.0 %), its spread against theirs a little larger (6.4 against 4.4 nm; its lateral spread against theirs
11 -- 12 nm against Spline 4Pi's 7 -- 8).  Its extras: from 1000 photons, 415 000 localizations the authors did not
report.  Whether they are molecules is judged by the structure, without a truth: a molecule lies on the filaments
the other blinks trace.  Of netfit's unpaired, 74 % have an authors' localization of another blink (more than 3
frames away) within 30 nm, against 86 % of Spline 4Pi's unpaired, 98 -- 99 % of either's paired, and 22 % of random
positions -- so most of netfit's extras lie on the filaments, and up to about a third may not.  Some of those are
where the authors fit nothing at all (the channels' bottom band, 140 -- 168 px, bright and dense, where Spline 4Pi's
ROIs do not fit either), which this measure cannot tell from invented molecules.  Undecided; the place to look next
is the training's flat background against the real movie's out-of-focus light (*uneven backgrounds*, off in these
trainings).

## 31. 3D drift over the hour-long tubulin movie (P6, 2026-11-25)

§25's P6: the sample's drift in x, y and z over Cell01 (360 000 frames, an hour), measured by *COMET* and *RCC* on
the phase z after *4Pi z*, against the authors' (`Cell01_DCresult.mat`'s `shifts`, a curve per frame).  The
script is `benchmarks/fourpi/tubulin_drift.py`, its outputs in `smappy_p6/` beside the data; the simulations are
`smappy.simulate.simulate_fourpi_drift` and `tests/test_drift_fourpi.py`.

**Which localizations.**  Our own fit covers 45 000 frames (below); the whole hour is the authors' own raw fits, `Cell01_642_tmpresult_20200912.mat`
(11.8 M), made into a 4Pi table: their x and y, their astigmatic z as the envelope (turned to our direction,
`-0.8975 zf + 473.7`, fitted to our envelope on the 283 802 pairs of the first 6000 frames, 53 nm scatter), their
phase (ours less a constant on those pairs, resultant 0.95).  Their CRLB holds x and y but not the phase's error
or the envelope's, so those are modelled from what they have, fitted to ours on the same pairs (the phase's from
photons, modulation and background, R² 0.94; the envelope's from photons and the widths' CRLB, R² 0.79).  So the
localizations are theirs, *4Pi z* and the drift ours: the methods compared on the same data.  *4Pi z* on them:
4.8 % expected wrong fringes, 86 % at least 90 % sure, the envelope scattering 1.4× its modelled error.

**What changed in the drift plugins, and why** (COMET version 6, RCC 3, Fiducials 5):

1. **Ghosts pull little, but the sure fringes are a free gain.**  A localization on the wrong fringe is a period,
   294 nm, off in z: beyond COMET's overlap at its 30 nm target width, and a side peak RCC's ±200 nm axial search
   does not reach.  On simulations with 4 % ghosts the curves from all, from the sure and from the right
   localizations alone were the same to 0.01 nm; with one in six on the wrong fringe (an envelope four times as
   wrong as it says), COMET's z came 5 % closer to the truth from the sure fringes (1.01 → 0.96 nm over three
   seeds), as close as from the right ones; RCC's did not move.  So *sure fringe* (0.9, both plugins): a table
   with `fringe_probability` is estimated from those at least that sure, and all are corrected.  On Cell01
   the curves from all and from the sure differ by 0.1 nm rms per axis on our 6000 frames, and over the authors'
   hour by 0.2 nm (COMET) and 0.3 -- 0.4 nm (RCC) -- each its noise, about.
2. **z on knots of its own.**  *4Pi z* takes the cavity's phase out in 1000-frame windows; what the cavity does
   within a window stays in z, the same for every molecule of a frame -- drift, as far as a drift correction can
   tell.  On our 6000 frames the cavity measured in 100-frame windows (1.5 nm as z) moved 3.8 nm rms away from the
   1000-frame one; over the authors' hour 5.9 nm.  COMET's spline at 2000 frames cannot follow it, and finer knots
   for all axes made x and y noisier (on simulations 0.16 → 0.39 nm).  So the spline has a *z knot spacing*, on
   auto an eighth of the knot spacing for a 4Pi table (250 frames) and the knot spacing otherwise
   (`drift.z_knot_frames`).  On simulations with Cell01's cavity wander, z against the drift the table shows went
   from 2.1 to 0.9 nm, x and y unchanged.  On Cell01 the z the fine knots add correlates with the cavity *4Pi z*
   missed, measured independently in 100-frame windows: 0.84 on our 6000 frames, 0.73 over the authors' hour (a
   30 % sample); the split-half noise of the fine z is 0.29 nm (0.16 at 2000), against a 12 nm swing.
3. **The envelope moves with z.**  Every drift correction (`Drift.apply`) now moves `z_env_nm` with `z_nm`: the
   table says one z twice, and *4Pi z* run again on a corrected table keeps the correction -- its cavity, phase
   less k z_env, takes the shift up as far as its windows follow it, which a sample's drift they do.  Before, a
   second *4Pi z* undid the z correction.
4. COMET's *RCC first* no longer makes RCC's error bar, two RCC runs whose result it threw away.

Order: *4Pi z* first, then the drift.  The drift on the phase z then takes out the sample's drift and what *4Pi z*
left of the cavity together.

**The whole hour** (the authors' fits; COMET as *RCC first* over 120 windows and then within 50 nm on a random
30 % of the localizations -- the whole within 300 nm is 19 billion pairs, 457 GB -- 3.5 min each; RCC over 120
windows, one per 3000-frame cycle as theirs, 3 min):

| ours | x vs theirs (rms) | y | z | peak to peak x, y, z (theirs 44, 77, 104) | correlation x, y, z |
| --- | --- | --- | --- | --- | --- |
| COMET, sure fringes, z knots 2000 | 0.5 nm | 0.5 | 34.0 | 45, 79, 80 | 0.999, 1.000, 0.35 |
| COMET, all, z knots 2000 | 0.6 | 0.6 | 33.9 | 45, 79, 80 | 0.999, 0.999, 0.35 |
| COMET, sure fringes, z knots 250 (the default) | 0.5 | 0.5 | 34.1 | 45, 79, 87 | 0.999, 1.000, 0.34 |
| RCC, sure fringes | 0.5 | 0.6 | 34.1 | 46, 79, 79 | 0.999, 0.999, 0.34 |
| RCC, all | 0.6 | 0.8 | 34.0 | 46, 79, 79 | 0.999, 0.999, 0.35 |

- **Lateral: the same drift**, to half a nanometre over an hour of 44 and 77 nm, by both methods.
- **z: different curves, the same z.**  The curves differ by 34 nm rms, most around cycle 61, yet the corrected z
  agree: on the 5.1 M localizations of their final table, our z less our drift minus their final z has a median
  per cycle that varies by 2.5 nm (sd; 14 nm peak to peak) over the 120 cycles -- 17 nm before our correction.
  The curves split the same movement differently.  Theirs is the phase's alone: their pipeline has no cavity, so
  their drift is the sample's and the cavity's together.  Ours is what remains after *4Pi z* has tied each window's
  z to the envelope.  The event at cycle 61 shows it: near frame 183 000 the envelope and the phase both step by
  about −45 nm (the sample moved; both curves step), and near frame 209 000 the envelope steps back by +60 nm
  while the phase does not.  *4Pi z* follows the envelope, so our raw z steps back and our drift with it; theirs,
  from the phase, never had the second step.  After each pipeline's own correction the z agree throughout.  Which
  is the sample's "true" z drift depends on the reference -- the objectives' focus (the envelope) or the
  interference's zero (the phase) -- and does not matter for the picture: a drift correction on the structure
  takes out whichever is left.
- **Sure fringes or all, coarse or fine z: the same against theirs** (within 0.1 nm rms), as it must be at
  their 3000-frame resolution; the fine z's gain is inside a cycle, which their curve cannot show (above).

**Our own fit of 45 000 frames** (Cell01's first 15 files, copied once the share was mounted; a sixteenth filled
the Mac's disk and was left).  `tubulin_fit.py` with P1's model (pupil, 680 nm, 100 nm beads) and settings: 760 s,
59 frames/s, 2.08 M localizations, 3.7 GB; *4Pi z*: 5.8 % expected wrong fringes, 86 % at least 90 % sure.  The
drift on all of them (COMET 15 -- 19 s, RCC 10 -- 16 s):

| ours | x vs theirs (rms) | y | z | peak to peak x, y, z (theirs 13, 36, 17) |
| --- | --- | --- | --- | --- |
| COMET, sure fringes, z knots 2000 | 0.6 nm | 0.6 | 4.3 | 16, 36, 6 |
| COMET, all, z knots 2000 | 0.6 | 0.6 | 4.3 | 15, 36, 6 |
| COMET, sure fringes, z knots 250 (the default) | 0.6 | 0.6 | 4.8 | 16, 36, 16 |
| RCC, sure fringes, 15 windows | 0.6 | 0.6 | 4.2 | 16, 37, 6 |

- Lateral again the same as theirs to 0.6 nm.  The z the fine knots add correlates 0.83 with the cavity *4Pi z*
  missed (3.6 nm rms), as on the first 6000 frames.
- **The corrected z against their final z**, on the 709 798 of ours paired with their final table (same frame,
  within 0.5 px): the median difference per cycle has an sd of 1.9 -- 2.0 nm with any of our curves (2.1 before
  ours).  Per 250 frames it is 4.0 nm with z knots 2000, 3.8 with RCC, 4.7 with 250: the fine z *disagrees* more
  with theirs inside a cycle, where their z, the phase's, still carries the cavity's wander and their per-cycle
  curve cannot take it out.
- **Which is right, measured on the structure itself.**  The drift estimated from one half of the field (x below
  or above its median), and judged on the other: pairs of its localizations within 15 nm laterally and more than
  1000 frames apart (1.95 M pairs, fringes 99 % sure), their z difference.  Robust spread, none / knots 2000 /
  1000 / 500 / 250 / 125: 24.05 / 23.96 / 23.87 / 23.69 / 23.58 / 23.55 nm on one half, 23.32 / 23.24 / 23.11 /
  22.94 / 22.79 / 22.77 on the other.  Finer z knots make the held-out filaments sharper in z, monotonically, to
  250; 125 adds nothing.  Taking the variances apart, 250 against 2000 removes about 3 nm rms per localization --
  the cavity's wander.  So the wander is error in z, the fine knots take it out, and the default of 250 is where
  the gain stops.

**Not done.**  The ghosts themselves (§25 item 4) stay: the drift no longer cares about them, the picture still
does.  A blink merged across fringes averages a ghost into its z
(`group`'s z is a mean); with the sure fringes taken first the estimate does not see it, but a merged picture
would.

## 32. The output figures (2026-11-25)

Chosen by Jonas on 2026-11-25; drawn by module-level functions (`fourpi.fringe_risk`, `roi_photons`,
`model_channel_sums`, `model_shares`, `channel_shares`, `molecule_model`) and the plugins' `draw_*`.

- ***4Pi z*'s main figure, four panels**: the stripes with the fringe boundaries dashed (half a period from each
  stripe, where the choice of fringe flips; no colour coding of points); the wrong-fringe risk along the
  envelope's z (expected wrong, share below 90 % sure, per z bin); the cavity's phase; and z by fringe on a log
  scale.  The stripes' grey scale now ends at the bins' 99.5th percentile: on the first 10 000 Nup96 frames a
  fiducial bead (13 % of the localizations within 50 nm, at z_env ≈ 470 nm) took the scale and left the stripes
  white.
- **z by fringe was tried and kept.**  On the 2017 tubulin (6000 frames, in situ model) the filaments' layer
  shows its ghosts as peaks exactly ±1 and ±2 periods (282 nm) away, each in its fringe's colour, about a tenth
  of the main peak -- the result line's 13.6 % expected wrong fringes, seen.  On Nup96, where the nucleus fills
  z, the fringes are bands side by side and show nothing; not misleading, and the page says to read the risk
  panel there.  A fringe's edges move with the cavity, so one structure can be split between colours.
- **The channels' split against the phase**, *4Pi z*'s figure *channels* (and so with every Spline 4Pi fit,
  whose *after the fit* runs *4Pi z*): each channel's share of the photons in bins of the fitted phase, against
  the model's prediction for the same localizations.  The per-channel photons are **ROI sums less the fitted
  background**, `roi_photons_ch0..3`, written by the 4Pi engine -- not a refit with photons free per channel,
  which absorbs the interference into the photons and leaves the phase undetermined (`psf.Global4PiPSF`), and
  would be a second fit of every molecule.  The model's split is summed over the same box (the table keeps
  `metadata["fourpi"]["roisize"]`).  It lives in *4Pi z* because the figure is about the phase and *4Pi z*
  already reads the model; the fit itself only writes the columns.  On Nup96 (in situ v2b) the four sinusoids
  follow the model to about 0.01 except where the bead's localizations fill one phase bin, where model and data
  bend together; on the tubulin (in situ) channel 3 sits about 0.02 above the model near ±π and channel 2's
  minimum is a little late -- the kind of offset the figure is for.  Spline 4Pi's version stays 5: the new columns
  move no number.
- **Spline 4Pi's Preview, figure *molecule***: the frame's brightest molecule whose fit is typical (a mean squared
  Pearson residual within 10 % of the frame's median; the brightest boxes are overlapping pairs), its data,
  model and residual in noise units in all four channels.  0.4 -- 0.6 s for the whole Preview on both sets.
  Nup96's residual is noise; the tubulin's shows lobes in channels 1 and 3, the model's shape off there.

## 33. Does 4Pi tolerate higher densities than astigmatism? (2026-11-25)

Jonas's hypothesis: a 4Pi image tolerates denser emitters than a single-objective astigmatic one, because a
molecule's light is split over the four channels by its phase -- two overlapping molecules at different phases
are bright in different channels, so in some channel each is much dimmer than its neighbour, and the pair
separates.  A simulation study, in `benchmarks/decode/density_4pi/`.

### Design

**The same molecules for both.**  One truth per movie -- positions, z, photons, blinking, the cavity's phase --
drawn as `netfit_4pi/movie_compare.py` draws it (§30, stage 3): molecules switching on at a rate that keeps
*d* on per 40 × 40 px, an exponential on-time of mean 1 or 3 frames, x, y uniform over the region all four
channels see, z uniform in ±500 nm, the cavity drifting.  It is drawn twice: as a 4Pi movie
(`fourpi.render_frames`, the model's own layout and maps) and as a single-channel astigmatic movie of channel 0's
pixels (`simulate.camera._draw_spline`).  Densities 0.5, 1, 2, 5, 10, 25 and 50 per 40 × 40 px; with 129-nm
pixels 40 × 40 px is 26.6 µm², so 0.019, 0.038, 0.075, 0.19, 0.38, 0.94 and 1.9 per µm².  Scored where both
read (the box less 8 px), near the focus (|z| < 300 nm) first, then over the whole ±500 nm.

**Fair photons.**  The default: the same molecule has the same total detected photons in both images (1000 ±
800, from 100 -- network A's range, Nup96's), and the same background summed over the 4Pi channels as the
single-channel image has per pixel (3.75 a channel, 15 in the astigmatic image).  The 4Pi image has four times the
pixels, and each its quarter of the background: per molecule the same light and the same background, only how it
is spread differs.  The variant: the astigmatic image with half the photons and half the background (one objective
of the two), the same molecules; there the floor below which a molecule counts neither way is 50 photons, not 100,
so that the same molecules are judged.

**The PSFs.**  4Pi: the Nup96 v2 bead model (§27).  Astigmatic: the v2 model's *first objective's* pupil alone,
as it was learnt -- every phase mode, γ, the blur -- drawn by uiPSF at the model's NA (1.35), emission (600 nm),
immersion (1.406) and pixel, and sampled into a Spline 3D calibration (`make_astig.py`).  Not the 4Pi channels'
incoherent part, Σ_c I_c: the two objectives' astigmatism is opposite (Noll 6, +46 nm and -32 nm), so their sum is
nearly round and would be a poor astigmatic microscope.  The first pupil is the same optics and aberrations, minus
the second objective and the interference.  The bounds at 1000 photons, background as above: lateral 5.7 -- 26 nm
astigmatic against 5.8 -- 13 nm for 4Pi over ±500 nm (the second objective's opposite astigmatism keeps the 4Pi
spot compact out of focus); z 20 -- 32 nm astigmatic, the 4Pi envelope's 19 -- 23 nm and its phase's 4.2 -- 5.7 nm.

**Methods, each with its own detection.**  netfit: the 4Pi network A of §30 (25 per 40 × 40 px, on-time 1 -- 3,
3-frame window, frame stack, early fusion) and two single-channel astigmatic networks trained the same way -- the
same parts, window, density, on-time and photon range, the background four times A's (6 -- 24) -- one at the
default photons and one at half (500 ± 400 from 50, background 3 -- 12); `jobs/astig_*.json`, trained on
decodebox.  Fitters: Spline 4Pi and Spline 3D with the plugins' default detection and fit (*focus* as
calibrated).

**Metrics against density**: recall, precision and Jaccard; lateral and z RMSE, absolute and as the robust width
of the error over each mode's own single-frame CRLB; for 4Pi the envelope z and z from the phase with the cavity
known (`fourpi.unwrap_known`), and the wrong-fringe share.  One axis per metric, both modes on it.

**The mechanism.**  (1) For two molecules of equal brightness at the same place, each channel's share of each
molecule's light, from the model, against their phase difference Δφ: the largest ratio between the two over the
channels.  (2) From the density runs, the recall of molecules with a neighbour on in the same frame within
250 nm laterally, binned by |Δφ| -- for 4Pi it should rise with |Δφ|, for astigmatism (binned by the same Δφ,
which there means nothing but Δz modulo the fringe period) it should not.

### Results (2026-11-25)

**Yes, but by little: the 4Pi network holds its numbers to about 1.2 times the density the astigmatic one does,
and the reason is the one Jonas proposed.**  The runs: `sweep.sh` on gpuvm (molecules per row: ~15 000 at the
sparse end, up to 90 000 at 50 per 40 × 40 px), the astigmatic networks trained on decodebox (`run.sh`, ~40 min
each), scored by `plot.py` from every molecule (`results/summary.json`, `metrics_vs_density.png`,
`mechanism.png`; the per-molecule `.npz` files, 170 MB, stay on gpuvm in `~/netfit/smappy_density/bench/
density_4pi/results`).  Near the focus, |z| < 300 nm, unless the column says otherwise; lateral and z errors over
each mode's own single-frame CRLB as robust widths:

| per 40² px (per µm²) | | netfit 4Pi | netfit astig. | netfit astig., half | Spline 4Pi | Spline 3D |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 (0.02) | recall / Jaccard (all z) | 0.988 / 0.923 | 0.990 / 0.872 | 0.955 / 0.766 | 0.908 / 0.818 | 0.917 / 0.818 |
| | lat. RMSE, width | 18.7 nm, 0.71 | 21.0 nm, 0.67 | 26.7 nm, 0.67 | 21.7 nm, 1.04 | 25.5 nm, 1.01 |
| | z RMSE, width | 52.7 nm, 0.69; phase: 7.6 nm, 3.9 % wrong | 48.2 nm, 0.64 | 57.5 nm, 0.63 | 61.9 nm, 1.05; phase: 9.3 nm, 4.8 % | 58.5 nm, 1.00 |
| 5 (0.19) | recall / Jaccard | 0.981 / 0.935 | 0.982 / 0.929 | 0.938 / 0.791 | 0.807 / 0.711 | 0.811 / 0.702 |
| | lat. RMSE, width | 20.0 nm, 0.73 | 24.0 nm, 0.69 | 28.1 nm, 0.70 | 29.7 nm, 1.14 | 36.1 nm, 1.13 |
| 25 (0.94) | recall / Jaccard | 0.945 / 0.888 | 0.942 / 0.875 | 0.890 / 0.731 | 0.323 / 0.229 | 0.305 / 0.212 |
| | lat. RMSE, width | 24.5 nm, 0.83 | 29.5 nm, 0.83 | 34.9 nm, 0.81 | 46.0 nm, 2.01 | 57.0 nm, 2.07 |
| 50 (1.9) | recall / Jaccard | 0.898 / 0.827 | 0.885 / 0.797 | 0.826 / 0.654 | 0.043 / 0.027 | 0.043 / 0.027 |
| | lat. RMSE, width | 29.5 nm, 0.99 | 35.7 nm, 1.03 | 41.1 nm, 0.99 | 49.5 nm, 3.83 | 61.8 nm, 3.85 |
| | z RMSE, width | 75.6 nm, 1.09; phase: 10.4 nm, 7.6 % wrong | 75.7 nm, 1.07 | 86.7 nm, 1.02 | 134 nm, 5.07; phase: 9.4 nm, 21.5 % | 144 nm, 5.26 |

(On-time 1 frame, 3-frame window; on-time 3 is the same picture, every network 5 -- 15 % more precise; all in
`summary.json`.)

**The networks.**  The two full-photon networks lose recall and precision with density almost alike.  From 5 to
50 per 40² px recall near the focus falls 0.981 → 0.898 (4Pi) and 0.982 → 0.885 (astigmatic), Jaccard over all z
0.935 → 0.827 and 0.929 → 0.797; the lateral error over the CRLB grows 0.73 → 0.99 and 0.69 → 1.03, z 0.74 →
1.09 and 0.66 → 1.07.  Read as density at equal quality: a Jaccard of 0.85 at 38 per 40² px for 4Pi and 31 for
astigmatism, recall near the focus of 0.90 at 49 and 42 -- 1.2 times, at both on-times.  In nm the 4Pi network is
ahead by its smaller bound, not its handling of neighbours: lateral 18.7 → 29.5 nm (×1.58) against 21.0 → 35.7 nm
(×1.70).  Over the whole ±500 nm 4Pi's lateral lead is larger (34.5 against 46.5 nm at 50), because its spot stays
compact out of focus -- the two objectives' astigmatism is opposite -- which is optics, at every density.  The
envelope's z is no better than astigmatism's (the bounds are alike, 19 -- 23 nm against 20 -- 32 at 1000 photons);
the phase's z is 6 -- 7 times better at every density, 7.6 → 10.4 nm with 3.9 → 7.6 % on a wrong fringe.  At the
sparse end both networks are less precise than at 2 -- 5 per 40² px (Jaccard 0.92 and 0.87 at 0.5, recall 0.99):
they were trained at 25, and invent molecules where there are few.

**Half the photons** (one objective) costs more than any density here: at 0.5 per 40² px the half-photon network
recalls 0.955 near the focus, what the full-photon networks still recall at 25.  Against that baseline 4Pi's
advantage is large at every density, but it is the second objective's photons, not the phase.

**The fitters do not see the difference.**  Spline 4Pi and Spline 3D lose molecules identically -- recall 0.81
and 0.81 at 5 per 40² px, 0.32 and 0.31 at 25, 0.04 at 50 -- because Spline 4Pi's detection finds candidates on
the channels' sum, which carries no phase; past detection both fits suffer the neighbours in their ROIs alike
(2× the CRLB at 25, 4× at 50).  Spline 4Pi's phase z keeps 7 -- 9 nm where it fits, but its wrong-fringe share
rises 4.8 → 21.5 %, the network's 3.9 → 7.6 %.  Only a method that reads the channels apart can use the phase
for density.

**The mechanism, shown directly** (`mechanism.png`).  *From the model* (`contrast.py`): a molecule's channel
shares swing between 0.15 and 0.36 with its phase (the modulation over a 4-px disc is 0.45 -- 0.47, less than the
bead model's coherence of 0.8, because A and B vary across the spot); for two equal molecules at one place the
channel where one is dimmest against the other has a ratio of 1.38 at Δφ = π/4, 1.8 at π/2 and 2.2 at π (1.6 at
-300 nm), and 1 in an astigmatic image.  *From the runs*: molecules near the focus with a neighbour on in the same
frame within 250 nm, densities 5 -- 50 pooled (~20 000 molecules each on-time), recall by |Δφ|:

| \|Δφ\| (rad) | 0 -- 0.5 | 0.5 -- 1.0 | 1.0 -- 1.6 | 1.6 -- 2.1 | 2.1 -- 2.6 | 2.6 -- 3.1 |
| --- | --- | --- | --- | --- | --- | --- |
| netfit 4Pi, on 1 | 0.767 | 0.772 | 0.804 | 0.816 | 0.831 | 0.848 |
| netfit astig., on 1 | 0.783 | 0.779 | 0.794 | 0.797 | 0.800 | 0.791 |
| netfit 4Pi, on 3 | 0.750 | 0.758 | 0.790 | 0.816 | 0.804 | 0.838 |
| netfit astig., on 3 | 0.773 | 0.776 | 0.790 | 0.792 | 0.776 | 0.794 |
| Spline 4Pi / 3D, on 1 | 0.17 / 0.17 | 0.16 / 0.16 | 0.16 / 0.16 | 0.17 / 0.16 | 0.16 / 0.16 | 0.16 / 0.15 |

The 4Pi network separates pairs better the more their phases differ, by 8 -- 9 points from Δφ ≈ 0 to π; the
astigmatic network, binned by the same Δφ (for it, Δz modulo the fringe period), is flat within 2 points.  At
Δφ ≈ 0 the 4Pi image is slightly *worse* (0.767 against 0.783): with the phases equal the channels split both
molecules alike, and what is left is a less astigmatic spot spread over four times the pixels.  So the phase is
the whole of 4Pi's gain at density, and it is a modest one, because the channel contrast it gives -- at most a
factor of 2.2 -- is far from the factor that would make a neighbour vanish in a channel.

**What this does not cover.**  One 4Pi model (Nup96's; a model with higher modulation should do better -- the
contrast grows as (1 + m) / (1 - m)) and one astigmatic counterpart (its first objective's 46 nm of Noll 6;
stronger astigmatism trades lateral for z precision); networks trained at one density (25 per 40² px); the
network A of §30 as it was, not retrained beside the astigmatic ones.  Real data adds what neither simulation has
(uneven background, the cavity's drift on a real model).

## 34. The tubulin model refined in situ, judged by the figures (2026-11-25)

§32's figures showed P1's bead model (§26: three one-bead scans, 680 nm, 100 nm beads, maps held from the bead
fields) off on Gabin's tubulin: in *4Pi z*'s *channels* figure channel 3 about 0.02 above the model near ±π, in
Spline 4Pi's Preview residual lobes in channels 1 and 3.  The task: refine the model in situ until they go, with
the fewest parameters, and judge it by them.  Scripts in `benchmarks/fourpi/insitu/` (`tubulin_misfit.py`,
`tubulin_channel_drift.py`, `tubulin_refine.py`, `tubulin_share_check.py`, `tubulin_blinks.py`; the README's
order), outputs in `smappy_refine/` beside the data.  Cell01 now has 45 000 frames; the molecules come from all
of them, the measures from the first 6000 as in §26 and §29, everything judged on the even blocks of 500 frames
and learnt from the odd ones.  Learning ran on gpuvm's H100: 1 -- 1.5 minutes a model.

**Most of the *channels* misfit was the figure's own.**  `fourpi.channel_shares` set the data's *median* share in
a phase bin against the model's *mean* share of the same localizations.  A bin holds molecules at every z, whose
shares swing far near the focus and little away from it, so they are skewed within a bin, and their mean swings
less than their median.  Frames drawn with P1's model itself at the tubulin's brightness and background (5500
photons, 34 a pixel), fitted by the 4Pi engine (`tubulin_share_check.py`), showed the data swinging 1.20 times the
model's line and channel 3 +0.011 near ±π, channel 1 −0.010 -- with the model the truth; even noise-free shares
equal to the model's swung 1.20 times.  The line is now the model's median too: the same simulation's residual
is below 0.001 in every channel, the swing 1.00 (`test_noise_free_data_equal_to_the_model_lie_on_its_line_...`).

**What is left of the misfit with P1's model** (`tubulin_misfit.py`; on the even blocks, the split in 24 phase
bins; the residual over the ROIs of 300 even frames, the half with the smaller χ² of those above 2000 photons --
the dense filaments put neighbours in the rest -- binned in 3 z × 4 phase cells, *systematic* the mean residual
image's n · mean² − 1 per pixel, 0 for noise):
- *The split*: rms 0.0038, 0.0024, 0.0035, 0.0034 about the model per channel; channel 3 +0.005 near ±π; channels 1
  and 3 swing 7 % more than the model, in phase with their own swing (coherence too low); the means within 0.003.
- *The residual*: χ² per pixel 1.32, 1.29, 1.29, 1.24; systematic 16.5, 15.8, 12.8, 9.1 per pixel.  Channel 2
  shows one dipole, up and down, in every cell of z and phase: a shift.  Channels 1 and 3 show dipoles that turn
  with the phase, opposite in the two -- what a lateral misalignment of the arms' foci does to the interference
  term -- and, beneath, a peak at the centre with a dark ring.
- Spline 4Pi's *channel drift* (x and y free per channel) puts channel 2 at 0.21 px (27 nm) in y and 0.07 px in x
  from where the registration has it, steadily over the movie, and channels 3 and 4 wandering by 0.16 -- 0.19 px.
  P1's maps came from the bead fields Cell06 -- 12 (§26), which the movie's alignment no longer is.

**In situ, a few parameters at a time.**  `tubulin_refine.py`: from P1's model (relearnt with its recipe, §29),
2935 molecules of the odd blocks, 1950 of the even held out, each alone within 1.5 boxes in its frame and the
frames beside it (`fourpi_insitu.alone`, new: §29's boxes held their neighbours' light), above the median photons
(5500); two rounds, tied to the cavity (§29, σ 0.2 rad).  The sets, smallest first; *held out* is the gain in
nats per held-out molecule over the bead model, first on these isolated molecules, then on §29's dense held-out set
(1900, deviance 2.8 a pixel); *split rms* the four channels'; *systematic* per pixel per channel:

| learnt | held out: isolated / §29's | split rms | ch 3 near ±π | systematic | uncertain fringes | z env vs phase (nm) | z vs their final; other fringe | env. z vs their final | x, y vs theirs (nm) | blink pairs x, y, z (nm) | period (nm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| beads (P1) | -- | .0038 .0024 .0035 .0034 | +.0053 | 16.5 15.8 12.8 9.1 | 7.17 % | 44.5 | 4.4 nm; 0.1 % | 26.9 | −1.9, +0.5 | 13.5, 15.3, 7.5 | 294.2 |
| maps | 5.8 / 3.7 | .0039 .0025 .0035 .0037 | +.0054 | 13.9 6.6 9.7 8.6 | 7.24 % | 44.7 | 4.4 nm; 0.1 % | 27.2 | −4.6, −4.7 | 13.5, 15.3, 7.5 | 294.2 |
| + misalignment Δ | 6.4 / 2.2 | .0038 .0025 .0033 .0036 | +.0050 | 12.4 6.2 8.9 7.7 | 7.19 % | 44.6 | 4.3 nm; 0.1 % | 26.9 | −4.9, −6.1 | 13.4, 15.2, 7.5 | 295.1 |
| **+ each channel's interference** | **8.5 / −2.0** | **.0022 .0017 .0025 .0021** | **+.0016** | **10.1 5.4 8.5 4.0** | **6.91 %** | **43.8** | **4.4 nm; 0.0 %** | **26.5** | −5.2, −7.9 | 13.4, 15.3, 7.4 | 294.9 |
| + blur | 4.9 / −20.8 | .0024 .0017 .0025 .0019 | +.0018 | 6.9 4.2 6.5 3.6 | 6.94 % | 44.2 | 4.4 nm; 0.0 % | 26.6 | −5.8, −8.8 | 13.3, 15.2, 7.5 | 296.5 |
| + pupils (everything) | 5.2 / −21.6 | .0021 .0017 .0026 .0024 | +.0028 | 4.9 3.9 5.3 2.8 | 6.95 % | 44.7 | 4.5 nm; 0.1 % | 26.9 | −6.1, −0.4 | 13.4, 15.2, 7.5 | 296.4 |

*Each channel's interference* is the coherence, the arms' balance, the phase offset and the share; *blink pairs*
the robust sd of the differences between one blink's fits in consecutive frames (`tubulin_blinks.py`, both sure of
their fringe), which needs no reference; the measures as `tubulin_evaluate.py` and `tubulin_compare.py` give them.

- **The maps remove channel 2's dipole** (systematic 15.8 → 6.6): its map moved 0.19 px in y and 0.11 px in x,
  channel 4's 0.16 px in y, the shift the channel drift measured.  But only once Spline 4Pi reads them: with P1's
  registration file, as P1's script set it, the fitter put the bead fields' maps back and the maps model gave the
  bead model's numbers to the last digit.  Spline 4Pi now says so (below).
- **The misalignment and each channel's interference take most of the rest**: Δ (x, y, z) from 28, 70, 106 to
  5, 48, 133 nm -- a 23 nm lateral misalignment less, which makes the dipoles that turn with the phase -- the
  coherence 0.985, 0.874, 0.992, 0.824 → 1.00, 0.93, 1.00, 0.89, the balance 0.44, 0.56, 0.36, 0.52 → 0.50, 0.51,
  0.45, 0.46, the phase offsets by 0.03 rad, the shares by 0.01.  The split's rms falls by 30 -- 40 %,
  channel 3 near ±π from 0.005 to 0.0016; the held-out molecules are explained 8.5 nats each better, the most of
  any set.  The tie does not matter here: untied and at σ 0.7 rad, 8.5 and 8.4 nats, the same values.
- **The blur and the pupils make the Preview cleaner and the model worse.**  The blur from 84 to 74 nm takes the
  centre peak and its ring out of channels 1 and 3 (systematic 10.1 → 6.9, 8.5 → 6.5), but explains the held-out
  isolated molecules worse than the channels alone (4.9 against 8.5 nats; untied 4.8: not the tie) and §29's dense
  ones 21 nats each worse than the beads, and moves the period 0.8 %.  So a narrower core fits the bright typical
  molecules of the Preview and not the rest: one Gaussian blur cannot do both.
- **No set changes what is measured.**  The uncertain fringes 7.2 → 6.9 %, the envelope against the phase 44.5 →
  43.8 nm, z against the authors' final 4.4 nm and 0.0 -- 0.1 % on another fringe for every model, the envelope
  against it 26.5 -- 27.2 nm, and a blink's two fits differ by 13.4, 15.3 and 7.5 nm in x, y and z under every
  model (1.7 -- 1.9 times their errors: the dense filaments' neighbours, not the model).  The learnt maps move x, y
  against the authors' by 5 -- 8 nm and widen y's spread from 8.4 to 9.3 nm; since the blinks' agreement does not
  change, that is the maps' new origin against the authors' registration (theirs, like P1's, from the bead
  fields), not lost precision.
- **What is left is a model-class limit, two of them.**  Channels 1 and 3 still swing 4 % more than the model
  (the split's first harmonic 0.0026 against a swing of 0.065), with their coherence at its bound of 1: the model
  cannot make them interfere more.  The light lost to the interference may be the 100 nm bead's blur of the
  fringe, which P1 deconvolved with an assumed size (§26: coherences 0.82 -- 0.99 at 100 nm, near the bound
  already), or the arms' pupils overlapping better on the molecules than on the beads -- the pupils, learnt here,
  did not find it.  And the core: one blur for every z and brightness (above).  Neither moves z.

**Compared with §29.**  §29's in situ learnt everything from molecules alone in their frame only and gained 93
nats per held-out molecule on §29's dense set by taking the neighbours' light up as blur and low coherence; it
measured z worse.  Learnt from isolated molecules, the same parameters lose 21.6 nats there and the channels'
set 2.0: they no longer describe the neighbours.  The dense set's deviance, 2.8 a pixel against 1.5 for the
isolated, is the neighbours'.

**The model** is kept beside P1's: `smappy_p1/pupil_680_100_held_insitu_channels_4pi.h5` (in situ, the maps, the
misalignment and each channel's interference).  It improves both figures and worsens no measure, and it is the
one to use for this movie with Spline 4Pi's *registration* on *the model's*.  In the PSF window: sample *in situ*,
mode *4Pi*, *bead model* P1's pupil model (saved with its recipe, `smappy_insitu/pupil_680_100_held_4pi.h5`),
*localizations* a Spline 4Pi fit of the movie with it, and under *more* *alone within* 1.5 boxes, *alone over* 1
frame, *learn* *maps, misalignment, channels*.  Run through `run_insitu` on the same molecules (a fifth held out at
random) it took 86 s on the H100, gained 9.4 nats per held-out molecule, raised no warning and learnt the same
values (Δ 5, 47, 131 nm; coherence 1.00, 0.93, 1.00, 0.89; the blur held at 84 nm).

**What changed.**
- `fourpi.channel_shares`: the model's line is its median (above); *4Pi z*'s page says why.
- *Spline 4Pi* says, as a notice, when its model learnt the channels' maps in situ and the *registration* is
  from a file or beads, which would replace them (`fit_4pi.maps_learnt_in_situ`).
- *4Pi in situ* (the PSF window): *alone within* (boxes) and *alone over* (frames beside, the molecule's own blink
  within a pixel not counted; `fourpi_insitu.alone`), defaults as before, 1 box and 0 frames; *learn*:
  *everything* (default, §27's set) or *maps, misalignment, channels* (`fourpi_insitu.learnt`, the misalignment
  loose in x and y too).  uiPSF's `InSituSettings.free` learns only the named shared parameters.
- The tubulin scripts: `TUBULIN_MAPS=model` fits with the model's maps (`tubulin_common.registration_settings`).

**Not done.**  Whether *maps, misalignment, channels* should be the window's default (here it was better; on
Nup96 everything was learnt and was fine, §27), and the isolation's; a blur that depends on z or brightness, or a
coherence that may exceed what the beads' size allows, for the 4 % of modulation and the core; the remaining
1.7× scatter of a blink's fits, which is the neighbours'; Nup96 with this set.
