# 4Pi in SMAPpy: what exists, what is wrong with it, and a plan

Status, 2026-10-28: M0-M5 are done (§6-§8, §11, §14): the fitter, the channel
layer (§9), real DCIMG data (§10), learning from beads in uiPSF v2 and the
bead average, both in the PSF calibration window, live DCIMG and simulated 4Pi
cameras (§16).  Of M7 the vectorial model is in and the window's default
(§17, §19), with a start that reaches one optimum (§20); the locked refit, multicolour and two cameras are not started.  M6
(in situ) waits on the in-situ learning; dual colour on the layer (M8) is
planned, not started.

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
work in progress elsewhere, which it builds on.

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
- **Dual colour** still runs on `dualfit`.  Its port is M8 (§9), as part of
  making the channel geometry one layer for every multi-channel fit.
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
Read as one channel's bead stack they had failed with "not a TIFF file".


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
