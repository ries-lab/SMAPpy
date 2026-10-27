# 4Pi in SMAPpy: what exists, what is wrong with it, and a plan

Status, 2026-10-27: M0-M4 are done (§6-§8).  4Pi runs on the channel layer of M8
(§9), and on real DCIMG data (§10).  M5, learning from beads in uiPSF v2, is
merged (§11); its window method is still to come.  Dual colour on the layer
(M8) and M6-M7 are proposals.

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
  vectorial is a later step (M7).  v2's `VectorPSF` already has the six dipole
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

**M5. uiPSF v2 4Pi from beads** (done but for the window method: §8)
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
  polarisations of the channels (D7).
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
- **Open: the bead kernel and the coherence trade.**  With a 20 nm bead the coherences rise to
  0.89–0.95 at the same deviance, although a 20 nm sphere averages a 265 nm fringe by only 2 %.  So
  the bead kernel is off by default, as in v1, until that is understood.  It also leaves the export a
  point emitter's, which the beads then fit worse.
- **Speed:** every shared column is a JVP (the 4Pi model has no linearisation yet, `Problem` falls back),
  so n ≤ 8 takes 14 min on the Mac's GPU.  Linearising the pupils' columns as `VectorPSF` does is the
  next step for speed.

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
  has its own entry.

**Still to do for M5.**
- The beads' fitted positions against v1's own (`locres`), bead by bead.
