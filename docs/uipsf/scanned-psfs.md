# Scanned PSFs: confocal, ISM, STED and MINFLUX beams, and phase masks

November 2026.  A study, not a plan yet: what it takes for uipsf to learn the PSFs of scanning microscopes --
an excitation beam focused into the sample, a pinhole in front of the detector, and for MINFLUX and STED a
phase mask or SLM that makes the beam's zero.  It also covers phase masks in the pupil in general, which
widefield engineered PSFs (double helix, tetrapod) need as well.  No code has been written.  It extends the
roadmap's "Excitation PSFs" and "Phase masks in the pupil" ideas (`roadmap.md`), and draws on three sources:

- uipsf's forward model as built (`model/psf.py`, `optics/`);
- the lab's simulators, SimuFLUX (`ries-lab/SimuFLUX`, Marin & Ries 2025) and MINFLUXexcitation (Deguchi &
  Ries 2024).  Both run Leutenegger's vectorial Debye library (chirp-z, Leutenegger et al. 2006) with the
  lab's masks.  They are what to build on for the physics and what to test against, though they do no
  fitting;
- the literature (references at the end).

## 1. What is new, in one paragraph

In a widefield z stack the detector images the emitter: one exposure gives a whole 2-D picture of the
detection PSF.  In a scanning microscope the detector is a point (or a small array).  The picture is built
up by moving the beam or the sample, so each count is one value of the *effective* PSF at one
beam–emitter offset:

    μ(r_s) = N · T · [ I_exc(r_e − r_s) · D(r_e − r_s) ] ⊛ bead  +  b,

where:
- I_exc is the focused excitation intensity;
- D is the detection efficiency, the emission PSF convolved with the pinhole;
- r_e is the emitter's position and r_s the beam's;
- T is the dwell time;
- ⊛ bead is the convolution with the bead's size.

Three things are new to uipsf:
- a second focusing problem, the excitation, at its own wavelength, with an input beam, a polarisation and
  often a mask;
- the product of two PSFs;
- data that are counts at scan positions rather than camera pixels: a regular raster over a bead or a
  single molecule.

**Scope (decided November 2026).**  Scanned beads and scanned single molecules only; no in situ learning
from MINFLUX localisations' counts.  Often only a single focal plane can be taken, not a z stack (§7a).

Most of the rest carries over:
- the pupil grid, Zernikes, the defocus and depth waves, the layered model, n_med;
- the bead kernel, the matrix DFT, the parameter/solver machinery and the emitter-independence contract;
- in situ's idea of learning shared parameters from many emitters, each with its own position.

## 2. What carries over, and where

| needed | in uipsf today | change |
| --- | --- | --- |
| pupil grid, k_z, Fresnel, apodisation | `optics/grid.py` `PupilGrid` | a second grid at λ_exc; Fresnel for light going *into* the sample (below) |
| vectorial fields | `optics/vectorial.py` `_vector_fields` | none: the focal field of x- and y-polarised input *is* `hx`, `hy` (below) |
| field to points | `optics/mdft.py` `MatrixDFT` | the "image" coordinates become scan positions; a non-separable DFT for TCPs |
| defocus, depth, tilted coverslip | `VectorPSF._waves`, `optics/layers.py`, `optics/medium.py` | none.  The 3-D top-hat needs depth and n_med from the start |
| Zernike / minimal pupil | `model/pupil.py` | an input-beam magnitude (Gaussian fill, decentred) in place of the Zernike magnitude; masks (§6) |
| bead size | `optics/kernels.py` `sphere_otf` | the same 3-D OTF, on the scan grid, applied after the product |
| pixel integration | `bin_pixels`, `blur_bin_matrices` | gone: scan samples are points.  A box along the fast axis for motion during the dwell |
| coherent sums of two fields | `VectorPSF.field_linearisation`, `model/fourpi.py` | reused for Jones pupils and for the coherent scattering of gold beads |
| channels sharing parameters | `model/channels.py` | ISM: one channel per SPAD element, sharing the excitation |
| Jacobian via tangent pupils | `VectorPSF.pupil_linearisation` | the product rule: an excitation tangent is dI_exc·D, a detection tangent I_exc·dD |
| likelihood | `noise.py` `Poisson` | photon counting: s = 0, no gain; optionally dead time (§5) |
| per-frame factors, bleaching | `model/intensity.py`, `Batch.frame` | per scan line for blinking single molecules (§5) |

## 3. The excitation field

**Reciprocity gives the vectorial part for free.**  Let an input beam with Jones vector (e_x, e_y) at the
back aperture be focused by an aplanatic objective.  Its angular spectrum in the sample is

    E(k) = A(k) · [ (e_x cos φ + e_y sin φ) t_p p̂ + (−e_x sin φ + e_y cos φ) t_s ŝ ],
    p̂ = (cos θ cos φ, cos θ sin φ, −sin θ),   ŝ = (−sin φ, cos φ, 0).

That is exactly the matrix uipsf already builds for detection.  `_vector_fields` returns
`hx = cos φ·p − sin φ·s`, the x-analyser response of the x, y and z dipoles; read the other way, it is
the (E_x, E_y, E_z) focal field of x-polarised input.  So `torch.stack([hx, hy], -1)` applied to (e_x, e_y)
is the excitation field.  Only two things change:

- the Fresnel coefficients are those for light going from the immersion through the coverslip into the
  sample.  They differ from detection's by the ratio of n·cos θ in the two end media;
- the apodisation (√cos θ_imm / cos θ_med in k-space, `grid.apodisation`) should be the same by
  reciprocity.  **Compared (S2, November 2026):** Leutenegger's library weights by 1/cos θ_imm and its
  "air-lens" transmission (`SimuFLUX/python/psfs/library/modifiers.py:64`, `transmission.py`); uipsf keeps
  √cos θ_imm / cos θ_med, which the reciprocity argument gives exactly up to n_imm/n_med.  About 1 % of the
  peak for a Gaussian, 0.12 against 0.06 of the peak at a top-hat's zero.  Weighted as the library weights,
  uipsf reproduces its fields to 0.05 % at 256 pupil samples -- once the library's E_z, n_m too small, is
  corrected (architecture.md, "The excitation beam").

A z position works as in a bead stack: kz_imm·z for the stage, kz_med·depth in the sample, the layered
model with a tilt.  Unlike the widefield case, defocus of the *beam* has two physically different sources:

- *the sample moved* (a stage or piezo z scan): the emitter's depth in the medium changes.  This is
  `BeadStackGeometry`'s z;
- *the beam refocused* (a deformable mirror or a varifocal element, as MINFLUX's z patterns may be): a phase
  in the pupil, kz_imm·Δz, with the emitter's depth unchanged.

With index mismatch the two give different PSFs.  The geometry must say which one moved
(`ScanGeometry`, §8).

**The intensity the emitter sees.**
- *A freely rotating dye:* absorption ∝ |E_x|² + |E_y|² + |E_z|² (SimuFLUX's `effIntensity`).
- *A fixed dipole:* |μ·E|².  Stallinga et al. 2026 show that this biases MINFLUX by up to ~25 % of L.
  Dyes immobilised on glass (in PVA, or on short linkers) can be fixed or partly so; then a scanned
  molecule's image depends on its orientation, two angles per molecule (emitter parameters), with a
  degree of rotational freedom as a third.  Freely rotating is the first model; the orientation one is
  needed when the residuals of single molecules differ from molecule to molecule in shape, not just
  brightness.
- *A fluorescent bead:* many fixed, random dyes, excited and collected without rotating.  The signal is
  Σ_d w_d |E_d|², with the z component weighted by how well a z dipole's emission is collected.  The
  weights can be computed from the detection model's dipole fields (`dipole_fields`: the energy per dipole
  row).
- *A gold bead* scatters coherently.  With an open pinhole the collected power is again ≈ Σ_d w_d |E_d|²,
  w_z < w_xy.  Behind a pinhole it is |Σ_d E_exc,d · E_det,d|², a coherent product of two fields.  That is
  the case `field_linearisation` was built for.

The w_z weighting matters for donuts.  A vortex donut's zero is filled by E_z when the circular
polarisation's handedness is wrong, so gold-bead and fluorescent-bead scans of the same beam differ
exactly there.

**The input beam** replaces the widefield pupil's magnitude.  Its parameters:
- the Gaussian fill, exp(−ρ²/w²), with w relative to the aperture.  Both lab simulators run nearly flat
  pupils, w_a = 7 mm against a 6.5 mm aperture;
- the beam's centre on the aperture (dx, dy): a decentred, truncated Gaussian;
- the polarisation, ellipticity and orientation: two numbers for an imperfect λ/4.

The objective's own transmission (the minimal pupil's cos θ^γ and the supercritical ring) belongs to the
objective, not the beam.  It could be held from the widefield calibration of the same objective, with
γ measured at another wavelength as an open question.

**Normalisation.**
- *Widefield:* each PSF integrates to one photon (`VectorPSF`, decision 5).
- *Excitation:* normalise to the beam's power through the aperture.  Then the photon count N is a
  brightness × power product, and patterns that share a laser (the x, y and z beams of a variable phase
  plate, the Gaussian scout beam) share it, so their *relative* heights are learnt, not absorbed per
  pattern.
- *Detection:* keeps uipsf's normalisation: D is a collection efficiency.

## 4. Detection and the pinhole

D(r) = [I_em ⊛ disc_pinhole](r − r_ph):
- I_em is the emission PSF at λ_em in sample coordinates: a `VectorPSF` intensity, with no pixels and no
  blur;
- the disc has the pinhole's diameter back-projected into the sample, in Airy units of λ_em/NA (SimuFLUX:
  1.22·λ_em/NA·AU);
- r_ph is its lateral misalignment.

The disc's OTF is a jinc, `2 J₁(2π q R)/(2π q R)`, a sibling of `sphere_otf`.  Two corrections to the
lab's simulators:
- SimuFLUX convolves the pinhole with the *excitation*-wavelength PSF and uses λ_em only for the diameter
  (`psf_vectorial.py:265-293`).  A learnt model should use λ_em: the Stokes shift sets the confocal gain
  and ISM's reassignment factor (Sheppard et al. 2020);
- the detection PSF can come from the widefield calibration of the same objective.  It is the same
  pupil, apart from the paths after the dichroic.

**Excitation and detection share the objective.**  Its aberrations are common in optical path (nm), not
in phase (rad); the paths after the dichroic are not common.  One parameterisation:
- shared Zernikes in nm of OPD, converted to radians at each wavelength;
- plus Zernikes of each path, the excitation's usually low order.

Patton et al. 2016 correct the two paths separately, so the path terms are needed.  uipsf's Zernikes are
in rad today, at one wavelength, so this is new.  It is also what makes a two-colour widefield model
physical.

**Descanned or not.**  A galvo-scanned confocal descans: D moves with the beam, and the product is a
function of r_e − r_s.  MINFLUX's EOD steers only the excitation (SimuFLUX `simulator.py:254-272`), so
there D is evaluated at r_e − r_galvo, not r_e − r_s.  Over a pattern of L ≤ 150 nm, D is nearly
constant, a ~1 % effect.  It matters for the larger scout patterns and for ISM-FLUX.

**ISM / SPAD arrays** (Castello et al. 2019; Slenders et al. 2025, ISM-FLUX):
- element c sees D_c(r) = [I_em ⊛ pixel_c](r − d_c): one shared excitation, one detection pupil, and an
  offset d_c for each element;
- the offsets are a lattice: pitch, rotation and centre, four numbers, not 2 per element;
- that is uipsf's channel model with a different "transform" and the excitation shared (`channels.SHARED`).

## 5. The data: scan geometry and the detector

**Bead scans** map onto uipsf's bead stacks almost one to one:
- data [E, C, K, Y, X], with K the z planes and Y, X the scan pixels;
- (x_e, y_e, z_e) per bead, photons, bleaching and background;
- the "pixel size" is the scan step.

Differences:
- *point samples.*  No pixel integration.  With continuous scanning, a box of length v·T_dwell along the
  fast axis instead; with step-and-dwell, nothing;
- *the scanner's calibration.*  Scale and rotation of the scan axes, and for galvos the line-to-line
  shift of bidirectional scans: a few numbers of the geometry;
- *a coarse z.*  Confocal stacks are usually 50–200 nm in z, top-hat donuts need ≤ 50 nm near the zero.

**Scanned single molecules** are bead scans with three complications:
- *they blink and bleach during the scan.*  A pixel is a time; a raster is minutes of lines.  An intensity
  factor per scan line under a prior that keeps it near one -- new: uipsf's `PerFrameFactor` is one factor
  per observation (a whole image), and this varies along the image's slow axis -- and a mask for lines after the molecule has bleached (a step test along the slow axis).
  Several fast frames of the same area, summed or fitted together, average the blinking down;
- *they are dim.*  Hundreds to a few thousand photons per molecule, so the shared beam is learnt from
  many molecules, as in situ learns from many blinks: the data layout is unchanged, E molecules sharing
  the pupil parameters;
- *they may be fixed dipoles* (§3).

They have advantages too: no bead size, so they see the zero as it is; and a molecule is the sample
MINFLUX actually measures.  On the coverslip they share one height, which is what makes single planes
workable (§7a).

**The detector** is a photon counter: Poisson, s = 0, no EM gain (`noise.Poisson`).  What to add:
- *dead time:* measured ≈ μ/(1 + μ·τ/T) for a non-paralysable SPAD, τ ≈ 20–80 ns.  At the crest of a
  confocal scan of a bright bead it flattens the peak.  Unmodelled, that looks like a broader beam and a
  shallower zero;
- *afterpulsing:* a background proportional to the signal;
- *saturation of the dye* at the beam's crest: I/(1 + I/I_sat), already on the roadmap's physics list.
  It too changes the donut's shape, not just its height (Neupane & Wang 2020: nonlinear signals change the
  zero's order).

The robust first choice is fitting with the crest masked (`Batch.mask`), with dead time as a parameter
only when the data ask for it.

## 6. Phase masks: one pupil component for detection and excitation

The roadmap's design stands: the pupil phase is the mask's, placed on the pupil by a few fitted
parameters, plus Zernikes on top as the system's aberrations.  What this study adds:

**Analytic masks first (decided November 2026).**  Φ(u′, v′), evaluated at transformed pupil coordinates
(u′, v′) = S⁻¹R⁻¹((u, v) − c):
- the vortex, ℓφ;
- the top-hat, π·[ρ < r_th], with SimuFLUX's r_th = 1/√2 + 0.03;
- the half-moon, π·[u′ < 0], plus a phase δ that moves its zero (SimuFLUX: 1.4 nm/°);
- the double helix as Fresnel zones of increasing vortex charge;
- the tetrapod from its published Zernike sum.

Each mask names its parameters -- centre c, rotation, scale S, phase factor α, and its own (r_th, δ, ℓ) --
**held by default at their design values, each freed by a tick**.  Freed, a parameter is a diagnostic of
misalignment rather than a model choice: the fit reports its value with its CRLB and the likelihood it
gains, as `diagnostics/probes.py` reports a probe, so "the mask is 0.04 of the pupil radius off towards
+x, worth 300 nats" is a sentence the calibration can say.  The centre is the first to support, being the
commonest misalignment and the best determined (an odd quantity, which even a single plane sees once the
objective's even modes are held, §7a).  The scale stays held unless asked: it trades with defocus and
spherical aberration.

**User masks as images, later.**  A phase image with its scale relative to the pupil, stored with the model
file; resampled at the transformed coordinates, so that it gets the same centre/rotation/scale parameters.
Bilinear sampling of an image has derivatives with respect to the placement only within pixels, and
`grid_sample` must be checked on MPS before relying on it; supersampling as for edges (below) applies.

**Edges.**  A step's position must have a derivative:
- supersample each pupil sample s × s (s = 4), average exp(iαΦ) over the subsamples, and analytic masks
  stay exact;
- the vortex's singularity is integrable and needs nothing;
- edges need P ≥ 128 to place a step to ~1 % of the aperture.  The matrix DFT's cost grows with P, but
  only in the pupil-side product.

**A mask is not always a phase.**
- An SLM modulates only the polarisation along its liquid-crystal axis; the other passes flat.
- Its unmodulated zero order adds a flat field.
- A variable phase plate (Deguchi & Ries 2024: EOM, half-pupil SLM, polariser) and easySTED's segmented
  wave plate (Reuss et al. 2010) act differently on the two input polarisations.

These are a Jones pupil, 2 × 2, and they are the physical causes of most of a donut's ε (§7).  So the pupil
interface needs a complex form beside (magnitude, phase): `pupil(p)` keeps returning (magnitude, phase) for
phase-only masks, and a `JonesPupil` returns [2, 2, P, P] complex, which the focusing contracts with the
input Jones vector.  The variable phase plate is then one shared aberrated pupil, a fixed mask, and one
scalar per pattern (the EOM's retardance), which is learnable from the x, y and z scans together.  Its
misalignments are diagnostics in the same way, held by default: the compensating half-wave plate's angle,
the SLM's retardance error and its axis angle (Deguchi's notebook), and the zero-order fraction η.

**Start with detection.**  The mask component is useful without any of the excitation work: a tetrapod or
double helix on the widefield bead pipeline, where z stacks are routine.  It is the cheapest way to settle
the mask's gradients, sampling and identifiability (S1, §9).

**Measured on real double-helix beads** (November 2026, `benchmarks/uipsf/real/dh_beads.py`, results in
`benchmarks/uipsf/results/dh_beads.jsonl`).  Two plates, one bead each: Moerner's (Barentine et al. 2025, 61
planes at 50 nm, 120 nm pixels, objective not stated, NA 1.35 assumed) and DHPSFU's (Zhang et al. 2025, a
Double Helix Optics far-red plate, 1.27 NA water, 207 nm pixels, the field's most isolated bead, every other
plane).  Photons from each stack's own noise (the gains are not published), so the deviances compare models
on a set, not sets.  Mean deviance per pixel (1 at the noise):

| model | Moerner | DHPSFU |
| --- | --- | --- |
| Zernikes to order 8, flat start | 95.7 | 70.9 |
| Zernikes to order 12, flat start | 68.9 | 66.7 |
| Pavani double helix + Zernikes to 4 | 46.1 | 67.2 |
| Pavani double helix + Zernikes to 8 | **28.6** | **50.9** |

- *Prasad's zones are not a real plate's design*: their lobes came out broad and smeared.  Pavani & Piestun
  2008's design -- the phase of a sum of Gauss–Laguerre modes LG(k, 2k + 1) -- reproduces Moerner's bead by
  eye (lobes, separation, rotation over ±1.2 µm) and is now `DoubleHelix`'s default.
- *The mask as the base wins*: the design with Zernikes to order 8 beats free Zernikes to order 12 by 1.0 M
  nats (Moerner) and 0.14 M (DHPSFU).  Part of that may be the start: free Zernikes from a flat pupil can stop in
  a poor optimum, and the mask is also how a fit of an engineered PSF gets a good one.
- *Neither reaches the noise*: 29 and 51 per pixel, a blur of 87 and 120 nm taking up the rest.  The plates are
  optimised beyond Pavani's equal-weight design, DHPSFU's more (it turns over ±2 µm, longer than five modes
  give).  Next: the modes' weights as the design's own parameters, or the mask as an image.
- *The centre as a diagnostic*: stable on Moerner's plate (0.10 then 0.09 of the pupil radius off in x, with
  Zernikes to 4 and to 8), not on DHPSFU's (0.25, then 0.05): where the design is wrong, the centre takes up
  what the design misses, and its CRLB (10⁻⁴) says nothing about that.  A freed placement is a misalignment
  only once the design fits.

## 7. What trades with what, and how to separate it

The roadmap names the mask's trades: decentration vs coma, scale vs defocus and spherical, α vs ε.  For
beams, more trades:

- **ε has many sources.**
  - a polarisation impurity (E_z fills a vortex's zero);
  - the SLM's unmodulated polarisation or zero order;
  - asymmetric aberrations (astigmatism and coma fill it, Antonello et al. 2017);
  - the bead's size (a 100 nm bead fills a 2-D donut's zero; 20–40 nm beads are needed);
  - dead time, background, and the dye's own non-zero absorption at the zero.

  The roadmap's rule fits here: *fit the physical causes, keep a scalar incoherent floor ε only as a
  nuisance*, and report what share of the measured zero each cause explains.  SimuFLUX's
  `zerooffset` = 0.005 of the peak (CFR ≈ 0.3 at L = 80 nm for 0.01, MINFLUXexcitation) is the scale to
  expect.
- **The order of calibrations:**
  1. the widefield calibration of the same objective, or a scan with the mask out: the objective and the
     system;
  2. the mask in, those Zernikes held, only the mask's and the beam's parameters free.

  The roadmap's *keep its aberrations* tick for *start from* serves step 2.  With single planes (below)
  step 1 is more than convenient: it is what fixes the signs a single plane cannot see.

## 7a. Single planes: what one focal plane can and cannot determine

Often only one plane can be scanned.  A z stack is what breaks the **twin** in widefield beads
(`conventions.md`): the pupil A(k) and its twin A*(−k) give the same intensity at height −z, so in the
focal plane itself they are indistinguishable.  For a single plane at focus this predicts (scalar light,
matched indices), and the measurements below confirm for detection PSFs:

- **determined:** the odd Zernikes (m odd: tilt, coma, trefoil), which the twin leaves alone
  (c_j → −(−1)^m c_j); the magnitude of the even ones; ε; the beam's fill;
- **not determined:** the *sign* of every even phase mode (m even: defocus, astigmatism, spherical) --
  and with them the sign of a twin-symmetric mask's decentration, since the twin also takes a mask's
  centre c to −c;
- **the emitters' height** is one with defocus: on the coverslip all share one height, so it is one shared
  focus offset, and only its size is determined.

**What a mask does to this.**  The twin maps a mask Φ(k) to −Φ(−k).  A mask that maps to itself (up to a
constant) leaves the degeneracy in place; one that does not, breaks it, because its fixed form excludes the
twin from the model:
- *twin-symmetric:* the top-hat (π·[ρ < r] → −π·[ρ < r] ≡ the same mod 2π), the half-moon, the flat
  Gaussian beam;
- *not:* the vortex, whose charge ℓ flips to −ℓ.  A charge-1 vortex with astigmatism gives a ring tilted
  one way or the other depending on the astigmatism's sign relative to the charge, so a 2-D donut in focus
  should see astigmatism's sign.  Not spherical's or defocus's, which are rotationally symmetric.

Vectorial light: the twin also conjugates the polarisation (circular L ↔ R).  That changes nothing for
linear input.  For circular input it matters as soon as the handedness is known: a model with the λ/4
held cannot reach the twin, and the measurements below show that this alone breaks it for a Gaussian beam
(an expectation of this section that S3 corrected: it had said vectorial light adds little).

**Measured** (November 2026, `benchmarks/uipsf/synthetic/single_plane_twin.py`, results in
`benchmarks/uipsf/results/single_plane_twin.jsonl`).  The detection PSF -- vectorial, unpolarised, NA 1.45 --
with astigmatism (Noll 5, 6: 0.2, 0.3 rad), coma (7, 8), trefoil (9) and spherical (11: 0.12 rad), each mask
0.05/0.03 of the pupil radius off; 10⁴ photons per bead and plane, 10 background per pixel.  The fits are 20
beads in one plane at focus, the phase (order 4), the mask's centre and every bead's position, photons and
background free, started once from the truth and once from the twin (even modes flipped, centre at −c, heights
mirrored).  Δ is what the twin start's optimum is worse by, over the 20 beads:

| mask | oil, matched: Δ, nats | where the twin start ends | water: Δ, nats |
| --- | --- | --- | --- |
| none | 0 | at the twin | 141 (stays at the twin) |
| vortex | 0 | **back at the truth** | 0, back at the truth |
| top-hat | 22 | at the twin, centre at −c | 494 (stays) |
| half-moon | 0 | at the twin, centre at −c | 180 (stays) |
| double helix | 0 | **back at the truth** | 0, back at the truth |
| tetrapod (Zernike design) | 2341 | a poor local optimum | 1812 (poor local optimum) |

- *The twin is exact in focus without a mask and with the twin-symmetric masks:* with matched indices the
  top-hat and the half-moon end at the twin with the centre mirrored, Δ = 0–22 nats over 20 beads, about one
  nat per bead or less: the data cannot choose.
- *A vortex or a double helix breaks it outright:* started at the twin, the fit walks back to the truth -- the
  even modes' signs and the centre's direction come back.  The tetrapod design breaks it too (the expected
  images differ by ~470 nats per bead), but a fit started at the twin stops in a poor local optimum: a
  second start is needed to find the truth, not more data.
- *Index mismatch breaks it partly:* beads on the coverslip in water, through oil, prefer the truth by 7 nats
  per bead without a mask (25 with the top-hat), but a fit started at the twin stays near it.  So the sign is
  in the data but the optimiser does not find it alone: **fit from both starts and keep the better**, as in
  situ's twin test does.
- *With the even modes known, the centre's direction is determined* for the twin-symmetric masks too: the
  centre mirrored alone costs 39 (half-moon) to 130 (top-hat) nats per bead in the expected images.
- *Precision* (CRLB, the same 20 beads): one plane determines every mode to 3–6 mrad (11 for the double helix's
  astigmatism) and the centre to 0.002–0.004 of the pupil radius when the aberrations are of the size above, about twice the CRLB of three
  planes (−400, 0, +400 nm).  With the even modes five times weaker, one plane without a mask determines them
  4× worse than three planes (8 against 2 mrad), as their first-order effect on a focused image vanishes at
  zero; the vortex keeps them at 4 mrad.  So precision is not what a single plane lacks; it is the sign.

**Measured for scanned beams** (S3, November 2026, `benchmarks/uipsf/synthetic/scanned_single_plane.py`,
results in `benchmarks/uipsf/results/scanned_single_plane.jsonl`; the model is `model/scanned.py`).  The
excitation at 640 nm through NA 1.4 oil, SimuFLUX's fill, the same aberrations and mask offset as above; dyes
behind an open detector, scanned 33 × 33 at 25 nm, N = 3·10⁴ (about 10⁴ counts in a Gaussian's scan), one
background count per position.  KL is nats per emitter between the truth's expected scan and the
alternative's, in one plane at focus; the fits are 20 emitters, the phase (order 4), the mask's centre and the
emitters free, started from the truth and from the twin with the λ/4 held:

| beam | twin, handedness flipped | twin, λ/4 held | fit from the twin (oil): Δ, nats | water: Δ |
| --- | --- | --- | --- | --- |
| Gaussian, circular | 0 | 327 | 0, **back at the truth** | 0, back at the truth |
| vortex, circular | 2791 | 2865 | 0, back at the truth | 0, back at the truth |
| top-hat, circular | 0.9 | 25 | 208, a poor local optimum | 366, poor local optimum |
| half-moon, x along its line | 0 | 0 | 0, **at the twin** | 55, stays at the twin |

- *The exact twin includes the handedness*: with it flipped, a Gaussian, a top-hat and a half-moon give the
  same scan in focus (0–0.9 nats per emitter in oil; 8–29 in water, the mismatch breaking it weakly).
- *A known circular handedness breaks it.*  A model with the λ/4 held cannot flip it, and that alone costs the
  twin 327 nats per emitter for a Gaussian and 25 for a top-hat: from the twin the Gaussian's fit walks back to
  the truth.  The handedness is known when the vortex's zero is dark (a vortex is dark only with one of the two,
  below) or from the setup.  The top-hat's fit from the twin stops in a poor local optimum, as the tetrapod's
  did above: **fit from both starts**.
- *Linear polarisation keeps the twin*: the half-moon with x light ends at the twin, the even modes flipped and
  the centre at −c, at the truth's objective.  It needs the even modes held, a vortex or circular scan of the
  same path, or planes.
- *The vortex breaks it outright* (its charge cannot flip), as in widefield; but in one plane its centre
  trades with coma (CRLB 8·10⁻³ of the pupil radius against 2·10⁻³ in three planes; coma 14 against 4 mrad), so
  a fit lands near the truth, not on it.
- *Precision* (CRLB, 20 emitters): one plane determines every mode to 4–15 mrad and spherical to 14–22, about 2–4
  × the CRLB of three planes at −300, 0, +300 nm of known beam defocus (the SLM's diversity; on the coverslip
  the same as a stage).  Holding the even modes leaves the odd modes' CRLB unchanged except for the top-hat's
  coma (6–8 → 4 mrad).  A half-moon's centre along its step line is not determined at all, the mask being the
  same all along it: only the centre across the line (2·10⁻³) means anything.

**What fills a vortex's zero** (`--zero`, the minimum within 100 nm of the axis against the crest, one cause at
a time from a perfect beam; SimuFLUX's ε = 0.005 is the scale):

| cause | zero / crest |
| --- | --- |
| none, or spherical 0.1 rad (symmetric) | 0 |
| wrong handedness: dye, fluorescent bead, gold | 0.86, 0.76, 0.61 |
| λ/4 off by 5° (χ = 40°) | 0.0069 |
| 20, 40, 100 nm bead | 0.0029, 0.012, 0.073 |
| astigmatism 0.1 rad | 0.0047 |
| coma 0.1 rad | 0.0014 |
| mask 0.05 of the pupil radius off | 0.0086 |

So a 40 nm bead already fills the zero beyond SimuFLUX's ε: 20 nm beads are what the zero needs, or the bead's
size fitted.  The three samples differ through E_z's weight (`model.scanned.sample_weights`: gold 0.65 of the
in-plane components', a fluorescent bead's fixed dyes 0.85, a rotating dye 1).

**Ways to break it without moving the stage**, in order of preference:
0. **Circular light of known handedness** (measured above): free for a Gaussian beam and a vortex; a top-hat
   needs both starts.
1. **Hold the even modes from a widefield calibration of the objective** (step 1 above).  The twin is
   then not in the model, and the single plane learns the mask's parameters, the odd modes and
   corrections.  The diagnostic use (is the mask centred?) needs no more than this.
2. **Phase diversity from the beam path.**  If the beam can be refocused -- a deformable mirror, a
   varifocal lens, or *the SLM that writes the mask*, with a known defocus added to it -- a few planes at
   ±200–400 nm of known defocus act as a stack.  Diversity works only with phases that are not
   twin-symmetric; defocus is not (m = 0, c → −c).  Adding the vortex scan to the top-hat scans of the
   same beam path helps for the same reason.
3. **Two or three planes**, if the stage or piezo can take them at all: not a full stack, but enough to
   fix the signs.
4. **Emitters at different heights in one plane** -- beads in a gel, a tilted coverslip -- each sampling
   another defocus, as in situ does: the twin is then broken only by the index mismatch, which on the
   coverslip alone already gave 7 nats per bead (measured, above) but needs both starts.  Last resort.
5. **A vortex scan of the same beam path** (or any mask that is not twin-symmetric) fixes the even modes'
   signs for the other patterns that share the pupil: the measured vortex and double helix leave no
   ambiguity in one plane.

Otherwise the fit is reported with the even modes' signs as undetermined (the solver finds one of two
equally good optima; both are written out, as the twin test does in situ), and spherical aberration is
taken from the layered model with the known depth and indices rather than learnt.

**A focal plane is also where a top-hat is least informative.**  The 3-D donut's zero is a point; its
strength along z is what MINFLUX uses, and a single xy plane sees only the in-plane rise.  If z scans are
impossible, an xz *line* scan (the beam refocused along one line) may be: it is worth asking whether the
instrument can scan xz.

## 8. How it would fit into uipsf

Module by module, as a sketch for discussion:

- `optics/focus.py`:
  - the excitation fields, from `_vector_fields` with the forward Fresnel coefficients;
  - the input beam (fill, centre, Jones vector);
  - the scan-position DFT, separable on a raster.
- `model/mask.py`: `PhaseMask` (analytic and image masks, placed by centre, rotation, scale and α) and
  `JonesPupil`; composed with `ZernikePupil` / `MinimalPupil` through the builder (`model/build.py`), so
  that widefield models get masks too.
- `model/scanned.py`: `ScannedPSF(excitation, detection, pinhole, bead)` with the same interface as
  `VectorPSF`:
  - `specs`, `__call__`, `pupil_linearisation` with the product rule, `describe`;
  - `kind` "confocal" | "beam" (open detector: gold or fluorescent beads under a wide pinhole) | "ism".
- `model/geometry.py`:
  - `ScanGeometry`: the scanner's scale, rotation and bidirectional shift, and which of stage or beam
    moves in z;
  - for single molecules on the coverslip, one shared height (a focus offset) rather than one per emitter,
    as `CoverslipStackGeometry` does for widefield molecules; a factor per scan line (`model/intensity.py`).
- `noise.py`: dead time as an option of `Poisson`.
- `workflows/scan_beads.py`, for bead, gold and single-molecule scans, single planes or stacks.
- Export: an excitation model has no spline in SMAPpy's sense.  What SMAPpy would use:
  - a 3-D effective PSF sampled on a grid (a confocal or ISM deconvolution, a MINFLUX estimator);
  - the beam's parameters for SimuFLUX.

  The model file (`export/model_file.py`) needs a modality field.
- SMAPpy:
  - readers already exist for Abberior `.msr` images (`io/msr.py`) and MINFLUX records;
  - a *modality* switch in the PSF calibration window (widefield, confocal, MINFLUX beam), each with its
    sample list (`psf_methods.SAMPLES`).

**Cost.**
- A 3-D bead scan of 41 planes × 41 × 41 scan pixels is the same work as a widefield stack: 3 field
  components per input polarisation instead of 6 dipole fields, and a second DFT for the detection.
- The product breaks the "linear in |field|²" shortcut only through the product rule, so the Jacobian
  stays one wide DFT per tangent.
- A single plane is 1/41 of that; many single molecules (hundreds of small rasters) cost what in situ's
  blinks cost today.

**Platform.**  Nothing needs an operation CUDA, MPS and the CPU lack, if masks stay analytic (above).

## 9. Suggested milestones

Each one ends, as uipsf's do, in parameters recovered from simulated data to their CRLB, then real data.

| | what | data | depends on |
| --- | --- | --- | --- |
| S1 | analytic phase masks, detection side: tetrapod and double helix; centre, rotation, scale, α held, each freeable as a diagnostic | simulated z stacks; widefield beads with a mask | the builder |
| S2 | excitation focusing: fields, input beam, Jones pupil, normalisation; the oracle.  **Built** (`model/excitation.py`) | Leutenegger's library through SimuFLUX, as fixtures (`scripts/uipsf/make_oracles_focus.py`) | S1 for the masks |
| S3 | beam scans with an open detector: Gaussian, vortex, top-hat, half-moon; bead size; w_z; ε's causes.  **First: the single-plane study (§7a)** -- which parameters one plane determines, with and without held even modes and with SLM defocus diversity, by Fisher information on simulations.  **Model and single-plane study built** (November 2026, `model/scanned.py`, §7a); real scans next | simulated, then 20–40 nm fluorescent and 80 nm gold scans | S2 |
| S4 | scanned single molecules: one shared height, a factor per line, blinking and bleaching masked, fixed dipoles as an option | simulated, then molecules on glass | S3 |
| S5 | confocal: pinhole, product, λ_em, shared OPD Zernikes, scanner geometry; dead time later.  **Next, before S4** (C1-C4 below) | Abberior confocal bead scans (`.msr`) and a custom microscope without metadata | S3 |
| S6 | ISM: SPAD elements as channels on a lattice | a SPAD-array dataset (Vicidomini lab, BrightEyes) | S5 |
| later | user masks as images | | S1 |

**Confocal raster scans first (decided November 2026).**  The next application is confocal-like raster scans,
and its purpose is **characterising the beam** -- its aberrations, polarisation, fill and the pinhole's
alignment -- not localising molecules in raster images; that may stay the only use.  So S5's confocal model
comes before S4, and the output is a report of the beam, not a spline for SMAPpy's fitter.  The steps:

- **C1, the confocal model** (**built**, `model/confocal.py`; architecture.md): μ(r_s) = N·I_exc(r_e − r_s)·D(r_e − r_s) + b, D the emission PSF at λ_em (a
  `VectorPSF` intensity) convolved with the pinhole's disc back-projected into the sample (its size in Airy
  units of λ_em/NA, its offset learnable), descanned.  One pupil for both paths, Zernikes in nm of OPD turned
  into phase at each wavelength; path terms only if the data ask.  Checked in its limits (pinhole → 0:
  I_exc·I_em; pinhole → ∞: S3's open detector) and against SimuFLUX's confocal model with its E_z and its
  wavelength corrected.
- **C2, reading scans:** Abberior `.msr` images with their scan step from the metadata, and a custom microscope
  with **no metadata**, whose scan step (and wavelengths, NA, pinhole) are typed: every acquisition setting
  must be enterable by hand.  Beads found by SMAPpy's bead detection and cut out; one plane or an xyz stack;
  the bidirectional line shift as a parameter.  The detectors are APDs with dead times under 100 ns, run
  below saturation: **dead time is not modelled** for now (scanned-psfs.md §5 keeps it as a later option).
- **C3, simulations, then the lab's example data** (to come): recovered to the CRLB with §7a's rules -- the
  even modes held from a widefield calibration or two to three planes for linear light, one plane for
  circular light of known handedness.
- **C4, the report:** the beam's parameters with their CRLBs, the pupil, the scan against the model and the
  residuals, the pinhole's offset.  The PSF calibration window's *modality* choice comes with it, not before.
  **Later, a check of the descanning** (Jonas, November 2026): the pinhole's offset fitted per bead (an
  emitter-scope ``pinhole.offset`` beside the shared one) and plotted against where the bead is in the field.
  A perfect descan gives one offset everywhere; a constant offset is a misaligned pinhole; an offset growing
  with the distance from the scan's centre is a descan whose gain or rotation does not match the scan's.  It
  waits for data with beads spread over the field.

**S1 paused (November 2026).**  The mask machinery is built and tested (`model/mask.py`, the model file,
`start_from`, the twin study §7a), and fitted on real double-helix beads (§6), but it is not in the GUI.  The
analytic Pavani design with Zernikes on top stops at 29 (Moerner) and 43 (DHPSFU) deviance per pixel: real
plates are optimised designs, not the textbook mode sum, and DHPSFU's Zernikes run to ±210 rad at the rim,
which points to a clipped aperture.  It waits for better data -- beads of a plate whose design is known, or a
measured phase map -- and for the masks as images.  Open ends when it resumes: learnable mode weights of the
design, NA free on DHPSFU, the vortex beads of Hulleman 2021 (4TU did not answer for six hours), the
*phase mask* choice in the calibration window.  S2 does not wait for it: the masks are there for the
excitation side.

S1 is useful on its own and settles the mask machinery on data where z stacks are routine.  S3 is the first
scanned PSF and the first publishable result: no modern maximum-likelihood retrieval of a MINFLUX or STED
beam from bead scans exists (Kromann et al. 2012 used Gerchberg–Saxton on a point gold bead), and none that
works from single planes or from single molecules.

**Oracle and licences (decided November 2026).**
- uipsf is BSD-3.  SimuFLUX is the lab's and will be relicensed, so its own code -- the simulator, the
  pattern and pinhole handling (`psf_vectorial.py`), Deguchi's mask modifiers (`library/pishift.py`,
  `halfmoon.py`, written by him, "based on" Leutenegger's paper; `phaseramp.py`, the vortex, by Jonas Ries)
  -- can be ported.  Any other file of `library/` is checked for its author before porting.
- Leutenegger's library (LGPL 2.1; `effField`, `field`, `circular`, `interface`, ... in
  `SimuFLUX/python/psfs/library/` and `MINFLUXexcitation/PSF_simulation/library/`) is used as an oracle
  only: a script outside the package writes its fields as fixtures, as `make_oracles_v1.py` does for v1.
  Reading its algorithms for ideas is fine; uipsf's focusing is written from the equations (Richards & Wolf;
  Török et al. 1995; Leutenegger et al. 2006) on uipsf's own grid and DFT.

## 10. Open questions

1. Which data first: the lab's MINFLUX, an Abberior system, or a confocal / SPAD-array dataset?  And on
   each: can the beam be refocused (a deformable mirror, a varifocal lens, a defocus written on the SLM),
   or an xz line be scanned, when the stage cannot take a z stack (§7a)?
2. Gold or fluorescent beads for beams, or both?  Gold gives a cleaner zero and no bleaching, but needs the
   coherent model behind a pinhole.
3. Is the objective's widefield calibration available for the same instrument, to hold the detection
   pupil, the objective's transmission and -- for single planes -- the even aberrations' signs?
4. How far into STED: only the depletion beam's shape (the same as a donut), or the effective STED PSF,
   1/(1 + I_STED/I_sat) × excitation, which is nonlinear in the beam?
5. Shared OPD Zernikes for excitation and detection: worth making the default for two-colour widefield
   too?

## References

The DOIs were checked against Crossref.  Most of the content notes come from the abstracts.

- Richards & Wolf 1959, Proc R Soc A, [10.1098/rspa.1959.0200](https://doi.org/10.1098/rspa.1959.0200):
  vectorial focusing.
- Török et al. 1995, JOSA A, [10.1364/JOSAA.12.000325](https://doi.org/10.1364/JOSAA.12.000325): focusing
  through an index mismatch.
- Leutenegger et al. 2006, Opt Express, [10.1364/OE.14.011277](https://doi.org/10.1364/OE.14.011277): the
  chirp-z Debye integral in SimuFLUX and MINFLUXexcitation.
- Caprile, Masullo, Stefani 2022, Comput Phys Commun (PyFocus),
  [10.1016/j.cpc.2022.108315](https://doi.org/10.1016/j.cpc.2022.108315): a second oracle, with masks and
  layers.
- Kromann et al. 2012, Opt Lett, [10.1364/ol.37.001805](https://doi.org/10.1364/ol.37.001805): STED pupil
  by Gerchberg–Saxton from gold-bead scans, the nearest precedent.
- Antonello, Burke, Booth 2017, Opt Commun,
  [10.1016/j.optcom.2017.06.037](https://doi.org/10.1016/j.optcom.2017.06.037): how each aberration fills
  2-D and 3-D donuts.
- Patton et al. 2016, Opt Express, [10.1364/OE.24.008862](https://doi.org/10.1364/OE.24.008862):
  excitation and depletion paths aberrated separately.
- Balzarotti et al. 2017, Science, [10.1126/science.aak9913](https://doi.org/10.1126/science.aak9913);
  Gwosch et al. 2020, Nat Methods, [10.1038/s41592-019-0688-0](https://doi.org/10.1038/s41592-019-0688-0):
  MINFLUX, parabolic or Gaussian beam models with a scalar background.
- Schmidt et al. 2021, Nat Commun, [10.1038/s41467-021-21652-z](https://doi.org/10.1038/s41467-021-21652-z):
  MINFLUX on a common microscope.
- Deguchi & Ries 2024, Light Sci Appl,
  [10.1038/s41377-024-01487-1](https://doi.org/10.1038/s41377-024-01487-1): the variable phase plate.
- Stallinga, Wang, Rieger 2026, Opt Express,
  [10.1364/oe.590144](https://doi.org/10.1364/oe.590144): absorption-dipole and aberration effects on
  MINFLUX; a case for a vectorial beam model.
- Sheppard, Castello, Tortarolo, Deguchi 2020, JOSA A,
  [10.1364/JOSAA.37.000154](https://doi.org/10.1364/JOSAA.37.000154): ISM's reassignment from the Stokes
  shift and both PSFs.
- Castello et al. 2019, Nat Methods, [10.1038/s41592-018-0291-9](https://doi.org/10.1038/s41592-018-0291-9):
  SPAD-array ISM.
- Slenders et al. 2025, Light Sci Appl,
  [10.1038/s41377-025-01883-1](https://doi.org/10.1038/s41377-025-01883-1): ISM-FLUX.
- Reuss et al. 2010, Opt Express, [10.1364/OE.18.001049](https://doi.org/10.1364/OE.18.001049): easySTED,
  a polarisation-dependent mask.
- Shechtman et al. 2015, Nano Lett, [10.1021/acs.nanolett.5b01396](https://doi.org/10.1021/acs.nanolett.5b01396)
  (tetrapod); Pavani et al. 2009, PNAS, [10.1073/pnas.0900245106](https://doi.org/10.1073/pnas.0900245106)
  (double helix).
- Petrov et al. 2017, Opt Express, [10.1364/OE.25.007945](https://doi.org/10.1364/OE.25.007945); Ferdman et
  al. 2020, Opt Express (VIPR), [10.1364/OE.388248](https://doi.org/10.1364/OE.388248): pupils of engineered
  PSFs from data, pixel-wise or Zernike, with no mask parameters.
- Hulleman, Rieger, Stallinga 2021, Nat Commun,
  [10.1038/s41467-021-26228-5](https://doi.org/10.1038/s41467-021-26228-5): the vortex PSF, fitted
  vectorially.
- Neupane & Wang 2020, Eng Res Express,
  [10.1088/2631-8695/ab743b](https://doi.org/10.1088/2631-8695/ab743b): the measured zero's order depends on
  the signal's nonlinearity.
- Marin & Ries 2025, SimuFLUX (`ries-lab/SimuFLUX`): the ε model and the simulator.
