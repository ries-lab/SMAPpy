# DECODE, DECODE-Plex and 4Pi in SMAPpy: what exists, and a plan

Status, 2026-11-09: investigation only, nothing written.  This page is the
design to start from.  It is written from five sources:
- the DECODE-Plex preprint (Sha et al. 2026) and its code,
  `ries-lab/DECODE-Plex`;
- DECODE's development tree, the private `ries-lab/DECODE_Internal`, read on
  `future/dev`, `future/dev_uiPSF_shah` (a local copy in
  `~/Downloads/DECODE_Internal-future-dev_uiPSF_shah`) and `4pi-simulation`;
- the published DECODE, `TuragaLab/DECODE` (Speiser et al. 2021);
- SMAPpy's and uipsf's own multi-channel and 4Pi code;
- DECODE-Plex's localizations of our dual-colour data, in
  `~/datalocal/decode_plex`.

The point is one framework for single-channel, multi-channel and 4Pi DECODE
inside SMAPpy, not a port of the DECODE package.  Where the existing code did
something that should change, the plan says what replaces it and why.

## 1. The method in one paragraph

**What the network does.**  DECODE localizes molecules at high density, where
their PSFs overlap and a single-emitter fit fails.  A U-Net reads a few
consecutive frames (three) and writes, for every pixel, the probability that
a molecule sits near it.  It also writes that molecule's offset from the
pixel centre in x, y and z, its photons and the background, and a standard
deviation for each.  The temporal context helps because a molecule is often
on in neighbouring frames.

**How it is trained.**  The network is trained only on simulated frames,
drawn afresh for every batch, so it never sees a frame twice.  The simulation
needs the microscope's PSF, a blinking model, the camera's noise and ranges
for the photons and background.  The loss is a likelihood.  The probabilities
form a mixture over the pixels, each pixel a Gaussian in its predicted
parameters, and the loss is the probability of the true molecules under that
mixture.  A count term matches the sum of the probabilities to the number of
molecules.

**What changes with a new PSF.**  A new PSF needs a new network.  It takes
about 4 h from scratch on an RTX 4090, and about 30 min started from an
earlier network (DECODE-Plex, Supplementary Figures 6 and 7).

## 2. What exists

### DECODE-Plex (published, `ries-lab/DECODE-Plex`, GPL-3.0)

**Input.**
- **Frames:** C channels of three frames each.  Mirrors and integer shifts are
  undone first, so the channels roughly overlap.
- **Offset maps:** one x and one y map per channel, the per-pixel offset of
  that channel from channel 0, computed from the affine registration.
- **Why the maps:** a trained model is valid for any ROI on the chip and any
  registration.  During training the ROI's place on the chip is sampled, so
  the network sees the offsets vary.

**Network** (`neuralfitter/models/model_param.py`, `DoubleMUnet`).
- A first U-Net reads each channel's frames.  In the v1 model this is one
  U-Net with shared weights (`unet_shared`) applied to each channel.  The paper
  says one U-Net per channel, and the v2 model has a `separate_ch` switch.
  Which one the published models used is to be checked (§10).
- A union U-Net reads their concatenated features.
- Heads write the outputs: probability, means, sigmas and background.

**Output** (`spec/model_out.py`, `ModelChannelMapGMM`).
- **Layout:** `[p × n_codes | photons × n_phot, x, y, z | their sigmas | bg × C]`.
- **Biplane:** one photon count, split by the calibration's ratio.
- **Colour:** one photon count per channel, sampled independently in
  training.  The colour is assigned after the fit from the ratio.

**Loss** (`neuralfitter/loss.py`, `GaussianMMLoss`).
- `Independent(Normal)` components over the `3 + n_phot` parameters.
- A count loss per code.
- A background MSE.

**Simulation.**
- **PSF:** cubic splines through the compiled `spline` package (C/CUDA, from a
  conda dev channel, not on PyPI).  It is not a torch op, does not run on MPS,
  and returns numpy arrays.
- **Rest:** EMCCD and sCMOS cameras, a uniform background per channel.

**Post-processing** (`processing/to_emitter.py`).
- 3×3 non-maximum suppression with a neighbour sum ("lookahead"), at
  thresholds 0.1 and 0.6.
- An optional de-biasing: the sub-pixel position histogram equalised per
  100 nm bin of z and per group of predicted uncertainty.

**Limits.**  Two channels are hard-coded in places: `setup_trafo`, the input
splitter (`case 2`), and the offset maps' single reference channel.  `cli/infer.py`
raises `NotImplementedError`, and training's entry names a config that is not
in the tree.

### The colour codes (`DECODE_Internal`, `future/dev_prob_colour`)

**What it is.**  `codes` is one probability map per colour, with the photons
and positions shared.

**The limit.**  It cannot be combined with photons per channel
(`target_generator.py:116`).  The mixture loss sums the code maps away
(`p.sum(1)`, `loss.py:364`), so codes only reach the count loss.

### 4Pi (`DECODE_Internal`, branch `4pi-simulation`, Hao Sha, 2024-10 to 2025-08)

**Status.**  65 commits ahead of `future/dev`.  Its README describes a
DECODE-4Pi that beats GlobLoc on simulated density sweeps and recovers the two
rings of Nup96 at low density.

What it adds:
- **PSF:** `CubicSplinePSF4Pi` (`simulation/psf_kernel.py:1494`), the IAB
  model per channel.  It reads uiPSF v1's `I_model` and the complex `A_model`
  as A = 2 Re, B = -2 Im (`io/psf.py`), with CRLB derivatives.
- **Simulation:** a phase on `FluorophoreSet` and the emitter sampler, and
  `Microscope4PiMultiChannel`.
- **Output and loss:** a phase output and its sigma (`activation_phase: tanh`,
  scaled to [0, 2π)), and a von Mises mixture loss
  (`loss.py`, `_vmmm_core_loss`).
- **Data and evaluation:** 4Pi frame readers and inference code, density
  sweep and CRLB notebooks.
- **Backbone:** an optional transformer, `LunarNet` (`models/LunarNet.py`, from
  the LUNAR / FD-DeepLoc line of work), not the default.

What should not be carried over:
1. **The phase is its own mixture.**  The loss is
   `3·(GMM(N, x, y, z), bg, 10·VMM(φ))`: two mixtures over the pixels, one for
   the position and photons and one for the phase.  A true molecule's phase
   can then be explained by another pixel than its position.  The right form
   is one mixture whose components are Normal(N, x, y, z) × von Mises(φ).  The
   hand-set weights 10 and 3 are a sign of the split.
2. **The channels' phase shifts may be counted twice.**  The kernel adds
   [0, π/2, π, 3π/2] per channel (`psf_kernel.py:1812`, `2183`) on top of the
   per-channel A and B.  SMAPpy's reader of the same files
   (`fourpi.load_uipsf_v1`) takes the per-channel A and B as already holding
   each channel's phase, and adds nothing.  One of the two is wrong for real
   data.  One bead stack fitted both ways settles it (§9, step 1).
3. **One background map for four channels** (`n_bg = n_phot = 1`,
   `tar_bg[:, 0]`).  The channels' backgrounds differ.
4. **The 4Pi forward model runs on the CPU in numpy** (three spline calls,
   then `np.multiply`).  Four channels are assumed, and the paths are
   `/home/shah/...`.

### Elsewhere

**`DECODE_Internal`.**  `future/multichannel` (2023, the start of DECODE-Plex),
`future/dev_onnx` (an export skeleton) and the uiPSF loaders (`245-uipsf`,
`247-uipsf`, merged into the 4Pi branch).

**DECODE Cloud.**  The `DECODE_Cloud_*` repos (a user API, a worker API, a job
fetcher, AWS infrastructure, some updated in autumn 2026) are a job service for
DECODE.  Whether it is alive decides whether it is a route for training (§8).

### What SMAPpy and uipsf already have

| need | where |
| --- | --- |
| channels as (n, C, h, w), mirrors and integer shifts undone, a camera ROI | `channels.ChannelSource`, `ChannelLayout`, the fitters' `open_source` |
| the registration between channels | `ChannelLayout` maps, `calibrate.dual`, `FourPiCalibration.maps` |
| multi-channel and 4Pi PSF models | spline calibrations, `fourpi.FourPiCalibration` (per-channel I, A, B), uipsf's models in torch on CUDA, MPS or CPU |
| field-dependent PSFs | `field_splines`, uipsf `model/field.py` |
| blinking, cameras, sCMOS noise maps | `simulate.kinetics`, `camera_db`, `scmos`, `uipsf/movie.py` |
| colour from the photons in two channels | *Assign colours*, its probabilistic mode (`docs/dual_color_assignment.md`, which compares it with DECODE-Plex's) |
| z from the phase, following the cavity's drift | *4Pi z*, `fourpi.py` |
| reference fitters and their CRLB | Spline 3D, Spline 3D 2C, Spline 4Pi |
| scoring against a simulation | *Ground truth* |
| reading DECODE's files | `io/decode.py` |

## 3. One framework

Every variant so far is DECODE with a flag: `codes` or `channels`, `is_4pi`,
`is_4pi_single_phot`.  The flags cannot be combined.  What they all differ in
is the molecule's parameter vector.  So **the model declares its parameters,
and everything else follows from that list**: the heads, the loss, the
targets, the sampling in training, the post-processing and the table's
columns.

| | channels | photons | extra | detection maps |
| --- | --- | --- | --- | --- |
| DECODE | 1 | N | -- | 1 |
| biplane, multiplane | C | N, split by the calibration | -- | 1 |
| ratiometric colour | C | N_c per channel, free | -- | 1 |
| colour by PSF or codes | 1 or C | N | -- | 1 per colour |
| 4Pi | 4 | N, through I, A, B | φ | 1 |
| 4Pi in colour (later) | 4 | per colour | φ | 1 per colour |

### 3.1 The parameter spec

Each parameter has four properties.

**A kind**, which fixes its head's activation and its component of the
mixture:
- *offset*: x, y, z; tanh; Normal.
- *scale*: photons, background; sigmoid times its range; Normal.
- *circular*: φ; predicted as (cos, sin) and read as an angle; von Mises with
  concentration 1/σ².

**A prior for training.**  This is the range the simulation samples
uniformly.

**A link to the channels**, for photons:
- one N split by fixed shares (biplane, or 4Pi's I);
- one N per channel (colour);
- one N per colour mixed by the spectral matrix (DECODE's
  `MultiChoricSplitter`).

**A column.**  The output uses SMAPpy's vocabulary, so the plugins after the
fit work unchanged:
- `x_nm`, `y_nm`, `z_nm` (`z_env_nm` for 4Pi), `photons`;
- `photons_ch0`, ... and `photons_err_ch0`, ...;
- `phase_rad`, `phase_err_rad`;
- the `*_err_*` columns, `background`, `prob`.

**The phase prior.**  The phase is sampled uniformly and independently of z,
as DECODE-Plex samples the colours' photons independently.  The network then
learns no relation between phase and z.  The cavity's drift does not matter at
training, and *4Pi z* unwraps afterwards as it does for Spline 4Pi.

### 3.2 Inputs

**Frames.**  C channels × T frames, from the channel layer.

**Extra input maps.**  A list rather than one hard-coded pair:
- **Registration:** the offset of each channel from channel 0, x and y,
  2(C-1) maps.  As in DECODE-Plex.
- **Place on the chip:** x and y in chip coordinates, for a field-dependent
  PSF.  This is FD-DeepLoc's approach (Fu et al. 2023).
- **Camera noise:** the variance per pixel, for an sCMOS with a noise map.

Each map the network reads makes one trained model valid for more ROIs,
cameras and places on the chip.  Training samples whatever the maps describe:
the ROI's place, the noise map's crop.

### 3.3 Network

DECODE-Plex's architecture, parametrised by C, T, the extra maps and the
output spec.  C = 1 with no extra maps is DECODE.  Two choices stay settings:
- per-channel or shared weights in the first stage (`separate_ch`);
- the backbone (the U-Net now; LunarNet if it earns its place).

### 3.4 Loss

**One mixture over the pixels.**  Each component is the product over the
parameters of its kind's distribution: Normal for offsets and scales, von
Mises for circular ones.  It is written out by hand
(log-sum-exp over the pixels of the summed log densities), not through
`MixtureSameFamily(Independent(Normal))`, which cannot mix kinds.

**The other terms.**
- The count loss is per detection map, as now.
- The background loss is per channel.

**Codes.**  The mixture runs over (pixel × code) pairs, so a code can carry
its own photons.  That is what `codes` cannot do now.

### 3.5 Simulation

**Renderer.**  Splines and IAB rendered in plain torch: a batched gather of
the 64 coefficients of each voxel the molecule touches, then the product with
the polynomial in the sub-voxel offsets.  It runs on CUDA, MPS and the CPU,
and has no conda dependency.  It is checked pixel by pixel against SMAPpy's
C++ `CSpline` and `CSpline4Pi` (§9, step 1).

**Speed.**  Measured on an M1 Max (2026-11) for one batch: 256 frames of
64 × 64, 25 spots each, 13 × 13 spline spots, background, Poisson and read
noise to uint16.  The C++ is `CSpline::value` from `csrc/models.hpp` (value
only), with the same accumulation and noise, `-O3`; DECODE's `spline` package
has only an Intel macOS build and was not timed.

| | 1 channel | 4Pi: 4 channels, I + A cos φ + B sin φ |
| --- | --- | --- |
| numpy, `smappy.simulate` now | 0.33 s | – |
| C++, 1 thread | 0.115 s | 0.85 s |
| C++, 10 threads | 0.018 s | 0.12 s |
| torch, CPU | 0.094 s | 0.68 s |
| torch, MPS | 0.013 s | 0.13 s |

Torch on the GPU is as fast as the C++ on every core, and matches
`io.calibration.evaluate_spline` to float precision.  A DECODE batch is
smaller than this one, so drawing it costs milliseconds; training's cost is
the network.  CUDA (a 4090, torch against DECODE's kernel) is not yet
measured.

**Everything else from SMAPpy:**
- kinetics (`simulate.kinetics`);
- the camera models with sCMOS maps (`camera_db`, `scmos`);
- a background per channel;
- the registration from the calibration;
- the ROI's place on the chip, sampled as DECODE-Plex does.

### 3.6 After the network

**Extraction.**  DECODE's extraction (`to_emitter`, the lookahead) is kept.

**De-biasing.**  Its de-biasing becomes a plugin of its own after the fit:
histogram equalisation of the sub-pixel positions per z bin and uncertainty
group.  Then it can be judged on the table and applied to any fitter's
output.

**Colour and z.**  Colour is *Assign colours*, and z from the phase is
*4Pi z*.

## 4. In SMAPpy

**A torch package beside uipsf.**  Working name `src/deeploc/`.  It is
imported only inside the functions that need it, as uipsf is.  It reuses
uipsf's device handling (`uipsf.device`) and reads its models.

**Inference is a fitter** (`_FitPlugin`) on the channel layer, so it gets these
for free:
- ROIs on the chip;
- DCIMG and quadrant files;
- live fits and *After the fit*.

**Training** gets a window (*Train*) and a CLI.  The window writes the job
and starts it locally or hands it to another machine.

**The model file** records everything inference must check:
- the parameter spec;
- the calibration's hash;
- the channel layout;
- the camera and its readout;
- the training ranges.

The fitter refuses a model whose calibration or layout differs from the
data's.  It warns when the data's photons or background lie outside the
ranges it was trained on.

## 5. What is taken from DECODE

The licence is Jonas's to decide (he is the author), so code can be taken
directly where it helps:
- the network blocks;
- the loss's count term;
- the extraction;
- the samplers;
- the 4Pi CRLB derivatives, as a check.

The DECODE package itself is not taken.  It is about 21 000 lines on Hydra,
Lightning 1.8, pydantic < 2 and the `spline` package, and much of it
duplicates SMAPpy:
- cameras;
- kinetics;
- the channel layout and registration;
- 4Pi models;
- colour assignment;
- phase to z.

What is taken goes into the new package, written to SMAPpy's house style.

## 6. Where the trained models come from

**Pretrained base models, then warm starts.**  A base model is trained once
per microscope and modality.  A user's calibration then starts from it, which
is the 30 minutes rather than the 4 hours.

The model file names its base model, so a warm start is traceable.  Base
models are trained over wide ranges.  DECODE-Plex found a wide range costs
little (its Supplementary Figure 9).

## 7. The training job

Training is a **headless job**, so it runs the same on every machine in §8.

**Input:**
- a settings file;
- the calibration (PSF, registration, camera);
- the ranges;
- optionally a base model.

**Output:** one model file, with checkpoints along the way.

**It runs anywhere:**
- a CLI, nothing from Qt or the GUI imported;
- CUDA, MPS or the CPU;
- resumable from a checkpoint, for a cluster's time limits and for a
  microscope PC that is needed for an acquisition;
- logs for TensorBoard;
- a validation set drawn once and kept, so curves are comparable.

**Pause and resume.**  On the microscope PC, training must yield to an
acquisition.  The job stops at a checkpoint when asked and resumes later.

## 8. Where to train (to be decided)

| machine | for |
| --- | --- |
| the microscope PCs' RTX 4090 | a user's warm starts and trainings, at night or between acquisitions; inference during the measurement needs the same GPU |
| a workstation | the same, without the conflict |
| an H100 cluster | base models, studies, benchmarks: many jobs in parallel. A small U-Net on 40 × 40 patches does not fill an H100; a guess, not measured, is 1.5-3 × a 4090 per job unless the batch grows |
| the Mac (M1 Max, MPS) | tests and development, small warm starts within the memory cap |
| DECODE Cloud, if alive | users without a GPU |

## 9. Plan

Each step is accepted on numbers, as the tests do elsewhere.

1. **Renderer.**  Torch splines and IAB, against the C++ models to float
   precision on CUDA, MPS and the CPU.  The 4Pi phase convention is settled
   on one bead stack (§2, item 2).
2. **C = 1.**  Network, joint loss, extraction.  It must reach DECODE's
   numbers on the SMLM Challenge's 3D set and the CRLB at low density
   (against Spline 3D's).
3. **C = 2.**  Biplane and colour.  It must reach DECODE-Plex's published
   numbers (Figure 1C, Supplementary Figures 4 and 5) and its localizations of
   our data in `~/datalocal/decode_plex`.  The Zenodo release
   (10.5281/zenodo.20088009) holds weights, PSFs and data.
4. **C = 4 with phase.**  Against Hao's branch and against Spline 4Pi and its
   CRLB on the same simulations, then on Nup96.
5. **GUI.**  The fitter on the channel layer, the *Train* window, the
   de-biasing plugin, plugin pages.
6. **Extra input maps.**  Chip coordinates (field-dependent PSFs) and sCMOS
   variance.

## 10. Open questions

- **Shared or separate first-stage weights?**  It stays a setting until the
  authors say which the published models used.
- **Is DECODE Cloud alive,** and could its worker run this trainer?
- **Which machine trains** (§8).
- **Is LunarNet worth keeping** as a backbone?  It should be judged on step
  3's numbers.

## References

- Speiser, A. et al. Deep learning enables fast and dense single-molecule
  localization with high accuracy.  *Nat. Methods* 18, 1082-1090 (2021).
  [doi:10.1038/s41592-021-01236-x](https://doi.org/10.1038/s41592-021-01236-x)
- Sha, H. et al. Multi-channel high-density single-molecule localization.
  bioRxiv (2026).
  [doi:10.64898/2026.07.24.740199](https://doi.org/10.64898/2026.07.24.740199)
- Li, Y. et al. Global fitting for high-accuracy multi-channel single-molecule
  localization.  *Nat. Commun.* 13, 3133 (2022) -- GlobLoc, the reference
  fitter.
- Liu, S. et al. Universal inverse modeling of point spread functions for
  SMLM localization and microscope characterization.  *Nat. Methods* 21,
  1082-1093 (2024) -- uiPSF, whose models are simulated.
- Fu, S. et al. Field-dependent deep learning enables high-throughput
  whole-cell 3D super-resolution imaging.  *Nat. Methods* (2023).
  [doi:10.1038/s41592-023-01775-5](https://doi.org/10.1038/s41592-023-01775-5)
  -- chip coordinates as an input.
