# DECODE, DECODE-Plex and 4Pi in SMAPpy: what exists, and a plan

Status, 2026-11-09: investigation only, nothing written; the published DECODE
first (§9).  This page is the design to start from.  It is written from five sources:
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

**The existing code is a baseline, not a reference.**  DECODE and DECODE-Plex
work well, but they still show pixelation artifacts, and nobody knows whether
their architecture, loss and training are optimal.  The 4Pi branch never
worked; it was an attempt.  So the new code need not reproduce any of them.
It is benchmarked against them and against the CRLB, and then tuned.  That
decides what is taken and what is written new (§5): modules with clean
interfaces, each of which can be swapped and benchmarked on its own.

**First the published DECODE, as a proof of concept (§9).**  Before the new
framework, the single-channel DECODE is ported into SMAPpy with its
algorithm unchanged: preparation, training, progress and inference, all
from SMAPpy.  That puts a working deep-learning fitter in users' hands
early, builds the parts the new fitter reuses (the job, the *Train* window,
the fitter, the benchmark), and gives the baseline it has to beat.  §3 to
§8 and §10 are the new fitter, a project of its own after that.

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
  Which one the published models used is to be checked (§11).
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

### What DECODE and DECODE-Plex leave open

Both work, and both are the baselines to beat.  Two things are not settled:
- **Pixelation.**  The localizations' sub-pixel positions are not uniform:
  they gather at places within the pixel, which shows as a grid in the
  rendered image.  The de-biasing after the fit (above) treats the symptom.
  Where it comes from is not known.  Candidates to test, none of them a
  finding: the per-pixel offsets near a pixel's edge, where two pixels can
  claim one molecule; the offset head's activation and range; how targets are
  assigned to pixels in training; the extraction's non-maximum suppression;
  aliasing in the network's downsampling and nearest-neighbour upsampling
  (§3.3).
- **Whether they are optimal.**  The architecture, the loss's weights, the
  ranges sampled in training and the training schedule were set once and
  never searched systematically.

### The colour codes (`DECODE_Internal`, `future/dev_prob_colour`)

**What it is.**  `codes` is one probability map per colour, with the photons
and positions shared.

**The limit.**  It cannot be combined with photons per channel
(`target_generator.py:116`).  The mixture loss sums the code maps away
(`p.sum(1)`, `loss.py:364`), so codes only reach the count loss.

### 4Pi (`DECODE_Internal`, branch `4pi-simulation`, Hao Sha, 2024-10 to 2025-08)

**Status.**  An attempt that never worked (Jonas, 2026-11).  It is 65 commits
ahead of `future/dev`.  Its README describes a DECODE-4Pi that beats GlobLoc
on simulated density sweeps and recovers the two rings of Nup96 at low
density; those are its author's claims and were not borne out.  The branch is
a source of parts and of ideas, not a result to reproduce.

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

What should not be carried over, and may be part of why it did not work:
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
   each channel's phase, and adds nothing, and so does the C++ fitter
   (`CSpline4Pi` in `csrc/models.hpp`: the channels share one phase with no
   offset).  One of the two is wrong for real data, most likely the branch.
   One bead stack fitted both ways settles it (§10, step 1).
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

### Other fitters built like DECODE (survey, 2026-11)

All output a probability per pixel with offsets and the other parameters, and
train on simulation.  The architecture numbers are from their code.  The Li
lab's code and SHOT's are GPL-3.0 and AGPL-3.0: ideas only, no code.

| fitter | what differs from DECODE | reported |
| --- | --- | --- |
| FD-DeepLoc (Fu 2023) | chip coordinates as input maps (CoordConv) into both U-Nets and the heads; a per-pixel occupancy loss (cross-entropy against the nearest pixel) added | field-dependent PSFs over ~180 µm |
| LiteLoc (Fei 2025) | half resolution, dilated convolutions (2 to 16); each frame encoded alone, joined at half resolution; 1.3 M vs 2.2 M parameters | 3.3× faster; 19 % lower lateral error with a Tetrapod, equal with astigmatism |
| LUNAR (Fu 2026) | ConvNeXt U-Net; attention along time per pixel, ±3 frames, temporal dropout; *SL* also learns Zernike aberrations from the data | 3D error over 9 frames 7.2 vs 11.1 nm (astigmatism), 10.3 vs 20.1 nm (Tetrapod) |
| SHOT (Séailles 2026) | offsets up to ±1.5 cells, so cells overlap; an optimal-transport loss (Sinkhorn) instead of the mixture, no non-maximum suppression; refined by rendering its estimate | E3D 0.75 vs 0.71 from the wider offsets alone; lower recall |
| AttentionUNet (Reinhard 2025) | attention over up to ~50 frames | mixed with density |
| SIMCODE (Cnossen 2025) | 6 intensities per pixel, then a fit of their modulation | the nearest thing to 4Pi; no deep-learning 4Pi fitter was found |

**On pixelation.**  LUNAR's extraction (a commit of 2025-02) forms a mixture
of the pixel and the neighbour its offset points to and takes its peak; the
commit reports far fewer grid artifacts, without retraining.  Its authors
say since (to Jonas, 2026-11) that it helps somewhat but does not remove
the pixelation: worth trying, not the whole answer.  No paper measures
sub-pixel bias in a controlled way.

**Image or volume outputs** (Deep-STORM, DeepSTORM3D, diffusion and flow
reconstructions) are limited by their grid and are not candidates.

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
| patterned illumination, SIMFLUX (an idea) | 1, a pattern cycle of frames | N per frame | -- (the pattern fitted after) | 1 |

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

**The phase from intensities instead (an idea, SIMCODE's route).**  SIMCODE
(Cnossen 2025) keeps the illumination pattern out of the network: it outputs
one photon count per pattern frame with its uncertainty (and whether the
molecule was on in that frame), and a small fit afterwards finds the position
from them, the pattern's phases and drift measured on the data.  For 4Pi the
same route is photons per channel (a link the spec has) and a fit of
I + A cos φ + B sin φ to the four counts.  It is not the same physics:
SIMFLUX's modulation scales the whole PSF, so the intensities hold all of it,
while in 4Pi A and B change each channel's shape with φ and z, which counts
throw away.  The direct phase head should therefore win; if it does not, the
network is not using the shape.  Not planned as a test.

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

**Why convolutional.**  The image is, to first order, the molecules convolved
with the PSF.  A shifted molecule gives a shifted image, and a CNN's output
shifts with its input, so that physics is built in.  It follows that:
- **the field of view must cover the PSF**: large kernels or dilations for
  large PSFs (a Tetrapod), where LiteLoc and LUNAR gained most;
- **a field-dependent PSF breaks the shift invariance** in a controlled way:
  chip coordinates as input maps, nothing more;
- **a CNN is equivariant only to shifts by its stride**: strides, pooling and
  nearest-neighbour upsampling alias, and sub-pixel position is what is
  measured.  Anti-aliased downsampling (BlurPool, Zhang 2019), smooth
  upsampling or a full-resolution path (HRNet, Sun 2019) are cheap tests
  against pixelation;
- **time is not a shift** (molecules do not move): attention or a recurrent
  cell (ConvGRU, which suits live fitting) over frames, convolutions in space.

**Candidates**, compared at matched parameters, FLOPs and training, with the
same heads, loss and extraction: DECODE's U-Net; a residual or ConvNeXt U-Net;
HRNet-style full resolution; a restoration network (NAFNet, Restormer); large
kernels or dilations.  Transformers over pixels (Swin, DETR-type set outputs,
Mamba scans) must learn the locality a CNN has, and are left last; in 3D
medical segmentation a tuned CNN U-Net beat most of them (Isensee 2024).

**Representation and decoding** come first, as in keypoint estimation, where
the sub-pixel bias was traced to decoding from a grid (DARK, Zhang 2020):
LUNAR's mixture peak (helps somewhat, its authors say), overlapping offsets
with an optimal-transport loss
(SHOT, keeping a σ per localization), coordinates as sub-pixel bins (SimCC,
Li 2022).

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
C++ `CSpline` and `CSpline4Pi` (§10, step 1).

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

**Extraction.**  DECODE's extraction (`to_emitter`, the lookahead) is the
first one, behind an interface, so another can replace it if the benchmarks
show it causes some of the pixelation.

**De-biasing.**  Its de-biasing becomes a plugin of its own after the fit:
histogram equalisation of the sub-pixel positions per z bin and uncertainty
group.  Then it can be judged on the table and applied to any fitter's
output.  It is a fallback: the aim is a network whose sub-pixel positions are
uniform without it, and the benchmark measures that (§10).

**Where it pixelates, measured (2026-11-11).**  On DECODE's example data, the
port's model trained on SMAP's spline pulls positions towards the pixel's
centre where the image says little.  The KS distance of the sub-pixel
positions from uniform is 0.018 within 150 nm of focus and 0.134 at 600 to
700 nm, and 0.038 below 2000 photons.  Density matters little (0.008 to
0.018), the training having been dense already.  Spline 3D on the same frames
and calibration stays at counting noise (0.001 to 0.006).  It is what has
been seen before: pixelation where the SNR is low, the density high or |z|
large.  So the redesign measures it this way on experimental data -- the
sub-pixel histograms in bins of |z|, photons, local density and predicted
error (`benchmarks/decode/pixelation/analyse.py`, `--by-condition`,
`--by-error`), against Spline 3D -- and tries output representations and
architectures that reduce it, before reaching for the de-biasing.  Whether a
better PSF model reduces it (the bead pupil model, the in situ one) is the
pixelation study of the proof of concept (`docs/decode/poc-plan.md` §8).

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

## 5. Taking code, or writing it new

The licence is Jonas's to decide (he is the author), so any of DECODE's code
can be taken.  Taking it is quicker; writing it new makes it easier to change
and tune.  Since the existing code is a baseline and not a reference (top of
the page), the balance is set piece by piece:

| piece | | why |
| --- | --- | --- |
| U-Net blocks | taken, as the first backbone | standard, not suspected; the backbone is an interface, so others can be tried |
| count loss | taken | small and understood |
| emitter samplers, the ROI's place on the chip | taken | small; the priors come from the parameter spec |
| extraction (`to_emitter`) | taken, as the first extractor | a suspect for pixelation, so behind an interface |
| de-biasing | taken, as a plugin after the fit | a fallback, judged on the table |
| 4Pi CRLB derivatives | read, as a check | SMAPpy's fitters already have the CRLB |
| heads and the mixture loss | new | driven by the parameter spec; the joint Normal × von Mises mixture does not exist |
| target assignment to pixels | new, with DECODE's as one option | a suspect for pixelation, to be benchmarked |
| renderer and simulation | new | torch, on SMAPpy's cameras, kinetics and calibrations (§3.5) |
| training loop, config, model file | new | headless, resumable, no Hydra or Lightning (§7) |

The DECODE package itself is not taken.  It is about 21 000 lines on Hydra,
Lightning 1.8, pydantic < 2 and the `spline` package, and much of it
duplicates SMAPpy:
- cameras;
- kinetics;
- the channel layout and registration;
- 4Pi models;
- colour assignment;
- phase to z.

**Modular, so it can be tuned.**  The backbone, the heads, the loss, the
target assignment, the extraction and the training schedule are each a part
with a setting, so one can be changed while the rest is held.  What is taken
goes into the new package, written to SMAPpy's house style.

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

## 9. First: the published DECODE in SMAPpy (proof of concept)

Status, 2026-11-09: agreed as the first step, not started.  The plan, its
milestones and what reading v0.10.2 found against this section:
[`decode/poc-plan.md`](decode/poc-plan.md).

**What it is.**  `TuragaLab/DECODE` v0.10.2 (2023, Speiser et al. 2021), one
channel, brought into SMAPpy as uiPSF was: a package of its own beside
`smappy`, changed wherever install, inputs or outputs need it, ported to
current Python and torch.  The method stays as published: the network
(`SigmaMUNet`), the loss, the simulation's statistics and the
post-processing (`SpatialIntegration`, `raw_th` 0.5).  Nothing of §3 goes
in; pixelation and the rest wait for the new fitter.  DECODE-Plex and 4Pi
are not part of it.

**Why first.**  The four things a user needs -- preparing a training,
starting it, watching it, fitting with the model -- are the same for any
network.  Built once on a method that works, they carry over to the new
fitter, which then only changes the network, the loss and the extraction.

### 9.1 The port

- **Where.**  `src/decode_smlm/`, a working name: `decode` is taken on PyPI
  and would shadow the original wherever both are installed.  torch is
  imported inside the functions that need it, as for uipsf; an extra
  `decode` installs it.
- **What it starts from.**  About 400 kB of Python in 77 modules, half of it
  `neuralfitter`.  What the proof of concept does not use is left behind:
  the notebooks, the example downloads, `plot`, `renderer`.
- **What changes.**
  - The `spline` package (compiled, from the `turagalab` conda channel)
    gives way to the torch renderer of §3.5.  The package is then pure
    pip and runs on CUDA, MPS and the CPU.
  - The calibration is read through `io.calibration.load_spline_calibration`:
    SMAPpy's HDF5 and SMAP's `_3dcal.mat` directly, and a uiPSF pupil model
    drawn at the sample's depth (`calibrate/depth.depth_spline`) -- what
    DECODE-Plex's users do by hand in uiPSF's demo, with a flip and a
    transpose.
  - The camera comes from SMAPpy's camera database, not typed.
  - torch 1.7's APIs move to current ones where they broke.
  - The job, the progress file and the model file of §9.3 to §9.5.
- **Licence.**  DECODE is GPL-3; Jonas expects to relicense it for the
  BSD wheel (2026-11).  Not a blocker for the work.
- **Accepted when** -- training is stochastic, so not bitwise:
  - the torch renderer draws the PSF `spline` draws, to float precision;
  - a model trained by the original v0.10.2 loads in the port and gives the
    same localizations on the same frames (inference is deterministic);
  - a model trained by the port scores as one trained by the original, same
    settings and calibration, within the spread between two runs of the
    original.
  The original runs in its own conda environment for this.  Its torch 1.7
  may not run on an RTX 40xx at all; then on an older GPU or the CPU.

### 9.2 Preparation

- **The PSF.**  Any calibration SMAPpy fits with (§9.1).  Checked once at
  the start of a job: the PSF as the training draws it against
  `evaluate_spline`, at a sub-pixel offset, which catches a flip or a half
  pixel.
- **The prefit,** a port of SMAP's *DECODE_training_estimates*.  Spline 3D
  fits a few hundred frames of the user's data (*focus* on *auto* for a
  pupil model), filtered on `logl_rel`, and the training's parameters are
  read off the table as SMAP does:

  | parameter | from |
  | --- | --- |
  | `lifetime_avg` | the mean blink length (`n_in_group`) less 1, at least 1 |
  | `intensity_mu_sig` | [1, 0.2] × mean photons / lifetime |
  | `bg_uniform` | the background's 1st to 98th percentiles, widened by 30 % |
  | `emitter_av` | the density, per 40 × 40 pixels |
  | `emitter_extent` z | the z range of the fits, within the calibration's |
  | `Camera` | the camera database: offset, gain, EM gain, pixel size, read noise |

  Every value stays editable.  Two figures check them: simulated frames
  against real ones in photons (DECODE's notebook does this by eye), and
  the prefit's photons, background and z with the distributions the
  training will draw overlaid.
- **The job folder.**  The output of preparation: DECODE's `param.yaml`,
  the calibration, and `job.json` (SMAPpy's version, the data it was
  prepared on, the ranges).  Everything training needs, so the folder can be
  copied to another machine.
- The functions are Qt-free in the package; the GUI is the first part of the
  *Train* window, on the open file.

### 9.3 Starting the training

- **One command, any machine:** `smappy-decode train FOLDER`.  Headless,
  nothing of Qt imported.  Installed by `pip install smappy-smlm[decode]` (or
  `uv tool install`): no conda, no compiled package beyond SMAPpy's wheels.
  CUDA on Linux and Windows, MPS on a Mac.
- **Here** the *Train* window starts it as a subprocess.
- **On a workstation** over the system's `ssh`, with a key (the host saved in
  Preferences): the job folder copied there, the job started under `nohup`,
  the model copied back.  No server and no daemon.  The workstation's
  SMAPpy version must be the job's; the job refuses otherwise.
- **Detached.**  Closing SMAPpy leaves the job running; the window lists the
  jobs it started and attaches to them again.
- **Stop, resume, warm start.**  A stop file is read after each epoch; the
  job resumes from `ckpt.pt` (DECODE's `InOut.checkpoint_init`) and starts
  warm from an earlier model (`InOut.model_init`).  That also lets a
  microscope PC yield its GPU to an acquisition (§7).

### 9.4 Progress

- The job appends a line per epoch to `progress.jsonl`: the training and
  test losses, DECODE's validation numbers on its fixed test set (Jaccard,
  lateral and axial RMSE, efficiency), the learning rate, the time per
  epoch.  Every few epochs a test frame with its truth and the predictions.
- The *Train* window follows the file -- here directly, on a workstation
  over `ssh` -- and draws the curves, the RMSE against the CRLB of the
  calibration at the training's photons, and the time left.  It has *Stop*.
- **Try on my data:** the latest model on a few real frames, its
  localizations over the prefit's Spline 3D fits.  On real data this says
  more than any loss.
- TensorBoard stays, for those who want it.

### 9.5 Inference in SMAPpy

- **A fitter,** *DECODE*, a `_FitPlugin` like the others on the channel
  layer: ROIs on the chip, every loader, live fits and *After the fit* come
  with it.  It runs in SMAPpy's process, on CUDA, MPS or the CPU.
- **The model file,** written by the job next to `ckpt.pt`: the weights and
  the parameters as JSON, no pickle -- the calibration's hash, the camera,
  the ranges (§4).  The fitter refuses a model whose calibration or camera
  differs from the data's, and warns where the data's photons or background
  lie outside the ranges.
- **What it does:** frames to photons through SMAPpy's camera, three frames
  at a time, the network, DECODE's post-processing.  The columns are
  SMAPpy's, as `io/decode.py` already names DECODE's: `x_nm`, `y_nm`,
  `z_nm`, `photons`, `background`, their errors, and the probability `prob`
  to filter on.  DECODE's `.h5` keeps loading through
  `io/decode.py`.
- **Checked on the benchmark (§10, step 2):** the same simulated frames
  fitted by Spline 3D and by DECODE, both against the truth.  This also
  catches x and y swapped or the pixel centre shifted, the likeliest
  mistakes of a port.
- A page for the fitter and one for the *Train* window, as for every plugin.

## 10. The new fitter: plan

**In SMAPpy's notation.**  The new fitter's code uses SMAPpy's conventions throughout -- x along
the columns, 0-based pixels with centres at whole numbers, calibrations as SMAPpy holds them,
SMAPpy's column names -- not DECODE's (x the rows, ``x0`` from 1, its own coefficient order).  The
port keeps DECODE's, behind one boundary (`docs/decode/poc-plan.md` §7).

Each step is accepted on numbers, as the tests do elsewhere.  Equivalence
with the existing code is not one of them: the new code has to match or beat
the baselines on the benchmark, not reproduce their outputs.

1. **Renderer.**  Torch splines and IAB, against the C++ models to float
   precision on CUDA, MPS and the CPU (this one is an equivalence: the
   simulation must be the PSF the fitters use).  The 4Pi phase convention is
   settled on one bead stack (§2, item 2).
2. **Benchmark.**  Before any network: fixed simulated sets over density, z,
   photons and background for each modality, drawn once and kept, plus the
   SMLM Challenge's 3D set and real data.  Each run reports detection
   (Jaccard, recall, precision), RMSE against the CRLB (Spline 3D's, Spline
   3D 2C's, Spline 4Pi's), the bias in z, **the sub-pixel histogram's
   uniformity, as the measure of pixelation**, and speed.  The baselines are
   run on it as they are, each in its own environment:
   - **DECODE-Plex** (`ries-lab/DECODE-Plex`, 2026-10), the main baseline and
     the main source of ideas, for C = 1 (its `conf.yaml`, `channels: 1`; not
     yet tried) and C = 2.  Its published weights where they fit (the Zenodo
     release, 10.5281/zenodo.20088009), and its localizations of our data in
     `~/datalocal/decode_plex`.
   - **The published DECODE** (`TuragaLab/DECODE`, v0.10.2, 2023), a second
     baseline for C = 1: the paper's method and its released models.
   - `DECODE_Internal` is not a baseline: its `future/dev` (2024-06) is older
     than DECODE-Plex.  It is read only for what DECODE-Plex lacks: the 4Pi
     attempt, the colour codes, LunarNet, the uiPSF loaders.

   **Built for the comparisons of step 4** (decided with Jonas, 2026-11-12): the benchmark and the
   fitter are modular from the start, so that a part can be switched and the result compared on
   the same sets -- the backbone (DECODE's U-Net, a ConvNeXt or residual U-Net, HRNet-style full
   resolution, NAFNet, large kernels or dilations; §3.3), the time model (none, per-pixel
   attention, a ConvGRU), the output representation and the extraction (DECODE's offsets, LUNAR's
   mixture peak, SHOT's overlapping offsets with an optimal-transport loss, SimCC's sub-pixel
   bins), the loss's terms and the training's schedule.  Each named in the run's settings and
   recorded with its results, the heads, loss and extraction otherwise unchanged; compared at
   matched parameters, compute and training.  The benchmark's report includes the Follow tab's
   per-epoch measures of today's training (`decode_smlm.training`): the error over each
   molecule's CRLB as a robust width, and the sub-pixel histogram (pixelation).

   **Built for C = 1** (2026-11-12): the sets, the report, a run's parts and how step 4 compares
   them are `docs/decode/benchmark.md` (code `benchmarks/decode/suite/`); the baselines' first
   results are in `docs/decode/poc-plan.md`, "Step 2, first round".
3. **C = 1.**  Network, joint loss, extraction.  At least DECODE's numbers on
   the benchmark.
4. **Tuning.**  One part at a time on C = 1, where it is cheapest.
   Pixelation first: LUNAR's mixture peak in the extraction (cheap, no
   retraining; helps somewhat), then anti-aliasing, the offsets' range and
   the loss (SHOT's), target assignment.  Then time
   (per-pixel attention against a ConvGRU), the backbones of §3.3, the loss's
   weights, the training ranges and schedule.  This needs the cluster or many
   nights on a 4090 (§8).
5. **C = 2.**  Biplane and colour.  At least DECODE-Plex's numbers on the
   benchmark.
6. **C = 4 with phase.**  No working baseline: against Spline 4Pi and its
   CRLB on the same simulations, then on Nup96.  Hao's branch is read for
   parts and ideas only.
7. **GUI.**  The fitter on the channel layer, the *Train* window, the
   de-biasing plugin, plugin pages.
8. **Extra input maps.**  Chip coordinates (field-dependent PSFs) and sCMOS
   variance.

## 11. Open questions

- **Shared or separate first-stage weights?**  It stays a setting until the
  authors say which the published models used.
- **Is DECODE Cloud alive,** and could its worker run this trainer?
- **Which machine trains** (§8).
- **Is LunarNet worth keeping** as a backbone?  It should be judged on the
  benchmark during tuning (§10, step 4).
- **Where does the pixelation come from?**  §2 lists the suspects; the
  benchmark tells them apart.

## References

- Speiser, A. et al. Deep learning enables fast and dense single-molecule
  localization with high accuracy.  *Nat. Methods* 18, 1082-1090 (2021).
  [doi:10.1038/s41592-021-01236-x](https://doi.org/10.1038/s41592-021-01236-x)
- Fei, Y. et al. Scalable and lightweight deep learning for efficient high
  accuracy single-molecule localization.  *Nat. Commun.* (2025) -- LiteLoc.
  [doi:10.1038/s41467-025-62662-5](https://doi.org/10.1038/s41467-025-62662-5)
- Fu, S. et al. Aberration-aware 3D localization microscopy via
  self-supervised neural-physics learning.  *Nat. Commun.* (2026) -- LUNAR.
  [doi:10.1038/s41467-026-73045-9](https://doi.org/10.1038/s41467-026-73045-9)
- Séailles et al. Optimal transport unlocks end-to-end learning for
  single-molecule localization.  ICLR (2026) -- SHOT.
  [arXiv:2512.10683](https://arxiv.org/abs/2512.10683)
- Reinhard et al. Improving single molecule localisation microscopy
  reconstruction by extending the temporal context.  bioRxiv (2025).
  [doi:10.1101/2025.04.05.647262](https://doi.org/10.1101/2025.04.05.647262)
- Cnossen, J. et al. Patterned illumination enables denser deep-learning
  based single-molecule localization microscopy.  bioRxiv (2025) -- SIMCODE.
  [doi:10.1101/2025.03.09.642203](https://doi.org/10.1101/2025.03.09.642203)
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
