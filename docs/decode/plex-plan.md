# DECODE-Plex in SMAPpy: the port, and step 5's plan

Status, 2026-11-19: planned and agreed with Jonas; the port's inference done -- the paper's three
published models tested (Fig. 1e, 2c, 2a) give the original's localizations here (§9); SMAP's
two-channel calibrations read (§10); the C = 2 baseline on 22 sets, which reproduces the paper's CRLB
demonstration (§12; §11 is the first round, superseded).  Next: the study (§7).

Decided (Jonas, 2026-11-19):
- **The licence.**  DECODE-Plex is his group's code: GPL-3, as it is published, is fine.  It goes
  where `decode_smlm` is, under the same notice.
- **The order** was left to this plan: port first, then a study of the C = 2 architecture (§7),
  then netfit's C = 2.  The study needs the baseline to compare with, and the published weights make
  the baseline cheap.
- **The benchmark's C = 2 data are the paper's own**: the live-cell biplane movie (Fig. 2c) and one
  dual-colour data set (Fig. 2a, which comes with a small test movie), each with its calibration,
  its registration and its published model.

## 1. What DECODE-Plex is, as read

**Not v0.10.2 with channels added.**  It is DECODE 1.0.0dev0 -- `DECODE_Internal`'s `future/dev`
rewrite -- with the multi-channel work on top: 165 modules, 21 000 lines, on PyTorch Lightning 1.8,
Hydra 1.2 / omegaconf 2.2, pydantic 1.10, structlog, and the compiled `spline` 0.11.1dev0.  It
already runs on torch 2.1.2 and Python 3.10.  Of what `decode_smlm` ported, only the U-Net
(`unet_param.py`) is the same file.  So the port is a second tree, not a patch on the first, and
`decode/__init__.py` imports every subpackage, so the import closure of inference alone is 152 of
the 165 modules.

**The input.**  C channels of a window of frames, and 2C *offset maps*: per pixel, x_c - x_0 and
y_c - y_0, channel c's position of that pixel less channel 0's, from the registration (an affine map
from SMAP's `_trafo.mat`, or uiPSF's shift), divided by 20.  Mirrors and integer shifts between the
halves of the chip are undone before (`dual_splitter_factory`), so the channels overlap to a few
pixels; the maps hold the rest.  In training the ROI's place on the chip is drawn anew
(`sampled.offset: auto`), so one model holds for any ROI.

**The network** (`SigmaMUNetv2`, built as `AggregateModel`), as every published model has it:
- `separate_ch`: one U-Net per channel; `separate_win`: each frame through it alone; `aux_shared`
  and `aux_union`: the offset maps appended to every frame's input and to the union's;
- a union U-Net reads all C x window x 64 features and the maps;
- `bigger`: depth 3 and 3, 64 features, ELU, strided convolutions, nearest upsampling --
  **22.8 M parameters** (2 x 7.5 M shared, 7.75 M union), against netfit v1f's 2.2 M;
- heads: the probability (one per *code*), the means `[photons x n_phot, x, y, z]`, their sigmas,
  a background per channel.

**Biplane and colour differ only in the photons:**
- biplane: `n_phot = 1`, one count, split [1, 1] to the planes by the photon transform, the PSF not
  renormalised, so the calibration's own ratio holds;
- colour: `n_phot = 2`, a count per channel drawn independently, over wide ranges (Fig. 2a: 200 --
  20 000 and 1000 -- 100 000), so the network learns no colour prior; the colour is assigned
  afterwards from the ratio (`notebook/3_color_assignment.ipynb`).

**The loss** is v0.10.2's `GaussianMMLoss` with the mixture over `3 + n_phot` parameters, a count
term per code and a background term per channel.

**Training:** AdamW, lr 1e-4, weight decay 0.1, ReduceLROnPlateau (0.5, patience 20), batch 16,
gradients clipped at 0.03 (norm), 10 000 samples an epoch, 600 -- 1000 epochs (6 -- 10 M samples),
40 emitters per 40 x 40 pixels, targets of at least 100 photons (summed over channels).

**Post-processing:** `SpatialIntegration`, raw_th 0.1, split at 0.6, as v0.10.2; an optional
de-biasing (`de_bias.py`) of the sub-pixel histogram.

**Limits:** two channels hard-coded in `setup_trafo`; `cli/infer.py` not implemented; the notebooks
are the interface.

## 2. Where the code goes

**`src/decode_smlm/plex/`**, a subpackage of the C = 1 port:
- one extra (`smappy-smlm[decode]`), one licence notice, one lint configuration, one test folder's
  conventions -- what the two have in common is everything around the code;
- a subpackage, not a merge: the trees diverged, and merging them would make every change to one a
  risk to the other's reproduction of its original;
- imported only where a DECODE-Plex model is used, as `decode_smlm` itself is (a test checks that
  `import decode_smlm` and `import smappy` leave it out);
- the Hydra `config/` folder comes along as data (`decode_smlm/plex/config/`): training composes
  from it, and a run's `param_run.yaml` is resolved from it.

Tests in `tests/decode_tests/plex/`, with `collect_ignore` without torch as for the rest.

## 3. What is copied verbatim, and what is adapted

**Verbatim, one commit:** `decode/` whole (without `__pycache__`), as `src/decode_smlm/plex/`, and
`config/` beside it, at 6138472.  Not copied: `cli/` (its `infer` raises `NotImplementedError`),
the notebooks, `setup.py`, `environment.yaml`.  `NOTICE.md` gains a paragraph: where the files come
from, the commit, the authors, the licence.

**Then one commit each:**
1. **Imports.**  `decode.` becomes `decode_smlm.plex.`, the absolute `import decode.x` included.
2. **The package's `__init__`** made lazy, as `decode_smlm`'s: DECODE-Plex's imports everything,
   torch, Lightning and plotting with it.
3. **Dependencies the inference does not need, behind functions or shims** -- the path from a
   checkpoint and frames to localizations should need torch, numpy, scipy, h5py and yaml only:
   - `structlog` (10 modules, `generic/logging.py`): a small shim over `logging` that takes its
     keyword arguments;
   - `deprecated` (22 modules): a shim decorator;
   - `pytorch_lightning` (14 modules): the checkpoint is read with `torch.load` and its
     `state_dict`'s `_model.` prefix stripped, no Lightning on the inference path; the Lightning
     module stays for training (§4);
   - `omegaconf` and Hydra: `param_run.yaml` is already resolved, so it is read as YAML;
   - `pydantic` 1 (11 modules): `pydantic.v1` where pydantic 2 is installed, or the models made
     dataclasses if they are few on the path;
   - `spline`: the renderer is `decode_smlm`'s torch spline (`simulation/spline_torch.py`), swapped
     in for training; inference does not render;
   - `seaborn`, `torchmetrics`, `dotted_dict`, `sklearn`, `skimage`: imported where used.
4. **The API breaks of `poc-plan.md` §3 that apply** (a CPU tensor indexed on the GPU, numpy on
   tensors, MPS), each found by a test, one commit each.
5. **The boundary to SMAPpy**, as `decode_smlm`'s `predict.py`: DECODE's x is the rows, SMAPpy's
   the columns; the calibration and the registration come in through SMAPpy's loaders, the
   localizations go out as SMAPpy's columns.  Kept in one module, `plex/predict.py`, so that the
   port keeps DECODE-Plex's conventions inside, as `decode_smlm` keeps v0.10.2's.

## 4. Training, later in this step

Inference first: the published models are the baseline on the paper's data, and that needs no
training.  Training is ported when the baseline has to be trained on something else (our own
two-colour beads, `~/datalocal/260709_MM_BeadCal_2C`) or when the study needs DECODE-Plex retrained
at matched compute.  Then the Lightning loop gives way to `decode_smlm.training`'s job runner (the
job folder, `progress.jsonl`, stop and resume), as v0.10.2's `live_engine` did, with DECODE-Plex's
sampler, simulation and loss behind it.

## 5. The reference outputs

The original runs on **decodebox**, in `~/miniforge3/envs/decode_plex`: a clone of `decode_dev`
(DECODE_Internal, torch 2.1.2, Lightning 1.8.6, pydantic 1.10, structlog, spline 0.11.2.dev0),
which differs from DECODE-Plex's `environment.yaml` only in minor versions (Hydra 1.3, omegaconf
2.3, seaborn 0.10, tifffile 2023, spline 0.11.2 for 0.11.1 -- the spline is not on the inference
path).  The versions are recorded with the outputs.  A script, `scripts/decode/reference_plex.py`,
imports only DECODE-Plex and writes:

1. **The architecture oracle**, in git (`tests/decode_tests/fixtures/plex_*.npz`, under 1 MB): a
   small `AggregateModel` (depth 1, 8 features; colour and biplane) with seeded weights, its output
   on fixed inputs, then `SpatialIntegration` on that output -- the network's wiring, the offset
   maps' placement, the activations and the post-processing, without a 90 MB weight file.
2. **The offset maps** from each published registration (`trafo_aux_factory`) for a fixed crop, and
   the preprocessing's output on a few raw frames (mirrors, shifts, gain, crop): the geometry,
   where a port is likeliest to be wrong by a pixel or a sign.
3. **The model oracles**, in `~/.cache/decode_smlm/plex/` (too large for git), each the published
   checkpoint on the first 200 frames of its movie through `infer_func.infer`: the raw output of
   the first 20 frames and all localizations.
   - **Fig. 1e**, the SMLM Challenge's biplane `MT0.N2.HD` with its truth: no registration (one
     model, both planes on one grid) -- the first to pass;
   - **Fig. 2c**, the live-cell biplane, a registration and a mirror, window 1;
   - **Fig. 2a**, dual colour on the small test movie, a registration, window 3, two photon counts.
   The paper's own localizations of the full Fig. 2c and Fig. 2b movies (`results.zip`) are a second
   check, of the whole movie, by hand.

**Accepted when:** the oracle's output equals the port's to float precision on the CPU and within
1e-5 on MPS and CUDA; the localizations the same, x and y within 1e-4 px and z within 0.05 nm; and
on Fig. 1e the Jaccard and RMSE against the challenge's truth are the original's.

## 6. The benchmark's C = 2 sets

As `benchmarks/decode/suite/sets.py` draws the C = 1 sets: drawn once, kept, recorded in
`manifest.json` with seed, size and hash, each emitter's CRLB kept with its truth.  A context per
published model, so its weights fit and no training is needed for the baseline:

| context | calibration and registration | model | window | photons | background (a pixel) |
|---|---|---|---|---|---|
| `biplane_2c` | Fig. 1c's bifocal bead calibration (`Fig1c_2c-..._3dcal.mat`, `_trafo.mat`) | Fig. 2c, live cell | 1 | 500 -- 30 000 | 0 -- 300 each |
| `colour_2a` | Fig. 2a's two-colour bead calibration (`Fig2a-..._3dcal.mat`, `_trafo.mat`) | Fig. 2a, actin / microtubules | 3 | 200 -- 20 000 and 1000 -- 100 000 per channel | 0 -- 100, 0 -- 500 |

Both on the paper's camera (EMCCD, gain 100, 5 e-/ADU, read noise 74.4 e-, 127 x 117 nm pixels),
z in +-600 nm (+-1000 for the biplane, as its training).

**Simulated, two sweeps each, as for C = 1:** density 1, 5, 15, 25, 50 per 40 x 40 px at the
training's photons; photons at three levels at density 1.  Drawn by `smappy.simulate` as a split
chip, the two halves through the calibration's registration and mirror -- the frames a real movie
gives, so that each fitter's own preprocessing is tested.
- **Biplane:** one photon count per emitter, each plane drawn with its own spline.
- **Colour:** two dyes, each emitter one of them, its photons split by the dye's ratio between the
  channels (the ratios from the paper's data: the median of each population on the photon scatter
  of Fig. 2a); the dye is kept with the truth, so the report adds the colour assignment's accuracy.

**Each emitter's CRLB** from Spline 3D 2C's model -- both channels, one emitter -- on the spot alone,
doubled in variance for the EM register.

**Real data, fixed frame ranges:** Fig. 2c's movie and Fig. 2a's test movie, kept on decodebox and
in the cache; the paper's localizations beside them.

**The reference fitter** is Spline 3D 2C, with the same calibration and registration.

**First, a reader** (checked 2026-11-19: SMAPpy reads none).  The paper's calibrations are SMAP's
global two-channel files: `SXY(2)`, a spline per channel (Fig. 1c: 26 x 26 x 198, 20 nm steps,
`z0` 100, `x0` 14, the chip split at row 245, `EMon` 1), with `normf`, `mirror` and `global`, beside
`parameters1/2/_g` and `SXY_g`; the `_trafo.mat` holds `trafo_raw` and `trafo_inv_raw`, each an
affine `ref` / `tar` pair in micrometres, the mirror inside it.  SMAPpy's dual calibration
(`calibrate.dual.load_dual_color_calibration`) is its own HDF5, so a converter comes first -- the two
splines in SMAPpy's orientation, the map as a projective `transformation` in its pixel convention,
the geometry -- tested against DECODE-Plex's offset maps of the same registration, which the model
oracles record as the network's input (§9).

## 7. After the port: the study, then netfit's C = 2

Before netfit's C = 2 is fixed, its architecture is studied on these sets, at matched compute and
training, three seeds a variant, against DECODE-Plex's numbers (Jonas, 2026-11-19).  The questions:

1. **Where the time model goes.**  netfit's ConvGRU (GMN7C) in each channel's network, after the
   channels are joined, or in both.  After the join it sees a blink in both channels at once,
   which should matter most for colour.
2. **Two networks and a union, or one.**  The alternative to DECODE-Plex's late fusion: channel 1
   resampled onto channel 0's grid by the registration, then one network over all channels from the
   first layer.  No offset maps; the cost is the interpolation, which correlates the noise of
   neighbouring pixels.
3. **Where the offset maps go:** every stage, as DECODE-Plex, or the union alone.
4. **Size.**  DECODE-Plex's network is ten times v1f's.  Screen B compared backbones at v1f's size
   only, so capacity is open at C = 1 too, and is tried there first, where it is cheapest.
   *Answered at C = 1 (`poc-plan.md`, "Network size at C = 1"): depth adds nothing, 64 features
   gain about 0.01 in Jaccard for 23 % in speed, DECODE-Plex's size nothing beyond that.*
5. **The photons for colour:** a count per channel (DECODE-Plex), or a total and a fraction, which
   gives the colour a head of its own.

What netfit can take from DECODE-Plex without the study: its input (the offset maps, the ROI's place
drawn over the chip), and the maps as the mechanism for chip coordinates in step 8.

## 8. Data

- **Zenodo record 21348704** (CC-BY-4.0).  Archives are read by range requests, so a single file is
  fetched without the whole archive: `outputs.zip` (2.4 GB; per model a 275 MB `.ckpt` with the
  optimizer's state, `param_run.yaml`, an empty `training.log`, `val_cache.pkl`), `calibration.zip`
  (4.6 GB), `results.zip` (12.7 GB), the movies (Fig. 2c 2.8 GB, Fig. 2a test 3.3 GB, Fig. 1e 41 MB).
- **Kept** in `~/.cache/decode_smlm/plex/` here (checkpoints, calibrations, the first frames of each
  movie: about 1.5 GB) and under `~/decode_plex/` on decodebox (the movies whole).
- `~/datalocal/decode_plex ` -- its name ends in a space -- holds DECODE-Plex's localizations of our
  own two-colour movies (2025-09), not the paper's.

## 9. As done (2026-11-19)

**The port** (`src/decode_smlm/plex/`), in commits of one kind each, as `decode_smlm` was: the tree
verbatim (ed938712); `decode.` renamed; the package's `__init__` and every subpackage's made lazy
(`plex/_lazy.py`); structlog and Deprecated replaced by the standard library; pydantic 1 by
`generic/validated.py` (fields with coercion, `validator`, `root_validator`, `construct`,
`validate_model`, `validate_arguments`, `StrictInt`); Lightning, Hydra, omegaconf, dotted_dict and
the compiled spline imported where used, and a run's YAML read under OmegaConf's two loader rules;
the checkpoint read without Lightning (`io/model.load_state_dict`, `weights_only`, the one
`pathlib.PosixPath` read as `PurePosixPath`); ruff's safe fixes, then the rest by hand.  Reading a
model and inferring imports torch, numpy, scipy, h5py and yaml only (a test checks).  No API break
of torch 2.14 or numpy 2.5 showed on the inference path.

**The oracles** (`scripts/decode/reference_plex.py`), from DECODE-Plex at 6138472 in
`~/miniforge3/envs/decode_plex` on decodebox -- a clone of `decode_dev` plus `dotted-dict`: torch
2.1.2, Lightning 1.8.6, pydantic 1.10.17, omegaconf 2.3, spline 0.11.2.dev0 (the versions are
recorded in each oracle's JSON).  The architecture fixture (0.7 MB, in git) is bitwise equal over
two runs.  The movies' first 300 frames come from range requests into the archives
(`reference_plex.py head`), so 250 MB of each is fetched, not 2.8 and 3.3 GB.

**The published models, 200 frames each, the port on MPS against the original on the CPU:**

| run | channels, window | localizations | found by both | max x, y (px) | max z (nm) | photons (rel.) |
|---|---|---|---|---|---|---|
| Fig. 1e, challenge biplane, no registration | 2 planes, 3 | 2 913 | 2 913 | 3.8e-6 | 0.0004 | 1.7e-6 |
| Fig. 2c, live-cell biplane, registration and mirror | 2 planes, 1 | 339 169 | 339 168 | 3.1e-5 | 0.034 | 4.1e-5 |
| Fig. 2a, two colours, registration | 2 colours, 3 | 146 466 | 146 465 | 2.1e-5 | 0.043 | 4.0e-5 |

The network's input is equal to 3e-6 (the offset maps, float32 on both sides) and its output to
1.2e-4.  The one localization each run leaves unpaired is a spot the post-processing put on a
neighbouring pixel (probabilities 0.27 -- 0.39), not one near the threshold.  The port on the CPU
matches as closely (Fig. 1e: x, y within 3.8e-6 px, z within 0.002 nm); MPS is ten times faster
(Fig. 1e: 4.4 s against 46 s for 200 frames).

**The original's own GPU is the looser reference.**  Its 4090 run differs from its CPU run by up to
0.09 in the raw output and by 20 of 339 149 localizations (Fig. 2c; Fig. 2a 7 of 146 466, Fig. 1e
1 of 2 913): torch uses TF32 for convolutions on Ampere and later by default.  The port on MPS
reproduces the CPU's counts exactly, so the CPU is the reference, and a comparison of fits made on
NVIDIA GPUs should allow for this.

**Tests** (`tests/decode_tests/plex/`): the wiring and the post-processing against the fixture, the
imports, the YAML loader, the pydantic replacement (fast); the three models on the CPU and on a GPU
(slow, from the cache, skipped without it).  The live-cell model finds 1 700 localizations a frame
at the notebook's threshold (0.1): most are at low probability, which the benchmark's filter, not
the port, has to deal with.

**Not done yet:** training (§4); uiPSF's `.h5` calibrations (Fig. 2b), which need omegaconf and
dotted_dict at the call; the multi-GPU path (`DataParallel`), untested; the de-biasing
(`de_bias.py`), not on the notebooks' path.

## 10. The reader and the C = 2 sets, as done (2026-11-19)

**The reader**, `smappy.calibrate.smap_dual.load_smap_dual_calibration` (tests in
`tests/test_smap_dual.py`).  Each `SXY` entry goes through SMAPpy's single reader, now one function
for one entry (`io.calibration.smap_spline`); the global calibration's `coeff` is each channel's raw
spline (`coeffrawref` / `coeffrawtar`), so the secondary stays as the camera saw it, as SMAPpy keeps
it.  The splines hold each channel's brightness relative to the main one (SMAP's `normf`: their sums
at focus differ by 0.78 for Fig. 1c, 4.6 for Fig. 2a).  The registration is `trafo_raw.tar`
(secondary to main), read as MATLAB applies it, on pixels from 1 in units of 1000, in the bead image
as SMAP loaded it (un-mirrored in x for EM gain); the reader takes it onto the chip's pixels from 0 in
the camera's orientation through the beads' camera ROI (`parameters_g.roi`).  How this was settled:
- on Fig. 2a (no EM) it agrees with DECODE-Plex's loader, given the same matrix, to 0.01 px (float32
  at its scale) once the division is left out as DECODE-Plex leaves it;
- the movies' own shift between the channels agrees with it: Fig. 2a's movie lies on the registration
  but for 5.3 rows, Fig. 2c's (EM-mirrored, as SMAP's EM coordinates need) but for 3.4 rows, which is
  how far each splitter moved since its beads: DECODE-Plex's runs say so too, shifting the movie's
  second channel by their `Test` offset where their maps take the `Simulation` one (Fig. 2a: -29
  against -35 rows; Fig. 2c: 43 against 47).  Its maps and its shifted frames agree within a pixel.

**DECODE-Plex treats SMAP's projective registration as affine** (`XYZTransformationMatrix` never
divides by the third coordinate).  On Fig. 2a's beads (226 px wide) that moves a point by up to
0.07 px; on Fig. 1c's (512 px) by up to 0.29 px, 37 nm, over the area sampled.  Its offset maps carry
that error, in training and in inference alike, so a model learns around it only where the error is
the same; the study (§7) and netfit take the registration with its division.

**The sets**, `benchmarks/decode/suite/sets_c2.py` (`$DECODE_SMLM_DATA/benchmark/v1/sets_c2/`, the
manifest `manifest_c2.json`), each context's converted calibration kept beside them as `_2ct.h5`:
- `biplane_2c`: Fig. 1c's bifocal beads, frames of 512 x 512 as that EMCCD records them, the Fig. 2c
  model's context (one photon count, drawn into both planes, each with its share), z +-1000 nm;
- `colour_2a`: Fig. 2a's two-colour beads, 512 x 226, its model's context; two dyes at 78 % and 92 %
  of their light in channel 1 -- the peak and the shoulder of log10(ch1 / ch0) in DECODE-Plex's own
  fit of the Fig. 2a movie (0.55 and 1.05), an estimate to replace by the dyes' known ratios; each
  channel's spline normalised at focus, as the model's `norm: auto`; z +-600 nm;
- each a density sweep (1, 5, 15, 25, 50 per 40 x 40 pixels of the main half) at the model's training
  photons and background, and a photon sweep (1000, 3000, 10 000 at density 1, each channel's
  background 2 % of its mean photons); 20 000 molecules a set, placed in the main half where the
  partner falls inside the secondary one;
- the camera the paper's (EMCCD, gain 100, 5 e-/ADU, read noise 74.4 e-), pixels 117 x 127 nm;
- each molecule's CRLB from both channels together (`crlb`): each channel's spot's Fisher matrix,
  the secondary's position derivatives through the registration's Jacobian, summed over x, y, z and
  the photons (one count, or two for colour) with a background per channel, doubled for the EM
  register.  At 10 000 photons the median is 7 -- 8 nm laterally and 15 nm in z (C = 1: 9 nm).

**The published models on the sets** (`plex_on_set.py`; the trial, 2000 molecules at 10 000 photons,
the second channel shifted by the registration's own offset): they find 96 % (biplane) and 99.8 %
(colour) of the molecules where both channels see the sample, so the reader, the drawing and
DECODE-Plex's conventions agree on where a molecule is.  Near the focus they are at the bound (colour,
|z| < 240 nm: x and y scattered by 6 -- 13 nm against a CRLB of 7); away from it they are not (at
z = -600 nm, y biased by 47 nm, scattered by 59), and z is biased by up to +-20 nm at +-360 nm.  Two
things the benchmark's scoring has to know:
- **score where both channels see the sample.**  Where a molecule's partner falls off the chip
  (Fig. 1c: the main half's rows below 52) DECODE-Plex pads its second channel with zeros, and the
  network reports spots at the edge it sees there;
- **the biplane model reports a band of false localizations** (rows 73 -- 103 of the main half, at
  probability about 0.6; 2447 against 1783 molecules on the trial) inside that area.  Not yet
  understood: it may be the training's sampled ROI positions never covering that part of the chip,
  or something the real movie has and the simulation lacks.

## 11. The baselines on the C = 2 sets, first round (2026-11-19; superseded by §12)

`benchmarks/decode/suite/run_c2.py` (`plex`, `spline`, `table`): DECODE-Plex's published model of each
context through its notebook's steps, on MPS (raw threshold 0.1, kept at prob >= 0.5), and SMAPpy's
Spline 3D 2C with the converted calibration written ROI-local (photons linked for biplane, free for
colour; kept within 3 MADs of the median log-likelihood and below the iteration limit), both scored
by `report.score_truth` where both channels see the sample, the network reads, and a 13-pixel fit box
stays inside both halves (7 pixels from their edges: 76 % of the molecules nearer an edge are not fitted
by Spline 3D 2C).  Frames read: DECODE-Plex 1 (biplane) and 3 (colour), Spline 3D 2C 1.  The reports
are `reports/{plex,spline}_c2.json`.

| set | fitter | frames | Jaccard | RMSE lat / ax (nm) | \|z\| < 300: lat / ax | its error / CRLB, \|z\| < 300 (robust) |
|---|---|---|---|---|---|---|
| biplane density 1 | DECODE-Plex | 1 | 0.50 | 48 / 61 | 24 / 55 | 2.8 / 2.0 |
| | Spline 3D 2C | 1 | 0.64 | 22 / 29 | 9 / 20 | 1.07 / 1.09 |
| biplane density 15 | DECODE-Plex | 1 | 0.69 | 56 / 80 | 29 / 73 | 3.2 / 2.7 |
| | Spline 3D 2C | 1 | 0.41 | 74 / 89 | 47 / 72 | 6.1 / 3.2 |
| biplane density 50 | DECODE-Plex | 1 | 0.65 | 67 / 107 | 36 / 96 | 4.7 / 5.4 |
| | Spline 3D 2C | 1 | 0.02 | 70 / 130 | 66 / 114 | 21 / 13 |
| biplane 10 000 photons | DECODE-Plex | 1 | 0.43 | 51 / 49 | 21 / 27 | 2.5 / 1.8 |
| | Spline 3D 2C | 1 | 0.79 | 26 / 31 | 6 / 12 | 1.09 / 1.04 |
| colour density 1 | DECODE-Plex | 3 | 0.27 | 47 / 43 | 24 / 34 | 1.9 / 1.9 |
| | Spline 3D 2C | 1 | 0.69 | 18 / 23 | 11 / 18 | 1.14 / 1.07 |
| colour density 15 | DECODE-Plex | 3 | 0.69 | 58 / 66 | 37 / 56 | 2.8 / 2.9 |
| | Spline 3D 2C | 1 | 0.30 | 84 / 130 | 78 / 131 | 9.0 / 9.8 |
| colour density 50 | DECODE-Plex | 3 | 0.70 | 78 / 104 | 59 / 97 | 5.6 / 6.3 |
| | Spline 3D 2C | 1 | 0.01 | 102 / 167 | -- | -- |
| colour 3000 photons | DECODE-Plex | 3 | 0.98 | 44 / 48 | 22 / 37 | 1.2 / 1.2 |
| | Spline 3D 2C | 1 | 0.76 | 35 / 45 | 25 / 39 | 1.04 / 1.10 |
| colour 10 000 photons | DECODE-Plex | 3 | 0.84 | 42 / 33 | 17 / 24 | 1.4 / 1.3 |
| | Spline 3D 2C | 1 | 0.72 | 20 / 22 | 12 / 18 | 1.09 / 1.08 |

All 16 sets are in the reports (`run_c2.py table`).  Speed on the Mac: DECODE-Plex 100 -- 320 ms a frame of
512 x 512 or 512 x 226 (MPS), Spline 3D 2C 3 -- 90 ms (CPU).

**What it says, with two findings open:**
- **Near the focus Spline 3D 2C is at the bound** (1.04 -- 1.14 of the CRLB) while the molecules are
  isolated, and collapses with density (Jaccard 0.41 / 0.30 at 15, 0.02 / 0.01 at 50), as Spline 3D does
  for one channel.  DECODE-Plex holds its detection at density (Jaccard 0.65 -- 0.73 from 15 to 50) but
  never reaches the bound: 1.2 -- 2.9 times it near the focus at low density, 5 -- 6 at density 50.
- **Spline 3D 2C fits about 15 % of the molecules twice** (density 1, both contexts), most of them far
  from focus, where the two channels' spots differ enough that the detection does not pair them, and
  misses about 7 % of those well inside both halves even at 10 000 photons.  Its Jaccard is understated
  by both: a property of its detection on these PSFs, worth fixing in SMAPpy, not a limit of the fit.
- **DECODE-Plex's false positives grow with the background, inside its training range**: on
  `colour_2a_density_1`, 7 a frame where channel 0's background is 0 -- 25 photons a pixel, 174 where it
  is 75 -- 100 (its training drew 0 -- 100); precision 0.26 there against 0.99 at the photon sweep's low
  background.  Not understood: a difference between these frames and its training's (the camera's noise,
  the background's form) is as likely as the model's own behaviour, and the Jaccards of the density sweep
  depend on it, so they are not yet the baseline's.  The biplane model's band of false localizations
  (§10) is part of the same question.

## 12. The baseline reflects DECODE-Plex's CRLB demonstration (2026-11-19)

Jonas pointed out that DECODE-Plex was shown to reach the two-channel CRLB on single emitters (the
paper's Fig. S4, `notebook/crlb.ipynb`), which §11's numbers did not show.  The demonstration uses its
own two models (*FigS4-biplane-crlb*, *FigS4-dual_color-crlb*), trained with a perfect camera on single
frames, photons 2000 -- 8000, background 0 -- 40; the colour one with channel 0's spline in both
channels (`channels_out [0, 0]`).  Reproducing it found three differences between §11's sets and the
world DECODE-Plex's models were trained in, each settled on DECODE-Plex's own code:

1. **DECODE-Plex reads SMAP's projective registration as affine** (`XYZTransformationMatrix`: the row
   vector times `trafo_inv_raw`, the third element dropped).  On Fig. 1c's 512-pixel chip that differs
   from SMAP's own by up to 0.29 px, in training as in inference.  The decisive test: the S4 biplane
   model on isolated molecules at 5000 photons, channel 1 placed by SMAP's map against DECODE-Plex's --
   RMSE / CRLB 3.1 -- 6.9 (x, y) with biases up to 29 nm, against **1.02 -- 1.27 with no bias**.  So the
   sets now place channel 1 as DECODE-Plex reads the registration (`smap_dual`, `reading="decode-plex"`).
   On real data the projective map is the microscope's: what the affine reading costs DECODE-Plex there,
   where the two differ, is a finding for netfit, which takes the registration with its division.
2. **SMAP keeps the mirrored channel's spline un-mirrored along the split** (`cspline.mirror` 2).
   DECODE-Plex's own splines, rendered on decodebox for each run's settings (`plex_psf.py`) and compared
   with SMAPpy's under every flip, show it: the two models of real movies (Figs. 2a, 2c), whose settings
   flip channel 1 along the split, see the camera only if the converter flips it back -- which it now
   does (`smap_dual`; this also fixes Spline 3D 2C on real data read through it).  The S4 models were
   trained differently -- biplane with channel 1 not flipped, colour with channel 0's spline mirrored --
   and their sets are drawn so (`sets_c2.CONTEXTS[...]["channel_1"]`).  The splines agree with
   DECODE-Plex's to 1 % near focus and 3 -- 8 % in shape at z = -600 nm, where SMAPpy's renderer cuts the
   spot to the spline's size less 3 pixels (DECODE-Plex draws it whole); position, z and brightness agree.
3. **The S4 colour model was trained on a 256 x 256 sensor** (its run's `sensor_size`): it misses 94 --
   100 % of the molecules beyond it.  The scoring keeps to the sensor each model was trained on
   (`run_c2.trained_sensor`); the other models' is 512 x 512.

**The baseline** (`run_c2.py table`, all 22 sets; Spline 3D 2C fitted with the same splines and map):

| set | fitter | frames | Jaccard | \|z\| < 300 nm: RMSE lat / ax (nm) | RMSE / CRLB lat / ax |
|---|---|---|---|---|---|
| biplane S4, 2000 photons | DECODE-Plex | 1 | 0.95 | 8.1 / 17.4 | 1.08 / 1.10 |
| | Spline 3D 2C | 1 | 0.96 | 7.6 / 16.1 | 1.06 / 1.13 |
| biplane S4, 5000 photons | DECODE-Plex | 1 | 0.95 | 4.7 / 9.6 | 1.12 / 1.13 |
| | Spline 3D 2C | 1 | 0.94 | 4.3 / 8.9 | 1.09 / 1.14 |
| biplane S4, 8000 photons | DECODE-Plex | 1 | 0.93 | 3.9 / 8.1 | 1.23 / 1.26 |
| | Spline 3D 2C | 1 | 0.89 | 3.3 / 6.7 | 1.13 / 1.15 |
| colour S4, 2000 photons | DECODE-Plex | 1 | 0.93 | 14.7 / 24.2 | 1.04 / 1.02 |
| | Spline 3D 2C | 1 | 0.80 | 14.5 / 24.0 | 1.14 / 1.27 |
| colour S4, 5000 photons | DECODE-Plex | 1 | 0.93 | 7.7 / 12.0 | 1.05 / 1.02 |
| | Spline 3D 2C | 1 | 0.77 | 7.5 / 11.9 | 1.06 / 1.18 |
| colour S4, 8000 photons | DECODE-Plex | 1 | 0.89 | 6.0 / 10.5 | 1.08 / 1.27 |
| | Spline 3D 2C | 1 | 0.78 | 5.5 / 8.5 | 1.07 / 1.19 |
| biplane Fig. 2c, density 1 | DECODE-Plex | 1 | 0.56 | 15.3 / 50.6 | 1.61 / 2.34 |
| | Spline 3D 2C | 1 | 0.65 | 8.3 / 18.5 | 1.48 / 1.53 |
| biplane Fig. 2c, density 25 | DECODE-Plex | 1 | 0.81 | 22.3 / 62.7 | 2.95 / 3.74 |
| | Spline 3D 2C | 1 | 0.27 | 56.4 / 87.3 | 11.6 / 10.8 |
| colour Fig. 2a, 10 000 photons | DECODE-Plex | 3 | 0.92 | 11.5 / 18.8 | 1.37 / 1.37 |
| | Spline 3D 2C | 1 | 0.72 | 12.1 / 18.6 | 1.46 / 1.32 |
| colour Fig. 2a, density 25 | DECODE-Plex | 3 | 0.82 | 33.3 / 57.1 | 3.75 / 4.07 |
| | Spline 3D 2C | 1 | 0.09 | 91.2 / 150.3 | 17.5 / 19.1 |

RMSE / CRLB is over all z; near the focus (its robust width, `crlb width ... |z| < 300` in the table) the
S4 models are at 1.02 -- 1.22 and Fig. 2's at 1.17 -- 1.5 at low density.

**What it says.**  On the conditions it was demonstrated on, DECODE-Plex reaches the two-channel CRLB,
as the paper shows, and so does Spline 3D 2C on isolated molecules; DECODE-Plex holds its detection and
much of its precision at density (Jaccard 0.78 -- 0.82 at 15 -- 25, Spline 3D 2C 0.09 -- 0.41).  Its
real-data models (EM camera, Fig. 2) are 1.2 -- 1.5 times the bound near focus at low density.
**Still open:** Spline 3D 2C's duplicates far from focus (its Jaccard of 0.77 -- 0.80 on isolated
molecules, §11), the Fig. 2 models' false positives at high background (§11), and the biplane band of
§10 -- each to look at again on these sets before the study.

## 13. The conventions on real movies (2026-11-19, first look)

Spline 3D 2C through the converted calibration (SMAP's projective reading), on the first frames of the
paper's movies, against DECODE-Plex's localizations of the same frames (the original's run, §9):

- **Fig. 2a** (its beads' ROI, no EM): the splitter moved by **5.8 rows** of the secondary half since the
  beads -- the shift that takes the offset between the two fitters to zero (scanned 4.5 -- 6.5; the
  likelihood is best at 5.5 -- 6.0), against 6 from DECODE-Plex's two offsets (`Test` -29, `Simulation`
  -35).  There, on 200 frames, 26 000 of Spline 3D 2C's 42 000 localizations lie within 150 nm of one of
  DECODE-Plex's, offset by -0.7 / -0.8 / -4.5 nm (x / y / z) and spread by 30 / 48 / 96 nm (1.48 MAD; the
  movie is dense, 230 fits a frame).
- **Channel 1's orientation is not decided by Fig. 2a**: flipped along the split (the converter now) and
  not, the fits are alike (median `logl_rel` -4.22 against -4.18, the same agreement): its astigmatic PSF
  is nearly symmetric along y.  The orientation rests on §12's evidence (DECODE-Plex's own splines).
- **Fig. 2c** (EM, a 248-pixel ROI) fits badly in every variant tried -- the movie's columns as they are
  and flipped, channel 1 flipped and not, its ROI at x = 266 and at 2, the splitter's move at 0 and +-3.4
  rows: median `logl_rel` -30 to -34, against -4 on Fig. 2a, and z 60 -- 420 nm from DECODE-Plex's.
  Something about this movie and its calibration is not yet understood; open.  **Not the EM register**:
  beads and movies were taken with the same EM settings and nothing should be flipped for it, only the
  channels against each other (Jonas, 2026-11-19) -- the converter flips nothing for EM in SMAPpy's frame
  (it only undoes SMAP's own mirror of the beads, `EMon` 1, back to the camera's orientation), and with
  no EM mirror anywhere (splines and registration as SMAP stored them, the movie as recorded) the fits
  are as bad (-30 to -32, z +380 -- +415 nm from DECODE-Plex's).  Since nothing varied here moves the fit,
  the cause is global: the live cell's high, structured background against a uniform one in the model,
  or a focus or depth offset (z some 400 nm from DECODE-Plex's throughout).  Checked since (Jonas: biplane,
  the focal offset in the two-channel model, photons linked): the fits did link the photons, and Spline 3D
  2C's link applies no ratio on top of the secondary spline's own share (`photon_factors`, 1 for a spline
  pair).  The movie's metadata (`Evolve512`, EM gain 100, gain 1, ROI **175**-0-248-512) gives the camera
  assumed and the ROI the earlier fits guessed wrong; at x = 175 the z agreement with DECODE-Plex is -53
  nm with 1467 matches (shift 0), but the fits stay at `logl_rel` -31.  The movie's background is 300 --
  700 photons a pixel and uneven (66 in the frame's left half, 709 in its right; DECODE-Plex's input shows
  the same level), and even the fits on a background under 60 stay at -16 to -21, a single-channel fit
  of one half at -30.  **The conventions hold on Fig. 2c too** (Jonas): at x = 175 and no shift, the
  best of every variant tried, Spline 3D 2C puts the molecules where DECODE-Plex does, offset by
  +16 / +7 / -53 nm (x / y / z; every other variant moves z by 120 -- 1500 nm).  What is not alike is
  the precision: spread by 73 / 71 / 201 nm (1.48 MAD), where DECODE-Plex's own uncertainties are
  27 / 28 / 74 nm, and 39 % of Spline 3D 2C's fits have no partner within 150 nm.  **The cause is the
  density, not a background:** DECODE-Plex finds 1260 molecules a frame (1.3 per um^2, nearest
  neighbour 340 nm at the median, 42 % of them at probability 0.6 -- 0.9), Spline 3D 2C 35; each of its
  fits has 6 -- 13 of DECODE-Plex's within 800 nm, and the 700 -- 2000 photons a pixel it fits as
  background are those neighbours' light (DECODE-Plex's own background there is 10 -- 95).  So Spline 3D
  2C is no baseline on this movie: Fig. 2c is DECODE-Plex against netfit only, and a fitter's
  conventions are tested on sparse data (S4's sets, Fig. 2a).  The paper's own fit of the
  movie (`results.zip`, all 10 000 frames) is the run of §9: on the first 200 frames 99.9 % of its
  localizations lie within 0.05 px of ours, differing by 0.002 px and 1.7 nm (99th percentile), as
  much as the original's own run on the 4090 does (TF32).
- What the affine reading costs DECODE-Plex on real data is not measured yet: on Fig. 2a's narrow chip
  the two readings differ by 0.07 px at most, too little to see here; Fig. 2c's chip is where it would
  show, once that movie fits.

## 14. Spline 3D 2C's duplicates and misses, found and fixed (2026-11-19)

**The duplicates were the peak finder on a streak, not the pairing.**  Far from focus an astigmatic
spot is a streak, and each channel's peak finder (strict 3 x 3 maxima of a 1.2-px difference of
Gaussians) finds a maximum at each end of it, 4 -- 5 px apart; the two channels' streaks lie across
each other, so their ends are not each other's nearest neighbours and are not paired.  A molecule
there became two or three candidates, every box held the whole spot, and every fit converged to it:
of the pairs of fits within 100 nm, 97 -- 100 % came from one simulated molecule, a median 15 -- 28 nm
apart (on `colour_2a_density_1`, 18 % of the molecules had two candidates; 19 -- 29 % of the isolated
ones in the 10 000-photon and S4 sets were fitted twice).  Candidates cannot be told from two real
molecules 5 px apart before the fit, so the engine now drops them after it: two fits of a frame less
than a pixel apart are one molecule's, and the one nearest its own candidate is kept
(`multichannel.one_per_molecule`, both two-colour fitters; Spline 3D 2C version 15, Gaussian 2D 2C 12;
`test_a_streak_far_from_focus_found_at_both_ends_is_fitted_once`).  Keeping the better
log-likelihood instead, or 0.5 / 1.5 / 2 px, changes the misses by less than 0.3 points.

**The misses were not lost candidates.**  After the fix (density 1, inside both halves):
`colour_2a` misses 9.1 % = 4.4 % with another molecule within 6 px (one fit takes both) + 4.7 % else;
at 10 000 photons 7.3 % = 4.6 % + 2.8 %, of which most have a neighbour 6 -- 10 px away and a fit with
1.5 -- 2.5 times the true photons (a defocused neighbour's light), 0.8 % no fit within 400 nm.
`biplane_2c` 14 % = 3.5 % neighbours + 3.0 % beyond |z| = 900 nm, where the spot is spread so wide
that neither channel has a peak within 8 px at all (10 000 photons, 2 % background), + 7.5 % mostly
dim molecules at |z| > 400 (median 2400 photons, background up to 300 a pixel).  The S4 sets miss none.
The 1.5 % with no candidate on `colour_2a_density_1` are the same: no peak in either channel.

**Jaccard before -> after** (`run_c2.py spline --refit`; precision 0.78 -- 0.98 -> 0.95 -- 1.00 at density
1 and on the photon and S4 sets, recall within 0.01 of before, RMSE near focus within 0.2 nm or lower):

| set | before | after | | set | before | after |
|---|---|---|---|---|---|---|
| biplane density 1 | 0.65 | 0.73 | | colour density 1 | 0.69 | 0.79 |
| biplane density 5 | 0.55 | 0.56 | | colour density 5 | 0.51 | 0.55 |
| biplane density 15 / 25 / 50 | 0.41 / 0.27 / 0.02 | 0.42 / 0.26 / 0.02 | | colour density 15 / 25 / 50 | 0.30 / 0.09 / 0.01 | 0.30 / 0.09 / 0.01 |
| biplane 1000 / 3000 / 10 000 photons | 0.63 / 0.74 / 0.79 | 0.63 / 0.74 / 0.81 | | colour 1000 / 3000 / 10 000 photons | 0.71 / 0.75 / 0.72 | 0.74 / 0.82 / 0.82 |
| biplane S4 2000 / 5000 / 8000 | 0.96 / 0.94 / 0.89 | 0.98 / 0.99 / 0.99 | | colour S4 2000 / 5000 / 8000 | 0.80 / 0.77 / 0.78 | 0.95 / 0.99 / 0.99 |

So on DECODE-Plex's own CRLB conditions (§12) Spline 3D 2C now detects as well as it (0.95 -- 0.99
against 0.89 -- 0.95) at the same precision; what is left of its gap at density 1 is overlap, and at
density 15 and above it is unchanged -- the single-emitter fit's limit, not a detection fault.

## 15. DECODE-Plex's false localizations on the sets (2026-11-19 -- 20)

On `colour_2a_density_1` DECODE-Plex's Fig. 2a model made 0.05 false localizations per 1000 px^2 in the
fifth of frames with the least background in channel 0 and 3.5 in the fifth with the most (precision
0.15 there), at about 0.6 true molecules per 1000 px^2.  Run on its own training simulation
(`scripts/decode/plex_own_frames.py`: DECODE-Plex's sampler, model and post-processing, nothing of
SMAPpy; scored by matching within 1 px and 500 nm in z, 2 px from the edge), it does not do this:

| frames (40 x 40 px) | Fig. 2a: false per 1000 px^2 | precision | Fig. 2c: false per 1000 px^2 | precision |
| --- | --- | --- | --- | --- |
| as trained, 32 molecules a frame | 0.84 -- 0.93 | 0.96 | 1.2 -- 1.4 | 0.92 |
| sparse, 0.8 a frame | 0.01 -- 0.02 | 0.97 | 0.03 -- 0.06 | 0.91 |

flat over each channel's background in both.  The camera is not the difference (the same EMCCD model;
DECODE floors where ours rounds, 0.025 photons), nor the level of the background (each half's median in
our frames is its recorded background within 1 %).  **It is the background's change between frames.**
DECODE-Plex's training holds each channel's background the same in the frames of a window (one value
per sample: 0.127, 0.127, 0.127 in the scaled input), where our C = 2 sets draw a new one every frame.
The Fig. 2a model reads three frames, and a frame brighter than its neighbours is molecules to it: the
false localizations follow the background less its neighbours' mean (correlation +0.75 in channel 0,
+0.59 for the background alone) -- 257 a frame where a frame is brighter than its neighbours by half
the range, 0.3 where it is darker -- bright ones (median 7000 photons) at probability 0.6, spread over
the frame and not near any molecule of this frame or the next.  The same set drawn with one background
in every frame (90 photons a pixel in both channels, near the top of channel 0's range) gives **0.005
false per 1000 px^2, precision 0.992, recall 0.997**.

So `colour_2a`'s sets are to be drawn with a background that changes slowly over the movie, as a real
movie's does; DECODE-Plex's colour results on the present sets understate it.  Fair to netfit too: its
simulation draws each frame's background on its own (`background_window`), as do DECODE's C = 1
simulation (`UniformBackground`, one value a frame) and the C = 1 sets (`sets.py`), so a steady
background is no harder for it, and the C = 1 comparisons stand.

**Redrawn** (`sets_c2.slow_background`: a level every 50 frames, straight in between, at most 2 % of
the range a frame), the ten density sets: DECODE-Plex's colour Jaccard 0.805 / 0.943 / 0.907 / 0.872 /
0.784 at density 1 / 5 / 15 / 25 / 50; on density 1 precision 0.78 (0.47 before), recall 0.995.  What
false localizations remain are not the frames' any more: they lie in a band along the scored area's
edge at the split (y = 190 -- 218 px, the last rows `run_c2.both_channels` keeps with its 7-pixel
margin; about 3 a cell of 27 x 26 px over 653 frames elsewhere, DECODE-Plex's own rate), and grow with
channel 1's background (0.055 a frame per photon a pixel; its drift adds nothing, channel 0's a
little).  Most likely the step between the two channels' backgrounds at the secondary half's edge, seen
inside the network's field: a property of DECODE-Plex at a split, on real movies too, and so scored as
it is -- netfit meets the same edge.  To compare the two away from it, a wider margin there.

**Biplane: DECODE-Plex's inference pads channel 1 with zeros (2026-11-20).**  The Fig. 2c model reads
one frame, yet on `biplane_2c_density_1` its precision was 0.54 at probability 0.5 (DECODE-Plex's own
sparse frames: 0.91).  Ruled out, in turn:

- *The PSF.*  The sets drew spots in smappy's box of the spline grid less 3 pixels (23 of 26), which
  cut 4 -- 8 % of the light at +-1000 nm; DECODE-Plex draws the whole grid.  The sets now draw the
  whole grid (`sets_c2.draw_spline_whole`): totals within 0.4 % of DECODE-Plex's at every z, shapes
  within 0.3 -- 1 % near focus and 1 -- 3 % at +-1000 nm.  The scores did not move.
- *The registration.*  Channel 1's place in the sets agrees with DECODE-Plex's offset maps within
  0.02 px over the chip; the maps `infer` builds (`trafo_aux_factory`) are its training sampler's, times
  the configuration's aux scale of 20, which the inference pre-processing divides out.
- *Whole-chip frames.*  DECODE-Plex's own sampler drawing 256 x 512 frames at the chip's corner
  (`plex_own_frames.py ... 80 256x512`, 300 frames, 0.6 molecules per 1000 px^2): precision 0.905,
  recall 0.806 at 0.5, as on its 40 x 40 crops, false localizations even over the chip.

What remained were two bands, rows y = 67 -- 93 px and columns x = 17 -- 78 px, holding 10,600 of the
13,000 false localizations, with no light under them in channel 0 (the 5 x 5 pixels' excess -580 ADU
against +71,000 for a molecule) and next to none in channel 1, reported bright (21,000 photons) and far
from focus (|z| 760 nm).  The tensor `infer` hands the network says why: **channel 1's first 47 rows and
last 13 columns are exactly zero photons.**  `dual_splitter_factory` moves channel 1 by the
transformation's global offset (`FrameShiftCropPad`, x 47, y -13 for this calibration) before the crop
and pads what comes from beyond the chip with zeros; its comment hopes the padding is cropped away, which
on a whole chip it cannot be.  Training never shows a channel without background, and the network
answers the step with molecules 5 -- 65 px inside it.  A property of DECODE-Plex's inference, so of its
use on real whole-chip movies of this camera too; `colour_2a` is not touched (its model's offset moves
nothing beyond the chip).

The fill decides the scores (`scratchpad` test, monkey-patching `FrameShiftCropPad`), DECODE-Plex Jaccard
on the biplane density sets 1 / 5 / 15 / 25 / 50, and z RMSE at |z| < 300 nm on density 1:

| fill of channel 1's shifted-in pixels | Jaccard | z RMSE, density 1 |
| --- | --- | --- |
| zeros (as published) | 0.512 / 0.752 / 0.882 / 0.765 / 0.731 | 59.1 nm |
| the edge row or column copied | 0.883 / 0.910 / 0.884 / 0.836 / 0.676 | 22.4 nm |
| each frame's median | 0.811 / 0.914 / 0.891 / 0.851 / 0.696 | 24.9 nm |
| a random pixel of the frame's channel 1 | 0.778 / 0.867 / 0.801 / 0.758 / 0.621 | 30.0 nm |

With the edge copied, density 1 gives precision 0.901 and recall 0.963 at 0.5: DECODE-Plex on its own
frames.  No fill is right at density 50, where copying an edge copies molecules.

**So the network reads only the channels' overlap** (`run_c2.overlap`, `fit_plex`; Jonas,
2026-11-20): any fill breaks the planes' intensity relation (biplane) or leaves a channel dark
(colour).  The overlap is found by sending a constant frame through DECODE-Plex's own splitter (biplane
rows 47 -- 255, columns 0 -- 498 of the main half; colour the whole crop), the frame is cut to it after
the split, mirror and integer shift, and the offset maps are cut from the whole crop's at the same
pixels (DECODE-Plex's ``roi_shift`` would shift them through the transformation, but is off by twice
the shift along the mirrored axis).  Every fitter is scored only there, 7 px in (`both_channels`).

The U-Net (depth 3) reads a multiple of 8; the overlap is not one (209 x 499).  Biplane density 1,
all on the same scored area, precision by distance from the padded / cut edge:

| far edges | Jaccard | bottom 4 -- 8 / 8 -- 15 px | right 4 -- 8 / 8 -- 15 px | ms a frame |
| --- | --- | --- | --- | --- |
| trimmed to 208 x 496 (DECODE-Plex's way) | 0.906 | 0.90 / 0.92 | 0.89 / 0.94 | 77 |
| mirrored out to 216 x 504 | 0.885 | 0.49 / 0.73 | 0.88 / 0.85 | |
| repeated edge pixel | | 0.43 / 0.60 | 0.82 / 0.80 | |
| zeros | | 0.04 / 0.09 | 0.23 / 0.31 | |
| trimmed box from all four corners | 0.905 | 0.91 / 0.92 | 0.93 / 0.93 | 308 |
| trimmed box and thin strips | 0.905 | 0.93 / 0.91 | 0.95 / 0.92 | 105 |

Every invented pixel puts false molecules up to 15 px inside, mirrored ones too; the trim loses the cut
rows' molecules (recall 0.40 in the right edge's last 4 px, 0.94 with strips).  **The strips**
(`_strips`, ``--edges strips``): the trimmed box once, then strips along the lower and right edges and their
corner, a multiple of 8 thick (32 px here), each taking over 13 px (half the spots' 26-px box) inside
the box's cut and reaching 13 px before that -- every kept localization had half a box of real pixels
on every side.  Each pass keeps the localizations in its own part (`_owned`); within 0.5 px of a line
between parts both passes' are kept and, frame by frame, the less probable of two within 1 px
dropped.  Pairs within 1 px from different passes: 5 of 20,016 localizations (a single pass has 367
such pairs of its own).  Time 1.4 x the trim, with the strips' batches as many pixels as the box's.
The last 4 rows above the split are poor (precision 0.32) whatever is done: the frame's edge, inside
the scoring margin.  **The benchmark trims** (``EDGES = "trim"``, Jonas): the same scores, the fastest, and the
strips an option for real movies, where the cut rows' molecules matter.  DECODE-Plex on the overlap, biplane Jaccard
0.906 / 0.925 / 0.899 / 0.851 / 0.733 at densities 1 / 5 / 15 / 25 / 50 (0.512 / 0.752 / 0.882 / 0.765
/ 0.731 with the zero padding); colour unchanged, 0.805 / 0.942 / 0.910 / 0.871 / 0.782
(`reports/plex_c2.json`; Spline 3D 2C scored on the same area, `reports/spline_c2.json`).

**C = 1 is not sensitive to the background's regime.**  netfit's GMN7C_s0 (trained on a background
drawn per frame) on B7 drawn again with a steady one (`SimulationSettings.background_jitter` 0.02: one
level a movie, 2 % from frame to frame; same molecules' statistics): Jaccard 0.568 on both, and on
sparse twins (2 per 40 x 40 px, 8192 samples) precision 0.961 / 0.961, recall 0.786 / 0.787.  So
DECODE-Plex's colour false localizations (above) are its training's steady window meeting jumps, not a
general weakness of multi-frame networks.

**Training on a steady background gains nothing** (gpuvm, 2026-11-20, `sbg_queue.sh`: GMN7C with
`background_jitter` 0.02, three seeds, 6 M samples each, against GMN7C_s0 / s2 trained per frame).
Jaccard, background jumping / steady:

| trained on | B7 | B7_l3 (on-time 3 frames) | B7_d2 (sparse) |
| --- | --- | --- | --- |
| per frame (2 seeds) | 0.568 / 0.568 | 0.696 / 0.697 | 0.757 / 0.759 |
| steady (3 seeds) | 0.549 / 0.569 | 0.680 / 0.700 | 0.610 / 0.754 |

On steady test background the two are equal (within 0.003; RMSE lateral / axial the same, 61/70 nm on
B7); on a jumping one the steady-trained network loses 0.02 Jaccard on B7 and 0.15 on sparse frames,
where its precision falls from 0.95 to 0.78 -- DECODE-Plex's colour failure, reproduced.  So netfit
keeps the background drawn per frame for C = 2 too: it costs nothing where the background is steady
and protects where it is not.  `background_jitter` stays, for test sets.

## 16. netfit at C = 2, Round 0: the design (2026-11-20)

Round 0 is netfit made to read two channels the way DECODE-Plex does, with netfit's own C = 1 recipe
elsewhere, so that the study of §7 starts from a working baseline and changes one thing at a time.

**The channels** come from SMAPpy's own dual calibration (`DualColorCalibration`, the `_2ct.h5` the C =
2 sets were drawn from), not DECODE-Plex's files: `layout()` gives each half's rectangle on the chip,
its flip, and the map between the halves' own pixels; `layer_splines()` the two splines in that
orientation.  The network's grid is channel 0's pixels.  Channel 1 is moved onto it by a whole-pixel
shift ``s`` (the map's mean offset over the overlap), so that a molecule at ``p`` in channel 0 is at
``p + d(p)`` in channel 1's image, ``d`` the registration's remainder: a fraction of a pixel near the
overlap's middle, several pixels at its corners where the map turns and scales (biplane: 0.036 of
shear, about 9 px over 256).  ``d`` at each pixel's centre is two input maps, as DECODE-Plex's offset
maps; channel 0's are zero and left out.  The network reads only the overlap, its far edges trimmed
to the U-Net's fold (§15).

**Simulation** draws crops of the overlap at random places, so the network sees every part of the
registration: molecules uniform in the crop, channel 0's spot from the main spline at ``p``, channel
1's from the secondary at ``p + d(p)``; a background per channel and frame (the C = 1 pilot: drawn per
frame costs nothing where it is steady, §15); the camera on each.  Photons: biplane one count, each
plane's share held in its spline (as the sets draw it); colour a total and the share in channel 1,
uniform over 0 -- 1 so that one network serves any pair of dyes (DECODE-Plex draws each channel's
photons on its own over a range).  The spots over the whole spline grid, as the sets and DECODE-Plex
draw them.

**The network** is DECODE-Plex's arrangement in netfit's parts: a per-frame U-Net for each channel
(weights of its own), reading the frame and the two offset maps; the time model per channel; the
channels' features joined, the offset maps beside them, into the union U-Net; heads for p, the
photons (one, or one per channel for colour: DECODE-Plex's), x, y, z, their sigmas, and a background
per channel.  The output's channels are described by a layout object (`network.Layout`), which the
loss, the extraction and the predictor read, so that C = 1 models are read as they were saved.
Netfit's C = 1 choices stay: 64 features (the size screen), GMN7C's ConvGRU with its normalised input
-- but over 3 frames, DECODE-Plex's window, not GMN7C's 7; each window drawn on its own, not streamed
(`draw_batch_2c` refuses ``movie_windows`` > 1); one mean on-time, 1 frame as the sets, not 1 -- 3 --
the mixture loss with the reported sigma, `mixture_peak`.

**Scored** on the C = 2 sets by `run_c2` like DECODE-Plex, on the same overlap; a netfit C = 2 test set
per context for the training's own scores.  Round 0 is done when both contexts train and score; the
target is DECODE-Plex's numbers (§15) at each density.

### Round 0, the first result (2026-11-20)

One seed a context, 6 M samples each (`benchmarks/decode/netfit_c2/R0_*.json`: 64 features, depth 2,
ConvGRU over 3 frames with an on-time of 1 frame, the mixture loss with the reported sigma; biplane on
gpuvm, 1.7 h; colour on decodebox, 2 h), scored by `run_c2 netfit` on the C = 2 sets beside DECODE-Plex
(§15's overlap, the same scored area).  Jaccard, and lateral / axial RMSE at |z| < 300 nm:

| set | netfit J | DECODE-Plex J | netfit lat / ax | DECODE-Plex lat / ax |
| --- | --- | --- | --- | --- |
| biplane density 1 / 5 / 15 / 25 / 50 | 0.879 / 0.900 / 0.895 / 0.863 / 0.797 | 0.906 / 0.925 / 0.899 / 0.851 / 0.733 | 10.5 / 25.4 at 1, 24.3 / 55.1 at 50 | 10.7 / 24.2, 25.9 / 65.1 |
| biplane 1 000 / 3 000 / 10 000 photons | 0.776 / 0.955 / 0.910 | 0.742 / 0.923 / 0.923 | 5.9 / 13.0 at 10 000 | 6.2 / 13.2 |
| colour density 1 / 5 / 15 / 25 / 50 | 0.954 / 0.959 / 0.926 / 0.876 / 0.780 | 0.805 / 0.942 / 0.910 / 0.871 / 0.782 | 10.3 / 18.3 at 1, 50.5 / 79.8 at 50 | 12.0 / 21.1, 47.1 / 82.1 |
| colour 1 000 / 3 000 / 10 000 photons | 0.927 / 0.987 / 0.971 | 0.769 / 0.985 / 0.922 | 10.8 / 18.3 at 10 000 | 11.5 / 18.8 |

netfit's errors are as small or smaller nearly everywhere and it finds more molecules at every density
(biplane recall 0.865 against 0.758 at density 50).  Where it loses is biplane's precision at low
density (0.905 against 0.934 at density 1): its extra false localizations are faint (median 650
photons) and far from any molecule of their frame, and 28 % of them lie within 1.5 px of a molecule in
the frame before or after (DECODE-Plex's: 1 %).  netfit was trained on molecules that stay on over
frames, so a bright spot next door is a faint tail to it; the C = 2 sets put each molecule in one frame
alone, so there is never a tail -- the sets' lack, as §15's background was.

**Colour assignment** (each matched molecule's share of its light in channel 1 against its dye's,
0.78 or 0.92; the dye called by the midpoint): netfit and DECODE-Plex alike at density 1 -- 15 and at
10 000 photons (share error sd 0.012 -- 0.020, the right dye 99.3 -- 100 %), and netfit's photons right
where DECODE-Plex's are 9 -- 15 % high.  At 1 000 photons netfit calls the right dye 90 % of the time
against 97 %: its share is pulled 0.045 towards 1/2 and its photons 7 % low, the molecules below its
training's total of 3 000 -- 40 000 (DECODE-Plex's channels were trained from 200 and 1 000 photons).

**Next** (open): the sets drawn with molecules that blink over frames, as real movies do and as the
time model is for -- or netfit on single frames for biplane, matching the sets as they are; colour's
training range down to the sets' 1 000 photons; then seeds, and the study of §7.

### Round 1: blinking sets, and biplane on one frame (2026-11-20)

**The blinking sets** (`sets_c2`'s ``<context>_blink_...``, molecules on for an exponential time of mean
1 frame, a truth row per frame on): harder for every fitter -- the rows of a molecule on for part of a
frame hold part of its photons, and the scoring counts every row from 100 photons (DECODE-Plex biplane
density 1: 0.792, against 0.906 without blinking; Spline 3D 2C 0.641) -- but the errors against the
bound are as before (DECODE-Plex 1.2 -- 1.3 on biplane).  Jaccard and lateral / axial RMSE at |z| <
300 nm:

| set | netfit, 3 frames | netfit, 1 frame | DECODE-Plex |
| --- | --- | --- | --- |
| biplane density 1 | 0.835, 16.4 / 36.7 | 0.785, 22.5 / 49.3 | 0.792, 22.5 / 47.7 |
| biplane density 15 | 0.821, 18.7 / 43.9 | 0.786, 23.6 / 51.1 | 0.762, 25.2 / 56.3 |
| biplane density 50 | 0.747, 24.0 / 51.7 | 0.711, 26.9 / 59.0 | 0.642, 32.8 / 78.9 |
| biplane 1 000 photons | 0.563, 27.4 / 70.8 | 0.562, 36.6 / 84.2 | 0.469, 36.9 / 86.7 |
| biplane 10 000 photons | 0.819, 14.3 / 31.0 | 0.752, 22.4 / 46.1 | 0.777, 22.5 / 48.7 |
| colour density 1 | 0.901, 20.6 / 36.4 | | 0.746, 21.7 / 38.4 |
| colour density 50 | 0.721, 41.2 / 69.5 | | 0.675, 44.6 / 77.4 |
| colour 1 000 photons | 0.561, 39.3 / 80.0 | | 0.341, 44.1 / 84.7 |

netfit reading three frames beats DECODE-Plex on all 16 sets, precision included (biplane density 1:
0.939 against 0.925) -- the echoes of §16's Round 0 were the one-frame sets'.  On one frame, as
DECODE-Plex's biplane model reads, netfit is DECODE-Plex's equal at low density (0.785 / 0.792, the
same errors) and better as the density rises (0.711 / 0.642 at 50): what the second channel's network
arrangement gives.  The three frames are worth 0.03 -- 0.07 in Jaccard and a quarter to a third of the
error near focus, the frames a molecule is on in pooled.

**Training** (6 M samples, batch 64, bf16, compiled; the early stop never fired): biplane on 3 frames
1.8 h on the H100 (35 s an epoch of 32 768), on 1 frame 0.9 h (17 s); colour on 3 frames 2.0 h on the
4090 (40 s).  Converged early: 90 % of the final Jaccard (on the training's own test frames) at 0.4 --
0.6 M samples (8 -- 10 minutes), 95 % at 0.7 -- 0.9 M (15 minutes), 99 % at 2.8 -- 3.5 M (about an
hour); the last 2.5 M samples gain the last 1 %.  6.7 M parameters (64 features, a U-Net per channel
and a ConvGRU per channel), 5.7 M on one frame.

**Fitting**, the sets' 512 x 512 frames (two 209 x 500 channels read): netfit 30 ms a frame on 3
frames, 8 ms on 1, on the H100 while a training shared it; DECODE-Plex 77 -- 85 ms on the Mac's GPU,
Spline 3D 2C 4 -- 93 ms on its CPU (growing with the density) -- different machines, so the speeds are
not yet a comparison.

**Colour from 500 photons** (`R1_colour.json`, 6 M samples, decodebox 2.0 h), on the blinking sets,
Jaccard against Round 0's colour (from 3 000) and DECODE-Plex: 1 000 photons 0.632 / 0.561 / 0.341,
3 000 photons 0.809 / 0.796 / 0.692, density 1 0.876 / 0.901 / 0.746, 10 000 photons 0.851 / 0.881 /
0.796, the other densities within 0.004 of Round 0.  The dimmest molecules are found more often; the
bright ones a little less (a wider range for the same network).  The dye (each matched molecule's share
in channel 1 against the midpoint of 0.78 and 0.92) is not called better: right 89 % at 1 000 photons
and 97 % at 3 000 (Round 0 the same), DECODE-Plex 98 % and 99 %.  netfit draws a molecule's share
uniform over 0 -- 1, so a dim one's share is pulled towards 1/2 and lands below the midpoint;
DECODE-Plex draws each channel's photons on its own (channel 0 200 -- 20 000, channel 1 1 000 --
100 000), which puts most of its training's shares between 0.7 and 1, where both dyes are -- a prior
that fits these dyes.  The fair test is netfit trained with the share over the dyes' window (0.6 --
1), as a user who knows the two dyes would set it.

**The dyes' shares are the data's to say** (Jonas, 2026-11-20): approximate brightness ratios from the
prefit, or typed.  netfit's colour training then draws each molecule's share around one of them
(`ChannelSettings.dyes`, normal with ``share_sd`` 0.03, inside ``share``) -- or, a ratio known only
roughly, uniform over a window (``share``).  Both queued after the study, three seeds each:
``S_colour_dyes`` (0.78 and 0.92, decodebox) and ``S_colour_win`` (0.6 -- 1, gpuvm).  The prefit's part
-- the dyes' ratios from the data's brightest isolated spots in both channels -- belongs to the fitter
in SMAPpy (step 5's item 11).

### The architecture study: early fusion wins (2026-11-21)

Five arrangements of Round 1's network, three seeds each (`benchmarks/decode/netfit_c2/study/`,
3 M samples -- half of Round 1's, so a little behind it; biplane on gpuvm, colour on decodebox, colour *share* on gpuvm), scored on the
blinking sets with 3 frames read:

* **base**: Round 1's -- a per-frame U-Net per channel, a time model per channel, the offset maps
  into every per-frame net and the union;
* **tjoin** (`two_time="joined"`): one time model over both channels' features;
* **tboth** (`"both"`): per channel and joined;
* **early** (`two_fuse="early"`): channel 1 resampled onto channel 0's grid inside the network (the
  registration's offsets, `grid_sample` at p + d(p)), one per-frame U-Net reading both -- 4.4 M
  parameters against 6.7 M;
* **auxu** (`two_aux="union"`): the offset maps only into the union;
* colour only, **share** (`photon_out="share"`): a total and channel 1's share instead of two counts.

Mean (sd over seeds), DECODE-Plex and Spline 3D 2C one run each:

| biplane, Jaccard | d 1 | d 15 | d 50 | 1 000 ph | 3 000 ph | 10 000 ph |
|---|---|---|---|---|---|---|
| DECODE-Plex | 0.792 | 0.762 | 0.642 | 0.469 | 0.662 | 0.777 |
| base | 0.814 (0.014) | 0.815 (0.001) | 0.737 (0.001) | 0.525 (0.022) | 0.737 (0.006) | 0.797 (0.018) |
| tjoin | 0.808 (0.013) | 0.822 (0.000) | 0.745 (0.002) | 0.541 (0.011) | 0.745 (0.003) | 0.784 (0.020) |
| tboth | 0.803 (0.012) | 0.820 (0.001) | 0.744 (0.002) | 0.530 (0.014) | 0.737 (0.008) | 0.785 (0.014) |
| **early** | **0.829** (0.011) | **0.827** (0.002) | **0.760** (0.001) | **0.564** (0.013) | **0.768** (0.006) | **0.801** (0.024) |
| auxu | 0.786 (0.012) | 0.790 (0.002) | 0.695 (0.003) | 0.527 (0.019) | 0.698 (0.007) | 0.771 (0.016) |

| biplane, \|z\| < 300 lat / ax nm | d 1 | d 15 | d 50 | 1 000 ph | 10 000 ph |
|---|---|---|---|---|---|
| DECODE-Plex | 22.5 / 47.7 | 25.2 / 56.3 | 32.8 / 78.9 | 36.9 / 86.7 | 22.5 / 48.7 |
| Spline 3D 2C | 15.6 / 32.6 | 39.3 / 58.9 | 54.1 / 83.3 | 30.6 / 64.7 | 15.9 / 35.2 |
| base | 16.5 / 39.1 | 19.1 / 42.8 | 24.2 / 53.1 | 25.4 / 60.3 | 15.1 / 33.7 |
| early | 16.3 / 35.5 | 18.4 / 40.8 | 23.4 / 51.6 | 25.1 / 61.7 | 14.4 / 30.6 |
| auxu | 17.6 / 47.5 | 19.8 / 51.6 | 27.6 / 67.1 | 29.7 / 84.7 | 16.2 / 41.3 |

| colour, Jaccard / dye right | d 1 | d 15 | d 50 | 1 000 ph | 3 000 ph | 10 000 ph |
|---|---|---|---|---|---|---|
| DECODE-Plex | 0.746 / 0.991 | 0.830 / 0.985 | 0.675 / 0.975 | 0.341 / 0.973 | 0.692 / 0.986 | 0.796 / 0.990 |
| base | 0.882 / 0.992 | 0.859 / 0.984 | 0.712 / 0.972 | 0.578 / 0.887 | 0.796 / 0.969 | 0.862 / 0.987 |
| tjoin | 0.888 / 0.993 | 0.863 / 0.983 | 0.717 / 0.975 | 0.574 / 0.933 | 0.797 / 0.971 | 0.864 / 0.989 |
| tboth | 0.889 / 0.994 | 0.864 / 0.984 | 0.720 / 0.975 | 0.579 / 0.926 | 0.794 / 0.976 | 0.862 / 0.989 |
| **early** | 0.886 / 0.993 | 0.863 / 0.985 | 0.720 / 0.973 | 0.597 / 0.936 | 0.796 / 0.980 | 0.868 / 0.989 |
| auxu (2 seeds) | 0.862 / 0.984 | 0.851 / 0.974 | 0.690 / 0.965 | 0.613 / 0.945 | 0.790 / 0.969 | 0.844 / 0.978 |
| share | 0.872 / 0.992 | 0.860 / 0.984 | 0.716 / 0.971 | 0.562 / 0.932 | 0.792 / 0.971 | 0.849 / 0.988 |

Colour at \|z\| < 300, lat / ax nm, 1 000 photons: DECODE-Plex 44.1 / 84.7, base 39.4 / 76.2, early
37.5 / 64.1; at 10 000: 24.3 / 44.1, 23.4 / 42.7, 22.8 / 39.9.  The dye seeds' spread at 1 000
photons is large (base 0.887 sd 0.056).

**Pixelation** (`run_c2.pixelation`: KS from uniform of the matched localizations' sub-pixel
positions on the network's grid, the larger of x and y; the truth's own, the counting noise, 0.01 --
0.02): at the counting noise for every fitter from density 1 to 50 and at 10 000 photons.  At 1 000
photons, biplane: DECODE-Plex 0.144, base 0.069 (0.006), tjoin 0.055, tboth 0.074, **early 0.046
(0.009)**, auxu 0.075, Spline 0.018 (Round 1's one-frame network 0.033).  It grows with \|z\|: base
0.030 at \|z\| < 150 nm to 0.171 at 500 -- 700, early 0.042 to 0.087 -- early halves it far from focus.
Colour, 1 000 photons: DECODE-Plex 0.091, base 0.028, early 0.025, tjoin 0.038, share 0.039.

**Speed** (density 15, the sets' 512 x 512 frames, ms a frame): biplane on the H100 base 17.3, tjoin
18.8, tboth 28.0, early 10.7, auxu 17.3; colour on the 4090 base 13.0, tjoin 15.0, tboth 22.8, early
7.6 (share 7.4 on the H100, not comparable).

**So early fusion is the arrangement**: the best Jaccard in biplane on every set, beyond the seeds'
spread, colour as good as the best, the best axial precision, the least pixelation, a third fewer
parameters and 40 % faster.  Joining the time models buys little (a little Jaccard at high density, a
worse z at 1 000 photons in biplane) and costs time; feeding the offset maps only to the union loses
everywhere but colour at 1 000 photons; *share* gains nothing over two counts.  Early with the halves
swapped and colour's dye priors: below, "The last runs".

**Early's dye calls, molecule by molecule** (colour, 1 000 photons; `dye_wrong.py`-style matching, each
model's matches of the same truth).  On the ~5 100 molecules both early and DECODE-Plex find, early
is right 96.3 / 97.9 / 97.6 % (seeds 0 -- 2) against DECODE-Plex's 97.4 %; at 3 000 photons 98.9 --
99.0 % against 98.7 %.  Early's lower overall number (93.6 %) is the 3 500 -- 4 300 dim molecules it
finds and DECODE-Plex misses, called right 83 -- 92 %.  Where early is wrong and DECODE-Plex right
(136 molecules, seed 0), the true photons are lower (median 810 against 1 000: a molecule on for part
of a frame) and the dye is almost always the 0.92 one, its share read 0.84 -- pulled towards 1/2.
Which dye a seed favours differs (seed 0 loses the 0.92 dye, 91 %; seed 2 the 0.78, 97 %), and the
seeds' wrong molecules hardly overlap (117 in all three, 1 236 in any): a share bias each training
settles on, not molecules that are hard.  base has it worse (0.92 dye 77 % on the common molecules);
``win`` (share 0.6 -- 1) removes it (98.3 %), the dye priors are the cure.

**Going on** (Jonas, 2026-11-21): early fusion is the arrangement optimised from here; a late-fusion
model with the offset maps as input -- DECODE-Plex's published design, `two_fuse="late"`, base or
tjoin -- stays as an option for the user.  None of the study's trainings converged: the in-focus
RMSE on the training's test frames still falls at their end, 3 M samples (92 epochs) (`training_vs_time.png`, scratch).

**The fair bound for a fit that reads several frames** (`sets_c2.window_crlb`, ``crlb_w3.npz``
beside each blinking set's truth; `report` scores ``*_w3`` beside the one-frame bound, and RMSE / CRLB
in focus, ``core.rmse_over_crlb_*``).  A molecule stays put while it is on, so in the frames it is on
within the window x, y and z are shared and each frame keeps its own photons and backgrounds: each
frame's information on x, y, z (its nuisances taken out, the Schur complement) adds.  A dim first or
last frame adds little; the window's total photons put into one frame would drop two frames'
background and sit 15 -- 28 % lower (colour, density 1).  Against the bound for the frames each
fitter reads, RMSE / CRLB in focus (|z| < 300 nm), lat / ax:

| | frames | 10 000 ph | 1 000 ph | density 15 |
|---|---|---|---|---|
| biplane DECODE-Plex | 1 | 1.13 / 1.17 | 1.33 / 1.49 | 1.50 / 1.45 |
| biplane Spline 3D 2C | 1 | 1.05 / 1.09 | 1.08 / 1.08 | 4.18 / 2.73 |
| biplane netfit Round 1 | 1 | 1.07 / 1.05 | 1.15 / 1.26 | 1.31 / 1.26 |
| biplane base | 3 | 1.18 / 1.25 | 1.22 / 1.38 | 1.52 / 1.48 |
| biplane early | 3 | 1.12 / 1.14 | 1.19 / 1.39 | 1.45 / 1.41 |
| colour DECODE-Plex | 3 | 1.33 / 1.35 | 1.77 / 1.95 | 2.18 / 2.08 |
| colour base | 3 | 1.17 / 1.18 | 1.31 / 1.45 | 1.81 / 1.64 |
| colour early | 3 | 1.15 / 1.12 | 1.23 / 1.20 | 1.83 / 1.68 |

Every fitter is at or above its bound now (the 0.6 -- 0.7 before were the one-frame bound).  The
3-frame networks use the neighbouring frames' photons less well than the 1-frame network uses its
own frame's: 12 -- 25 % above the bound in focus, the 1-frame netfit 5 -- 15 %.  Room for the time
model.

**The multi-emitter bound** (`sets_c2.joint_crlb`, ``crlb_joint_w1.npz`` / ``crlb_joint_w3.npz``
beside the density sets' truth; `smappy.training.joint_crlb` made two-channel: channel 1's
derivatives through the registration, a background per channel and overlap group, the photons per
row and channel, x, y, z one per molecule over the frames read, C = 1's prior of 1 pixel and the
range's sd).  Local: each molecule frees the spots whose patch overlaps one of its own in either
channel and holds the rest known -- at density 50 a frame's patches all chain.  At density 1 it is
the single-emitter bound (median 0.999).  RMSE / CRLB in focus against it, each fitter's window,
lat / ax:

| | frames | d 15 | d 25 | d 50 |
|---|---|---|---|---|
| biplane DECODE-Plex | 1 | 1.33 / 1.31 | 1.46 / 1.48 | 1.68 / 1.77 |
| biplane netfit Round 1 | 1 | 1.16 / 1.12 | 1.19 / 1.17 | 1.22 / 1.19 |
| biplane base | 3 | 1.34 / 1.32 | 1.41 / 1.47 | 1.55 / 1.56 |
| biplane early | 3 | 1.28 / 1.26 | 1.34 / 1.45 | 1.44 / 1.46 |
| colour DECODE-Plex | 3 | 1.77 / 1.75 | 1.83 / 1.82 | 1.93 / 1.96 |
| colour base | 3 | 1.53 / 1.44 | 1.58 / 1.62 | 1.70 / 1.69 |
| colour early | 3 | 1.51 / 1.44 | 1.57 / 1.54 | 1.66 / 1.65 |

Crowding's own cost taken out, the 1-frame network stays within 22 % of its bound up to density 50;
the 3-frame networks are 28 -- 66 % above theirs, and the gap grows with density: they use the
neighbouring frames' photons least where they would help most.  The time model's job.

**A bound at inference** (`smappy.netfit.bounds.plugin_crlb`, off by default; ``NETFIT_CRLB`` in
`run_c2.fit_netfit`): the single-emitter bound from the fit's own position, z, photons and the
network's background per channel, on both channels, and over the frames read by linking each
detection to the nearest one within a pixel in the neighbouring frames.  On early's fits it is the
truth's bound within 1 % at the median (80 % within 5 -- 12 %), 1 and 3 frames, density 1 -- 15 and
1 000 photons; some 15 s for 10 000 localizations.  The network's reported sigma is above the
3-frame plug-in bound for 85 -- 99 % of localizations (median 1.2 -- 1.6 times it).

**For later: the time model of the best arrangement** (Jonas, 2026-11-21).  Time is one of the
optimisations of early fusion, not part of the architecture study: GMN7C's full recipe at C = 2 --
streaming in `draw_batch_2c` (C = 1: 20 s an epoch instead of 35, the same numbers), 5 -- 7 frames,
on-times mixed over 1 -- 3 frames -- tested on C = 2 sets drawn with a mean on-time of 3 frames
beside the 1-frame ones, which show how the networks pool a molecule across frames (the 3-frame
networks sit 28 -- 66 % above their multi-emitter 3-frame bound, the 1-frame network 16 -- 22 %).

**The last runs** (3 seeds each, scored as the rest, 2026-11-21).  *eswap*, early fusion with channel
1 as the main -- the grid and positions follow it -- is early's equal in biplane (Jaccard 0.829 /
0.826 / 0.759 at densities 1, 15, 50 against 0.829 / 0.827 / 0.760; 0.535 against 0.564 at 1 000
photons; pixelation 0.040 against 0.046) and in colour, where channel 1 is the bright one, a shade
better: Jaccard 0.881 / 0.867 / 0.726 and 0.590 at 1 000 photons, RMSE / CRLB in focus 1.13 / 1.09
at 10 000 photons against 1.15 / 1.12, the multi-emitter bound at density 50 1.63 / 1.64 against
1.66 / 1.65, right dye 95.0 % against 93.6 %.  Which channel is main hardly matters; the bright one,
if anything.  The dye priors -- *win* (share 0.6 -- 1) and *dyes* (normal around 0.78 and 0.92, sd
0.03), both on base, not early -- do what they were for: the share's bias at 1 000 photons goes from
-0.04 -- -0.06 to -0.01 -- 0.00, its spread from 0.040 to 0.027 -- 0.030, the right dye from 89 %
(base) to 96.5 / 94.1 % (97.3 % both, seed 0).  They cost recall on the dim: Jaccard at 1 000 photons
0.49 / 0.47 against base's 0.58 (recall 0.50 / 0.47 against 0.56 at seed 0, precision the same
0.996 -- 0.998); at density 1 -- 50 and 10 000 photons nothing changes, pixelation halves (0.017 /
0.018 against 0.028).  A network that knows the share must be near one of two values keeps fewer of
the spots too dim to tell which.  So the priors belong on early -- *early + win*, *early + dyes*, on
eswap's channel order -- and their recall at 1 000 photons is the thing to watch.

**Cost, measured fairly** (`benchmarks/decode/netfit_c2/study/timing.py`; every architecture with
the same script on gpuvm's H100, nvidia-smi empty before and after each).  Epochs to 99 % of the
final test Jaccard, from the runs' own logs (machine-free: every epoch draws the same 32 768
samples), times one epoch there (epochs 2 and 3: drawing, steps, test evaluation); fit of the
density-1 set (646 frames) after a warm-up.  *eswap* is timed as early, *win* and *dyes* as share.

| | params | epochs to 99 % | s / epoch | train h | fit ms / frame |
|---|---|---|---|---|---|
| biplane base | | 58 | 30.3 | 0.49 | 16.3 |
| biplane tjoin | | 57 | 31.3 | 0.50 | 17.7 |
| biplane tboth | | 57 | 35.7 | 0.56 | 27.2 |
| biplane early | 4.4 M | 56 | 25.6 | 0.40 | 9.7 |
| biplane auxu | | 60 | 31.0 | 0.51 | 16.3 |
| colour base | | 54 | 30.7 | 0.46 | 7.1 |
| colour tjoin | | 56 | 31.4 | 0.49 | 7.7 |
| colour tboth | | 59 | 36.0 | 0.59 | 11.8 |
| colour early | 4.4 M | 58 | 31.3 | 0.50 | 4.2 |
| colour auxu | | 54 | 41.2 | 0.61 | 7.1 |
| colour share | | 56 | 30.9 | 0.48 | 7.1 |

Every architecture reaches 99 % in 53 -- 60 epochs (1.7 -- 2.0 M samples), the same within the
seeds; the time to train is the epoch's.  Drawing one epoch's batches alone takes 17.6 -- 17.9 s for
all of them -- it runs in a thread beside the steps, so it sets no architecture's pace, but it is
more than half of the fastest epoch: a faster network gains less than its steps suggest, and
streaming (C = 1: 20 s an epoch instead of 35) is worth having before the time model.  Early is the
cheapest to train in biplane and as cheap as base in colour, and the fastest to fit in both (9.7 and
4.2 ms a frame against 16.3 and 7.1).

**A dim channel, and molecules on for 3 frames** (2026-11-21; `sets_c2`'s ``<context>_blink_l3_density_15``,
mean on-time 3 frames, and colour's ``_blink_dimmain_3000`` / ``_dimsecond_3000``: the same molecules,
one dye, 3 000 photons, 97 % of its light in channel 1 -- the main channel dim -- or in channel 0;
the trained models, no retraining: their share is uniform over 0 -- 1, their on-time 1 frame).
`report.CORE_PHOTONS_MIN` adds a photon threshold to the in-focus core.

*The dim channel.*  Jaccard, dim main / dim registered channel: early 0.791 / 0.812, eswap (channel 1
its main) 0.808 / 0.798, base 0.797 / 0.804, tjoin 0.792 / 0.809, DECODE-Plex 0.657 / 0.805.  Each
network loses 0.01 -- 0.02 when its own main channel is the dim one, precision unchanged, RMSE / CRLB
1.10 -- 1.16 either way; the share's bias -0.01 -- -0.02 / +0.02 -- +0.03.  Early's resampling of the
registered channel costs nothing extra; DECODE-Plex loses 0.15 with the dim main channel and puts a
+0.12 bias on the share with the dim second one.

*Density 15, on-time 1 against 3*, mean over seeds, RMSE / CRLB in focus against each fitter's
window (single / multi-emitter, lat / ax, ≥ 2 000 photons in the frame; DECODE-Plex biplane 1 frame):

| | J, on 1 / 3 | single, on 1 | single, on 3 | multi, on 1 | multi, on 3 |
|---|---|---|---|---|---|
| biplane DECODE-Plex | 0.762 / 0.849 | 1.64 / 1.78 | 2.07 / 2.25 | 1.32 / 1.43 | 1.57 / 1.71 |
| biplane early | 0.827 / 0.907 | 1.41 / 1.45 | 1.62 / 1.90 | 1.15 / 1.15 | 1.27 / 1.50 |
| biplane tjoin | 0.822 / 0.902 | 1.50 / 1.52 | 1.83 / 2.27 | 1.23 / 1.20 | 1.44 / 1.79 |
| biplane base | 0.815 / 0.895 | 1.48 / 1.59 | 2.04 / 2.41 | 1.21 / 1.26 | 1.60 / 1.91 |
| colour DECODE-Plex | 0.830 / 0.882 | 2.36 / 2.27 | 2.55 / 2.38 | 1.76 / 1.80 | 1.93 / 1.90 |
| colour early | 0.863 / 0.917 | 2.03 / 1.91 | 2.03 / 1.89 | 1.49 / 1.49 | 1.53 / 1.50 |
| colour eswap | 0.867 / 0.919 | 1.99 / 1.89 | 1.98 / 1.89 | 1.46 / 1.48 | 1.49 / 1.50 |
| colour tjoin | 0.863 / 0.917 | 2.04 / 1.91 | 2.02 / 1.92 | 1.50 / 1.49 | 1.52 / 1.53 |
| colour base | 0.859 / 0.913 | 2.07 / 1.92 | 2.04 / 1.95 | 1.52 / 1.50 | 1.54 / 1.55 |

Longer on-times find more and localise better for every fitter (biplane early in focus 18.4 / 40.8 nm
to 10.3 / 24.6), but in biplane the 3-frame networks reach less of what the extra frames offer: early
from 1.15 to 1.27 / 1.50 of the multi-emitter bound, tjoin to 1.44 / 1.79, base to 1.60 / 1.91 --
late fusion pools frames worst, axially most.  In colour nothing changes with the on-time (1.5 for
all) and the arrangements tie; colour sits at 1.5 of its bound even bright, biplane at 1.15 -- not
yet understood.  Pixelation at density 15 is at the counting noise for all (KS 0.008 -- 0.022); dye
calls 98.3 -- 98.5 % on 1, 99.3 % on 3 for all, DECODE-Plex the same with photons 11 % high.  So the
time model's work is biplane's, and trained with on-times of 1 -- 3 frames it should close what early
already closes best.

**Robust metrics and outliers** (2026-11-21; `report`'s core now leaves out errors beyond
``CORE_TRIM`` = 5 of their bound on any axis -- ``*_trim`` -- and counts them, ``outliers*``; the
robust width, ``crlb_width_*``, beside it).  In focus, ≥ 2 000 photons in the frame (the photon sets
all), multi-emitter bound for each fitter's window, lat / ax; outliers against the single-emitter
bound:

| | d 15: all / trimmed / robust | outliers d 15 / d 50 | d 50 robust | on 3, d 15 robust |
|---|---|---|---|---|
| biplane early | 1.15 / 1.15, 1.14 / 1.13, 1.13 / 1.24 | 1.5 / 9.6 % | 1.25 / 1.37 | 1.23 / 1.37 |
| biplane tjoin | 1.23 / 1.20, 1.18 / 1.17, 1.20 / 1.26 | 1.9 / 11.6 % | 1.35 / 1.45 | 1.33 / 1.47 |
| biplane base | 1.21 / 1.26, 1.19 / 1.24, 1.22 / 1.31 | 2.1 / 12.6 % | 1.38 / 1.54 | 1.36 / 1.57 |
| biplane DECODE-Plex | 1.32 / 1.43, 1.27 / 1.27, 1.23 / 1.31 | 3.8 / 21.8 % | 1.55 / 1.58 | 1.36 / 1.48 |
| colour eswap | 1.46 / 1.48, 1.34 / 1.36, 1.21 / 1.29 | 4.8 / 25.1 % | 1.41 / 1.50 | 1.27 / 1.32 |
| colour early | 1.49 / 1.49, 1.37 / 1.40, 1.24 / 1.36 | 5.5 / 26.4 % | 1.46 / 1.53 | 1.34 / 1.40 |
| colour tjoin | 1.50 / 1.49, 1.41 / 1.38, 1.30 / 1.33 | 5.8 / 27.8 % | 1.49 / 1.55 | 1.38 / 1.38 |
| colour base | 1.52 / 1.50, 1.41 / 1.40, 1.33 / 1.37 | 6.4 / 28.7 % | 1.51 / 1.59 | 1.42 / 1.43 |
| colour DECODE-Plex | 1.76 / 1.80, 1.54 / 1.55, 1.46 / 1.54 | 8.9 / 34.1 % | 1.68 / 1.74 | 1.53 / 1.60 |

The order holds -- early (eswap in colour) first on every robust measure, every set -- but the
distances shrink: most of what separates the fitters at density 15 is how often they fail, not how
precise they are when they do not (DECODE-Plex biplane 1.32 / 1.43 is 1.23 / 1.31 robust, against
early's 1.13 / 1.24).  At density 50 and on molecules on for 3 frames real gaps stay (biplane
robust ax 1.37 early, 1.54 -- 1.57 base).  Colour's gap to biplane is mostly its outliers: 3 -- 4
times as many, robust widths 1.21 -- 1.33 against 1.13 -- 1.22.  Isolated (the photon sets),
outliers are 0.1 -- 1 % and nothing changes.  The outlier rate goes beside Jaccard as a headline
number.

**Flagging them from the fit** (`flag.py`, scratch): the reported sigma over the plug-in bound ranks
outliers with AUC 0.92 (biplane early), 0.88 (colour early), 0.80 at density 50 -- dropping its
highest 5 % removes 69 % / 38 % of the outliers at density 15; the probability (AUC 0.47 -- 0.58)
and the photons (0.45 -- 0.52) do not.  An outlier's error is a median 1.7 -- 2.2 of its reported
sigma (0.9 -- 1.0 for the rest): the network knows it is unsure, not how much.

**Linking the channels' photons**, two quick trainings against early's own 1 M-sample snapshot (same
schedule to there).  The bound: unlinked / linked 1.28 in z for biplane (its planes' different focus
makes the ratio tell z; with plane 0's spline in both it is 1.000), 1.000 for colour (one focus).
*Biplane unlinked* (each plane its own count, share 0.25 -- 0.75): z loses what the bound says --
multi-emitter ax 1.38 → 1.76 at density 1, 1.30 → 1.71 at 15, 1.21 → 1.49 at 10 000 photons -- the
Jaccard 0.01 -- 0.05, outliers 3 -- 6 times as many.  *Colour linked* (each molecule exactly one of
the two dyes): no gain where the bound has none (density 1, 10 000 photons) but in crowding -- lat /
ax 1.63 / 1.63 → 1.44 / 1.52 at density 15, 1.89 / 1.91 → 1.59 / 1.70 at 50, outliers halved
(1.7 → 1.1 %, 4.2 → 2.2 %), dyes 98.3 → 99.0 %; at 1 000 photons the dye calls fall (86.5 → 81.4 %,
a wrong call now a whole swap).  So linking is worth its bound in biplane and robustness against
neighbours in colour, half of colour's gap to biplane at density 15.

### The standard report, filters, and how to go on (Jonas, 2026-11-21)

**Decided.**  (1) Precision is reported over |z| < 500 nm, not 300: normalised by the bound, the
order of the fitters is the in-focus one (checked on the four sets below: one flip, bright biplane,
where the 3-frame networks lose some 6 % against their bound at 300 -- 500 nm and DECODE-Plex does
not -- which the wider band now shows).  (2) How colour uses the dyes is the user's choice at
training (and a GUI option): ratios from the prefit or typed, a wider range, or independent counts
(isotropic -- what four dyes in one data set will need).  Tested next: the ratios known with a broad
spread (``S_colour_dyesb``, sd 0.06, 3 seeds) against independent counts (early), on sets with
known ratios.  (3) Both outlier flags -- the reported sigma over the plug-in bound and the goodness
of fit -- are options; the GUI should say what each costs (a tooltip, a colour, a line).
(4) Land after this step.

**The standard report** (`suite/standard_c2.py`, ``python standard_c2.py CONTEXT RUN ...``): four
sets -- 10 000 and 1 000 photons isolated, density 15 with molecules on for 1 and for 3 frames --
and Jaccard; outliers (matched, error beyond 5 of its own single-emitter bound for the frames read,
any axis); the inliers' RMSE / CRLB against the multi-emitter bound for the frames read (density
sets) or the single-emitter one; pixelation at 1 000 photons, KS of the inliers less the truth's at
the same molecules (outliers pixelate no differently: KS 0.058 all, 0.058 inliers, biplane early);
the right dye on inliers; and sigma honesty at density 15, the robust width of the inliers' errors
over their reported sigma (1 honest, below it cautious).  |z| < 500 nm; density sets from 2 000
photons in the frame.  Mean (sd) over 3 seeds:

biplane

| model | J 10k | J 1k | J d15 | J on3 | out d15 % | out on3 % | 10k lat | 10k ax | 1k lat | 1k ax | d15 lat | d15 ax | on3 lat | on3 ax | pix 1k | dye 1k % | dye d15 % | σ lat | σ ax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DECODE-Plex | 0.777 | 0.469 | 0.762 | 0.849 | 4.1 | 7.8 | 1.11 | 1.14 | 1.30 | 1.49 | 1.18 | 1.26 | 1.26 | 1.37 | 0.130 |  |  | 1.04 | 1.18 |
| early | 0.801 (0.024) | 0.564 (0.013) | 0.827 (0.002) | 0.907 (0.001) | 1.5 (0.1) | 3.5 (0.1) | 1.17 (0.02) | 1.18 (0.02) | 1.18 (0.02) | 1.33 (0.04) | 1.14 (0.01) | 1.19 (0.01) | 1.17 (0.00) | 1.30 (0.02) | 0.032 (0.009) |  |  | 0.96 (0.01) | 0.93 (0.01) |
| eswap | 0.812 (0.021) | 0.535 (0.002) | 0.826 (0.001) | 0.910 (0.002) | 1.7 (0.2) | 3.2 (0.1) | 1.15 (0.00) | 1.17 (0.00) | 1.16 (0.01) | 1.29 (0.03) | 1.12 (0.01) | 1.17 (0.00) | 1.18 (0.00) | 1.31 (0.03) | 0.028 (0.001) |  |  | 0.92 (0.00) | 0.89 (0.00) |
| tjoin | 0.784 (0.020) | 0.541 (0.011) | 0.822 (0.000) | 0.902 (0.000) | 1.9 (0.1) | 5.0 (0.1) | 1.19 (0.01) | 1.24 (0.02) | 1.30 (0.01) | 1.55 (0.05) | 1.16 (0.01) | 1.19 (0.02) | 1.23 (0.01) | 1.36 (0.02) | 0.042 (0.012) |  |  | 0.86 (0.00) | 0.85 (0.01) |
| base | 0.797 (0.018) | 0.525 (0.022) | 0.815 (0.001) | 0.895 (0.001) | 2.3 (0.1) | 5.9 (0.3) | 1.20 (0.00) | 1.26 (0.02) | 1.25 (0.01) | 1.47 (0.03) | 1.15 (0.01) | 1.22 (0.01) | 1.27 (0.01) | 1.43 (0.02) | 0.055 (0.006) |  |  | 0.86 (0.01) | 0.85 (0.01) |

colour

| model | J 10k | J 1k | J d15 | J on3 | out d15 % | out on3 % | 10k lat | 10k ax | 1k lat | 1k ax | d15 lat | d15 ax | on3 lat | on3 ax | pix 1k | dye 1k % | dye d15 % | σ lat | σ ax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DECODE-Plex | 0.796 | 0.341 | 0.830 | 0.882 | 10.0 | 12.5 | 1.31 | 1.30 | 1.56 | 1.65 | 1.39 | 1.43 | 1.46 | 1.47 | 0.069 | 97.5 | 98.7 | 1.02 | 1.13 |
| early | 0.868 (0.005) | 0.597 (0.022) | 0.863 (0.002) | 0.917 (0.003) | 5.9 (0.2) | 7.1 (0.1) | 1.14 (0.00) | 1.11 (0.01) | 1.24 (0.01) | 1.25 (0.01) | 1.26 (0.01) | 1.29 (0.02) | 1.30 (0.01) | 1.27 (0.00) | 0.011 (0.005) | 93.6 (1.9) | 98.7 (0.0) | 0.90 (0.01) | 0.88 (0.01) |
| eswap | 0.861 (0.008) | 0.590 (0.032) | 0.867 (0.002) | 0.919 (0.003) | 5.2 (0.1) | 6.2 (0.1) | 1.12 (0.01) | 1.10 (0.01) | 1.24 (0.02) | 1.27 (0.03) | 1.24 (0.01) | 1.25 (0.01) | 1.27 (0.02) | 1.25 (0.01) | 0.017 (0.005) | 95.0 (0.9) | 98.5 (0.1) | 0.92 (0.01) | 0.90 (0.02) |
| tjoin | 0.864 (0.003) | 0.574 (0.013) | 0.863 (0.000) | 0.917 (0.001) | 6.1 (0.1) | 7.5 (0.1) | 1.13 (0.01) | 1.11 (0.02) | 1.26 (0.01) | 1.28 (0.02) | 1.28 (0.02) | 1.28 (0.02) | 1.30 (0.03) | 1.25 (0.01) | 0.023 (0.009) | 93.3 (2.1) | 98.5 (0.1) | 0.84 (0.01) | 0.85 (0.01) |
| base | 0.862 (0.004) | 0.578 (0.017) | 0.859 (0.002) | 0.913 (0.002) | 6.8 (0.3) | 8.3 (0.2) | 1.14 (0.01) | 1.15 (0.02) | 1.30 (0.03) | 1.42 (0.08) | 1.28 (0.01) | 1.31 (0.01) | 1.33 (0.01) | 1.30 (0.01) | 0.014 (0.005) | 88.7 (5.6) | 98.6 (0.1) | 0.85 (0.01) | 0.88 (0.02) |

Early fusion (eswap -- the bright channel main -- in colour) leads or ties on every number but
bright biplane, where DECODE-Plex reading one frame is 4 -- 5 % closer to its bound; it has a third
to a half of the outliers of late fusion and DECODE-Plex, the least pixelation in biplane, and the
most honest sigma (0.89 -- 0.96; late fusion 0.85, cautious; DECODE-Plex 1.04 / 1.18,
overconfident in z).  Cost: `timing.py`'s table above.

**Flagging outliers from the fit alone** (`flag2.py`, scratch; the plug-in bound and the goodness
from the saved fits, `run_c2.plugin_bound`, `run_c2.netfit_goodness`): AUC of the sigma over the
plug-in bound 0.74 -- 0.87 for the networks (lower than over the truth's bound, 0.88 -- 0.92), of
the goodness 0.61 -- 0.71, of the two together no better than the first.  For DECODE-Plex at 1 000
photons the sigma says nothing (0.46 -- 0.58) and the goodness does (0.71 -- 0.74).  Cost on the
Mac's CPU, 10 000 localizations: the goodness 2 s, the plug-in bound 8.5 s (python loops; the
network 0.1 s on the GPU).

**After filtering** (`filt.py`, scratch; each fitter keeps its best 95 / 90 / 80 % by a flag,
scored on what remains, |z| < 300 nm): keeping 90 % by sigma over bound halves the outliers
(biplane early 1.5 → 0.6 %, colour early 5.5 → 3.4 %) and brings the RMSE / CRLB down (colour
early 1.49 → 1.37) while the robust width hardly moves (1.24 → 1.22) -- it cuts the tails; the
recall falls with the share dropped.  The goodness removes false positives too (precision biplane
DECODE-Plex 0.972 → 0.989, early 0.968 → 0.975), which the sigma does not.  At an equal share kept
the order does not change, and no filter brings a weaker fitter to early unfiltered (biplane density
15: DECODE-Plex keeping 80 % 1.17 / 1.27, early keeping all 1.15 / 1.15).  So: the sigma over the
plug-in bound for positions, the goodness for false positives, both options; neither decides
between models.

**Going on.**  Land.  Then one training round on the winner -- early, the bright channel main --
for longer (6 M samples, 3 seeds), which says whether the bright out-of-focus loss and colour's
crowded outliers are undertraining, and the dye prior chosen from ``S_colour_dyesb``; then the time
model (streaming first: drawing is half an epoch); then real data and the SMAPpy fitter, with the
dyes' ratios from a prefit and the flags as options.

**The dyes known, broadly** (``S_colour_dyesb``: early, each molecule's share drawn around 0.78 or
0.92 with sd 0.06, 3 M samples, 3 seeds; against early's independent counts, standard report):
outliers a quarter fewer (density 15 5.9 → 4.5 %, on 3 frames 7.1 → 5.2 %), the inliers closer to
the bound everywhere (density 15 1.26 / 1.29 → 1.23 / 1.23, 1 000 photons 1.24 / 1.25 → 1.21 /
1.20), the right dye at 1 000 photons 93.6 → 97.5 % with no whole swaps (the exact link: 81 %),
density 50 Jaccard 0.720 → 0.739 -- but the dim found less: Jaccard at 1 000 photons 0.597 →
0.548 (10 000 photons 0.868 → 0.857), and the sigma more cautious (0.90 / 0.88 → 0.85 / 0.82).  So a
broad prior buys dye calls and robustness in crowding with recall of the dimmest, as the priors on
base did (§16, "The last runs"); both stay the user's choice.

**Cheaper checks, and streaming** (2026-11-21).  The goodness of fit is now drawn and summed in torch
(every spot of a chunk of frames with `render.SplineRenderer`, the deviance an image, each
localization's square by a box filter, a background level per frame): the same numbers as the
loop (largest difference 0.008 on 18 000 localizations), about 0.1 s for 10 000 on a GPU shared
with a training -- what the network takes -- and 0.5 -- 1 s on a CPU.  The plug-in bound's spots and
Fisher matrices likewise (`bounds.information`, identical to the loop): about 0.4 s for 10 000 on
the GPU, some 9 s on a CPU, where its cost is the work itself (seven spline evaluations a spot and
channel; splitting the renderer's contraction gave the same numbers no faster).  Streaming
(`draw_batch_2c`'s ``movie_windows``, windows cut from one movie with its crop, as C = 1) halves the
drawing at 4 -- 8 windows a movie; tonight's runs use 8: early (eswap in colour) and tjoin, on-times
mixed over 1 -- 3 frames (``lifetime_max_frames`` 3), window 3, 6 M samples, 3 seeds a context
(``M_*``), after the 6 M runs on on-time 1 (``L_*``).  5 or 7 frames come later, on the best of
these, as for C = 1 (Jonas).

**Twice the training** (``L_*``: the winner, early -- eswap in colour -- to 6 M samples, on-time 1,
3 seeds a context, 1.5 h each on one GPU; against the 3 M runs ``S_*``, standard report, |z| < 500
nm).  Biplane: the bright out-of-focus loss was undertraining -- 10 000 photons 1.17 / 1.18 →
1.12 / 1.15, DECODE-Plex 1.11 / 1.14, a tie now -- and everything else moves a little the same way
(1 000 photons Jaccard 0.564 → 0.595, axial 1.33 → 1.25; outliers on 3 frames 3.5 → 2.8 %; the
sigma 0.96 / 0.93 → 0.98 / 0.97).  Colour gains far less: outliers 5.2 → 4.6 % at density 15
(6.2 → 5.5 % on 3 frames), the inliers 1.24 / 1.25 → 1.22 / 1.23, so its crowded outliers are mostly
not undertraining -- the broad dye prior did more (5.9 → 4.5 %, above).  The right dye at 1 000
photons 95.0 → 91.1 % (seeds' sd 2.7, Jaccard there 0.590 → 0.604 with sd 0.06): within the seeds'
scatter for now, to watch.  So 6 M for a model to keep; the comparisons go on at 3 M where the
order is what is asked.

**The time model** (``M_*``, 2026-11-22: early -- eswap in colour -- and tjoin, window 3, on-times
drawn over 1 -- 3 frames, streamed 8 windows a movie, 6 M samples, 3 seeds a context; early against
``L_*``, which differ only in the on-times and the streaming).  Mixed on-times do not help early
even where molecules stay on: on the on-time 3 set outliers 2.9 against 2.8 % in biplane (colour 5.2
against 5.5 %), precision the same.  They cost the dim molecules on for one frame, which most of the
sets are: Jaccard at 1 000 photons 0.595 → 0.536 in biplane and 0.604 → 0.555 in colour, biplane's
precision there 1.14 / 1.25 → 1.19 / 1.31, and colour's right dye at 1 000 photons 91 → 82 % --
whole swaps, as with the photons linked exactly.  A network that expects a molecule to persist
under-calls one that does not.  Streaming was neutral at C = 1 and is likely not the cause here; it is
not separated.  tjoin gains from the mixed on-times on the on-time 3 set (outliers biplane 5.0 → 4.4
%, colour 7.5 → 6.3 % against the 3 M runs) and still loses to early everywhere but the dye call
(colour 95.8 % at 1 000 photons, DECODE-Plex 97.5 %).  So the model is early, trained on on-time 1:
its three frames already pool a molecule that stays.  Streaming makes a run 1 h 05 instead of 1 h 30
on one GPU.  Next: 5 and 7 frames on early with on-time 1 (and one 3 M run with streaming and on-time
1, which separates the streaming); when a GPU is idle, one run trained on on-time 3 alone, tested
on both on-times (Jonas); then real data and the SMAPpy fitter.

### Real data: the paper's movies (2026-11-22)

netfit's 6 M models (early; eswap in colour; 3 seeds) on the first 200 frames of Figs. 2a and 2c,
against the paper's own DECODE-Plex run of the same frames (prob > 0.6 for both;
`benchmarks/decode/suite/real_c2.py`, `real_c2_compare.py`).  The seeds agree to a nanometre, so one
line each.  Both networks were trained with the same EM camera (DECODE-Plex's `param_run.yaml`), so
neither has an advantage there, though Fig. 2a was taken without EM gain.

**Fig. 2a (colour, 226 columns, the beads' ROI).**  The splitter moved **5.8 rows** since the beads
(§13): read without it, 39 % of netfit's localizations have one of DECODE-Plex's within 150 nm and the
shares fall to 0 and 1; with it, 80 % of ours and 72 % of DECODE-Plex's are paired, offset -0.3 / -12.6
/ +6.3 nm and spread 11 / 22 / 39 nm (x / y / z).  The y offset vanishes at 5.95 rows, the move
DECODE-Plex's own `Test` offset holds; the spread and the goodness are best at 5.5 -- 5.8.  netfit finds
486 molecules a frame, DECODE-Plex 528.

* **Precision across frames**, from molecules found again 3 frames later (the nearest within 100 nm, the
  differences' robust width over sqrt 2; each fitter's own pairs, 9 % of its localizations):
  netfit 21 / 26 / 55 nm, DECODE-Plex 25 / 28 / 68 nm.  Against the reported sigma of the same pairs
  netfit is honest (0.9 -- 1.0) and DECODE-Plex overconfident (1.4 -- 1.8).  Pairs 1 frame apart are
  not a measure for a network reading 3 frames: two of their inputs are the same frames, the errors
  correlated, and netfit came out three times better than its own sigma.
* **Pixelation** (KS of the sub-pixel positions): 0.010 -- 0.015 against 0.031.
* **The dyes**, from independent counts, no prior: the share in channel 1 peaks at **0.785** (most
  molecules, and nearly all the brightest) and at **0.92**, the two dyes the sets were drawn with.
  DECODE-Plex's run keeps one total, so its shares cannot be compared here.
* The goodness of fit: median 8, 90th percentile 41 -- 46.

**Fig. 2c (biplane, a 248-column camera ROI of a 512-wide calibration; `TwoChannels.from_calibration`'s
``shape``).**  Out of both models' training: the frames' median is about 500 photons a pixel in each
plane (the training background 0 -- 300), a live cell crowded to 1 300 -- 2 300 molecules a frame.
netfit finds 2 100 -- 2 350 a frame, DECODE-Plex 1 260; 50 % of ours and 77 % of DECODE-Plex's are
paired, offset +26 / +20 / +25 nm, spread 60 / 52 / 190 nm.  Across 3 frames both are as precise
(31 / 28 / 69 nm against 31 / 29 / 79 nm); netfit's sigma is cautious (0.7), DECODE-Plex's overconfident
(1.2); pixelation 0.022 -- 0.039 against 0.057.  The goodness says neither explains the frames (median
netfit 50 -- 64; DECODE-Plex's localizations drawn the same way 389 on 50 frames).  So Fig. 2c is not a
test of either until a model is trained for its background and density.

**What follows for the fitter**: the registration moves between the beads and the measurement by
whole pixels (a splitter, a day), and an uncorrected move loses half the molecules -- the fitter needs
it, measured from the movie (the shares collapsing to 0 and 1, the goodness, pairing with a prefit)
or typed; and the training's background range is set from the movie (a prefit), not from the beads.

**Fig. 2c's background, trained for** (``R_biplane_early_bg1000_s0``: early, each plane's background
drawn over 0 -- 1 000 photons a pixel where the others drew 0 -- 300, 6 M samples, streamed; 2026-11-22).
The movie's floor -- each pixel's 5th percentile over 100 frames -- is 200 photons at the median and
670 at the 95th percentile, and the earlier models' background stopped at their range's top.  This
one's reaches 325 (95th percentile), but nothing else moves: 2 100 localizations a frame (DECODE-Plex
1 260), 51 % of ours paired with DECODE-Plex's and 75 % of theirs, precision across 3 frames 31 / 28 /
70 nm (the same), the goodness's median 66 (50 -- 64 before).  So the range was not what kept the movie
out of reach.  The light a flat background misses is uneven -- out-of-focus molecules and the cell
-- and without a truth this movie cannot say which fitter is right.  On the simulated sets the wider
range costs the dim molecules: Jaccard at 1 000 photons 0.609 → 0.490, pixelation 0.032 → 0.058, at
10 000 photons 0.832 → 0.856.  The range stays the prefit's; an uneven background in the
simulation is the open question for live-cell movies.

**Photons, defined** (Jonas, 2026-11-22): a biplane's linked N is the molecule's total over both
planes; a colour's total is N1 + N2, each its channel's own count.  SMAP's converted two-channel
calibrations had not met it.  Their PSFs carry the beads' split (Fig. 1c 0.987 and 0.767 at focus,
Fig. 2a 0.901 and 4.18) but no stored share, so Spline 3D 2C read one photon per channel: a linked
biplane N came out 2 N_draw for 1.754 N_draw of light, and a colour's second count was off by its
split.  Each channel's own light at focus is now its share (`psf.signal_integral`, Spline 3D 2C
version 16).  netfit's biplane reports the draw's N times the planes' light (`fit_two.biplane_total`),
and its training records its ranges so; the prefit converts by the same factor, not a fixed 2.
Checked on the 10 000-photon sets, both fitters, matched molecules, |z| < 500 nm:

| | photons / light drawn | colour: total - (N1 + N2) | each channel / truth | lateral / axial σ or bound over the CRLB |
|---|---|---|---|---|
| biplane, Spline 3D 2C | 0.999 | | | 1.016 / 1.053 (reported σ) |
| biplane, netfit | 0.997 | | | 1.006 / 1.006 (bound, 1 frame), 1.006 (3 frames, against crlb_w3) |
| colour, Spline 3D 2C | 1.000 | 1.5e-11 | 0.996 / 1.000 | 1.019 / 1.073 |
| colour, netfit | 1.003 | 1e-7 relative | 1.002 / 1.003 | 0.997 / 0.993 (1 frame), 0.998 / 0.994 (3 frames) |

The bounds use the network's draw counts with the PSFs as drawn, so the convention changes no bound.  On
Fig. 2a the two fitters' counts now agree in both channels (Spline 3D 2C 0.91 and 0.92 of netfit's;
shares 0.794 and 0.798 at the median).  The suite's biplane Spline 3D 2C photons change with it (from
2 N_draw to 1.754 N_draw); none of its scores reads them.

**An uneven background, simulated** (`benchmarks/decode/suite/uneven_bg.py`, 2026-11-22; Jonas: DECODE trained
on a flat background predicted realistic background maps, and Perlin backgrounds in training did not
help).  The density-15 sets with a smooth random field added -- noise filtered 16 pixels wide, scaled to
0 -- 100 or 0 -- 300 photons a pixel, the same in every frame, through the camera -- and, as the control,
the field's mean spread evenly; fitted by the 6 M models trained on a flat background.  Scored against the
flat set's truth and bounds, so the added light's own cost is in every ratio; the comparison is uneven
against even:

| density 15 | Jaccard | outliers | inliers lat / ax | right dye |
|---|---|---|---|---|
| colour, the set | 0.873 | 4.8 % | 1.23 / 1.23 | 98.7 % |
| colour, + even (field 0 -- 100's mean) | 0.858 | 6.0 % | 1.33 / 1.34 | 97.7 % |
| colour, + field 0 -- 100 | 0.783 | 7.7 % | 1.42 / 1.52 | 95.8 % |
| colour, + even (0 -- 300's mean) | 0.553 | 10.5 % | 1.48 / 1.57 | 90.5 % |
| colour, + field 0 -- 300 | 0.524 | 20.5 % | 1.59 / 1.85 | 91.0 % |
| biplane, the set | 0.834 | 1.3 % | 1.11 / 1.16 | |
| biplane, + even (0 -- 100's mean) | 0.814 | 1.8 % | 1.22 / 1.25 | |
| biplane, + field 0 -- 100 | 0.787 | 2.1 % | 1.23 / 1.32 | |
| biplane, + even (0 -- 300's mean) | 0.791 | 3.2 % | 1.39 / 1.50 | |
| biplane, + field 0 -- 300 | 0.656 | 7.7 % | 1.42 / 1.66 | |

The unevenness costs beyond its light: Jaccard 0.03 -- 0.14, outliers up to twice, axial worst.  The colour
model's dimmer channel was trained to 100 photons a pixel, so at 0 -- 300 it is out of range either way.
The network's background at each matched molecule, within the training's range, is 0.89 -- 1.00 of the
truth's (the set's level and the field there) at the median, but follows the field's shape weakly: the
correlation of its excess with the field 0.06 -- 0.44 per channel.  So the local level is roughly right
and the 16-pixel structure is not mapped -- less than DECODE's maps were remembered to be.  A training
with an uneven background is worth one run (a field per sample, smooth at a few widths), judged by this
script's uneven sets against the flat-trained model and on Fig. 2c.

**The background maps themselves** (`uneven_bg.map_errors`: the network's background output at every pixel
against the truth's, the set's level a frame and channel plus the field, on the grid; *gain* the slope of
the map's pattern on the field's, 1 following it, 0 flat):

| | channel | network / truth | rms error | pattern gain (corr) |
|---|---|---|---|---|
| the sets as drawn (flat) | all | 0.98 -- 1.03 | 2 -- 3 % | |
| + even, inside the training's range | biplane, colour 1 | 0.98 -- 1.03 | 1 -- 3 % | |
| + field 0 -- 100 | colour 0 / 1 | 0.85 / 0.97 | 20 / 8 % | 0.06 / 0.05 (0.25) |
| + field 0 -- 100 | biplane 0 / 1 | 0.97 / 1.01 | 13 / 4 % | 0.02 / 0.24 (0.07 / 0.53) |
| + field 0 -- 300 | colour, biplane | 0.52 -- 0.92 | 17 -- 52 % | -0.05 -- 0.08 |

A flat background is predicted to 2 -- 3 %, its pixel noise 1 -- 2 photons; a field barely: the map keeps the
area's level and follows 2 -- 24 % of its pattern.  The colour model's dimmer channel, trained to 100
photons a pixel, saturates above it (0.85 at an even 120).  The training crops are 40 pixels, each one
level: the network never had to map a background inside a crop.  An uneven background in training is
the experiment; these maps are its measure, beside the localizations' scores.

**The C runs so far** (3 M samples, seed 0, standard report; against `S_*`, the same unstreamed):
streamed alone (`P_*`) biplane Jaccard at 1 000 photons 0.569 → 0.533, outliers 1.4 → 1.8 %; colour the
same but the right dye at 1 000 photons 95.3 → 89.3 % -- within the seeds' scatter (0.03, 2.7 points), in
the direction the streamed time-model runs went; at C = 1 streaming was as good.  The job stays streamed,
as one channel's, while more seeds decide (Jonas, 2026-11-22).  On-time 3 alone (biplane): no gain on molecules on for 3 frames (outliers 3.9 against 3.6 %, the
same precision), Jaccard at 1 000 photons 0.477 -- on-time 1 stays.  5 frames: Jaccard no better
(biplane 10 000 photons 0.765 against 0.814; colour equal); their precision waits for the sets' 5-frame
bounds (`standard_c2` has 1 and 3).

**The window's defaults checked against the winners** (2026-11-22): one channel is GMN7C -- the ConvGRU
with norm over 7 frames, on-times 1 -- 3, streamed, 48 features; two channels the `L_*` runs -- early
fusion with the bright channel the grid, 64 features, the ConvGRU over 3 frames, on-time 1, spots over
the whole grid, streamed (being tested at C = 2); the density the data's (LD, HD or UHD, chosen for the sample -- Jonas: settings, not models), targets from 100
photons, the rate decaying every 327 680 samples, 6 M samples.  Still not the winners': GMN7C's
fine-tune to the prefit's on-time.

### End to end on our own data (2026-11-22)

SMAPpy's chain on a real two-colour movie, every step the plugins' and the training window's own:
the movie of 2025-12-11 (M2, microtubules and actin, CF660C and Phalloidin-647, Evolve, EM gain 100,
ROI 150-0-226-512, 200 000 frames) and the beads of the next day (Zenodo 21348704's
`Fig2a-251212_MM_BeadCal_2C_M2`, 11 positions, the raw stacks behind DECODE-Plex's Fig. 2a calibration).

- **The calibration**: *PSF calibration*, dual colour, bead average, the arrangement the microscope
  database's: 47 pairs, 28 used, in 6 s.  The database had M2's Evolve setup unmirrored; its beads (and our
  2026-07 ones: 44 pairs against 7) pair only with the reflected arm mirrored along y -- corrected
  (`microscopes.json`).  Our 2026-07-09 beads do not serve this movie: its spots pair under the 2025-12
  beads' map (a peak 32 times the median of the spot-difference histogram) and not under theirs (2.9);
  the splitter was changed between.
- **The prefit** (Spline 3D 2C, 4 blocks of 50 frames, 19 s): two colours, the brighter half the second,
  shares at 0.795 with a tail towards 0.9; no move -- the shares are narrowest at none (sd 0.051) and every
  2 x 3 pixel move spreads them.  The job: early fusion, independent counts, the window's estimates.
- **The training**: on decodebox by `smappy-decode train`, 6 M samples in 52 min (17 s an epoch);
  its test set at the end Jaccard 0.785, RMSE 40 / 43 nm, KS 0.005.
- **The fit** (*Localize/DECODE*, frames 2 000 -- 2 399, *measure the move* and the goodness on, on this
  Mac): 4.2 frames a second; the move measured -0.25 rows, 0 columns; *after the fit*'s colours kept
  89.5 %.  Against DECODE-Plex's fit of the same frames (the paper's code, `beadpsf-swap`, prob > 0.6 both):

  | | per frame | paired | precision across 3 frames, the 172 000 both found | sigma honesty | KS | dyes' shares |
  |---|---|---|---|---|---|---|
  | netfit, our calibration | 741 | 62 % of ours | 19.0 / 21.6 / 47.6 nm | 1.51 / 1.26 / 1.04 | 0.003 | 0.79, 0.94 |
  | DECODE-Plex | 490 | 90 % of theirs | 21.4 / 25.2 / 51.6 nm | 2.01 / 1.76 / 1.67 | 0.007 | 0.79, 0.94 |

  offset between them 1 / -18 / 20 nm, spread 14 / 23 / 49 nm.  Each fitter's own pairs (not the same
  molecules): netfit 24 / 28 / 70 nm, DECODE-Plex 23 / 26 / 57 -- netfit's include 50 % more molecules and
  a wider z (5th -- 95th percentile -435 -- 619 nm against -350 -- 307).
- **Found on the way**: the biplane photon total imported the compiled fitter, which a training host
  lacks (fixed: `io.calibration.light_at_focus`); the run's record kept the move before it was measured
  (fixed); the prefit cannot pair a moved splitter, and *measure the move* needs a trained network -- the
  spot-pair vote (each channel's spots, the differences' histogram) finds a move with none, a candidate
  for the prefit; the training's test set has no bound at C = 2, so the outliers and inliers' numbers are
  missing from a two-channel *Follow*.

**Which density to train at** (2026-11-22; `D_*_d15_s0`, `D_*_d25_s0` against the default `P_*` at 50 --
three seeds for colour, one for biplane -- on the blink sets at 1, 5, 15, 25 and 50 per 40 x 40 px).
Jaccard, biplane / colour:

| trained at | 1 | 5 | 15 | 25 | 50 |
|---|---|---|---|---|---|
| 50 | 0.832 / 0.877 | 0.821 / 0.889 | 0.824 / 0.865 | 0.801 / 0.834 | 0.758 / 0.725 |
| 25 | 0.861 / 0.898 | 0.846 / 0.903 | 0.836 / 0.866 | 0.810 / 0.833 | 0.745 / 0.701 |
| 15 | 0.873 / 0.908 | 0.852 / 0.905 | 0.835 / 0.865 | 0.802 / 0.823 | 0.719 / 0.676 |

Below the training density's neighbourhood a sparser training finds 3-4 % more and is closer to the bound
(colour lateral RMSE/CRLB 1.16 against 1.26 at density 1); at 15 they tie; above, the dense training wins,
and the sparse one's outliers grow fastest (colour at 50: 38 % trained at 15, 32 % at 25, 27 % at 50;
at density 5 already 1.4 against 1.0 %).  The loss from training too sparse is larger than the gain from
training to match, so **if in doubt, train at the higher density** (Jonas).

Trained at 5 (`D_*_d5_s0`, one seed each), Jaccard biplane / colour 0.894 / 0.922 at density 1, 0.850 / 0.896
at 5, 0.807 / 0.823 at 15, 0.625 / 0.587 at 50.  Only at density 1 does it gain over the 15-trained model
(+0.02 / +0.014, about the seeds' scatter).  At its own density it ties in Jaccard (0.850 against 0.852,
0.896 against 0.905) and has twice the outliers (1.3 against 0.5 %, 2.4 against 1.4 %).  Matching the
training to the data is not the optimum: training somewhat denser than the data does as well or better, and LD
(15) is low enough for sparse samples; no sparser choice is needed in the window.

**The window's densities** (Jonas, 2026-11-22): HD (25), the default, and UHD (50), with any other typed
under *manual*.  25 is never more than about 0.025 in Jaccard behind the best of the trainings at any test
density; LD (15) gained at most 0.012 over it, at density 1 only, within the seeds' scatter, and had more
outliers at every density from 5 up, so it is no longer offered (DECODE's and netfit's jobs still read it).
The window states the density per µm² of the sample (Jonas): HD 1, UHD 2.5, converted to the jobs' 40 x 40
pixels with the movie's pixel size.  At these sets' 117 x 127 nm pixels (23.8 µm²) HD is 24, the 25 tested,
and UHD 59, a little denser than the 50 tested; with smaller pixels the same µm² density is fewer molecules
in 40 x 40 pixels -- the overlap the network must undo is the sample's, set by the PSF's size in nm.

**Streaming at C = 2, three seeds each** (2026-11-22, standard report; S unstreamed on the code of 2026-11-21,
P streamed and U unstreamed on today's): biplane Jaccard at 1 000 photons S 0.564 (0.013), P 0.541 (0.008),
U 0.545; outliers at density 15 1.5 / 1.7 / 1.7 %; colour Jaccard at 1 000 photons S 0.590 (0.032), P 0.590
(0.041), right dye at 1 000 photons S 95.0 (0.9), P 91.6 (2.0), U 83.9 %; every other number within 0.01.
On the same code, streamed and unstreamed agree (biplane 0.541 against 0.545); the gap to S is the code's
or one seed's, not streaming's.  The two-channel job stays streamed, as one channel's.  U's two more seeds
settle the gap as one seed's: U over three seeds is S's -- biplane Jaccard at 1 000 photons 0.560 (0.014)
against 0.564 (0.013), colour 0.590 (0.040) against 0.590 (0.032), right dye at 1 000 photons 90.3 (4.6)
against 95.0 (0.9) %, every other number within 0.01 -- so today's code has not regressed, and P's 0.541
against U's 0.560 (biplane, 1 000 photons) is streaming's cost there at most, within two sd.

**Uneven backgrounds in training, built** (2026-11-22; `SimulationSettings.background_uneven`,
`simulate.background_modulation`): a share of the movies has its background multiplied by
1 + c (2 f - 1), f a smooth field (bicubic from 2 -- 6 knots a side, so 7 -- 40 pixels of a 40-pixel frame)
fixed over the movie, c uniform up to ``background_contrast`` (0.8: a pixel up to 9 times another, Fig.
2c's halves 10).  The mean is kept; two channels share the choice and most of the shape (correlation
0.87 in the median), each its own contrast.  The background term then takes the map as its target, and
the in-memory test set's bound each molecule's own background.  Off by default.  `B_*_uneven_s0` (half
the movies uneven, otherwise P's) train on gpuvm; judged by the background maps' pattern gain
(`uneven_bg.map_errors`, 2 -- 24 % for a flat-trained model) and the flat sets' standard report.

**Fig. 2c's prefit over the movie** (`suite/fig2c_prefit.py`, the window's own prefit with its blocks
placed; the whole movie fetched, 10 000 frames, camera ROI 175-0-248-512):

| frames | background, photons a pixel | photons, mean / sd | per 40 x 40 (SMAP's measure) |
|---|---|---|---|
| 0 -- 1 000 | 0 -- 1 517 | 6 500 / 9 300 | 1.0 |
| 9 000 -- 10 000 | 0 -- 195 | 2 200 / 2 700 | 2.1 |
| across the movie | 0 -- 1 249 | 2 800 / 4 200 | 1.7 |

The cell bleaches: the late frames' background is an eighth of the early ones', their photons typical
of a dye; the early photons look inflated by overlapping molecules.  Two models from the last frames'
estimates (`F_fig2c_last_s0`, flat; `F_fig2c_last_uneven_s0`, half the movies uneven) train on decodebox,
to be fitted on the first and the last 300 frames (`last300.tif`).

**UHD's 2.5 per µm² checked** (`D_*_d59_s0`, 59 in 40 x 40 at these pixels, against the 50 tested): Jaccard
biplane / colour at densities 1, 5, 15, 25, 50: 0.837 / 0.874, 0.822 / 0.882, 0.822 / 0.863, 0.799 / 0.832,
0.757 / 0.723 -- the model at 50 within 0.007 everywhere (0.832 / 0.877 ... 0.758 / 0.725), outliers and
RMSE / CRLB alike (outliers at 50: 11.1 / 26.5 % against 12.2 / 26.9 %).  UHD stays 2.5.

**Uneven backgrounds in training, judged** (`B_*_uneven_s0` against `P_*_s0`, one seed, 3 M samples):

| | biplane P | biplane B | colour P | colour B |
|---|---|---|---|---|
| map's pattern gain, channel 0 / 1 (field 0 -- 300) | 0.06 / 0.10 | 0.92 / 0.45 | 0.05 / 0.17 | 0.56 / 0.85 |
| map's pattern correlation, 0 / 1 | 0.23 / 0.44 | 0.99 / 0.88 | 0.41 / 0.76 | 0.94 / 0.98 |
| uneven density 15: Jaccard | 0.683 | 0.766 | 0.614 | 0.820 |
| uneven density 15: outliers % | 5.8 | 3.6 | 17.3 | 11.6 |
| uneven density 15: RMSE / CRLB lat, ax | 1.45, 1.65 | 1.41, 1.50 | 1.63, 1.77 | 1.56, 1.59 |
| uneven density 15: right dye % | | | 91.6 | 94.9 |
| flat density 15: Jaccard | 0.824 | 0.822 | 0.864 | 0.860 |
| flat 1 000 photons: Jaccard | 0.541 (P's 3 seeds) | 0.595 | 0.590 (3 seeds) | 0.554 |
| flat 1 000 photons: RMSE / CRLB lat, ax | 1.19, 1.37 | 1.22, 1.48 | 1.24, 1.31 | 1.31, 1.34 |

The uneven sets' bounds are the flat truth's, so their ratios hold the added light's cost for both.  The
network trained on uneven backgrounds maps them (it learnt the field, where the flat-trained one returns
nearly a constant) and finds what the flat-trained one loses on them: Jaccard + 0.08 biplane, + 0.21
colour, outliers a third fewer.  On flat data it costs a few per cent of precision at 1 000 photons
(3 -- 8 %, axial worst) and nothing at density 15; biplane's Jaccard at 1 000 photons rose, colour's fell,
each within the seeds' spread.  Colour's channel 0 is mapped at 0.71 of the truth with the field or with
as much light spread flat (0.68) -- the background beyond the training's range, a range matter and not
the pattern's.  Uneven backgrounds are worth having in the window's training (a share of the movies, as
here), with a seed or two more before the default changes; Fig. 2c's F models are the real-data check.

Colour's channel 0 short of its field: the training draws channel 0's background from [0, 100] photons
a pixel (its prefit's range), and the uneven set puts the level, 71, under a field of 61 -- 212 -- most
pixels beyond anything the network saw.  Biplane's ranges reach higher and its channel 0 is mapped at
0.99.  A prefit over a movie with such a background would have the range to match.

**The seeds** (`B_*_uneven_s{0,1,2}` against `P_*_s{0,1,2}`, mean (sd)):

| | biplane P | biplane B | colour P | colour B |
|---|---|---|---|---|
| flat 10 000 photons: Jaccard | 0.789 (0.028) | 0.830 (0.008) | 0.853 (0.009) | 0.854 (0.005) |
| flat 1 000 photons: Jaccard | 0.541 (0.008) | 0.550 (0.033) | 0.590 (0.041) | 0.596 (0.030) |
| flat density 15: Jaccard | 0.824 (0.002) | 0.822 (0.000) | 0.865 (0.001) | 0.862 (0.001) |
| flat density 15: outliers % | 1.7 | 1.8 | 5.3 | 5.7 |
| flat 1 000 photons: RMSE / CRLB lat, ax | 1.19, 1.37 | 1.21, 1.42 | 1.24, 1.31 | 1.27, 1.31 |
| flat density 15: RMSE / CRLB lat, ax | 1.13, 1.17 | 1.15, 1.20 | 1.25, 1.26 | 1.27, 1.29 |
| flat 1 000 photons: pixelation KS | 0.038 | 0.017 | 0.009 | 0.011 |
| flat 1 000 photons: right dye % | | | 91.6 (2.0) | 94.2 (1.0) |
| uneven density 15: Jaccard, seeds 0 / 1 / 2 | 0.683 (s0) | 0.766 / 0.767 / 0.768 | 0.614 (s0) | 0.820 / 0.819 / 0.819 |
| map's pattern gain 0 / 1, seeds 0 / 1 / 2 | | 0.92 / 0.45, 1.03 / 0.37, 0.95 / 0.40 | | 0.56 / 0.85, 0.40 / 0.88, 0.56 / 0.98 |

Over three seeds the flat sets' cost is 1 -- 4 % of precision (axial at 1 000 photons worst, 4 %) and none in
Jaccard or the dye; seed 0's colour Jaccard at 1 000 photons (0.554) was the seed.  The gain on the uneven
set is the same in every seed.  Biplane's pixelation halves on the simulation too.  The default can change.

**Fig. 2c, the models from the last frames' estimates** (`real_c2.py fig2c|fig2c_last`, 300 frames each,
prob > 0.6; F trained on the last 1 000 frames' prefit, bleached -- background 0 -- 195; P and B on the suite's
biplane ranges, R on backgrounds up to 1 000; no DECODE-Plex localizations for the last frames):

| | F flat | F uneven | R bg 1 000 | P (flat) | B (uneven) |
|---|---|---|---|---|---|
| first frames: per frame (Plex 1 257) | 2 224 | 1 967 | 2 915 | 3 047 | 2 937 |
| first frames: of Plex's matched | 0.62 | 0.63 | 0.75 | 0.77 | 0.77 |
| first frames: sigma lat / ax nm | 70 / 310 | 41 / 152 | 42 / 105 | 39 / 102 | 36 / 99 |
| first frames: z 5 -- 95 % (Plex -601 -- 691) | -1046 -- 425 | -300 -- 496 | -873 -- 610 | -889 -- 616 | -647 -- 691 |
| first frames: pixelation KS | 0.031 | 0.022 | 0.030 | 0.033 | 0.016 |
| first frames: goodness median / 90 % | 112 / 1160 | 204 / 1343 | 55 / 298 | 57 / 321 | 100 / 594 |
| last frames: per frame | 554 | 629 | 617 | 641 | 636 |
| last frames: sigma lat / ax nm | 40 / 134 | 34 / 105 | 39 / 98 | 40 / 102 | 37 / 106 |
| last frames: z 5 -- 95 % | -1028 -- 495 | -484 -- 560 | -942 -- 621 | -947 -- 644 | -812 -- 653 |
| last frames: pixelation KS | 0.023 | 0.012 | 0.038 | 0.035 | 0.019 |
| last frames: goodness median / 90 % | 17 / 131 | 35 / 318 | 23 / 101 | 24 / 115 | 30 / 213 |

A model trained for the late, bleached frames fails on the early ones (their background up to 1 500, eight
times its range): fewer molecules, twice the sigma, a fifth fewer of DECODE-Plex's found.  On the late
frames it is no better than the suite's models -- fewer localizations, z as wide -- so the window's prefit
over the whole movie (blocks spread over it) is the right one, not the last frames'.  Trained on uneven
backgrounds, every model pixelates half as much on the real movie (0.016 against 0.033, 0.012 against
0.023) and puts z narrower, B's nearest DECODE-Plex's (-647 -- 691 against -601 -- 691); molecules found and
matched as many.  Their goodness is worse, but the statistic draws a flat background per frame
(`goodness`), and these models put more of the light into their background map and less into the spots
(photons 3 344 against 3 754): it penalises the very thing they learnt.  Judged with the network's own
background map instead (`goodness(background_map=)`, `real_c2.background_map`), goodness median / 90 %:

| | F flat | F uneven | R bg 1 000 | P (flat) | B (uneven) |
|---|---|---|---|---|---|
| first frames, a level a frame | 112 / 1160 | 204 / 1343 | 55 / 298 | 57 / 321 | 100 / 594 |
| first frames, the network's map | 66 / 1325 | 48 / 1385 | 84 / 362 | 87 / 443 | 58 / 241 |
| last frames, a level a frame | 17 / 131 | 35 / 318 | 23 / 101 | 24 / 115 | 30 / 213 |
| last frames, the network's map | 15 / 101 | **2.3 / 11** | 36 / 114 | 19 / 119 | 12 / 53 |

With their own maps the models trained on uneven backgrounds explain the movie best, and the flat-trained
ones get no better (their maps are near a constant, and where they differ they are wrong): on the early
frames B's 90th percentile is half P's; on the late frames the model trained for them with uneven
backgrounds leaves noise alone around its molecules (0 is noise; P 19, B 12, F flat 15).  Fig. 2c's
background is uneven, and a network that has learnt such backgrounds puts the light where it is.  For
the window: uneven backgrounds in training (half the movies, contrast up to 0.8) once the seeds confirm
the flat sets' small cost; and the goodness a network is judged by draws its own map.

**Fig. 2c, the window's job over the whole movie** (`fig2c_prefit.py MOVIE 0 10000 8 50`: 8 blocks of 50
frames spread over it, the window's prefit and job; one seed each, 6 M samples; first frames scored
against DECODE-Plex, 300 frames each):

| | per frame | of Plex's matched | offset x, y nm | sigma lat / ax nm | z 5 -- 95 % | pixelation | last frames: per frame |
|---|---|---|---|---|---|---|---|
| P (suite's ranges) | 3 047 | 0.77 | 26, 19 | 39 / 102 | -889 -- 616 | 0.033 | 641 |
| B (suite's, uneven) | 2 937 | 0.77 | 26, 20 | 36 / 99 | -647 -- 691 | 0.016 | 636 |
| the window's job, flat | 1 241 | 0.41 | 43, 32 | 49 / 190 | -721 -- 540 | 0.014 | 400 |
| the window's job, uneven | 1 828 | 0.62 | 34, 25 | 33 / 110 | -352 -- 449 | 0.014 | 479 |
| + robust photons | 1 714 | 0.51 | 35, 24 | 59 / 247 | -543 -- 563 | 0.017 | 503 |
| + robust photons, uneven | 1 981 | 0.62 | 33, 22 | 34 / 122 | -270 -- 378 | 0.044 | 547 |
| + robust photons, density 50 | 2 972 | 0.68 | 22, 20 | 57 / 198 | -670 -- 584 | 0.017 | 719 |
| + robust photons, density 50, z +-1 um | 3 455 | 0.73 | 24, 20 | 58 / 175 | -671 -- 718 | 0.026 | 807 |

The window's job trained from this movie lost to the suite's hand-set ranges.  Its estimates
(`job.json`): photons 2 836 +- 4 163, background 0 -- 1 249, z -1 862 -- 1 940 (the calibration's
limits), density HD (1 per µm², 24 in 40 x 40).  What each cost:

* **The photons' spread** was a prefit bug.  2 % of the fully-on fits -- overlaps in the crowded first
  frames, up to 50 times the median, past the logl cut -- set the sd alone; the training draws the flux
  normal and cut at 0, so a quarter of its molecules had next to no photons, targets it cannot find.
  Fixed (`prefit.intensity_fully_on`: the fits 3 robust sd above the median in log photons left
  out, 2 480 +- 1 300).  It recovers part of what was lost (0.41 to 0.51 flat), not most.
* **The density** is most of it: at 50 in 40 x 40 the job finds as many as P and matches 0.68, and its
  offset to DECODE-Plex is P's again (the sparse models' extra 10 -- 15 nm is their missing near
  neighbours, matched to the wrong molecule).  The live cell's first frames hold 1 300 -- 2 300
  molecules a frame (the prefit's measured density, 1.7 in 40 x 40, counts only what Spline 3D 2C
  separates, over the whole movie).  The window's choice for such a movie is UHD; the density study had found as much
  ("training too sparse costs more than too dense").
* **The z range**: the blinks' 1st -- 99th percentile, widened by 20 %, reaches the calibration's
  limits; DECODE-Plex's 5 -- 95 % is -601 -- 691 and the prefit's -921 -- 784.  The tails are
  molecules out of focus in a live cell (the logl cut leaves them: -1 695 -- 1 593 at 1 MAD).  At
  +-1 µm the job matches 0.73 and its z is the nearest to DECODE-Plex's.
* Still left: the sigma, half again P's at every density.  The background range (0 -- 1 249, against P's
  0 -- 300) and the photons' distribution (normal, against P's uniform 500 -- 30 000) are being trained.
