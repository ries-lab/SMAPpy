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
