# DECODE-Plex in SMAPpy: the port, and step 5's plan

Status, 2026-11-19: planned, agreed with Jonas the same day; the port started.  It details
`docs/decode.md` §10, step 5 (C = 2, biplane and colour), as `poc-plan.md` detailed §9 for C = 1.
Written after reading `ries-lab/DECODE-Plex` at 6138472 (2026-10-20) and its data release
(Zenodo 10.5281/zenodo.20088009, record 21348704).

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

**To check before drawing:** that SMAPpy reads these SMAP two-channel calibrations and `_trafo.mat`
as one dual calibration (`load_dual_color_calibration`), mirror included; if not, the reader is
added first, and is tested against DECODE-Plex's offset maps (§5, item 2).

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
