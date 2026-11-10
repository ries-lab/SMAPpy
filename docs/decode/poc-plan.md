# The DECODE proof of concept: the plan, step by step

Status, 2026-11-09: a plan, nothing written.  It details `docs/decode.md` §9,
the published DECODE ported into SMAPpy as `src/decode_smlm/`.  It was written
after reading `TuragaLab/DECODE` at tag v0.10.2, its `spline` package
(`TuragaLab/SplinePSF` v0.10.0), SMAP's `DECODE_training_estimates.m`, and
SMAPpy's calibration loader, DECODE reader, fitter plugin and camera database.
§3 to §8 and §10 of `decode.md` (the new fitter) are not touched here.

What the reading found that §9 does not say, or says otherwise, is in §6 below,
for a decision.  The plan follows §9 as written wherever §6 does not name a
choice.

## 1. Milestones

Each milestone ends in a number or a test.  The dependencies: (a) comes first,
because every equivalence check needs its outputs; (b) and (c) can then run in
parallel; (e) and (h) are torch-free in large part and can start any time.

### (a) The original v0.10.2, running

**What v0.10.2 needs.**  Python 3.7 or 3.8 and torch **1.12.0**, pinned in
`conda/decode/meta.yaml` (not torch 1.7), numpy 1.19 and `spline=0.10.0` from
the `turagalab` conda channel.  `spline` is built for linux-64, win-64 and
osx-64.  Only the Linux build has CUDA.  It is compiled with
`--gpu-architecture=sm_37`, so it carries PTX that the driver JIT-compiles for
any newer GPU.

**Will it run on an RTX 4090 (sm_89)?**  Probably yes.  torch 1.12's cu116
builds carry sm_86 code, and sm_86 code runs on sm_89.  This is the first thing
to try, with one `torch.cuda` matmul and one `spline.PSFWrapperCUDA` forward.

**Where:**
- **Linux with a 4090** (a workstation): everything on the GPU.  This is the
  machine for the reference trainings.
- **Windows with a 4090** (a microscope PC): the network on CUDA, the
  simulation on the CPU (`Hardware.device_simulation: cpu`).  There is no CUDA
  spline on Windows.
- **The Mac:** an osx-64 environment under Rosetta (`CONDA_SUBDIR=osx-64`), CPU
  only.  Enough for the oracle files below, not for training.

**What is made here.**  A script, `scripts/decode/reference_v0102.py`, runs in
that environment and imports only `decode`.  It writes:
1. **The renderer oracle:** `forward_rois` and `forward_frames` of
   `CubicSplinePSF` for about 1000 emitters, including emitters at the frame's
   edges, half outside the ROI and outside the calibration's z range.  It is
   computed on a synthetic SMAP-format `_3dcal.mat` that the script writes
   itself, so it can be checked into `tests/decode_tests/fixtures/`.
2. **The method oracle:** `GaussianMMLoss`, the targets and the
   post-processing (`SpatialIntegration`) on fixed input tensors, also from the
   synthetic file.
3. **The model oracle:** the example package's model run on its first 500
   frames, through the Fit notebook's pipeline: the raw network output (`npz`)
   and the localizations (`h5`).  The example package is too large for git, so
   this goes to the cache (§4).
4. **The reference trainings:** two runs on the 4090 with the example
   package's `param_run.yaml` and calibration, for a fixed number of epochs.
   Their per-epoch metrics go to `benchmarks/decode/results/original_run{1,2}.jsonl`;
   they are in the checkpoint's `log` dict, so no TensorBoard export is needed.

**Done when:**
- the environment is recorded (`scripts/decode/environment_v0102.yml` and
  `conda list --explicit`) and rebuilds;
- fixtures 1 and 2 total under 1 MB, and two CPU runs of the script give them
  bitwise equal;
- the time per epoch on the 4090 is measured and written here, and the epoch
  count E for the reference runs is set from it (each run under about 8 h);
- both reference runs finish.

### (b) The port's skeleton on current torch

**How the code comes over.**
- **Verbatim first.**  The kept modules (§2) are copied as they are at
  v0.10.2, with DECODE's copyright and licence notice, in one commit of their
  own.  Every later change is then a visible diff.
- **The imports.**  `decode.` becomes `decode_smlm.`.  The `spline` import
  goes, replaced in (c).
- **The API breaks of §3**, fixed one commit each.

**The package's frame**, as uipsf's:
- the extra `decode = ["torch>=2.4"]`;
- `tests/decode_tests/` with `collect_ignore` when torch is missing (as
  `uipsf_tests`);
- ruff with uipsf's rules;
- the devices from `uipsf.device`: CUDA, MPS, CPU, and no float64 on the
  device.

**The tests.**  DECODE's own pytest tests of the kept modules come along,
adapted to the new imports.  That is about 3000 of its 5600 test lines: loss,
target generator, post-processing, scale transforms, emitter generator,
emitter set, matching, evaluation, dataset, frame processing.

**Done when:**
- the ported DECODE tests pass on torch ≥ 2.4 on the CPU and on MPS (CUDA by
  hand on the 4090);
- the loss, targets and post-processing on the fixed inputs equal the method
  oracle to 1e-6 relative;
- a test asserts that `import smappy` and plugin discovery leave `torch` out
  of `sys.modules`.

### (c) The renderer swap and its equivalence

**What it is.**  `decode_smlm/simulation/spline_torch.py`, a torch
`CubicSplinePSF` with the original's interface (`forward`, `forward_rois`).
Each emitter's 64 coefficients per voxel are gathered and multiplied with the
monomials, `16*pz + 4*py + px` with px the fastest; the ROIs are added into
the frames with `index_add_`.  The prototype in the old scratchpad is gone; it
is small to write again.

**It reproduces the `spline` kernel, not `evaluate_spline`.**  The two differ
at the ROI's edges (§6, item 6):
- the reference `ref0 = (x0 - 1, x0 - 1, z0)` and `coord2impl`'s negated
  coordinates;
- 1e-10 for a pixel outside the coefficient grid in x or y;
- z clamped to the grid.

**Axes.**  Inside `decode_smlm`, x is the frame's first axis, as in DECODE
(§6, item 4).  The swap to SMAPpy's x happens at exactly two places: the
calibration coming in and the localizations going out.

**The calibration** comes in through `load_spline_calibration`, so it arrives
already un-mirrored (§6, item 5).

**Done when:**
- on the renderer oracle, max |Δ| / max(PSF) < 1e-6 in float32, on the CPU and
  MPS (and CUDA by hand);
- the same SMAP file read raw (DECODE's way) and through
  `load_spline_calibration` plus the swap draws the same frames;
- the renderer agrees with `evaluate_spline` to 1e-6 inside the ROI, with the
  edge difference pinned by a test;
- an epoch's simulation (10 000 frames of 40 × 40) is timed on MPS and on the
  4090 and written here, and is no slower than the original's CUDA `spline`.

### (d) A model the original trained, in the port

**What it is.**  `decode_smlm.io.load_original(param_run.yaml, model.pt)`
reads an original model with its parameters.  The frames go through the Fit
notebook's pipeline:
1. `camera.backward`, from counts to photons;
2. `AutoCenterCrop(8)`;
3. `Mirror2D(-1)` if the model was trained on a mirrored calibration;
4. `AmplitudeRescale`.

The threshold is the notebook's, `raw_th` 0.1 (§6, item 3).

**Done when**, on the model oracle (500 frames):
- **On the CPU:** the network output differs from the original's by at most
  1e-4 of each channel's range.  The localizations are the same, apart from at
  most 0.1 % that sit on a threshold or a tie, and the rest agree to 1e-3 nm.
- **On MPS and CUDA:** the same, but positions agree to 1e-2 nm.
- **The axes:** the port's table in SMAPpy's columns, matched to Spline 3D's
  fit of the same frames, has median offsets |Δx| and |Δy| below 10 nm.  This
  catches x and y swapped and a half-pixel shift.

### (e) The prefit and the job folder

**The estimates.**  `decode_smlm.prefit.estimates(locs, camera, calibration,
area)` is SMAP's `usecurrent_callback` and `setz`, ported with every rule they
apply (§6, item 7).  The prefit itself, Spline 3D on a few hundred frames, is
a function on the source.  This part is torch-free.

**The job folder**, written and read in smappy, torch-free (§5).

**The two check figures.**  The figure functions are module-level, so the
page can use them:
- simulated frames against real ones, in photons;
- the prefit's photons, background and z, with the distributions the training
  will draw overlaid.

**Done when:**
- on a table from `smappy.simulate` with a known lifetime, photon count,
  background range, z range and density, the estimates come back within 10 %;
- a job folder written, read back and written again is byte-identical;
- a job folder moved to another path still trains, because every path in it
  is relative;
- on the example package, our estimates are listed beside its
  `param_run.yaml` in the page.  This is information, not a test.

### (f) The training CLI: progress, stop and resume

**The commands.**  `smappy-decode train FOLDER`, `smappy-decode stop FOLDER`
and `smappy-decode status FOLDER`.

**The loop.**  It is `live_engine_setup`, restructured; `setup_trainer` stays
as it is.  What is added:
- a seeded test set, saved in the folder;
- the RNG states in the checkpoint;
- the stop file, read after every epoch, and SIGTERM, both handled as a stop
  at a checkpoint;
- the checkpoint written atomically (a temporary file, then a rename);
- the model file and a `progress.jsonl` line every epoch;
- the auto-restarts (`GMMHeuristicCheck`, up to 5) written to the progress
  file;
- a lock file (host and pid), so that two runs never share a folder;
- a refusal of a job whose `smappy_version` is not the running one.

**Done when:**
- **A tiny job runs** (16 × 16 frames, 8 batches, 3 epochs, CPU) in under 20 s
  (slow tier), and every progress line is valid.
- **Stop and resume lose nothing:** stopped after epoch 2 and resumed, epoch 3
  has the same loss bitwise as an uninterrupted run's epoch 3 (CPU,
  `num_workers=0`).
- **A warm start** from a model file starts from its weights.
- **The port trains as the original does:** on the 4090, with the reference
  job and E epochs, the port's final test-set Jaccard, lateral and axial RMSE
  and efficiencies lie within max(|run 1 − run 2|, 2 %) of the original's two
  runs (§9.1's third criterion).
- One epoch on the M1 Max is timed and written here.

### (g) The fitter plugin

**What it is.**  `DecodeFit(_FitPlugin)` at `Localize/DECODE`.
- **`engine()`** returns a frame-window engine rather than
  `LocalizationEngine`.  It takes blocks of frames with one frame of overlap on
  each side, so a block's edges see the same three frames as one long run
  (§6, item 18).  It then runs the network and `SpatialIntegration`, and
  returns the localizations in SMAPpy's columns, on the chip.
- **No detection part:** the finder is not used, and its part is hidden
  (`FormState.shown`).
- **`describe()`** reads the model file's metadata without torch (§5).  It
  refuses a model whose calibration hash or camera differs from the data's,
  and warns when the data's photons or background lie outside the training
  ranges.
- **Live fits and *After the fit*** come from `_FitPlugin` unchanged.
- **The tests that every fitter meets:** discovery, `test_control_width`, and
  `gui/assistant_actions` for any button.

**Done when:**
- the plugin's table on the model oracle's frames equals (d)'s;
- 500 frames in blocks of 37 give the same table as one block of 500;
- a live fit of a growing file gives the same table;
- a camera ROI moves the positions by the ROI's corner;
- `describe()` leaves `torch` out of `sys.modules`;
- a model trained on calibration A is refused on data calibrated with B.

### (h) The Train window, here and then over ssh

**Here.**  `gui/train_window.py`, opened by a `Plugin.window`.
- **Prepare:** the prefit runs on the open file, the estimates appear as an
  editable table, and the two figures of (e) are shown.
- **Start:** `sys.executable -m decode_smlm.cli train FOLDER` as a detached
  subprocess.  The jobs are listed in the config directory and attached to
  again after a restart.
- **Follow:** `progress.jsonl` is tailed.  The window draws the curves, the
  RMSE against the CRLB (smappy's spline CRLB at the training's mean photons
  and background) and the time left.  It has *Stop*.
- ***Try on my data*:** the fitter with the latest model on a few frames, its
  localizations drawn over the prefit's.

**Over ssh.**
- the system's `ssh` with `BatchMode=yes`, and the host saved in Preferences;
- the folder copied over with `scp -r` (rsync where present);
- the job started with `nohup smappy-decode train`;
- the progress followed with `ssh host tail -c +OFFSET`;
- the model copied back;
- the versions checked with `smappy-decode --version`.

**Done when:**
- a slow GUI test prepares a job on simulated data, starts the tiny job here,
  sees two progress lines drawn, stops it and resumes it;
- the ssh path runs against a fake `ssh`, a script that runs the command here
  (fast tier), and has been run by hand against a real workstation;
- the panel page passes `test_panel_docs`.

### (i) The benchmark against Spline 3D and the CRLB

**Where:** `benchmarks/decode/`, with its results as `jsonl`, as uipsf's
benchmarks are.

**The data:**
- fixed simulated sets from `smappy.simulate`, drawn once with their seeds
  kept: the example calibration and one astigmatic SMAPpy calibration, over
  density (emitters per 40 × 40 pixels: 1, 5, 15, 25, 50), photons,
  background and z;
- the SMLM Challenge 2016 3D-AS set, from DECODE's gateway (119 MB), which has
  the truth;
- the example package's real frames, with no truth: compared with Spline 3D,
  and rendered.

**The fitters:** Spline 3D, the original DECODE (run by the script of (a), in
its own environment) and the port.

**The metrics:**
- Jaccard, recall and precision;
- lateral and axial RMSE, and DECODE's efficiencies;
- RMSE / √CRLB at low density;
- the z bias per 100 nm bin;
- how uniform the sub-pixel histogram is (a KS statistic on frac(x / px)), as
  the measure of pixelation;
- the frames per second on the 4090 and on MPS.

**Done when** one report holds all of these, and:
- the port is within the two original runs' spread on every metric;
- at a density of 1 per 40 × 40 and at least 2000 photons, DECODE's lateral
  RMSE is at most 1.2 √CRLB and at most 1.1 × Spline 3D's;
- at 25 per 40 × 40, DECODE's Jaccard is above Spline 3D's (the paper's
  claim, reproduced);
- at high photons, the mean Δx and Δy are below 2 nm and Δz below 5 nm.

### (j) The pages

**The fitter's page,** `plugins/docs/Localize/DECODE.md`:
- *What it does*, with Speiser et al. 2021 cited where DECODE is first named;
- *How it works*, in the code's own steps: three frames in; the U-Net
  (`SigmaMUNet`); probability, offsets, photons and their σ; the mixture loss;
  `SpatialIntegration`;
- *In detail*, with the constants: `split_th` 0.6, the scaling rules of
  `autoset_scaling`, the photon filter at 100 photons on the training labels;
- *Differences from the original DECODE*, the §9.1 changes, concretely.

**The Train window's page,** in `docs/panels/`, with *Controls*.

**`docs/decode/`:** `architecture.md` (what is built), `data.md` (the cache
and the gateway) and `conventions.md` (the axes and the mirror, from §6).

**Glossary entries:** U-Net, mixture model.

**The figures** are drawn in seconds without a trained network (§6, item 19):
the renderer against real frames, the prefit's distributions, and
`SpatialIntegration` on a synthetic probability map.

**Done when** `test_plugin_docs` and `test_panel_docs` pass with the figures
(`--slow`), and the rendered pages have been looked at.

## 2. DECODE's modules: kept, adapted, dropped

v0.10.2 is 78 modules and 11 800 lines without its tests (17 400 with them).
"Kept" means copied and changed only for §3.  "Adapted" means its logic is
kept with a changed boundary.  "Dropped" means SMAPpy has it, or the proof of
concept does not use it.

| module | lines | | what happens |
| --- | --- | --- | --- |
| `neuralfitter/models/model_param.py` | 322 | kept | `SigmaMUNet` (and `DoubleMUnet`, its base); `SimpleSMLMNet` dropped |
| `neuralfitter/models/unet_param.py` | 342 | kept | |
| `neuralfitter/utils/last_layer_dynamics.py` | 46 | kept | `grad_mod`, on by default |
| `neuralfitter/models/model_speced_impl.py` | 111 | kept | imported by `models/__init__` |
| `neuralfitter/models/unet_parts.py` | 118 | dropped | used by nothing on the path |
| `neuralfitter/loss.py` | 277 | kept | `GaussianMMLoss`; `PPXYZBLoss` dropped |
| `neuralfitter/target_generator.py` | 420 | kept | `ParameterListTarget`, `DisableAttributes`, `UnifiedEmbeddingTarget`; the `LongTensor` check fixed (§3) |
| `neuralfitter/scale_transform.py` | 324 | kept | |
| `neuralfitter/coord_transform.py` | 69 | kept | |
| `neuralfitter/post_processing.py` | 650 | kept, trimmed | `SpatialIntegration`, `LookUpPostProcessing` and their base, about 350 lines; the consistency post-processing dropped |
| `neuralfitter/em_filter.py` | 78 | kept | |
| `neuralfitter/frame_processing.py` | 132 | kept | `Mirror2D`, `AutoCenterCrop`, `get_frame_extent` |
| `neuralfitter/dataset.py` | 365 | kept | the live, a-priori and inference datasets |
| `neuralfitter/train_val_impl.py` | 133 | kept | |
| `neuralfitter/utils/{processing,progress,probability,dataloader_customs}.py` | 249 | kept | |
| `neuralfitter/train/live_engine.py` | 493 | adapted | `setup_trainer` and `setup_dataloader` kept; the loop is rewritten as the job runner (f), about 300 lines |
| `neuralfitter/train/random_simulation.py` | 61 | adapted | the calibration through smappy, the torch renderer |
| `neuralfitter/utils/logger.py` | 155 | adapted | `DictLogger` writes `progress.jsonl`; `SummaryWriter` optional |
| `neuralfitter/utils/log_train_val_progress.py` | 163 | adapted | the metrics to `progress.jsonl`; seaborn's `distplot` to matplotlib |
| `neuralfitter/inference/inference.py` | 262 | adapted | `Infer` kept for (d); `LiveInfer` dropped (the fitter's engine streams) |
| `neuralfitter/inference/{pred_tif,utils}.py` | 260 | dropped | |
| `neuralfitter/{weight_generator,de_bias,sampling}.py` | 343 | dropped | not on the path (`weight_gen=None`); de-biasing is §3.6's, later |
| `neuralfitter/utils/padding_calc.py` | 20 | kept | the background module uses it |
| `simulation/psf_kernel.py` | 782 | adapted | `CubicSplinePSF` on the torch renderer, about 250 lines new; `DeltaPSF` kept; `GaussianPSF` and the spline's derivatives and CRLB dropped (SMAPpy's fitter has the CRLB) |
| `simulation/{emitter_generator,structure_prior,simulator,noise_distributions}.py` | 503 | kept | the method's statistics |
| `simulation/camera.py` | 241 | kept | `Photon2Camera`, its values from the camera database (§6, item 8) |
| `simulation/background.py` | 236 | kept | `UniformBackground`; the rest trimmed |
| `generic/{emitter,process,slicing,utils}.py` | 1285 | kept | `EmitterSet` runs through everything |
| `evaluation/{evaluation,match_emittersets,metric}.py` | 738 | kept | the validation numbers; their plots to matplotlib |
| `evaluation/{predict_dist,utils}.py` | 298 | kept, trimmed | `px_pointer_dist` and the helpers the logs use |
| `utils/param_io.py`, `types.py`, `reference_files/` | 278 | kept | `RecursiveNamespace`, `autoset_scaling`, `reference.yaml` as data |
| `utils/hardware.py` | 59 | adapted | `mps` accepted (§3) |
| `utils/checkpoint.py` | 63 | adapted | `map_location`, RNG states, no `log`, atomic writes |
| `utils/model_io.py` | 121 | adapted | the model file of §5; the original's `model.pt` read for (d) |
| `utils/calibration_io.py` | 38 | replaced | `load_spline_calibration` and the axis swap, about 40 lines |
| `utils/{emitter_io,frames_io,loader,emitter_trafo}.py` | 637 | dropped | SMAPpy reads frames and writes tables |
| `utils/{bookkeeping,dependency,notebooks,deprecate_warning,example_helper}.py` | 269 | dropped | `example_helper`'s download moves to `scripts/decode/fetch_examples.py` |
| `plot/`, `renderer/` | 694 | dropped | |
| `test/` | 5600 | ~3000 ported | to `tests/decode_tests/` |

**In all:** about 7 500 lines kept, about 1 000 adapted, about 600 new (the
renderer, the job runner, the model file, the prefit), about 3 300 dropped.

## 3. API breaks between torch 1.12 and now, found by reading

Most breaks of the torch 1.7 era were already fixed in v0.10.2: it uses
`rounding_mode` and `nonzero(as_tuple=False)`.  What is left:

1. **`torch.load`**, at `utils/checkpoint.py:49`, has no `map_location`, so
   a CUDA checkpoint cannot be loaded on a Mac.  (Read first as a
   `weights_only=True` failure too, the default since torch 2.6; tried on a
   4090 run's checkpoint, it is not: the logger's dict is lists of floats.)
   `model_io.py` loads a plain `state_dict` and works.
2. **Legacy type classes as type checks.**  `isinstance(frame_ix,
   torch.LongTensor)` (`target_generator.py:219`) and `torch.IntTensor` or
   `LongTensor` (`generic/slicing.py:26`) are false for a CUDA or MPS tensor.
   Written as `dtype` checks instead.
3. **No MPS anywhere.**
   - `hardware._specific_device_by_str` refuses anything but `cpu` and
     `cuda`.
   - `live_engine` falls back to the CPU when CUDA is missing.
   - `Infer.get_max_batch_size` asserts CUDA.
4. **Sampling on MPS is not verified.**  DECODE's camera draws Poisson, then
   Gamma for the EM gain (`noise_distributions.py`).  Whether MPS implements
   `poisson` and `_standard_gamma` is to be checked.  If it does not, the noise
   is drawn on the CPU; it is a small part of the cost.
5. **`torch.meshgrid` without `indexing=`** (`coord_transform`,
   `target_generator`, `psf_kernel`) only warns, and its behaviour, `ij`, is
   kept.  `indexing="ij"` is written out.
6. **The DataLoader on macOS starts its workers by *spawn*,** so the dataset
   must pickle.  It does without the `spline` object, whose `__getstate__`
   workaround goes.  `pin_memory=True` warns on MPS, so it is set only for
   CUDA.
7. **Plots.**  seaborn's `distplot` (in the logs and `evaluation.py`) was
   deprecated in seaborn 0.11 and is gone from newer releases; the plots use
   matplotlib.  The `seaborn==0.10` pin goes.
8. **Old pins.**  `scipy<=1.8` and `tifffile<=2022.4.8` go with the modules
   that needed them.  `importlib.resources.open_text` becomes `files()`.
   `gitpython` (for `decode_state`) and `deprecated` are dropped.
9. **TensorBoard.**  `SummaryWriter.add_graph` traces with the JIT.  It is
   already inside a `try`, and TensorBoard is optional (§9.4).

Not found: `np.float`-style aliases, `torch._six`, removed linalg functions.

Found only by running it, in (b):

10. **A CPU tensor indexed with GPU indices.**  `GaussianMMLoss` indexes its
    pixel centres (a CPU tensor) with the detections' indices, which are on
    the model's device.  torch 1.12 accepts a CUDA index into a CPU tensor;
    torch 2.14 refuses it.  DECODE's own tests miss it: the mixture-model
    loss's test class has no GPU test.  The method oracle on MPS found it.
11. **numpy functions on torch tensors** (`np.sqrt`, `np.unravel_index`) are
    deprecated in numpy 2: a warning per call, two thousand in the tests.

## 4. Test data and the tiers

**The data:**
- **DECODE's example packages**, through its `gateway.yaml`, still served by
  EMBL's ownCloud (checked 2026-11-09).
  - `experimental_data_workflow`, 1.35 GB: real frames, beads, a
    `_3dcal.mat`, a prefit, a trained model and its `param_run.yaml`.  This is
    the model oracle and the real data of (d) and (i).
  - `smlm_challenge`, 119 MB: the 2016 challenge's 3D sets with their PSF.
  - `colab_uhd_2020-07-23`, 42 MB: a calibration and parameters for training.
  - All of them are fetched by `scripts/decode/fetch_examples.py` into
    `~/.cache/decode_smlm/` (`$DECODE_SMLM_DATA`), checked against the
    gateway's sha256, and never put in git.  Tests that need them skip
    without them, as `SMAPPY_TEST_CAL` does.
- **Simulated data from `smappy.simulate`**, made in the test: tables for the
  prefit, frames for the fitter and the benchmark.
- **Our real data:** `docs/decode.md` names only dual-colour data
  (`~/datalocal/decode_plex`).  A single-channel 3D data set with its bead
  calibration is needed for (i) (§7).

**Fast tier** (each under 1 s, CPU):
- the renderer against its oracle;
- the method oracle (loss, targets, post-processing);
- the model file written and read, and its metadata read without torch;
- the job folder round trip;
- the `progress.jsonl` writer and reader;
- the prefit on simulated tables;
- the ssh path with a fake `ssh`;
- `describe()` without torch;
- the ported DECODE unit tests.

**Slow tier** (4 to 20 s):
- the tiny job: three epochs, then stop and resume;
- the Train window test;
- the model oracle through `load_original` and through the plugin (cache
  only);
- the block and live equivalence of the engine;
- the pages' figures.

**Long tier** (over 20 s):
- a short training (about 20 epochs on 40 × 40, MPS or CUDA) whose test-set
  Jaccard rises above a set value, to be measured in (f).

**By hand, in `benchmarks/decode/`:**
- the reference trainings, and the port's;
- the benchmark (i);
- the GPU checks on the 4090 and the M1 Max.

## 5. The interfaces the new fitter reuses

Everything here is free of DECODE's internals.  The new fitter of §10 adds a
method behind the same four things.  The torch-free parts live in smappy
(`src/smappy/training/`); the method lives in `decode_smlm`.

**The job folder:**

    job.json          schema 1: method ("decode-v0.10.2"), smappy_version, created, the data it was
                      prepared on (path, frames, camera ROI), calibration (file, sha256 of the coefficients),
                      camera (database entry and its values), ranges (photons, background, z, density),
                      base_model, seed, device ("auto")
    method.yaml       the method's own settings: here DECODE's param.yaml, paths relative to the folder
    calibration.h5    the calibration as smappy loaded it (`save_spline_calibration`), so the far end
                      reads one format whatever the user had
    test_set.pt       the fixed validation set, drawn once from `seed`
    ckpt.pt           checkpoint: weights, optimizer, scheduler, epoch, RNG states (weights_only-safe)
    model.safetensors the model file, rewritten every epoch
    progress.jsonl    one line per event
    samples/          a test frame, its truth and the predictions every k epochs (npz)
    STOP              present = stop at the next checkpoint
    lock.json         host, pid, started; removed on exit
    train.log         stdout and stderr

**`progress.jsonl`.**  One JSON object per line, appended and flushed, so a
reader can follow it over `tail`.  Every line has `kind`, `time` (ISO) and
`epoch`.
- `start` (versions, device), `epoch`, `sample` (the path of an npz),
  `restart` (why), `stopped`, `finished` and `error` (a message).
- An `epoch` line holds `loss_train`, `loss_test`, `lr`, `seconds`, and
  `metrics`, a dict of the method's own values (DECODE's: `jaccard`,
  `rmse_lat_nm`, `rmse_ax_nm`, `rmse_vol_nm`, `eff_lat`, `eff_ax`).
- A reader ignores keys it does not know, so a new method adds metrics
  without changing the window.

**The model file.**  It is safetensors: an 8-byte length, a JSON header, the
raw tensors.
- **Metadata** goes in the header's `__metadata__`, as one JSON string:
  - the method and its architecture, with its parameters;
  - the input (frames per window, channels, `frames_mirrored`) and the output
    channels;
  - the scaling;
  - the camera;
  - the calibration's sha256;
  - the training ranges;
  - the base model, the epochs and the smappy version.
- **The header reads without torch** in about 20 lines of Python, which is
  what lets `describe()` check a model on every form change.
- **No pickle.**  The format is written and read by our own short code, about
  80 lines; the `safetensors` package is used only in a test, as a check.
- This is §9.5's "weights and parameters as JSON, no pickle", made concrete
  (§6, item 16).

**The fitter.**  The plugin talks to the method through two calls:
`decode_smlm.load_predictor(path, device) -> Predictor`, and
`Predictor(frames_photons: (n, h, w) float32, first_frame) -> columns`.  A
`Predictor` also gives `window` (the frames it needs on either side) and
`metadata`.  The plugin's engine handles the blocks, the overlap, the chip
offset and the columns.  The new fitter brings a different `Predictor`, chosen
by the model file's `method`.

## 6. What §9 gets wrong or leaves open

To decide; none of it has been changed in `decode.md`.

1. **The torch version.**  v0.10.2 pins torch 1.12.0 and Python ≥ 3.7 and
   < 3.9, not torch 1.7.  It probably runs on a 4090 (§1a).
2. **The entry point, and which DECODE is the baseline.**
   - v0.10.2 trains through `decode.neuralfitter.train.live_engine`; there is
     no `train/train.py`.
   - SMAP's `DECODE_training_estimates` calls `decode.neuralfitter.train.train`,
     and its yaml has `Simulation.PSF_min_mod_max`, which v0.10.2 does not
     know.  So the lab's users ran a later development DECODE, not v0.10.2.
   - **Question:** is v0.10.2 the baseline, as §9 says, or the version the
     lab used?
3. **The threshold.**
   - §9 says `raw_th` 0.5.  That is what the training's validation uses
     (`reference.yaml`).
   - The published Fit notebook infers with `SpatialIntegration(raw_th=0.1)`.
   - The plan keeps both: 0.5 for validation, 0.1 as the fitter's default (a
     setting), and `prob` to filter on afterwards.
4. **The axes.**
   - DECODE's x is the frame's first axis, the rows: `tifffile` gives
     (n, rows, columns), and `Offset2Coordinate` and the kernel index x
     first.  A SMAP spline's first axis is also the rows.
   - So DECODE's x is SMAPpy's y.
   - `io/decode.py` reads `xyz[:, 0]` as `x_nm`, unswapped.
     `test_foreign_formats.py` builds its file by hand and cannot catch this.
   - (d) settles it on the example package.  If it is confirmed,
     `io/decode.py` gets a small fix of its own.
5. **The mirror.**
   - SMAP builds the spline of an EMCCD from x-mirrored beads.
   - DECODE takes that spline as it is, and the Fit notebook flips the frames
     to match: `Mirror2D(dims=-1)`, commented "you might need to comment this
     line out".
   - SMAPpy un-mirrors the spline on load instead.
   - So a model the port trains needs no flip.  A model the original trained
     on such a calibration needs the frames flipped.
   - The model file records `frames_mirrored`.
6. **"The PSF `spline` draws" is not `evaluate_spline`.**
   - Where they differ: outside the coefficient grid in x and y (1e-10 against
     the edge value, clamped), and in how the ROI is placed (`ref0 = x0 - 1`
     against centring by `floor((nx + 1 - R) / 2)`).
   - The port reproduces `spline`, checked against oracle files from the
     original's environment.  `spline` does not install on an arm64 Mac.
   - `spline`'s CPU kernel loops `j < npx` in `kernel_roi`; this is harmless
     for its square ROIs.
7. **The prefit table differs from SMAP's code:**
   - **Density.**  `emitter_av` is not the measured density.  SMAP shows the
     density and then sets `emitter_av` from a choice, LD / HD / UHD = 15 / 25
     / 50 (`finalizejson`).  Which should the port do?
   - **Lifetime.**  It is `make_statistics2`'s `lifetime.mu` less 1, at least
     1.  The estimator behind `lifetime.mu` is to be read from SMAP.
   - **Intensity.**  [1, 0.2] × mean photons of the grouped localizations /
     lifetime, × 1.3 when `lifetime.mu` < 0.8.
   - **Background.**  The 1st to 98th percentile of the ungrouped
     localizations, widened by 30 % of that range on each side, its lower end
     at least 0.9 × the 0.05th percentile.
   - **z.**  The 1st to 99th percentile, widened by 20 % on each side,
     clamped to the calibration's `fminmax` range; ±750 nm without z.
   - **Read noise.**  Hard-coded: 74.4 e⁻ with EM gain, 1.5 e⁻ without.
     `spur_noise` stays the yaml default.
   - **The table it reads.**  SMAP reads the fit already loaded, grouped and in
     the ROI, not a new Spline 3D prefit.
   - **2D data without a calibration.**  SMAP draws a Gaussian spline from the
     widths (`make2Dastig`).  Out of scope here.
8. **The camera database has no `spur_noise` and no `qe`.**
   - The defaults: QE 1, because photons are detected photons, as everywhere
     in SMAPpy.  Spurious charge 0.0015 with EM gain and 0 without, as the
     notebook suggests.  Both are settings.
   - DECODE's camera is one EMCCD model.  An sCMOS is em_gain `None` with one
     read noise for the chip; a per-pixel noise map does not fit it.
9. **Fresh data per epoch, not per batch.**  `mode: acquisition` simulates
   10 000 frames once per epoch, and the batches are drawn from those.
   `decode.md` §1's "drawn afresh for every batch" is true of `mode: samples`.
   Nothing to change for the port, but §1's wording is DECODE-Plex's.
10. **The test set is neither seeded nor kept.**  It is drawn again in every
    setup, a resume included, and no RNG state is saved.  §9.3's resume and
    §9.4's comparable curves need the seed and the saved set of (f).  This
    changes the training's plumbing, not the method.
11. **Auto-restart.**  A pathological loss restarts the training from scratch,
    up to 5 times.  That has to show in `progress.jsonl` and in the window, and
    must not be mistaken for a resume.
12. **The models the original saves.**  `LoadSaveModel` writes `model_0.pt`,
    `model_1.pt`, ... once an hour, and the checkpoint every epoch.  The port
    writes one model file every epoch (§5).
13. **The size.**  §9.1's "about 400 kB of Python in 77 modules": 78 modules,
    11 800 lines without the tests, 17 400 with them.  §5's "about 21 000
    lines on Hydra, Lightning 1.8, pydantic < 2" describes a later tree;
    v0.10.2 uses none of the three.
14. **The z range and its scaling.**  `autoset_scaling` sets `z_max` from the
    upper bound only (× 1.2).  An asymmetric range from the prefit whose lower
    bound exceeds 1.2 × the upper puts training targets outside the scaled
    range.  Whether the z head then saturates is to be read in
    `SigmaMUNet`'s activations.  The prefit should warn, or keep the range
    symmetric.
15. **MPS:** §3, items 3 and 4.
16. **The model file.**  "JSON, no pickle" needs a concrete format, because
    `torch.save` is pickle.  safetensors (§5) gives that, with metadata that
    reads without torch.
17. **The crop at inference.**  `AutoCenterCrop(8)` cuts the frames to a
    multiple of 8, losing up to 7 pixels at the edges.  The plan keeps it for
    (d).  Afterwards, the fitter pads instead and drops the padded
    detections.  This changes only edge pixels and is stated on the page.
18. **The ends of a stack.**  `InferenceDataset` repeats the first and last
    frame to make their three-frame windows.  A block engine needs one frame
    of overlap to match a single run (g).
19. **The pages' figures cannot show the network.**  The repository may hold
    no trained model (it is large, and data stays out of git), and a page's
    figures must take seconds.  So the page shows the renderer, the prefit and
    the post-processing; the network's results are in the benchmark's report.
20. **The licence of the copied files.**  Until the relicence is written
    down, the files under `src/decode_smlm/` keep DECODE's GPL notice.  Can
    they then be in a released wheel (§9.1 says the relicence is for the BSD
    wheel)?

## 7. Decided, and still open

Decided (Jonas, 2026-11-10):
- **The machine:** (a) and the reference trainings run on a Linux
  workstation with a 4090, reached over ssh from the Mac.
- **The baseline:** v0.10.2, as a first step.  Later, DECODE's versions on
  git (not DECODE-Plex) are compared, to see what a later one added; one may
  replace v0.10.2 if it has new features or was tuned.
- **The licence:** the copied files go into the package, and into releases,
  for now.

Still open:
- The prefit's density: measured, or LD / HD / UHD as in SMAP (§6, item 7)?
  Suggested: measured, with the nearest of the three preselected.
- Which single-channel 3D data set with its bead calibration is the real data
  for (i)?

## 8. The Linux workstation, as found (2026-11-10)

`pc-ries100` (ssh alias `decodebox`): Ubuntu 26.04, glibc 2.43, an RTX 4090
(24 GB, driver 595, CUDA 13.2), 32 cores, 122 GB RAM.  The disk is 96 % full,
**76 GB free**: the runs keep to a few GB, and anything larger is asked first.
Another user works at its desktop (FPGA tools, the GPU idle).  The Claude app's
processes are kept off the local network by macOS even with *Local Network*
allowed, so ssh goes through a connection Terminal.app opens and shares
(`ControlMaster` in `~/.ssh/config`; `ssh -fN decodebox` once per login).

**v0.10.2 is already installed**, `~/miniforge3/envs/decode_env`: DECODE 0.10.2,
torch 1.12.0 (CUDA 11.3, sm_86 code, runs on the 4090), spline 0.10.0, numpy
1.19.5, Python 3.8.  It needs one variable on glibc ≥ 2.41, which refuses
libraries that ask for an executable stack (`libtorch_cpu.so` does):
`GLIBC_TUNABLES=glibc.rtld.execstack=2`.  No file is changed.

**spline 0.10.0's GPU renderer collapses above ~2000 emitters a call.**
`CubicSplinePSF.forward`, 200 frames of 40 x 40, 26 x 26 ROIs, best of two
(`~/decode_poc/bench_spline.py`):

| emitters | 0.10.0 CPU | 0.10.0 GPU | 0.11.2.dev0 GPU |
|---:|---:|---:|---:|
| 2 000 | 52 ms | 1.1 ms | 0.7 ms |
| 5 000 | 110 ms | 1 676 ms | 1.1 ms |
| 20 000 | 425 ms | 10 684 ms | 2.5 ms |
| 50 000 | 1 040 ms | 26 777 ms | 5.9 ms |

GPU and CPU agree to 2e-7 in both.  0.10.0 starts one block per emitter, each
launching a child grid (dynamic parallelism); above the device runtime's
2048 pending launches the rest go through a slow virtualised pool.
SplinePSF's 2c7f598 (2023-06-21, "change forward roi computation to non
dynamic parallelism") replaced it with one flat grid, and 47c7dd3 did the
same for the derivatives.  An epoch of 10 000 frames at 20 emitters is
~200 000 emitters: about 110 s of GPU rendering with 0.10.0.  0.10.0 also
fails DECODE's own `test_psf_kernel.py` on the 4090 ("an illegal memory
access", in `test_many_drv_roi_forward`).

**So the baseline runs with SplinePSF 1f85d10** (0.11.0.dev0, 2023-06-22, on
`Haydnspass/SplinePSF`, the fork where the fix was made; not on `TuragaLab/SplinePSF`):
both fixes, still Python 3.8, and from v0.10.0 to it only `spline_psf_gpu.cu`
and the build change -- the Python binding and the CPU code are v0.10.0's.
It is `~/miniforge3/envs/decode_fast`, a clone of `decode_env` with spline
swapped, built in `spline_build` (CUDA 12.4's nvcc, gcc 12, sm_86 and sm_89)
from a clone in `~/decode_poc/SplinePSF`; `decode_env` is untouched.  With it
DECODE 0.10.2 renders 50 000 emitters in 6.0 ms, agrees with its CPU to
3e-7, and passes `test_psf_kernel.py` (45 passed, 1 xpassed).  The port's
renderer is timed against this.

**A later DECODE is also there**, for the comparison of versions (§7):
`~/miniforge3/envs/decode_dev`, DECODE 1.0.0.dev0 from `ries-lab/DECODE_Internal`
(`~/Code/decode/DECODE_Internal`, branch `4pi-simulation`, 2850 commits,
uncommitted changes in `psf_kernel.py`, `model.py`, `loss.py`,
`setup_cfg.py`), torch 2.1.2 (CUDA 12.1), spline 0.11.2.dev0 built from
`GabinAgbale/SplinePSF` (`~/Code/SplinePSF`).

### (a) as run, 2026-11-10

- **The environment** is recorded in `scripts/decode/`: `environment_v0102.yml` (what it is
  made of), `v0102_linux64.explicit.txt` (`conda list --explicit` of `decode_fast`) and
  `setup_v0102.sh`, which rebuilds it with the spline swap.
- **Fixtures 1 and 2**, `tests/decode_tests/fixtures/v0102_renderer.npz` (539 KB) and
  `v0102_method.npz` (66 KB), from `scripts/decode/reference_v0102.py oracles` on the CPU with one
  thread; two runs give them bitwise equal (their sha256 in `v0102_oracles.json`).  The renderer's:
  1032 emitters on a 32 x 24 frame, 16 frames, from a synthetic 9 x 9 x 15 calibration whose
  coefficients are multiples of 2^-12 (exact in float32, so they compress); ROIs and frames for
  all, the derivatives for every 7th (148, every kind of emitter among them).  The GPU agrees
  with the CPU to 1.6e-7 on all four.  The method's: the loss and its gradient, the targets
  before and after rescaling, `SpatialIntegration` at raw_th 0.1 and 0.5.
- **The model oracle**, `reference_v0102.py model`, in `~/.cache/decode_smlm/v0102_model/`:
  the package's model on its first 500 frames (176 x 176 after the crop), 137 351
  localizations, and the raw output of the first 100 frames (124 MB; all 500 would be 620 MB).
  CPU and GPU: the same localizations, x and y within 1.5e-5 px, z within 0.023 nm; the raw
  channels within 8e-5.
- **The Fit notebook ignores `frames_meta.yaml`'s camera.**  It calls
  `autofill_dict(meta['Camera'], param.to_dict())`, which fills the camera's four values *with*
  the whole parameter tree instead of putting them into its `Camera`, so `param.Camera` stays
  the training's: EM gain 100, where the frames were taken at 50 (`frames_meta.yaml`, and
  "EM50" in the original file name).  The photons it reports are half the camera's.  The oracle
  keeps this -- it is what DECODE's users got -- and the port's Fit step must not copy it.
- **The package's `param_run.yaml` (2020-07) does not load in v0.10.2 as it is.**  v0.10.2
  fills what a file leaves out from its `reference.yaml`, which adds StepLR's `step_size` to
  the file's ReduceLROnPlateau, and the scheduler refuses it.  The reference runs use v0.10.2's
  own StepLR (10 epochs, 0.9); everything else is the package's: 32 768 frames an epoch
  (`pseudo_ds_size`), 50 emitters a frame on average, lr 6e-4, weight decay 0.1, 1000 epochs
  (`scripts/decode/make_params.py`).
- **Model files.**  The live engine saves after every epoch with no metric, so `LoadSaveModel`
  moves to the next name each time, through three (`max_files=3`): `model_0.pt` to `model_2.pt`
  are the last three epochs, not hourly snapshots.  About 50 MB a run with the checkpoint.
- **The time per epoch on the 4090: 26 s** (the simulation of 1.64 million emitters 0.6 s of
  it, 512 batches of 64 at 0.03 s), so 1000 epochs are 7.2 h, and **E = 1000**, the package's
  own.  Two runs at once take 46 s an epoch each, no faster in all than one after the other.
- **The reference runs** started at 11:52 (`scripts/decode/train_reference_v0102.sh
  original_run1`, `original_run2`, in `~/decode_poc/runs_reference/`), together, to end about
  02:00 on 2026-11-11.  Their per-epoch metrics are in each checkpoint's `log` (`eval/jac`,
  `eval/rmse_lat`, ..., the loss channels); they go to
  `benchmarks/decode/results/original_run{1,2}.jsonl` when they end.

### (b) as done, 2026-11-10

- **The port is `src/decode_smlm/`, 8 800 lines, and `tests/decode_tests/` (35 files).**  The first
  commit (e7376e7f) is the 58 files and 34 test files of v0.10.2 unchanged; every change since
  is a commit of its own: the rename and the dropped modules, then §3's items 1, 2, 3, 5, 6, 7,
  8 and 9, the two of §3 that only running found (10, 11), and ruff.  DECODE's licence is
  `src/decode_smlm/LICENSE.txt`, with `NOTICE.md` (version, commit, authors) and an entry in
  `THIRD_PARTY_NOTICES.md`.
- **Dropped beyond §2's table:** the consistency post-processing (scikit-learn), `log_frames`
  (DECODE's plot package; its figures went only to TensorBoard), `SCMOS` (unfinished upstream),
  `EmitterSet`'s `.h5` and `.csv` (`.pt` stays), the `deprecated` package, `LiveInfer` and the
  inference CLI.  `CubicSplinePSF` raises until (c).
- **The frame:** the extra `decode = ["torch>=2.4", "tqdm>=4.60"]`; TensorBoard optional (the
  live engine then logs to the `DictLogger` alone); devices CUDA, then MPS, then the CPU
  (`utils.hardware.default_device`, `DECODE_DEVICE` overrides), and a parameter file's `cuda`
  falls back to what the machine has; `src/decode_smlm/ruff.toml`, uipsf's rules, with DECODE's
  remaining findings listed as ignores.
- **Item 4, sampling on MPS:** torch 2.14 draws Poisson and Gamma there; the moments, and
  `Photon2Camera`'s output, agree with the CPU's.
- **The tests:** 460 pass, 34 skipped (DECODE's own skips, CUDA-only tests, one asset
  download), on torch 2.14.1, numpy 2.5, Python 3.12, on the CPU and on the M1 Max's GPU.
  DECODE's GPU tests run on whichever GPU there is.  Not yet run on CUDA with current torch.
- **Against the method oracle** (`tests/decode_tests/test_port.py`): the targets are bitwise
  the original's; the loss, its gradient and the post-processing at raw_th 0.1 and 0.5 within
  1e-6 of the original's largest value, on the CPU and on MPS.
- **`import smappy` and plugin discovery load neither torch nor `decode_smlm`** (a test).
- **Test changes, for an offline suite:** downloads skipped (`asset_handler`, `web`); a local
  network for `torch.hub`'s UNet; files to `tmp_path`; `test_nms` at 1e-6 (a float32 step:
  0.50099993 against 0.50099999); DECODE's `test_load_init` passed a path as the device.

