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

### (f2) The batch size fitted to the GPU -- the first optimisation

**Outcome (2026-11-11): measured and rejected.**  The choice works (`decode_smlm/batch.py`), but
batch 256 needs three to five times the epochs of 64 to reach the same test loss, and is at most
1.4 times as fast an epoch; several trainings at once through MPS give 1.4 times the throughput
without changing one (§8, "(f) and (f2)").  What follows is the plan as it was written.

**Why first.**  Every later study trains (the benchmark (i), the training's ranges, clipping,
DECODE's later versions), so anything that makes a training faster is worth having before them.
The package's batch of 64 frames of 40 x 40 leaves a large GPU partly idle.  On gpuvm's H100 the
network's step (forward, backward, AdamW; TF32; 2026-11-11) costs per frame:

| batch | a step | a frame |
|---:|---:|---:|
| 64 (the package's) | 16.2 ms | 253 us |
| 128 | 22.9 ms | 179 us |
| 256 | 42.7 ms | 167 us |
| 512 | 81.4 ms | 159 us |

So at 64 the H100 runs at about 60 % of its rate per frame, and at most 1.6 times the frames a
second are there to gain on the GPU's part of an epoch (11.3 s of 12.9).  Two trainings at once
are not the way to it: without NVIDIA's MPS two processes take turns on the GPU, and gpuvm's 4
cores, 62 % busy with one training, are what a second one would run short of.

**What it is.**
- **The batch measured, not guessed:** before the first epoch, a few seconds of the network's step
  at 64, 128, 256, ... frames on the training's device, the batch taken as the smallest within
  10 % of the best time a frame, and below the memory the device has.  The CPU and a small GPU
  keep 64.  `batch_size: auto` in `method.yaml`; a number is used as it is, and 64 is the
  original's training, kept for comparisons with it.
- **The learning rate follows the batch:** a larger batch takes fewer, larger steps through the
  same 32 768 frames an epoch, and DECODE's rate (and StepLR's schedule, in epochs) were tuned for
  64.  The rule -- linear scaling with a warm-up, or the square root -- is what the study below
  decides.
- **Recorded:** the batch and the rate chosen, and the measurement, in `progress.jsonl`'s `start`
  line and the model file's metadata.

**Done when:**
- the reference job at the chosen batch on the H100 reaches the batch-64 runs' final test metrics
  ((f)'s table, within (f)'s tolerance), and the time to get there is measured against theirs
  (3.4 h);
- the choice is measured and written here for the 4090, the H100 and MPS;
- the choosing is tested on the CPU with a given timing (no GPU needed), and a job with a number
  keeps it.

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

### (h), the design (2026-11-11; its open points decided with Jonas the same day)

Built on the patterns the GUI has (surveyed 2026-11-11); nothing in the window imports torch -- a
training is always a process of its own.

**How it opens.**  A plugin `Localize/Train DECODE` whose `window = "decode training"`, and the
DECODE fitter's *Train a model...* button (`Plugin.window_buttons`); the opener registered in
`gui/app.py` beside the PSF calibration's (one instance, `show()` and `raise_()`), named in
`gui/assistant_actions.WINDOW_BUTTONS`.  It is handed the fitter's settings when there are any (the
movie, the camera, the calibration), else the session's first file.  Its page is
`docs/panels/DECODE training.md` (with *Controls*, in `test_panel_docs.PANELS`).

**The window: a list of jobs on the left, three tabs on the right.**
- **Jobs.**  Every job folder this computer prepared or started, with its state (`progress.state`
  and `lock.json`: new, running, interrupted, stopped, finished, error), its epoch and where it
  runs.  Kept as `config_dir()/training_jobs.json`, as `calibration_list` keeps calibrations
  (`{"version": 1, "jobs": [...]}`, a missing folder dropped from view, never deleted); so after a
  restart every job is listed again and a running one followed again.
- **Prepare.**  The movie, the camera (the fitters' `CameraSettings` form) and the calibration,
  filled from the fitter; the density (LD / HD / UHD, §7).  *Prefit* runs `training.prefit.prefit`
  (Spline 3D on 10 blocks) and `decode_smlm.prefit.estimates` on a worker thread (the PSF window's
  `_Worker`: a `QThread`, `progress` and `done` signals, a cancel event, errors as a result, never a
  crash), then shows the estimates as a form one can edit (photons, background, z, on-time,
  density) and the two figures of (e) (simulated frames beside real ones, the ranges over the
  prefit's).  The five estimates are the form; the whole ``method.yaml`` is behind *advanced*.
  *Write job* writes the folder (`training.job.write`) beside the movie,
  ``<movie>_decode/<calibration>_<date>/`` -- the model belongs to the data it was prepared on --
  and lists it.
- **Train.**  Where: *this computer*, or one of the hosts saved in Preferences (a new page,
  *Training hosts*: an ssh alias, the remote folder for jobs, the remote command, by default
  `smappy-decode`).  *Start*:
  - here: ``sys.executable -m decode_smlm.cli train FOLDER`` detached (`handoff.own_session()`),
    stdout and stderr to ``train.log``; refused with a line saying how when this Python has no torch
    (the PSF window's *install torch* offered);
  - on a host: the system's ``ssh`` with ``BatchMode=yes`` (no password prompt: a key or an open
    control socket, which the page explains), the folder copied with ``rsync -a`` (``scp -r`` where
    there is none), the version checked (``smappy-decode --version``, to add), then ``nohup
    smappy-decode train`` there.  Every ssh call is a `QProcess` (`batch_window`'s pattern), never a
    blocking call on the GUI thread.
  *Stop* writes ``STOP`` here or over ssh (``smappy-decode stop``); *Resume* is *Start* again.
- **Follow.**  A repeating `QTimer` (5 s here, 30 s on a host; never `singleShot` with code that can
  raise) reads ``progress.jsonl`` -- here with `progress.read`, on a host by ``tail -c +OFFSET`` over
  ssh into a local copy -- and redraws: the train and test loss, Jaccard, the lateral and axial
  RMSE against the CRLB at the training's mean photons and background (a line, from SMAPpy's spline
  CRLB), the seconds an epoch and the time left, the restarts.  Canvases from
  `gui/canvas.canvas_class()`.
- **Try.**  *Try on my data*: the newest model (copied back from a host first) through the DECODE
  fitter's preview path on a few frames of the movie, its localizations drawn over the prefit's.
  *Use this model* puts it into the DECODE fitter's *model* (a `Signal`, as the PSF window's
  `calibrated`).

**Why a training needs no shared drive.**  It reads no frames: a job folder is ``job.json``,
``method.yaml`` and ``calibration.h5`` (about 30 MB with the calibration), the frames are simulated,
and what comes back is the model file (about 9 MB) and ``progress.jsonl``.  Copied over ssh in
seconds.  The prefit, the one step that reads the movie, runs here in SMAPpy.

**Units.**  The network already sees photons: the package trains with ``convert2photons: true`` --
frames drawn in photons, through the camera model to ADU, back to photons -- and the fitter feeds it
SMAPpy's photons (`to_photons`, then the training's QE and spurious charge), which is how (d)
matched the original.  What a model keeps of its camera is the noise it learnt (EM gain, read
noise): so the training's camera is the job's, from the prefit's data, and a model is checked
against the data's camera as against its calibration.

**Tests** (the plan's done-when, made concrete): a slow window test, offscreen, prepares a job on
frames simulated with a spline (`smappy.simulate`), trains a tiny one here (the training tests'
`tiny_job` sizes), sees two progress lines drawn, stops and resumes it; the ssh path against a fake
``ssh`` on ``PATH`` (a script running the command here, fast tier); `test_panel_docs`,
`test_assistant_gui`, `test_control_width`.

### The fitter run on a workstation (2026-11-11, decided with Jonas)

Inference stays in SMAPpy: everything before the network is the fitters' (the camera's photons, the
camera ROI, the mirror, the blocks, live fits) and *after the fit* follows it, so frames are never
sent to another program.  Locally it is fast on a GPU (the M1 Max's: 107 -- 120 frames/s of the
example's 180 x 179, 20 000 frames in about 3 minutes) and too slow on a CPU alone (3.4 frames/s
there, about 0.4 for 512 x 512: hours a movie).  So the DECODE fitter gets an option to **run on a
workstation**: SMAPpy's own fitter, the same code, headless there (as `smappy-batch` runs a chain),
reading the movie from the shared drive and writing the table back to it -- no frames copied --
with the result loaded here when it is done.  A live fit always runs here.  The hosts are the Train
window's (Preferences, *Training hosts*).  To decide how much it matters, (i) measures the frames a
second of inference on the CPU, the Apple GPU, the 4090 and the H100 at 256 x 256 and 512 x 512.

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
   - (d) settled it on the example package: confirmed (§8, "(d) as done"), and
     `io/decode.py` fixed in a commit of its own.
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

- **The training's density is chosen,** LD / HD / UHD as in SMAP (§6, item 7),
  and the measured one is shown beside it, not used: the density can change
  during an acquisition, and a model trained at a low one learns less from each
  frame.
- **The batch size is fitted to the GPU, early** (2026-11-11), before the studies that
  train, so that they run as fast as the GPU allows: (f2).  Done the same day, with the answer
  no: batch 256 needs three to five times the epochs; the batch stays 64, and the H100's spare
  capacity goes to two or three trainings at once through MPS (§8, "(f) and (f2)").

- **SMAPpy's notation for the integration** (2026-11-11): the new fitter (`docs/decode.md` §10)
  works in SMAPpy's conventions throughout -- x along the columns, 0-based pixel coordinates with
  centres at whole numbers, ``x0`` from 0, calibrations as SMAPpy holds them (``(64, nz, ny, nx)``,
  its monomial order, the camera's orientation), SMAPpy's column names and units -- with no
  conversion layer.  The port (`decode_smlm`) keeps DECODE's, to stay comparable to the original,
  and its boundary to SMAPpy stays in `utils/calibration_io.py`, `predict.py` and `io.py`.  Why:
  every conversion between the two is a place to be wrong, as the ``x0`` of (the pixelation study,
  §8) was, a whole pixel unnoticed until a fit was compared with Spline 3D spot by spot.

Still open:
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
  `eval/rmse_lat`, ..., the loss channels), exported by `scripts/decode/export_log_v0102.py` to
  `benchmarks/decode/results/original_run{1,2}.jsonl`: both ran their 1000 epochs, in 14.7 h
  together.

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


### (c) as done, 2026-11-10

- **The renderer** is `src/decode_smlm/simulation/spline_torch.py`: SplinePSF's kernel
  (`spline_psf.c`, read for it) in torch, behind DECODE's own `CubicSplinePSF`, of which only
  `_spline_impl` changed.  The emitters are grouped by voxel offset -- `frame2roi_coord` leaves
  one lateral kind and one of Z in z (163 kinds in an epoch of the example package) -- so each
  kind's coefficient slab is gathered once and drawn with one batched product; the ROIs are added
  into a canvas padded by a ROI on every side (32-bit indices, no mask) on the device, and the
  frames handed over once.  DECODE's chunking of the frames on the CPU is not used.
- **Against the renderer oracle:** ROIs 1.3e-7, frames 0.8-1.6e-7, derivatives 1.9e-7 of the
  largest value, on the CPU, MPS and CUDA (the 4090, torch 2.15) -- the original's own GPU is
  1.3-1.6e-7 from its CPU.  One departure: SplinePSF's CPU kernel looped a non-square ROI's
  second axis to `npx`.
- **The calibration** comes through `load_spline_calibration` (`utils/calibration_io.py`, in
  place of `SMAPSplineCoefficient`), transposed to DECODE's `(ny, nx, nz, 64)` with the two
  lateral monomial digits exchanged.  §6, item 4 holds: DECODE's x is the rows, SMAPpy's y.
  Without a mirror the coefficients are DECODE's reading bitwise, and the example package's
  frames come out bitwise.
- **The mirror (§6, item 5)** needed more than the plan said.  At the reference SMAPpy's `x0`
  implies, the un-mirrored spline drew the PSF two pixels off; at the right one (`n - r`), DECODE's
  ROI window, which is not mirror-symmetric (voxels 0 to n - 2 of an axis), still cut another edge
  column: 16 % of the peak at 800 nm defocus.  The un-mirrored coefficients are moved one voxel
  along the columns: then the PSF is the file's mirrored to 1.5e-7 at every z, except for an
  emitter exactly on a pixel centre, where DECODE's own window jumps (pinned by a test).
- **`evaluate_spline` (§6, item 6)** agrees inside the ROI to 6.7e-7 (z of +-900 nm); on the
  first row and column it extrapolates where the kernel gives 1e-10, up to 25 % of the peak.
- **Speed, one epoch of the example package** (`benchmarks/decode/bench_renderer.py`: 32 768
  frames of 40 x 40, 1.64 M emitters, its 26 x 26 x 197 calibration; the plan's 10 000 frames
  were the reference parameters', not the package's):

  | | original (`spline`, 1f85d10) | port (torch) |
  |---|---:|---:|
  | RTX 4090, idle, alternated three times (2026-11-11) | 0.30 s | 0.22 s |
  | RTX 4090, under the reference runs' load, alternated | 0.46 s | 0.43 s |
  | M1 Max, MPS | -- | 0.96 s |
  | CPU, 8 threads (workstation / M1 Max) | 49.6 s / -- | 8.3 s / 3.8 s |

  On the way there: masked 64-bit indices for the frames had made MPS take 16 s, and
  DECODE's chunking on the CPU had made the 4090 take 0.84 s.  On the idle 4090 the port is
  1.4 times as fast as the original's CUDA `spline` (0.217-0.224 s against 0.300-0.306 s; the
  same image sum to 1e-11 of it).
- **Defaults:** `CubicSplinePSF`'s device is the machine's, not `cuda:0`.

### (d) as done, 2026-11-10

- **`decode_smlm.io`:** `load_original(param_run.yaml, model.pt, frames_mirrored=...)` and
  `OriginalModel.raw` / `.localize`, the Fit notebook's pipeline as §1 (d) lists it, with the
  camera always the caller's (`training_camera(param)` is what the notebook in effect used);
  `to_columns` puts the positions back on the frames as given, in SMAPpy's names.  `model.pt` is
  a plain state dict and loads with `weights_only=True`.  `frames_mirrored` is not in either file
  and is an argument.  Tests: `tests/decode_tests/test_original_model.py` (slow; skipped without
  the package and the oracle, which `reference_v0102.py model` now also writes the 500 frames for,
  `frames_500.tif`, 32 MB).
- **Against the model oracle:** the network output within 6e-5 of each channel's range on the CPU
  and MPS, 1.4e-4 against the original's CUDA output.  The localizations: all 137 351, in the
  original's order, on the CPU, MPS and CUDA; x and y within 1.5e-5 px, z within 0.03 nm, photons,
  background and probability within 2e-5 of their largest value (the sigmas within the test's 1e-4).  §1 (d)'s 1e-3 nm cannot be met by any
  float32 run: one step of a float32 at 176 px is 1.5e-5 px, 0.002 nm, and the original's own CPU
  and GPU differ by that.  The test asks for 2 steps.  CPU 0.29 s a frame (8 threads), MPS 0.012 s.
- **TF32.**  Current torch lets cuDNN convolve in TF32 on Ampere and later; on the 4090 that moved
  the network output by 1.5 % of a channel's range against the original's (which, on torch 1.12,
  was within 8e-5 of its CPU).  Inference switches it off for its call.  Whether training may use
  it is for (f): it is faster, but the reference runs did not use it in effect.
- **The axes (§6, item 4) are settled:** DECODE's x is the rows.  Matched to SMAPpy's Spline 3D on
  the same frames (probability above 0.7, nearest within a pixel), the medians differ by 0.9 nm in
  x, 1.0 nm in y and 0.8 nm in z (41 000 pairs); with the frames not flipped back, 5.4 nm in x.
  `smappy/io/decode.py` read `xyz[:, 0]` as x: it now reads DECODE's x as `y_nm`, its own commit.
  It cannot undo the crop or the flip, which the file does not record (its docstring says so).
- **The package's calibration is of beads without EM gain** (`EMon` 0), its frames with: the Fit
  notebook's flip matched them, and SMAPpy, which un-mirrors only a spline of EM beads, keeps it as
  it is and only warns ("EM gain mismatch") -- for Spline 3D the frames were flipped by hand.  A
  model the port trains on such a calibration will need the same flip for EM frames; the model
  file's `frames_mirrored` (§5) carries it, and (g) must set it from the calibration's and the
  data's EM settings, not from the user.
- **The pixel size's order is not recorded.**  The package's `px_size` is (127, 117) in DECODE's
  (rows, columns) order, so DECODE's nm put 127 nm on SMAPpy's y; whether SMAP's camera meant 127
  along x is not in the files.  `to_columns` takes the pixel size from the caller in SMAPpy's
  (x, y) order.

### (e) as done, 2026-11-10

- **The estimates** are `decode_smlm/prefit.py`, numpy only: `import decode_smlm` no longer loads
  torch, its subpackages imported when first used (`decode_smlm/__init__.py`, a commit of its
  own; DECODE's warning for GPUs below compute capability 3.7 dropped).  SMAP's rules with its
  constants (`usecurrent_callback`, `setz`, `finalizejson`) for the on-time, background, z and
  camera, SMAP's read noise where the camera gives none, and `method_params`, a job's
  `method.yaml`: DECODE's minimal file (`param_friendly.yaml`'s sections) that its loader fills
  from `reference.yaml`.
- **Three departures from SMAP, all measured:**
  - *The brightness is that of the fully-on frames* (Jonas, 2026-11-10): DECODE draws a
    molecule's photons a frame while it is on, and the frames it switches in get less, so the
    middle frames of blinks of 3 frames or more measure it directly -- their mean and standard
    deviation, of the fits whose `logl_rel` is at most 1 MAD below the median (overlapping
    molecules fit worse and look brighter; losing most fits is fine) and neither at the
    iteration limit nor within 10 % of the calibration's z limit (`fully_on`).  On simulations of
    1.5 and 3 frames' on-time it gives the molecules' 2667 and 1333 photons to 3 %; SMAP's rule
    (the blinks' mean photons over the lifetime, sd 0.2 of it) stays for fewer than 200 such
    frames.  SMAP's rule had given 30 465 photons on the example package, twice its own value:
    Spline 3D's poor fits, far from focus, reach 150 000 photons and pulled the mean.
  - *The on-time is the histogram's mean over SMAP's range, not its three-exponential fit's.*  A
    sum of decaying exponentials is log-convex and blink lengths are not (few in 1 frame, most in
    1 or 2): the fit gave 1.95 frames for blinks of 1.64 on a SMAPpy simulation, 3.3 for 2.5 on
    exponential ones.  SMAP's cut at max(95th percentile, 13) frames is kept; it shortens long
    on-times (2.70 for 3).
  - *The prefit is a fit of its own* (`smappy/training/prefit.py`), not the layer's: 10 blocks of 50
    consecutive frames spread from the acquisition's first frame to its last (it is densest at
    its start), each fitted and grouped on its own, with a 150 nm box, not 50 -- the dim first and
    last frames of a blink scatter further, and 50 nm cut blinks of 2.48 and 3.95 frames to 2.22
    and 3.19 (150 nm: 2.44 and 3.81).
- **On a simulation** of known on-time, brightness, background and z (`tests/decode_tests/test_prefit.py`):
  the lifetime within 10 % at 1.5 and 3 frames, the brightness within 5 %, its spread between the
  molecules' own and 1.3 times it (shot noise on top), background and z as the rules say.
- **The job folder** is `smappy/training/job.py`, torch-free: `job.json` (schema 1), `method.yaml`,
  `calibration.h5`; written, read and written again it is byte-identical, moved it is complete
  (`check`), and an absolute path in `method.yaml` is refused (`tests/test_training_job.py`).
  That a moved folder still *trains* is for (f), which runs it; `random_simulation` now loads the
  calibration through SMAPpy (`spline_psf`), as the training needs.
- **The two check figures** are `decode_smlm/figures.py`: `draw_frames` (a data frame beside one
  simulated by DECODE's own simulation, `simulate_frames`, in photons, on one scale) and
  `draw_ranges` (the prefit's fully-on photons, background and z under the training's draws).
  With SMAP's brightness the simulated frames were three times brighter than the data; with the
  fully-on frames' the spots match (99.9th percentile of the pixels 1 219 against 1 317 photons
  at LD), and what is left is the background's uniform draw over SMAP's widened range.
- **On the example package** (10 blocks of 20 of its first 500 frames, flipped as its beads were
  taken without EM gain, camera from `frames_meta.yaml`), beside its `param_run.yaml` -- whose
  photons are in its EM gain of 100, half the frames' 50:

  | | estimates | `param_run.yaml` | its value at gain 50 |
  |---|---|---|---|
  | lifetime | 1.0 (on-time 1.70 frames) | 1.0 | 1.0 |
  | intensity | 12 672 +- 9 559 (3 286 fully-on frames) | 7 000 +- 3 000 | 14 000 +- 6 000 |
  | background | 33 -- 340 | 20 -- 200 | 40 -- 400 |
  | z | +-995 (the calibration's limit) | +-800 | +-800 |
  | density | measured 3.5 a 40 x 40 frame | UHD, 50 | |
  | read noise | 74.4 e- | 58.8 e- | |

  DECODE draws the brightness from a normal distribution; the fully-on frames' is closer to a
  log-normal (sd 0.75 in log), so the normal of the same mean and spread puts about 10 % of the
  molecules at almost no photons -- invisible to the training, so they only lower its density.

### Training speed, 2026-11-10 (for (f))

Measured on gpuvm's H100 (4 vCPUs) with `benchmarks/decode/bench_training.py`: the port's live
engine on the example package, each phase timed with the device synchronised.

- **An epoch, 19.4 s before:** training 15.8 s (512 batches), the next epoch's simulation 2.4 s,
  the test set with its post-processing 0.9 s, the rest 0.4 s.
- **Two exact changes, 12.9 s after:** the loss's mixture log-probability cut to the target slots
  a batch uses (padded to 250 for about 50 a frame; half the GPU's time; `loss.py`, the method
  oracle unchanged); an epoch's emitters split into frames lazily, in the worker that reads a
  frame (`LazySplit`; 1.6 s of the simulation's 2.4).  1000 epochs: 3.6 h, from 5.4 (the
  original on the 4090: 7.2).
- **Measured, not adopted:** 2 data-loader workers instead of 4, 12.2 s (this machine's 4 cores);
  fused AdamW, no gain; all of torch.distributions' argument checks off, 0.6 s (the one that
  failed runs is now off on its own, below); the network channels-last, training 8.4 s with TF32 and twice as slow
  without.
- **TF32** (cuDNN's, torch's default on the H100) is 1.6x of the training's speed: 11.3 s against
  18 s an epoch.  It is not what failed runs: in a study of 10 seeds x 3 epochs each
  (`--watch`), TF32 on failed 1 of 10, TF32 off 2 of 10, TF32 on with DECODE's gradient
  clipping (`grad_mod`, `clip_grad_norm_` at 0.03) none, and the test loss after 3 epochs had a
  median of 520, 526 and 376 (366 -- 393).  The original's cuDNN 8.3 evidently did not use TF32
  on the 4090 ((d): its GPU output within 8e-5 of its CPU's).
- **The failures were torch's argument check, not a NaN.**  A run caught failing (seed 17, its
  batch saved) had a finite network output: 40 -- 60 pixels a frame saturated at the clamp,
  p = sigmoid(8), and the rest at its floor.  Normalised, the mixture's 1600 weights summed to 1
  within 9.5e-7 in the batch before; in the next float32 rounding went past the 1e-6 that
  `torch.distributions.constraints.simplex` allows, and `Categorical` refused it -- a ValueError
  before DECODE's restart could see anything.  The mixture's `Categorical` is now built
  unvalidated (`loss.py`), which changes no number.  Without clipping the gradient's norm
  reaches 10^7 -- 10^8 in the first epochs (with it, 0.03), likely what saturates the network;
  the original's reference runs ran unclipped to epoch 359 without the check failing, its
  summation order on the 4090 being another.  Whether to train clipped is for (f) and (i):
  better at 3 epochs, but not the original's training.

### (f) as done, 2026-11-10 -- the comparison with the original 2026-11-11

- **The runner** is `decode_smlm/training.py`, `train(folder)`: the live engine's loop around a
  job folder, `setup_random_simulation` and `setup_trainer` as they are.  Added: the job's seed,
  the test set drawn once and kept (`test_set.pt`), every random state in the checkpoint (taken
  before the next epoch's frames are drawn), the checkpoint written atomically, the `STOP` file
  and SIGTERM as a stop at the checkpoint, `model.safetensors` and a `progress.jsonl` line every
  epoch, DECODE's restarts as `restart` lines (each reseeded from the job's seed), `lock.json`,
  and a refusal of a job another SMAPpy version prepared (`--allow-other-version`).  Left out:
  TensorBoard and the four matplotlib figures DECODE drew every epoch only to close them.
- **The files** are SMAPpy's, torch-free: `smappy/training/model_file.py` (safetensors written and
  read with numpy, the metadata a JSON document in the header, read without the tensors) and
  `smappy/training/progress.py` (appended and fsynced; NaN written as null; a line a crash cut
  short skipped, and the next line kept off it).  `smappy.__version__` no longer imports the
  fit pipeline (a training machine may have SMAPpy's sources without its compiled fitter).
- **The command** is `smappy-decode train | stop | status FOLDER` (`decode_smlm/cli.py`).
- **The tests** (`tests/decode_tests/test_training.py`, a tiny job of 16 x 16 frames, 8 batches,
  3 epochs, about a second each on the CPU, so not the slow tier): every progress line valid;
  stopped after epoch 1 and resumed, epoch 2's losses bitwise the uninterrupted run's; a STOP
  file; a warm start from a model file; the lock; the version refusal; `status`.  The model and
  progress files: `tests/test_training_files.py`.
- **A moved folder trains:** the reference job written on the Mac (`scripts/decode/reference_job.py`),
  copied to gpuvm and trained there.  macOS's spawned data-loader workers found a function local
  to `SMLMLiveDataset.sample` that the lazy split held and could not pickle: now module-level, with
  a test.
- **One epoch of the reference job**: 196 s on the M1 Max (MPS, 4 workers), 12.2 s on the H100.
- **The comparison**, §1 (f)'s last criterion, **holds.**  The reference job with seeds 0 and 1,
  1000 epochs each on gpuvm's H100 (`port_run0`, `port_run1`, unclipped, TF32 on, no restart, 3.4 h
  each), against the original's two runs on the 4090.  The means of the last 20 epochs (the test
  sets differ: the original drew its own per run):

  | | original 1 | original 2 | port 0 | port 1 | \|Δ of the means\| | allowed |
  |---|---:|---:|---:|---:|---:|---:|
  | Jaccard | 0.365 | 0.382 | 0.370 | 0.370 | 0.003 | 0.017 |
  | precision | 0.987 | 0.990 | 0.987 | 0.987 | 0.002 | 0.020 |
  | recall | 0.367 | 0.383 | 0.372 | 0.372 | 0.003 | 0.016 |
  | RMSE lateral, nm | 47.0 | 47.9 | 47.9 | 46.9 | 0.08 | 0.95 |
  | RMSE axial, nm | 74.0 | 70.6 | 72.2 | 72.1 | 0.17 | 3.4 |
  | RMSE volume, nm | 87.7 | 85.3 | 86.6 | 86.0 | 0.20 | 2.4 |
  | efficiency lateral | 0.210 | 0.218 | 0.209 | 0.215 | 0.002 | 0.008 |
  | efficiency axial | 0.265 | 0.288 | 0.274 | 0.275 | 0.002 | 0.023 |
  | test loss | 283.9 | 278.1 | 286.6 | 279.2 | 1.9 | 5.8 |

  Allowed is max(\|original 1 - original 2\|, 2 %).  The curves agree from epoch 50 on (Jaccard
  at 100: 0.325 and 0.342 both ways); before it the port's runs were ahead (0.005 and 0.055
  against 0.000 and 0.008 at epoch 10).  The recall is low for both because the test frames hold
  50 emitters on 40 x 40 pixels, as the package's training does.  The logs:
  `benchmarks/decode/results/` (`original_run{1,2}.jsonl`, `port_run{0,1}_progress.jsonl`).

### (g) as done, 2026-11-10

- **The plugin** is `Localize/DECODE` (`smappy/plugins/fit_decode.py`), a `_FitPlugin` whose
  settings have the shared parts under their names (source, microscope, output, *after the fit*)
  and none of detection, the fit region, the channels or the fit: the plan's "hidden" detection
  part is simply absent.  Two hooks were added to `_FitPlugin.run`, nothing else changed in it:
  `provenance` (the file's record, the model's here) and `region_finder` returning None.
- **The engine** is `smappy/window_engine.py`, `WindowEngine`: the last `2 window` frames of a
  block are kept in front of the next, frames whose later neighbours have not been read wait,
  the first and last frame of the movie are repeated (DECODE's `InferenceDataset`), and a jump
  in the frame numbers ends one run and starts another.  `flush`, which a live fit calls after
  every block, holds nothing it could localize; the end is a new `end()`, which `pipeline.drive`
  calls at the source's end when an engine has one.
- **The predictor** is `decode_smlm/predict.py`: `load_predictor(path, device, frames_mirrored=)`
  builds the network from the model file (the architecture from an allowed list, the activation
  a class of `torch.nn` -- nothing in the file is evaluated, where DECODE's own parse `eval`s
  strings), takes photons with `window` frames of context, and gives pixel columns on the frames
  as given.  `export_original` makes a model file of a model the original trained, with the
  sha256 and EM state of its calibration.  `io.to_columns` now shares `pixel_columns` with it.
- **The model file's metadata changed:** `frames_mirrored` (a constant false) is replaced by
  `calibration_em_on`, the EM state of the calibration's beads, which the training reads from the
  folder's calibration.  The fitter flips the frames when the data's EM state differs from it --
  set from the calibration and the data, never by the user, as (d) asked.  Added: `trained_on`,
  the training's photons, background and z ranges, against which the fit is checked afterwards
  (more than 5 % outside is a warning in the run's text).
- **Done when**, each a test:
  - the package's model, exported, through the plugin on the oracle's 500 frames (MPS; 30 on the
    CPU): all 137 351 localizations, in order, within 1e-4 px (`test_fit_plugin.py`, slow);
  - blocks of 1, 7 and 37 frames give the table of one block, exactly; so does a fit from frame
    10 (with frame 9 as context) -- with the network one frame per batch: the CPU's convolutions
    take another path for batches under 32 and round differently, by 5e-4 of a channel's range
    in the untrained test network, which would hide a wrong context frame;
  - a live fit of an NDTiff written in blocks of 3 to 13 frames gives the finished file's table;
  - a camera ROI moves the positions by its corner (non-square pixels);
  - `describe()` with a refusal leaves `torch` out of `sys.modules` (`tests/test_fit_decode.py`);
  - a model trained on calibration A is refused, at Run and in the form, on data checked against
    B; its own passes.  With no calibration to compare (none chosen, none listed for the file's
    microscope), the form says the model is unchecked.
- **The test network** is the tiny job's, three frames in, randomly initialised: it localizes 40
  frames of 32 x 32 in milliseconds, and the plumbing tests need no trained model and no data.
- **Not in the shipped GUI yet:** the plugin is in the tree (*Localize > DECODE*) but not pinned
  in `default.gui.yaml`, until the benchmark (i) says it should be.
- **Left for later:** item 17's padding in place of the crop (edge pixels are still lost, as the
  page says); the range check writes to the run's text, not to the file, since it is made after
  the file is written.

### (f) and (f2), 2026-11-11

- **Convergence per epoch is the original's** (`benchmarks/decode/results/convergence.png`):
  the test loss came within 10 % / 5 % / 2 % of its final value at epochs 46 / 73 / 149 and
  30 / 60 / 145 in the original's runs, 43 / 78 / 146 and 31 / 54 / 141 in the port's.  By time
  the port was 4.3 times as fast (0.5 h to 2 % on the H100 against 2.1 h on the 4090 shared by two
  runs), which is the hardware.  Since 2 % is reached at about 145 epochs, (f2)'s studies compare
  runs of 200.
- **The batch on `auto`** is `decode_smlm/batch.py`: `measure` (the network's step on random
  frames, a copy of the model, a generator of its own -- no random state of the training touched),
  `choose` (the smallest batch within 10 % of the best time a frame), `scaled_lr` (`sqrt`,
  `linear`, `none`) and `resolve`, called once before the first epoch; the checkpoint keeps the
  batch and rate, so a resume or a restart never measures again.  The `start` line has
  `batch` (size, rate, how chosen, the times), the model file `batch_size` and `learning_rate`.
  `benchmarks/decode/bench_batch.py` prints a machine's choice:

  | GPU | 64 | best | chosen | gain a frame |
  |---|---:|---:|---:|---:|
  | H100, TF32 (the training's default) | 253 us | 154 us (1024) | 256 | 1.51x |
  | H100, TF32 off | 463 us | 362 us (1024) | 256 | 1.22x |
  | RTX 4090 | 265 us | 249 us (128) | 64 | -- |
  | M1 Max, MPS | 4042 us | 3637 us (256) | 128 | 1.05x |

  Only the H100 gains: the 4090 is full at the package's 64.
- **A larger rate needs a warm-up.**  Batch 256 at the square root's rate (0.0012, no warm-up,
  the 4090, seed 0) restarted five times for a pathological loss after epoch 1 -- the most
  DECODE allows -- and its training loss jumped from 506 to 1828 at epoch 7; batch 64 at 0.0006
  never restarted.  `lr_warmup_epochs` (with `lr_warmup_from`, by default the rate the linear rule
  gives `batch_base`) ramps the rate linearly over the first epochs, outside the scheduler, so
  the checkpoint and a resume are untouched (a test: stopped inside the ramp and resumed, bitwise
  the run through).
- **The study: batch 256 learns too little an epoch to be worth it.**  On the 4090 (the H100
  kept free; whether batch 256 learns as much an epoch does not depend on the card), 200 epochs of
  the reference job, seed 0, against the port's batch-64 runs (`benchmarks/decode/batch_study.py`,
  `batch_study.png`):

  | 200 epochs | restarts | test loss, 180-200 | Jaccard | batch 64 there at epoch |
  |---|---:|---:|---:|---:|
  | batch 64, seed 0 / seed 1 | 0 / 0 | 291 / 283 | 0.353 / 0.360 | -- |
  | 256, sqrt rate (0.0012), no warm-up | 5 | 302 | 0.278 | 70 / 36 |
  | 256, linear rate (0.0024), 5 epochs' warm-up | 1 | 302 | 0.328 | 61 / 35 |
  | 256, sqrt rate, 5 epochs' warm-up | 1 | 304 | 0.313 | 60 / 34 |

  Batch 64 gets as far in 35 -- 70 epochs as 256 in 200; an epoch at 256 would be at most 1.4
  times as fast on the H100.  An epoch's 32 768 frames are 512 steps at 64 and 128 at 256, and
  the larger rate does not make up for the steps.  So no seed-1 run.
- **Nor does a smaller batch.**  Batch 32 costs 1.24 times as much a frame on the H100 (317 against
  255 us; 1.17 times on the 4090), 16 costs 1.85 times.  Two runs of 32, seed 0, 200 epochs:

  | 200 epochs | test loss, 180-200 | Jaccard | lateral RMSE | at batch 64's seed-0 loss (291) |
  |---|---:|---:|---:|---:|
  | batch 64, seed 0 / seed 1 | 291 / 283 | 0.353 / 0.360 | 46.3 nm | epoch 200 |
  | 32, rate 0.0006 (the 4090) | 291 | 0.382 | 49.3 nm | epoch 151 |
  | 32, rate 0.00042 (the H100) | 290 | 0.378 | 48.7 nm | epoch 137 |

  It ends where 64 ends; 30 % fewer epochs at 1.25 times the time an epoch is at most 15 % in time,
  less than the gap between 64's two seeds.  It finds a few more molecules, harder ones, so its RMSE
  is larger.  Batch 16 is not tried.  The batch stays 64.
  `batch_size: auto` and the warm-up stay as options, off unless a job asks for them.
- **The cores did not matter.**  gpuvm went from 4 cores to 32 (and 125 GB) on 2026-11-11; an
  epoch at 64 took 12.4, 12.7 and 13.0 s with 4, 8 and 16 data workers (it was 12.9 with 4 on 4
  cores).  The training is the GPU's.
- **Several trainings at once, through MPS, use the idle part of the H100** without changing a
  training.  Without MPS the processes take turns (two at once: 26.4 s an epoch each, nothing
  gained).  Through MPS, run as the user with its pipes in a directory of its own (only the
  processes given `CUDA_MPS_PIPE_DIRECTORY` use it; `echo quit | nvidia-cuda-mps-control` stops it):

  | at once | an epoch, each | epochs a minute, together |
  |---:|---:|---:|
  | 1 | 12.7 s | 4.7 |
  | 2 | 18.4 s | 6.5 (1.38x) |
  | 3 | 26.5 s | 6.8 (1.43x) |

  So a study of several runs (seeds, variants, (i)'s models) runs two or three at once through
  MPS; one run alone gains nothing.

### The pixelation study, so far (2026-11-11)

On DECODE's example data, before (i): does a better PSF model reduce pixelation, in Spline 3D and in
DECODE?  Scripts in `benchmarks/decode/pixelation/` (`common.py` says how the data are read).

- **The data:** the example package's movie (U2OS, alpha+beta tubulin, an Evolve 512 at EM gain 50,
  20 000 frames of 180 x 179) and the 11 bead stacks it was calibrated from (200 slices 10 nm apart,
  without EM gain), DECODE_Internal's copy on decodebox.  Pixels 117 x 127 nm, 5 e-/ADU, offset 398.6
  (SMAP's fit of the movie, `frames_sml.mat`); NA 1.43, indices 1.516 / 1.335, 600 nm (the 2024 uiPSF
  v1 model of the beads; it used 50 nm slices, the files and SMAP's calibration say 10).
- **The mirror:** beads without EM gain, movie with it, and a pupil model cannot be mirrored in
  SMAPpy, so the movie is flipped once into the beads' orientation and everything works there.
- **The models:** A, SMAP's spline (DECODE's); B, SMAPpy's pupil model of the beads (converged, most
  beads used, Noll 6 at -0.59 rad); C, B refined in situ on the movie (running).
- **Spline 3D** (fit box 13, *focus as calibrated*): 2.34 M localizations each with A and B, 1.8 M
  kept (photons >= 1000, |z| <= 700 nm, log-likelihood within 3 MADs, not at the iteration limit).
  Flat within a pixel at every photon count, |z| and density (KS 0.001 -- 0.006, counting noise).
  Its z histogram has humps at -550 and +450 nm (A; B puts them at -450 and +420) that shrink to half
  or less at the median log-likelihood cut: misfits, mostly.  B piles fits up at its spline's ends,
  +-794 nm.
- **DECODE with A** (the port's reference model, prob >= 0.7): 3.3 M kept.  Pixelated where the image
  says little -- KS 0.018 within 150 nm of focus, 0.134 at 600 -- 700 nm, 0.038 below 2000 photons;
  density matters little -- with positions pulled to the pixel's centre (a one-bin spike in y) and
  sloped histograms in between.  Its z histogram has no humps and little beyond +-500 nm.  As seen
  before: pixelation where the SNR is low, the density high or |z| large; for the redesign
  (`docs/decode.md` §3.6).
- **Running:** C, and DECODE's trainings on B (the H100, through MPS) and then C, 1000 epochs each;
  then DECODE with B and C on the movie as taken, the fitter flipping it, the positions mirrored
  back for the analysis (`fit_decode.py`).
- **C, the in situ model, is a mismatch** (`compare_fits.py`: the same spots fitted with two models,
  matched within half a pixel).  Spline 3D with C gave a sharp z peak at -100 nm and a narrow one at
  -670 nm, the shape unchanged by the log-likelihood cut -- but C fits 81 % of 1.82 M spots worse
  than A (logl_rel lower by a median 0.21 a pixel; worse in 90 % between -200 and 0 nm) and maps A's
  -200 to +200 nm onto -147 to -15 nm: it compresses z near the focus, which is the peak.  Its
  DECODE training was stopped.  Why in situ learning went wrong on these data (it started from B
  and held its gamma and the phase above order 6) is SMAPpy's in situ learning's question.
- **A DECODE model trained on a calibration SMAPpy made was a pixel off: fixed.**  DECODE with B
  localized 0.962 and 0.988 px off Spline 3D with B in x and y (DECODE with A: 0.012 and 0.009).
  `calibration_io` took ``x0`` as SMAP's, 1-based, less one; SMAPpy writes its own 0-based
  (``(roi_size - 1) / 2``), and its fitter never reads it.  Now SMAP's convention is used where the
  calibration carries SMAP's parameters (``emmirror``, ``ROIxy``), SMAPpy's elsewhere, and a centre
  that is not the grid's middle is refused (a test).  The whole-pixel shift leaves the pixelation
  below as it is (0.04 px of it in the sub-pixel histograms); B's model, trained again with the fix,
  matches Spline 3D with B on 1.47 M spots (32 k before) at median offsets of 1.5, -0.7 and -1.5 nm in
  x, y and z, z within 27 nm (robust spread, |z| < 300 nm) of Spline 3D's.
- **DECODE with B against DECODE with A** (prob >= 0.7, 3.3 M each), KS from uniform, the larger of x
  and y:

  | | DECODE A | DECODE B |
  |---|---:|---:|
  | all, x / y | 0.007 / 0.008 | 0.005 / 0.003 |
  | \|z\| 0 -- 150 nm | 0.018 | 0.013 |
  | \|z\| 150 -- 300 nm | 0.013 | 0.026 |
  | \|z\| 300 -- 450 nm | 0.044 | 0.051 |
  | \|z\| 450 -- 600 nm | 0.081 | 0.068 |
  | 1000 -- 2000 photons | 0.038 | 0.025 |

  The bead pupil model helps near the focus and for dim spots, a little; the growth with |z| stays.
  So the pixelation far from focus is the network's, not the PSF model's: for the redesign
  (`docs/decode.md` §3.6).
- **DECODE pulls z inward below the focus.**  Matched with Spline 3D with A, the spots it puts at
  -400 to -300 nm come back at a median -271, at -500 to -400 at -173 (above the focus: +400 to +500
  at +420).  Its z histograms have no humps and little beyond +-500 nm; part of Spline 3D's humps are
  misfits, but DECODE's z far below the focus is compressed.

### (i), first round: simulated sets on the example's spline (2026-11-11)

Scripts in `benchmarks/decode/benchmark/`: `simulate_sets.py`, `fit_sets.py`, `score_sets.py`,
`plot_scores.py`; data in `$DECODE_SMLM_DATA/benchmark/A/`.

- **The sets:** A, SMAP's spline of the example package (the beads' orientation, so nothing is
  flipped).  Five densities -- 1, 5, 15, 25 and 50 emitters per 40 x 40 pixels -- of about 20 000
  emitters each, single frames of 64 x 64 pixels drawn as the package's training draws them: x and y
  uniform, z uniform in +-700 nm, photons 7000 +- 3000 (at least 1000), background 20 -- 200 a frame,
  the training's EMCCD (gain 100, 5 e-/ADU, read noise 58.8 e-).  Drawn by `smappy.simulate`'s
  `_draw_spline`, not by the training's renderer.  Seeds kept (`sets.json`).
- **The fitters:** Spline 3D (fit box 13, *as calibrated*); the package's published model; the
  original's two reference runs, their final weights through the port's inference (which (d) showed
  gives the original's localizations); the port's two runs.
- **The scoring:** DECODE kept at prob >= 0.5, Spline 3D within 3 MADs of the median log-likelihood
  and not at the iteration limit; matched per frame by the Hungarian algorithm within 250 nm
  laterally and 500 nm axially; scored inside a margin of 7 pixels (Spline 3D's fit box cannot
  leave the frame; DECODE localizes to the edge).
- **Every criterion of §1 (i) for these sets holds:**
  - the port within the original runs' spread at every density (Jaccard at 25: port 0.822,
    original 0.822 and 0.835, package 0.822; lateral RMSE 46.8 against 46.9 and 47.6);
  - at density 1 and >= 2000 photons, DECODE's lateral RMSE 1.11 -- 1.12 times the CRLB (Spline 3D's
    own precision at the same emitters), Spline 3D's 1.19: DECODE is closer to the bound on the same
    spots -- probably the noise, which Spline 3D approximates as Poisson after the excess factor and
    the network learnt as it is;
  - at 25 per 40 x 40, Jaccard 0.82 against Spline 3D's 0.03;
  - DECODE's mean offsets at most 1.8 nm in x and y, 1.1 nm in z, at every density.
- **DECODE finds more even sparse:** recall 0.96 against Spline 3D's 0.77 at density 1, 0.95 against
  0.55 at 5 (defocused spots reach into each other's fit boxes).
- **DECODE pulls z towards the focus at the range's ends:** +8 to +13 nm at -650 nm, -11 to -15 at
  +650, the same in all five models; Spline 3D -7 to +6.  The experimental data showed the same
  (the pixelation study, above): a property of the trained network.
- **B, SMAPpy's own bead calibration** (the pixelation study's pupil model of the same beads,
  `--calibration B`; 2026-11-11): the same sets drawn with B, fitted by Spline 3D with B and the
  port's model trained on B after the x0 fix (``port_B``, 1000 epochs on the H100).  The same
  picture as on A: at density 1 and >= 2000 photons DECODE 1.11 times the CRLB, Spline 3D 1.23;
  Jaccard 0.89 against 0.63 at density 1, 0.77 against 0.04 at 25, 0.64 at 50; offsets at most 2.0 nm
  in x and y, 3.7 in z (at 50).  A model trained on a calibration SMAPpy made works as one trained on
  SMAP's.
- **The example's real frames: the precision each fitter claims against the one the data show**
  (`benchmarks/decode/pixelation/consecutive_frames.py`, 2026-11-11).  A molecule on in two
  consecutive frames is localized twice, independently, so the spread of the difference (1.4826 MAD,
  pairs mutually nearest within 100 nm) against ``sqrt(s_1^2 + s_2^2)`` of the claimed precisions
  says whether a fitter's errors are what it says.  In bins of the claim, 0.6 -- 1.7 M pairs:

  | claimed (nm, an axis) | Spline 3D A, x | DECODE A, x | Spline 3D B, x | DECODE B, x | Spline 3D B, z | DECODE B, z |
  |---|---:|---:|---:|---:|---:|---:|
  | 0 -- 5 (z: 0 -- 10) | 2.48 | 1.20 | 2.56 | 1.29 | 2.99 | 1.25 |
  | 5 -- 10 (10 -- 20) | 1.54 | 1.13 | 1.53 | 1.16 | 1.78 | 1.12 |
  | 10 -- 15 (20 -- 30) | 1.29 | 1.11 | 1.26 | 1.11 | 1.31 | 1.14 |
  | 15 -- 20 (30 -- 45) | 1.19 | 1.06 | 1.15 | 1.06 | 1.09 | 1.15 |
  | 20 -- 30 (45 -- 70) | 1.21 | 1.00 | 1.13 | 1.01 | 0.88 | 1.09 |
  | 30 -- 50 (70 -- 120) | 0.93 | 0.90 | 0.97 | 0.91 | 0.71 | 0.95 |

  DECODE's predicted uncertainties are what the data show within 10 -- 30 %, at every claim; Spline
  3D's CRLB is too optimistic for the bright spots -- 2.5 times below 5 nm, 3 times in z below 10
  nm -- and right only for the dim ones.  An error the CRLB leaves out grows the ratio where the
  claim is small: a model error of the PSF, the EM register's excess noise (a factor 1.41 the
  fitter's Poisson CRLB may lack), or the sample moving a few nm between frames, which DECODE's
  sigma, learnt on simulated frames without motion, would not know either -- its 1.2 -- 1.3 at
  the smallest claims bounds that.  For SMAPpy's fitters, worth its own look: the CRLB is what
  the precision filters and NeNA comparisons read.  (The ratio pooled over all claims is not
  meaningful: a mixture's robust spread against the root mean square of its claims.)
- **Not yet:** the SMLM Challenge 2016 3D-AS set (trainings running); the density test by
  summing frames.

### (h), first step: the window here (2026-11-11)

- **The window** is `gui/train_window.py`, opened by `Localize/Train DECODE` (`plugins/train_decode.py`)
  and the DECODE fitter's *Train a model...* (`window_buttons`); `ControlWindow.open_decode_training`,
  `use_decode_model` (*Use this model* into the fitter), the assistant's `open_window` and
  `WINDOW_BUTTONS`; its page `docs/panels/DECODE training.md`, the plugin's `Localize/Train DECODE.md`.
- **Its work** is module-level: `run_prefit`, `write_job` (beside the movie, `default_folder`),
  `start_here` (``python -m decode_smlm.cli train`` detached, `handoff.own_session`), `stop` (the STOP
  file), `draw_progress`, `try_on_data`.  The list of jobs and the training hosts are
  `smappy/training/jobs.py` (``training_jobs.json``; a job's state from its progress and lock, a
  dead training *interrupted*).
- **Tests:** `tests/test_training_jobs.py`, `tests/test_train_window.py` (a job prepared from a
  simulated movie through Spline 3D, edited estimates written, the curves, the window's list and its
  refusal without torch, the command it starts), `tests/decode_tests/test_train_window_run.py` (slow: a
  tiny job started the window's way, followed, stopped through the STOP file, resumed).
- **Not yet:** training on a host over ssh, its Preferences page, following it there; the simulated
  frames beside real ones in the preparation figure (needs torch here); the CRLB line on the curves.

### Inference speed (2026-11-11)

`benchmarks/decode/benchmark/bench_inference.py`: the port's model, (i)'s density-15 frames tiled to
camera sizes, the predictor with its post-processing, TF32 off.

| device | 256 x 256 | 512 x 512 | 20 000 frames of 512 x 512 |
|---|---:|---:|---:|
| RTX 4090 | 264 frames/s | 67 | 5 min |
| H100 PCIe | 234 | 53 | 6 min |
| M1 Max, Apple GPU | 64 | 15 | 23 min |
| M1 Max, CPU (8 threads) | 2.7 | 0.7 | 8 h |

Any GPU fits a movie in minutes; a CPU alone does not, which is what the fitter on a workstation is
for.  The 4090 is slightly faster than the H100 here: the network is small, and the PCIe H100 is no
faster on it than a gaming card.
