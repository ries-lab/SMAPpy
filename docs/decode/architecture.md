# DECODE in SMAPpy: what is built

The proof of concept of `docs/decode.md` §9, as it stands (2026-11-12).  The plan and its record
are `poc-plan.md`; the axes, the mirror and the calibration's centre are `conventions.md`; where
the data come from is `data.md`.

## Two packages, one boundary

* **`decode_smlm`** (GPL-3.0, `src/decode_smlm/NOTICE.md`) is DECODE v0.10.2 ported to current
  PyTorch: its simulation, network, loss, post-processing and evaluation, in DECODE's own
  conventions so that it stays comparable to the original.  Added to it: `training.py` (a job
  folder trained, with its progress, lock, STOP file, checkpoint and model file), `predict.py` (a
  model file put to use: `load_predictor`, `Predictor`, `export_original`), `prefit.py` (SMAP's
  training estimates, torch-free), `batch.py`, `figures.py` (the preparation's two check figures),
  `serve.py` (a predictor for a SMAPpy on another computer) and `cli.py` (`smappy-decode train |
  stop | status | serve`).  `import decode_smlm` loads no torch.
* **`smappy`** (BSD-3) holds what is no method's: `smappy/training/` -- the job folder (`job.py`),
  the progress file (`progress.py`), the model file (`model_file.py`), the prefit on Spline 3D
  (`prefit.py`), the list of jobs and the training hosts (`jobs.py`), a job on a host over ssh
  (`remote.py`), the CRLB reference of the Follow curves (`reference.py`) -- and the fitter's side:
  `window_engine.py` (blocks of frames with their context), `plugins/fit_decode.py` (*Localize >
  DECODE*), `decode_remote.py` (the network on a training host), `plugins/train_decode.py` and
  `gui/train_window.py` (*Localize > Train DECODE*), and the *Training hosts* page of Preferences
  (`gui/preferences.py`).

The boundary is three modules of `decode_smlm`: `utils/calibration_io.py` (a SMAPpy calibration
in DECODE's layout), `predict.py` (frames in photons in, SMAPpy's pixel columns out) and `io.py`
(the original's files).  Everything on SMAPpy's side is in SMAPpy's conventions; nothing converts
in between.  SMAPpy imports `decode_smlm` only inside the functions that need a network, and a
test checks it.

## What happens, from a movie to a table

1. **Prepare** (the Train window, torch-free).  Spline 3D fits blocks spread over the movie
   (`training.prefit`), more of them while too few molecules were fully on in a frame; SMAP's
   rules turn the fits into the training's ranges (`decode_smlm.prefit.estimates`) -- photons,
   background, z, the time a molecule is on -- which the user may edit.  `write_job` writes the
   job folder beside the movie: `job.json` (where it came from, the camera, the ranges, SMAPpy's
   version), `method.yaml` (DECODE's parameters), `calibration.h5` (the calibration, copied).
2. **Train**, here or on a training host.  `smappy-decode train FOLDER` simulates its frames from
   the folder's files alone -- the movie is not needed -- and writes `progress.jsonl` (a line an
   epoch), `ckpt.pt` (to resume), `model.safetensors` (every epoch) and, while running,
   `lock.json`.  On a host the folder is sent over ssh and the progress, lock and model copied
   back (`training.remote`); the Train window reads the same files either way.
3. **Fit** (*Localize > DECODE*).  The movie is read in blocks, cut to the camera ROI and turned
   into photons as every fitter does; `WindowEngine` keeps each block's last frames as the next
   one's context; the predictor runs the network and DECODE's `SpatialIntegration` and gives
   pixel columns, which become SMAPpy's localizations on the chip.  With *run on* a training
   host, only the predictor's step happens there (`decode_remote.RemotePredictor` talking to
   `smappy-decode serve`).  *After the fit* follows as for every fitter.

## The files

| file | written by | read by |
|---|---|---|
| `job.json`, `method.yaml`, `calibration.h5` | `training.job.write` | the training; the window |
| `progress.jsonl` | the training, a line an epoch | the window, `smappy-decode status` |
| `lock.json`, `STOP` | the training; the window or `smappy-decode stop` | the window; the training |
| `ckpt.pt`, `test_set.pt` | the training | the training, resumed |
| `model.safetensors` | the training, every epoch; `predict.export_original` | the fitter |

A model file holds tensors and JSON: the architecture by name from an allowed list, the scaling,
the camera, the calibration's sha256 and its beads' EM state, the ranges it was trained on.
Nothing in it is run.

## Tests

`tests/decode_tests/` (DECODE's own tests ported, and the port against the original's outputs in
`fixtures/`), with torch; `tests/test_train_window.py`, `test_training_jobs.py` and
`test_training_remote.py` torch-free.  The benchmarks are `benchmarks/decode/`.
