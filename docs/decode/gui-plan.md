# DECODE in the GUI: netfit and two channels (plan, 2026-11-22)

What the user sees stays **DECODE**: *Localize/DECODE* fits, *Localize/Train DECODE* trains.  Behind them
there are now two methods and two channel counts: the port of DECODE v0.10.2 (`decode_smlm`,
``decode-v0.10.2``), netfit (`smappy.netfit`, ``netfit-v1``) for one channel, and for two netfit's
C = 2 (plex-plan §16) and DECODE-Plex's published models (`decode_smlm.plex`, inference only).  This
plan extends the two plugins to all of them.  It does not add a fitter or a window.

## 1. The model choice

One field, *model*, under *more* in the training window's preparation, its choices depending on the
calibration (one channel or a dual one):

| channels | choice (label) | method, settings | trains here | default |
| --- | --- | --- | --- | --- |
| 1 | *DECODE* | ``decode-v0.10.2``, the original's SigmaMUNet | yes (`decode_smlm.training`) | |
| 1 | *DECODE, optimised* | ``netfit-v1``, `settings.Parts`' defaults (poc-plan, "netfit's settings decided") | yes (`smappy.netfit.train`) | ✓ |
| 2 | *DECODE-Plex* | the published models, inference only | no: an exported model is fitted, not trained | |
| 2 | *DECODE-Plex-like* | ``netfit-v1``, two channels, late fusion (``two_fuse`` "late", offset maps) | yes | |
| 2 | *aligned channels* (label open, §5) | ``netfit-v1``, two channels, early fusion, the bright channel main | yes | ✓ |

The fitter has no model choice: it reads the method and the channels from the model file's header
(`model_file.read_metadata`, numpy alone) and shows what that model needs.  *DECODE-Plex* is not a
training choice, because the port has no training loop (plex-plan §4).  Its paper's models are fitted
after an export, as DECODE's are (`predict.export_original`).  *DECODE-Plex-like* is the way to train
that architecture here: what the study compared it as.

## 2. Inference: the same fitter

*Localize/DECODE* (`plugins/fit_decode.py`) takes every model:

- **I1, the method.**  `read_model` accepts ``decode-v0.10.2`` and ``netfit-v1``.  `DecodeFit.model`
  builds `decode_smlm.predict.load_predictor` or `smappy.netfit.predict.Predictor`.  Both give the
  `WindowEngine` the same interface: ``window``, a call on frames with their context, SMAPpy's pixel
  columns.  netfit's predictor gains what DECODE's has and it lacks: the automatic batch, the mirror
  for an EM register, ``qe`` and spurious noise.  A one-channel netfit model is then fitted exactly as
  a DECODE one.
- **I2, two channels** (a model whose header has ``channels``).  The fitter shows a *channels* part,
  hidden for one channel (`FormState.shown`):
  - the camera ROI's corner, from the file's metadata as the other fitters read it (Fig. 2c's 175, 0);
    the movie may cut into the channels (`TwoChannels.from_calibration`'s ``shape``, done);
  - the registration: the model's own.  Its header carries the dual calibration's map and geometry (a
    few hundred numbers), so the model is all the fitter needs.  Today the header has only the
    calibration's sha256, so this changes the model file;
  - ***moved since the beads*** (rows, columns): the splitter's move between the bead calibration and
    the measurement, *auto* or typed.  Fig. 2a lost half its molecules without its 5.8 rows (plex-plan
    §16, real data).  *auto* measures it on the first frames: the move that makes the network's
    goodness of fit best over a few frames (each try a fraction of a second on a GPU), or the
    two channels' images cross-correlated through the map (colour; biplane's planes differ in focus).
    The result goes in the run's record and in the form's hint.
  - The frames are cut by `TwoChannels.cut` per block of the `WindowEngine`, and positions are reported
    on the main half, as the two-colour spline fitters report them.
- **I3, the columns.**
  - Per channel: ``photons_0`` / ``photons_1`` and their errors, ``background_0`` / ``background_1``.
  - Colour also gets the share in channel 1, ``ratio``, which is Spline 3D 2C's name.  Where the model
    knows its dyes, ``channel`` is the nearest dye; otherwise the user's *Assign colours* does it from
    the share, as for any two-colour fit.
  - Biplane: one ``photons`` (linked).
- **I4, the checks**, as options with their cost in the tooltip (Jonas, 2026-11-21):
  - *goodness of fit*: a ``goodness`` column (`netfit.goodness`), about 0.1 s for 10 000
    localizations on a GPU, 1 s on a CPU; it flags false positives;
  - *bound from the fit*: ``crlb_x_nm`` etc. and the sigma's ratio to them (`netfit.bounds`), about
    0.4 s for 10 000 on a GPU, 9 s on a CPU; it flags position outliers.
  - Neither is a filter: the layer's filter is.
- **I5, DECODE-Plex's published models**: `export_original`'s two-channel twin, the checkpoint and
  its `param_run.yaml` to a ``.safetensors`` with method ``decode-plex``, inferred by
  `decode_smlm.plex` behind the same predictor interface.  This comes last: it is the baseline, not
  the tool.
- **The training's ranges** checked after the run (`range_warnings`), per channel: Fig. 2c's
  background, twice the training's, is what that check is for.

## 3. Training: the same window

*Train DECODE* (`gui/train_window.py`, `training/`):

- **T1, the job by method.**  ``job.json`` already names its method.
  - A netfit job holds ``settings.json`` (`netfit.settings.Settings.to_dict`) where DECODE's holds
    ``method.yaml``, beside the calibration: a spline ``.h5``, or a dual ``_2ct.h5`` for two channels.
  - ``smappy-decode train FOLDER`` (`decode_smlm.cli`) dispatches on the method:
    `decode_smlm.training` or `smappy.netfit.train`.  So *run here*, the training hosts
    (`training/remote.py`), the jobs list and *Follow* work unchanged.  netfit already writes
    `progress.jsonl` in the shared form (`training.progress`), snapshots included.
  - The stop rule applies to both.
- **T2, the prefit sets the ranges, for netfit as for DECODE.**  `decode_smlm.prefit.Estimates`
  (photons and their spread, background, z, on-time) map onto `SimulationSettings`: photons as a
  range around the estimate, background from its least to its most, the most taken high.  Fig. 2c's
  lesson: the background's 95th percentile per pixel, not its median.  z and on-time are mapped too.
  The density stays a choice (LD / HD / UHD), as now.
- **T3, two channels.**  The calibration decides: a dual ``_2ct.h5`` (the PSF calibration window's
  dual mode, or converted from SMAP's) gives two channels.  What else is needed:
  - ***kind***, biplane or colour.  From the calibration where it says (two planes' focus offset
    against one focal plane), else chosen.
  - ***prefit***: Spline 3D 2C on the movie's blocks, as the one-channel prefit runs Spline 3D.  It
    gives each channel's photons and background, z, and for colour each molecule's share.  It also
    gives the splitter's move since the beads (I2's *auto*, measured once here and kept in the job as
    the fitter's starting value).
  - ***main channel***: *auto*, the brighter channel from the prefit (Jonas: the bright one main), or
    either half.
  - ***dyes*** (colour), Jonas's four choices (2026-11-21):
    - *from the prefit*: the share histogram's peaks, which Fig. 2a's showed clearly at 0.785 and 0.92;
    - *typed*: shares as a list;
    - *broad*: either of the above with a wide spread (``share_sd`` 0.06; fewer outliers, the dimmest
      found less);
    - *independent counts*: no prior, the default for several dyes or none known (isotropic; Fig. 2a's
      dyes were found this way too).
    The choice, the shares and their spread go into the model's header, so the fitter can name the dye.
  - ***architecture***: §1's choice; under *more*, the window (3), the samples (6 M for a model to keep,
    plex-plan §16), streaming always.
- **T4, Try** (a few frames fitted with the model so far): through I1's predictor dispatch, the
  channels as I2 cuts them.
- **T6, the scoreboard** (*Follow*, the early stop): the standard report's numbers (plex-plan §16)
  every epoch, at no extra cost.  The test set is drawn once with each molecule's bound
  (`testset.draw`, `metrics.crlbs_torch`, ``crlb_pooled`` for the frames read), so per epoch they are
  array operations on the matched pairs:
  - *outliers*: the share beyond 5 times the single bound;
  - *inliers' RMSE / CRLB*, lateral and axial;
  - *pixelation* (KS less the truth's own);
  - *right dye* (colour);
  - *sigma honesty*, beside the Jaccard.
  Two channels' test set has no bound yet (`train.py`: "two channels' joint bound is not written
  yet").  `netfit.bounds.information` draws it once at the start, under a second on a GPU.  The early
  stop keeps its rule (loss, Jaccard, efficiencies); the new numbers are shown, not stopped on.
- **T5, the pages.** *Localize/DECODE*'s page and the training window's (`docs/panels/`) say what each
  model is, with the study's numbers where they help choose: early fusion against DECODE-Plex in
  outliers, precision and pixelation; netfit against DECODE in Jaccard and pixelation.  The tooltips
  give the checks' costs.  The assistant's actions follow for the new fields (`window_actions.py`, the
  tests that enumerate them).

## 4. Order

1. **I1**: netfit one-channel models in *Localize/DECODE*.  The smallest step that makes netfit usable:
   models trained by script are fitted in the GUI.
2. **T1, T2**: netfit jobs in the training window, one channel; *DECODE, optimised* the default.
3. **I2 -- I4**: two-channel models in the fitter.  The model file carries the registration; the move
   since the beads and the checks are options.
4. **T3, T4**: two-channel jobs: Spline 3D 2C's prefit, the kind, the main channel, the dyes.
5. **T6** with step 2 (one channel) and step 4 (two).
6. **I5**: DECODE-Plex's published models, exported, as a baseline in the GUI.
7. **T5** alongside each step, not after: each step's page, tests (a simulated two-channel movie
   trained a few epochs and fitted, as the one-channel tests do) and the slow tier.

## 5. Decided (Jonas, 2026-11-22), and open

- **One channel's default is netfit** (*DECODE, optimised*); *DECODE* under *more*.
- **No "netfit" in the GUI's names**: the method's name is in *Model info...* and the file only.
- **Early fusion's label** is open; "early fusion" is possible but not telling.  What it is: channel
  1 resampled onto channel 0's pixels through the registration, both read as one two-colour image
  from the first layer; late fusion gives each channel its own first layers and joins them later.
  Candidates: *aligned channels* (proposed), *channels overlaid*, *joint image*, *channels read
  together*; its counterpart stays *DECODE-Plex-like*.
- **The channels' move since the beads: measured by the goodness, reported as a move.**  A scan over
  shifts of the registration on a few frames, each fitted by the network and drawn back
  (`netfit.goodness`).  The shift where the goodness is best is the move, a correction to the
  registration at inference, never a retraining.  The goodness at that shift is the model's fit to
  the data: still poor there means a model mismatch (Fig. 2c's background), not a move.  So one scan
  answers both, and the form says both.  Retraining is not needed for a move: the network takes the
  registration as offset maps at its input, and a translation leaves them within what it saw
  (Fig. 2a, fitted with its 5.8 rows by a model trained on the beads' map).  A map changed beyond a
  translation (scale, turn) is a new bead calibration's matter.  Learning the whole map from the data
  (colour: the same molecules in both channels) is possible later.
- **Fig. 2c's background**: whether a range set from the prefit is enough or the simulation needs an
  uneven background.  The model training now (R_biplane_early_bg1000) answers that for this movie.
- **DECODE-Plex training** (porting its Lightning loop): not planned; *DECODE-Plex-like* trains the
  architecture here.
