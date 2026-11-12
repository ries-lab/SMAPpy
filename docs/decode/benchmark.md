# The new fitter's benchmark (step 2)

`docs/decode.md` §10, step 2: a fixed benchmark, built before any new network, that every fitter
and every variant of the new one is scored on the same way.  Step 4 changes one part of the network
at a time -- the backbone, the time model, the output and its extraction, the loss, the schedule --
and needs the numbers to differ only by that part.  So the sets are drawn once and kept, the report
is one function of a fit and its truth, and a run names its parts.

The code is `benchmarks/decode/suite/`; the data, fits and reports are under
`$DECODE_SMLM_DATA/benchmark/v1/` (`~/.cache/decode_smlm`).  The results are in
`docs/decode/poc-plan.md`, "Step 2, first round".

## The sets

`sets.py` defines them and draws the simulated ones (`python sets.py draw`), recording each with
its seed, its file's size and hash in `manifest.json`.

**Simulated** (decided with Jonas, 2026-11-12).  Calibration B, SMAPpy's pupil model of the example
package's beads; frames of 64 x 64 pixels of 117 x 127 nm; about 20 000 emitters a set, x and y
uniform over the frame, z uniform in +-700 nm; the example package's training camera (EMCCD, gain
100, 5 e-/ADU, read noise 58.8 e-).  Two sweeps, each changing one thing:

| sets | density (per 40 x 40 px) | photons | background (a pixel) |
|---|---|---|---|
| `density_1`, `_5`, `_15`, `_25`, `_50` | 1, 5, 15, 25, 50 | normal 7000 +- 3000, at least 1000 | uniform 20 -- 200 |
| `photons_1000`, `_3000`, `_10000` | 1 | 1000, 3000, 10 000 | 2 % of the photons: 20, 60, 200 |

The first sweep is the training's own prior at rising density; the second the signal to noise
from poor to good at a fixed ratio of background to signal, inside the training's range so that it
measures the fitter, not the range it was trained on.  z is not a sweep: every set spreads it, and
the report bins it.  Background was a third sweep in the proposal and was dropped: inside 20 -- 200
it matters less than the photons.

**Drawn as an EMCCD reads them.**  The beads were taken without EM gain, an EM register reads the
chip out mirrored, so the PSF is drawn mirrored in x (`matched_to_em`), and every fitter gets the
frames as a real movie's and mirrors its own calibration where it must -- Spline 3D the spline
(`matched_to_em`), DECODE the frames (`frames_mirrored`).  The sets of (i) were drawn in the beads'
orientation and fitted with the mirror switched off; these test the path a user's data takes.

**Each emitter's CRLB is kept with its truth**: the Poisson bound of SMAPpy's spline fitter on the
spot alone and noise-free, doubled in variance for the EM register's excess noise -- in 13 x 13
pixels (`crlb_*`, Spline 3D's fit box) and in 25 x 25 (`crlb_*_full`, the calibration's whole
spot, which a network seeing the whole frame can use; up to a third lower in z far from focus).
The median bound laterally is 11 nm in the density sets, 30, 17 and 9 nm at 1000, 3000 and 10 000
photons.

**The SMLM Challenge 2016, 3D-AS**: the four training sets (`MT0.N1/N2.LD/HD`) with their truth, a
model per noise level; one median shift per fit taken off before scoring, as (i) did (the
calibration's centre is the bead average's).  No CRLB: the truth has no background per emitter.

**Real data**, a fixed range of frames of each movie, kept where it is:
- `example` -- the example package's movie (U2OS, alpha+beta tubulin, EM gain 50), frames 0 -- 4999,
  context B;
- `fig4e` -- the DECODE paper's Fig. 4e raw Micro-Manager movie, frames 2000 -- 6999, with the
  model the GUI trained for it (its own beads, `261111_beads_3Dcal`) and that job's camera.

**Contexts.**  A network is trained for a calibration and a camera, not for a set.  Each set names
its context -- `B`, `challenge_N1`, `challenge_N2`, `fig4e` -- and a run gives one model per context:
one training on B covers every simulated set and the example movie.  Spline 3D takes the context's
calibration.

## The report

`report.py`, one JSON per run (`reports/<run>.json`), every set scored by the same function.
Localizations are kept by the run's `keep` (a network's at `prob` >= 0.5; Spline 3D's within 3
MADs of its median log-likelihood and not at the iteration limit), matched per frame by the
Hungarian algorithm within 250 nm laterally and 500 nm in z, inside a margin of 7 pixels -- (i)'s
rules, unchanged.

| measure | key | what it says |
|---|---|---|
| detection | `jaccard`, `recall`, `precision` | what was found and what was invented |
| accuracy | `rmse_lat_nm`, `rmse_ax_nm`, `eff_lat`, `eff_ax` | the challenge's numbers |
| own claims | `claim_width_lat` / `_ax`, `claim_width_by_z` | each error over the fitter's claimed uncertainty, 1 where right; z per 200 nm of true z, bias off |
| against the bound | `rmse_over_crlb_lat` / `_ax` | RMSE over the matched emitters' root-mean-square CRLB |
| | `crlb_width_lat` / `_ax`, `crlb_width_by_photons` | 1.4826 MAD of each error over its own emitter's CRLB: Follow's measure, blind to mismatches; 1 at the bound |
| | `..._full` | the same against the whole spot's bound |
| bias | `bias_x/y/z_nm`, `z_bias_nm` | mean offsets; z's in 100 nm bins of the true z |
| pixelation | `ks_x`, `ks_y`, `ks_by_abs_z`, `subpixel_x/y` | KS distance from uniform of the matched positions within their pixel, in bins of |z| (the truth is uniform there) |
| speed | `frames_per_s` | wall time of the fit, file and peak finding included, the device recorded |

Real sets have no truth: localizations a frame, the KS distance of the kept ones (at least 1000
photons) by |z| and photons, and the pair check -- a molecule on in two frames localized twice;
each pair's difference over the root of its two claimed variances, its robust width, overall and by
bins of the claim, 1 where the claimed precision is the real one.  The two frames are as far apart
as the fitter needs for its two fits to share no input: 3 for a model reading 3 frames, whose
errors in consecutive frames correlate by 0.73 (`precision_check.py`, which tests a model's claims
on a simulated movie of molecules on for several frames).  On real data the ratio grows with the
frames apart for every fitter (the sample moves), so it is compared at the least such distance.

`python report.py --compare RUN RUN ...` prints the runs side by side with their parts and costs.

**Single frames and temporal context.**  The simulated sets are independent single frames: a
model reading several frames sees no molecule in its neighbours, so it acts as its own single-frame
counterpart (decided with Jonas, 2026-11-12: every network compared will read several frames in the
end, so no single-frame model is trained beside it).  What the network does alone -- its
uncertainty, its pixelation -- is measured there; what temporal context adds on a multi-frame movie
(`precision_check.py`, molecules on for several frames).

## A run, and how step 4 uses it

A run is a YAML file in `suite/runs/`:

```yaml
name: decode_port
fitter: decode_smlm           # the adapter that runs it (run.py's ADAPTERS)
parts:                        # each named; the new fitter reads the same names from its settings
  backbone:   {name: decode_unet, ...}
  time_model: {name: frame_stack, ...}
  output:     {name: decode_offsets, ...}
  extraction: {name: decode_lookahead, ...}
  loss:       {name: decode_ppxyzb, terms: [count, mixture, background]}
  schedule:   {name: decode_v0102, ...}
models: {B: ..., challenge_N1: ..., challenge_N2: ..., fig4e: ...}   # one per context
fit:  {threshold: 0.1}
keep: {min_prob: 0.5}
```

`run.py RUN` fits every set whose context has a model, through the fitter's plugin with the frames
as they are, keeps each fit (`fits/<run>/<set>.hdf5`) and writes the report with the run's spec,
what each model file records of itself (`model_record`: parameters counted from its tensors,
architecture and its settings, the training's ranges, epochs, batch, learning rate) and the cost.
A fit is kept and only scored again unless `--refit`, so the scoring can change without refitting.

**The parts and their candidates** (from `docs/decode.md` §2, §3.3, §3.4):

| part | the baseline's | candidates for step 4 |
|---|---|---|
| `backbone` | `decode_unet` (SigmaMUNet, 48 features, depth 2 + 2) | residual / ConvNeXt U-Net, HRNet-style full resolution, NAFNet, large kernels or dilations, anti-aliased down- and upsampling |
| `time_model` | `frame_stack` (each frame's features concatenated) | none (one frame), per-pixel attention over +-3 frames (LUNAR), a ConvGRU |
| `output` | `decode_offsets` (p, photons, x, y, z in +-1 px, sigmas, background) | overlapping offsets +-1.5 px (SHOT), sub-pixel bins (SimCC) |
| `extraction` | `decode_lookahead` (SpatialIntegration or NMS) | LUNAR's mixture peak, none (SHOT) |
| `loss` | `decode_ppxyzb`: count, mixture, background | optimal transport (Sinkhorn, SHOT), an occupancy term (FD-DeepLoc), the terms' weights |
| `schedule` | `decode_v0102` (AdamW, StepLR, 1000 epochs) | the early stop of today's training, ranges, batch |

**A variant states only its difference**: `base: decode_port` takes that run's YAML and replaces
what this one names (a part whole; `fit` and `keep` key by key).  The first are
`decode_port_mixpeak` and `decode_port_mixpair`, the port's models with the extraction changed to
the mixture's peak (`extraction.py`; switched into the port's predictor for the benchmark until the
new fitter has its own extraction part) -- results in `poc-plan.md`, "Step 4's first part, early".

**Matched comparisons.**  Two runs are compared when they differ in one part and their costs are
close: the parameters (from the model file), the epochs and the training's wall time on one GPU,
and the frames a second at inference.  The report records all four; a comparison at unmatched cost
says so.  A part's settings beyond its name go into the run's YAML and the model file, so that a
report says exactly what was trained.  Network code is written new for the parts whose sources are
GPL or AGPL (LiteLoc, SHOT): ideas only.

**The spread between trainings comes first** (Jonas, 2026-11-12).  Two trainings of one model from
different seeds need not reach the same minimum: the original DECODE's two reference runs differed
by 0.013 in Jaccard at density 25 ((i), `poc-plan.md`).  So once there is a good base model, it is
trained several times (seeds, at least 3 -- 5, two at a time on gpuvm and the workstation), each scored
on the fixed test set (`netfit testset`) and the benchmark, and the spread of every number recorded.
A difference between two variants counts only where it exceeds that spread; until then a single
training's gain is a hint, not a result.

**Training targets below 100 photons** (Jonas, 2026-11-12, to test in step 4).  netfit's version 1
trains, as `decode_B` was trained, on targets from 100 photons in the frame (`target_photons_min`,
DECODE's `emitter_label_photon_min`): a dimmer molecule is drawn but is no target, so the count term
teaches the network to call it background.  Without the limit the network is shown every molecule's
true position, which may teach it dim events better -- and help the molecules just above 100 photons
most.  One setting; judged by the evaluation's 100-photon rule (`metrics.PHOTONS_MIN`), recall and
precision near 100 -- 1000 photons especially.

**What step 4 will need that is not here yet**: the training's own wall time in the model file
(today only epochs), FLOPs per frame, and the training on gpuvm launched from a run's YAML; all
belong to the new fitter (step 3), which reads `parts` from its settings.

## Not in the first round

- **DECODE-Plex** (`ries-lab/DECODE-Plex`): no environment of it here, and a C = 1 model of it
  would have to be trained on calibration B in its own environment.  It is the baseline for C = 2
  (step 5), where its published weights fit; a C = 1 run is added when it is installed.
- **The published DECODE package's model** was trained on SMAP's spline A, which these sets are not
  drawn with; (i)'s sets on A compared it with the port (equal within the runs' spread).
- **Other modalities** (C = 2, 4Pi) get their sets when their steps come (5, 6).
