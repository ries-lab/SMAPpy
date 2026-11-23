# netfit with field-dependent PSFs: the plan (2026-11-23)

`docs/decode.md` §3.2 and step 8: chip coordinates as input maps, FD-DeepLoc's way (Fu et al. 2023,
[10.1038/s41592-023-01775-5](https://doi.org/10.1038/s41592-023-01775-5)).  Why now:
`docs/uipsf/field-dependence.md` §11 found that one central spline with a dz(x, y, z) correction
afterwards works for M5's mild field, but gives no precision back and breaks down from about twice the
synthetic field and on Zenodo's 177 µm field.  Node splines (Spline 3D blending four splines per spot)
match a fit with the exact local pupil.  So a network for such data must learn the field: drawn with the
local PSF, told where on the chip each pixel is.  Jonas chose per-pixel u, v maps over FiLM scalars.

Until this change `netfit/render.py` drew every spot with `cal.coeff`, the field's centre, and ignored a
calibration's `field/` group.

## 1. Drawing with the local PSF

* `SplineRenderer(cal, field=True)` keeps the node splines (`cal.field_splines`, `(G, 64, nz, ny, nx)`) on
  the device, and `spots(..., chip=)` takes each spot's chip position (x, y pixels): the four nodes and
  weights as `field_splines.bilinear` (clamped at the grid's edge, a single row or column without a
  neighbour), and each pixel's 64 coefficients the weighted sum of the four nodes' -- what
  `CSplineField::value` does in `csrc/models.hpp`.  The blend is of coefficients per voxel, so the spot is
  exactly the blended spline's, not a blend of four spots clamped separately.
* A spot's square is fixed by its position in the frame, as before; only its coefficients change.  The
  chip position of a spot is the crop's corner on the chip plus its position in the crop.
* A calibration without `field/` draws as before, `chip` ignored.

## 2. Where the crops are

Training frames (40 × 40 px) are crops of the chip.  With the field on, each movie gets a corner drawn
uniform so that its centre lies anywhere over the nodes' span (the beads' span: `pupil.field_extent`)
widened by half a frame each side: the network sees the corners, and a little beyond them, where the
PSF is the edge's (clamped, as Spline 3D clamps it).

## 3. The input maps

Two maps beside the frames, u and v per pixel:

    u = (clip(chip_x, lo_x, hi_x) − centre_x) · pixel_x / 50 µm

and v alike in rows; ``lo``, ``hi`` the nodes' span, ``centre`` its middle -- `FieldPolynomial`'s centre
(`pupil.field_centre`: the middle of the beads' span), ``pixel`` the training's camera pixel (nm).  The
clip makes the maps stop where the PSF stops changing: beyond the beads the fitter and the drawing both
use the edge's PSF, and the network sees nothing it was not trained on.  The values are recorded in the
model file (§5), and inference computes the maps from them and the file's camera ROI, so one network
serves any ROI of the chip.

Into the network as the C = 2 offset maps go (`two_aux` "every"): beside each frame into the per-frame
U-Net, and again into the union U-Net.  FD-DeepLoc also feeds them to the heads; the union's last
features feed the heads here, so they are left out there.  A C = 1 network with maps gets `aux` = 2; a
network without them is built exactly as before, so every saved model loads as it was trained.

## 4. Settings, job, record

* `SimulationSettings.field`: "off" (the default: a settings file without it, every model so far), "on"
  (refused for a calibration without `field/`) or "auto" (on when the calibration has `field/`).  Resolved
  when the training starts and recorded.
* The training job (`netfit/job.py`, `settings_from_estimates`) writes "auto", so a job on a
  field-dependent calibration trains with the field: the default Jonas asked for.  The *Train DECODE*
  window builds its settings there and needs no change.
* Refused for now: the field with two channels (no dual calibration has `field/`), with a fixed test set
  (`testset.py` keeps no chip positions; the training's own test set is drawn with them), and with an EM
  mismatch between beads and camera (`io.calibration.matched_to_em` cannot mirror a field: it needs the
  chip's width).
* The model file's header gets `"field": {"centre_px", "lo_px", "hi_px", "pixel_nm", "scale_um", "grid"}`
  and the test set's bound is drawn with each molecule's local PSF.

## 5. Inference

* `netfit.predict.Predictor(corner=(ox, oy))`: the frames' corner on the chip, the camera ROI's offset.
  For a model with `field` it makes the maps for the cropped frames (the crop's corner added) and hands
  them to the network.  `frames_mirrored` is refused for a field model.
* `training.predictors.load(..., corner=)`, the DECODE fitter passes `camera.roi_offset`; on a training
  host the corner goes in the server's hello (`decode_smlm.serve`).
* The fitter's notice (`fit_decode.model_problems`): a model trained without the field on a calibration
  that has one says so; a field model on data whose ROI reaches beyond the nodes' span says that there
  the PSF is the edge's.

* The correction afterwards (branch `claude/fd-correction`: `DecodeFit.field_dependent(settings)`, true
  when the model's calibration has `field/`, and *After the fit*'s *field correction*) must not run on a
  model trained with the field: its bias is not the centre spline's.  The flag is the header's `field`
  (`training.predictors.field_dependent(meta)`); `claude/fd-correction` already makes `field_dependent`
  return False for such a model, reading `meta.get("field")` itself.

## 6. Tests

* The renderer's field spot equals `CSplineField` -- the C++ fit's model (`fit_cspline_field` at zero
  iterations, or its model image) at the same position and chip place -- and equals `blend`'s spline drawn
  by the plain renderer; without `field/` it is unchanged.
* The maps: values at known pixels with a camera ROI offset; the clip; the predictor's crop.
* A tiny training with the field (a few steps on the CPU) writes the record, and the predictor reads it.

## 7. The evaluation

Calibrations from `benchmarks/uipsf/synthetic/field_correction_study.py`'s `truth(field, scale)` and
`splines`: the node splines with the central spline as `coeff`, written as a SMAPpy calibration (21 px box,
20 nm planes, ±800 nm).  Fields: synthetic × 1 and × 2 (±50 µm), and Zenodo's (±80 µm).

Trained on gpuvm / decodebox, 6 M samples each (the job's defaults otherwise, GMN7C's ConvGRU over 7
frames, density 50 per 40 × 40 px):
- **netfit, centre**: the same calibration with the field off (what netfit did until now);
- **netfit, FD**: with the field on.

On gpuvm's H100 an epoch of 32 768 samples took 13 s with the centre PSF and 18 s with the field (the
four nodes' gather; synthetic × 2, a 9 × 9 grid; 2026-11-23): 6 M samples about 40 and 55 minutes, the
six trainings about five hours on one GPU, under three on gpuvm and decodebox together.

Scoring, run on this Mac: `fd_study.py score` (Spline 3D on 13 × 13 ROIs at every molecule, three z
starts; the networks on the CPU).  A smoke run of Spline 3D alone on synthetic × 2 at 5000 photons, 150
windows a place: |z bias| at most 10.6 nm near focus and 24.5 nm over ±600 nm, the corners' SD 1.4 and
1.3 times the local CRLB (few molecules a bin; the full run takes 600 a place).

Test data: movies drawn with the node splines over the whole field (the truth; §11 found node splines
match the local pupil), sparse (a few molecules per 40 × 40 px) so that Spline 3D's peak finding is not
what is compared, 5000 and 1500 photons, background 20 and 100.  Fitted by both networks and by Spline 3D
with the node splines (the same frames through SMAPpy's fitter, `chip_xy` from the ROI).

Reported, near focus first (|z| < 300 nm), then over ±600 nm:
- z bias (median fitted − true z) on a 5 × 5 grid of field positions, per z bin -- the map that §11's
  correction afterwards could not flatten;
- lateral and axial RMSE and SD at the four corners and the centre, beside the CRLB of the local PSF;
- Jaccard (the networks only).

Done when the FD network's z bias at the corners is within Spline 3D's, its precision there near the
CRLB, and the centre network's failure is measured beside it.  The numbers go in §8 below.

## 8. Results

(to come)
