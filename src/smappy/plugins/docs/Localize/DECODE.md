---
version: "1"
covers: [smappy.plugins.fit_decode.DecodeFit, smappy.plugins.fit_decode.DecodeFit.checked_calibration, smappy.plugins.fit_decode.model_problems, smappy.plugins.fit_decode.range_warnings, smappy.plugins.fit_decode.read_model, smappy.plugins.fit_decode._mirrored, smappy.window_engine.WindowEngine, smappy.pipeline.drive, smappy.training.model_file.read_metadata]
---

## What it does

DECODE (Speiser et al. 2021,
[doi:10.1038/s41592-021-01236-x](https://doi.org/10.1038/s41592-021-01236-x))
localizes molecules with a neural network instead of fitting a model of the
[PSF](plugin:Panels/Glossary) to each spot.  The network was trained on
frames simulated with this microscope's PSF, where the true positions are
known.  Given a frame, it says for every pixel how likely a molecule is
there, and where in the pixel, in z, how bright, and how uncertain each of
these is.

This helps most where spots overlap.  A fitter like
[Spline 3D](plugin:Localize/Spline 3D) needs each spot on its own in its fit
box, and at high density it misses or merges molecules.  The network has
learnt what two spots in one place look like.  On sparse data with bright
molecules a spline fit is as precise and needs no training.

The fitter needs a **trained model**, a `.safetensors` file.  A training
writes one into its job folder (`smappy-decode train`), and a model that the
original DECODE trained can be exported to one.  A model belongs to one
calibration, one camera and one range of brightness and background, the ones
it was trained on: it gives confident answers for data of another PSF too,
and they are wrong.  So the fitter checks what it can (below).

There is no peak finding and no fit box: the form has none of the other
fitters' detection and fit parts.

## How it works

**The frames become photons.**  The counts are converted with the camera:
the [offset](plugin:Panels/Glossary) taken off, times the
[conversion](plugin:Panels/Glossary), divided by the
[EM gain](plugin:Panels/Glossary).  The network was trained on frames
simulated with the camera's noise, so the photons are not divided by the
excess-noise factor of an EMCCD as a likelihood fit's are.

**Each frame is seen with its neighbours.**  A model trained with a frame
window of three sees the frame before and after the one it localizes: a
molecule that is on in the next frame too is easier to tell from noise.  The
movie is read in blocks.  The last frames of a block are kept and put in
front of the next, and the frames whose later neighbours have not been read
yet wait for them.  So the localizations do not depend on the block size, and
a live fit of a file still being written gives the table the finished file
gives.  At the first and last frame of the movie that frame is repeated in
place of the neighbour it does not have, as DECODE did.

**The network.**  The frames are cut to a multiple of 8 pixels in each
direction, as the network's two halvings need, and scaled as in training.
Out comes, for every pixel, a probability of a molecule, its offset within
the pixel in x and y, its z, its photons, the background, and an uncertainty
for each of x, y, z and the photons.

**Probabilities become localizations.**  A pixel whose probability is the
largest in its 3 × 3 neighbourhood and above *threshold* starts a
localization; its probability is summed with that of the four pixels it
shares an edge with, capped at 1.  A neighbouring pixel with a probability
above 0.6 starts a second one, so two molecules in adjacent pixels are both
kept.  Where the summed probability is above *threshold*, the position,
photons and uncertainties are read from the network's output at that pixel.

**Positions go on the chip.**  The positions are put back on the frames as
they were read -- the crop and any flip undone -- and the camera ROI's corner
is added, as for every fitter.

## In detail

**The crop.**  Frames whose sides are not a multiple of 8 lose up to 7
pixels of each side, split between the two edges (the extra one at the top
and the left).  Molecules on those edge pixels are not localized.

**The mirror.**  SMAPpy keeps a bead calibration in the orientation of the
camera its beads were taken with, and an EMCCD read out through its
EM register mirrors the image in x.  The model file records whether the
calibration's beads were taken with EM gain (`calibration_em_on`).  Data
taken the other way round -- beads without EM gain, data with it, as in
DECODE's example data -- are flipped before the network sees them, and the
positions flipped back.  Where either side does not say, nothing is flipped,
as Spline 3D does not flip.

**The checks before a run.**  The model file keeps the sha256 of its
calibration's spline coefficients.  If *calibration* is given, or the
calibration list has one for this file's microscope and state (as Spline
3D's *auto* finds it), it must be the same calibration, or the run is
refused.  With no calibration to compare, the form says the model is not
checked.  These checks read the model file's header only, without loading
the network.

**The check after a run.**  The fit's photons and background are compared
with what the training drew.  More than 5 % of the localizations brighter
than the training's mean plus three spreads, or with a background outside
its range, is a warning in the run's text, and so in the session's history:
the network has seen no molecule like those, and their numbers are
extrapolated.

**What the network ran on.**  The training's quantum efficiency and spurious
charge are taken off the photons as DECODE's camera model did.  The
network runs without TF32 on NVIDIA GPUs, which moved its output by 1.5 % of
a channel's range.  Different batch sizes round differently on the CPU, by
float32's last digits.

**The window.**  The localizations of each frame come from that frame and
its neighbours, the frames before and after only as context.  A jump in the
frame numbers, as in a live acquisition that skipped frames, is treated as
the end of one movie and the start of another.

## Parameters

### model.path

A job folder's `model.safetensors` is rewritten after every epoch of a
training, so a training still running can be tried; take it from a finished
training for the fit that counts.

### model.calibration

Set it when the calibration list does not know this microscope.  It is only
compared with the model, never used to localize: the PSF the network knows
is the one it was trained on.

### model.threshold

0.1 is DECODE's Fit notebook's.  A low threshold keeps uncertain detections,
which carry their `prob`: filtering the table by `prob` afterwards (0.5 to
0.7) is the usual way to trade missed molecules against false ones, and can
be undone.

### model.batch

32 suits a GPU with a few GB; lower it if the GPU runs out of memory, raise
it on a large one for speed.  On the CPU it changes the last digits only.

## Output

The table has the columns of every fitter -- `frame`, `x_nm`, `y_nm`, `z_nm`,
`photons`, `background`, their uncertainties `x_err_nm`, `y_err_nm`,
`xy_err_nm`, `z_err_nm`, `photons_err`, `background_err` -- and `prob`, the
network's probability that the localization is a molecule.  The
uncertainties are the network's own estimate of its error, not a CRLB from a
likelihood: they are right on average for data like the training's, and less
so for data unlike it.

No `logl` or `iterations` is written: nothing is fitted.  The file records
the model, the device, the threshold, whether the frames were flipped and the
model's own metadata.

## References

* Speiser A, Müller LR, Hoess P, et al. Deep learning enables fast and
  dense single-molecule localization with high accuracy. *Nat Methods* 18,
  1082 (2021).
  [doi:10.1038/s41592-021-01236-x](https://doi.org/10.1038/s41592-021-01236-x)
