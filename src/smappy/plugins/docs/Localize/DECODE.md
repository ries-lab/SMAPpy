---
version: "1"
covers: [smappy.plugins.fit_decode.DecodeFit, smappy.plugins.fit_decode.DecodeFit.checked_calibration, smappy.plugins.fit_decode.model_problems, smappy.plugins.fit_decode.range_warnings, smappy.plugins.fit_decode.read_model, smappy.plugins.fit_decode._mirrored, smappy.window_engine.WindowEngine, smappy.pipeline.drive, smappy.training.model_file.read_metadata, smappy.plugins.fit_decode.remote_predictor, smappy.decode_remote.RemotePredictor, decode_smlm.serve.serve, decode_smlm.neuralfitter.models.model_speced_impl.SigmaMUNet, decode_smlm.neuralfitter.loss.GaussianMMLoss, decode_smlm.neuralfitter.post_processing.SpatialIntegration, decode_smlm.utils.param_io.autoset_scaling]
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
direction, as the network's halvings need, and scaled as in training: the
middle of the training's background taken off, divided by a fiftieth of its
mean photons.  The network (`SigmaMUNet`) is two
[U-Nets](plugin:Panels/Glossary) in a row.  The first sees each of the three
frames alone, the same network for each; the second sees what the first made
of all three together, and so learns what a molecule on in two frames looks
like.  Four small heads read its output, and give, for every pixel:

* the probability that a molecule is there;
* its offset from the pixel's centre in x and y (at most a pixel either way),
  its z and its photons;
* an uncertainty for each of those four;
* the background.

**How it learnt.**  A training shows it simulated frames whose molecules it
knows, and adjusts it to make those molecules probable.  Its answer for a
frame is read as a [mixture model](plugin:Panels/Glossary): one bell curve a
pixel, centred on the position, z and photons the pixel predicts and as wide
as its uncertainties, weighted by its probability.  The loss -- the number a
training makes small -- is how improbable the true molecules are under that
mixture, plus how far the summed probabilities are from the number of
molecules in the frame, plus the background's squared error.  So a pixel's
uncertainty is learnt as the spread of its own errors, and its probability
as how often a molecule is there.  Molecules of fewer than 100 photons are
drawn in the training frames but not counted in the loss: the network is not
asked to find what is too dim to be found.

**Probabilities become localizations.**  A pixel whose probability is the
largest in its 3 × 3 neighbourhood and above *threshold* starts a
localization; its probability is summed with that of the four pixels it
shares an edge with, capped at 1.  A neighbouring pixel with a probability
above 0.6 starts a second one, so two molecules in adjacent pixels are both
kept.  Where the summed probability is above *threshold*, the position,
photons and uncertainties are read from the network's output at that pixel.

```figure-setup
import torch
from decode_smlm.neuralfitter.post_processing import SpatialIntegration

def blob(shape, centre, peak, width=0.7):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return peak * np.exp(-((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / (2 * width ** 2))

# a probability map as the network gives it: one molecule alone, spread over its pixel's
# neighbours; two in adjacent pixels; and a faint smear that is no molecule
shape = (12, 16)
p = blob(shape, (3.3, 4.6), 0.75) + blob(shape, (9.0, 6.0), 0.8, 0.45) + blob(shape, (10.0, 6.0), 0.7, 0.45) \
    + blob(shape, (13.0, 3.0), 0.09, 1.2)
p = np.clip(p, 0, 0.95).astype(np.float32)
summed = SpatialIntegration._nms(torch.from_numpy(p)[None], torch.add, 0.1, 0.6)[0].numpy()
```

```figure The probabilities a network gives (left) and what they become (right): a molecule's probability spread over its pixel and the next is summed into one localization (circled) where the summed value passes the threshold; two molecules in adjacent pixels, each above 0.6, stay two; the faint smear passes 0.1 but sums to too little to count at 0.5.
axes = fig.subplots(1, 2)
for ax, image, title in ((axes[0], p, "network's probability"), (axes[1], summed, "summed at the maxima")):
    shown = ax.imshow(image, cmap="magma", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
yy, xx = np.nonzero(summed >= 0.5)
axes[1].scatter(xx, yy, s=120, facecolors="none", edgecolors="cyan", linewidths=1.5)
fig.colorbar(shown, ax=axes, shrink=0.8, label="probability")
```

**Positions go on the chip.**  The positions are put back on the frames as
they were read -- the crop and any flip undone -- and the camera ROI's corner
is added, as for every fitter.

**The network can run on a workstation.**  Without a GPU the network is slow:
a laptop's CPU localizes about one frame of 512 × 512 pixels a second, a GPU
50 to 250.  With *run on* set to a training host (*File > Preferences*,
*Training hosts*, the hosts the [training window](plugin:Panels/DECODE training)
trains on), everything but the network still happens here -- the movie is
read, cut to the camera ROI, turned into photons and cut into blocks -- and
each block is sent there over ssh, through the network, and its
localizations sent back.  The model file goes there at the start; it need
not be there before.  The table is the same as here, but for the last digits
where two GPUs round differently.  A live fit always runs here.

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

**The network's numbers.**  Two U-Nets of depth 2, 48 features each
(DECODE's defaults), three frames in.  The probability is a logistic
function of the network's value, clamped to plus or minus 8 before it (so
between 0.0003 and 0.9997); the offsets and z a hyperbolic tangent, so the
offset stays within a pixel and z within plus or minus `z_max`; the photons
and the background a logistic function times `phot_max` and `bg_max`; an
uncertainty a logistic function times 3, plus 0.001.  The scales come from
the training's ranges where its file does not set them (`autoset_scaling`):
the input offset the middle of the background range, the input scale a
fiftieth of the mean photons, `phot_max` the mean photons plus 8 spreads,
`z_max` 1.2 times the top of the z range, `bg_max` 1.2 times the top of the
background range.  A model file keeps the values it was trained with.

**The second molecule's 0.6.**  Of the summed probability's maxima, only
those of a pixel that is the largest of its 3 × 3 neighbourhood start a
localization; a pixel next to one, above 0.6 on its own, starts another.
DECODE's constant (`split_th`), unchanged: lower, a molecule spread over two
pixels would become two.

**The window.**  The localizations of each frame come from that frame and
its neighbours, the frames before and after only as context.  A jump in the
frame numbers, as in a live acquisition that skipped frames, is treated as
the end of one movie and the start of another.

## Differences from the original DECODE

The network, its training and its post-processing are DECODE v0.10.2's,
ported to current PyTorch: a model the original trained gives the original's
localizations on the same frames, and one the port trained scores as the
original's do (within the spread of two of its runs).  What can change the
numbers:

* **The mirror is decided, not assumed.**  The original took the frames as
  they came, and its example data -- beads without EM gain, a movie with it
  -- needed a flipped calibration made by hand.  Here a model file records
  its beads' EM state and the frames are flipped when the data's differs.
* **The calibration's centre.**  SMAPpy reads SMAP's calibrations with
  SMAP's 1-based centre and its own with its 0-based one; read the other way,
  a model trained on a SMAPpy calibration localized a pixel off.
* **No TF32.**  The original ran with PyTorch's default, which on recent
  NVIDIA GPUs computes the convolutions in TF32 and moved the output by
  1.5 % of a channel's range; the port turns it off for inference.

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

### model.where

Worth it when this computer has no GPU: the frames go there as photons, four
bytes a pixel, so a movie of 512 × 512 pixels needs about 50 MB a second at a
GPU's pace -- a wired network carries that, a slow one makes the network the
limit.  The example movie's 20 000 frames of 180 × 179 took 153 s on a
workstation's H100 from a laptop, 166 s on the laptop's own GPU.  *device*
then names the device there.

## Output

The table has the columns of every fitter -- `frame`, `x_nm`, `y_nm`, `z_nm`,
`photons`, `background`, their uncertainties `x_err_nm`, `y_err_nm`,
`xy_err_nm`, `z_err_nm`, `photons_err`, `background_err` -- and `prob`, the
network's probability that the localization is a molecule.  The
uncertainties are the network's own estimate of its error, not a CRLB from a
likelihood: they are right on average for data like the training's, and less
so for data unlike it.  On DECODE's example movie, a molecule localized in two
consecutive frames moved between them by what the two uncertainties say,
within 10 to 30 % at every size of uncertainty -- where Spline 3D's CRLB was
up to 2.5 times too small for the bright spots.

No `logl` or `iterations` is written: nothing is fitted.  The file records
the model, the device, the threshold, whether the frames were flipped and the
model's own metadata.

## References

* Speiser A, Müller LR, Hoess P, et al. Deep learning enables fast and
  dense single-molecule localization with high accuracy. *Nat Methods* 18,
  1082 (2021).
  [doi:10.1038/s41592-021-01236-x](https://doi.org/10.1038/s41592-021-01236-x)
