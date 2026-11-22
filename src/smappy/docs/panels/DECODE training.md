---
title: DECODE training
summary: Prepares a DECODE training from a movie -- a short Spline 3D fit of it sets what the training simulates -- writes it as a job folder beside the movie, trains it on this computer or a workstation over ssh, follows its curves and tries the model on the data.
widget: smappy.gui.train_window.TrainWindow
covers: [smappy.gui.train_window.run_prefit, smappy.gui.train_window.run_prefit_two, smappy.netfit.prefit_two.prefit_two, smappy.netfit.prefit_two.dye_shares, smappy.netfit.prefit_two.kind_of, smappy.netfit.job.two_channel_settings, smappy.gui.train_window.two_channel_problem, smappy.gui.train_window.netfit_params, smappy.netfit.job.settings_from_estimates, smappy.netfit.job.train_folder, smappy.netfit.metrics.outliers_and_inliers, smappy.gui.train_window.prepared_by_hand, smappy.gui.train_window.write_job, smappy.gui.train_window.default_folder, smappy.gui.train_window.start_here, smappy.gui.train_window.stop, smappy.gui.train_window.draw_progress, smappy.gui.train_window.try_on_data, smappy.gui.train_window.start_on_host, smappy.gui.train_window.HostFollower, smappy.training.jobs.status, smappy.training.remote, smappy.training.remote.others, smappy.training.remote.adopt, smappy.training.progress.converged, smappy.training.progress.stop_status, smappy.training.reference.robust_width, smappy.training.reference.subpixel_histogram, decode_smlm.training.test_crlbs, smappy.training.joint_crlb.bounds, smappy.gui.train_window._split_panel, smappy.gui.train_window.rule_line, smappy.training.jobs.Status, smappy.gui.preferences.HostsPage, smappy.gui.train_window.simulated_frames, smappy.gui.train_window.simulated_frames_two, smappy.gui.train_window.channels_side_by_side, smappy.netfit.bounds.drawn_crlb, smappy.training.prefit.prefit, decode_smlm.prefit.estimates, smappy.gui.train_window.in_the_sample, smappy.calibrate.depth.for_sample]
---

## What it does

[DECODE](plugin:Localize/DECODE) localizes with a neural network that has
learnt what molecules look like on this microscope and this camera.  It
learns that from frames it simulates: molecules drawn with the bead
calibration's [PSF](plugin:Panels/Glossary), as bright, as dense and over the
same background and z range as in the data, with the camera's noise.  This
window sets that up from a movie, trains it, and lets you watch it learn.

Two networks can be trained, chosen as *model* under the form's *more*:
*DECODE, optimised*, SMAPpy's own, by default, and *DECODE*, the
original's.  Both are trained from the same estimates, in the same job
folder, and fitted by the same [DECODE](plugin:Localize/DECODE) fitter; what
differs is the network and how it learns (the fitter's page compares them).

A training takes hours, on a GPU; a computer without one trains on a
workstation, over ssh.  Nothing is copied there but a folder of a few files:
the training draws its own frames, so the movie stays here.

## How it works

**Prepare.**  Choose the movie the model is for and its *PSF calibration* (from beads).
*Prefit* fits a few blocks spread over the movie with
[Spline 3D](plugin:Localize/Spline 3D) and links the localizations into
blinks.  From them come the training's ranges: the photons of a molecule in
a frame it is fully on in, the background, the z range and how many frames a
blink lasts.  They appear under *Estimates*, to change if you know better --
or, without a prefit, to type in yourself: *Write job* then writes the job
from them as they stand;
the figure shows the prefit's values with the ranges the training will draw
from over them.  With torch installed here, a frame as the training would
simulate it is drawn beside a frame of the movie, on one scale, with both
their pixel values: the spots should look as bright and as wide, the
background as grey and as grainy.  It is drawn again a moment after an
estimate is changed.  *Write job* writes the job folder beside the movie, in
`<movie>_decode/`, and lists it on the left.

**Train.**  Choose a job in the list, where it trains (*run on*), and
*Start*.  The
training runs as a program of its own, so it goes on when this window or
SMAPpy is closed, and is listed again when it opens.  *Stop* stops it after
the epoch it is in, with its checkpoint written; *Start* again goes on from
there.

**It stops when it no longer improves.**  *epochs* is the most it trains.
With *stop when it no longer improves* on, it finishes earlier once nothing
the test set measures has improved over the last 100 epochs, and never
before epoch 150:

- the loss by more than 1 % *of how far it has fallen so far* (from epoch
  10 to its best before the window) -- not of its value, which is the
  density's: 24 at 25 molecules in 40 x 40 pixels, 300 at 50;
- its [Jaccard index](plugin:Panels/Glossary) and its lateral and axial
  efficiencies by more than 0.005;
- its lateral and axial RMSE by more than 1 %.

(The Jaccard index is the molecules found divided by those found, missed and
invented together: 1 is perfect.  The efficiency is the SMLM Challenge's
score of the Jaccard index and the RMSE together.)  Each is first taken as
the median of 11 epochs, so one epoch that happens to come out well or badly
does not decide.  On seven trainings of 500 and 1000 epochs this stopped
between epochs 327 and 397, with the Jaccard within 0.005 of the last
epoch's.  A shorter window trusts a flat stretch too early: with 50 epochs
the same trainings lost up to 0.012 in Jaccard, with 75 up to 0.007.  The
RMSE alone is no measure of progress -- it rises while a training learns, as
it finds more of the harder molecules -- so it only keeps a training going
while it still falls.  The window, the minimum and the tolerances are under
*more*.  Turned off, it trains every epoch, as DECODE did.

**On a workstation.**  The workstations are listed in *File > Preferences*,
*Training hosts* ([how to connect one](plugin:Panels/Training hosts), step by
step) (*Training hosts...* at the top of this window, and at the
foot of the DECODE fitter's *model*, which offers the same list as its
*run on*): a name, what `ssh` is given (an alias of your
`~/.ssh/config`, or `user@host`), the folder job folders go to there, and
the command that trains there -- `smappy-decode` where SMAPpy is installed
with its `decode` extra.  ssh must log in without a password (a key): the
window cannot type one.  *Test* logs in and asks the command, and says which
GPU it found.  A job started there is copied there first; its progress is
copied back every 15 seconds, so it is followed here as one trained here,
and its model when it finished, stopped or is asked for (*Try*, *Use this
model*).  Its checkpoint stays there, and *Start* again resumes it there.  A
job records the SMAPpy version that prepared it, and a training refuses a
job of another version unless the host's *train jobs another SMAPpy version
prepared* is on.

**Follow.**  Nine panels, read from the job's progress file every few
seconds: the loss, the test set's Jaccard index, and where in their pixel
the latest epoch's localizations fall; the RMSE, each localization's error
over its molecule's own CRLB, and the seconds an epoch; and, in the bottom
row, the latest epoch's error over CRLB split by how close the nearest other
molecule is and by how many of the frames the network reads the molecule is
on in.  Above them the
state, the epoch, the time trained so far and at most how long is left; below
them what the stop rule waits for, with its numbers -- *waiting for: loss
(+2.6 % of its fall, needs ≤ 1.0 %) · settled: Jaccard, …*.  On the curves the
rule judges, the last 100 epochs are shaded and a dashed line marks their
median: where the curve stands now.  The first epochs (5 %, at least 5) are
left out of each panel's range, and the RMSE's and the error over CRLB's
before the Jaccard reaches a quarter of its best -- a handful of matches,
hundreds of nm off; they are still drawn, beyond the edge.

**Pixelation** is the network pulling its positions towards the pixel
centres: the histogram of where in their pixel the localizations fall --
0 the centre, 0.5 the edge -- is flat for unbiased positions and peaks at the
centre for pixelated ones.  The test set's molecules are placed uniformly, so
any shape is the network's.  It fades as a training goes on.

**Error over CRLB** divides each matched localization's error by a
[CRLB](plugin:Panels/Glossary) of its own molecule -- its photons,
background and z, the training's own PSF, doubled in variance with EM gain --
and reports how widely that scatters, as the width of a Gaussian (1.4826
times the median absolute deviation: two molecules merged or a mismatch,
whose errors are many bounds, do not count, with no cut-off to choose).  At
the bound it is 1.  Three bounds are computed for every molecule of the test
set when a training starts:

* **alone** -- the single-emitter bound: the molecule on its own spot with a
  background of its own.  It leaves the neighbours out, so a dense training
  stays above 1 against it: about 2 at 50 molecules in 40 x 40 pixels.
* **+ neighbours** -- the joint bound of the molecule and every molecule whose
  spot overlaps its own in the frame, all their positions, photons and the
  background fitted together.  A spot beside another is localized worse than
  alone, because some of its photons could be the other's; how much worse
  this bound says.  Where two molecules are closer than the spot's width it
  grows without limit -- they cannot be told apart -- while a network puts a
  merged pair near its middle, a pixel or so off.  So each position carries
  a prior of a pixel laterally and of the training's z range axially, which
  caps the bound there and moves a bound of a tenth of a pixel by half a
  percent.
* **+ neighbours, frames** -- the same over the frames the network reads (three
  by default): a molecule on in the frames before and after is at the same
  place there, so its position is one parameter over them, and its bound
  falls -- to the single frame's over the root of three when it is on in all
  three at the same brightness.

The error over CRLB panel draws the width against the first (solid) and the
last (dashed, *joint*).  The bottom row splits the latest epoch's: by the
nearest other molecule's distance, laterally and axially, and by the frames
read the molecule is on in, laterally, with the number of matches in each bin
in grey.  Against *alone* the closest pairs are far above 1 and molecules on
in three frames can be below it; against the last bound both should sit near
1 if the network uses what the frames hold.  A curve below 1 for the closest
pairs is not an error: the bound is for an estimator that is never biased,
and a network that has learnt where molecules lie can do better there.  The
joint bounds take 6 s at every start of a training at the HD density, 17 s
at UHD; *joint CRLB* under the Train tab's *more* turned off leaves them out,
and the panels then show the single-emitter bound alone.  A
training from before these numbers existed shows the panels empty, and so
does an optimised model's: its test set has the single-emitter bound and the
bound pooled over the frames read, not the joint ones, and *joint CRLB* does
not apply to it.


**Try.**  *Try on my data* runs the chosen job's newest model -- written
after every epoch, so a training still running can be tried -- on the first
20 frames of its movie, and draws the first frame with what the network
found there at a probability of at least 0.5.  It is a look, not a fit:
nothing is saved.  A two-channel model reads the frames cut into its
channels by the job's calibration at the movie's camera ROI, as the DECODE
fitter cuts them, and its localizations are drawn where channel 0 saw them.

**Use.**  *Use this model*, beside the jobs, puts the chosen job's newest
model into the DECODE fitter's *trained model*.

The DECODE fitter's *Train a model...* opens this window with the fitter's
movie, microscope, PSF calibration and *run on* filled in.

## Controls

### Training hosts...
*File > Preferences*, *Training hosts*: the workstations *run on* offers.
*run on* reads the list again when Preferences closes.

### ?
This page.

### Jobs
Every job this computer prepared or started, running ones first, each with a
symbol and a colour for its state:

- ▶ *running*, with its epoch;
- ○ *not started*;
- ⏸ *paused*: stopped with *Stop*; *Start* goes on from its last
  checkpoint;
- ✕ *failed*: an error, a training that refused to start, or one that
  *died* -- ended without saying so, because the computer slept or the
  process was killed.  *Start* goes on from its last checkpoint here too;
  the tooltip and *Follow* say why it failed;
- ✓ *done*: *converged at epoch N* when it stopped improving, or *all N
  epochs*, with the test set's Jaccard and lateral RMSE.

*run on* is where it trains.  A job on a workstation that could not be
reached says *not reached*, and its tooltip what ssh said.

Jobs in a training host's folder that this list lacks -- started from
another computer, or taken off this list -- are found every minute, in the
background, and listed in italics with ↗ before their name.  Double-click
one to add it: its job files are copied here, and its progress is followed
from then on as for a job sent from here.

### Use this model
Puts the chosen job's newest model into the DECODE fitter's *trained model*
(from a workstation, copied back first).  It does not fit anything.

### Add existing job...
List a job folder made elsewhere, or one taken off the list.  The movie is
not chosen here but under *Prepare*.

### Remove from list
Takes the job off the list; its folder stays where it is, and a training
running goes on.

### Prepare
The movie, its PSF calibration, the density to train at, the *model* to train
and a model to *start from*.

*model* (under *more*): *DECODE, optimised* trains SMAPpy's own network,
*DECODE* the original's.  For one channel the optimised network reads 7
frames through a ConvGRU and is trained on blinks lasting 1 to 3 frames on
average.  That one network served brief and long blinks within 2 % of a
network trained for either.  The prefit's *on-time* is therefore not used
for it yet; a short fine-tune to the measured on-time is to come.  For two
channels it reads 3 frames, and is trained on blinks of 1 frame on average,
which on simulated data did as well as mixed on-times on molecules on for 3
frames and better on brief dim ones.  The optimised one's job holds `settings.json` in
place of DECODE's `method.yaml`: the same photons, background, z range,
on-time, density and camera, SMAPpy's settings for the rest.  Its *Follow*
draws two more curves: the share of the test set's molecules more than 5
times their [CRLB](plugin:Panels/Glossary) off (*outliers*, on the Jaccard's
panel), and the RMSE of the rest over their CRLB (dotted, on the error /
CRLB panel).  Together they say how often it fails and how good it is
otherwise, at no cost: the bounds are the test set's, computed once.  A
two-channel job's bound is both channels' together, from the spots as the
training drew them: the molecule's x, y and z shared, its photons one count
(biplane) or one per channel (colour), each channel's own background.

*start from* is a warm start: the training begins with that model's
network instead of a random one, and learns the new data from there.  A
model trained for other data -- another sample, another density, a
calibration of another day -- is still a better start than a random
network: it already knows what a spot is, and in earlier tests with DECODE
it trained faster (not yet measured in SMAPpy).  It must be a network of the
same shape and of the same *model* (the window says so when it is not).  The DECODE fitter's *Train
a model...* fills in the fitter's own model.

The density is the number of molecules on in a frame, per µm² of the sample:
*HD* is 1, *UHD* 2.5, and *manual* takes any other number, typed in
*molecules*.  It is the sample's and the acquisition's: choose it for the data,
the same for either model.  The job itself counts molecules in a frame of
40 x 40 pixels, as SMAP's and DECODE's do; the window converts with the movie's
pixel size, so at 100 nm pixels (16 µm²) *HD* is 16 molecules in 40 x 40
pixels and *UHD* 40, and at 160 nm (41 µm²) 41 and 102.

The window starts on *HD*.  In the two-channel studies (pixels of 117 x 127 nm,
where 1 per µm² is 24 in 40 x 40 pixels) a model trained at 25 in 40 x 40
pixels was never far from the best at any density from 1 to 50: it found 2 to
3 % more molecules than a model trained at 50 on sparse data, tied with it at
25, and lost 1 to 2 % at 50.  Choose *UHD* for data you know to be dense.

**If in doubt, train at the higher density.**  A model trained dense
localizes sparse data well, but not the other way round.  A model trained at
0.6 per µm² found at most 1 % more than one trained at 1, and only on sparse
data; above it found fewer and misplaced more: at 2 per µm², 38 % of the
molecules it found against 27 % for a model trained at 2.  One trained at 0.2
per µm² gained only on the sparsest data and had twice the outliers even at its
own density.

*in the sample* appears for a calibration made with the pupil (Zernike)
method, and is [Spline 3D](plugin:Localize/Spline 3D)'s *focus*: the
calibration is drawn for a molecule at each height above the coverslip, with
the objective's focus *focus depth* into the sample, and the network learns
that PSF rather than the beads' on the coverslip.  Its z is then the height
above the coverslip.  *auto* finds the focus from the movie's lowest
molecules, as Spline 3D does, before the prefit; *in situ* learns it, in
minutes; *typed* takes it as typed; *as calibrated* uses the beads' spline
as it is.  Its *sample index* is Spline 3D's: what the sample is in -- on
*auto* the calibration's own, *blinking buffer* (1.345) for fixed cells in a
dSTORM buffer when the beads were in water.  The drawing never includes the bead's size, nor the beads'
damping of the light beyond the critical angle: those belonged to the
calibration, not to a molecule.  One network serves one focus: data taken
at another depth want a network of their own.  The DECODE fitter's
*Model info...* says which focus a model was trained for.

### two channels
A two-channel calibration (a `_2ct.h5`, from the PSF calibration window's
dual mode or converted from SMAP's) trains a network that reads a split
camera's two halves as one molecule.  This part appears for such a
calibration only.  *model* is then *DECODE, optimised* (its *aligned
channels*: the second channel resampled onto the first's pixels and read
together from the network's first layer) or *DECODE-Plex-like* (each
channel read apart and joined later); DECODE's original reads one channel.

*Prefit* fits the blocks with [Spline 3D 2C](plugin:Localize/Spline 3D 2C)
instead.  The estimates are of each molecule's total photons, as one
channel's are.  Beside them it finds what only two channels have:

- *channels are*, on *auto*: two focal planes when the calibration's two
  PSFs come into focus 200 nm or more apart, else two colours.
- each channel's background range, which the training draws apart;
- *read as the grid*, on *auto*: the brighter half.  On the C = 2 study's
  sets that half as the grid gave fewer outliers.
- for two colours, the dyes: where the molecules' shares of their photons
  in the second channel gather.  On simulated data drawn at 0.78 and 0.92
  they were found at 0.775 and 0.925.

*dyes* says what the training does with them:

- *independent counts*, the default: each channel's photons drawn on their
  own, for any dyes or several.
- *from the prefit* or *typed* (*shares*): each molecule one of those
  dyes, its share drawn around it by *spread*.  0.03 is narrow, 0.06 broad.
  On the study's crowded sets the broad prior had a quarter fewer outliers
  and named the dye right more often, but found fewer of the dimmest
  molecules.

*moved since the beads* moves the registration for the prefit, as the
DECODE fitter's own does for the fit.  A splitter that moved since the
beads (Fig. 2a's 5.8 rows) leaves Spline 3D 2C pairing the wrong spots.
The fitter's *measure the move* finds it.  The job draws the two channels
with the beads' registration either way: the network learns all of it.

A two-channel job's preview shows each channel's corner of the data, cut as
the network reads them (the main half on the left), beside the same two as
the training draws them.

### Prefit
Fits the blocks and fills *Estimates*, with the calibration drawn in the
sample where *in the sample* says so; the status line then names the focus.  It takes as long as Spline 3D takes
on those frames, seconds to a minute.  A sparse movie, where too few molecules were
seen fully on in a frame (fewer than 200 such frames), is fitted further,
four times the blocks at a time, up to the whole movie: the photons are
measured on those frames, and the rule that would stand in for them,
from whole blinks, can be several times off.

### Estimates
The training's ranges, editable, and typed by hand where no prefit was run.
The photons are those of a frame a molecule is fully on in, not of a whole
blink.

### Advanced: method.yaml
DECODE's whole parameter file, as it will be written, to edit by hand.  Used
as written when shown.  For *model* *DECODE* only: the optimised model has no
such file, and the button is off.

### Write job
Writes the job folder and selects it: from the prefit's estimates as edited,
or, with no prefit, from those typed in, the camera read from the movie and
*microscope*.

### Train
Where it trains (*run on*: this computer, or a training host), the most
epochs, and whether it may stop before them.

### Start
Starts the chosen job where *run on* says, or goes on with it from its last
checkpoint.  Here it needs torch, SMAPpy's `decode` extra; on a training
host the job folder is copied there first.

### Stop
Stops the chosen job after the epoch it is in.

### Follow
The chosen job's curves.

### Try
The chosen job's newest model on the movie it was prepared on.

### Try on my data
The first frames of the movie through the newest model, drawn with what it
found at a probability of at least 0.5.
