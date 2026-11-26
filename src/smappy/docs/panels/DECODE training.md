---
title: DECODE training
summary: Prepares a DECODE training from a movie -- a short Spline 3D fit of it sets what the training simulates -- writes it as a job folder beside the movie, trains it on this computer or a workstation over ssh, follows its curves and tries the model on the data.
widget: smappy.gui.train_window.TrainWindow
covers: [smappy.gui.train_window.run_prefit, smappy.gui.train_window.run_prefit_two, smappy.netfit.prefit_two.prefit_two, smappy.netfit.prefit_two.dye_shares, smappy.netfit.prefit_two.kind_of, smappy.netfit.job.two_channel_settings, smappy.gui.train_window.two_channel_problem, smappy.gui.train_window.netfit_params, smappy.netfit.job.settings_from_estimates, smappy.netfit.job.train_folder, smappy.netfit.metrics.outliers_and_inliers, smappy.gui.train_window.prepared_by_hand, smappy.gui.train_window.write_job, smappy.gui.train_window.default_folder, smappy.gui.train_window.start_here, smappy.gui.train_window.stop, smappy.gui.train_window.draw_progress, smappy.gui.train_window.try_on_data, smappy.gui.train_window.start_on_host, smappy.gui.train_window.HostFollower, smappy.training.jobs.status, smappy.training.remote, smappy.training.remote.others, smappy.training.remote.adopt, smappy.training.progress.converged, smappy.training.progress.stop_status, smappy.training.reference.robust_width, smappy.training.reference.subpixel_histogram, decode_smlm.training.test_crlbs, smappy.training.joint_crlb.bounds, smappy.gui.train_window._split_panel, smappy.gui.train_window.rule_line, smappy.training.jobs.Status, smappy.gui.preferences.HostsPage, smappy.gui.train_window.simulated_frames, smappy.gui.train_window.simulated_frames_two, smappy.gui.train_window.channels_side_by_side, smappy.netfit.bounds.drawn_crlb, smappy.training.prefit.prefit, decode_smlm.prefit.estimates, smappy.gui.train_window.in_the_sample, smappy.calibrate.depth.for_sample, smappy.gui.train_window.psf_kind, smappy.gui.train_window.model_choices, smappy.gui.train_window.on_time_range, smappy.gui.train_window.prefit_blocks, smappy.gui.train_window.frames_read, smappy.gui.train_window.run_prefit_4pi, smappy.gui.train_window.fourpi_params, smappy.gui.train_window.fourpi_job_calibration, smappy.gui.train_window.simulated_frames_4pi, smappy.gui.train_window.fourpi_side_by_side, smappy.netfit.fourpi.prefit, smappy.netfit.fourpi.cavity_motion, smappy.netfit.fourpi.settings_from_prefit, smappy.gui.train_window.run_prefit_gaussian, smappy.gui.train_window.gaussian_drawn, smappy.calibrate.gaussian_psf.gaussian_calibration, smappy.calibrate.gaussian_psf.sigma_range, smappy.calibrate.gaussian_psf.as_widths, smappy.calibrate.gaussian_psf.dual_calibration, smappy.calibrate.gaussian_psf.width_ratio, smappy.netfit.prefit_two.prefit_two_gaussian, smappy.gui.train_window.run_prefit_gaussian_two, smappy.gui.train_window.gaussian_transform]
---

## What it does

[DECODE](plugin:Localize/DECODE) localizes with a neural network that has
learnt what molecules look like on this microscope and this camera.  It
learns that from frames it simulates: molecules drawn with the bead
calibration's [PSF](plugin:Panels/Glossary), as bright, as dense and over the
same background and z range as in the data, with the camera's noise.  This
window sets that up from a movie, trains it, and lets you watch it learn.

The calibration decides which networks can be trained, chosen as *model*
under the form's *more*; a line under *PSF calibration* says what kind it
is.  For one channel: *DECODE, optimised*, SMAPpy's own, by default, and
*DECODE*, the original's.  For two channels (a split camera, biplane or two
colours): *DECODE, transformed channels* by default, and
*DECODE-Plex-like*.  For a 4Pi model (the four phase channels of a 4Pi
microscope): *DECODE, 4Pi*, which also gives each molecule's interference
phase.  With *PSF* *Gaussian* there is no file: the *channels* part
decides, one channel or two.  All are trained from the same estimates, in the same kind of job
folder, and fitted by the same [DECODE](plugin:Localize/DECODE) fitter; what
differs is the network and how it learns (the fitter's page compares them).

**Without beads.**  2D data rarely have a bead calibration.  With *PSF* set
to *Gaussian*, the network learns Gaussian spots instead, as wide as the
data's spots are: the PSF that [Gaussian 2D](plugin:Localize/Gaussian 2D)
fits.  It then reports each spot's width, `sigma_nm`, where a network
trained on a calibration reports z.

A training takes hours, on a GPU; a computer without one trains on a
workstation, over ssh.  Nothing is copied there but a folder of a few files:
the training draws its own frames, so the movie stays here.

## How it works

**Prepare.**  Choose the movie the model is for and its *PSF calibration* (from beads).
*Prefit* fits blocks of 50 frames spread over the movie (*prefit frames*,
500 in all) with
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

**With Gaussians**, the prefit is [Gaussian 2D](plugin:Localize/Gaussian 2D)
with a free width, and the estimates hold the range of widths the training
draws (*width* ... *to*) in place of z.

**Train.**  Choose a job in the list, where it trains (*run on*), and
*Start*.  The
training runs as a program of its own, so it goes on when this window or
SMAPpy is closed, and is listed again when it opens.  *Stop* stops it after
the epoch it is in, with its checkpoint written; *Start* again goes on from
there.

**It stops when it no longer improves.**  An optimised model trains on at
most *samples* simulated samples (6 million: about 180 epochs of 32,768),
set when its job is prepared; *DECODE*'s most is *epochs*, which the Train
tab shows for a DECODE job only.
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

### PSF
What the network learns the PSF from.  *PSF calibration* is a calibration
file: a bead average or a pupil model, whichever the file holds (the window
reads which, and offers *in the sample* for a pupil model).  *Gaussian* needs
no file: the training draws Gaussian spots over a range of widths, for 2D
data taken without beads.

A molecule out of focus is wider, and a Gaussian widens alike above and below
focus, so z would be ambiguous: a network cannot learn whether a wide spot
was above or below.  The Gaussians are therefore drawn by their width, not
by z, and the network reports the width.  Each spot is the Gaussian
integrated over the pixels and holds all its photons, so the network's
photons are Gaussian 2D's.  Behind the scenes the Gaussians are written as an
ordinary spline file, `calibration.h5` in the job folder, with the width
where a bead spline has z; both networks train on it unchanged.

What a Gaussian leaves out is the real PSF's rings and tails.  The positions
suffer least from that, because the error is symmetric; at high photon counts
some of the tails go into the background, and the photons come out a little
low, as they do with Gaussian 2D.

### channels
With *PSF* *Gaussian*: the chip read whole, or a split camera's *two
channels*, with the same fields as the fitters' *channels* part.  For two
channels the prefit is [Gaussian 2D 2C](plugin:Localize/Gaussian 2D 2C), and
the training draws each molecule in both channels as two colours.  Both
channels' spots come from one width axis: a molecule out of focus is wider in
both.  The second channel's spot is a fixed number of times as wide as the
main channel's.  That ratio is the median of the molecules' width ratios
(the estimates' *second channel's width*).  The job's calibration is then a
two-channel file of two Gaussian splines, with the registration below; it
trains *DECODE, transformed channels* or *DECODE-Plex-like*, which read two
channels.

### channel registration
With two Gaussian channels: where the second channel lies on the chip, as
Gaussian 2D 2C finds it.  *from this movie* measures it once before the
prefit, from Gaussian fits of frames spread over the movie, as the fitter
does.  *file* reads one from *Calibrate transform* or an earlier run.  The
fitter later moves it, typed or measured, as for a bead calibration.

### Prepare
The movie, its PSF calibration, the density to train at, the *model* to train
and a model to *start from*.

The line under *PSF calibration* says what the file is: one channel (a
bead average, or a pupil model drawn at the focus in the sample, and whether
it varies over the chip), two channels (two focal planes or two colours,
told by whether the two PSFs come into focus 200 nm or more apart), or 4Pi
(a 4Pi model, as [Spline 4Pi](plugin:Localize/Spline 4Pi) fits with).

*model* (under *more*) offers what the calibration trains.  One channel:
*DECODE, optimised* trains SMAPpy's own network, *DECODE* the original's.
Two channels: *DECODE, transformed channels* (the second channel resampled
onto the first's pixels through the calibration's registration, both read
together from the network's first layer) or *DECODE-Plex-like* (each
channel read apart and joined later).  A choice the new calibration does
not offer moves to its default.

*frames read* (under *more*, the optimised models only) is how many frames
the network reads around the one it fits, through a ConvGRU.  On *auto* it
is 7 for one channel and 3 for two.  For one channel 7 frames were 1.6 nm
better than 5 on blinks lasting 3 frames, and 3 frames lost 1.3 nm on blinks
of 1; a shorter window trains and fits faster.  *samples* is how many
simulated samples it trains on at most: 6 million, past which the studies
saw the Jaccard no longer change and the error over CRLB fall by under 2 %.
The optimised one's job holds `settings.json` in
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

*uneven backgrounds* (*auto*: on for two channels, off for one; SMAPpy's networks only, not *DECODE*,
whose simulator draws a flat background): half the training's movies have a
background that varies smoothly over the field, up to about ten times
between its brightest and darkest parts, as a cell's autofluorescence or an
out-of-focus nucleus makes it; the network learns to fit that background
under each molecule.  On the two-channel studies' test sets it lost nothing
to a model trained on flat backgrounds and halved the pixelation (the pull
of positions towards pixel centres).  On a live cell it found half as many
molecules again.  For one channel it is off on *auto*: on the one-channel
test sets it cost 0.005 in Jaccard and doubled the pixelation, and no
one-channel live-cell movie with a truth has shown its gain there yet;
choose *on* for a cell with a strongly uneven background.

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
calibration only.  *model* is then *DECODE, transformed channels* or
*DECODE-Plex-like* (above); DECODE's original reads one channel.

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

*channel move* is the *move* of the two-channel fitters' *channel
registration*: *none*, *typed* or *measured*.  A splitter that moved since the beads (Fig. 2a's 5.8 rows)
leaves Spline 3D 2C pairing the wrong spots; the prefit moves the
registration by it first.  *measured* measures it as
[Spline 3D 2C](plugin:Localize/Spline 3D 2C) does, with x and y free in
each channel over the whole movie, and keeps it beside the movie, where the
fitter finds it again.  Over time, the prefit takes its mean.  The job draws
the two channels with the beads' registration either way: the network
learns that, and the fitter moves it.

A two-channel job's preview shows each channel's corner of the data, cut as
the network reads them (the main half on the left), beside the same two as
the training draws them.

### 4Pi
A 4Pi model (the PSF calibration window's 4Pi mode, or uiPSF's) trains
*DECODE, 4Pi*: a network that reads the four phase channels at once and gives
each molecule's x, y, the envelope's z, its photons and its interference
phase.  This part appears for such a model only: *channels* and
*registration* say where the channels are on the movie's frames and how they
map onto the first, as in Spline 4Pi, and the training draws its frames in
that layout -- the job's calibration is the 4Pi model with the movie's
layout and maps written in.

*Prefit* runs Spline 4Pi and [4Pi z](plugin:Analysis/After fit/4Pi z) on the
movie's first frames (*prefit frames*, at least 2000: 4Pi z follows the
cavity's phase over a quarter of them).  It gives the photons' range (every
fit counted: a 4Pi table is not grouped into blinks, so the on-time is not
measured and the training draws 1 -- 3 frames), each channel's background
range, z, and how fast the cavity's phase drifts, which the status line
says.  On the Ries-lab Nup96 and the 2017 tubulin movies it drifts by about
3e-5 rad a frame, far less than the 0.002 the training draws by default.

The preview shows the four channels of the data cut onto the network's grid,
side by side, beside the same four as the training draws them.  The job is
trained and followed as any other, and its model fitted by the DECODE fitter,
which then runs 4Pi z after the fit.  On simulations such a network reached
the phase's CRLB, and on the Nup96 it found a third more pores than Spline
4Pi, the two rings as far apart.

### Prefit
Fits *prefit frames* of the movie (under the form's *more*: 500 by default,
in blocks of 50 consecutive frames spread from its start to its end) and
fills *Estimates*, with the calibration drawn in the
sample where *in the sample* says so; the status line then names the focus.  It takes as long as Spline 3D takes
on those frames, seconds to a minute.  A sparse movie, where too few molecules were
seen fully on in a frame (fewer than 200 such frames), is fitted further,
four times the blocks at a time, up to the whole movie: the photons are
measured on those frames, and the rule that would stand in for them,
from whole blinks, can be several times off.

### Estimates
The training's ranges, editable, and typed by hand where no prefit was run.
The photons are those of a frame a molecule is fully on in, not of a whole
blink.  Fits far brighter than the rest (more than three robust standard
deviations above the median, in log photons) are left out: they are
overlapping molecules, and in a crowded movie they alone would set the
spread, which the training would then fill with molecules too dim to find.

*DECODE* draws a molecule's photons from a normal distribution, the mean
and spread shown.  SMAPpy's networks draw them uniformly over a range
instead, shown as *photons* ... *to*: from half the fully-on fits' 1st
percentile to 1.5 times their 99th.  A draw about the mean almost never
trains the brightest molecules a crowded movie holds, and the network then
fits them worse; on a live cell, a network trained over the range was as
precise as DECODE-Plex's, one trained from the mean and spread half as
precise again.

*DECODE* trains at one *on-time*: the frames a blink is seen in, on
average, as the prefit measured it.  SMAPpy's networks draw each training
sample's mean on-time over a range instead, *on-time* ... *to*: one network
for every dye whose mean lies in it.  This on-time is an exponential's mean,
so a blink is seen in about one frame more.  For one channel the range starts
at 1 to 3 frames: one network trained over it served brief and long blinks
within 2 % of a network trained for either.  For two channels it starts at 1,
which on simulated data did as well as 1 to 3 on molecules on for 3 frames
and better on brief dim ones.  Either is widened to take in the prefit's
on-time where it lies outside.  The same number at both ends trains one
on-time.

The z range is the 2nd to 98th percentile of the localizations' z, of the
fits that pass the same likelihood cut as the fully-on frames, widened by
15 % of that range on either side, never narrower than -500 to 500 nm --
a model is often used again on a sample focused elsewhere -- and kept inside
the calibration's.  Type a wider range for a model to serve other samples.  Poor
fits are mostly molecules far out of focus.  With them included, a live
cell's range reached the calibration's limits, and the network was
trained for z where there were hardly any molecules.

With *PSF* *Gaussian* the widths take the place of z: *width* ... *to*, the
Gaussian's sigma in the sample, in nm.  The prefit's range is the 2nd to
98th percentile of the widths Gaussian 2D fitted in frames a molecule was
fully on in, widened by 15 % of that range on either side, and never
narrower than 15 % below and above their median, so that a model serves spots
a little sharper or wider than one movie's.  The partly-on first and last
frames of a blink are left out because they are dim and their widths
scatter.  Type the range instead where no prefit was run, or for a model to
serve other data: about 1 to 1.6 pixels' worth at focus, more for a thick
sample.  With two Gaussian channels *width* is the main channel's, and
*second channel's width* how many times as wide the other channel's spot is.

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
epochs (for a *DECODE* job; an optimised one trains to its *samples*), and
whether it may stop before them.

### Start
Starts the chosen job where *run on* says, or goes on with it from its last
checkpoint.  Here it needs torch, SMAPpy's `decode` extra; on a training
host the job folder is copied there first.

Under the Train form the window says how long the chosen job will take where
*run on* says: the epochs it will most likely run (an optimised model's
*samples*; for *DECODE*, about 400 with *stop when it no longer improves*,
else every epoch) times the seconds an epoch takes there.  Those seconds come
from your earlier jobs on the same machine and device when there are any, and
otherwise from benchmark rates (an H100, an M1 Max), so it is an estimate to
the nearest hour, not a promise.  When the job would train on the CPU, or
longer than 12 hours, the line turns orange with what an NVIDIA GPU would
take, and *Start* asks before it begins: a training on the CPU takes days.

### Stop
Stops the chosen job after the epoch it is in.

### Follow
The chosen job's curves.

### Try
The chosen job's newest model on the movie it was prepared on.

### Try on my data
The first frames of the movie through the newest model, drawn with what it
found at a probability of at least 0.5.  The status line says how many
molecules per µm² it found a frame, and the density it was trained at.  When
it finds more, it says so: a model trained at a higher density (*UHD*, or the
number typed under *manual*) will find more and place them better.  This is
the check on the density the prefit cannot make.  In a crowded frame the
prefit's fit separates only a fraction of the molecules: on a live cell
crowded in its first frames it saw 0.16 per µm², where the network found
more than 1.  A model trained too sparse still finds more than it was trained
at, so a first short training, tried, tells you which density to train at.
