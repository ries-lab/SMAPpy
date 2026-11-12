---
title: DECODE training
summary: Prepares a DECODE training from a movie -- a short Spline 3D fit of it sets what the training simulates -- writes it as a job folder beside the movie, trains it on this computer or a workstation over ssh, follows its curves and tries the model on the data.
widget: smappy.gui.train_window.TrainWindow
covers: [smappy.gui.train_window.run_prefit, smappy.gui.train_window.prepared_by_hand, smappy.gui.train_window.write_job, smappy.gui.train_window.default_folder, smappy.gui.train_window.start_here, smappy.gui.train_window.stop, smappy.gui.train_window.draw_progress, smappy.gui.train_window.try_on_data, smappy.gui.train_window.start_on_host, smappy.gui.train_window.HostFollower, smappy.training.jobs.status, smappy.training.remote, smappy.training.remote.others, smappy.training.remote.adopt, smappy.training.progress.converged, smappy.training.jobs.Status, smappy.gui.preferences.HostsPage, smappy.gui.train_window.simulated_frames, smappy.training.reference.crlb_reference, smappy.training.prefit.prefit, decode_smlm.prefit.estimates]
---

## What it does

[DECODE](plugin:Localize/DECODE) localizes with a neural network that has
learnt what molecules look like on this microscope and this camera.  It
learns that from frames it simulates: molecules drawn with the bead
calibration's [PSF](plugin:Panels/Glossary), as bright, as dense and over the
same background and z range as in the data, with the camera's noise.  This
window sets that up from a movie, trains it, and lets you watch it learn.

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
the test set measures has improved over the last 100 epochs: its loss by
more than 0.5 %, its [Jaccard index](plugin:Panels/Glossary) and its lateral
and axial efficiencies by more than 0.005, its lateral and axial RMSE by
more than 1 % -- and never before epoch 150.  (The Jaccard index is the
molecules found divided by those found, missed and invented together: 1 is
perfect.  The efficiency is the SMLM Challenge's score of the Jaccard index
and the RMSE together.)  Each is first taken as the median of 11 epochs, so
one epoch that happens to come out well or badly does not decide.  On the
trainings of the proof of concept (1000 epochs each) this stopped between
epochs 340 and 576, with the Jaccard and both efficiencies within 0.005 of
the 1000th epoch's: half to two thirds of the time saved.  The RMSE alone is
no measure of progress -- it rises while a training learns, as it finds more
of the harder molecules -- so it only keeps a training going while it still
falls.  Turned off, it trains every epoch, as DECODE did.

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

**Follow.**  The losses, the test set's Jaccard and RMSE, and the seconds
an epoch, read from the job's progress file every few seconds, with the time
left.  The dashed lines on the RMSE are the
[CRLB](plugin:Panels/Glossary) of the median spot the training draws: its
photons, background and z drawn as the training draws them, alone and
without noise, from the job's calibration, the bound SMAPpy's spline fitter
gives there -- doubled in variance with EM gain, whose excess noise the
bound leaves out.  It is a guide, not a bound on the curve: the test set's
spots overlap, and its RMSE is over the spots found, of which the dim ones
are fewer.  A curve that levels out well above it has room left; one near it
has little.

**Try.**  *Try on my data* runs the newest model on the first frames of the
movie and draws what it found.  *Use this model* puts it into the DECODE
fitter.

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

### Add existing job...
List a job folder made elsewhere, or one taken off the list.  The movie is
not chosen here but under *Prepare*.

### Remove from list
Takes the job off the list; its folder stays where it is, and a training
running goes on.

### Prepare
The movie, its PSF calibration, the density to train at, and a model to
*start from*.

*start from* is a warm start: the training begins with that model's
network instead of a random one, and learns the new data from there.  A
model trained for other data -- another sample, another density, a
calibration of another day -- is still a better start than a random
network: it already knows what a spot is, and in earlier tests with DECODE
it trained faster (not yet measured in SMAPpy).  It must be a network of the
same shape (the window says so when it is not).  The DECODE fitter's *Train
a model...* fills in the fitter's own model.

*LD*, *HD* and *UHD*
are SMAP's: 15, 25 and 50 molecules in a frame of 40 x 40 pixels.  A model
trained dense also localizes sparse data well: the example package's, trained
at 50, found 96 % of the molecules at one in 40 x 40 pixels.

### Prefit
Fits the blocks and fills *Estimates*.  It takes as long as Spline 3D takes
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
as written when shown.

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

### Use this model
Puts the newest model into the DECODE fitter's *trained model* (from a workstation,
copied back first).
