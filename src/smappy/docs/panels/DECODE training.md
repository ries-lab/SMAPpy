---
title: DECODE training
summary: Prepares a DECODE training from a movie -- a short Spline 3D fit of it sets what the training simulates -- writes it as a job folder beside the movie, trains it on this computer or a workstation over ssh, follows its curves and tries the model on the data.
widget: smappy.gui.train_window.TrainWindow
covers: [smappy.gui.train_window.run_prefit, smappy.gui.train_window.prepared_by_hand, smappy.gui.train_window.write_job, smappy.gui.train_window.default_folder, smappy.gui.train_window.start_here, smappy.gui.train_window.stop, smappy.gui.train_window.draw_progress, smappy.gui.train_window.try_on_data, smappy.gui.train_window.start_on_host, smappy.gui.train_window.HostFollower, smappy.training.jobs.status, smappy.training.remote, smappy.gui.preferences.HostsPage, smappy.gui.train_window.simulated_frames, smappy.training.reference.crlb_reference, smappy.training.prefit.prefit, decode_smlm.prefit.estimates]
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

**Train.**  Choose a job on the left, where it trains (*run on*), and
*Start*.  The
training runs as a program of its own, so it goes on when this window or
SMAPpy is closed, and is listed again when it opens.  *Stop* stops it after
the epoch it is in, with its checkpoint written; *Start* again goes on from
there.

**On a workstation.**  The workstations are listed in *File > Preferences*,
*Training hosts* (*Training hosts...* at the top of this window, and at the
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
Every job this computer prepared or started, with its state and where it
trains: *new* (not trained yet), *running*, *interrupted* (its training
stopped without saying so: the computer slept or the process was killed --
*Start* goes on from its last checkpoint), *stopped*, *finished* or *error*
(also a training that refused to start; *Follow* says why).  A job on a
workstation that could not be reached says *not reached*, and its tooltip
what ssh said.

### Add folder...
List a job folder made elsewhere, or one taken off the list.

### Remove from list
Takes the job off the list; its folder stays where it is.

### Prepare
The movie, its PSF calibration and the density to train at.  *LD*, *HD* and *UHD*
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
Where it trains (*run on*: this computer, or a training host) and how long.  The test loss is within 2 % of its final value after
about 150 epochs; the example data's models were trained for 1000.

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
