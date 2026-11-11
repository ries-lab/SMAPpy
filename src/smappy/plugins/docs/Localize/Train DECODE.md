---
version: "1"
covers: [smappy.plugins.train_decode.TrainDecode, smappy.gui.train_window.TrainWindow]
---

## What it does

The [DECODE](plugin:Localize/DECODE) fitter needs a network trained for the
microscope and the data: their PSF, their camera, how bright the molecules are
and how dense.  This opens the
[DECODE training window](plugin:Panels/DECODE training), which prepares such a
training from a movie, runs it on this computer or a workstation, follows it
and tries the model on the data.  The window's page explains how.

The DECODE fitter's *Train a model...* button opens the same window, with the
fitter's movie, camera and calibration as its start.

This section only opens the window, with the *Open* button in its title bar.
In a script, a job folder is prepared with `smappy.training.job.write` and
trained with `smappy-decode train FOLDER`.
