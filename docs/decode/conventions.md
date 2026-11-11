# DECODE in SMAPpy: the conventions

The port keeps DECODE's conventions, the integration SMAPpy's (`poc-plan.md` §7, "SMAPpy's
notation"); the boundary between them is `decode_smlm/utils/calibration_io.py`, `predict.py` and
`io.py`.  These are the places where the two differ, each of which has cost a pixel once.

## The axes

* **SMAPpy:** x along the columns, y along the rows; pixel centres at whole numbers, 0-based.  A
  calibration is `(64, nz, ny, nx)` in SMAPpy's monomial order, in the orientation of the camera
  its beads were taken with.
* **DECODE:** x is a frame's first axis, the rows -- `tifffile` gives `(n, rows, columns)`, and
  `Offset2Coordinate` and the PSF kernel index x first; a SMAP spline's first axis is the rows too.
  So DECODE's x is SMAPpy's y.  `calibration_io` transposes a SMAPpy calibration to DECODE's
  `(ny, nx, nz, 64)` with the two lateral monomial digits exchanged; without a mirror the
  coefficients are DECODE's own reading of SMAP's file, bitwise.
* **Out of the predictor** come SMAPpy's pixel columns (`x_pix` along the columns), on the frames
  as given; the engine adds the camera ROI's corner and the pixel size.

## The mirror

An EMCCD read out through its EM register mirrors the image in x.  SMAP builds an EMCCD's spline
from x-mirrored beads, and DECODE takes that spline as it is -- its Fit notebook flips the frames
to match, a line "you might need to comment out".

SMAPpy holds a calibration in its beads' orientation instead and records their EM state
(`em_on`).  A model file records it too (`calibration_em_on`), and the fitter flips the frames
when the data's EM state differs (`plugins.fit_decode._mirrored`), and the positions back.  Where
either side does not say, nothing is flipped, as Spline 3D flips nothing.  The un-mirrored
coefficients are moved one voxel along the columns, because DECODE's ROI window is not
mirror-symmetric; then the PSF is the file's mirrored to 1.5e-7 (`poc-plan.md` §8, "(c)").

## The calibration's centre

`x0`, the spline's lateral centre: SMAP writes it 1-based, SMAPpy 0-based (`(roi_size - 1) / 2`).
`calibration_io` reads SMAP's convention where the file carries SMAP's parameters (`emmirror`,
`ROIxy`) and SMAPpy's elsewhere, and refuses a centre that is not the grid's middle.  Before this,
a model trained on a SMAPpy calibration localized a whole pixel off (the pixelation study,
`poc-plan.md` §8) -- unnoticed until its fit was compared with Spline 3D's spot by spot.

## Photons

The fitter converts counts as every SMAPpy fitter does, `(counts - offset) * e/ADU / EM gain`,
without the excess-noise factor (the network learnt the noise as it is); the predictor then takes
off the training's quantum efficiency and spurious charge as DECODE's `Photon2Camera.backward`
did.  The network is trained on photons (`convert2photons`), so a model carries its camera's
noise but not its gain.
