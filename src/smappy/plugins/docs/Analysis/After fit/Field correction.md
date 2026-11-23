---
version: "1"
covers: [smappy.field_correction.measure, smappy.field_correction.FieldCorrection, smappy.field_correction.basis, smappy.field_correction.kept, smappy.field_correction.corrected, smappy.field_correction.has_field, smappy.plugins.field_correction.CorrectField, smappy.plugins.field_correction.fitted_with, smappy.plugins.field_correction.calibration_for, smappy.plugins.field_correction.draw_correction]
---

## What it does

The PSF is not the same everywhere on the camera.  Across a large field of
view its [astigmatism](plugin:Panels/Glossary) and coma change, so the shape a
molecule at a given height makes depends on where it is.  A PSF calibration
with *field dependence* learns this from the beads and saves it as splines on
a grid of places over the chip (node splines).
[Spline 3D](plugin:Localize/Spline 3D) blends them for each spot and fits
each molecule with the PSF of its own place.

A fit with one PSF for the whole chip, the calibration's centre spline,
is wrong away from the centre.  Its z is scaled and bent differently at
each place, by tens of nanometres at the edges of a typical field.  DECODE's
network is such a fit: it is trained on the centre spline.  This plugin
measures that bias and subtracts it from `x_nm`, `y_nm` and `z_nm`, as a
smooth function of where each localization was fitted.

It runs as a step of [*after the fit*](plugin:Analysis/After fit/After the fit),
offered only where the fit's calibration has a field.  In Spline 3D, ticking
it makes the fit use the centre spline and correct afterwards, instead of
using the node splines.  Here it runs on an open table that was fitted with
such a calibration.

Use it for DECODE fits with a field-dependent calibration.  For Spline 3D,
the node splines are the better choice.  They match a fit with the exact local
PSF and keep its precision, at 1.7 times the cost of one spline.  The
correction moves the mean, but a molecule fitted with the wrong PSF has lost
precision that nothing gives back.

## How it works

```figure-setup
import numpy as np
from smappy.calibrate.core import spline_coefficients
from smappy.field_correction import measure
from smappy.field_splines import FieldSplines, blend
from smappy.io.calibration import SplineCalibration
from smappy.plugins.field_correction import draw_correction
from smappy.psf import SplinePSF

NZ, BOX, ROI = 41, 15, 11

def astigmatic(e):
    """A Gaussian stand-in for a measured PSF: two focal lines e apart, in units of the stack."""
    z = np.linspace(-1, 1, NZ)[:, None, None]
    y, x = np.mgrid[:BOX, :BOX] - (BOX - 1) / 2
    sx, sy = 1.3 + 0.8 * (z - e) ** 2, 1.3 + 0.8 * (z + e) ** 2
    return (np.exp(-x ** 2 / (2 * sx ** 2) - y ** 2 / (2 * sy ** 2)) / (2 * np.pi * sx * sy)).astype(np.float32)

ell = lambda x, y: 0.1 + 0.3 * x / 1000                     # astigmatism growing across the chip in x
xs = ys = np.linspace(0, 1000, 5)
field = FieldSplines(np.stack([spline_coefficients(astigmatic(ell(x, y))) for y in ys for x in xs]), xs, ys)
cal = SplineCalibration(blend(field, (500, 500)), 50.0, NZ // 2, x0=(BOX - 1) / 2, field_splines=field)
correction = measure(cal, 3000.0, 10.0, roi=ROI)
```

**The bias is measured from the calibration itself.**  The node splines are
the PSF of each place.  Spots are drawn with them on an 11 × 11 raster over
the field, every 25 nm in z, at the table's median photons and background.
They are noise-free, so what comes out is the bias alone.  Each spot is
fitted with the centre spline, and the fit's error is noted against where the
fit put the spot.  Nothing is measured on beads: the model knows where every
spot truly was.  Beads would also bring their own size, and their own
positions, which are not known.

**A smooth function of the fitted position.**  The errors in x, y and z are
fitted with polynomials of the chip position times polynomials of the fitted
z.  The correction then reads only what every localization has: where it was
fitted, and how high.

```figure The z bias the correction subtracts, against the fitted z, at the centre of the field (black), the middle of its edges (blue) and its corners (red).  At the centre the centre spline is right and the bias is zero; at the two edges in x, where the stand-in's astigmatism is weaker or stronger than at the centre, z is stretched or squeezed, by up to a hundred nanometres.
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 3.6))
draw_correction(ax, correction)
```

**Subtracted, never extrapolated.**  Each localization's bias is evaluated at
its fitted chip position and z, and subtracted.  Outside the field the
node splines cover, or outside the z range the bias was measured on, a
localization takes the value at the edge.  The plugin warns with how many
there were.

```figure Spots drawn at the field's edge (x = 950 px), fitted with the centre spline: the z found against the true z before (grey) and after the correction (red).  Before, z is squeezed and bent; after, it lies on the line.
import matplotlib.pyplot as plt
k = np.arange(10, 31)
s = (BOX - ROI) // 2
rois = (astigmatic(ell(950, 300))[k, s:s + ROI, s:s + ROI] * 3000 + 10).astype(np.float32)
truth = (NZ // 2 - k) * 50.0
plain = SplineCalibration(cal.coeff, 50.0, NZ // 2, x0=(BOX - 1) / 2)
fit = SplinePSF(plain)
out = fit.unpack(fit.fit(rois, 50, 1))
d, _ = correction(950 + out["x_roi"] - (ROI - 1) / 2, np.full(len(k), 300.0), out["z_nm"])
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.plot(truth, truth, color="0.7", lw=1)
ax.plot(truth, out["z_nm"], "o", color="0.4", ms=4, label="centre spline")
ax.plot(truth, out["z_nm"] - d[:, 2], "o", color="C3", ms=4, label="corrected")
ax.set(xlabel="true z (nm)", ylabel="z found (nm)")
ax.legend(frameon=False)
```

## In detail

The spots are drawn at the ROI's central pixel, with the photons and
background of the table's median localization.  They are fitted with 50
iterations of the spline fitter, in the fit box the fit used (its record's,
else 13 pixels).

The z range is the spline's, less a tenth of it at each end, where a fit runs
into the spline's edge.  A fit that ends more than 300 nm from the truth is
counted as lost.  A fit that ends outside the range is left out of the
polynomial fit.

The polynomials are the monomials $u^i v^j$ with $i + j \leq 4$, where $u$
and $v$ are the chip position scaled to $[-1, 1]$ over the node splines'
field.  They are multiplied by Legendre polynomials $P_0 \ldots P_{14}$ of
the fitted z, scaled to $[-1, 1]$ over the range.  The bias wiggles along z:
degree 6 left 21 nm in the study below, degree 14 left 2.7 nm.  Each axis is
fitted on its own by least squares.

**Why at the table's brightness.**  The bias of a fit with the wrong PSF
depends a little on photons and background, even without noise.  The fit
converges to the model PSF closest to the data, and the Poisson weights
decide which is closest.  With background, those weights shift between the
spot's core and its tails as the photons fall.  In the simulation study, a
correction measured at 20 000 photons and applied at 1000 left 3–6 nm rms in
z.  One measured at the data's brightness left 0.2–2 nm.

**When it fails.**  Beyond about twice the field of a well-aligned
microscope, the fitted z is no longer a smooth function of the true one.  It
is scaled down to a third, or the fit is lost.  No correction can recover
that.  The plugin measures how well the polynomial followed the bias on its
own raster (rms in z).  It warns when that is above 5 nm, or when more than
2 % of the raster's fits were lost.  Then fit with the field: Spline 3D
without this step.

**The depth in the sample.**  When the fit's spline was drawn for a focus
inside the sample (Spline 3D's *focus*, or the depth DECODE's network was
trained at), the node splines are drawn there too, and the bias is measured
there.  The table's record names the depth.

The measured correction is kept for the session, for each calibration, depth,
fit box and brightness (to 5 %), so a batch measures it once.  It takes
about a second.

## Parameters

### calibration
Leave it empty: the table's record names the calibration and the depth it
was drawn for.  A file typed here is drawn as calibrated, unless it is the
table's own.

### photons
Type it only to measure at another brightness than the table's median: for
a table whose molecules are a mix of very dim and very bright ones, say.

### background
As *photons*: the table's median is right for almost every table.

### fit_box
The fit box changes the bias a little, because a larger box sees more of the
spot's tails.  The fit's own is used when the table records it.

## Output

`x_nm`, `y_nm` and `z_nm` corrected on every localization.  The table records
the correction in its metadata (`field_correction`): the calibration, the
depth, the brightness, the ranges, and how well the polynomial followed the
bias.  A second run on the same table is refused, because it would subtract
the bias twice.

The result line says by how much z moved (rms), and how well the correction
followed the bias.  The figure is the first one above, for the table's
calibration: the bias subtracted against z at the centre, the edges and the
corners.  Curves that grow far apart, or a warning that the correction
failed, mean the field is too strong for one spline.

## References

- The simulation study behind it: `docs/uipsf/field-dependence.md`, §11, in
  SMAPpy's repository.
