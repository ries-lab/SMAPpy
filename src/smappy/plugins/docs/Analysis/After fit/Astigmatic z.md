---
version: "1"
covers: [smappy.astigmatic_z.z_from_widths, smappy.astigmatic_z.Defocus, smappy.astigmatic_z.AstigmaticCalibration, smappy.astigmatic_z.fit_calibration, smappy.astigmatic_z.simulator_calibration, smappy.astigmatic_z._one_to_one, smappy.astigmatic_z._refine, smappy.astigmatic_z._grid_search, smappy.plugins.astigmatic_z.AstigmaticZ, smappy.plugins.astigmatic_z.calibration_of, smappy.plugins.astigmatic_z.draw_widths, smappy.plugins.astigmatic_z.draw_z, smappy.psf.GaussianPSF]
---

## What it does

A cylindrical lens in the detection path gives a molecule's image
[astigmatism](plugin:Panels/Glossary): wide in x when the molecule is above
the focus, wide in y when it is below, and round in between.  How elongated
the spot is says how high the molecule is.  Reading that height from the two
widths of an elliptical Gaussian is the original way of 3D localization
microscopy (Huang et al. 2008,
[doi:10.1126/science.1153529](https://doi.org/10.1126/science.1153529)), and
it is what this plugin does: it writes `z_nm` and its uncertainty `z_err_nm`
onto every localization.

It needs the widths: fit with [Gaussian 2D](plugin:Localize/Gaussian 2D) and
*elliptical* ticked, which writes `sigma_x_nm` and `sigma_y_nm` and their
errors.  And it needs to know how the widths change with z, a *calibration*:
measured on beads in the [PSF calibration](plugin:Panels/PSF calibration)
window with the method *elliptical Gaussian*, or typed.  For simulated frames
the simulator's own curves can be typed (the defaults are its defaults), and
*Fit this* in *Simulate* types them in.

Gaussian 2D runs it by itself once the fit is done: with *elliptical* ticked,
its [*after the fit*](plugin:Analysis/After fit/After the fit) offers
*astigmatic z*, with these settings under *z calibration*.  It runs there
before any drift correction, so that the drift is corrected in z as well.

Use it for a quick 3D view of astigmatic data, or when no spline calibration
is at hand.  A spline fit ([Spline 3D](plugin:Localize/Spline 3D)) uses the
measured PSF itself, every pixel of it, and is more precise: an elliptical
Gaussian is not quite the shape of a defocused spot, and two widths say less
than the whole image.

## How it works

```figure-setup
import tempfile
from pathlib import Path
import tifffile
from scipy.spatial import cKDTree
from smappy.plugins import Context
from smappy.plugins.fit import (CameraSettings, GaussianFit, GaussianFitSettings,
                                GaussianModelSettings, OutputSettings, SourceSettings)
from smappy.simulate import ISOLATED_NM, LabellingSettings, camera_frames
from smappy.astigmatic_z import simulator_calibration, z_from_widths
from smappy.plugins.astigmatic_z import draw_widths
frames, truth = camera_frames(300, seed=2, labelling=LabellingSettings(efficiency=0.1),
                              astigmatism=(300.0, 400.0))
folder = Path(tempfile.mkdtemp())
tifffile.imwrite(folder / "frames.tif", frames)
fit = GaussianFitSettings(source=SourceSettings(path=str(folder / "frames.tif")),
                          camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1),
                          model=GaussianModelSettings(elliptical=True),
                          output=OutputSettings(path=""))
locs = GaussianFit().run(Context(), fit).locs
cal = simulator_calibration()
found = z_from_widths(locs["sigma_x_nm"], locs["sigma_y_nm"], cal,
                      locs["sigma_x_err_nm"], locs["sigma_y_err_nm"])
# each localization's true z: the isolated, bright spot it came from, in its frame
true_z = np.full(len(locs), np.nan)
clean = (np.asarray(truth["photons"]) > 500) & (np.asarray(truth["neighbour_nm"]) > ISOLATED_NM)
lf, tf = np.asarray(locs["frame"]), np.asarray(truth["frame"])
for f in np.unique(lf):
    here, there = np.flatnonzero(lf == f), np.flatnonzero((tf == f) & clean)
    if len(there):
        d, i = cKDTree(np.c_[truth["x_nm"][there], truth["y_nm"][there]]).query(
            np.c_[locs["x_nm"][here], locs["y_nm"][here]])
        true_z[here[d < 50]] = np.asarray(truth["z_nm"])[there[i[d < 50]]]
```

**1. The calibration.**  Each width follows the curve of a defocused
Gaussian beam: narrowest at its own focus and growing on either side,

$$s(z) = s_0 \sqrt{1 + \left( \frac{z - c}{d} \right)^2}$$

where $s_0$ is the narrowest width, $c$ the height at which that axis is in
focus and $d$ the *focal depth*, how far from focus the width has grown by
$\sqrt{2}$.  The cylindrical lens puts the two foci apart: x in focus above
the round spot, y below.  A calibration from beads adds Huang's two terms
for what a real objective does far from focus (*In detail*).  Its z is 0
where the spot is round.

```figure Simulated astigmatic frames fitted with the elliptical Gaussian: each spot's two widths (blue x, red y) at its true height, on the calibration's curves.  The widths follow the curves, and the scatter about them is what limits z.
fig.set_size_inches(6, 3.2)
ax = fig.subplots()
has = np.isfinite(true_z)
draw_widths(ax, found.z_nm[has], locs["sigma_x_nm"][has], locs["sigma_y_nm"][has], cal,
            truth=true_z[has])
```

**2. z from both widths.**  For each localization the plugin looks for the
height whose two widths on the curves are closest to the two it measured,
each weighted by how well the fit measured it (`sigma_x_err_nm`,
`sigma_y_err_nm`).  Both widths are used, not only how elongated the spot
is, so a spot that is too wide in both directions -- two molecules
overlapping, a molecule far out of focus -- is recognised: it matches no
height.

**3. The error.**  `z_err_nm` is small where the widths change fast with z
and were measured well, and large for a dim spot, whose widths are
uncertain.  A spot whose widths match no height gets a correspondingly
larger error, and one whose best height is the end of the calibrated range
-- whose true z could be anywhere beyond it -- gets the whole range as its
error.  So a filter on `z_err_nm` in the Render tab removes them without a
flag column of their own.

```figure The z found from the widths against the true z, the isolated spots of the same simulation: on the line, with no offset and no change of scale.  The bars are each one's z_err_nm.
fig.set_size_inches(6, 3.4)
ax = fig.subplots()
has = np.isfinite(true_z) & (found.z_err_nm < 200)
ax.errorbar(true_z[has], found.z_nm[has], yerr=found.z_err_nm[has], fmt=".", ms=3,
            color="0.3", ecolor="0.75", elinewidth=0.6)
ax.plot([-400, 400], [-400, 400], color="C1", lw=1)
ax.set_xlabel("true z (nm)")
ax.set_ylabel("z from the widths (nm)")
```

## In detail

**The curves.**  One per axis,

$$s(z) = s_0 \sqrt{1 + u^2 + A u^3 + B u^4}, \quad u = \frac{z - c}{d}$$

with $A = B = 0$ for the plain beam (typed curves have them under *more*).
Far from focus the polynomial under the root can turn negative with $A$ and
$B$; it is held at $10^{-6}$ there, so the width and its slope stay finite.
z = 0 is where $s_x = s_y$ -- a calibration from beads is moved there after
its fit -- and the calibration carries the range of z it is trusted over:
the beads' range, or *z range* for typed curves.

**The estimate.**  For widths $\sigma_x, \sigma_y$ with errors
$\delta_x, \delta_y$, z minimises

$$\chi^2(z) = \left( \frac{\sigma_x - s_x(z)}{\delta_x} \right)^2 + \left( \frac{\sigma_y - s_y(z)}{\delta_y} \right)^2$$

within the range.  The errors are the fit's
[CRLB](plugin:Panels/Glossary) (Cramér-Rao lower bound: the precision the
fitter expects from the photons, the background and the PSF), which Gaussian
2D writes for the elliptical fit.  Huang et al. minimised a distance between the square roots of the
widths instead, an unweighted stand-in for these weights.

The minimum is found in two steps, vectorised over the whole table.  The
start is a lookup of $\sigma_x^2 - \sigma_y^2$, the quantity SMAP looked z up
by: tabulated on 2001 points of the range, it is one-to-one with z on a
stretch around the round spot (for the plain curves with equal depths,
everywhere -- it is linear in z), and one interpolation gives every
localization a start.  The few whose value lies beyond that stretch start
from a grid search on 201 points.  Six Gauss-Newton steps then refine each
start, no step longer than a tenth of the range.  A million localizations
take about 0.6 s; a grid search for all of them took 3.5 s.

**The error.**  From the CRLB through the curves' slopes,

$$\frac{1}{\delta_z^2} = \left( \frac{s_x'(z)}{\delta_x} \right)^2 + \left( \frac{s_y'(z)}{\delta_y} \right)^2$$

With two widths and one unknown, $\chi^2$ at the minimum follows a
chi-squared distribution with one degree of freedom when the curves describe
the spot; $\delta_z$ is multiplied by $\sqrt{\chi^2}$ where that exceeds 1
(Birge's ratio).  On fitted simulated frames this leaves `z_err_nm` about a
fifth larger than the scatter of z about the truth: a little conservative,
and much larger for a spot whose widths disagree.  A z that ends on an end
of the range gets the range's width as its error, as does a row
without widths; their z is the end, or NaN.  All errors are capped at the
range's width.

**Without width errors** -- a table from a fitter that did not write them --
every width gets one error, estimated from the residuals at the minimum: the
median of $\chi^2$ with one degree of freedom is 0.455, so
$\delta^2 = \mathrm{median}(r_x^2 + r_y^2) / 0.455$.  The result line says
so, with the value.

**Which table.**  The localizations, whatever the layer shows: z is computed
for each from its own widths, and a grouped layer's blinks then take the
mean z and the combined error, as any fitted z.  Every row is computed, not
only the filtered ones.

## Parameters

### calibration
*from a file* for real data, made from beads in the PSF calibration window
(its *Use in Astigmatic z* puts the file into Gaussian 2D's *after the fit*,
with *elliptical* and *astigmatic z* ticked).  *typed* for simulated
frames, or curves taken from elsewhere.

### path
A file the PSF calibration window's *elliptical Gaussian* method saved,
`<name>_astigz.h5` beside the bead stacks.  The beads should have been fitted
in the data's fit box.

### typed
The defaults are the simulator's: widths of 130 nm, x in focus at +300 nm
and y at −300 nm, focal depths of 400 nm.

### typed.z_range_nm
Keep it to the heights the curves describe.  Too narrow, and molecules beyond
it end on its edge with a large error; too wide, and far out of focus the
two widths grow alike and say little about z.

### z_scale
Beads on the coverslip measure how the widths change as the *objective*
moves.  For a molecule in water above the coverslip, seen through an oil
objective, the [index mismatch](plugin:Panels/Glossary) (refractive-index
mismatch: oil and water bend light differently) moves the focus less than the
objective moves, so its apparent z is compressed; this factor stretches it
back.  Typical values are 0.7–0.8 for
oil into water; 1 for simulated data or an index-matched sample.  Both `z_nm`
and `z_err_nm` are multiplied by it.

## Output

`z_nm` and `z_err_nm` on every localization, replacing any the table had.
The result line gives the median error, how many localizations ended on an
edge of the range (or had no widths) and how many have widths that match no
height ($\chi^2 > 9$, which one in 370 good spots reaches by chance).

The figure has two panels: the widths of the localizations at the z found,
over the calibration's curves -- a good fit has them along the curves, with
the spots that match no height away from both -- and a histogram of z, the
localizations with an error under a tenth of the range in grey, the rest in
orange.  A pile of orange at the ends of the range means molecules beyond
the calibration: a calibration from a longer stack, or a narrower filter on
`z_err_nm`, is the answer.

## References

- Huang B, Wang W, Bates M, Zhuang X (2008) Three-dimensional super-resolution
  imaging by stochastic optical reconstruction microscopy. *Science*
  319:810–813.  The method and the defocus curves with their $A$ and $B$.
  [doi:10.1126/science.1153529](https://doi.org/10.1126/science.1153529)
