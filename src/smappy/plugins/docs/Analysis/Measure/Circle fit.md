---
version: "1"
covers: [smappy.plugins.circle_fit.circle_fit, smappy.plugins.npc.fit_circle, smappy.plugins.circle_fit.draw, smappy.plugins.circle_fit.CircleFit.run]
---

## What it does

Fits a circle to the localizations: where its centre is, how large its
radius is, how precisely both are known, and how well a circle describes
them.  It is for structures that are rings -- nuclear pores, endocytic
pits seen from above, ring-shaped assemblies -- and for finding the centre
of anything round.

It runs on the view, on the localizations in the drawn
[ROI](plugin:Panels/Glossary) (region of interest), and, because it is a
measurement, also in every ROI of the [ROI manager](plugin:Panels/ROI manager)
or every [cluster](plugin:Panels/Glossary) as one of the
[per-region steps](plugin:Panels/Glossary): a radius per pore, to histogram
or to filter by.  [Centre ROIs](plugin:Analysis/Regions/Centre ROIs) moves
each ROI onto the centre it finds.

Each visible [layer](plugin:Panels/Glossary) (one picture of the
localizations, with its own filter and colours) is fitted on its own, as the
layer shows it: its filter, and merged blinks where it merges them.  A
circle needs at least 4 localizations (3 with a fixed radius); in a region
with fewer the numbers are left empty.

The fit takes a few milliseconds, so the plugin offers a *follow ROI* tick:
the circle is fitted again while the ROI is dragged.

## How it works

```figure-setup
import os, tempfile
import yaml
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.plugins.circle_fit import circle_fit, draw

# rings of 50 nm radius on a grid
truth_r = 50.0
centres = [(x, y) for x in range(600, 9000, 800) for y in range(600, 9000, 800)]
structure = {"elements": [{"circle": truth_r, "centre": [cx, cy, 0], "density": 0.06}
                          for cx, cy in centres]}
path = os.path.join(tempfile.mkdtemp(), "rings.yaml")
with open(path, "w") as f:
    yaml.safe_dump(structure, f)
locs = simulate(SimulationSettings(n_frames=4000, seed=3,
                                   structure=StructureSettings(file=path),
                                   labelling=LabellingSettings(efficiency=1.0)))
bright = np.asarray(locs["photons"]) > 200
x, y = np.asarray(locs["x_nm"])[bright], np.asarray(locs["y_nm"])[bright]
s = np.asarray(locs["xy_err_nm"])[bright]

def near(c, radius=100):
    keep = np.hypot(x - c[0], y - c[1]) < radius
    return x[keep], y[keep], s[keep]

fits = [circle_fit(*near(c)) for c in centres]
```

**1. A start.**  The fit starts at the median of the localizations, with
the median distance from there as the radius.  The median, not the mean, so
a few stray localizations do not move the start.

**2. The circle.**  The centre and the radius are then adjusted until the
distances of the localizations from the circle are as small as they can be
-- in the sense of a *soft* least squares: a localization close to the circle
counts with its squared distance, as in ordinary least squares, but one
further than about *outlier scale* counts only with its distance.  So the
unspecific localizations inside and around a ring, which ordinary least
squares would let pull the circle towards them, have little say.  With a
*fixed radius*, only the centre is moved, which holds even on a ring of
which only an arc was labelled; it then starts from the free circle's
centre, since from the median, on the arc, the circle mirrored onto the
arc's outer side fits as well.

```figure One simulated ring of 50 nm radius: each label blinks several times, so the localizations come in small groups.  The fitted circle (black) with its centre lies on the true ring (dashed).
ax = fig.subplots()
c = centres[10]
px, py, ps = near(c)
draw(ax, [{"name": "", "colour": "#1f77b4", "x": px, "y": py, "values": fits[10]}])
t = np.linspace(0, 2 * np.pi, 200)
ax.plot(c[0] + truth_r * np.cos(t), c[1] + truth_r * np.sin(t), "--", color="#d62728", lw=1)
```

**3. How precise, how good.**  From how the localizations scatter around
the circle, and how many there are, the plugin estimates the error of the
centre and of the radius: roughly the scatter divided by the square root of
their number.  How well a circle describes them is the root mean square
distance of the localizations from it; where the table has the
[localization precision](plugin:Panels/Glossary), also that distance in
units of each localization's own precision (the reduced chi-square): about
1 for a ring thinner than the precision, more for a wider ring, or for
something that is not a ring at all.

```figure Fitted radius of each of the simulated rings, with its error bar; the truth is the red line.  The error bars cover the truth about two times in three, as one standard deviation should.
ax = fig.subplots()
r = np.array([f["radius_nm"] for f in fits])
e = np.array([f["radius_err_nm"] for f in fits])
order = np.argsort(r)
ax.errorbar(np.arange(len(r)), r[order], yerr=e[order], fmt=".", color="#1f77b4",
            ecolor="0.6", elinewidth=0.8)
ax.axhline(truth_r, color="#d62728", lw=1)
ax.set_xlabel("ring (sorted by radius)")
ax.set_ylabel("fitted radius (nm)")
```

## In detail

**The loss.**  For localizations at $(x_i, y_i)$ and a circle of centre
$(x_0, y_0)$ and radius $r$, the residual of each is its distance from the
circle,

$$d_i = \sqrt{(x_i - x_0)^2 + (y_i - y_0)^2} - r .$$

What is minimised is the soft-L1 loss
$\sum_i 2 c^2 \left( \sqrt{1 + (d_i/c)^2} - 1 \right)$, with $c$ the
*outlier scale*: quadratic for $|d_i| \ll c$, linear for $|d_i| \gg c$.  It
is minimised by Gauss-Newton with reweighting -- each step a weighted least
squares with weights $w_i = 1/\sqrt{1 + (d_i/c)^2}$ -- for up to 30 steps,
or until no parameter moves by more than 0.001 nm.  This is the same fit the
ring template of [Segment regions](plugin:Analysis/Regions/Segment regions)
uses for each candidate pore.

**The errors** are those of a weighted least-squares fit at its minimum:
the covariance $s^2 (J^T W J)^{-1}$, with $J$ the derivatives of the $d_i$
by $x_0$, $y_0$ and $r$, $W$ the weights, and
$s^2 = \sum_i w_i d_i^2 / (n - p)$, $p$ the number of fitted parameters (3,
or 2 with a fixed radius, whose error is then 0).  On simulated rings they
match the scatter of the fitted radius over repeats.  For very few
localizations, or a ring with only a short arc labelled, they are
optimistic; so they are when the localizations of one blink, which lie
together, count separately -- on a layer that merges blinks each counts
once.

**Goodness of fit.**  `rms_nm` is $\sqrt{\frac{1}{n}\sum_i d_i^2}$, over
all localizations, so the unspecific ones raise it.  `chi2_red` is
$\frac{1}{n-p}\sum_i (d_i/\sigma_i)^2$ with $\sigma_i$ each localization's
precision (`xy_err_nm`).

**Background.**  The soft loss limits how far unspecific localizations
pull, but does not remove them: localizations spread evenly over the ROI
lie mostly outside a ring smaller than it and pull its radius outwards by a
few nm when they are as many as a third of the ring's.  The centre stays.
Filter them out before (by precision, by merging blinks), or keep the ROI
close around the ring.

**The radius.**  A ring's localizations are spread around its true radius
by their precision.  The radius fitted is the distance of the ring of
labels, plus a small bias of order $\sigma^2 / 2r$ (under 1 nm for 8 nm
precision on a 50 nm ring); the centre has none.

## Parameters

### radius_nm
Fix it when the radius is known (a nuclear pore's) and the rings are sparse
or only partly labelled: the centre is then much better determined.  Leave
it at 0 to measure the radius.

### scale_nm
About the width of the ring as it appears -- its own width and the
precision together, 10 to 30 nm.  Much smaller and the fit sees only the
localizations right on the circle; much larger and it becomes ordinary
least squares, which background pulls.

## Output

* **The numbers**, per layer (the second layer's with `_L2`, and so on):
  `n_locs`, the centre `x_nm`, `y_nm` and the radius `radius_nm`, with
  their errors `x_err_nm`, `y_err_nm`, `radius_err_nm`, and `rms_nm` and
  `chi2_red`.  In the ROI manager they are columns of the results table.
* **The figure**: the localizations with the fitted circle and its centre.

A good fit has the circle through the middle of the ring, a radius error a
small fraction of the radius, and a reduced chi-square not far above 1.  A
circle through a blob or a crescent fits too -- every set of points has a
best circle -- but with a large `rms_nm`, a large chi-square and a radius
that differs from ROI to ROI.

## Differences from SMAP

Ported from SMAP's evaluator *circleFitter*
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), the part that fits
a circle to the localizations.

* **A robust fit.**  SMAP minimises the algebraic distance of the points
  from a circle by least squares (`implicitfit`), which the localizations
  inside a ring pull towards the centre; here the geometric distance, with
  the soft-L1 loss.
* **Errors and a goodness of fit.**  SMAP gives none.
* **Left out**: SMAP's second fit of a blurred ring to a rendered image of
  the ROI, with a pixel size and blur of its own, the counts within 1.5
  radii and the asymmetry.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
