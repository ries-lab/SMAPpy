---
version: "1"
covers: [smappy.plugins.roi_centre.CentreROIs.run, smappy.plugins.roi_centre.roi_centre, smappy.plugins.circle_fit.circle_fit, smappy.roi_manager.core.ROIProject.move_roi]
---

## What it does

Moves every used [ROI](plugin:Panels/Glossary) (region of interest: a shape
kept in the [ROI manager](plugin:Panels/ROI manager), inside which things
are measured) so that its centre sits on the centre of the structure in it.
A segmenter puts an ROI where its filter peaked, a hand where the eye
landed; a measurement that assumes the structure in the middle -- a radial
profile, an average over many ROIs, a line through the centre -- wants the
ROI on it.

The centre is that of a fitted circle, by the same fit as
[Circle fit](plugin:Analysis/Measure/Circle fit), for rings such as nuclear
pores; or the median of the localizations, for a blob or anything not round.
Run it after the ROIs are found and before they are measured, in a chain as
well.

## How it works

```figure-setup
from smappy.plugins.circle_fit import circle_fit
rng = np.random.default_rng(2)
truth = np.array([(400.0, 400.0), (1200.0, 450.0), (800.0, 1150.0)])
offsets = np.array([(70.0, -50.0), (-60.0, -60.0), (40.0, 75.0)])
xs, ys = [], []
for c in truth:
    phi = rng.uniform(0, 2 * np.pi, 120)
    r = 50 + rng.normal(0, 7, 120)
    xs.append(c[0] + r * np.cos(phi)); ys.append(c[1] + r * np.sin(phi))
x, y = np.concatenate(xs), np.concatenate(ys)
size = 260
moved = []
for c in truth + offsets:
    for _ in range(2):
        inside = np.hypot(x - c[0], y - c[1]) <= size / 2
        fit = circle_fit(x[inside], y[inside])
        c = np.array([fit["x_nm"], fit["y_nm"]])
    moved.append(c)
```

**1. Each ROI's localizations.**  For each used ROI, the localizations
inside it as the ROI manager shows them: the layer's
[filter](plugin:Panels/Glossary) (the lowest and highest values of columns a
localization must lie between), and merged blinks where the layer merges
them.

**2. Its centre.**  A circle is fitted to them (*a fitted circle*) and its
centre taken -- with the radius fitted too, or fixed at *fixed radius* --
or the median of their x and of their y (*the median*).

**3. The move.**  The ROI is moved so that its centre is there; a drawn
outline and the ROI's named lines move with it.  With *iterations* above
1 the ROI is measured again from its new place and moved again: an ROI far
off its structure sees only part of it at first, and more after the first
move.

```figure Three rings and their ROIs as a segmenter left them (dashed), off the rings' centres, and after Centre ROIs with two iterations (solid): each sits on its ring.
from matplotlib.patches import Circle
ax = fig.subplots()
ax.plot(x, y, ".", color="0.6", markersize=2)
for c in truth + offsets:
    ax.add_patch(Circle(c, size / 2, fill=False, ls="--", color="#d62728"))
for c in moved:
    ax.add_patch(Circle(c, size / 2, fill=False, color="#1f77b4"))
ax.plot(truth[:, 0], truth[:, 1], "k+", markersize=8)
ax.set_aspect("equal")
ax.set_xlim(100, 1500); ax.set_ylim(150, 1450)
ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
```

All the moves are one step of the ROI manager's undo (*Undo ROI edit*).
Results measured before the move describe the old place, so the per-region
steps measure the moved ROIs again.

## In detail

**When an ROI is left where it is.**  An ROI without localizations, or with
too few for a circle (4, or 3 with a fixed radius), is not moved; neither is
one whose measured centre falls outside the ROI itself.  A circle fitted to
a few background localizations can land anywhere, and an ROI walked off its
structure is worse than one a little off-centre.  The text says how many
were left.

**The median** is of x and of y separately.  On a ring cut by the ROI's
edge it is pulled towards the part inside; *iterations* takes it the rest
of the way.

## Parameters

### centre
*a fitted circle* for rings; it finds the centre of a ring whose
localizations are all on one side of the ROI, where the median would not.
*the median* for anything that is not a ring.

### radius_nm
For rings of known size, such as nuclear pores (about 50 nm): the centre is
then found even when only part of the ring is labelled.

### iterations
1 when the ROIs are already close, 2 or 3 when a segmenter left them a
third of their size or more off.

## Output

The ROIs move in the ROI manager; the line under the panel says how many
moved, by how much (median and largest), and how many were left.

## Differences from SMAP

Ported from SMAP's evaluator *CenterSites*, *centre all* in the ROI manager
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* **Fewer ways.**  SMAP also offers the mean, and a *mask*: a ring template
  correlated with the site's image.  The mean is the median's less robust
  twin; finding rings with a template is the job of
  [Segment regions](plugin:Analysis/Regions/Segment regions).
* **The circle** is the soft-L1 fit of Circle fit, radius fixed or free;
  SMAP's *fitring* is a least-squares circle of fixed radius.
* **A centre outside the ROI is not taken.**  SMAP moves the site wherever
  the fit went.
* **No z.**  SMAP can also centre a site in z; an ROI here has no z.
* **A Regions plugin**, so a chain can centre the ROIs between finding and
  measuring them, rather than an evaluator that moves the site it measures.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
