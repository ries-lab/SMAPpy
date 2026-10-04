---
version: "1"
covers: [smappy.plugins.npc.count_corners, smappy.plugins.npc.fit_circle]
---

## What it does

A nuclear pore has eight corners, and each corner of the Nup96 ring holds
four copies of the protein.  A corner shows in the image when at least one of
its four copies was labelled and detected.  How many of a pore's corners show
is therefore a measure of how well the sample was labelled.  Across many
pores, that number gives the *effective labelling efficiency*: the fraction
of the protein that was labelled and seen
(Thevathasan et al. 2019, [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)).

This evaluator counts the corners that show, one pore (one ROI) at a time.
The count goes into the site table, and
[NPC Labeling Efficiency](plugin:ROIManager/Analyze/NPC Labeling Efficiency)
turns the counts of all pores into the efficiency.

It is not in a new evaluation pipeline by itself: add it in the evaluation
window when the sites are pores.  It needs ROIs centred roughly on the pores,
such as those of [NPC](plugin:ROIManager/Segment/NPC), and pores seen face
on.  It uses the localization precision (`xy_err_nm`) when the table has it.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.plugins.npc import (NPCCornersSettings, count_corners, draw_corners,
                                true_corners)
from scipy.spatial import cKDTree
sim = SimulationSettings(n_frames=3000, seed=2,
                         structure=StructureSettings(preset="npc"),
                         labelling=LabellingSettings(efficiency=0.3))
locs = simulate(sim)
xy = np.column_stack((locs["x_nm"], locs["y_nm"]))
precision = np.asarray(locs["xy_err_nm"])
tree = cKDTree(xy)
copies = np.column_stack((locs.metadata["copies"]["x_nm"],
                          locs.metadata["copies"]["y_nm"]))
labelled = true_corners(locs)
settings = NPCCornersSettings()
```

**1. Centre.**  A circle of the set *radius* is fitted to the ROI's
localizations, which places the pore's centre.  The ROI's own centre is the
starting point.

**2. The ring.**  The localizations within ±*ring width* of the ring are
kept, as long as their precision is good enough to tell one corner from the
next: better than 0.4 of the distance between two corners along the ring.
At the default radius that is 15.7 nm.

**3. Rotation.**  The pore can be turned by any angle.  Its rotation is the
angle at which the eight corners best match the kept localizations.  Precise
localizations count more than imprecise ones.

**4. Count.**  The ring is cut into eight equal segments, each centred on a
corner.  A segment with at least *min per corner* localizations is a corner
that shows.

```figure Six simulated pores at 30 % labelling: the localizations kept (blue) and the others (grey), the ring band (dotted), the borders between the segments, and the corners counted (filled red).  The truly labelled corners of each pore are given in the title, beside the count.
fig.set_size_inches(7.5, 5)
axes = fig.subplots(2, 3)
for ax, copy in zip(axes.ravel(), range(6, 12)):
    near = tree.query_ball_point(copies[copy], 150)
    x, y = xy[near, 0], xy[near, 1]
    values = count_corners(x, y, copies[copy] + 10, settings, precision[near])
    draw_corners(ax, x, y, values, settings, precision[near])
    ax.set_title(f"counted {values['n_corners']}, labelled {labelled[copy]}", fontsize=9)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])
```

## In detail

**The centre** minimises $\sum_i \rho(d_i - R)$ over all the ROI's
localizations, with $d_i$ the distance of localization $i$ from the centre, $R$
the *radius*, and $\rho$ the soft-L1 loss with a scale of the *ring width*
(see [NPC](plugin:ROIManager/Segment/NPC)).  The radius reported is then
fitted free, on the localizations within $R + \Delta R$ of that centre;
$\Delta R$ is the *ring width*.

**The kept localizations** are those with
$R - \Delta R < \rho_i < R + \Delta R$ and $\sigma_i < 0.4 \cdot 2\pi R / n$,
with $\rho_i$ the distance from the centre, $\sigma_i$ the precision
(`xy_err_nm`) and $n$ the *corners*.  Without a precision column, every
localization in the band is kept.

**The rotation** is the weighted circular mean of the angles $\theta_i$ taken
$n$ times over, so that all eight corners fall on one direction:

$$\phi = \frac{1}{n} \arg \sum_i w_i e^{i n \theta_i}, \qquad w_i = \left( \frac{\rho_i}{\sigma_i} \right)^2 ,$$

the weight being one over the square of the angular precision.  The corners
are at $\phi + 2\pi k / n$, and a localization belongs to the corner $k$ whose
segment, of width $2\pi/n$ centred on it, holds its angle.

**Stray localizations.**  A localization between two corners, such as a fit
of two molecules that were on in the same frame, opens a corner by itself
when *min per corner* is 1.  In simulations this adds about 0.2 corners per
pore at a labelling efficiency of 0.15, 0.07 at 0.3, and nearly nothing at
0.5.  A minimum of 2 removes it but loses corners whose label blinked only
once, which costs more: it lowers the efficiency by 0.02 to 0.05.

## Parameters

### radius_nm
The radius of the protein's ring, a little less than the outer edge seen in
the image.  50 nm, with a *ring width* of 20 nm, takes 30 to 70 nm from the
centre, enough for Nup96 at 54 nm.

### min_locs
1 is right for ungrouped data, where a molecule usually shows in several
localizations.  See *In detail* for what changes at 2.

## Output

One row per ROI in the site table:

* `n_corners`, the corners that show, 0 to *corners*;
* `n_ring_locs`, the localizations counted (in the band, precise enough);
* `n_within_locs`, all localizations within $R + \Delta R$ of the centre;
* `radius_nm`, the radius fitted free;
* `rotation_deg`, the rotation $\phi$, between $-180/n$ and $180/n$ degrees;
* `x_nm`, `y_nm`, the fitted centre.

The figure shows the pore with the ring band, the segment borders and the
corners: filled when counted, empty when not.  It is redrawn for each ROI as
the ROI manager walks the list.  A good pore shows its localizations in
clusters, one in each counted segment.  Clusters cut by a border mean the
rotation is off.  This happens on pores with one or two corners, where the
rotation is poorly defined, and does not change their count much.

## Differences from SMAP

Ported from SMAP's `ROIManager/Evaluate/NPCLabelingQuantify_s`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* **Rotation.**  SMAP fits the sawtooth $\mathrm{mod}(\theta - a, \pi/4)$ to a
  constant by least squares, started from a circular mean.  Here the weighted
  circular mean is the answer: it has no start value and no local minima.
* **Centre.**  SMAP fits it by plain least squares; here a soft-L1 loss keeps
  the background in the ROI from pulling it.
* **Left out.**  SMAP's split of the count by time (the frames before and
  after each of 50 time points), and its counts on the simulation's true
  positions.  The truth is compared in
  [NPC Labeling Efficiency](plugin:ROIManager/Analyze/NPC Labeling Efficiency)
  instead, from the simulation recipe, without extra columns.

## References

* Thevathasan JV, Kahnwald M, Cieśliński K, et al. Nuclear pores as
  versatile reference standards for quantitative superresolution
  microscopy. *Nat Methods* 16, 1045 (2019).
  [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
