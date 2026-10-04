---
version: "2"
covers: [smappy.plugins.npc.count_corners, smappy.plugins.npc.segments, smappy.plugins.npc.fit_circle]
---

## What it does

A nuclear pore has eight corners, and each corner of the Nup96 ring holds
four copies of the protein.  How many corners a pore shows, and how many
localizations it has, together tell how well the sample was labelled and how
often a fluorophore blinks (Thevathasan et al. 2019,
[doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)).

This evaluator measures both, one pore (one ROI) at a time:

* the **corners** the pore shows, counted with its precise localizations;
* its **localizations**, a little less strictly selected;
* and **SMAP's corner count**, for SMAP's way of analysing them.

The numbers go into the site table, and
[NPC Labeling Efficiency](plugin:ROIManager/Analyze/NPC Labeling Efficiency)
turns them into the labelling efficiency.  Its page explains why the counts
are taken this way.

It is not in a new evaluation pipeline by itself: add it in the evaluation
window when the sites are pores.  It needs ROIs centred roughly on the pores,
such as those of [NPC](plugin:ROIManager/Segment/NPC), pores seen face on,
grouped localizations, and the localization precision (`xy_err_nm`).

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.plugins.npc import NPCCornersSettings, count_corners, draw_corners, true_corners
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

**1. Centre.**  A circle of the set *radius* (55 nm) is fitted to the ROI's
localizations better than *corner precision* (20 nm), starting from the
ROI's centre, and again to those near the first answer.

**2. The ring band.**  The localizations better than *corner precision*, from
40 to 70 nm from the centre (*radius* ± *ring width*), are the ones the
corners are counted with.  Nearer the centre a corner's segment is narrower
than a localization's spread, so those are left out.

**3. Rotation.**  The pore can be turned by any angle.  Its rotation is the
angle at which the eight corners best match the counted localizations,
precise ones counting more.

**4. Corners.**  The ring is cut into eight equal segments, each centred on a
corner.  A segment with a counted localization is a corner seen:
`n_corners`.

**5. Localizations.**  The localizations better than *count precision*
(30 nm) within *count window* (100 nm) of the centre: `n_localizations`.

```figure Six simulated pores at 30 % labelling: the localizations counted for the corners (blue) and the others (grey), the ring band (dotted), the segment borders, and the corners seen (filled red).  The truly labelled corners of each pore are given in the title beside the count.
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

**The centre** minimises $\sum_i \rho(d_i - R)$ over the precise
localizations, $d_i$ the distance of localization $i$ from the centre, $R$
the *radius*, and $\rho$ the soft-L1 loss with the *ring width* as its scale
(see [NPC](plugin:ROIManager/Segment/NPC)); then twice more over those within
$R + 2\Delta R$ of the answer, $\Delta R$ the *ring width*.  With fewer than 3
precise localizations, all of them are used.

**The rotation** is the weighted circular mean of the angles $\theta_i$ of
the counted localizations, taken $n$ times over so that all corners fall on
one direction:

$$\phi = \frac{1}{n} \arg \sum_i w_i e^{i n \theta_i}, \qquad w_i = \left( \frac{\rho_i}{\sigma_i} \right)^2 ,$$

$n$ the *corners*, $\rho_i$ the distance from the centre and $\sigma_i$ the
precision, the weight one over the square of the angular precision.  The
corners are at $\phi + 2\pi j / n$; a localization belongs to the corner
whose segment, of width $2\pi/n$ centred on it, holds its angle, and a
segment with at least *min per corner* is a corner seen.

**Why a cutoff, and why 20 nm.**  A localization with a precision of 15 nm
or more crosses into a neighbouring segment often enough to open an empty
corner, and with many blinks per fluorophore that happens in many pores;
with no cutoff the efficiency came out up to 32 points too high on
simulations.  A tight cutoff loses the corners whose blinks were all dim.
20 nm is between the two, and the analysis models both what spills and what
is lost, so it does not have to be exact.

**SMAP's count**, `n_corners_smap`, is that of SMAP's
`NPCLabelingQuantify`: every localization from 30 to 70 nm from the centre
with a precision better than 0.4 of the arc between two corners at 50 nm
(15.7 nm), around the same centre and with the rotation found the same way.

**Without a precision column** every localization is counted, for the
corners and for N, and the analysis can only use SMAP's method.

## Parameters

### radius_nm
The radius of the protein's ring.  Leave the band (*radius* ± *ring width*)
where the segmenter has it, so that what is counted is what was judged.

### precision_nm
The cutoff the analysis models (see *In detail*): the same as the
segmenter's *judge on precision*.

### n_precision_nm
Loose enough that most of a pore's localizations count, tight enough that
the scattered imprecise localizations of neighbouring pores do not.

### n_window_nm
About the ring's radius plus three of the count's precisions.

### min_locs
1.  The analysis assumes it.

## Output

One row per ROI in the site table:

* `n_corners`, the corners seen with the precise localizations;
* `n_localizations`, the localizations better than *count precision*;
* `n_corners_smap`, SMAP's count;
* `n_ring_locs`, the localizations the corners were counted with;
* `radius_nm`, the radius fitted free to them, and `ring_radius_nm`, their
  median distance from the centre (the analysis takes the ring's radius
  from it);
* `rotation_deg`, the rotation $\phi$, between $-180/n$ and $180/n$ degrees;
* `x_nm`, `y_nm`, the fitted centre.

The figure shows the pore with the ring band, the segment borders and the
corners, filled when seen, and is redrawn for each ROI as the ROI manager
walks the list.

## Differences from SMAP

Ported from SMAP's `ROIManager/Evaluate/NPCLabelingQuantify_s`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), whose count is
kept as `n_corners_smap`.

* **What is counted.**  SMAP counts the corners only, with every
  localization better than 15.7 nm from 30 nm out.  Here the main count uses
  a cutoff the analysis models and starts at 40 nm, and the localizations per
  pore are counted as well, which is what lets the analysis take the blinks
  into account.
* **Centre.**  SMAP fits it by plain least squares on every localization;
  here a soft-L1 loss on the precise ones.
* **Rotation.**  SMAP fits the sawtooth $\mathrm{mod}(\theta - a, \pi/4)$ by
  least squares, started from a circular mean; here the weighted circular
  mean is the answer.
* **Left out**: SMAP's split of the count by time and its counts on the
  simulation's true positions; the truth is compared in
  [NPC Labeling Efficiency](plugin:ROIManager/Analyze/NPC Labeling Efficiency)
  instead.

## References

* Thevathasan JV, Kahnwald M, Cieśliński K, et al. Nuclear pores as
  versatile reference standards for quantitative superresolution
  microscopy. *Nat Methods* 16, 1045 (2019).
  [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
