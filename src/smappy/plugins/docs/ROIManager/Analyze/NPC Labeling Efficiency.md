---
version: "1"
covers: [smappy.plugins.npc.labeling_efficiency, smappy.plugins.npc.fit_likelihood, smappy.plugins.npc.fit_sqrt_lsq, smappy.plugins.npc.corner_model, smappy.plugins.npc.true_corners]
---

## What it does

Not every copy of a protein in an SMLM image is seen.  Some are never
labelled.  Some are labelled with a dye that never switches on, or not
brightly enough to be fitted.  The fraction that is labelled *and* seen is the
*effective labelling efficiency* (ELE).  It limits what can be counted and
what structures can be resolved.  It is also a direct test of a labelling
protocol.

Nuclear pores make it measurable (Thevathasan et al. 2019,
[doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)).
Each pore has eight corners, each holding four copies of Nup96.  A corner
shows when at least one of its four copies was seen.  How many corners show
across many pores therefore depends only on the ELE, and this plugin finds
the ELE that explains it.

It reads the corner counts that
[NPC Corners](plugin:ROIManager/Evaluate/NPC Corners) wrote into the site
table.  So the order is: find the pores with
[NPC](plugin:ROIManager/Segment/NPC), add NPC Corners to the evaluation
pipeline and evaluate, then run this.  Only the ROIs that are used count.
A hundred pores give the ELE to about ±0.01 to 0.02.

## How it works

```figure-setup
from smappy.plugins.npc import (LabelingEfficiencySettings, corner_model,
                                labeling_efficiency, draw_histogram, true_corners,
                                NPCSegmentSettings, segment_npcs, NPCCornersSettings,
                                count_corners)
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from scipy.spatial import cKDTree
sim = SimulationSettings(n_frames=3000, seed=2,
                         structure=StructureSettings(preset="npc"),
                         labelling=LabellingSettings(efficiency=0.4))
locs = simulate(sim)
xy = np.column_stack((locs["x_nm"], locs["y_nm"]))
precision = np.asarray(locs["xy_err_nm"])
tree = cKDTree(xy)
counted = []
for site in segment_npcs(locs, NPCSegmentSettings()):
    if site["use"]:
        near = tree.query_ball_point(site["center"], 150)
        counted.append(count_corners(xy[near, 0], xy[near, 1], site["center"],
                                     NPCCornersSettings(), precision[near])["n_corners"])
settings = LabelingEfficiencySettings()
fitted = labeling_efficiency(counted, settings)
truth = labeling_efficiency(list(true_corners(locs).values()),
                            LabelingEfficiencySettings(bootstrap=0))
```

**1. The histogram.**  The corner counts of all used ROIs are collected into a
histogram: how many pores show 0, 1, ... 8 corners.

**2. The model.**  With an ELE $p$, a corner of four copies shows with the
chance $p_c = 1 - (1 - p)^4$.  The eight corners show independently of each
other, so the number that show follows a binomial distribution.

**3. The fit.**  The ELE is the value whose binomial distribution best
explains the histogram.  Only the pores with *fit from* to *fit to* corners
(3 to 8) take part.  Pores with fewer corners are the ones a segmentation is
likely to miss or reject, so they are left out, and the model is rescaled to
the same range.

**4. The error.**  The pores are resampled with replacement and the fit is
repeated (*bootstrap*, 100 times).  The spread of the answers is the error.

```figure The corner model.  Left: how many corners show, for ELEs from 0.2 to 0.8.  Above about 0.6, nearly every corner shows and the histograms all look alike; this is where the method loses its power.  Right: a simulated envelope at 40 % labelling, segmented, counted and fitted (red), beside the corners that were truly labelled in the same simulation (blue).
fig.set_size_inches(8, 3)
left, right = fig.subplots(1, 2, gridspec_kw={"wspace": 0.35})
k = np.arange(9)
for p, colour in zip((0.2, 0.3, 0.4, 0.6, 0.8), ("#fde725", "#5ec962", "#21918c", "#3b528b", "#440154")):
    left.plot(k, corner_model(p, 8, 4), "o-", color=colour, ms=3, label=f"ELE {p:.1f}")
left.set_xlabel("corners seen"); left.set_ylabel("fraction of pores")
left.legend(fontsize=7, frameon=False)
draw_histogram(right, fitted, settings, truth)
```

## In detail

**The model.**  With $n$ the *corners*, $m$ the *proteins per corner*, and $p$
the ELE:

$$p_c = 1 - (1 - p)^m, \qquad P(k) = \frac{n!}{k!\,(n-k)!} \, p_c^k (1 - p_c)^{n-k} .$$

It assumes that every copy is seen independently and with the same chance,
and that every corner of every pore holds all $m$ copies.

**The likelihood.**  With $h_k$ the number of pores showing $k$ corners, and
the fit range $a \leq k \leq b$, the ELE maximises

$$\log L(p) = \sum_{k=a}^{b} h_k \log \frac{P(k)}{\sum_{j=a}^{b} P(j)} ,$$

the likelihood of the binomial distribution renormalised to the range
(bounded search over $0 < p < 1$).  Because of the renormalisation, a pore
missing below $a$ does not bias the answer.  It only costs precision.  The
pores that do take part must be a fair sample within the range: a
segmentation that preferred pores with eight corners over those with three
would still bias the answer.

**SMAP's least squares** is offered as the *fit* choice
*SMAP's least squares*.  It minimises

$$\sum_{k=a}^{b} \left( \sqrt{A\,P(k)} - \sqrt{h_k} \right)^2$$

over an amplitude $A$ and $p$, started from the number of pores and 0.4.  The
square root roughly evens out the Poisson noise of the counts.  On a hundred
or more pores the two agree to well within the error; the likelihood needs no
amplitude and stays right for small counts.

**The bootstrap** draws as many pores as there are, with replacement, from a
generator with a fixed seed, so a rerun gives the same error.

**Simulated data.**  When the sites come from a simulated table, the plugin
also reports the efficiency it was simulated with.  It draws the labelled
molecules again from the recipe the table carries (as
[Ground Truth](plugin:Analysis/Measure/Ground Truth) does), counts each pore's
labelled corners, and fits those counts for the same pores the same way.  A
pore under a site is the `copy` most of the site's localizations belong to.
The difference between the two numbers is what segmenting and counting cost.
In simulations it is below 0.02 at an ELE of 0.3 to 0.7.  Below 0.2, stray
localizations between corners bias it upwards (see
[NPC Corners](plugin:ROIManager/Evaluate/NPC Corners)).

**Limits.**  The ELE is well measured between about 0.2 and 0.7.  Above that,
almost every corner shows whatever the exact ELE, the histogram is nearly all
eights, and the error grows; at 0.9 a simulation fits anywhere from 0.8 to 1.

## Parameters

### per_corner
4 for Nup96 and the other proteins with 32 copies per pore.  For a protein
with 16 copies (8 corners of 2), set 2.

### fit_min
3, as in SMAP.  Raise it if the segmentation rejects many sparse pores, and
lower it on a clean sample to use more of them.  The likelihood is unbiased
for any choice, as long as the pores in the range are a fair sample.

### fit_max
Usually equal to *corners*.

### fit
The likelihood is the default.  SMAP's least squares is kept to compare
results with SMAP.

### bootstrap
100 is enough for the error to two digits.  0 skips it, and the error is then
reported as nan.

## Output

* **The text**: the ELE with its bootstrap error, the number of pores and how
  many of them were in the fit range.  On a simulation, also the simulated
  efficiency and the ELE of the same pores' labelled corners.
* **The figure**: the histogram of corners seen (grey), the model at the
  fitted ELE (red, with dots on the fit range), and on a simulation the
  labelled corners of the same pores (blue).
* **Data**: `efficiency`, `error`, `n_sites`, `counts`, `corner_probability`
  ($p_c$), and on a simulation `simulated_efficiency`, `true_efficiency` and
  `true_counts`.

A good fit follows the histogram over the whole fit range.  A histogram with
more pores at both ends than the model allows means the pores are not alike.
That happens with uneven labelling across the cell, or with sites that are
not pores.

## Differences from SMAP

Ported from SMAP's `ROIManager/Analyze/NPCLabelingEfficiency` and
`shared/fitNPClabeling` ([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* **The fit.**  SMAP fits by least squares on the square root of the
  histogram, with a free amplitude.  Here the default is the likelihood
  renormalised to the fit range, which needs no amplitude and is correct for
  small counts.  SMAP's fit is kept as a choice.
* **The error.**  SMAP bootstraps 20 times; here 100, with a fixed seed.
* **Left out.**  SMAP's PSF-width and file-number selections (use the layer's
  filters and the ROI manager's *use*), the Gaussian fit to the localizations
  per pore with its "LE = mean / 32", and the copy to a separate page.
* **Ground truth.**  SMAP reads true positions from columns of the simulated
  table (`xnm_gt`); here the truth is drawn again from the recipe, so the
  table needs no extra columns.

## References

* Thevathasan JV, Kahnwald M, Cieśliński K, et al. Nuclear pores as
  versatile reference standards for quantitative superresolution
  microscopy. *Nat Methods* 16, 1045 (2019).
  [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)
  -- the method and the corner model.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
