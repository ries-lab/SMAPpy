---
version: "2"
covers: [smappy.npc.localization_classes, smappy.npc.joint_pmf, smappy.npc.fit, smappy.plugins.npc.joint_efficiency, smappy.plugins.npc.labeling_efficiency, smappy.plugins.npc.fit_likelihood, smappy.plugins.npc.fit_sqrt_lsq, smappy.plugins.npc.true_corners]
---

## What it does

Not every copy of a protein in an SMLM image is seen.  Some are never
labelled; some are labelled with a dye that never switches on, or never
brightly enough.  The fraction that is labelled and seen is the *effective
labelling efficiency* (ELE).  It limits what can be counted and what can be
resolved, and it is the direct test of a labelling protocol.

Nuclear pores make it measurable (Thevathasan et al. 2019,
[doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)).
Each pore has eight corners of four copies of Nup96.  A corner shows when
one of its copies is seen, so how many corners show, across many pores,
depends on the ELE, and this plugin finds the ELE that explains it.

Counting corners naively is biased, in both directions.  A localization
imprecise enough lands in the neighbouring corner's segment and opens a
corner that is empty; the more often a fluorophore blinks, the more often.
And a filter on precision, which stops that, loses the corners whose
blinks were all dim.  On simulations both are large: up to 10 points of
efficiency at 5000 photons per blink, and far more at 500 (the study in
`studies/npc_le` has the numbers).

So the plugin fits a model of a whole pore instead.  It uses two numbers per
pore from [NPC Corners](plugin:ROIManager/Evaluate/NPC Corners): the corners
seen with the precise localizations, and the localizations the pore has.  The
model knows what spills and what is lost, and the localizations tell how
often a fluorophore blinks -- which is what it needs to work out the dim
fluorophores the precise ones miss.  It gives the ELE and the blinks per
copy.

It uses only the pores that show at least *fit from* corners (5).  Pores with
fewer are the ones a segmentation is likely to miss, and junk rarely shows
5; leaving both out of the fit, and out of the model, means their loss does
not bias the answer.  SMAP's analysis, the corner histogram alone, is kept as
the *corners (SMAP)* method.

The order is: find the pores with [NPC](plugin:ROIManager/Segment/NPC), add
NPC Corners to the evaluation pipeline and evaluate, then run this.  Use
grouped localizations, imaged until the fluorophores have bleached.

## How it works

```figure-setup
from smappy.group import group
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings, BlinkingSettings)
from smappy.plugins.npc import (NPCSegmentSettings, NPCCornersSettings, segment_npcs,
                                count_corners, joint_efficiency, draw_joint, true_corners,
                                labeling_efficiency, LabelingEfficiencySettings)
from smappy import npc as model
from scipy.spatial import cKDTree
sim = SimulationSettings(n_frames=3000, seed=2, structure=StructureSettings(preset="npc"),
                         labelling=LabellingSettings(efficiency=0.4),
                         blinking=BlinkingSettings(blinks=3, activation="all"))
raw = simulate(sim)
locs, _ = group(raw, attach_columns=False)
xy = np.column_stack((locs["x_nm"], locs["y_nm"]))
sigma = np.asarray(locs["xy_err_nm"])
tree = cKDTree(xy)
cs = NPCCornersSettings()
k, n, radii = [], [], []
for site in segment_npcs(locs, NPCSegmentSettings()):
    if site["use"]:
        near = tree.query_ball_point(site["center"], 150)
        v = count_corners(xy[near, 0], xy[near, 1], site["center"], cs, sigma[near])
        k.append(v["n_corners"]); n.append(v["n_localizations"]); radii.append(v["ring_radius_nm"])
fitted = joint_efficiency(k, n, sigma, cs, float(np.nanmedian(radii)))
labelled = labeling_efficiency(list(true_corners(raw).values()),
                               LabelingEfficiencySettings(fit_min=0, bootstrap=0))["efficiency"]
```

**1. Two histograms.**  For every pore with at least *fit from* corners: how
many corners it shows, and how many localizations it has.

**2. What a blink becomes.**  From the precisions of every localization in
the files, the plugin works out the chance that one blink gives a
localization good enough for the corners, one good enough only for the
localization count, or neither.  This is measured on the whole field, not on
the pores, so it does not depend on which pores were found.

**3. The pore.**  32 copies, 4 to a corner.  Each is labelled with the ELE,
and blinks a number of times that is, on average, the *blinks per copy*.  A
corner localization lands in a neighbouring segment with a chance worked out
from its precision and its distance from the centre.  From this the model
gives, exactly, how likely each combination of corners seen and
localizations is.

**4. The fit.**  The ELE and the blinks per copy are the values that make the
two histograms most likely, among the pores with at least *fit from*
corners.  The errors are from how sharply the likelihood falls off around
them.

```figure A simulated nuclear envelope at 40 % labelling and 3 blinks per copy, segmented, counted and fitted.  The model (red) is fitted to both histograms at once, over the pores with 5 corners or more.  The truly labelled corners of the simulation give the ELE in the title.
fig.set_size_inches(8, 3.2)
draw_joint(fig, fitted, labelled)
```

## In detail

**What a blink becomes.**  With $\sigma$ running over the precisions of
every localization of the files, $r$ the ring's radius (the median distance
of the counted localizations from the centres), the band $[r_1, r_2]$ (40 to
70 nm), the cutoffs $c$ (*corner precision*) and $c_N$ (*count
precision*), and the window $W$, the chances are averages over the field:

$$a = \left\langle [\sigma < c] \, P(r_1 \leq \rho \leq r_2) \right\rangle, \qquad b = \left\langle [\sigma < c_N] \, P(\rho \leq W) \right\rangle - a ,$$

$\rho \sim N(r, \sigma)$ the localization's distance from the centre.  A
corner localization at distance $\rho$ crosses into a given neighbour's
segment when its angular error exceeds the margin $\pi/8 \mp \delta$, with
$\delta = 5.6\,\mathrm{nm}/r$ the angle of the corner's two copies either side
of its centre:

$$\varepsilon = \left\langle \frac{1}{2} \left[ \Phi\left( -\frac{(\pi/8 - \delta)\rho}{w} \right) + \Phi\left( -\frac{(\pi/8 + \delta)\rho}{w} \right) \right] \right\rangle , \qquad w = \sqrt{\sigma^2 + (4\,\mathrm{nm})^2},$$

averaged over $\rho$ in the band and over the corner localizations, the
4 nm for the tilt of the pore and the rotation fit.

**One copy** is labelled with probability $L$ (the ELE) and blinks
$B \geq 1$ times, $B$ geometric with mean $1/p$ (a fluorophore imaged until
it bleaches), so its generating function is

$$G(u) = 1 - L + L \frac{p u}{1 - (1 - p) u} .$$

**One corner.**  Each blink is a corner localization ($a$), a counted one
only ($b$), or neither; a corner localization stays, or spills left or
right, with $(1 - 2\varepsilon, \varepsilon, \varepsilon)$.  With $z$
counting the localizations, $G(1 - a - b + bz + az\,w_S)^4$ is the
generating function of the corner's localizations with the channels in $S$
empty ($w_S$ the share left), and inclusion and exclusion over $S$ give the
weight of each of the 8 combinations of (stays, spills left, spills right).

**The ring.**  A corner is seen when one of its localizations stays, or a
neighbour spills one into it.  Going round the ring, a corner's state is
whether it is seen so far and whether it spills right; the next corner
decides the first.  That is a $4 \times 4$ transfer matrix $M(x, z)$, $x$
counting the corners seen, and the trace of $M^8$ is the generating function
of the pair (corners seen $k$, localizations $N$).  Evaluated at the roots of
unity and transformed back by a 2D FFT, it gives $P(k, N \mid L, p)$
exactly.

**The likelihood** over the pores $i$ with $k_i \geq k_{\mathrm{min}}$
(*fit from*) is

$$\log \mathcal{L}(L, p) = \sum_i \log \frac{P(k_i, N_i \mid L, p)}{P(k \geq k_{\mathrm{min}} \mid L, p)} ,$$

maximised by Nelder-Mead from three starts; the errors are the square roots
of the inverse of its curvature, by finite differences.  Conditioning on
$k \geq k_{\mathrm{min}}$ is what makes a segmentation that loses sparse
pores harmless: those pores are out of the sample and out of the model
alike.

**What it was tested on.**  Simulated Nup96 pores tilted by up to 10°,
imaged to full bleaching, with 500 or 5000 photons per blink, 1 to 10 blinks
per copy and ELEs of 0.35 to 0.7, segmented with the same checks among
blobs, filled clumps and filaments, 1000 pores at a time.  Fitted from 5
corners, the ELE came out within about 3 points at 5000 photons and within
about 3 at 500 with three blinks.  Most of what is left is junk that reaches
5 corners, largest at high ELE and many blinks, where the histogram is least
sensitive: −11 points at 500 photons, 10 blinks and ELE 0.7, −6 fitted from 6
corners.  Without the junk the fit is within 1.5 points at 5000 photons.

**What it assumes.**  Every pore has 32 copies, and all its localizations are
its own; the same ELE and blinking in every pore; a geometric number of
blinks; a blink's brightness independent of its fluorophore; and one grouped
localization per blink.  A grouping that leaves a blink in several rows --
smappy's links within a fixed 50 nm, which a dim frame at 500 photons misses
-- reads as more blinks.  On a layer that is not grouped at all the plugin
still runs, and its text says so.

**Simulated data.**  When the sites come from a simulated table, the plugin
also reports the efficiency it was simulated with, and fits the labelled
corners of the same pores (drawn again from the recipe, as
[Ground Truth](plugin:Analysis/Measure/Ground Truth) does; a site's pore is
the `copy` most of its localizations belong to).

**SMAP's method** fits the histogram of SMAP's corner count
(`n_corners_smap`) alone, with the binomial model, $p_c = 1 - (1 - L)^4$ the
chance a corner shows,

$$P(k) = \frac{8!}{k!\,(8-k)!} \, p_c^k (1 - p_c)^{8-k} ,$$

by the likelihood renormalised to *fit from* .. *fit to*, or by SMAP's least
squares on $\sqrt{A\,P(k)}$ against the square root of the histogram, with a
bootstrap error.  It measures the corners as counted, strays and losses
included.

## Parameters

### method
The joint model by default.  SMAP's method needs no precision and gives
numbers to compare with SMAP.

### fit_min
5.  6 removes more junk and costs where few pores show 6 corners (one blink
per fluorophore, low ELE); 4 keeps the sparse pores a segmentation is most
likely to miss or miscentre.

### corners
8 for the nuclear pore.

### per_corner
4 for Nup96 and the other proteins with 32 copies per pore; 2 for one with 16.

### fit_max
SMAP's method only; usually 8.

### fit
SMAP's method only.  The likelihood is right for small counts; SMAP's least
squares is kept to compare.

### bootstrap
SMAP's method only.  100 gives the error to two digits.

## Output

* **The text**: the ELE and the blinks per copy with their errors, and the
  pores fitted of all.  On a simulation also the simulated efficiency and the
  ELE of the same pores' labelled corners.  A note when the layer is not
  grouped.
* **The figure**: the corners seen and the localizations per pore of the
  pores fitted (grey), and the model at the fitted values (red).  A good fit
  follows both.  More pores at either end of the localization histogram than
  the model allows mean pores that are not alike -- uneven labelling, or
  junk among them.
* **Data**: `efficiency`, `error`, `blinks`, `blinks_error`, `pores`, the
  chances `a`, `b`, `eps`, the corner `counts`; on a simulation
  `simulated_efficiency` and `true_efficiency`.

With SMAP's method, the corner histogram and the binomial at the fitted
efficiency.

## Differences from SMAP

Ported from SMAP's `ROIManager/Analyze/NPCLabelingEfficiency` and
`shared/fitNPClabeling` ([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)),
which are kept as the *corners (SMAP)* method.

* **The model.**  SMAP fits the histogram of corners counted on every
  localization with the binomial model.  The joint model also counts the
  localizations, models the spills between corners and the corners lost to
  the cutoff, and fits the blinks with the ELE.
* **The fit range.**  SMAP fits from 3 corners by least squares on the square
  root of the histogram; here the likelihood, conditioned on 5 corners or
  more, for the model and the data alike.
* **Left out**: SMAP's PSF-width and file-number selections (the layer's
  filters and the ROI manager's *use* do it), the Gaussian fit to the
  localizations per pore with its "LE = mean / 32", and the copy to a
  separate page.

## References

* Thevathasan JV, Kahnwald M, Cieśliński K, et al. Nuclear pores as
  versatile reference standards for quantitative superresolution
  microscopy. *Nat Methods* 16, 1045 (2019).
  [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)
  -- the method and the corner model.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
