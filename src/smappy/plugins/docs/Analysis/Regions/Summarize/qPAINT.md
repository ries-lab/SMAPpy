---
version: "1"
covers: [smappy.plugins.binding_kinetics.Qpaint.run, smappy.plugins.binding_kinetics.pooled_kinetics, smappy.plugins.binding_kinetics.draw_qpaint, smappy.binding_kinetics.pooled_index, smappy.binding_kinetics.sites_from]
---

## What it does

[qPAINT](plugin:Panels/Glossary) counts the binding sites of a region from
how often an imager binds to it
([Jungmann et al. 2016](https://doi.org/10.1038/nmeth.3804)): a region with
four docking strands is bound four times as often as one with one.  The
rate of one strand, the **influx**, depends on the imager's concentration
and the buffer, so it is calibrated on regions whose number of sites is
known -- single docking strands, or DNA origami with a designed number.

This reads what
[Binding kinetics](plugin:Analysis/Measure/Binding kinetics) measured in
every region, calibrates the influx on the regions marked as calibration,
and gives every region its number of sites.  It also draws the bright and
dark times of all regions' events together, which says whether the
kinetics are what the imager should give.

Mark the calibration regions with an ROI keyword (*Annotate ROIs*: a list
such as "kind" with "calibration" and "sample"), or calibrate on every
region when they all have the same known number.  The influx it finds is
then typed into *Binding kinetics*' *influx*, which writes `n_sites` into
each region's row -- for *Filter ROIs*, *Histograms* or a column.

## How it works

```figure-setup
from smappy.simulate import simulate_binding_sites
from smappy import binding_kinetics as bk
from smappy.plugins.binding_kinetics import draw_qpaint, pooled_kinetics
N = 20000
sites = [1] * 30 + [2] * 15 + [3] * 15
sim = simulate_binding_sites(sites, N, tau_bright=5.0, tau_dark=600.0, seed=6)
frame, spot = np.asarray(sim["frame"]), np.asarray(sim["spot_true"])
found = [bk.region_kinetics(frame[spot == k + 1], spans={0: (0, N - 1)})
         for k in range(len(sites))]
values = [f[0] for f in found]
calibration = np.array(sites) == 1
```

**1. The influx.**  Over the calibration regions, their dark frames and
their dark periods are summed and the index of all of them taken together
is divided by the *known sites*: one strand's binding rate.  A region with
many events weighs more than one with few, as in Picasso's pooled fit.

**2. Sites.**  Every region's qPAINT index over the influx.

```figure Sixty simulated regions with one, two or three docking strands, each strand free for 600 frames on average over 20000, calibrated on the thirty single strands.  Top: the bright and dark times of every event of every region.  Bottom: the qPAINT index and the sites it gives, the calibration regions in blue; the three numbers stand apart.
pooled = pooled_kinetics([v["dark_total_frames"] for v in values[:30]],
                         [v["dark_periods"] for v in values[:30]],
                         np.concatenate([a["bright_frames"] for _, a, _ in found]))
index = np.array([v["qpaint_index"] for v in values])
fig.set_size_inches(7.0, 5.6)
fig.set_layout_engine("constrained")
draw_qpaint(fig, np.concatenate([a["bright_frames"] for _, a, _ in found]),
            np.concatenate([a["dark_frames"] for _, a, _ in found]),
            index, index / pooled["influx"], calibration, pooled, pooled["influx"])
```

## In detail

For calibration regions $k$ with dark frames $D_k$ and dark periods $n_k$
(the numerator and denominator of their mean dark time, `dark_total_frames`
and `dark_periods`; see *Binding kinetics*), the influx is

$$\phi = \frac{\sum_k n_k - 1}{N \sum_k D_k},$$

with $N$ the *known sites*, and its standard error is
$\phi / \sqrt{\sum_k n_k}$.  The sites of region $j$ are
$q_j / \phi$, $q_j$ its qPAINT index.  Picasso instead fits the cumulative
distribution of the calibration picks' dark times taken together; summing
the likelihood's terms is the same pooling without the fit's bias on the
ends of the acquisition.

The bright and dark times drawn are every event's, from the arrays each
region kept; the dashed lines are the mean bright time and the calibration
regions' mean dark time.  Regions the region filter drops are left out.

## Parameters

### known_sites
A whole number for designed structures; for single docking strands 1.

### keyword_list
The calibration regions are those with *keyword* in this list.  The
keywords are ROIs' only: regions of an id column (clusters) are calibrated
on all of them, or with a typed *influx*.

### influx
Set it to count sites with an influx from another sample of the same
imager and buffer; nothing is calibrated then.

## Output

The text gives the influx with its error and the regions it came from, and
the mean, median and standard deviation of the sites.  `data` holds the
influx, each region's sites by its id and the calibration regions' ids.

A good calibration gives sites that cluster at whole numbers.  A broad,
smeared distribution means too few events per region (run longer, or use
larger regions with more strands), regions holding more than one structure,
or sticky sites that *Binding kinetics*' curation values should have
removed first.

## References

* R. Jungmann, M. S. Avendaño, M. Dai, J. B. Woehrstein, S. S. Agasti,
  Z. Feiger, A. Rodal, P. Yin, "Quantitative super-resolution imaging with
  qPAINT", *Nature Methods* 13, 439-442 (2016),
  [doi:10.1038/nmeth.3804](https://doi.org/10.1038/nmeth.3804).
* J. Schnitzbauer, M. T. Strauss, T. Schlichthaerle, F. Schueder,
  R. Jungmann, "Super-resolution microscopy with DNA-PAINT", *Nature
  Protocols* 12, 1198-1228 (2017),
  [doi:10.1038/nprot.2017.024](https://doi.org/10.1038/nprot.2017.024) --
  Picasso's influx calibration.
