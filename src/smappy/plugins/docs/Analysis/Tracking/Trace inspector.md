---
version: "1"
covers: [smappy.plugins.trace_inspector.TraceInspector.run, smappy.plugins.trace_inspector.trace_values, smappy.plugins.trace_inspector.running_median, smappy.plugins.trace_inspector.draw_trace]
---

## What it does

Looks at one [trace](plugin:Panels/Glossary) closely -- a MINFLUX `tid`, a
linked track, a bead -- to see whether it is what it should be: its
positions against time with a running median, the spectrum of those
positions, its quality attributes (`efo`, `cfr`, photons, `fbg`) against
time, and the numbers that tell one still molecule from a molecule that
drifts, moves or was joined by a second one.  It is SMAP's
Cluster_MINFLUX_Roi.

It runs **per region**: in the ROI manager or a block over the
[id attribute](plugin:Panels/Glossary), each trace is inspected and its
numbers go into the [region table](plugin:Panels/Glossary), for *Filter
regions* and the summaries.  On the view it takes the longest trace the
selection holds -- draw an ROI around one to choose it, or give its id in
*trace* -- and with *gallery* the longest few side by side, each turned onto
its own long axis, as the MFX scripts' gallery of traces.

## How it works

```figure-setup
from smappy.plugins.trace_inspector import trace_values, draw_trace

rng = np.random.default_rng(0)
t = np.arange(2000) * 1e-3
x = rng.normal(0, 3, 2000) + 8 * np.sin(2 * np.pi * t / 1.5)    # 3 nm noise, a slow wander
y = rng.normal(0, 3, 2000)
cfr = np.clip(rng.normal(0.3, 0.05, 2000) + (t > 1.5) * 0.3, 0, None)   # a second molecule at 1.5 s
values = trace_values(t, {"x": x, "y": y}, {"cfr": cfr})
```

**1. The trace.**  The localizations of the trace in time order.

**2. Its numbers.**  Per axis three spreads: the standard deviation; a
robust one, 1.4826 times the median absolute deviation, which a few
outliers do not inflate; and the precision from consecutive localizations,
std(differences) / √2, which drift and slow wander do not reach.  A still
molecule gives three equal numbers close to its `xy_err_nm`; drift or
motion leave the consecutive one alone and raise the others.  Also the
number of localizations, the duration, the median time between them and
the median of each quality attribute.

**3. Its figures.**  The positions against time, each axis about its mean,
with the running median over *running median* localizations; the
amplitude spectrum of the positions, put on a regular grid of the median
time step first (MINFLUX's clock is irregular); the quality attributes
against time, each relative to its median.

```figure A simulated trace: 3 nm noise and a slow 8 nm wander in x, and a second molecule switching on at 1.5 s that raises the cfr. The wander shows in the running median and as the low frequency in the spectrum; the cfr's jump marks the second molecule.
fig.set_size_inches(10.0, 2.8)
draw_trace(fig, t, {"x": x, "y": y}, {"cfr": cfr}, 50, "s")
```

The wander raises x's standard deviation to about 6 nm, while the
precision from consecutive localizations stays at the 3 nm of the noise.

## Parameters

### trace
The id of the trace to inspect on the view, as the region table or a
colour by the id attribute shows it; -1 the longest in the selection.

### window
20 localizations follow a motor's steps; more smooth a still molecule's
noise into its drift.

### gallery
6 to 12 fit side by side; each panel is one trace, its long axis
horizontal.

## Output

The text and, per region, the values: `n_locs`, `duration`, `dt_median`,
per axis `x_sd_nm`, `x_robust_nm`, `x_consecutive_nm`, and the medians
`efo_median`, `cfr_median`, `photons_median`, `fbg_median` where the table
has them.  Three panels, or the gallery.

## Differences from SMAP

Ported from SMAP's Cluster_MINFLUX_Roi (Ries 2020).  SMAP's spread
measures, the medians of the quality attributes and the plots against time
are kept; its precision against binned photons is *Stability*'s random
scatter here, its time-step histogram is the median time step, and its
option to inspect the localizations of the steps StepsMINFLUX found is
*Find steps*' plateaus as a region.

## References

* Ries J (2020).  SMAP: a modular super-resolution microscopy analysis
  platform for SMLM data.  *Nat Methods* 17, 870-872.
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
