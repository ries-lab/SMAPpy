---
version: "2"
covers: [smappy.plugins.quality_filter.trace_cfr, smappy.quality.robust_upper, smappy.quality.efo_valley, smappy.plugins.quality_filter.QualityFilter.run]
---

## What it does

Sets the two cuts a [MINFLUX](plugin:Panels/Glossary) measurement is
usually filtered with, at values found in the measurement itself rather
than fixed numbers: an upper bound on the **cfr** (the centre-frequency
ratio, the photons at the pattern's centre over those at its offsets: high
when two molecules are on at once, or the background is high) and on the
**efo** (a localization's photon rate: twice as high when two molecules
emit together).  The cuts depend on the sample, the dye and the buffer; a
number that suits one measurement throws away good localizations, or keeps
bad ones, in the next.

The cuts become ordinary filter bounds on the layers, shown and editable
in the filter panel like any other, and recorded when the plugin is a step
of a chain.

## How it works

```figure-setup
from smappy import quality as Q

rng = np.random.default_rng(0)
# one molecule at a time: cfr about 0.3, efo about 100 kHz; 10 % with two on:
# cfr higher and spread out; 20 % of the localizations at twice the rate
cfr = np.r_[rng.normal(0.3, 0.08, 9000), rng.uniform(0.6, 1.2, 1000)]
efo = np.r_[rng.normal(100e3, 15e3, 8000), rng.normal(200e3, 25e3, 2000)]
upper = Q.robust_upper(cfr, 2.0)
valley = Q.efo_valley(efo)
```

**1. What it measures on.**  The localizations the layer shows, without the
layer's own bounds on cfr and efo -- so running it again finds the same cut
rather than cutting into what it kept the first time.

**2. The cfr bound.**  The median of the cfr plus *k* robust standard
deviations.  A robust standard deviation is 1.4826 times the median
absolute deviation: the spread of the bulk, which the few molecules with a
high cfr do not widen.

**3. The efo bound.**  The histogram of the efo, smoothed, has a peak where
one molecule emits.  Where two molecules emitted together there is a
second peak near twice that rate, and the bound is the lowest point between
the first peak and twice it.  Where there is no second peak, the bound is
where the histogram has fallen to 5 % of its peak.

```figure A simulated measurement, one molecule at a time with cfr about 0.3 and efo about 100 kHz, a tenth of it with a second molecule on (cfr spread to 1.2) and a fifth at twice the rate: the cfr bound (left) lies past the single molecules, the efo bound (right) in the valley before the two-molecule peak.
fig.set_size_inches(7.0, 2.8)
a, b = fig.subplots(1, 2)
a.hist(cfr, bins=80, range=(0, 1.2), color="0.35")
a.axvline(upper, color="tab:red")
a.set_xlabel("cfr")
a.set_ylabel("localizations")
a.set_title(f"cfr ≤ {upper:.2f}", fontsize=8)
b.plot(valley.centres / 1e3, valley.counts, color="0.2")
b.axvline(valley.bound / 1e3, color="tab:red")
b.axvline(valley.peak / 1e3, color="0.6", ls=":")
b.set_xlabel("efo (kHz)")
b.set_title(f"efo ≤ {valley.bound / 1e3:.0f} kHz", fontsize=8)
```

**4. cfr measured once a trace.**  A 2D sequence measures the cfr once a
trace, in an iteration before the localizations it vouches for, so most
localizations have none -- and a filter leaves out a localization without
a value.  Where fewer than half the localizations have a cfr, the cut is
the trace's: each trace's cfr is the mean of the values measured in it,
missing ones left out (what merging the trace gives too), written onto all
its localizations as `trace_cfr`.  The threshold is found over the traces,
one value each, so a long trace does not count more, and the bound is set
on `trace_cfr`.  A trace is then kept or dropped whole, the same on the
localizations and on the merged traces; one whose cfr was never measured
has no `trace_cfr` and is dropped.

## In detail

**The efo histogram** has 120 bins from 0 to the 99.5th percentile of the
positive values, smoothed by a Gaussian of 2 bins.  The valley is the
minimum between the highest bin and twice its rate; it counts as a valley
when the smoothed histogram rises again by a fifth after it.  At least 20
values are needed.

## Parameters

### k
2 keeps 95 % of a normal spread of single molecules; 3 is more lenient
where the cfr's own scatter is wide (few photons), 1.5 stricter.

## Output

The bounds set, on `cfr` or on `trace_cfr` (also in the text, with the share of localizations each
keeps) and two panels: the cfr histogram and the smoothed efo histogram,
each with its bound.

## References

* pyMINFLUX (ETH D-BSSE), the automatic cfr and efo thresholds:
  [github.com/bsse-scf/pyMINFLUX](https://github.com/bsse-scf/pyMINFLUX).
