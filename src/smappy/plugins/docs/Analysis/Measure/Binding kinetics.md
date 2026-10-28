---
version: "1"
covers: [smappy.binding_kinetics.binding_events, smappy.binding_kinetics.dark_times, smappy.binding_kinetics.dark_ml, smappy.binding_kinetics.fit_cumulative, smappy.binding_kinetics.pooled_index, smappy.binding_kinetics.trace, smappy.binding_kinetics.frame_spread, smappy.binding_kinetics.uniform_p, smappy.binding_kinetics.region_kinetics, smappy.plugins.binding_kinetics.acquisition, smappy.plugins.binding_kinetics.draw_trace, smappy.plugins.binding_kinetics.BindingKinetics.run, smappy.simulate.binding.simulate_binding_sites]
---

## What it does

In [DNA-PAINT](plugin:Panels/Glossary) an imager strand in solution binds to
a docking strand on the target and comes off again, so every site blinks
again and again for the whole acquisition.  How often a site is bound says
how many docking strands it carries, and *when* it was bound says whether it
is a good site at all.  This plugin reads both from one site's
**time trace**: the frames its localizations fall in.

It links the localizations of the site into
[binding events](plugin:Panels/Glossary) and measures

* the [bright time](plugin:Panels/Glossary) of each event and the
  [dark time](plugin:Panels/Glossary) before it, and their means;
* the **qPAINT index**, the rate of binding events, and with an *influx*
  (the rate of one site) the number of binding sites -- the method of qPAINT
  ([Jungmann et al. 2016](https://doi.org/10.1038/nmeth.3804)), as Picasso
  ([Schnitzbauer et al. 2017](https://doi.org/10.1038/nprot.2017.024))
  computes it for its picks;
* how evenly the events spread over the acquisition: the first and last
  frame, the longest gap, the longest event, the mean frame and its spread.
  A sticky imager (one event hundreds of frames long) or a site that was
  lost or appeared half way shows here, and *Filter ROIs* can then drop it.

It runs **once per region** -- an [ROI](plugin:Panels/Glossary) of the ROI
manager, or a cluster of an id column, as a per-region step or a chain's
block -- or **on the view**, where the selection (a drawn ROI round one
site) is taken as one site.  A region is one structure: its localizations
are linked in time alone, whatever their positions, so draw the ROIs round
single sites.  The figure is the trace, which the ROI manager shows while
Up and Down step through the ROIs; Space un-ticks a bad one.

It is not *Time Course*, which follows the whole field (medians over a
hundred windows) to check an acquisition, nor *Blinking*, which describes
a dye's blinks.  The influx is calibrated, and the regions' kinetics pooled,
by [qPAINT](plugin:Analysis/Regions/Summarize/qPAINT).

## How it works

```figure-setup
from smappy.simulate import simulate_binding_sites
from smappy import binding_kinetics as bk
from smappy.plugins.binding_kinetics import draw_trace
N = 20000
sim = simulate_binding_sites([1, 4, 2], N, tau_bright=5.0, tau_dark=1500.0,
                             sticky=[(2, 9000, 9700)], seed=4)
frame, spot = np.asarray(sim["frame"]), np.asarray(sim["spot_true"])
```

**1. Binding events.**  The localizations of the region (or its merged
blinks) are joined wherever fewer than *link gap* + 1 frames separate them:
one binding of an imager is one event, even if the fit missed a frame of
it.  An event's bright time runs from its first frame to its last, both
counted.

**2. Dark times.**  The dark time before an event is the number of frames
from the end of the event before it to its start.  Gaps of *link gap* frames
or fewer were joined in step 1, so every dark time is longer.

**3. Mean times and the qPAINT index.**  The mean dark time is a maximum
likelihood estimate that also counts the dark frames before the first
event and after the last (*In detail*); the qPAINT index is the rate of
binding, about one over the mean dark time.  A site with four docking
strands is bound four times as often as one with one.

**4. Sites.**  With an *influx* -- the qPAINT index of one docking strand,
calibrated on regions whose number is known -- the sites of a region are its
qPAINT index over the influx.

```figure Traces of three simulated sites, each binding for 5 frames on average and each docking strand free for 1500 frames: one strand (top), four (middle), and two with a sticky imager held for 700 frames (bottom).  Each stem is an event at its start, as high as its localizations; below, the events so far against the diagonal of a constant rate.  The four-strand site climbs four times as steeply; the sticky one is a single tall stem and nothing in its cumulative count gives it away.
fig.set_size_inches(7.0, 7.6)
fig.set_layout_engine("constrained")
subs = fig.subfigures(3, 1)
for k, sub in enumerate(subs):
    f = frame[spot == k + 1]
    values, arrays, ev = bk.region_kinetics(f, spans={0: (0, N - 1)})
    draw_trace(sub, ev.first, ev.last, ev.n, (0, N - 1),
               title=f"{values['n_events']} events, qPAINT index "
                     f"{values['qpaint_index']:.2g} /frame, longest event "
                     f"{values['longest_bright_frames']:.0f} frames")
```

**5. Curation.**  A good site binds at a constant rate from the first frame
to the last, its events short.  The values below each say how far a site is
from that, so a condition such as
`longest_bright_frames < 50 & peak_window_fraction < 0.8` keeps the good
ones.

## In detail

**Linking.**  The region's blinks (the layer's merged blinks, else runs of
consecutive frames) are sorted by first frame and a new event starts where
the dark time before a blink is longer than the link gap $T$; per file, so
two files' frames are never joined.  This is Picasso's
`link(..., r_max=999999, max_dark_time=3)` for a pick.  The dark time before
event $i$ is

$$d_i = s_i - \max_{j < i} e_j - 1 ,$$

with $s_i$ its first frame and $e_j$ the last frames of the events before:
Picasso's convention since its version 0.11.3, and *Blinking*'s.

**The mean dark time.**  For dark times exponential with mean $\tau$, a
frame counted bright from half covered, an observed $d_i$ is longer than
$T$ and only its excess over $T + 1/2$ is exponential.  The dark frames
before the first event (from the first frame of the acquisition) are a
complete exponential dark time; those after the last event an incomplete
one.  The likelihood is then maximised by

$$\hat{\tau} = \frac{D_0 + \sum_{i} \left( d_i - T - \frac{1}{2} \right) + D_1}{n},$$

with $D_0$ and $D_1$ the dark frames before the first event and after the
last, and $n$ the dark periods that ended -- the events, one per file more
than the dark times between them.  Leaving $D_0$ and $D_1$ out, as a mean of
the dark times between events does, makes $\hat{\tau}$ short by about
$n/(n+1)$: by 9 % at ten events.  Where the acquisition's frames are not
known, $D_0$ and $D_1$ are left out and $n$ is the number of dark times
between events.

**The qPAINT index** is the unbiased rate

$$q = \frac{n - 1}{D_0 + \sum_i \left( d_i - T - \frac{1}{2} \right) + D_1},$$

not $1/\hat{\tau}$, which overestimates a rate by $n/(n-1)$ -- 11 % at ten
events.  Its standard error is $q/\sqrt{n}$, so a site needs about 25 events
for its count to be good to 20 %.

**The acquisition.**  Its first and last frame are, in order: *acquisition*
when set (frames 0 to *acquisition* - 1); the file's own number of frames,
where it records one (a simulation does); the first and last localization of
the file in the whole table.

**The cumulative fit** (*estimator*) is Picasso's: a least-squares fit of
$a \left( 1 - e^{-t/\tau} \right) + c$ to the cumulative count of the sorted
times, $\tau$ held between the shortest and the longest; the mean for two
values or fewer.  The free offset $c$ absorbs the link gap, but the ends of
the acquisition are not counted, and on simulated single sites with ten
events each the mean dark time came out about 20 % too short.  It is offered to compare with Picasso.

**The bright time** is the mean of the events' bright times (the cumulative
fit fits it as it fits the dark time).  Two docking strands bound at once
are one event, so a site of many strands has slightly long events.

**Missed bindings.**  A binding shorter than about half a frame lights no
frame and is not seen; its dark times on either side are seen as one.  At a
mean bright time of 5 frames that is about 10 % of them, and the index is
that much low -- for every site alike, so the calibrated influx carries it
and the counts are not affected.

**Curation values** (Picasso's frame analysis rejects a cluster whose mean
frame is outside 20 to 80 % of the acquisition, or that has more than 80 %
of its localizations in one twentieth of it): `mean_frame_rel`,
`std_frame_rel` (0.29 for a constant rate) and `peak_window_fraction`, each
over the localizations, each frame as a fraction of its own file's
acquisition.  `uniform_p` is the p-value of a Kolmogorov-Smirnov test of the
events' first frames against a constant rate.

## Parameters

### link_gap
Picasso's default is 3 frames.  Longer joins two bindings of a site with
many strands into one; shorter splits a binding the fit missed a frame of.
Keep it the same for calibration and sample.

### estimator
*maximum likelihood* unless the numbers are to be compared with Picasso's.

### influx
From [qPAINT](plugin:Analysis/Regions/Summarize/qPAINT) on regions of known
site number, with the same *link gap*; it depends on the imager's
concentration and the buffer, so calibrate it on every sample.

### frames
Set it when the table's localizations do not reach the first or the last
frame (a filter on frames, a merged file).

## Output

A row of the region table per region (and the values of a run on the view):

* `n_localizations`, `n_events`, `locs_per_event` -- the localizations, the
  binding events and the localizations in each.
* `first_frame`, `last_frame`, `span_frames` -- of the localizations.
* `longest_dark_frames`, `longest_bright_frames` -- the longest gap between
  events and the longest event: a sticky imager is an event hundreds of
  frames long.
* `mean_bright_frames`, `mean_dark_frames` -- the plain means.
* `tau_bright_frames`, `tau_dark_frames` and in seconds `tau_bright_s`,
  `tau_dark_s` -- by the *estimator*; NaN seconds without a
  [frame time](plugin:Panels/Glossary).
* `qpaint_index`, `qpaint_index_err` -- the binding rate per frame and its
  error.
* `n_sites`, `n_sites_err` -- the index over the *influx*; NaN without one.
* `dark_total_frames`, `dark_periods` -- the numerator and denominator of the
  mean dark time, which *qPAINT* sums over regions.
* `mean_frame_rel`, `std_frame_rel`, `peak_window_fraction`, `uniform_p` --
  how evenly the localizations and events spread over the acquisition.

The region keeps the trace in *windows* windows of the acquisition (`frame`,
`events`, `localizations` and `cumulative`, the fraction of the events begun
so far), which *Curves* averages over regions of one file, and each event's
first frame, bright time and dark time.  The figure is the trace: a good
site's cumulative count follows the diagonal; a step is a sticky period, a
bend a site lost or gained during the acquisition.

## References

* R. Jungmann, M. S. Avendaño, M. Dai, J. B. Woehrstein, S. S. Agasti,
  Z. Feiger, A. Rodal, P. Yin, "Quantitative super-resolution imaging with
  qPAINT", *Nature Methods* 13, 439-442 (2016),
  [doi:10.1038/nmeth.3804](https://doi.org/10.1038/nmeth.3804) -- counting
  sites from the binding rate.
* J. Schnitzbauer, M. T. Strauss, T. Schlichthaerle, F. Schueder,
  R. Jungmann, "Super-resolution microscopy with DNA-PAINT", *Nature
  Protocols* 12, 1198-1228 (2017),
  [doi:10.1038/nprot.2017.024](https://doi.org/10.1038/nprot.2017.024) --
  Picasso, whose pick kinetics this ports.
