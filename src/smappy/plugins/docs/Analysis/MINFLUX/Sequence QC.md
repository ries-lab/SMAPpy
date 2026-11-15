---
version: "1"
covers: [smappy.minflux_qc.outcome, smappy.minflux_qc.final_iteration, smappy.minflux_qc.iteration_events, smappy.minflux_qc.per_iteration, smappy.minflux_qc.time_per_iteration, smappy.minflux_qc.catching_offsets, smappy.minflux_qc.traces, smappy.minflux_qc.exponential_mean, smappy.minflux_qc.summary, smappy.plugins.sequence_qc.SequenceQC.run, smappy.plugins.sequence_qc.shown_files, smappy.plugins.sequence_qc.index_profile, smappy.simulate.minflux.simulate_attempts]
---

## What it does

Shows how a [MINFLUX](plugin:Panels/Glossary) measurement ran: how many of
the molecules the microscope started to localize it lost on the way and
why, how much background it saw, how close its first guesses were to where
each molecule turned out to be, how many localizations and photons each
molecule gave, and where the time went.  These are the plots abberior's
ParaFlux draws, and the ones its wiki tells a MINFLUX user to look at
before trusting a measurement or changing the sequence or the buffer.

MINFLUX localizes a molecule in steps, called **iterations**.  The first,
the *scouting* iteration, looks for a molecule with a wide beam pattern.
If it finds one, the *catching* iterations close in on it, each with a
smaller pattern.  The *final* iteration, with the smallest pattern, then
repeats for as long as the molecule stays on, and each repetition is one
localization.  An attempt can end at any iteration: too few photons (the
molecule went dark), a failed centre-frequency-ratio check (`cfr`, a sign
of two molecules on at once), or a position out of the scanner's range.
The localizations of one molecule while it was on form a
[trace](plugin:Panels/Glossary) (`tid`), which ParaFlux calls a *burst*.

It needs the file opened with **every iteration** ticked (File > Open, or
the Load plugin's *Abberior MSR* part): a normal table holds only the
localizations, not the attempts that failed or the iterations before.
It reads every row of the files the layer shows, whatever the layer's
filter: that filter starts on the localizations (`final` = 1), and the
attempts that failed have no position for a filter or a drawn ROI to keep.

## How it works

```figure-setup
from smappy.simulate import simulate_attempts
from smappy.io import minflux
from smappy import minflux_qc as q
from smappy.plugins import sequence_qc as sq

passed = (0.08, 0.9, 0.6)              # per iteration before the final one
seq = {"iterations": [{"Mode.pattern": "hexagon", "patGeoFactor": g}
                      for g in (0.79, 0.84, 0.42, 0.28)]}
rec = simulate_attempts(4000, passed=passed, locs_per_burst=15.0,
                        catching_nm=(60.0, 20.0, 8.0), seed=2)
t, _ = minflux.table(rec, sequence=seq, every=True)
c = lambda name: np.asarray(t[name])
events = q.iteration_events(c("iteration"), c("sta"), c("final"), c("vld"))
traces = q.traces(c("tid"), c("sta"), c("final"), c("time_s"), None, c("photons"))
offsets = q.catching_offsets(c("iteration"), c("final"), c("tid"), c("trace_index"),
                             np.c_[c("x_nm"), c("y_nm")])
```

**1. The attempts, iteration by iteration.**  Every scouting event is an
attempt.  For each iteration before the final one, it counts how many
attempts reached it and how they ended there, as a share of all the
attempts: the 100 % line is every molecule the scouting iteration looked
at.  Why an iteration ended is the instrument's status code (`sta`):
passed, failed the cfr check, dark (no signal) or out of range.  The final
iteration repeats, so it is counted per row instead: the share of its rows
that were valid localizations is in the panel's title.

```figure Simulated attempts with known pass rates (8 %, 90 % and 60 % in the three iterations before the final one, so 8 %, 7.2 % and 4.3 % of the attempts pass): the bars and the crosses, the truth, agree.  Most attempts end in scouting, dark -- in a real measurement too, where most of the scouting iteration's rows find nothing.  The losses in iteration 2 are 80 % dark and 20 % failed cfr checks, as simulated.
fig.set_size_inches(5.0, 3.2)
ax = fig.subplots()
sq.draw_events(ax, events)
truth = 100 * np.cumprod(passed)
ax.plot(range(len(truth)), truth, "kx", ms=8, label="truth")
ax.set_ylim(0, 12)
ax.legend(fontsize=7, frameon=False)
```

**2. The background.**  Imspector estimates the background (`fbg`) in
each iteration where background sensing is on; the median and quartiles
per iteration are drawn.  With background sensing off -- as in many files
-- there is nothing to draw, and the panel says so.

**3. The catching offsets.**  Each catching iteration gives a position.
How far it lies from the localization it led to -- the first localization
of the trace after it -- shows how well the sequence closes in: the offsets
should shrink with each iteration and scatter around zero.  A mean far from
zero in one direction points to a misaligned pattern or pinhole; a large
spread in the first iterations is expected, since their patterns are wide.

```figure The offset of each catching iteration's position from the localization it led to, mean ± standard deviation per axis, for attempts simulated to scatter by 60, 20 and 8 nm in iterations 0, 1 and 2 (grey bars), about molecules localized to 4 nm.
fig.set_size_inches(5.0, 3.0)
ax = fig.subplots()
for k, s in enumerate((60.0, 20.0, 8.0)):
    ax.bar(k, 2 * np.hypot(s, 4.0), bottom=-np.hypot(s, 4.0), width=0.5, color="0.9")
sq.draw_offsets(ax, offsets, ["x_nm", "y_nm"])
```

**4. Bursts: localizations and photons.**  For every trace that gave at
least one localization (a valid burst) it counts the localizations and
adds up the photons of all its rows, the failed iterations included (`eco`
+ `ecc`, the photon budget).  The number of localizations per burst is
roughly exponentially distributed; its mean is the estimate drawn.
ParaFlux's guidance: around 15 for immunostained cells, more for
DNA-PAINT, under 10 means molecules are lost early.

```figure Localizations per valid burst for bursts simulated with a mean of 15: the estimate comes back.
fig.set_size_inches(5.0, 3.0)
sq.draw_bursts(fig.subplots(), traces)
```

**5. Where the time goes, and the parameters per iteration.**  Each row
lasts until the next one starts; adding that up per iteration gives the
share of the acquisition each iteration took -- usually most of it in
scouting, looking for molecules.  Any attribute (*per iteration*, the
photons by default) is drawn per iteration as median and quartiles, and
along the first *first localizations* of every trace, which shows a trace's
start settling in.

**6. efo against cfr.**  For the localizations whose cfr was measured,
their photon rate (`efo`) against their cfr: one molecule gives one cloud
at low cfr; two molecules on at once give a second one at higher cfr and
higher efo.  In 2D sequences cfr is measured once a trace, so the panel is
a scatter of few points.

## In detail

**The final iteration** is the highest iteration that holds a localization
(a row with `final` = 1: valid and marked final by the instrument).  The
iterations below it are counted per attempt: the share of the attempts
that reached iteration k and ended with outcome o is the number of rows of
iteration k with that outcome over the number of rows of iteration 0.
In a 3D sequence, whose iterations alternate between xy and z, every
iteration is counted; ParaFlux draws only the complete xy-and-z steps.

**The outcomes** come from `sta`, which the two record layouts code
differently (abberior's wiki, "MINFLUX Data Parameters"): m2410 1 valid
final, 2 valid, 6 failed cfr check, 8 no signal, 9 DAC out of range, 11
background measurement, 4 and 5 derived or reserved; m2205 0 valid, 4089
cfr, 4096 no signal, 4100 out of range.  A code not in either list counts
as passed where the row is valid, else as other.

**Why a trace ended** is its last row's code, a background measurement (11)
aside.  A valid burst whose last row is a localization ended with the
measurement.

**The catching offsets** pair each non-final row that has a position with
the localization of the same trace that has the same `trace_index` -- the
number of the localization a row leads to.  The offset is the row's
position minus that localization's; the mean and the population standard
deviation are taken per iteration and axis.

**Localizations per burst**: the mean over the valid bursts, which is the
maximum-likelihood estimate of a geometric distribution's mean.

**The time** per row is the time to the next row, all rows sorted by
`time_s`; the last row gets none.

**Per trace** it also keeps a [region table](plugin:Panels/Glossary) of
`tid`, "per tid: Sequence QC": each trace's rows (`n_rows`), localizations
(`n_localizations`), photon budget, duration and its last status
(`end_sta`), for the region filter and the summaries.

## Parameters

### parameter
`photons` (the final iteration's `eco`, Imspector's effective counts at
the pattern's offsets) shows whether the catching iterations collect what
the sequence's photon limits ask; `efo` and `fbg` whether the signal or the
background changes along the sequence; `xy_err_nm` the precision each
iteration reaches.

### first_localizations
About 25 shows a trace's start creeping into place, which in 10-30 % of
traces lasts 5 to 30 localizations; longer only averages fewer traces.

## Output

Nine panels: the attempts per iteration, the background per iteration,
the catching offsets, localizations per burst, photons per burst, the
chosen attribute per iteration, efo against cfr, the time per iteration,
and the chosen attribute along each trace's start.  The text gives the
attempts, localizations and bursts, the share passed per iteration, why
the bursts ended, and the scouting's share of the time.  The numbers are
kept with the result for a batch to compare measurements: `passed_itr{k}`,
`localizations_per_event`, `fbg_itr{k}_khz`, `locs_per_burst`,
`photons_per_burst`, `ended_dark` and the other reasons,
`scouting_time_fraction`.

A good measurement of a sparse, well-blinking sample passes 40-50 % of
the attempts that reach the catching iterations into the final one
(ParaFlux's rule of thumb for the last bar), loses few to the cfr check,
and gives bursts of 15 or more localizations.  Large cfr losses mean too
many molecules on at once (the buffer); large dark losses, too little
excitation or molecules that blink off too soon.

## References

* abberior's wiki, "ParaFlux" and "MINFLUX Data Parameters m2410"
  (wiki.abberior.rocks, read 2026-11-15; a login from abberior).
* Balzarotti F et al. (2017).  Nanometer resolution imaging and tracking
  of fluorescent molecules with minimal photon fluxes.  *Science* 355,
  606-612.  [doi:10.1126/science.aak9913](https://doi.org/10.1126/science.aak9913)
  -- MINFLUX.
