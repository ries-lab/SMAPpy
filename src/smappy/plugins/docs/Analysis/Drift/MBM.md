---
version: "1"
covers: [smappy.mbm.measurements_of, smappy.mbm.choose, smappy.mbm.drift_of, smappy.mbm.replace_correction, smappy.beads.select_beads, smappy.beads.fit_offsets_and_spline, smappy.beads.bead_metrics, smappy.beads.MadRule, smappy.beads.draw_bead_tracks, smappy.io.msr.bead_table, smappy.drift.Drift.at, smappy.simulate.beads.simulate_bead_tracks]
---

## What it does

A [MINFLUX](plugin:Panels/Glossary) microscope keeps its sample in place
with [MBM](plugin:Panels/Glossary) (beam-line monitoring): a few beads fixed
on the coverslip are measured every couple of seconds, and Imspector
subtracts their mean displacement from every localization.  That works
while every bead sits still.  A bead that comes loose, is lost and found
again in another place, or is dim and noisy, moves every localization by
its share of the mean -- with four beads, a quarter of whatever it does.
And the beads' own scatter goes into the correction undamped: in 3D a bead
is measured to about 10 nm from one point to the next.

This plugin reads the beads' tracks from the file, leaves out the ones that
disagree with the others, and fits one smooth curve to the rest.  It then
corrects the localizations from scratch with that curve, in place of
Imspector's correction.

Use it on MINFLUX data from an Abberior `.msr` whose measurements recorded
MBM beads (File > Open shows "MBM n beads" beside them).  The beads are
read from the file itself, so nothing else needs to be opened; a bead file
opened beside the measurement (*MBM beads as a file* in File > Open) is
corrected with it.  The plugin reads the localizations of each file in the
[selection](plugin:Panels/Glossary), whatever the layer shows, and corrects
the whole of each file.

Some files keep the beads only from before the measurement starts.  There
is then no track to correct with: the curve is flat after the beads' last
point, and the plugin warns that the beads cover only part of the
measurement.

## How it works

```figure-setup
from smappy.simulate import simulate_bead_tracks
from smappy.beads import select_beads, MadRule, draw_bead_tracks
t = np.arange(10.0, 200.0, 2.0)
truth = np.column_stack([10 * np.sin(t / 30), 0.05 * t, 3 * np.cos(t / 50)])
beads = simulate_bead_tracks(t, truth, n_beads=5, jumps=[(2, 120.0, 25.0)],
                             precision_nm=2.0, seed=0)
xyz = np.column_stack([beads["x_nm"], beads["y_nm"], beads["z_nm"]])
chosen = select_beads(beads["time_s"], xyz, beads["tid"],
                      ["R0", "R1", "R2", "R3", "R4"], knot=8.0)
every = select_beads(beads["time_s"], xyz, beads["tid"], knot=8.0,
                     rule=MadRule(k=1e9))
# Imspector's: each bead from its first point, averaged
shift = []
for k in range(1, 6):
    rows = beads["tid"] == k
    shift.append(np.column_stack([np.interp(t, beads["time_s"][rows],
                                            xyz[rows, a] - xyz[rows, a][0])
                                  for a in range(3)]))
imspector = np.mean(shift, axis=0)
```

**1. The beads' tracks.**  Each measurement keeps, for each of its beads,
a position every couple of seconds.  They are on the same clock as the
localizations.

**2. One curve through all of them.**  Each bead sits at its own place, and
all of them move together with the drift.  So every bead's track is the
same curve plus that bead's own place (its offset).  The curve and the
offsets are fitted together.  The curve is a smooth spline with a value
every *knot spacing* seconds, 8 s by default.  A bead seen only part of the
time counts for that part.

**3. Each bead judged against the others.**  Each bead is compared with
the curve the *other* beads give.  Leaving it out matters when there are
few: a bead inside the fit pulls the curve towards itself and hides part of
its own error.  Three numbers describe how a bead deviates:

* *noise*: how much it scatters from one point to the next;
* *wander*: how far it moves slowly on its own;
* *jump*: the largest sudden step.

A bead is left out when one of these is well above the other beads'.  Then
the curve is fitted again without it, and the beads are judged again, until
nothing changes.

```figure Five beads (thin lines) on the curve they agree on (black); bead R2 (red, dashed) jumps by 25 nm at 120 s and is left out.  Right: each bead's deviation from the curve -- the jump is plain, the others scatter by their precision.
fig.set_size_inches(9, 5.4)
draw_bead_tracks(fig, chosen, ("x", "y", "z"))
```

**4. The correction, from scratch.**  Every localization gets Imspector's
uncorrected position (`x_nc_nm`) minus the new curve at its time.  The
beads' tracks are uncorrected, so the curve already holds everything
Imspector corrected: it replaces Imspector's correction rather than adding
to it.  Running the plugin again gives the same table.

```figure The drift as Imspector corrected it (orange: the mean of all five beads) and as this plugin does (black), against the simulated drift (grey, wide), in x, each about its mean.  Imspector's carries a fifth of the jump from 120 s on, and the beads' scatter; the curve through the four good beads is 0.4 nm rms from the simulated drift.
fig.set_size_inches(7, 2.8)
ax = fig.subplots()
ours = chosen.drift_nm(t)
centred = lambda v: v - v.mean()
ax.plot(t, centred(truth[:, 0]), color="0.75", lw=5, label="simulated")
ax.plot(t, centred(imspector[:, 0]), color="C1", lw=1, label="Imspector's")
ax.plot(t, centred(ours[:, 0]), color="k", lw=1, label="this plugin")
ax.set_xlabel("time (s)"); ax.set_ylabel("drift x (nm)"); ax.legend(fontsize="small")
```

**5. Your own choice.**  *Select beads...* under the settings opens the
[bead selector](plugin:Panels/Bead selector): the beads in a table with
their numbers, and their tracks.  A bead can be included or excluded there
by hand, whatever the rule says; the choice is written into *by hand*, so a
chain or a batch makes it again.

## In detail

Point *i* of bead *b*, at time t, is modelled as

$$x_i = o_b + d(t_i) + \varepsilon_i$$

with *o_b* the bead's offset and *d* a cubic B-spline with a coefficient
every *knot spacing*.  All offsets and coefficients are one linear
least-squares problem, as in the [fiducial](plugin:Analysis/Drift/Fiducials)
correction, with the same small penalty on the coefficients' second
differences (10^-3 of the mean data weight per coefficient).  A constant
can move from every offset into the curve; the curve is fixed at zero at
the measurement's first bead point, which is where Imspector's own
correction is zero too, so the corrected positions stay where Imspector put
them up to the difference of the two curves.

A bead's numbers come from its residual about the fit of the other used
beads, with its own offset fitted over its points:

* *noise* is $1.4826\,\mathrm{MAD}(\Delta r) / \sqrt{2}$ per axis, added in
  quadrature: the median absolute deviation of the differences between
  consecutive points, which is blind to slow motion;
* *wander* is $\sqrt{\mathrm{rms}^2 - \mathrm{noise}^2}$, the part of the
  residual that noise does not explain;
* *jump* is the largest distance between the medians of five points before
  and five points after any point;
* *seen* is the fraction of the beads' time span the bead covers.

For each of noise, wander and jump, a bead is out when its value exceeds
the median of the used beads by more than *robust sigmas* times their
robust standard deviation (1.4826 MAD) **and** by more than *margin*.  The
margin is there because a few beads that agree have a tiny spread: without
it, a bead 0.3 nm noisier than the other three would count as bad.
Rejecting and refitting repeat until the set of beads no longer changes, at
most 10 times; if fewer than *min beads* would be left, the least deviant
come back.  A choice by hand is never overruled.

Each measurement is fitted on its own, since Imspector finds its beads
anew for each and reuses the names.  The curves of a merged file's
measurements are laid end to end; each covers its measurement's
localizations as well as its beads, held flat beyond the beads.  The curve
is sampled every eighth of the knot spacing (at most every 0.5 s) and
interpolated linearly at each localization's `time_s`.

On a real 3D measurement with four beads, all of Imspector's beads with no
rejection give a curve 0.7 nm (x, y) and 0.08 nm (z) rms from Imspector's
correction with a knot every 8 s, against 2.6 nm with a knot every 2 s,
which follows every point of the beads' scatter.

## Parameters

### knot_s
8 s follows changes of drift over tens of seconds and averages the beads'
scatter over about four points each.  Shorter follows faster drift and more
of the scatter; below the beads' period (2 s) the curve has nothing to fix
it between points.

### start
*Imspector's beads* begins from the beads Imspector used, which is what
its correction was made from; *all beads* lets the rule consider the ones
Imspector set aside as well.

### rejection.k
Larger keeps more beads.  With a handful of beads the spread is uncertain,
and *margin* usually decides.

### rejection.margin_nm
A bead is never rejected for being less than this worse than the others'
median.  Set it from the precision of the beads: about the beads' noise,
2 nm by default.

### rejection.min_beads
At least two beads are needed to tell which one moved.

### beads
Written by the bead selector; it can be typed too.  With several
measurements in the selection a bead's name starts with its measurement's
label up to the first underscore (`261019-133540/R0`).

### use_z
On *auto*, z is corrected for a 3D measurement.  Off leaves z as Imspector
corrected it.

### reference
*as estimated* keeps each measurement's curve at zero at its first bead
point, as Imspector's; *first frame* and *last frame* set it to zero at
the first or the last time of the measurement.

## Output

The corrected table, as one undo step; `x_nc_nm` and the other uncorrected
columns stay as they were, so the plugin can be run again with other
beads.  The text says how many beads there were, how many were used, how
many the rule left out and how many were chosen by hand, then the range of
the drift.  The figures are the drift curve and, per measurement, the
beads on it and each bead's deviation.  A good result has every used bead
scattering evenly about zero on the right; a used bead whose deviation has
a step or a slope is one to exclude by hand.
