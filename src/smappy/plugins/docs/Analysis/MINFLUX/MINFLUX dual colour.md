---
version: "2"
covers: [smappy.dual_colour.link_traces, smappy.dual_colour.fit_track_offset, smappy.dual_colour.bead_offset, smappy.dual_colour.mbm_colour_drift, smappy.dual_colour.offset_curve, smappy.dual_colour.direction, smappy.pairing.pair_traces, smappy.pairing.pair_offsets, smappy.plugins.dual_colour.DualColour.run, smappy.plugins.dual_colour._bead_like, smappy.dual_colour.offset_trends, smappy.plugins.dual_colour.draw_field, smappy.plugins.dual_colour.trends_line]
---

## What it does

A two-colour [MINFLUX](plugin:Panels/Glossary) measurement localizes each
dye with its own laser, the two interleaved in time: a kinesin with one
head labelled for 640 nm and the other for 560 nm is seen in both.  Two
things are not in the file.  Which [trace](plugin:Panels/Glossary) of one
colour belongs with which of the other -- the instrument numbers each
colour's traces on its own.  And how far apart the colours are: the two
lasers do not meet in exactly one place, by some 10 nm, which matters
when the heads themselves are 8 nm apart.

This plugin does both, in this order:

* **the offset between the colours**, from a bead measured in both colours
  (the usual way: a measurement of its own before the tracking, better one
  after it too), or, where none is open, from the molecules themselves;
  plus the colours' slow drift against each other over the measurement,
  from the [MBM](plugin:Panels/Glossary) beads where they were recorded in
  both colours;
* **colour 1 moved onto colour 0** by that offset;
* **the traces joined into molecules**: a trace of each colour measured at
  the same time *and* in the same place.  Every localization gets a
  `pair_id`, one number per molecule, in one colour or both.

The colour is the laser, `channel` (0 the longer wavelength) as the
`.msr` reader writes it, so the bead and the tracking measurement agree
even when their threads ran in opposite order; a table without it uses the
thread, `thi`.

Regions of `pair_id` -- [Segment regions](plugin:Analysis/Regions/Segment regions)'
*from an id attribute*, with its *MINFLUX kinesin* preset -- are then one
molecule each, which is what
[Find steps](plugin:Analysis/Tracking/Find steps)' *pairs* and the trace
inspector need.  The [MINFLUX kinesin](plugin:Analysis/MINFLUX/Workflow/MINFLUX kinesin)
chain runs it with the steps before and after.

It reads the localizations the layer's filter keeps, of the measurements
the selection holds; a bead measurement is read whole.  On a table of one
colour it says so and changes nothing, so that a chain made for two colours
runs on one as well.

## How it works

```figure-setup
from smappy import dual_colour as DC

rng = np.random.default_rng(0)
c_true, lead_true = np.array([-1.5, 11.5]), -6.0
rows, t, tid = [], 0.0, [100, 5000]
for k in range(50):
    walk = k < 40
    phi = rng.uniform(0, 2 * np.pi)
    u = np.array([np.cos(phi), np.sin(phi)])
    start = rng.uniform(0, 20000, 2)
    tid = [tid[0] + 1, tid[1] + 1]
    for j in range(120):
        ts = t + 3e-3 * j
        col = j % 2                  # the colours alternate in time
        p = start + (u * 400.0 * (ts - t) if walk else 0.0)
        if col:
            p = p + c_true + (lead_true * u if walk else 0.0)
        rows.append((tid[col], col, ts, *(p + rng.normal(0, 4.0, 2))))
    t += 0.5
a = np.array(rows)
links = DC.link_traces(a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2], a[:, 3], a[:, 4],
                       max_distance=100.0)
both = links.both
fit = DC.fit_track_offset(links.dx[both], links.dy[both], links.ux[both], links.uy[both],
                          links.displacement_nm[both] > 30)
```

**1. The offset from a bead measurement.**  A bead in the sample measured
in both colours, in a measurement of its own: its traces are paired by
time, each colour-1 localization matched with the colour-0 one nearest in
time, and the offset is the median of their differences -- over the
emitters, the median of each one's median.  On *auto* the open measurement
that is not being paired and was taken last before the tracking is used,
and one taken after it, if any, too.  A measurement read together with the
tracking (one `.msr`, merged) is taken for beads when its typical
localization is in a trace of 1000 or more: a bead is followed for
thousands, a motor for tens to hundreds.

**2. Before and after.**  With a bead measurement before and one after,
the offset at any time between is the first one, plus the MBM colour drift
since (step 4), plus whatever of the difference between the two the drift
does not explain, spread evenly over the time between.  That part is
reported: a large one says the drift does not follow the colours.

**3. The offset from the molecules.**  Without a bead measurement -- or
when *offset from* says so -- each molecule seen in both colours gives an
offset, and the offset is fitted to them all.  A molecule that walks adds
its own part: the two colours label different places of it (for a
kinesin, the two heads, half a step apart along the track), so its offset
is the registration *c* plus a *lead* *a* along the direction it walks
in.  With molecules walking in many directions the two are told apart:
*d* = *c* + *a* *u*, *u* the direction.  A still molecule tells only *c*.
The fit is robust, so the few traces of two different molecules that pass
as one count for nothing.

```figure Simulated molecules, 40 walking in random directions and 10 still, colour 1 offset by (−1.5, 11.5) nm and 6 nm behind colour 0 along the motion: each molecule's offset (dots) spreads round the registration along the direction it walks; the fit finds the registration (cross) and, from the spread, the lead.
fig.set_size_inches(7.6, 3.2)
ax, bx = fig.subplots(1, 2)
mv = links.displacement_nm[both] > 30
ax.plot(links.dx[both][~mv], links.dy[both][~mv], "s", ms=4, color="0.6", label="still")
ax.plot(links.dx[both][mv], links.dy[both][mv], "o", ms=4, color="0.3", label="walking")
ax.plot([fit.c[0]], [fit.c[1]], "x", ms=11, mew=2, color="tab:red", label="fit")
ax.plot([c_true[0]], [c_true[1]], "+", ms=13, color="k", label="truth")
ax.set_xlabel("dx (nm)")
ax.set_ylabel("dy (nm)")
ax.set_aspect("equal", adjustable="datalim")
ax.legend(fontsize=6, frameon=False)
ax.set_title(f"registration ({fit.c[0]:.1f}, {fit.c[1]:.1f}) nm", fontsize=8)
res = np.column_stack([links.dx[both], links.dy[both]]) - fit.c
along = res[mv, 0] * links.ux[both][mv] + res[mv, 1] * links.uy[both][mv]
bx.hist(along, bins=15, color="0.6")
bx.axvline(fit.a, color="tab:red", lw=1.5, label=f"fit {fit.a:.1f} nm")
bx.axvline(lead_true, color="k", lw=1, ls="--", label="truth -6 nm")
bx.set_xlabel("colour 1 ahead along the motion (nm)")
bx.set_ylabel("molecules")
bx.legend(fontsize=6, frameon=False)
bx.set_title("the lead along the track", fontsize=8)
```

**4. The colours' drift from the MBM beads.**  Where the measurement
recorded its MBM beads in both colours, each bead's colour-1 position less
its colour-0 one is followed over time.  Each bead has a bias of its own
-- in the measurement looked at, tens of nanometres and different for
every bead -- so its absolute value says nothing; how it changes does.
Each bead's difference is fitted with an offset of its own and one smooth
curve through them all, the beads that disagree left out, as
[MBM drift correction](plugin:Analysis/Drift/MBM) does with positions.  The
curve, zero at the beads' first point and held flat after their last, is
added to the offset.

**5. Colour 1 moved.**  Colour 1's localizations of the measurements being
paired are moved by the offset at their time, `x_nm` and `y_nm`, and
`x_nc_nm`, `y_nc_nm` where Imspector kept them.  The shift is recorded
with the table: a second run takes it off first and moves from the
positions as read.

**6. Molecules.**  A trace of colour 0 and one of colour 1 that overlap in
time are candidates.  Each colour-1 localization is matched with the
colour-0 one nearest in time (within *match within*); the two traces are
one molecule when enough localizations match (*matched at least*) and
enough of the matches -- *for a share of* -- are no further apart than
*apart at most*.  The share, not the median distance: two molecules whose
paths cross are close for a while either side of the crossing, and a
median can pass where a share does not.  The accepted candidates are
joined greedily, the most close matches first, so that a 560 nm trace that
stopped and started again on the same kinesin joins its 638 nm trace
(and the other way round), as long as no two traces of one colour in a
molecule overlap in time: two at once are two molecules.

**7. Is one offset enough?**  The offset applied is one shift (plus the MBM
drift).  What each molecule in both colours has left of its offset once
that is taken off is checked for what one shift would miss: a gradient
across the field -- a plane fitted to the leftovers against x and y -- and
a drift in time -- a line against time.  The text says, for each, whether a
slope stands out of its error, and the fifth panel draws each molecule's
leftover as an arrow at its place.  A slope there says the bead or the
molecules' average does not hold everywhere in the field or all through
the measurement.  With few molecules -- the kinesin measurement has a dozen
in both colours -- two at the edge of the field decide a plane, so a
gradient is fitted only from 20 molecules, a drift in time from 8.  The
molecules' own spread (the dyes' places on each molecule, a few nm) is
what a slope has to stand out of.

```figure Simulated: 40 molecules over 4 x 3.5 um whose offset left over grows by 1.5 nm per um along x, 2 nm of scatter (left), and 12 of them (right): with 40 the arrows turn from left to right across the field and the gradient is found; with 12 none is offered.
from smappy.plugins.dual_colour import draw_field
fig.set_size_inches(7.6, 3.2)
g = np.random.default_rng(3)
fx, fy = g.uniform(-2000, 2000, 40), g.uniform(-1000, 2500, 40)
ft = g.uniform(0, 600, 40)
left = np.column_stack([1.5 * fx / 1000 + g.normal(0, 2, 40), g.normal(0, 2, 40)])
ax, bx = fig.subplots(1, 2)
draw_field(ax, fx, fy, left, DC.offset_trends(fx, fy, ft, *left.T))
draw_field(bx, fx[:12], fy[:12], left[:12],
           DC.offset_trends(fx[:12], fy[:12], ft[:12], *left[:12].T))
```

## In detail

**Matching in time.**  Within two traces, for each colour-1 localization
the colour-0 one before and the one after it in time, the nearer taken;
further apart than *match within*, it is left out.

**Direction and displacement.**  A molecule's direction is the long axis
of its colour-0 localizations (their first principal component), turned
forward in time; how far it moves, the median position along that axis of
its last fifth less that of its first, so that the ends' noise is averaged
rather than its extremes taken.  It is still below *moving from*.

**The fit.**  Least squares of *d* = *c* + *a* *u* over the molecules
(two equations each, one per axis), reweighted with Tukey's biweight on
the length of each molecule's residual, tuning constant 4.685 robust
standard deviations, the spread 1.4826 times the median residual and
never under 1 nm.  *a* is fitted only with three or more moving molecules
whose directions are not all one: with every track parallel, a lead and a
registration cannot be told apart, and the plugin says so.  When the
offset comes from the molecules, a first pass joins them with *apart at
most* widened to 100 nm (or three times it), the fit is made on that
pass's molecules, each less the MBM drift at its time, and the molecules
are joined again with the offset taken off.

**The MBM drift.**  A bead's two colours are the two entries of its slot
(`gri`) in the bead table; each colour-1 point is matched with the
colour-0 point of the same bead nearest in time, within 2 s.  The curve is
`beads.select_beads`' spline, a coefficient every *drift knot spacing*,
with its rules for leaving a bead out.

**Times across measurements.**  A bead measurement's time is put on the
tracking's clock through the two acquisition dates (to the second), so
that a bead measurement six minutes before the tracking sits six minutes
before it.  Where the bead measurement has no MBM beads of its own, the
drift between it and the tracking's first MBM point is not measured, and
the plugin warns.

**`pair_id`** is written as a recipe with the rule *max*, so a merged
trace keeps its molecule; -1 outside the measurements paired, outside the
layer's filter, or without a trace.  Molecules are numbered by when they
start.

**What is left** (`offset_trends`).  For each molecule in both colours,
its median offset as measured less the offset curve at its time.  Two fits,
each on its own: dx and dy against x and y (a plane, slopes in nm per um),
and against time (a line, nm per 100 s).  Each molecule counts once.  Both
are robust as the offset fit is -- Tukey's biweight on the length of each
molecule's residual, constant 4.685, the spread 1.4826 times the median
residual and never under 1 nm -- and their errors come from the kept
molecules' scatter.  A slope "changes" when it is more than three times
its error: four slopes across the field are tested, and at twice the error
one would stand out by chance one time in five.  Fewer than 20 molecules:
no plane; fewer than 8: no line.  It is a diagnostic: nothing it finds is
applied.

## Parameters

### offset.source
*auto* takes the bead measurements where one is open beside the tracking,
else the molecules.  *the molecules* is worth comparing with the beads:
the two should agree within the fit's error, and when they do not, the
bead measurement does not describe the tracking -- other detectors, laser
power or sequence, or drift between the two.

### offset.knot_s
The colours drift slowly (1 to 2 nm per 100 s in the measurement looked
at) and the MBM beads are measured every 15 to 20 s with a few nanometres
of noise: a short spacing follows the noise.

### pairing.max_distance_nm
The two labels' distance in the molecule, plus twice the localization
precision, plus what is left of the registration: for kinesin heads
(at most about 16 nm apart) at 5 nm precision, 30 nm.  Wider joins more
chance neighbours; *for a share of* keeps most of those out.

### pairing.min_fraction
0.5 by default.  Raise it where molecules are dense, lower it for a
molecule whose colours separate for a while (a cargo that swings).

## Output

`pair_id` on the localizations; colour 1 moved; a region table of
`pair_id`, "per pair_id: MINFLUX dual colour", per molecule: whether it is
in both colours (`both`), the traces joined (`n_traces`), localizations per
colour (`n_0`, `n_1`), the matches (`n_matched`), their share within the
distance (`within`) and median distance (`separation_nm`), the offset left
after the move (`dx_nm`, `dy_nm`), and how far and fast colour 0 moved
(`displacement_nm`, `velocity_nm_per_s`).  The constant offset is kept as
a named transformation ("560 nm onto 638 nm") for
[Transform](plugin:Analysis/Process/Transform).

The text adds a line on what is left after the offset: the slopes across
the field and over time with their errors, or that there were too few
molecules to tell.

Five panels: each molecule's offset as measured, with the offset used; the
same over time with the offset curve -- the molecules should scatter round
it, with no trend away from it; the MBM colour drift, each bead about its
own offset and the curve; and how far colour 1 is ahead of colour 0 along
the motion, with the fitted lead; and what each molecule has left of its
offset, as an arrow at its place in the field -- arrows that turn or grow
across the field are a gradient one shift misses, arrows pointing every
way are the molecules' own spread.

## Differences from SMAP

Ported in idea from SMAP's MINFLUX plugins Segment_cotracks,
DCoffset_Beads_ROIs and StepsMINFLUX_dc (Ries 2020).

* **Pairing**: SMAP takes for each thread-0 trace in a site the most
  frequent thread-1 trace within its time span; here every trace is
  joined at once, by time and place, a molecule allowed several traces of
  a colour one after another, without sites.
* **The offset**: SMAP measures it on beads in drawn ROIs and applies it by
  hand (*offx*, *offy*); here it is found -- from a bead measurement, two
  bracketing ones, or the molecules with their lead -- and applied, with
  the MBM colour drift, which SMAP did not use.
* **The colour** is the laser, not the thread.

## References

* Ries J (2020).  SMAP: a modular super-resolution microscopy analysis
  platform for SMLM data.  *Nat Methods* 17, 870-872.
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
