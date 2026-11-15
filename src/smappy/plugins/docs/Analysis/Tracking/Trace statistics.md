---
version: "1"
covers: [smappy.traces.runs, smappy.traces.axes, smappy.traces.rotate, smappy.traces.consecutive_precision, smappy.traces.steps, smappy.traces.rolling_std, smappy.traces.tails, smappy.traces.cfr_cut, smappy.traces.statistics, smappy.traces.convergence, smappy.plugins.trace_statistics.TraceStatistics.run]
---

## What it does

Measures every [trace](plugin:Panels/Glossary) (a MINFLUX `tid`) or track
(a linked `track_id`) at once, as a molecule's path: how far it went from
its first localization to its last and how fast, which way, how elongated
its cloud of localizations is, how precise its localizations are judged
from consecutive ones, and how far it wanders from a straight line.  It
also finds each trace's **start** -- the first localizations of a still
molecule that creep into place in MINFLUX, or the approach of a motor
before it settles on its track -- and can measure each trace without it.

The numbers are kept as a [region table](plugin:Panels/Glossary) of the
[id attribute](plugin:Panels/Glossary), "per tid: Trace statistics", for
*Filter regions*, the summaries and *Regions to attribute*.  On request it
also writes onto the localizations each track turned onto its own axes
(`x_rot_nm`, `y_rot_nm`: the long axis along x, pointing the way the
molecule moved) and which localizations are its tail (`in_tail`), to
filter or colour by.

Use *Id statistics* for a quick count, centre, spread and the statistics of
other attributes per trace; use this for the trace's path.  It reads the
localizations, not merged blinks, and those the layer's filter keeps.

## How it works

```figure-setup
from smappy import traces as T

rng = np.random.default_rng(1)
# a motor: 0.5 nm per ms at 30 degrees for 200 ms, localized to 3 nm
t1 = np.arange(200) * 1e-3
s = 500 * t1
x1 = s * np.cos(np.radians(30)) + rng.normal(0, 3, 200)
y1 = s * np.sin(np.radians(30)) + rng.normal(0, 3, 200)
# still molecules whose first 5 to 25 localizations creep in from 20-60 nm
xs, ys, ts, ids, truth = [], [], [], [], []
for k in range(40):
    m, n_creep, d0 = 120, rng.integers(5, 26), rng.uniform(20, 60)
    creep = np.r_[np.linspace(d0, 0, n_creep), np.zeros(m - n_creep)]
    a = rng.uniform(0, 2 * np.pi)
    xs.append(creep * np.cos(a) + rng.normal(0, 2, m))
    ys.append(creep * np.sin(a) + rng.normal(0, 2, m))
    ts.append(np.arange(m) * 1e-3)
    ids.append(np.full(m, k + 2))
    truth.append(n_creep)
X = np.r_[x1, np.concatenate(xs)]
Y = np.r_[y1, np.concatenate(ys)]
TT = np.r_[t1, np.concatenate(ts)]
I = np.r_[np.full(200, 1), np.concatenate(ids)]
r = T.runs(I, TT)
x, y, t = X[r.order], Y[r.order], TT[r.order]
st = T.statistics(r, x, y, t)
xr, yr = T.rotate(r, x, y, st)
tail = T.tails(r, x, y, t, "still", 20, 30, 3.0)
```

**1. Each trace in time order.**  The localizations of each trace (one
value of the id attribute; -1 is none) are put in the order of `time_s`,
or of `frame` where there is no clock.

**2. Path and speed.**  The displacement is the distance from the trace's
first localization to its last; the velocity is that displacement over the
time between them.  For a molecule that moves steadily, as a motor, it is
its speed; for one that diffuses or stays put it is small and mostly
noise.

**3. The track's axes.**  The spread of a trace's positions has a long and
a short direction (the eigenvectors of its covariance).  The standard
deviations along them, their ratio (the *eccentricity*: 1 for a round
cloud, large for a line) and the long axis' angle -- turned to point the
way the molecule went, from its first quarter to its last -- are measured.
Turned onto these axes, a motor's track runs along x.

```figure A simulated motor track, 0.5 nm per ms at 30° for 200 ms with 3 nm precision, as measured (left) and turned onto its own axes (right): it runs along x, forward in time (colour), its width across the track the precision.
fig.set_size_inches(7.0, 2.8)
left, right = fig.subplots(1, 2)
one = r.code == 0
left.scatter(x[one], y[one], c=t[one], s=4, cmap="viridis")
left.set_aspect("equal")
left.set_xlabel("x (nm)")
left.set_ylabel("y (nm)")
right.scatter(xr[one], yr[one], c=t[one], s=4, cmap="viridis")
right.set_xlabel("along the track (nm)")
right.set_ylabel("across (nm)")
right.set_ylim(-30, 30)
left.set_title(f"angle {st['angle_deg'][0]:.1f}°, {st['velocity'][0]:.0f} nm/s", fontsize=8)
```

**4. Precision from consecutive localizations.**  The difference between
neighbouring localizations of a trace scatters by √2 times their
precision, so std(differences) / √2 estimates the precision -- for a still
molecule exactly, for a moving one blurred by what it moved in between.
Across a straight track (the short axis) the motion does not enter: it is
the precision.

**5. The trend across the track.**  The spread across the track beyond
that scatter -- √(std across² − precision across²) -- is how far the path
bends away from a straight line: 0 for a straight track.

**6. The start of a trace (tail).**  With a *tail rule* chosen, each
trace's start is found:

* *still molecule*: the trace's settled position is the median of its
  localizations from *settled from* over *settled over* of them.  The tail
  is every localization before the first one that comes within *reach*
  robust standard deviations of that position, the robust standard
  deviation being 1.4826 times the median distance of the settled
  localizations from it;
* *moving molecule*: the rolling standard deviation of the steps between
  consecutive localizations, over *rolling window* of them, falls below
  *settled below*; the tail is everything before.

With *numbers without the tail* ticked (the default), every number above is
measured on the rest of the trace.  Per trace it reports the tail's length
in localizations, how far its first localization was from the settled
position, and how long the tail lasted.

```figure Forty simulated still molecules whose first 5 to 25 localizations creep in from 20-60 nm, localized to 2 nm: the tail the still-molecule rule finds against the simulated one (left), and the pooled distance of each trace's n-th localization from its settled position (right).
fig.set_size_inches(7.0, 2.8)
left, right = fig.subplots(1, 2)
left.plot(truth, tail.n[1:], "o", color="0.3", ms=4)
left.plot([0, 26], [0, 26], ":", color="0.6")
left.set_xlabel("simulated creep (localizations)")
left.set_ylabel("tail found")
conv = T.convergence(r, tail.distance, 40)
right.plot(conv["index"], conv["mean"], "o-", ms=3, color="0.2")
right.set_xlabel("localization in its trace")
right.set_ylabel("distance from settled (nm)")
```

The rule stops a few localizations early where the creep's last steps are
within the scatter: those localizations are as good as the rest.

**7. Cut after a cfr violation.**  For MINFLUX, *cut after cfr above* ends
each trace at its first localization whose `cfr` exceeds the limit -- a
second molecule switched on -- and measures it without the rest.

## In detail

**Covariance and axes.**  Per trace, from the sums of x, y, x², y² and xy
over its rows: the sample variances (ddof 1) and covariance, the
eigenvalues λ± = tr/2 ± √(tr²/4 − det); `std_long_nm` = √λ+, `std_short_nm`
= √λ−, θ = ½ atan2(2 c_xy, v_x − v_y), turned by π where the mean of the
trace's last quarter lies behind the mean of its first quarter along θ.
`angle_deg` is θ in degrees, 0-360.

**Consecutive precision.**  The differences between rows k and k+1 of the
same trace, both kept; their sample standard deviation over √2.  On the
rotated coordinates for `precision_long_nm` and `precision_short_nm`.

**The still rule's spread** is a median absolute distance, 1.4826 times
its median: a robust standard deviation that one outlier in the settled
window does not move.  A trace shorter than *settled from* has no settled
position and no tail.

**The moving rule's rolling standard deviation** is centred on each
localization, over *rolling window* steps of the same trace (fewer at its
ends), needing three.

**Velocity** is in nm per second with `time_s`, nm per frame without.  The
median time between consecutive localizations is `dt_median_s` (or
`_frames`).

## Parameters

### tail.rule
*still molecule* for immobile emitters -- DNA-PAINT sites, labelled
structures, beads -- where a trace should stay put once settled; *moving
molecule* for motors and tracking, where the trace moves on and only its
noise settles.  A still rule on moving molecules finds nothing useful: the
"settled" position is a moment of the path.

### tail.without_tail
Untick it to measure the whole trace, tail included, and see how much the
tail changes the numbers.

### tail.reference_from
Later than the longest creep expected: 20 is past most MINFLUX creep
(5-30 localizations); traces shorter than this plus a few have no tail.

### tail.reference_window
30 localizations give a settled position to about a fifth of the
precision.

### tail.k
3 robust sigmas: a settled localization is beyond it about once in a
hundred.  Smaller finds longer tails and starts cutting into the settled
part.

### tail.rolling_window
10 steps, the MFX scripts' choice.

### tail.rolling_limit_nm
About twice the localization precision: a moving molecule's steps scatter
by √2 times the precision plus its motion between localizations.

### cfr_limit
0.8 as abberior's sequences check during the acquisition; lower to be
stricter.  Localizations whose cfr was not measured (NaN) never cut.

### write_rotated
For step finding by eye, kymographs along the track and motor-PAINT maps of
track directions.

## Output

A region table of the id attribute with, per trace: `n_locs`,
`displacement_nm`, `duration_s`, `velocity_nm_per_s`, `std_long_nm`,
`std_short_nm`, `eccentricity`, `angle_deg`, `precision_x_nm`,
`precision_y_nm`, `precision_long_nm`, `precision_short_nm`, `trend_nm`,
`dt_median_s`, `photons_sum`, and with a tail rule `tail_n`,
`tail_reach_nm`, `tail_duration_s`; with a cfr cut `cut_n`.  Four panels:
the velocity, the precision across the track and the eccentricity over the
traces, and the pooled distance of each trace's n-th localization from its
settled position, beside 120 nm/√N for the photons collected by then.

A good still-molecule measurement shows precisions close to `xy_err_nm`,
eccentricities near 1 and short tails; two emitters in one trace show as
a large spread and eccentricity.  A motor shows a high eccentricity, a
steady velocity and a small trend.

## References

* Schmidt R et al. (2021).  MINFLUX nanometer-scale 3D imaging and
  microsecond-range tracking on a common fluorescence microscope.  *Nat
  Commun* 12, 1478.
  [doi:10.1038/s41467-021-21652-z](https://doi.org/10.1038/s41467-021-21652-z)
  -- precision from a trace's localizations.
* Deguchi T et al. (2023).  Direct observation of motor protein stepping
  in living cells using MINFLUX.  *Science* 379, 1010-1015.
  [doi:10.1126/science.ade2676](https://doi.org/10.1126/science.ade2676)
  -- tracks turned onto their axes, precision from consecutive differences.
