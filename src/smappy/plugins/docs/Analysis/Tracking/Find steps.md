---
version: "7"
covers: [smappy.bnp_steps.bnp_steps, smappy.bnp_steps.noise_sd, smappy.steps.greedy, smappy.steps.s_peak, smappy.steps.fit_from, smappy.steps.autostepfinder, smappy.steps.refine, smappy.steps.split_merge, smappy.steps.step_table, smappy.steps.erlang_shape, smappy.steps.dwell_models, smappy.plugins.find_steps.model_line, smappy.plugins.find_steps.FindSteps.run, smappy.plugins.find_steps.pooled_steps, smappy.plugins.find_steps.steps_file, smappy.traces.moving_direction, smappy.plugins.find_steps.draw_steps, smappy.plugins.find_steps.rois_ids, smappy.dual_colour.co_movement]
---

## What it does

Finds the steps a molecule takes along its path -- a kinesin or dynein
walking 8 nm at a time, measured with [MINFLUX](plugin:Panels/Glossary) at
a few hundred localizations per second -- in every
[trace](plugin:Panels/Glossary) or track at once, and gathers what they say
about the motor: the step size, how long it waits between steps (the dwell
time), how many hidden sub-steps those waits suggest, and its velocity.

Each trace's positions are turned onto the direction it walks in where it
moves -- the parts where it is stuck left out -- forward in time, and the
steps are found along it with
AutoStepFinder (Loeff, Kerssemakers et al. 2021,
[doi:10.1016/j.patter.2021.100256](https://doi.org/10.1016/j.patter.2021.100256)),
SMAP's StepsMINFLUX's method, or with BNP-Step (Rojewski et al. 2024,
[doi:10.1016/j.bpj.2024.01.008](https://doi.org/10.1016/j.bpj.2024.01.008)),
which learns the number of steps by Bayesian sampling -- slower, and
better where the steps are small against the noise.  Where the expected
step size is known, steps smaller than it are merged and larger ones split.  The per-trace numbers are
a [region table](plugin:Panels/Glossary) of the
[id attribute](plugin:Panels/Glossary); the pooled step sizes and dwell
times are drawn and fitted.  On request the plateaus go onto the
localizations, or into a file of one localization per plateau.

## How it works

```figure-setup
from smappy import steps as S

rng = np.random.default_rng(0)
levels, pos, starts = [], 0.0, []
while len(levels) < 600:
    d = max(3, int(round(rng.gamma(2.0, 15.0))))     # dwells: 2 hidden sub-steps
    starts.append(len(levels))
    levels += [pos] * d
    pos += 8.0
x = np.array(levels[:600]) + rng.normal(0, 3, 600)    # 3 nm localization precision
t = np.arange(600) * 1e-3
order, Scurve = S.greedy(x, 150)
best, s3 = S.s_peak(Scurve)
found = S.autostepfinder(x)
b = S.split_merge(x, found.boundaries, 8.0)
from smappy.bnp_steps import bnp_steps
bnp = bnp_steps([x], sweeps=200).boundaries[0]
```

**1. One coordinate per trace.**  The localizations of each trace (at least
*at least* of them, those the layer's filter keeps) in time order, turned
onto the direction the trace walks in -- or x or y, with *steps along*.
That direction is found where the motor moves: the trace's mean position
over windows of *moving window* (0.1 s), and between neighbouring windows
how far it went; the windows it moved in faster than *moving from*
(100 nm/s) are summed into one direction.  A motor that sits still for
half its trace, wandering a little across its track, is turned along where
it walked, where the long axis of all its localizations is pulled across
by the wander.  A trace that never moves faster gets its long axis
(*track axis from*: *all its localizations* takes that always).

**2. A greedy fit.**  Starting from one level, the plateau whose best split
removes the most squared error is split, again and again.  A plateau of
fewer than four points is not split.

**3. How many steps: the S-score.**  The splits each plateau would make
next form a step function of their own, the *counter-fit*.  While the fit
has not yet found all real steps, the counter-fit holds some of them and
fits nearly as well; once it has, the counter-fit only splits noise and
fits much worse.  The ratio of their squared errors, the S-score, peaks at
the number of real steps.  The curve is taken minus one, and minus a line
from zero to its end, before its peak is read.

```figure A simulated motor track, 8 nm steps after waits of 30 ms on average, 3 nm precision: the S-score against the number of steps fitted peaks near the number of steps there are (left); the fit at that number (right).
fig.set_size_inches(7.0, 2.8)
a, c = fig.subplots(1, 2)
a.plot(np.arange(len(s3)), s3, color="0.2")
a.axvline(best, color="tab:red", lw=0.8)
a.axvline(len([s for s in starts if 0 < s < 600]), color="0.6", ls=":")
a.set_xlabel("steps fitted")
a.set_ylabel("S-score (baseline removed)")
a.set_title(f"peak at {best}; dotted: the true number", fontsize=8)
c.plot(t, x, ".", ms=1.5, color="0.6")
c.step(t, S.fit_from(x, b), where="post", color="tab:red", lw=1)
c.set_xlabel("time (s)")
c.set_ylabel("position (nm)")
```

**4. Two rounds.**  The steps the first round missed are looked for in what
is left of the trace after it, the residual.  A round counts only where its
S-score peaks above *threshold*.  The steps of both rounds, *overshoot*
times as many, make the fit.

**Or BNP-Step.**  With *method* on *BNP-Step*, a trace is a level plus up
to M possible steps, each switched on or off, with a height and a time of
its own, and Gaussian noise.  A sampler draws, again and again, each switch
given the rest, the heights, the times and the noise, starting from
AutoStepFinder's steps; a step is kept where it explains the data better
than its cost in the prior, so the number of steps is learned rather than
chosen by a threshold.  The answer is the sample the data make most
probable.  Every trace of the table is sampled at once.

```figure BNP-Step on the same track, without the expected step size: the steps it keeps (red) against the true number (title).
fig.set_size_inches(7.0, 2.6)
a = fig.subplots()
a.plot(t, x, ".", ms=1.5, color="0.6")
a.step(t, S.fit_from(x, bnp), where="post", color="tab:red", lw=1)
a.set_xlabel("time (s)")
a.set_ylabel("position (nm)")
a.set_title(f"{len(bnp)} steps found; {len([s for s in starts if 0 < s < 600])} there", fontsize=8)
```

**5. Split and merge.**  With *split/merge to* set to the expected step
size s: two steps in a row that add up to less than 1.4 s, the first under
0.7 s, become one; a step over 1.4 s gets a plateau inserted halfway.  Ten
rounds, each followed by refining the boundaries: every boundary moves by
up to two points to where the points lie closer to their own level.
Without it the boundaries are refined once.

**6. Per step, per trace, pooled.**  Each plateau's level is its points'
mean (or median, *levels*); a step is the difference between two levels,
a dwell the time from a plateau's first point to the next plateau's (the
first and last plateaus, cut by the trace's ends, left out).  Pooled over
all traces: the step size (mean and standard deviation of |size|, and the
fraction of forward steps), the mean dwell, and the dwell distribution's
**Erlang shape** -- mean² / variance: 1 when one event sets the pace
(exponential dwells), n when n events in a row do.  The velocity is the
mean step times its net direction over the mean dwell.

**Dwell models.**  The dwells are fitted by maximum likelihood with three
models, as MINFLUXStepsAnalyze compared them: one rate (an exponential:
one event sets the pace), a gamma of free shape (n equal events in a row;
its shape is n), and two different rates in a row (a hypoexponential).
They are compared by the Akaike information criterion, AIC = 2 x parameters
- 2 log-likelihood: the lowest is the model the dwells support, and a
difference of a few units is no evidence either way.

**7. Two colours.**  On a two-colour MINFLUX measurement, *two colours* on
*auto*: the traces [MINFLUX dual colour](plugin:Analysis/MINFLUX/MINFLUX dual colour)
joined into molecules (`pair_id`, taken when no other id is chosen) are
turned into one frame -- the direction the molecule walks in, about the
centre of its colour-0 localizations -- so both colours lie along the same
line, on one clock, and each colour's localizations get their own steps.  The colour is
the laser, `channel` (0 the longer wavelength), where the reader knew it,
else the thread, `thi`.  A motor and its cargo, or the two heads of one
motor in two colours, then step side by side in one plot (SMAP's
StepsMINFLUX_dc), and the region table holds the median step of each colour
(`step_median_channel0_nm`, `step_median_channel1_nm`; `..._thi0_nm` where
the colour is the thread).  The step count, dwell and velocity in the
table are colour 0's.  *as one trace* fits both colours' localizations as
one, which is only right for one label in two colours.

**8. Moving together.**  A molecule in two colours should move as one: the
same way, at the same speed, close together.  Two molecules that met by
chance can pass the pairing's distance test for a moment and then part.
Over the time both colours are seen, cut into windows of *window* (0.5 s;
0, one window over the whole overlap -- the two lines of a linear stepper
compared), a straight line x(t), y(t) is fitted to each colour.  A window
where the molecule moves (the faster colour at *moving from* or more)
matches when the two directions are within *directions within* (30°), the
speeds within *speeds within* (a factor 1.5) and the two lines within
*lines within* (50 nm) at the window's centre.  The share of the moving
windows that match is `comove_share`.  A molecule that never moves, or
whose colours are seen too briefly to fit a line through more than a step
or two, is not judged (NaN): that says nothing either way.

**9. In an ROI.**  Run as a per-region step in the ROI manager, it measures
the ROI's molecule and proposes the direction it walks in as the ROI's
angle -- set only by *Apply proposals* -- which the ROI then shows its
molecule turned by, and the steps of anything else measured in it are
along.  A two-colour molecule that was judged is proposed a keyword in the
*colours* list too: *dual* when at least *together from* (half) of its
moving windows match, *apart* when fewer do -- in both colours, but not
moving as one.  [Segment regions](plugin:Analysis/Regions/Segment regions)
names the ROIs *single* or *dual* when it makes them; *Apply proposals*
turns the *dual* ones that do not move together into *apart*.

## In detail

**The split.**  For a plateau [a, e) the best split k maximises the squared
error removed, SS(a, e) − SS(a, k) − SS(k, e), over 2 ≤ k − a ≤ e − a − 2
(any k for plateaus of two or three points); the plateau with the largest
such gain is split next (Kerssemakers et al. 2006).  SMAP ranked by
(level difference)² × width instead, which favours a split at a plateau's
end; the gain does not.

**The S-score** with n steps is SS(counter-fit) / SS(fit), the counter-fit's
plateaus running between all current plateaus' next best splits.  Up to
⌈n_points / 4⌉ steps are tried.

**Overshoot** multiplies the number of steps taken from the two rounds'
lists in their order (the first round's, then the second's new ones), as
SMAP's *coarseness* did.

**BNP-Step's model** (Rojewski et al. 2024): x_n = F + Σ_m b_m h_m [n ≤ k_m]
+ noise of precision η, with b_m ~ Bernoulli(γ/M) (γ = 1), h_m ~ N(0,
*step prior*²), k_m uniform over the points, F normal about the trace's end
and η Gamma about the noise estimated from consecutive differences
(1.4826 MAD / √2).  M is twice the most steps AutoStepFinder finds in the
batch, at least 20.  Each sweep draws every b_m (a logistic draw on the
change of the squared error), every k_m (Metropolis, proposed uniformly),
F and the heights jointly (their Gaussian conditional) and η (its Gamma
conditional); the kept sample is the one of highest posterior.  The code
departs from the authors' in computing every squared error from prefix
sums of the trace instead of points × steps matrices, in sampling all
traces as one batch, in starting from AutoStepFinder (they start from the
prior and run 50 000 sweeps) and in priors scaled to the data; on simulated
staircases it finds the same number of steps as their sampler.  The
boundaries are not refined afterwards: the sampler's times already are
where the data put them.

**Moving together** (`co_movement`).  Each window's lines are least
squares in x and y against time, centred on the window; the angle is
between the two velocity vectors, the speed ratio the faster over the
slower (≥ 1), the distance between the two lines' positions at the
window's centre.  The windows are the overlap divided into whole windows
nearest *window* long.  A window is judged only when each colour has at
least 5 localizations in it and, at the faster colour's speed, is seen over
*judged from* (30 nm) or more: a line through 40 ms of a stepping motor is
one plateau or one step, and on the kinesin data a colour seen that briefly
came out five times slower and 136° off.  The medians over the judged
windows are `comove_angle_deg`, `comove_speed_ratio` and
`comove_distance_nm`; `comove_windows` counts them.

**Not done** from AutoStepFinder: its merging of steps by their error and
its bootstrap errors; from StepsMINFLUX: the running-window smoothing, the
"bad step" flag near gaps in time, manual editing and the movie.

## Parameters

### along
Along the track for motors on straight filaments; x or y where the motion
is known to be along an axis, or the track bends so its long axis means
little.

### min_locs
A trace must hold several steps for the S-score to peak: 20 localizations
at least, more for small steps in large noise.

### method
AutoStepFinder takes a second for a hundred traces; BNP-Step tens of
seconds, and is the better choice for small steps in noisy traces, or when
the number of steps should not hang on a threshold.

### sweeps
A few hundred from AutoStepFinder's start is usually enough; more where the
traces are long and the steps many.  The run time grows with it linearly.

### height_nm
Steps much smaller than it cost more prior than they gain in fit, so they
are dropped; on *auto* three times the noise.  Set it near the expected
step (8 nm for kinesin) when the noise is small.

### threshold
0.15 is AutoStepFinder's.  Lower finds steps in noisier traces and
starts finding noise.

### overshoot
Below 1 takes fewer, more certain steps; above 1 more, splitting dwells.

### stat
The median is robust to a few outliers within a plateau; the mean is
AutoStepFinder's own.

### in_rois
For the motors chosen by hand or by
[Segment regions](plugin:Analysis/Regions/Segment regions)' *from an id
attribute*: only the ids of the used ROIs are fitted, so the pooled steps
and dwells are theirs.  A region filter would not do it: the per-trace
numbers never see one.

### step_nm
8 for kinesin and dynein along microtubules, 36 for myosin V.  Leave it at
0 to see what the data find without a prior.

### colours
Each colour of a molecule has only its own localizations, about half of
the molecule's: dwells shorter than a few localizations per colour merge.

### comove.window_s
0.3-1 s for a kinesin: several steps per window, so the lines are not one
step's.  0 compares one line per colour over everything, right for a
motor that walks straight and steadily; windows also follow a curved
microtubule and a motor that pauses.

### comove.min_share
0.5 calls a molecule *apart* when most of its moving windows disagree.  Raise
it to keep only molecules that move together throughout.

### window_s
A few tens of localizations per window: at a few hundred localizations a
second, 0.1 s.  Shorter windows see a kinesin's single steps and its
pauses between them as moving and stuck.

### min_speed
Above the windows' noise (a few nm over 0.1 s, some tens of nm/s) and
below the motor's: 100 nm/s for kinesin at saturating ATP; lower it for a
slow motor.

### output
*plateaus on the localizations* writes `plateau` (an id per plateau, -1 for
none) and `step_level_nm`, to colour by or to filter on; *steps as a file*
adds a file of one localization per plateau, at its points' mean position,
with its level, dwell and size -- MotorPAINT's map of steps.

## Output

A region table of the id attribute, "per tid: Find steps": `n_steps`,
`step_median_nm`, `step_mean_nm`, `dwell_mean_s`, `velocity_nm_per_s`,
`erlang_shape`, the direction of the track (`angle_deg`) and, from the
moving parts, the share of windows it moved in (`moving_share`); with two
colours the median step of each, and how the colours move together:
`comove_share`, `comove_angle_deg`, `comove_speed_ratio`,
`comove_distance_nm`, `comove_windows` -- NaN where it could not be judged,
and the values [Filter regions](plugin:Analysis/Regions/Filter regions)
keeps molecules by (`comove_share >= 0.5`).  The text counts the
two-colour molecules that move and those of them that do not move
together.  The text adds the dwell models, the best
first with each one's parameters and its AIC above the best.  The figure:
the upper half, the longest traces (in an ROI, its molecule) side by side,
the localizations as dots joined in time by a thin line (*localizations as
a line*) and the steps over them, a two-colour molecule's colours in
magenta (colour 0) and green on one clock from its first localization in
either and from colour 0's first plateau; below, the step sizes, and the
dwell times with the three models (the best drawn thicker), the mean dwell
and the velocity in its title.  The histograms' bins grow with the number
of steps.  Steps clustered at the expected size,
mostly forward, and an Erlang shape near the number of events per step
(about 2 for kinesin's alternating heads) look like a motor; a broad size
distribution with many backward steps, like steps fitted to noise.

## Differences from SMAP

Ported from SMAP's StepsMINFLUX and MINFLUXStepsAnalyze (Ries 2020).

* **All traces at once**, not one ROI-manager site at a time; the per-trace
  results are a region table, the pooled analysis part of the same run.
* **The split criterion** is the squared error removed, not SMAP's (level
  difference)² × width.
* **The dwell analysis** fits an exponential, a gamma and a two-rate
  hypoexponential by maximum likelihood on the dwells and compares them by
  AIC; MINFLUXStepsAnalyze fitted hypoexponential and Erlang models of
  fixed order to histograms and cumulative distributions.
* SMAP's smoothing modes disagreed between finding the boundaries and
  measuring the levels; here both use the same positions.

## References

* Loeff L, Kerssemakers JWJ, Joo C, Dekker C (2021).  AutoStepfinder: a
  fast and automated step detection method for single-molecule analysis.
  *Patterns* 2, 100256.
  [doi:10.1016/j.patter.2021.100256](https://doi.org/10.1016/j.patter.2021.100256)
* Rojewski A, Schweiger M, Sgouralis I, Comstock M, Pressé S (2024).  An
  accurate probabilistic step finder for time-series analysis.  *Biophys
  J*.  [doi:10.1016/j.bpj.2024.01.008](https://doi.org/10.1016/j.bpj.2024.01.008)
  -- BNP-Step; preprint
  [doi:10.1101/2023.09.19.558535](https://doi.org/10.1101/2023.09.19.558535),
  code [LabPresse/bnp-step](https://github.com/LabPresse/bnp-step).
* Kerssemakers JWJ et al. (2006).  Assembly dynamics of microtubules at
  molecular resolution.  *Nature* 442, 709-712.
  [doi:10.1038/nature04928](https://doi.org/10.1038/nature04928)
  -- the greedy fit and the counter-fit.
* Deguchi T et al. (2023).  Direct observation of motor protein stepping
  in living cells using MINFLUX.  *Science* 379, 1010-1015.
  [doi:10.1126/science.ade2676](https://doi.org/10.1126/science.ade2676)
* Ries J (2020).  SMAP: a modular super-resolution microscopy analysis
  platform for SMLM data.  *Nat Methods* 17, 870-872.
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
