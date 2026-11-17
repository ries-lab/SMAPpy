---
version: "3"
covers: [smappy.tracking.analysis.gather, smappy.tracking.analysis.steps_of, smappy.tracking.analysis.lag_sums, smappy.tracking.analysis.msd_covariance, smappy.tracking.analysis.lag_errors, smappy.tracking.analysis.optimal_lags, smappy.tracking.analysis.fit_msd, smappy.tracking.analysis.msd_analysis, smappy.tracking.analysis.jumps, smappy.tracking.analysis.jumps_over_lags, smappy.tracking.analysis.in_focus, smappy.tracking.analysis.fit_jumps, smappy.tracking.analysis.jump_density, smappy.tracking.spectrum.pieces_of, smappy.tracking.spectrum.piece_loglik, smappy.tracking.spectrum.state_array, smappy.tracking.spectrum.spectrum, smappy.tracking.spectrum.peaks, smappy.tracking.spectrum.per_localization, smappy.tracking.hmm.steps_of, smappy.tracking.hmm.track_loglik, smappy.tracking.hmm.state_probabilities, smappy.tracking.hmm.fit_hmm, smappy.tracking.hmm.per_localization, smappy.plugins.tracking_analysis.TrackingAnalysis.run]
---

## What it does

Measures how [tracks](plugin:Panels/Glossary) move.  It needs tracks, an
id attribute that says which localizations belong to one particle -- the
*track id*: on *auto*, `track_id` from
[Link tracks](plugin:Analysis/Tracking/Link tracks), else MINFLUX's `tid`,
else a simulation's `track_id_true`.  Two analyses, each ticked on or off:

* **MSD** ([mean squared displacement](plugin:Panels/Glossary)): how far a
  particle has moved, squared and averaged, after 1, 2, 3, ... frames.  For
  free diffusion it grows in a straight line whose slope is the
  [diffusion coefficient](plugin:Panels/Glossary) *D*.  It is fitted for
  every track and for all tracks pooled.  The number of points fitted is
  chosen per track as Michalet showed it must be
  ([Michalet 2010](https://doi.org/10.1103/PhysRevE.82.041914)): few when
  the localization error is small compared with a frame's motion, more
  when it is large.  It also gives the
  [anomalous exponent](plugin:Panels/Glossary) α, which tells free
  diffusion (1) from confined (below 1) and directed motion (above).
* **Jump distances**: the distances particles move in 1, 2, 3, ...
  frames, as a mixture of one to three
  [diffusive states](plugin:Panels/Glossary), each with its *D* and its
  fraction -- a bound and a free population, say.  In 2D tracking, fast
  particles leave the thin focus within a few frames and are seen less
  than their number; with an *axial range* the fractions are corrected
  for that, as Spot-On does
  ([Hansen et al. 2018](https://doi.org/10.7554/eLife.33125)), with the
  frame-by-frame correction of saSPT
  ([Heckert et al. 2022](https://doi.org/10.7554/eLife.70169)).
* **D spectrum** (off by default): the same jumps shared out over a fine
  grid of *D* rather than one to three states -- saSPT's *state arrays*
  (Heckert et al. 2022) -- so the number of populations is not chosen
  but read off the peaks.  Each track is cut into short pieces, and each
  piece gets its own *D*, which can be written onto its localizations
  (`D_post_um2_per_s`) to colour, filter or split the tracks by, and
  which shows a particle that changes state within a track.
* **State switching** (off by default): states that change in time, as a
  [hidden Markov model](plugin:Panels/Glossary) -- ExTrack's
  ([Simon et al. 2023](https://doi.org/10.1083/jcb.202208059)).  It gives
  each state's *D*, the rates at which particles switch between them
  (binding and unbinding, say), how long a state lasts, and the state of
  every localization (`hmm_state`).

All take the [localization precision](plugin:Panels/Glossary) into
account: a molecule that does not move still seems to, by its localization
error, and an analysis that ignores it finds every molecule mobile.

Time comes from the file's frame time (recorded by the camera, set in
*File > File properties*, or here under *frame time*); a table with its own clock (`time_s`, MINFLUX)
uses that.  It analyses the localizations the layer's filter and the ROI
keep.  Per region it runs too: in each ROI of the ROI manager, or in a
block over `track_id`, once per track.  With *write per track*, each
track's *D* and α are written onto its localizations
(`track_D_um2_per_s`, `track_alpha`), to colour or filter the tracks by.

## How it works

```figure-setup
from smappy.simulate import simulate_tracks
from smappy.tracking.analysis import (gather, msd_analysis, lag_errors, jumps,
                                      jumps_over_lags, fit_jumps, jump_density,
                                      in_focus)
one = simulate_tracks(n_frames=1000, density_um2=0.02, d_um2_s=(0.1,),
                      track_frames=40.0, precision_nm=30.0, seed=1)
two = simulate_tracks(n_frames=1000, density_um2=0.04, d_um2_s=(0.01, 0.5),
                      fractions=(0.35, 0.65), track_frames=40.0,
                      precision_nm=20.0, seed=2)
thin = simulate_tracks(n_frames=1500, density_um2=0.02, d_um2_s=(0.05, 3.0),
                       fractions=(0.5, 0.5), track_frames=30.0,
                       precision_nm=25.0, focal_range_nm=700.0, seed=3)
three = simulate_tracks(n_frames=1500, density_um2=0.02, d_um2_s=(0.02, 0.3, 2.0),
                        fractions=(0.3, 0.3, 0.4), track_frames=30.0,
                        precision_nm=25.0, seed=7)
hmm_locs = simulate_tracks(n_frames=800, density_um2=0.02, d_um2_s=(0.02, 1.0),
                           fractions=(0.5, 0.5), switch_probability=0.05,
                           track_frames=40.0, precision_nm=25.0, seed=4)
switching = simulate_tracks(n_frames=1000, density_um2=0.01, d_um2_s=(0.02, 2.0),
                            fractions=(0.5, 0.5), switch_probability=0.03,
                            track_frames=100.0, precision_nm=25.0, seed=12)

def tracks(locs):
    return gather(locs["track_id_true"], locs["frame"], locs["x_nm"],
                  locs["y_nm"], err_xy=locs["xy_err_nm"], frame_time_s=0.01)
```

**1. Displacements.**  The tracks are sorted by id and time.  For every
lag of k frames, the squared distance between every two localizations of a
track exactly k frames apart is averaged: that track's MSD at lag k.
Pairs across a dark frame count at their true lag.  The MSD of all tracks
pools the pairs of every track.

**2. The offset.**  A particle's measured position is off by its
localization error, so even at a lag of zero frames the MSD would not be
zero: in two dimensions it starts at $4\sigma^2$, with $\sigma$ the
precision.  It is taken from the precisions of the localizations
(*offset*: from the precisions), or fitted with the slope.

**3. How many points.**  The first MSD points are averaged over many
pairs but carry the localization error; the later ones are averaged over
few.  Somewhere between is the number of points that gives the most
precise *D*.  It depends on how long the track is and on how large the
error is compared with a frame's motion, and it is computed for each track
(see *In detail*).

**4. D and α.**  A straight line through the offset fitted to those points
gives *D*; the slope of the MSD above the offset on a log-log plot gives α.

```figure Left: simulated tracks at D = 0.1 µm²/s with 30 nm precision; the MSD of all of them (dots), the points the fit used (blue) and the line through the offset. Right: how precise D is when fitted to the first p points, for a track of 50 positions, at three ratios of the localization error to a frame's motion: when the error is small one point is best, when it dominates many are, and fitting the wrong number can be several times worse.
fig.set_size_inches(9, 3.6)
left, right = fig.subplots(1, 2)
r = msd_analysis(tracks(one))
k = int(r.pooled.lags[0])
left.plot(r.lag_s[:15], r.msd_nm2[:15] / 1e6, "o", ms=3, color="0.6")
left.plot(r.lag_s[:k], r.msd_nm2[:k] / 1e6, "o", ms=4, color="C0")
line = np.linspace(0, r.lag_s[14], 50)
D = r.pooled.D_um2_per_s[0]
left.plot(line, r.pooled.offset_nm2[0] / 1e6 + 4 * D * line, color="C3",
          label=f"D = {D:.3f} µm²/s (truth 0.1)")
left.set_xlim(0, line[-1])
left.set_ylim(0, None)
left.set_xlabel("lag (s)")
left.set_ylabel("MSD (µm²)")
left.legend(fontsize=8)
for x, colour in ((0.1, "C0"), (10.0, "C1"), (100.0, "C3")):
    err = lag_errors(50, x)
    right.plot(np.arange(1, len(err) + 1), err, color=colour, label=f"x = {x:g}")
    best = int(np.nanargmin(err))
    right.plot(best + 1, err[best], "o", color=colour)
right.set_yscale("log")
right.set_xlabel("MSD points fitted")
right.set_ylabel("relative error of D (one axis)")
right.legend(fontsize=8)
```

**5. Jump distances.**  The distances between localizations of a track
exactly 1, 2, ... frames apart are collected, up to *lags* frames.  Each
diffusive state gives jumps of a known shape -- wider for a larger *D*
and a longer lag, and widened by the two localizations' errors -- and the
states' *D* and fractions are those that make the observed jumps, at all
lags together, most likely.

```figure Two populations, 35% at 0.01 µm²/s and 65% at 0.5 µm²/s. Left: D per track, on a log axis -- two groups, broad because each track is short. Right: the jumps between consecutive frames, the two states fitted to them and their sum (black), which give back both coefficients and fractions.
fig.set_size_inches(9, 3.6)
left, right = fig.subplots(1, 2)
t = tracks(two)
r = msd_analysis(t)
d = r.per_track.D_um2_per_s
d = d[np.isfinite(d) & (d > 0)]
left.hist(d, bins=np.logspace(-3.5, 0.5, 40), color="0.6")
left.set_xscale("log")
for truth in (0.01, 0.5):
    left.axvline(truth, color="C3", lw=1)
left.set_xlabel("D per track (µm²/s)")
left.set_ylabel("tracks")
d2, err = jumps(t, 1)
fit = fit_jumps(d2, err, 2, 0.01, 2)
jump = np.sqrt(d2)
grid = np.linspace(0, np.quantile(jump, 0.995), 200)
right.hist(jump, bins=grid[::4], density=True, color="0.75")
parts = jump_density(grid, fit, float(np.median(err)), 2, 0.01)
for i, part in enumerate(parts):
    right.plot(grid, part, color=f"C{i}",
               label=f"{fit.D_um2_per_s[i]:.3f} µm²/s, {100 * fit.fraction[i]:.0f}%")
right.plot(grid, parts.sum(0), color="k", lw=1, label="both")
right.set_xlabel("jump (nm)")
right.set_yticks([])
right.legend(fontsize=8)
```

**6. The focus.**  A camera sees a particle only while it is within a few
hundred nm of the focal plane -- the *axial range*.  A slow particle stays
there for many frames; a fast one diffuses out of it in z within a few.
So the fast state makes fewer jumps than its number of particles, fewer
still at longer lags, and a fit that counts jumps finds it too small a
fraction.  For each *D* the fraction of particles still in focus after
1, 2, ... frames is computed (left in the figure below), and each state's
share of the jumps at each lag is its fraction of the particles times
that.  The fractions reported are then of the particles.

```figure Half the particles slow (0.05 µm²/s), half fast (3 µm²/s), seen in a 700 nm focus. Left: the fraction of particles still in focus after each frame, for several D. Middle: fitted without the axial range, the fast state's share of the jumps falls with lag (dots, the jumps) away from the flat model (line), and the fast fraction comes out far too small. Right: with it, the model follows the jumps' fall and the fractions of the particles (dashed) come back at a half each.
fig.set_size_inches(10.5, 3.4)
left, middle, right = fig.subplots(1, 3, gridspec_kw={"wspace": 0.35})
k = np.arange(1, 11)
for D, colour in ((0.05, "C0"), (0.5, "C2"), (3.0, "C1"), (10.0, "C3")):
    left.plot(k, in_focus([D], 0.01, 700.0, 10)[0], "o-", ms=3, color=colour,
              label=f"{D:g} µm²/s")
left.set_ylim(0, 1.02)
left.set_xlabel("frames (10 ms)")
left.set_ylabel("still in focus")
left.legend(fontsize=7)
t = tracks(thin)
d2, err, lag = jumps_over_lags(t, 7)
from smappy.plugins.tracking_analysis import _seen_per_lag
for ax, focus, name in ((middle, 0.0, "no axial range"),
                        (right, 700.0, "axial range 700 nm")):
    fit = fit_jumps(d2, err, 2, lag * 0.01, 2, focal_range_nm=focus, step_s=0.01)
    own = _seen_per_lag(d2, err, lag, fit, 2, 0.01)
    for i in range(2):
        ax.plot(k[:7], fit.seen[i], color=f"C{i}",
                label=f"{fit.D_um2_per_s[i]:.2g} µm²/s: {100 * fit.fraction[i]:.0f}%")
        ax.plot(k[:7], own[i], "o", ms=3, color=f"C{i}")
        if focus:
            ax.axhline(fit.fraction[i], color=f"C{i}", ls="--", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("lag (frames)")
    if not focus:
        ax.set_ylabel("share of the jumps")
    ax.set_title(name, fontsize=9)
    ax.legend(fontsize=7)
```

**7. The D spectrum.**  Each track is cut into pieces of at most *piece*
jumps.  For every piece and every *D* of a grid of 100, from 0.001 to 100
µm²/s, the probability of its jumps is computed exactly -- the jumps of a
piece are not independent, since each two consecutive ones share a
localization and so its error.  Then the share of the jumps at each *D*
(the *occupation*) is found that makes all pieces together most probable,
with each piece weighed over the *D*s by how well each explains it.  The
populations are the peaks of the occupations; their fractions are the
occupations under each peak.  Each piece's *D* is its average over the
*D*s by that weight.  With an *axial range*, the occupations are
corrected for the focus as the jumps are.

```figure Left: three populations, 30% at 0.02, 30% at 0.3 and 40% at 2 µm²/s (dotted); the D spectrum finds three peaks at their D with their fractions, without being told how many there are. Right: a particle that switches between 0.02 and 2 µm²/s; each piece of 10 jumps of its track gets its own D (dots), which follows the true D (grey) from piece to piece.
from smappy.tracking.spectrum import spectrum, peaks
fig.set_size_inches(9, 3.4)
left, right = fig.subplots(1, 2)
t = tracks(three)
spec = spectrum(t)
found = peaks(spec)
left.plot(spec.D_um2_per_s, spec.occupation, color="C0")
for d, f in found:
    left.annotate(f"{d:.2g} µm²/s, {100 * f:.0f}%",
                  (d, spec.occupation[np.argmin(np.abs(spec.D_um2_per_s - d))]),
                  textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
for truth in (0.02, 0.3, 2.0):
    left.axvline(truth, color="C3", lw=0.6, ls=":")
left.set_xscale("log")
left.set_ylim(0, None)
left.set_xlabel("D (µm²/s)")
left.set_ylabel("occupation")
t = tracks(switching)
spec = spectrum(t)
p = spec.pieces
state = np.asarray(switching["state_true"])[t.rows]
k = next(k for k in range(t.n_tracks) if (p.track == k).sum() >= 8
         and np.count_nonzero(np.diff(state[t.track == k])) >= 2)
mine = np.flatnonzero(p.track == k)
rows = np.flatnonzero(t.track == k)
true_D = np.array([0.02, 2.0])[state[rows]]
right.step(t.step[rows] * 0.01, true_D, where="post", color="0.6", label="truth")
middle = t.step[p.first[mine] + p.n_jumps[mine] // 2] * 0.01
right.plot(middle, spec.D_post_um2_per_s[mine], "o", color="C0", label="each piece")
right.set_yscale("log")
right.set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
right.set_ylim(0.005, 10)
right.set_xlabel("time in the track (s)")
right.set_ylabel("D (µm²/s)")
right.legend(fontsize=7)
```

**8. State switching.**  Each particle is in one of *states* states at
any moment, each with its *D*, and switches from one to another at a
fixed rate -- so many times a second.  The model follows each track
localization by localization: for each state it keeps how likely the
particle is to be in it, and where the particle truly is given the
localizations so far (each is off by its precision).  The *D*s and rates
that make all tracks most likely are the answer.  The fractions follow
from the rates: in the long run, a particle spends in each state the share
of time the rates into and out of it give.  Each localization's state is
then the most likely given the whole track, before and after.

```figure Particles switching between 0.02 and 1 µm²/s, 2.5 times a second each way (a switch every 20 frames), half the time in each. Left: the fitted states, with the rates on the arrows -- 2.5/s each way and half in each come back. Right: one track, the probability that the particle is fast at each localization (blue) against when it truly was (grey).
from smappy.tracking.hmm import fit_hmm, per_localization as hmm_states
from smappy.plugins.tracking_analysis import draw_switching
fig.set_size_inches(9, 3.4)
left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.6]})
t = tracks(hmm_locs)
fit = fit_hmm(t, 2)
draw_switching(left, fit)
prob, state = hmm_states(t, fit.params, len(hmm_locs))
truth = np.asarray(hmm_locs["state_true"])
ids = np.asarray(hmm_locs["track_id_true"])
frames = np.asarray(hmm_locs["frame"])
counts = np.bincount(ids[ids > 0])
k = next(k for k in np.argsort(-counts) if len(set(truth[ids == k])) > 1)
here = np.flatnonzero(ids == k)
here = here[np.argsort(frames[here])]
time_s = (frames[here] - frames[here][0]) * 0.01
right.step(time_s, truth[here], where="post", color="0.6", label="truth: fast")
right.plot(time_s, prob[here, 1], "o-", ms=3, lw=0.8, color="C0",
           label="probability fast")
right.set_ylim(-0.05, 1.05)
right.set_xlabel("time in the track (s)")
right.legend(fontsize=7)
```

## In detail

**The MSD.**  For a track with positions $\mathbf{r}_i$ in frames $f_i$,

$$\mathrm{MSD}(k) = \frac{1}{N_k} \sum_{f_j - f_i = k} |\mathbf{r}_j - \mathbf{r}_i|^2 ,$$

over all $N_k$ pairs $k$ frames apart, overlapping -- the usual estimator.
Pooled, the sums and counts of all tracks are added.  Lags up to 50 frames
are computed, every one over the whole table at once, so 50,000 tracks of a
million localizations take a few seconds.

**The model.**  For free diffusion in $d$ dimensions, with frame time
$\Delta t$ and a localization error $\sigma_a$ per axis,

$$\mathrm{MSD}(k) = 2 d D k \Delta t + 2 \sum_a \sigma_a^2 ,$$

the offset $4\sigma^2$ in 2D.  *from the precisions* takes $\sigma^2$ as
the mean of the track's `xy_err_nm`² (and `z_err_nm`² in 3D).  It is the
static error; a particle that moves during the exposure is blurred, which
makes the true offset smaller (Michalet 2010).  *fitted* fits the
intercept too, which needs no precisions but leaves *D* less certain
([Michalet & Berglund 2012](https://doi.org/10.1103/PhysRevE.85.061916)).

**How many points.**  With the offset known, $D$ is the slope through it,
$\hat{D} = \sum_k k \, (\mathrm{MSD}(k) - o) / (2 d \Delta t \sum_k k^2)$
over $k = 1 \ldots p$; with it fitted, the slope of an ordinary straight
line.  Michalet (2010) showed that the error of $\hat{D}$ has a minimum
in $p$, set by the track's length $N$ and the reduced localization error

$$x = \frac{\sigma^2}{D \Delta t} ,$$

and gave an empirical formula for it.  That formula is not used here:
the optimum is computed from the model itself.  The displacements of a
Brownian track with Gaussian errors are jointly Gaussian, with covariances
$2 D \Delta t$ times the frames two displacements share plus $\sigma^2$
where their ends meet; the MSD points are means of their squares, and for
Gaussian variables $\mathrm{Cov}(a^2, b^2) = 2 \, \mathrm{Cov}(a, b)^2$, so
the full covariance of the MSD points follows exactly.  From it, the
variance of $\hat{D}$ for every $p$, and the $p$ with the least.  (Against
20,000 simulated tracks the covariance agrees to a few percent.)  Since
$x$ needs $D$, Michalet's iteration: $D$ from $p = N/10$, $x$ from $D$, $p$
from $x$, until $p$ repeats, at most 10 rounds.  A track with dark frames
is taken as the gap-free track over its span.  For $x \ll 1$ the optimum
is one point with the offset known (two with it fitted); for $x = 100$ and
50 positions it is about 35.

**α** is the slope of $\log(\mathrm{MSD}(k) - o)$ against $\log k$ over
the same points, or the first two at least, where the MSD is above the
offset; NaN for a track that hardly moves.  Per track it is noisy: read
the median, or the pooled value.

**Jump distances.**  For the jumps $r$ between localizations exactly $L$
frames apart ($L = 1 \ldots$ *lags*, every pair of every track), a state
with coefficient $D_s$ gives per axis a Gaussian of variance

$$s^2 = 2 D_s L \Delta t + \sigma_1^2 + \sigma_2^2 ,$$

the two localizations' errors from their precisions, so $r^2 / s^2$ is
chi-square with $d$ degrees of freedom.  The fractions and coefficients
of the states maximise the likelihood of all jumps, by expectation
maximisation: the fractions are the mean memberships, and each $D_s$
solves its weighted likelihood equation by a fixed point.  At most 200,000
jumps are fitted (a random subset, the same each run).  The BIC
(Bayesian information criterion, $(2S - 1) \ln n - 2 \ln \mathcal{L}$ for
$S$ states and $n$ jumps) compares fits with different *states*: the
lower is better.  The jumps of one track are not independent, so the
likelihood is overconfident, and no errors are given; with several lags
the same displacement enters more than once, which makes it more so but
moves no estimate.

**The focus.**  The focus is the slab $|z| \leq h/2$, $h$ the *axial
range*.  A particle in focus in one frame is equally likely anywhere in
it -- as the particles of a cell are at any moment -- and moves in z by a
Gaussian of variance $2 D \Delta t$ per frame.  The fraction still in
focus in each of the next $k$ frames, $F_s(k)$, is computed frame by
frame: the density in the slab is carried on 200 cells by the probability
of moving from each cell into each other, and whatever lands outside is
dropped.  Looking once a frame is what the camera does; Spot-On instead
uses the continuous absorbing series and widens the slab by an empirical
$a\sqrt{D} + b$, which this does not need (saSPT's choice).  Against
200,000 simulated particles $F$ agrees to 0.1%.  State $s$'s share of the
jumps at lag $L$ is then

$$\pi_s(L) = \frac{f_s F_s(L)}{\sum_q f_q F_q(L)} ,$$

$f_s$ its fraction of the particles.  Bleaching takes every state alike
and cancels.  This couples each $D_s$ to every fraction, so the EM fit
without the focus is the start, and the likelihood is then maximised over
the $D_s$ and $f_s$ (L-BFGS, with the exact gradient but for the slope of
$F$ in $D$, taken by a difference).

Spot-On also uses only each track's first four jumps at each lag; here
every jump is used.  The correction holds for particles anywhere in the
focus, while a track's first localizations are often of a particle that
has just come in at its edge, and will soon leave: on simulated tracks
(like the figure's) the first four jumps gave the fast state 79% for a true
50%, all jumps 51%.  In 3D the fit with an axial range uses the lateral
jumps alone: the particles that stay in focus are those that moved less
in z, so their z jumps are too short (a 10% lower *D* on simulated
tracks), while x and y do not care.  A particle that comes back into
focus is taken as a new one; a linker with a *memory* may join it to its
old track across the gap, which the model does not count.

**The D spectrum.**  Each track's jumps between consecutive localizations
are cut, from its start, into pieces of at most *piece* jumps, the last
shorter.  Along a piece, jump $j$ over $k_j$ frames has per axis the
variance $2 D k_j \Delta t + s_j^2 + s_{j+1}^2$ ($s_j^2$ localization $j$'s
error variance, from its precision), and jumps $j-1$ and $j$ share
localization $j$, so they covary by $-s_j^2$.  The jumps of a piece are
therefore jointly Gaussian with a tridiagonal covariance, whose
determinant and quadratic form one pass along the piece gives (its LDL
factorisation); the log-likelihood $\ell_{pg}$ of piece $p$ at grid point
$g$ is the sum over the axes.  saSPT puts the localization error on a
grid as well, one value for all; here each localization's own precision
is used, which drops that grid.

The occupations follow by variational Bayes, saSPT's scheme: a Dirichlet
over them with one pseudo-count per grid point, and, round by round,

$$R_{pg} \propto e^{\ell_{pg}} \, e^{\psi(\alpha_g) - \psi(\sum_h \alpha_h)} , \qquad \alpha_g = 1 + \sum_p n_p R_{pg} ,$$

with $n_p$ the piece's number of jumps and $\psi$ the digamma function,
until no occupation $\alpha_g / \sum_h \alpha_h$ moves by more than
$10^{-7}$ (at most 5000 rounds; saSPT stops after 200, when single grid
points can still be several percent from where they settle, though the
fraction under a peak no longer moves).  The occupations are shares of
the jumps.  With an *axial range* they are divided by each *D*'s fraction
still in focus after one frame, $F(1)$ above, and renormalised: a piece's
jumps are mostly between consecutive frames.  Each piece's *D* is
$\exp \sum_g R_{pg} \ln D_g$, the posterior mean of $\ln D$.  The
peaks are the local maxima of the occupations, each with the grid points
down to the minima on either side; a peak with less than 2% is not
reported.

Against simulated tracks with three populations it gives back their *D*
within 20% and their fractions within 0.04 (with 2,600 pieces), and a
piece's *D* tells a slow from a fast state in 97% of pieces or more.
Unlike the jumps' states, it needs no number of states and its
likelihood is right for the correlated jumps.  Close populations are
where it is weakest: on simulated tracks two halves a factor of two apart
in *D* still gave two peaks, but at 62% and 38%; a factor of three gave
55% and 45%.  Short pieces spread every peak out.

**State switching.**  The state is a Markov chain with a rate $k_{ab}$
(per second) from state $a$ to $b$; over $k$ frames it moves by
$\exp(Q k \Delta t)$, $Q$ the rate matrix.  A particle's true position
moves from localization $i$ to the next, $k_i$ frames later, by a
Gaussian of variance $2 D_a k_i \Delta t$ per axis, $a$ the state at $i$:
the state holds over the interval.  What is measured is the true position
plus the localization's error, of variance $s_i^2$ per axis.

The likelihood of a track integrates out the true positions.  For each
state of the current localization the filter keeps its log-probability
and a Gaussian of the true position, and goes on localization by
localization: from each state $a$, a Kalman step -- the predicted
position's variance grows by $2 D_a k \Delta t$, the next localization
updates it, and how well it was predicted is the step's likelihood --
then into each state $b$ by the transition matrix.  The exact belief is a
mixture with a Gaussian per past sequence of states, which grows without
bound; the Gaussians that end in the same state are merged into one with
their mean and variance at every step.  That is ExTrack's merging with the
shortest memory (its *frame_len* of 1).  With one state nothing is merged
and the filter is exact: it agrees with the D spectrum's Gaussian of a
whole track to $10^{-9}$.

The fit maximises the summed log-likelihood over $\ln D$, $\ln k_{ab}$
and the probabilities of the state a track starts in (L-BFGS, the
gradient by differences), from the jump fit's *D*s and a switch every 20
frames.  The starting probabilities are free rather than the chain's
long-run ones, because how tracks start -- a fast particle enters a thin
focus more often -- need not follow the rates.  The fractions reported
are the long-run ones, $\pi Q = 0$; a state's dwell time is
$1 / \sum_{b \neq a} k_{ab}$.  At most 50,000 jumps are fitted, a random
subset of whole tracks beyond (the same each run); two states take a few
seconds for 10,000 jumps, three about four times as long.

Each localization's state is the posterior of the state over the jump it
starts (the last of a track: the jump it ends), by the forward-backward
algorithm with the fitted parameters.  There the jumps are taken as
independent given the states, each with the error variance
$s_i^2 + s_{i+1}^2$, the standard hidden Markov model: the merged filter
has no exact way back.  On the simulations here it gets 95% (three
states) to 98% (two) of localizations right.

Against simulated tracks (two states, 0.02 and 1 µm²/s, 2.5/s each way)
the rates come back within 15%, at three states (six rates) within 30%,
the fractions within 0.04.  Two things are left out.  A jump across dark
frames is taken in the state it started in throughout.  And a particle
that leaves a thin focus ends its track: fast particles leave sooner, so
the tracks seen for long are those that went slow, which the likelihood
does not count.  Six simulations of each (0.05 and 2 µm²/s, 2.5/s each
way): without a focus the rates came back at 2.49 ± 0.16 and
2.66 ± 0.19/s and the fast fraction at 0.48; in a 700 nm focus at 1.99
and 3.77 ± 0.63/s and 0.35.  A factor for staying in focus at each step
made it no better -- whether a particle stays depends on its z, which the
model does not follow -- so it is not applied, and with an *axial range*
the run warns: take the fractions from the jump distances or the D
spectrum there, and read the rates with care.

**MINFLUX.**  A table with `time_s` has no regular frames; its
localizations are put on a grid of the median interval between
consecutive localizations of a track, each at the nearest step.

## Parameters

### id_column
*auto* takes the first of `track_id`, `tid` and `track_id_true` the table
has, and the list says which.  Choose another integer attribute to compare
two linkings (Link tracks' *output attribute*), or `group_id` to treat each
merged blink as a track.  An id of -1 is no track.

### msd_fit.offset
*from the precisions* when the fitter's precisions are trustworthy (the
default); *fitted* when they are not, or when motion blur is large -- a
fast particle over a long exposure.

### msd_fit.lags
Leave on *auto*.  A fixed number is what most papers use and what Michalet
showed to be often far from the best: too few points for slow particles
with large errors, too many for fast ones.

### msd_fit.min_length
In 2D, a track of 5 localizations gives a *D* uncertain by about 55%
even with a small localization error, one of 20 by about 25%: for
per-track values to mean something, 20 or more.  The pooled MSD uses every
track.

### jump_fit.states
Fit 1, 2 and 3 and compare the BIC: a state more that does not lower it by
at least about 10 (a common rule of thumb) is not supported by the data.  Two states fitted to one
population come out as two close values of *D*.

### jump_fit.lags
Several lags pin *D* and the fractions down better than one: longer lags
spread the states apart against the localization error, which helps for
slow states, and with an *axial range* the fall of the fast state's share
with lag is itself information.  Spot-On uses about 7.  Too many, and
particles that change state within the lag blur the states together, and
few fast particles are left at the longest lags.

### axial_range_nm
The depth around the focal plane in which a particle is detected, not
the PSF's width: about 700 nm for 2D tracking with HILO illumination is a
common value; measure it as the z range over which beads are still
detected and fitted.  0 when nothing leaves -- 3D tracking over a deep
range, MINFLUX, or particles on a membrane.  A value too large
undercorrects (the fast fraction too small), one too small overcorrects.
It applies to the jump distances and the D spectrum, not to the MSD or
the state switching (see *In detail*).

### switching_fit.states
Compare the BIC of 1, 2 and 3 as for the jump distances.  Three states
have six rates, which need many switches to pin down: tracks long against
the dwell times, or many of them.  A state the data do not support comes
out with a *D* close to another's or with rates at their bounds.

### spectrum_fit.split
saSPT's 10.  Shorter pieces follow a particle that changes state more
closely, but each says less about its *D*, which broadens the spectrum's
peaks; longer pieces sharpen the peaks but blur a change of state within
a piece.

### spectrum_fit.d_min
The grid has 100 points, log-spaced.  Keep it a decade beyond the slowest
and fastest populations expected: occupation piled up at either end says
the grid is too short.  A *D* below what the localization error can tell
from zero, roughly $\sigma^2 / \Delta t$, is immobile.

## Output

A line per analysis: the pooled *D* and α with the lags used and the
offset, the median *D* and α per track; the states' *D* and fractions
with the BIC, and with an *axial range* also each state's share of the
jumps at one frame, before the correction.  A figure: the pooled MSD with
the line fitted, the histogram of *D* per track, the jumps at one frame
with the fitted states, and, with several *lags*, each state's share of
the jumps per lag -- the model's (line), the jumps' own (dots, the mean
of their memberships) and, with an *axial range*, its fraction of the
particles (dashed).  Per region, the values (`msd_D_um2_per_s`,
`msd_alpha`, `track_D_median_um2_per_s`, `jump_D1_um2_per_s`,
`jump_fraction1`, ..., `jump_seen1`, ..., `spectrum_D1_um2_per_s`,
`spectrum_fraction1`, ..., `n_pieces`), the pooled MSD curve (`lag_s`,
`msd_nm2`) and the spectrum (`spectrum_D_um2_per_s`,
`spectrum_occupation`).  The D spectrum adds a line with its peaks and
their fractions and a panel of the occupations (with an *axial range*,
of the particles, and of the jumps dashed); *write per localization*
writes `D_post_um2_per_s`, each piece's *D* on its localizations (NaN on
a track of one localization).  State switching adds a line with each
state's *D*, fraction and dwell time and the rates, the values
`hmm_D1_um2_per_s`, `hmm_fraction1`, `hmm_dwell1_s`, `hmm_rate12_per_s`
(from state 1 to 2), ..., `hmm_bic`, and a panel of the states with the
rates on arrows; *write per localization* writes `hmm_state` (0 the
slowest state, -1 none) and `hmm_p1`, `hmm_p2`, ... (each state's
probability), to colour the tracks by state or to measure where the
particles bind.

Run on the view, the fit of every track is also kept as a
[region table](plugin:Panels/Glossary) of the track id ("per track_id:
Tracking analysis"): `n_locs`, `D_um2_per_s`, `alpha`, `lags` and
`reduced_error` (the localization error over D times the step), NaN for a
track shorter than *minimum length*.  *Summarize/Histograms* shows their
distributions, and the [region filter](plugin:Analysis/Regions/Filter regions)
keeps the tracks of one population -- `alpha` below 0.7, say -- without a
attribute per value; *write per track* is for colouring the tracks.  Not when
an ROI is drawn, since the table stands for every track the layer shows.

A good result: the MSD a straight line over its first points, α near 1
for free diffusion, the states' curves following the jumps' histogram,
and the dots of the share per lag on their lines.  Without an *axial
range*, dots of the fast state that fall with lag below its flat line say
that particles leave the focus: set one.  A
pooled MSD that bends down is confinement or a mixture of slow and fast
particles; one that bends up, drift or directed motion.  Per-track *D*
of short tracks is broad even for one population: compare populations by
the jumps' states, not by the width of that histogram.

## Differences from SMAP

Based on SMAP's *AnalyzeSPT*
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), rewritten.  The
main changes:

* SMAP fitted the MSD at lags 2 to 4, always, with *D* the slope over 4
  and the frame time hard-coded at 30 ms.  Here the lags are the optimum
  per track, the offset is the localization error's, *D* is divided by
  $2d$ (3D as well), and the frame time is the file's.
* SMAP fitted a lognormal to the histogram of *D* per track.  Here the
  jumps are fitted with diffusive states, at several lags and corrected
  for the focus, which separates populations that the per-track *D* of
  short tracks blurs together.
* SMAP's diffusion maps and its interactive track view are not ported;
  the tracks are coloured by `track_D_um2_per_s`, or piece by piece by
  `D_post_um2_per_s`, instead.

## References

* Michalet X. Mean square displacement analysis of single-particle
  trajectories with localization error: Brownian motion in an isotropic
  medium. *Phys Rev E* 82, 041914 (2010).
  [doi:10.1103/PhysRevE.82.041914](https://doi.org/10.1103/PhysRevE.82.041914)
  -- the optimal number of MSD points.
* Michalet X, Berglund AJ. Optimal diffusion coefficient estimation in
  single-particle tracking. *Phys Rev E* 85, 061916 (2012).
  [doi:10.1103/PhysRevE.85.061916](https://doi.org/10.1103/PhysRevE.85.061916)
  -- a known localization error makes D more precise.
* Hansen AS, Woringer M, Grimm JB, Lavis LD, Tjian R, Darzacq X. Robust
  model-based analysis of single-particle tracking experiments with
  Spot-On. *eLife* 7, e33125 (2018).
  [doi:10.7554/eLife.33125](https://doi.org/10.7554/eLife.33125)
  -- the jumps at several lags, corrected for particles leaving the focus.
* Heckert A, Dahal L, Tjian R, Darzacq X. Recovering mixtures of
  fast-diffusing states from short single-particle trajectories. *eLife*
  11, e70169 (2022).
  [doi:10.7554/eLife.70169](https://doi.org/10.7554/eLife.70169)
  -- the loss from the focus computed frame by frame (saSPT).
* Simon F, Tinevez J-Y, van Teeffelen S. ExTrack characterizes
  transition kinetics and diffusion in noisy single-particle tracks.
  *J Cell Biol* 222, e202208059 (2023).
  [doi:10.1083/jcb.202208059](https://doi.org/10.1083/jcb.202208059)
  -- the hidden Markov model with the localization error integrated out.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
