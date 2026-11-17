"""Diffusive states that switch: a hidden Markov model of the tracks, its
rates of switching, and the state of every localization -- ExTrack's model
(Simon et al. 2023), re-implemented from its paper rather than ported.

**Why.**  The jump fit and the D spectrum (`analysis.fit_jumps`,
`spectrum`) find how many particles are in each state, but not how fast
they change state: a static mixture cannot give binding and unbinding
rates, which are what most single-particle tracking of a transcription
factor or a receptor is after.  Here the state is a Markov chain in time,
switching from a to b at the rate k_ab (per second), and the fit gives
the rates, the D of each state and so the fraction in each (the chain's
stationary distribution).

**The model.**  A particle's true position r_i at localization i moves to
r_{i+1} by a Gaussian of variance 2 D_a k dt per axis over the k frames
between them, a the state at i (the state holds over the interval: the
simulator's convention, `simulate_tracks`); the state then moves on by the
k-frame transition matrix exp(Q k dt).  What is measured is x_i = r_i plus
the localization's error, of variance s_i^2 per axis from its precision.
The true positions are integrated out (the error of a localization is
shared by the two jumps it ends and starts, which the jump fit ignores):

* **the filter** (`track_loglik`) carries, for each state of the current
  localization, its probability so far and a Gaussian of the true
  position given it, through a Kalman step per localization.  The exact
  belief is a mixture over every past sequence of states; the Gaussians of
  the sequences that end in the same state are merged into one by their
  mean and variance each step -- ExTrack's merging with the shortest
  memory (its *frame_len* of 1; the interacting-multiple-model filters of
  tracking radar do the same).  On simulated tracks it gives back D and the
  rates within the scatter of the seeds (tests);
* **the fit** (`fit_hmm`) maximises the summed log-likelihood over log D,
  log rates and the initial state probabilities (L-BFGS, the gradient by
  differences: 5 parameters for 2 states, 11 for 3), started from the
  jump fit's D and fractions and a switch every 20 frames.  The initial
  probabilities are free, not the chain's stationary ones, because which
  state a track starts in depends on how tracks start (a fast particle
  enters a thin focus more often); the fractions reported are the
  stationary distribution of the rates, which only the switching inside
  tracks determines;
* **each localization's state** (`state_probabilities`) is the posterior
  of the state over the jump it starts (the last localization of a track:
  the jump it ends), by the forward-backward algorithm on the jumps with
  the fitted parameters -- here each jump's error variance is
  s_i^2 + s_{i+1}^2 and the jumps are independent given the states, the
  standard HMM, because the merged filter has no exact backward pass.  It
  is a classifier on parameters fitted with the right likelihood.

What it leaves out, and says so on the page: a jump across dark frames is
taken at the state it started in throughout; and a particle that leaves a
thin focus ends its track.  Fast particles leave sooner, so the tracks
that are seen long are those that went slow, which the likelihood does
not count.  Six simulations of each (0.05 and 2 um^2/s, half each, a
switch every 20 frames, so 2.5 /s each way): without a focus the rates
came back at 2.49 +- 0.16 and 2.66 +- 0.19 /s and the fast fraction at
0.48; in a 700 nm focus at 1.99 and 3.77 +- 0.63 /s and 0.35.  A factor
per step for staying in focus (`analysis.in_focus` at one frame) was
tried and made it no better -- whether a particle stays depends on its
z, which the filter does not follow -- and lowered the fast D by 12%.
ExTrack's loss probability is the same for every state and has the same
blind spot.  In a thin focus the fractions should come from the jump fit
or the D spectrum with the axial range, and the rates read with care:
here they came out low from slow to fast and high back.

**Crowding.**  A wrong link that carries a fast track onto a slow
neighbour, which then goes on as that particle, is in the data a fast
particle that slowed down: a switch, which no model of the track can tell
from a real one.  So the rates grow with the density.  Two states 50
times apart in D (`studies/tracking/two_states.py`, the survey's item 5)
came out 10--20% high where the fast state's rms jump in a frame was
0.13--0.18 of the mean distance between particles, 60% high at 0.28 and
twice as high at 0.4; with no switching at all, rates of 0.3, 1.3 and
2.7 /s were found.  An outlier term (a link to anywhere, with a fitted
probability) took a sixth of the wrong links and part of the bias, slowed
the fit two to three times and pulled the fast D down, and was left out.
`crowding` measures that ratio from the data, and above `CROWDED` the
plugin warns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .analysis import Tracks

# the most jumps fitted: a random subset of tracks beyond (the same one each run)
MAX_JUMPS = 50_000
FLOOR_D = 1e-6                     # um^2/s
# the fast state's rms jump over the spacing beyond which the rates come
# out 10--20% high and more (the docstring's *Crowding*)
CROWDED = 0.15


def nearest_spacing_nm(x, y, frame) -> float:
    """The mean distance from each localization to the nearest one in
    its frame (nm), NaN if no frame has two: for points at random,
    ``1 / (2 sqrt(density))``, `simulate.spacing_nm`.  One tree over the
    positions with the frames set far apart along a third axis."""
    from scipy.spatial import cKDTree
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = np.asarray(frame, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
    if ok.sum() < 2:
        return float("nan")
    far = 10.0 * (np.ptp(x[ok]) + np.ptp(y[ok]) + 1.0)
    pts = np.column_stack([x[ok], y[ok], f[ok] * far])
    d, _ = cKDTree(pts).query(pts, k=2)
    near = d[:, 1][d[:, 1] < far / 2]                 # a neighbour in the same frame
    return float(near.mean()) if len(near) else float("nan")


def crowding(params: "HMMParams", dt_s: float, spacing_nm: float) -> float:
    """The fastest state's lateral rms jump in one step, ``sqrt(4 D dt)``,
    over the spacing between particles (`nearest_spacing_nm`)."""
    D = float(np.max(params.D_um2_per_s)) * 1e6
    return float(np.sqrt(4.0 * D * dt_s) / spacing_nm) if spacing_nm > 0 else float("nan")


@dataclass
class Steps:
    """The tracks of two or more localizations, longest first, for a pass
    in lockstep: localization j of track t is row ``start[t] + j`` of
    ``x`` (positions, nm) and ``err`` (variance per axis, nm^2); ``lag``
    (frames) is per row the gap to the next localization of its track."""
    x: np.ndarray
    err: np.ndarray
    lag: np.ndarray
    start: np.ndarray
    length: np.ndarray
    track: np.ndarray                 # the `Tracks` index of each, longest first
    rows: np.ndarray                  # each row's row in the `Tracks`
    dt_s: float

    @property
    def n_jumps(self) -> int:
        return int((self.length - 1).sum())


def steps_of(t: Tracks, max_jumps: Optional[int] = None) -> Steps:
    """The tracks of ``t`` with a jump, longest first; with ``max_jumps``,
    a random subset of tracks with about that many jumps (the same each
    run)."""
    counts = t.counts()
    keep = np.flatnonzero(counts >= 2)
    if max_jumps and (counts[keep] - 1).sum() > max_jumps:
        order = np.random.default_rng(0).permutation(keep)
        take = np.cumsum(counts[order] - 1) <= max_jumps
        keep = np.sort(order[take])
    order = keep[np.argsort(-counts[keep], kind="stable")]
    first_row = np.concatenate([[0], np.cumsum(counts)[:-1]])
    length = counts[order]
    rows = np.concatenate([first_row[k] + np.arange(n) for k, n in zip(order, length)]) \
        if len(order) else np.zeros(0, np.int64)
    start = np.concatenate([[0], np.cumsum(length)[:-1]]) if len(order) else \
        np.zeros(0, np.int64)
    step = t.step[rows]
    lag = np.zeros(len(rows), np.int64)
    if len(rows) > 1:
        lag[:-1] = np.diff(step)
    return Steps(t.xyz[rows], t.axis_errors()[rows], lag, start, length, order, rows,
                 t.dt_s)


@dataclass
class HMMParams:
    """D per state (um^2/s, slowest first), the rate matrix Q (per second;
    the rows sum to 0) and the probabilities of the state a track starts
    in."""
    D_um2_per_s: np.ndarray
    Q_per_s: np.ndarray
    initial: np.ndarray

    @property
    def states(self) -> int:
        return len(self.D_um2_per_s)

    def stationary(self) -> np.ndarray:
        """The fraction of the time in each state: pi Q = 0."""
        S = self.states
        if S == 1:
            return np.ones(1)
        A = np.vstack([self.Q_per_s.T, np.ones(S)])
        b = np.zeros(S + 1)
        b[-1] = 1.0
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        pi = np.maximum(pi, 0)
        return pi / pi.sum()

    def dwell_s(self) -> np.ndarray:
        """The mean time a state lasts once entered."""
        out = -np.diag(self.Q_per_s)
        with np.errstate(divide="ignore"):
            return np.where(out > 0, 1.0 / out, np.inf)

    def transitions(self, lags: np.ndarray, dt_s: float) -> np.ndarray:
        """The transition matrices over each of ``lags`` frames (a stack)."""
        from scipy.linalg import expm
        return np.stack([expm(self.Q_per_s * k * dt_s) for k in lags])


def _transition_logs(params: HMMParams, lag: np.ndarray, dt_s: float):
    """log T over each row's lag, as an index into a stack and the stack."""
    lags, index = np.unique(np.maximum(lag, 1), return_inverse=True)
    T = params.transitions(lags, dt_s)
    return index, np.log(np.maximum(T, 1e-300))


def track_loglik(s: Steps, params: HMMParams) -> np.ndarray:
    """Each track's log-likelihood (of its positions after the first,
    given the first), by the merged filter: per state of the current
    localization, its log-probability and the Gaussian of the true
    position (mean and variance per axis)."""
    from scipy.special import logsumexp
    S = params.states
    D = np.maximum(params.D_um2_per_s, FLOOR_D) * 1e6                 # nm^2/s
    T = len(s.length)
    if not T:
        return np.zeros(0)
    index, logT = _transition_logs(params, s.lag, s.dt_s)
    logw = np.broadcast_to(np.log(np.maximum(params.initial, 1e-300)), (T, S)).copy()
    mu = np.repeat(s.x[s.start][:, None, :], S, axis=1)                 # T x S x d
    P = np.repeat(s.err[s.start][:, None, :], S, axis=1)
    out = np.zeros(T)
    for j in range(1, int(s.length.max())):
        A = int(np.count_nonzero(s.length > j))                         # longest first
        at = s.start[:A] + j
        lag = s.lag[at - 1]
        q = 2.0 * D[None, :, None] * (lag * s.dt_s)[:, None, None]       # A x S x 1
        pred = P[:A] + q
        var = pred + s.err[at][:, None, :]
        resid = s.x[at][:, None, :] - mu[:A]
        L = -0.5 * (resid * resid / var + np.log(2 * np.pi * var)).sum(axis=2)   # A x S
        gain = pred / var
        mu_post = mu[:A] + gain * resid
        P_post = pred * s.err[at][:, None, :] / var
        # from state a (columns) into state b: a's jump, then T over the lag
        joint = (logw[:A] + L)[:, :, None] + logT[index[at - 1]]           # A x a x b
        new = logsumexp(joint, axis=1)                                    # A x b
        weight = np.exp(joint - new[:, None, :])                          # over a
        mean = np.einsum("tab,tad->tbd", weight, mu_post)
        spread = np.einsum("tab,tabd->tbd", weight,
                           P_post[:, :, None, :]
                           + (mu_post[:, :, None, :] - mean[:, None, :, :]) ** 2)
        logw[:A], mu[:A], P[:A] = new, mean, spread
        ends = s.length[:A] == j + 1
        out[:A][ends] = logsumexp(new[ends], axis=1)
    return out


def _jump_emissions(s: Steps, params: HMMParams) -> np.ndarray:
    """Per row with a next localization, the log-density of its jump in
    each state, the jumps independent (`state_probabilities`): rows x S."""
    D = np.maximum(params.D_um2_per_s, FLOOR_D) * 1e6
    x = np.zeros(s.x.shape)
    x[:-1] = s.x[1:] - s.x[:-1]
    e = np.zeros(s.err.shape)
    e[:-1] = s.err[1:] + s.err[:-1]
    var = 2.0 * D[None, :, None] * (s.lag * s.dt_s)[:, None, None] + e[:, None, :]
    var = np.where(s.lag[:, None, None] > 0, var, 1.0)        # a track's last row: unused
    return -0.5 * (x[:, None, :] ** 2 / var + np.log(2 * np.pi * var)).sum(axis=2)


def state_probabilities(s: Steps, params: HMMParams) -> np.ndarray:
    """Each localization's posterior over the states (rows x S, in the
    order of ``s``): of the jump it starts, the last of a track of the jump
    it ends; by the scaled forward-backward algorithm on the jumps."""
    from scipy.special import logsumexp
    S = params.states
    n = len(s.x)
    out = np.full((n, S), np.nan)
    if not len(s.length):
        return out
    em = _jump_emissions(s, params)
    index, logT = _transition_logs(params, s.lag, s.dt_s)
    alpha = np.zeros((n, S))
    T = len(s.length)
    a = np.broadcast_to(np.log(np.maximum(params.initial, 1e-300)), (T, S)).copy()
    jumps = s.length - 1
    for j in range(int(jumps.max())):
        A = int(np.count_nonzero(jumps > j))
        at = s.start[:A] + j
        if j:
            prev = alpha[at - 1] + em[at - 1]
            a = logsumexp(prev[:, :, None] + logT[index[at - 1]], axis=1)
        else:
            a = a[:A]
        alpha[at] = a - logsumexp(a, axis=1)[:, None]
    beta = np.zeros((n, S))
    for j in range(int(jumps.max()) - 1, -1, -1):
        A = int(np.count_nonzero(jumps > j))
        at = s.start[:A] + j
        last = jumps[:A] == j + 1
        nxt = at + 1
        b = logsumexp(logT[index[at]] + (em[nxt] + beta[nxt])[:, None, :], axis=2)
        b[last] = 0.0
        beta[at] = b - logsumexp(b, axis=1)[:, None]
    has = np.zeros(n, bool)
    for j in range(int(jumps.max())):
        A = int(np.count_nonzero(jumps > j))
        has[s.start[:A] + j] = True
    post = alpha + em + beta
    post = np.exp(post - logsumexp(post, axis=1)[:, None])
    out[has] = post[has]
    ends = s.start + s.length - 1
    out[ends] = out[ends - 1]
    return out


@dataclass
class HMMFit:
    """The fitted model, its log-likelihood and BIC over the jumps fitted."""
    params: HMMParams
    loglik: float
    bic: float
    n_jumps: int
    n_tracks: int

    @property
    def fractions(self) -> np.ndarray:
        return self.params.stationary()


def _pack(p: HMMParams) -> np.ndarray:
    S = p.states
    off = ~np.eye(S, dtype=bool)
    init = np.maximum(p.initial, 1e-9)
    return np.concatenate([np.log(np.maximum(p.D_um2_per_s, FLOOR_D)),
                           np.log(np.maximum(p.Q_per_s[off], 1e-9)),
                           np.log(init[1:] / init[0])])


def _unpack(theta: np.ndarray, S: int) -> HMMParams:
    D = np.exp(theta[:S])
    off = ~np.eye(S, dtype=bool)
    Q = np.zeros((S, S))
    Q[off] = np.exp(theta[S:S + S * (S - 1)])
    Q[np.diag_indices(S)] = -Q.sum(axis=1)
    a = np.concatenate([[0.0], theta[S + S * (S - 1):]])
    init = np.exp(a - a.max())
    return HMMParams(D, Q, init / init.sum())


def fit_hmm(t: Tracks, states: int = 2, start: Optional[HMMParams] = None,
            max_jumps: int = MAX_JUMPS) -> HMMFit:
    """D, rates and initial probabilities of ``states`` switching states,
    by maximum likelihood (`track_loglik`), started from ``start`` or the
    jump fit at one frame with a switch every 20 frames."""
    from scipy.optimize import minimize
    from .analysis import fit_jumps
    S = int(states)
    s = steps_of(t, max_jumps)
    if s.n_jumps < 4 * S:
        raise ValueError(f"{s.n_jumps} jumps: too few for {S} state{'s' if S > 1 else ''}")
    if start is None:
        # every jump between consecutive localizations, at its own lag
        inner = np.ones(len(s.x), bool)
        inner[s.start + s.length - 1] = False
        a = np.flatnonzero(inner)
        d2 = ((s.x[a + 1] - s.x[a]) ** 2).sum(axis=1)
        err = (s.err[a + 1] + s.err[a]).mean(axis=1)
        guess = fit_jumps(d2, err, t.dims, s.lag[a] * t.dt_s, S)
        rate = 0.05 / t.dt_s
        Q = np.full((S, S), rate / max(S - 1, 1))
        np.fill_diagonal(Q, 0.0)
        np.fill_diagonal(Q, -Q.sum(axis=1))
        start = HMMParams(np.maximum(guess.D_um2_per_s, 1e-4), Q, guess.fraction)
    theta0 = _pack(start)
    rate_lo, rate_hi = np.log(1e-3 / (t.dt_s * s.length.max())), np.log(5.0 / t.dt_s)
    bounds = ([(np.log(FLOOR_D), np.log(1e4))] * S
              + [(rate_lo, rate_hi)] * (S * (S - 1)) + [(-20.0, 20.0)] * (S - 1))
    theta0 = np.clip(theta0, [b[0] for b in bounds], [b[1] for b in bounds])
    scale = float(s.n_jumps)

    def negative(theta):
        return -float(track_loglik(s, _unpack(theta, S)).sum()) / scale

    result = minimize(negative, theta0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 300})
    p = _unpack(result.x, S)
    order = np.argsort(p.D_um2_per_s)
    p = HMMParams(p.D_um2_per_s[order], p.Q_per_s[np.ix_(order, order)], p.initial[order])
    loglik = -result.fun * scale
    k = S + S * (S - 1) + (S - 1)
    return HMMFit(p, loglik, k * np.log(s.n_jumps) - 2 * loglik, s.n_jumps, len(s.length))


def per_localization(t: Tracks, params: HMMParams, n_rows: int):
    """Each localization's posterior over the states and its likeliest
    state, on the rows of the table (``n_rows``): NaN and -1 where it is in
    no track of two localizations or more."""
    s = steps_of(t)
    post = state_probabilities(s, params)
    prob = np.full((n_rows, params.states), np.nan)
    state = np.full(n_rows, -1, np.int64)
    rows = t.rows[s.rows]
    prob[rows] = post
    have = np.isfinite(post).all(axis=1)
    state[rows[have]] = np.argmax(post[have], axis=1)
    return prob, state
