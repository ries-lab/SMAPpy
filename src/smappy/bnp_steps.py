"""BNP-Step: Bayesian non-parametric step finding, every trace at once.

BNP-Step (Rojewski, Schweiger, Sgouralis, Comstock & Pressé, Biophys J 2024,
doi:10.1016/j.bpj.2024.01.008) models a trace as a level F plus up to M
possible steps, each switched on or off by a load b_m, with a height h_m and
a time t_m:

    x_n = F + sum_m b_m h_m [t_n <= t_m] + noise,   noise ~ N(0, 1/eta)

The loads have Bernoulli(gamma / M) priors -- the weak limit of a beta-
Bernoulli process, so the number of steps is learned, not chosen -- the
heights and F Normal priors, the times a uniform prior over the data's
times and eta a Gamma prior.  A Gibbs sampler draws each load given the
rest, F and the heights jointly from their Gaussian conditional, each time
by a Metropolis step proposed from the prior, and eta from its Gamma
conditional; the answer is the sample of highest posterior (MAP).  A step
adds to the points *before* its time, so F is the trace's last level.

Ported from the authors' code (LabPresse/bnp-step, c8c6621; MIT licence
pending at the time of the port), the same model and the same conditional
draws, with these departures:

* **No points x loads matrices.**  The original forms N x M Heaviside
  matrices for every update.  Every quantity the sampler needs is a sum
  over prefixes of the trace, so it is computed from the cumulative sum of
  the data at the M step positions: the change of the squared error when a
  load is switched or a time moved, the normal equations of F and the
  heights, the squared error for eta.  An update costs O(M), not O(N M).
* **Every trace at once.**  The traces of a table are sampled side by side
  as arrays of (traces, loads); a trace is independent of the others, so
  this is the original sampler run on each, in one loop.
* **Started from AutoStepFinder** (`steps.autostepfinder`) rather than from
  the prior, so a few hundred sweeps reach the posterior's mode where the
  original runs 50 000 from a random start.  ``start="prior"`` starts as
  the original does.
* **Priors in the data's units.**  The original's defaults (heights and F
  of standard deviation 60, noise precision about 10) are for its force
  traces; here the heights' prior is ``height_sd`` (by default three times
  the noise estimated from consecutive differences, which a step must
  stand out of), F's is centred on the trace's end, and eta's is centred
  on the noise estimate.  The weak limit M is set per batch from the start.
* **Systematic scan**: loads, times, F and heights, eta in that order each
  sweep (the original draws the four blocks in random order); both leave
  the posterior unchanged.  A tempered draw of F and the heights scales
  their covariance by the temperature and not their mean, which the
  original also scales.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class BNPResult:
    """What `bnp_steps` found for each trace: the boundaries (sorted plateau
    starts, as `steps.autostepfinder` gives them) of the MAP sample, the
    number of steps in it and its mean over the sweeps' second half, and
    each trace's MAP log posterior."""
    boundaries: List[np.ndarray]
    n_steps: np.ndarray
    n_steps_mean: np.ndarray
    log_posterior: np.ndarray
    noise_sd: np.ndarray


def noise_sd(x) -> float:
    """The noise of a trace from its consecutive differences, robustly: a
    step moves one difference, the noise all of them."""
    d = np.diff(np.asarray(x, np.float64))
    if len(d) < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


class _Batch:
    """The traces padded to one length, their prefix sums and the sampler's
    state, all (traces, ...) arrays."""

    def __init__(self, traces: Sequence[np.ndarray]):
        self.n = np.array([len(x) for x in traces], np.int64)
        T, N = len(traces), int(self.n.max())
        self.cum = np.zeros((T, N))
        self.sumsq = np.zeros(T)
        self.offset = np.zeros(T)
        for i, x in enumerate(traces):
            x = np.asarray(x, np.float64)
            self.offset[i] = x.mean()            # centred: the priors are about 0
            xc = x - self.offset[i]
            c = np.cumsum(xc)
            self.cum[i, :len(c)] = c
            self.cum[i, len(c):] = c[-1]
            self.sumsq[i] = float(xc @ xc)
        self.total = self.cum[np.arange(T), self.n - 1]

    def prefix(self, k):
        """S_x[k], the sum of the first k + 1 points, for (T, ...) indices."""
        rows = np.arange(len(self.n)).reshape((-1,) + (1,) * (np.ndim(k) - 1))
        return self.cum[rows, k]


def _ss(batch: _Batch, F, a, k):
    """Each trace's squared error, F a (T,), a = b h and k (T, M)."""
    P = np.minimum(k[:, :, None], k[:, None, :]) + 1.0
    cross = F * batch.total + np.sum(a * batch.prefix(k), axis=1)
    square = F * F * batch.n + 2 * F * np.sum(a * (k + 1), axis=1) + np.einsum(
        "ti,tij,tj->t", a, P, a)
    return batch.sumsq - 2 * cross + square


def bnp_steps(traces: Sequence[np.ndarray], sweeps: int = 400,
              height_sd: Optional[float] = None, gamma: float = 1.0,
              weak_limit: Optional[int] = None, start: str = "autostepfinder",
              anneal: bool = False, seed: Optional[int] = 0) -> BNPResult:
    """BNP-Step on each trace of ``traces`` (1-D arrays of positions, in
    time order).  ``height_sd``: the prior's standard deviation of a step
    (None: three times each trace's noise).  ``weak_limit``: the most steps
    a trace may have (None: twice the most AutoStepFinder finds, at least
    20).  ``anneal``: the original's simulated annealing, from a
    temperature of 10 down to 1 over the first half of the sweeps."""
    from .steps import autostepfinder
    rng = np.random.default_rng(seed)
    traces = [np.asarray(x, np.float64) for x in traces]
    T = len(traces)
    if T == 0:
        return BNPResult([], np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    batch = _Batch(traces)
    n = batch.n
    sd = np.array([noise_sd(x) for x in traces])
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nanstd(np.concatenate(traces)) or 1.0)
    h_sd = np.full(T, height_sd) if height_sd else 3.0 * sd
    chi = 1.0 / h_sd ** 2                            # the heights' prior precision
    starts = ([autostepfinder(x).boundaries for x in traces] if start == "autostepfinder"
              else [np.zeros(0, np.int64)] * T)
    M = int(weak_limit or max(20, 2 * max((len(b) for b in starts), default=0)))
    # F, the last level: its prior centred on the trace's end, as wide as the trace
    tail = np.array([x[-max(1, len(x) // 10):].mean() for x in traces]) - batch.offset
    f_ref = tail
    psi = 1.0 / np.maximum(np.array([np.std(x) for x in traces]), sd) ** 2
    # eta: Gamma of shape phi, mean eta_ref -- here the noise estimate, weakly
    phi, eta_ref = 1.0, 1.0 / sd ** 2
    p_on = gamma / M

    # the state
    b = np.zeros((T, M), bool)
    k = (rng.random((T, M)) * n[:, None]).astype(np.int64)
    h = rng.normal(0.0, h_sd[:, None], (T, M))
    F = f_ref.copy()
    eta = eta_ref.copy()
    for i, bound in enumerate(starts):
        bound = np.asarray(bound, np.int64)[:M]
        if not len(bound):
            continue
        edges = np.r_[0, bound, n[i]]
        x = traces[i] - batch.offset[i]
        levels = np.array([x[s:e].mean() for s, e in zip(edges[:-1], edges[1:])])
        # a step at plateau start s adds (level before - level after) to the points before s
        b[i, :len(bound)] = True
        k[i, :len(bound)] = bound - 1
        h[i, :len(bound)] = levels[:-1] - levels[1:]
        F[i] = levels[-1]

    best_lp = np.full(T, -np.inf)
    best_b, best_k = b.copy(), k.copy()
    counts = []
    for sweep in range(sweeps):
        temp = (1.0 + 9.0 * max(0.0, 1.0 - sweep / (0.5 * sweeps))) if anneal else 1.0
        a = np.where(b, h, 0.0)

        def energy(m, km, hm):
            """h (E(k) ...) of the module docstring: the squared error's
            change, given the others, of load m on at index km, height hm."""
            Pm = np.minimum(km[:, None], k) + 1.0
            others = np.sum(a * Pm, axis=1) - a[:, m] * Pm[:, m]
            return hm * ((km + 1.0) * (2 * F + hm) - 2 * batch.prefix(km) + 2 * others)

        # --- the loads, one at a time in random order, all traces together
        for m in rng.permutation(M):
            d_on = energy(m, k[:, m], h[:, m])          # SS(on) - SS(off)
            log_on = (np.log(p_on) - 0.5 * eta * d_on) / temp
            log_off = np.log(1.0 - p_on) / temp
            on = log_on - log_off > -np.log(1.0 / rng.random(T) - 1.0)   # logistic draw
            b[:, m] = on
            a[:, m] = np.where(on, h[:, m], 0.0)

        # --- the times: Metropolis from the prior for the loads that are on
        for m in rng.permutation(M):
            proposal = (rng.random(T) * n).astype(np.int64)
            off = ~b[:, m]
            k[off, m] = proposal[off]
            if (~off).any():
                d = energy(m, proposal, h[:, m]) - energy(m, k[:, m], h[:, m])
                accept = ~off & (np.log(rng.random(T)) < -0.5 * eta * d / temp)
                k[accept, m] = proposal[accept]

        # --- F and the heights: their joint Gaussian conditional
        bf = b.astype(float)
        P = np.minimum(k[:, :, None], k[:, None, :]) + 1.0
        L = np.zeros((T, M + 1, M + 1))
        L[:, 0, 0] = eta * n + psi
        L[:, 0, 1:] = L[:, 1:, 0] = eta[:, None] * bf * (k + 1)
        L[:, 1:, 1:] = eta[:, None, None] * bf[:, :, None] * bf[:, None, :] * P
        L[:, np.arange(1, M + 1), np.arange(1, M + 1)] += chi[:, None]
        q = np.zeros((T, M + 1))
        q[:, 0] = eta * batch.total + psi * f_ref
        q[:, 1:] = eta[:, None] * bf * batch.prefix(k)        # h_ref = 0
        chol = np.linalg.cholesky(L)
        mean = np.linalg.solve(L, q[:, :, None])[:, :, 0]
        z = rng.standard_normal((T, M + 1, 1))
        draw = mean + np.sqrt(temp) * np.linalg.solve(np.swapaxes(chol, 1, 2), z)[:, :, 0]
        F, h = draw[:, 0], draw[:, 1:]

        # --- eta: its Gamma conditional
        a = np.where(b, h, 0.0)
        ss = np.maximum(_ss(batch, F, a, k), 0.0)
        shape = (n / 2 + phi + temp - 1) / temp
        rate = (phi / eta_ref + ss / 2) / temp
        eta = rng.gamma(shape, 1.0 / rate)

        # the log posterior (constants left out), and the best sample kept
        lp = (n / 2 * np.log(eta) - eta * ss / 2
              + np.sum(np.where(b, np.log(p_on), np.log(1 - p_on)), axis=1)
              - 0.5 * chi * np.sum(h * h, axis=1)
              + (phi - 1) * np.log(eta) - phi * eta / eta_ref
              - 0.5 * psi * (F - f_ref) ** 2)
        better = lp > best_lp
        best_lp[better] = lp[better]
        best_b[better], best_k[better] = b[better], k[better]
        if sweep >= sweeps // 2:
            counts.append(b.sum(axis=1))

    boundaries = []
    for i in range(T):
        kk = np.unique(best_k[i][best_b[i]] + 1)
        boundaries.append(kk[(kk > 0) & (kk < n[i])].astype(np.int64))
    return BNPResult(boundaries, np.array([len(x) for x in boundaries]),
                     np.mean(counts, axis=0) if counts else np.zeros(T), best_lp,
                     np.sqrt(1.0 / eta))
