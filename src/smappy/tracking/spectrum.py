"""The spectrum of diffusion coefficients: how the tracks' jumps share out
over a fine grid of D, with no number of states chosen, and each piece of a
track's posterior over that grid -- the *state arrays* of saSPT (Heckert et
al. 2022), re-implemented from the paper rather than ported.

**Why.**  The jump fit (`analysis.fit_jumps`) takes 1 to 3 states and
treats every jump as independent.  But consecutive jumps of a track share
a localization, so their errors are anti-correlated, and the number of
states is a choice the data should make.  Here:

* **pieces.**  Each track's jumps between consecutive localizations are
  cut into pieces of at most *split* jumps (saSPT's 10): a particle that
  changes state within a track is then seen in each state for a piece,
  while a piece is still long enough to say which;
* **the likelihood of a piece** at each D of the grid is the exact
  Gaussian one of its whole jump vector, per axis: jump j over k_j frames
  has the variance 2 D k_j dt + s_j^2 + s_{j+1}^2 (its two localizations'
  errors), and jumps j - 1 and j share localization j, so they covary by
  -s_j^2.  The covariance is tridiagonal, its determinant and quadratic
  form one LDL pass along the piece (`piece_loglik`), vectorised over every
  piece of a length and every D at once.  saSPT grids the error too, one
  sigma for all; here each localization's own precision is used, which
  drops that axis of the grid, as the rest of smappy trusts the fitter's
  precision (NOTES, "offset from the precisions");
* **the occupations** of the grid -- the share of the jumps at each D --
  are found by variational Bayes, saSPT's scheme: a Dirichlet over the
  occupations, a flat prior of one pseudo-count per D, and each piece
  assigned to the D's in proportion to its likelihood times
  exp(psi(alpha_D) - psi(sum alpha)), weighted by its number of jumps,
  until no occupation moves by more than 1e-7 (at most 5000 rounds,
  `state_array`).  The result is a smooth spectrum whose peaks are the
  populations, as many as the data support;
* **the focus.**  A piece's jumps are mostly between consecutive frames,
  so a state's share of them is its fraction of the particles times the
  fraction still in focus one frame on (`analysis.in_focus` at one frame,
  saSPT's `defoc_corr`); dividing by that gives the fractions of the
  particles.  Jumps across a dark frame are a little more likely lost,
  which this ignores.

Each piece's posterior over D is what is written per localization: its
mean of log D, as a D (`D_post`), so the tracks can be coloured, filtered
or split by it, and a track that changes state shows it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .analysis import Tracks, in_focus

# saSPT's grid runs from 0.01 to 100 um^2/s; MINFLUX and slow membrane
# proteins need a decade lower
D_MIN, D_MAX, D_POINTS = 1e-3, 1e2, 100
# pieces handled at once: memory is pieces x grid x a few doubles
CHUNK = 20_000


def d_grid(d_min: float = D_MIN, d_max: float = D_MAX, points: int = D_POINTS
           ) -> np.ndarray:
    """The diffusion coefficients (um^2/s), log-spaced."""
    return np.logspace(np.log10(d_min), np.log10(d_max), int(points))


@dataclass
class Pieces:
    """Every track's consecutive jumps, cut into pieces of at most *split*:
    per piece its track, number of jumps and first row (into the `Tracks`,
    whose rows are sorted by track and time); a piece of n jumps spans the
    rows first .. first + n."""
    track: np.ndarray
    n_jumps: np.ndarray
    first: np.ndarray

    def __len__(self) -> int:
        return len(self.track)


def pieces_of(t: Tracks, split: int = 10) -> Pieces:
    """The pieces of every track with a jump: its jumps between consecutive
    localizations, in runs of ``split`` from its start, the last shorter."""
    split = max(int(split), 1)
    n = len(t.track)
    if n < 2:
        z = np.zeros(0, np.int64)
        return Pieces(z, z, z)
    same = t.track[1:] == t.track[:-1]          # a jump from row i to i + 1
    jump_rows = np.flatnonzero(same)
    if not len(jump_rows):
        z = np.zeros(0, np.int64)
        return Pieces(z, z, z)
    tr = t.track[jump_rows]
    starts = np.concatenate([[True], tr[1:] != tr[:-1]])
    index = np.arange(len(tr))
    rank = index - np.maximum.accumulate(np.where(starts, index, 0))
    begins = rank % split == 0
    piece = np.cumsum(begins) - 1
    first = jump_rows[begins]
    n_jumps = np.bincount(piece)
    return Pieces(tr[begins], n_jumps.astype(np.int64), first)


def piece_loglik(t: Tracks, pieces: Pieces, d_um2_s: np.ndarray) -> np.ndarray:
    """The log-likelihood of every piece at every D (pieces x grid): the
    Gaussian of its jumps per axis, with the tridiagonal covariance of
    shared localizations, summed over the axes."""
    D = np.asarray(d_um2_s, float) * 1e6                  # nm^2/s
    out = np.zeros((len(pieces), len(D)))
    errs = t.axis_errors()
    for L in np.unique(pieces.n_jumps):
        which = np.flatnonzero(pieces.n_jumps == L)
        rows = pieces.first[which][:, None] + np.arange(L + 1)[None, :]   # P x (L+1)
        lag_s = (t.step[rows[:, 1:]] - t.step[rows[:, :-1]]) * t.dt_s    # P x L
        for lo in range(0, len(which), CHUNK):
            part = slice(lo, lo + CHUNK)
            r, lag = rows[part], lag_s[part]
            total = np.zeros((len(r), len(D)))
            for axis in range(t.dims):
                x = t.xyz[r[:, 1:], axis] - t.xyz[r[:, :-1], axis]          # P x L
                s2 = errs[r, axis]                                           # P x (L+1)
                noise = s2[:, :-1] + s2[:, 1:]
                total += _tridiagonal_loglik(x, lag, noise, s2[:, 1:-1], D)
            out[which[part]] = total
    return out


def _tridiagonal_loglik(x, lag_s, noise, shared, D):
    """log N(x; 0, C) for each row of ``x`` (P x L) and each D, C with
    diagonal 2 D lag + noise and off-diagonal -shared (P x (L-1)), by
    the LDL factorisation of a tridiagonal matrix, one jump at a time."""
    P, L = x.shape
    a = 2.0 * D[None, :] * lag_s[:, 0, None] + noise[:, 0, None]          # P x G
    d = a
    y = np.broadcast_to(x[:, 0, None], d.shape)
    quad = y * y / d
    logdet = np.log(d)
    for j in range(1, L):
        b = -shared[:, j - 1, None]                                        # P x 1
        a = 2.0 * D[None, :] * lag_s[:, j, None] + noise[:, j, None]
        ell = b / d
        d = a - ell * b
        y = x[:, j, None] - ell * y
        quad = quad + y * y / d
        logdet = logdet + np.log(d)
    return -0.5 * (quad + logdet + L * np.log(2 * np.pi))


def state_array(loglik: np.ndarray, weights: np.ndarray, prior: float = 1.0,
                max_iter: int = 5000, tol: float = 1e-7):
    """The occupations of the grid (summing to 1) and each piece's
    posterior over it (pieces x grid), by variational Bayes on a Dirichlet
    of ``prior`` pseudo-counts per grid point, the pieces weighted by
    ``weights`` (their numbers of jumps).

    Each piece's likelihoods are scaled to their largest once, so a round
    is two products of that matrix with a vector -- 0.1 ms for 2500 pieces
    on 100 D's, where the same in log space took 5 ms -- and the rounds go
    on until no occupation moves by more than ``tol``.  The scheme is
    EM's, slow where neighbouring D's share a population: after saSPT's
    200 rounds single grid points were still 8% from where they settle,
    though the fraction under a peak no longer moved."""
    from scipy.special import digamma
    loglik = np.asarray(loglik, float)
    w = np.asarray(weights, float)
    G = loglik.shape[1]
    lik = np.exp(loglik - loglik.max(axis=1, keepdims=True))      # P x G, max 1
    alpha = np.full(G, float(prior)) + w.sum() / G
    occupation = alpha / alpha.sum()
    for _ in range(int(max_iter)):
        e = np.exp(digamma(alpha) - digamma(alpha.sum()))
        norm = lik @ e                                              # per piece
        alpha = prior + e * (lik.T @ (w / norm))
        new = alpha / alpha.sum()
        moved = np.abs(new - occupation).max()
        occupation = new
        if moved < tol:
            break
    e = np.exp(digamma(alpha) - digamma(alpha.sum()))
    R = lik * e[None, :]
    R /= R.sum(axis=1, keepdims=True)
    return occupation, R


@dataclass
class Spectrum:
    """The D spectrum of a set of tracks: the grid (um^2/s), the share of
    the jumps at each D (``seen``) and of the particles (``occupation``,
    the same without a focal range), and per piece its track, rows and
    posterior mean of log D, as a D."""
    D_um2_per_s: np.ndarray
    occupation: np.ndarray
    seen: np.ndarray
    pieces: Pieces
    D_post_um2_per_s: np.ndarray
    n_jumps: int
    focal_range_nm: float = 0.0

    def fraction_between(self, lo: float, hi: float) -> float:
        """The fraction of the particles with D in [lo, hi) (um^2/s)."""
        D = self.D_um2_per_s
        return float(self.occupation[(D >= lo) & (D < hi)].sum())


def spectrum(t: Tracks, d_um2_s: Optional[np.ndarray] = None, split: int = 10,
             focal_range_nm: float = 0.0, prior: float = 1.0) -> Spectrum:
    """The state array of the tracks ``t`` on the grid ``d_um2_s``
    (`d_grid` by default)."""
    D = d_grid() if d_um2_s is None else np.asarray(d_um2_s, float)
    pieces = pieces_of(t, split)
    if not len(pieces):
        raise ValueError("no track has two localizations: no jumps")
    ll = piece_loglik(t, pieces, D)
    seen, R = state_array(ll, pieces.n_jumps, prior)
    occupation = seen
    focus = bool(focal_range_nm) and focal_range_nm > 0
    if focus:
        kept = in_focus(D, t.dt_s, float(focal_range_nm), 1)[:, 0]
        occupation = seen / np.maximum(kept, 1e-12)
        occupation = occupation / occupation.sum()
    post = np.exp(R @ np.log(D))
    return Spectrum(D, occupation, seen, pieces, post, int(pieces.n_jumps.sum()),
                    float(focal_range_nm) if focus else 0.0)


def per_localization(t: Tracks, spec: Spectrum, n_rows: int) -> np.ndarray:
    """Each piece's posterior D on the localizations it spans, NaN on the
    rest (rows of the table, ``n_rows`` of them); a localization shared by
    two pieces takes the later."""
    out = np.full(n_rows, np.nan)
    p = spec.pieces
    if not len(p):
        return out
    piece = np.repeat(np.arange(len(p)), p.n_jumps + 1)
    offset = np.arange(len(piece)) - np.repeat(np.cumsum(p.n_jumps + 1) - (p.n_jumps + 1),
                                               p.n_jumps + 1)
    rows = p.first[piece] + offset
    out[t.rows[rows]] = spec.D_post_um2_per_s[piece]
    return out


def peaks(spec: Spectrum, min_fraction: float = 0.02):
    """The spectrum's populations: each local maximum of the occupations
    with its basin, the grid points down to the minima on either side,
    as (D at the maximum, fraction of the particles in the basin), largest
    D last; basins of less than ``min_fraction`` are left out."""
    occ = spec.occupation
    G = len(occ)
    if not G:
        return []
    # minima split the grid into basins; a flat stretch splits at its start
    rising = np.diff(occ) > 0
    cuts = [0] + [i + 1 for i in range(len(rising) - 1)
                  if not rising[i] and rising[i + 1]] + [G]
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        f = float(occ[lo:hi].sum())
        if f >= min_fraction:
            out.append((float(spec.D_um2_per_s[lo + int(np.argmax(occ[lo:hi]))]), f))
    return out
