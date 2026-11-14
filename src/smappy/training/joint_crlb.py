"""The Cramer-Rao bound of every emitter of a simulated movie, its neighbours and its other frames
included: what a network that sees overlapping spots and several frames could reach at best.

The single-emitter bound (`decode_smlm.training.test_crlbs`) leaves out what a dense frame costs:
a spot beside another is localized worse than alone, because some of each one's photons could be
the other's.  At 50 molecules in 40 x 40 pixels a network stayed at twice that bound, and the bound,
not the network, was most of the gap.  The bound that counts both is the inverse of one Fisher
matrix over every parameter of the frame -- each emitter's x, y, z and photons, and the background
-- and this module computes it for every emitter of a simulated movie, from each spot's image and
its derivatives on its patch.

**One matrix per group.**  Two emitters are coupled only through the pixels both their patches
cover, so the frame's matrix is block diagonal over the groups of emitters whose patches overlap,
chained (`overlap_groups`): a few emitters each at the densities DECODE trains on, a matrix of
``4 k + 1`` for a group of ``k``, never the frame's.  Each group has a background of its own,
constant over the pixels its patches cover: an emitter alone is then the single-emitter model
exactly (its patch, its own background), and the joint bound is that bound wherever a spot has no
neighbour, so the two compare.  Pixels outside the frame are cut away: a spot at the edge has fewer.

**Temporal context.**  A network reading frames ``t - w .. t + w`` sees a molecule on in several
of them at the same place: its x, y and z are one parameter over them, its photons and the
backgrounds one per frame.  The matrix of the window is the frames' matrices added with those
parameters shared (``context=w``), its groups now chained through a molecule's frames as well; the
bound of a molecule on in all three frames at the same brightness is the single frame's over 3 in
variance.  Each frame of the movie is a window's centre in turn.

**Numerically**, the matrix mixes pixels and photons, a photon count of thousands, nm of z: its
entries span ten orders, and DECODE's float32 ``torch.inverse`` was off by up to half a percent
already on one spot's 5 x 5 (a test) and by more on a group's.  It is inverted in float64 after
scaling by its diagonal, ``D^-1/2 I D^-1/2``, which changes nothing exact and brings the scaled
matrix's condition to what the geometry makes it; through its Cholesky factor where that is well
determined, else by its eigenvalues with the smallest held at 1e-14 of the largest, so a singular
matrix gives a large finite bound and its condition, never an exception.

**Where the bound diverges** -- two emitters closer than the spot's width -- it does so for real:
at zero separation two positions cannot be told apart, and the bound of an unbiased estimator
grows without limit as they approach (Ram, Ward & Ober 2006, https://doi.org/10.1073/pnas.0508047103).
A network is not unbiased: it puts a merged pair near its middle, an error of a pixel or so,
never more than the overlap allows.  ``prior`` adds what is known before the frame is seen --
each position within a pixel or so of the spot, z within the training's range -- as a Gaussian's
information, ``1 / sigma^2`` on the diagonal: the per-configuration form of the Bayesian bound (van
Trees), which caps the bound near ``sigma`` where the frame says nothing and leaves it where the
frame says much (a prior of a pixel moves a bound of a tenth of a pixel by half a percent).  It is
a reference, not a bound per molecule: an estimator biased towards the right answer beats it there.

Everything takes arrays; DECODE's spline draws them in `decode_smlm.training.test_bounds`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: the smallest eigenvalue kept, relative to the largest, of a scaled Fisher matrix: a singular
#: matrix's bound is then about 1e7 times the spot's, finite, and its condition says why
EIGEN_FLOOR = 1e-14
#: the largest (4 n, pixels) derivative matrix of a frame built dense (`frame_fisher`)
DENSE_ELEMENTS = 20_000_000


@dataclass
class Spots:
    """The emitters of a movie, a row each emitter in each frame it is on.

    ``frame`` (n,) the frame, ``ident`` (n,) the molecule (the same in every frame it is on),
    ``corner`` (n, 2) the patch's first row and column on the frame, ``model`` (n, h, w) the
    spot's expected photons per pixel without background, ``grads`` (n, 4, h, w) its derivatives by
    x, y, z and photons (any units: the bounds come back in the same), ``background`` (n_frames,)
    each frame's expected background per pixel, ``shape`` the frame's (rows, columns)."""
    frame: np.ndarray
    ident: np.ndarray
    corner: np.ndarray
    model: np.ndarray
    grads: np.ndarray
    background: np.ndarray
    shape: tuple


@dataclass
class Bounds:
    """Per spot: ``sd`` (n, 4) the bound's standard deviation of x, y, z and photons;
    ``condition`` (n,) the condition of its group's scaled matrix (above 1e10 the frame alone
    cannot separate the group); ``group`` (n,) how many spots its group has."""
    sd: np.ndarray
    condition: np.ndarray
    group: np.ndarray


# ----------------------------------------------------------------------------- one frame
def _pixels(corner, h, w, shape):
    """Each spot's patch as flat pixel indices on the frame and which of the patch's pixels are on
    it: ``(index (n, h*w), inside (n, h*w))``."""
    rr = corner[:, 0, None, None] + np.arange(h)[None, :, None]
    cc = corner[:, 1, None, None] + np.arange(w)[None, None, :]
    rr, cc = np.broadcast_arrays(rr, cc)
    inside = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    index = np.where(inside, rr * shape[1] + cc, 0)
    return index.reshape(len(corner), -1), inside.reshape(len(corner), -1)


def overlap_groups(corner, h, w) -> np.ndarray:
    """Each spot's group: spots whose patches (``h`` x ``w`` from their ``corner``) share a pixel,
    chained.  The test is on the patches' rectangles, which is what couples two spots."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(corner)
    if n == 0:
        return np.zeros(0, int)
    d = np.abs(corner[:, None, :] - corner[None, :, :])
    a, b = np.nonzero((d[..., 0] < h) & (d[..., 1] < w))
    return connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)[1]


def frame_fisher(corner, model, grads, background: float, shape, excess: float = 1.0):
    """The Fisher matrix of one frame's spots, in the frame's own parameters: each spot's x, y, z
    and photons (``4 a + k``), then a background per overlap group.  Returns ``(fisher, groups)``,
    the matrix a scipy sparse COO of size ``4 n + n_groups`` (each spot's patch on the others'
    only where they overlap) and each spot's group.  ``excess`` is the camera's excess noise
    factor in variance: an EM register at high gain, 2, halves the information.

    Poisson pixels: ``I_ij = sum_p d_i mu_p d_j mu_p / mu_p`` with ``mu`` the background and every
    spot drawn together (`overlap_groups`), each group's background ``d mu / d b = 1`` on the pixels
    its patches cover."""
    from scipy.sparse import coo_matrix, csr_matrix, diags
    n, h, w = model.shape
    size = shape[0] * shape[1]
    groups = overlap_groups(corner, h, w)
    n_groups = int(groups.max()) + 1 if n else 0
    index, inside = _pixels(corner, h, w, shape)
    mu = np.full(size, float(background))
    np.add.at(mu, index[inside], model.reshape(n, -1)[inside].astype(float))
    inv = 1.0 / np.maximum(mu, 1e-12)
    # the spots' derivatives as one (4n, pixels) matrix, the frame's matrix one product: dense
    # where the patches cover much of the frame (40 x 40 pixels, patches of 26: ten times faster),
    # sparse where they do not
    rows = np.repeat(np.arange(4 * n).reshape(n, 4)[:, :, None], h * w, axis=2)
    cols = np.repeat(index[:, None, :], 4, axis=1)
    keep = np.repeat(inside[:, None, :], 4, axis=1)
    values = grads.reshape(n, 4, -1)[keep].astype(float)
    if 4 * n * size <= DENSE_ELEMENTS:
        g = np.zeros((4 * n, size))
        g[rows[keep], cols[keep]] = values                         # a pixel once per spot
        gw = g * inv[None, :]
        spots = coo_matrix(gw @ g.T)
        cross = gw.sum(axis=1)                                     # sum over the patch of d mu / mu
    else:
        g = csr_matrix((values, (rows[keep], cols[keep])), shape=(4 * n, size))
        gw = g @ diags(inv)
        spots = (gw @ g.T).tocoo()
        cross = np.asarray(gw.sum(axis=1)).ravel()
    # a group's background: the pixels its patches cover, each once
    label = np.full(size, -1)
    label[index[inside]] = np.repeat(groups, h * w)[inside.ravel()]
    covered = label >= 0
    bb = np.bincount(label[covered], inv[covered], n_groups)
    b_rows = np.repeat(groups, 4) + 4 * n
    r = np.r_[spots.row, np.arange(4 * n), b_rows, 4 * n + np.arange(n_groups)]
    c = np.r_[spots.col, b_rows, np.arange(4 * n), 4 * n + np.arange(n_groups)]
    v = np.r_[spots.data, cross, cross, bb] / float(excess)
    return coo_matrix((v, (r, c)), shape=(4 * n + n_groups,) * 2), groups


# ----------------------------------------------------------------------------- inversion
def variances(fisher: np.ndarray, condition: bool = True):
    """The diagonal of a Fisher matrix's inverse and the condition of its scaled form (None
    unless ``condition``).

    Scaled by its diagonal first (``A = D^-1/2 I D^-1/2``: exact, and the units no longer
    matter); inverted through its Cholesky factor, a twentieth of the eigenvalues' time on the
    groups of a dense frame, where that is well determined (the factor's diagonal within 1e-6 of
    its largest), else by its eigenvalues with the smallest held at `EIGEN_FLOOR` of the largest.  A
    parameter without information (a zero diagonal) is infinite."""
    from scipy.linalg import solve_triangular
    f = np.asarray(fisher, float)
    d = np.diag(f).copy()
    none = ~(d > 0)
    d[none] = 1.0
    s = 1.0 / np.sqrt(d)
    a = f * s[:, None] * s[None, :]
    a = 0.5 * (a + a.T)
    # the spline's far tails make couplings of 1e-300, subnormal numbers that slowed the factor of a
    # dense frame's window (400 parameters) five times; below 1e-15 of the diagonal they are nothing
    a[np.abs(a) < 1e-15] = 0.0
    var, cond = None, None
    if not none.any():
        try:
            low = np.linalg.cholesky(a)
            diag = np.abs(np.diag(low))
            if diag.min() > 1e-6 * diag.max():
                inv_low = solve_triangular(low, np.eye(len(a)), lower=True)
                var = (inv_low ** 2).sum(0) / d
        except np.linalg.LinAlgError:
            pass
    if var is None or condition:
        lam, v = np.linalg.eigh(a) if var is None else (np.linalg.eigvalsh(a), None)
        top = max(float(lam[-1]), 1e-300)
        floor = EIGEN_FLOOR * top
        if var is None:
            var = (v ** 2 / np.maximum(lam, floor)[None, :]).sum(1) / d
        cond = top / max(float(lam[0]), floor)
    var[none] = np.inf
    return var, cond


# ----------------------------------------------------------------------------- the movie
def one_blas_thread():
    """A context holding BLAS to one thread, where threadpoolctl is installed: OpenBLAS's threads
    made the factors of a dense frame's window (a few hundred parameters, one after another)
    three times slower on four cores (0.85 s against 2.6 s for 64 frames, 2026-11-14)."""
    import contextlib
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(1, user_api="blas")


def frame_matrices(spots: Spots, excess: float = 1.0) -> dict:
    """Every frame's `frame_fisher`, ``{frame: (rows, fisher, n_groups)}``: what `bounds` adds
    up, computed once for several of its calls."""
    out = {}
    for f in np.unique(spots.frame):
        rows = np.flatnonzero(spots.frame == f)
        fisher, groups = frame_fisher(spots.corner[rows], spots.model[rows], spots.grads[rows],
                                      float(spots.background[int(f)]), spots.shape, excess)
        out[int(f)] = (rows, fisher, int(groups.max()) + 1)
    return out


def bounds(spots: Spots, *, context: int = 0, prior=None, excess: float = 1.0, matrices=None,
           condition: bool = True) -> Bounds:
    """Every spot's joint bound (`Bounds`): its frame's neighbours and, with ``context`` = w, the
    frames ``t - w .. t + w`` around its own, a molecule's x, y and z shared across them.

    ``prior``: None, or the standard deviations ``(x, y, z, photons)`` (in the derivatives' units,
    None for none) of a Gaussian's information added to each molecule's parameters -- what caps
    the bound where two spots merge (the module's docstring).  ``matrices``: `frame_matrices`
    of the same spots and ``excess``, for several calls.  ``condition`` False leaves it NaN,
    for speed."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(spots.frame)
    sd = np.full((n, 4), np.nan)
    cond_out = np.full(n, np.nan)
    group = np.zeros(n, int)
    if n == 0:
        return Bounds(sd, cond_out, group)
    frames = np.unique(spots.frame)
    per_frame = frame_matrices(spots, excess) if matrices is None else matrices
    prior_info = None
    if prior is not None:
        prior_info = np.array([0.0 if p is None else 1.0 / float(p) ** 2 for p in prior])
    for t in frames:
        t = int(t)
        window = [f for f in range(t - context, t + context + 1) if f in per_frame]
        rows = np.concatenate([per_frame[f][0] for f in window])
        idents, slot = np.unique(spots.ident[rows], return_inverse=True)
        n_id, n_rows = len(idents), len(rows)
        # the window's parameters: each molecule's x, y, z; each row's photons; each group's background
        r_all, c_all, v_all = [], [], []
        at, b_at = 0, 3 * n_id + n_rows
        for f in window:
            frows, fisher, n_groups = per_frame[f]
            k = len(frows)
            s = slot[at:at + k]
            local = np.empty(4 * k + n_groups, int)
            local[:4 * k] = np.c_[3 * s, 3 * s + 1, 3 * s + 2, 3 * n_id + at + np.arange(k)].ravel()
            local[4 * k:] = b_at + np.arange(n_groups)
            r_all.append(local[fisher.row])
            c_all.append(local[fisher.col])
            v_all.append(fisher.data)
            at += k
            b_at += n_groups
        size = b_at
        if prior_info is not None:
            diag = np.r_[np.tile(prior_info[:3], n_id), np.full(n_rows, prior_info[3])]
            r_all.append(np.arange(len(diag)))
            c_all.append(np.arange(len(diag)))
            v_all.append(diag)
        big = coo_matrix((np.concatenate(v_all), (np.concatenate(r_all), np.concatenate(c_all))),
                         shape=(size, size)).tocsr()
        big.sum_duplicates()
        _, part = connected_components(big, directed=False)
        centre = np.flatnonzero(spots.frame[rows] == t)
        wanted = np.unique(np.r_[part[3 * slot[centre]], part[3 * n_id + centre]])
        for p in wanted:
            members = np.flatnonzero(part == p)
            var, cond = variances(big[members][:, members].toarray(), condition)
            full = np.full(size, np.nan)
            full[members] = var
            mine = centre[(part[3 * slot[centre]] == p)]
            xyz = full[3 * slot[mine, None] + np.arange(3)[None, :]]
            sd[rows[mine], :3] = np.sqrt(xyz)
            sd[rows[mine], 3] = np.sqrt(full[3 * n_id + mine])
            cond_out[rows[mine]] = np.nan if cond is None else cond
            in_part = np.isin(3 * slot, members)          # the rows whose molecule is in this group
            group[rows[mine]] = int(in_part.sum())
    return Bounds(sd, cond_out, group)


def single_bounds(spots: Spots, excess: float = 1.0) -> np.ndarray:
    """Each spot alone on its whole patch with a background of its own, (n, 4): the single-emitter
    bound, which `bounds` reaches for a spot with no neighbour inside the frame."""
    model = spots.model.reshape(len(spots.model), -1).astype(float)
    g = spots.grads.reshape(len(model), 4, -1).astype(float)
    mu = np.maximum(model + spots.background[spots.frame][:, None], 1e-12)
    d = np.concatenate([g, np.ones_like(model)[:, None]], 1)       # (n, 5, P)
    fisher = np.matmul(d / mu[:, None], d.transpose(0, 2, 1)) / float(excess)
    # all at once, scaled as `variances` scales; one by one only where that fails (a spot outside
    # the spline's z, whose derivatives are zero)
    diag = np.einsum("nii->ni", fisher)
    if np.all(diag > 0):
        s = 1.0 / np.sqrt(diag)
        try:
            inv = np.linalg.inv(fisher * s[:, :, None] * s[:, None, :])
            var = np.einsum("nii->ni", inv)[:, :4] * s[:, :4] ** 2
            if np.all(var > 0):
                return np.sqrt(var)
        except np.linalg.LinAlgError:
            pass
    return np.array([np.sqrt(variances(f, False)[0][:4]) for f in fisher])


# ----------------------------------------------------------------------------- what a bound is split by
def nearest_neighbour(frame, xy) -> np.ndarray:
    """Each spot's lateral distance to the nearest other spot of its frame (xy's units); inf for
    one alone."""
    from scipy.spatial import cKDTree
    frame = np.asarray(frame)
    xy = np.asarray(xy, float)
    out = np.full(len(frame), np.inf)
    for f in np.unique(frame):
        i = np.flatnonzero(frame == f)
        if len(i) > 1:
            out[i] = cKDTree(xy[i]).query(xy[i], k=2)[0][:, 1]
    return out


def frames_on(frame, ident, context: int) -> np.ndarray:
    """In how many of the frames ``t - context .. t + context`` each spot's molecule is on (1 ..
    2 context + 1): what temporal context has to work with."""
    frame = np.asarray(frame, int)
    ident = np.asarray(ident)
    on = set(zip(frame.tolist(), ident.tolist()))
    return np.array([sum((f + d, m) in on for d in range(-context, context + 1))
                     for f, m in zip(frame.tolist(), ident.tolist())])
