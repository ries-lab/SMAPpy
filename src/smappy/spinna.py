"""SPINNA: the oligomeric states of molecules from their nearest-neighbour
distances.

Ported from Picasso's ``picasso/spinna.py`` (Jungmann lab, MIT licence;
Masullo, Kowalewski et al., Nat Commun 2025;16:4202,
doi:10.1038/s41467-025-59500-z).  The question: of the molecules a DNA-PAINT
experiment resolved one by one (the centres of *Molecular mapping*, RESI),
which fraction sits in monomers, which in dimers, trimers, or in complexes
with a second protein?  Single molecules cannot be counted into complexes
directly -- a label is missing from most complexes, the label sits a few
nanometres off its protein, and complexes lie closer to each other than the
eye can separate -- so SPINNA simulates.  Every candidate mixture of
*structures* is placed at random in the observed window, turned, labelled
with the labelling efficiency and blurred by the label uncertainty, and the
mixture whose nearest-neighbour distances look most like the data's wins.

What is Picasso's here, by function:

* `search_space` is ``generate_N_structures``: the candidate numbers of each
  structure that use up exactly the molecules there are (observed / the
  labelling efficiency) -- a linear system with one equation per target,
  whose free unknowns are put on a grid of ``granularity`` values from none
  to as many as there are molecules for, the others solved.  The dependent
  structures are the first ones that make the system solvable (Picasso finds
  them with its ``rref``; a rank test is the same choice).
* `proportions` is ``StructureMixer.convert_counts_to_props``: a structure's
  share of the *molecules* (all targets), in %, not of the structures.
* `simulate` is ``StructureSimulator.run``: centres uniform in the window,
  each structure turned at random in the plane (or in 3D, or not), the
  label uncertainty a Gaussian per coordinate, and the labelling efficiency
  kept as Picasso keeps it -- exactly ``int(n * efficiency)`` of each
  structure's molecules of a target, drawn without replacement, rather than
  each molecule on its own coin.
* `neighbour_orders` is ``StructureMixer.get_neighbor_idx`` with
  ``get_max_nn``: for every pair of targets, the 1st to k-th nearest
  neighbours, k the most partners a structure gives a molecule (a dimer 1,
  a trimer 2, a heterodimer's A 1 of B); one direction for two targets.
* `nn_distances` is ``get_NN_dist`` (a k-d tree, a point not its own
  neighbour); `nnd_score` is ``NND_score``: the two-sample Kolmogorov-Smirnov
  statistic of every pair and order, averaged, 1 when nothing compares.
* `fit` is ``SPINNA.fit_stoichiometry`` with its brute force and
  ``fit_coarse_to_fine`` (farthest-point sampling, ``_farthest_point_sampling``,
  then every candidate within 30 percentage points, L1, of the coarse winner,
  ``get_subset_N_structures``); each candidate simulated ``n_sims`` times
  (``N_sim``) and the distances pooled.
* The bootstrap is ``_run_bootstrap``: 20 data sets simulated from the best
  mixture, the neighbourhood of the best (30 points, L1) fitted to each, the
  standard deviation of the proportions found its error -- a parametric
  bootstrap: it says how much the answer moves with the randomness of
  placement and labelling, not how wrong the model is.
* `fit_efficiency` is ``fit_le`` and ``get_le_from_props`` (Hellmeier et al.,
  Nat Methods 2024;21:1702): a protein carrying two tags A and B, simulated
  with every label detected, as A alone, B alone and an A-B pair at the tags'
  distance; the fraction of B seen with an A partner is A's efficiency.

Where it departs, and why:

* **The window is SMAPpy's**: the region, the drawn ROI or the area the
  molecules occupy (`plugins.spatial_statistics.Window`), uniform inside it.
  Picasso's ``MaskGenerator`` -- a blurred density map of the data, centres
  drawn in proportion to it -- is not ported: a region of even density (a
  cell's ROI each, `per_region`) does the same job in the open.
* **Files apart**: a table of several files is several windows; each file
  gets the candidate's structures in proportion to its share of the
  molecules, and the distances of all files are pooled.  Picasso takes one
  data set.
* **The coarse pass covers the grid.**  Picasso samples 10 % of the
  candidates; on a grid of one free structure and granularity 21 that is two
  candidates, 50 % and an end, and the fine pass (30 points around the
  winner) can never reach a true 20 %.  Here the sampling goes on until every
  candidate is within half that radius of one sampled (and is at least
  10 %), so the fine pass always reaches the region the coarse winner stood
  for.
* The free counts are rounded before the dependent ones are solved, so each
  candidate uses up exactly the molecules there are; Picasso truncated both.
  Duplicate candidates (a grid finer than the counts) are dropped.
* The KS statistic is computed from the two sorted samples directly
  (`ks_statistic`); ``scipy.stats.ks_2samp`` gives the same number but also
  an exact p-value that is not used and costs more than the rest.
* One random generator, seeded, so a run can be repeated.
* Not ported: the Bayesian search (``fit_bayesian``), fitting the label
  uncertainty and comparing models (``compare_models``) -- run the fit with
  each model and compare the scores -- and the masks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# Picasso's colours of the 1st to 4th nearest neighbours (``NN_COLORS``)
NN_COLOURS = ("#2880C4", "#97D8C4", "#F4B942", "#363636")
# bootstrap rounds (``N_BOOTSTRAPS``)
N_BOOTSTRAPS = 20
# the fine pass and the bootstrap look this far around a winner, in
# percentage points of the proportions, L1 (``BOOTSTRAP_DISTANCE`` with
# ``BOOTSTRAP_DISTANCE_METRIC`` = 1)
NEIGHBOURHOOD_PCT = 30.0
# the coarse pass samples at least this fraction of the candidates
COARSE_FRACTION = 0.1
# Picasso's GUI defaults: grid steps, simulations per candidate, label
# uncertainty (nm), labelling efficiency
GRANULARITY = 21
N_SIMS = 10
LABEL_UNC_NM = 5.0
EFFICIENCY = 0.5
# the histograms' bin (``plot_NN``'s binsize)
HIST_BIN_NM = 4.0
TARGET_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ROTATIONS = ("plane", "3D", "none")


# ---------------------------------------------------------------- structures

@dataclass
class Structure:
    """One kind of complex: where its molecules sit (nm, about its centre)
    and which target each is (0 for A, 1 for B, ...)."""
    name: str
    xyz: np.ndarray          # (m, 3)
    target: np.ndarray       # (m,) int

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(-1, 3)
        self.target = np.asarray(self.target, dtype=np.int64).reshape(-1)
        if len(self.xyz) != len(self.target):
            raise ValueError(f"{self.name}: a target for every molecule")
        if not len(self.xyz):
            raise ValueError(f"{self.name}: a structure needs a molecule")

    def count(self, target: int) -> int:
        return int(np.sum(self.target == target))


def letter(target: int) -> str:
    return TARGET_LETTERS[target] if target < len(TARGET_LETTERS) else f"T{target + 1}"


def parse_structure(name: str, text: str) -> Structure:
    """A structure from ``"A -10 0; A 10 0"``: its molecules separated by
    ``;`` (or new lines), each ``x y`` or ``x y z`` in nm, after an optional
    letter naming its target (A when none; B the second target, ...)."""
    xyz, targets = [], []
    for part in re.split(r"[;\n]", str(text)):
        tokens = part.replace(",", " ").split()
        if not tokens:
            continue
        target = 0
        if tokens[0].isalpha():
            if len(tokens[0]) != 1:
                raise ValueError(f"{name}: {tokens[0]!r} is not a target; a target is "
                                 "one letter, A for the first")
            target = TARGET_LETTERS.index(tokens[0].upper())
            tokens = tokens[1:]
        try:
            numbers = [float(t) for t in tokens]
        except ValueError:
            raise ValueError(f"{name}: {part.strip()!r} is not 'x y' or 'x y z' in nm, "
                             "after an optional target letter") from None
        if len(numbers) not in (2, 3):
            raise ValueError(f"{name}: {part.strip()!r} has {len(numbers)} numbers; "
                             "a molecule is 'x y' or 'x y z'")
        xyz.append(numbers + [0.0] * (3 - len(numbers)))
        targets.append(target)
    if not xyz:
        raise ValueError(f"{name}: no molecules; write them as 'A -10 0; A 10 0'")
    return Structure(name, np.array(xyz), np.array(targets))


def n_targets_of(structures: Sequence[Structure]) -> int:
    return int(max(int(s.target.max()) for s in structures)) + 1


def target_counts(structures: Sequence[Structure], n_targets: int) -> np.ndarray:
    """(targets, structures): how many molecules of each target a structure
    holds (``_find_target_counts``)."""
    return np.array([[s.count(t) for s in structures] for t in range(n_targets)],
                    dtype=float).reshape(n_targets, len(structures))


def neighbour_orders(structures: Sequence[Structure], n_targets: int
                     ) -> List[Tuple[int, int, int]]:
    """``(t1, t2, k)``: the pairs of targets whose 1st to k-th nearest
    neighbours tell the structures apart, k the most partners of the second
    target a structure gives a molecule of the first (``get_max_nn``); one
    direction for two targets, and none where no structure pairs them."""
    out = []
    for t1 in range(n_targets):
        for t2 in range(t1, n_targets):
            k = 0
            for s in structures:
                n1, n2 = s.count(t1), s.count(t2)
                k = max(k, n1 - 1 if t1 == t2 else (min(n1, n2) if n1 and n2 else 0))
            if k > 0:
                out.append((t1, t2, int(k)))
    return out


def order_name(order: Tuple[int, int, int], names: Optional[Sequence[str]] = None) -> str:
    t1, t2, _ = order
    a = names[t1] if names else letter(t1)
    b = names[t2] if names else letter(t2)
    return f"{a} to {b}"


# -------------------------------------------------------------- the search

def search_space(structures: Sequence[Structure], totals: Sequence[float],
                 granularity: int) -> np.ndarray:
    """Every candidate as numbers of each structure, (n, structures), each
    using up the ``totals`` of molecules per target (observed over the
    labelling efficiency) -- ``generate_N_structures``.

    The structures that make the target equations solvable are dependent, the
    rest free: each free one takes ``granularity`` values from 0 to as many as
    the molecules allow, the dependent ones follow, and a candidate needing a
    negative number is not one."""
    n_t = len(totals)
    a = target_counts(structures, n_t)
    totals = np.asarray(totals, dtype=float)
    n_s = a.shape[1]
    if n_s < n_t:
        raise ValueError(f"{n_t} targets need at least {n_t} structures")
    dependent: List[int] = []
    for j in range(n_s):
        if np.linalg.matrix_rank(a[:, dependent + [j]]) > len(dependent):
            dependent.append(j)
        if len(dependent) == n_t:
            break
    if len(dependent) < n_t:
        missing = [letter(t) for t in range(n_t) if not a[t].any()]
        raise ValueError("no structure holds target " + ", ".join(missing) if missing
                         else "the structures cannot account for every target")
    free = [j for j in range(n_s) if j not in dependent]
    a_dep, a_free = a[:, dependent], a[:, free]
    if free:
        with np.errstate(divide="ignore", invalid="ignore"):
            reach = np.where(a_free > 0, totals[:, None] / a_free, np.inf)
        top = np.floor(reach.min(axis=0))
        bases = [np.linspace(0.0, m, max(int(granularity), 2)) for m in top]
        grid = np.round(np.array(list(product(*bases)), dtype=float))
    else:
        grid = np.zeros((1, 0))
    rhs = totals[:, None] - a_free @ grid.T
    solved = np.linalg.lstsq(a_dep, rhs, rcond=None)[0].T
    solved = np.round(solved)
    keep = np.all(solved >= 0, axis=1)
    counts = np.zeros((int(keep.sum()), n_s), dtype=np.int64)
    counts[:, dependent] = solved[keep]
    counts[:, free] = grid[keep]
    if not len(counts):
        raise ValueError("no mixture of these structures uses up the molecules there are")
    return np.unique(counts, axis=0)


def proportions(counts: np.ndarray, structures: Sequence[Structure]) -> np.ndarray:
    """Each structure's share of all molecules, in % (``convert_counts_to_props``)."""
    counts = np.atleast_2d(np.asarray(counts, dtype=float))
    sizes = np.array([len(s.xyz) for s in structures], dtype=float)
    mols = counts * sizes
    total = mols.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, 100.0 * mols / total, 0.0)


def farthest_points(points: np.ndarray, n_samples: int, cover: float = 0.0) -> np.ndarray:
    """A spread subset (``_farthest_point_sampling``): the point nearest the
    centroid, then again and again the one farthest from those taken -- at
    least ``n_samples``, and on until every point is within ``cover`` of one
    taken (L1)."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    first = int(np.argmin(np.abs(points - points.mean(axis=0)).sum(axis=1)))
    taken = [first]
    reach = np.abs(points - points[first]).sum(axis=1)
    while len(taken) < n and (len(taken) < n_samples or reach.max() > cover):
        nxt = int(np.argmax(reach))
        taken.append(nxt)
        reach = np.minimum(reach, np.abs(points - points[nxt]).sum(axis=1))
    return np.array(taken)


def around(props: np.ndarray, centre: np.ndarray, radius: float = NEIGHBOURHOOD_PCT
           ) -> np.ndarray:
    """The candidates within ``radius`` percentage points (L1) of ``centre``
    (``get_subset_N_structures``)."""
    return np.flatnonzero(np.abs(props - centre).sum(axis=1) <= radius + 1e-9)


# ----------------------------------------------------------------- simulation

def _rotated(xyz: np.ndarray, n: int, rotation: str, rng) -> np.ndarray:
    """(n, m, 3): the structure turned at random ``n`` times."""
    if rotation == "plane":
        theta = rng.uniform(0.0, 2 * np.pi, n)
        c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
        x, y = xyz[None, :, 0], xyz[None, :, 1]
        z = np.broadcast_to(xyz[None, :, 2], (n, len(xyz)))
        return np.stack([c * x - s * y, s * x + c * y, z], axis=-1)
    if rotation == "3D":
        from scipy.spatial.transform import Rotation
        r = Rotation.random(n, random_state=rng).as_matrix().reshape(n, 3, 3)
        return np.einsum("nij,mj->nmi", r, xyz)
    if rotation == "none":
        return np.broadcast_to(xyz[None], (n, len(xyz), 3)).copy()
    raise ValueError(f"no rotation {rotation!r}: " + ", ".join(ROTATIONS))


def simulate(structures: Sequence[Structure], counts: Sequence[int], window,
             efficiency: Sequence[float], label_unc_nm: Sequence[float], rng,
             z_range: Optional[Tuple[float, float]] = None, rotation: str = "plane"
             ) -> List[np.ndarray]:
    """The observed molecules of each target, (n, 2) -- or (n, 3) with a
    ``z_range`` -- from ``counts[i]`` of each structure placed uniformly in
    ``window`` (anything with ``uniform(n, rng)``, `Window`), turned, blurred
    by the label uncertainty and thinned by the labelling efficiency
    (``StructureSimulator.run``).  A turn in 3D without a ``z_range`` is seen
    from above, its projection."""
    n_t = len(efficiency)
    dims = 3 if z_range is not None else 2
    out: List[List[np.ndarray]] = [[] for _ in range(n_t)]
    for s, n in zip(structures, counts):
        n = int(n)
        if n <= 0:
            continue
        centres = window.uniform(n, rng)
        pts = _rotated(s.xyz, n, rotation, rng)[:, :, :dims]
        pts[:, :, 0] += centres[:, None, 0]
        pts[:, :, 1] += centres[:, None, 1]
        if dims == 3:
            pts[:, :, 2] += rng.uniform(z_range[0], z_range[1], n)[:, None]
        for t in range(n_t):
            mine = s.target == t
            if not mine.any():
                continue
            p = pts[:, mine, :].reshape(-1, dims)
            p = p + rng.normal(0.0, float(label_unc_nm[t]), p.shape)
            seen = int(len(p) * float(efficiency[t]))
            if seen < len(p):
                p = p[rng.choice(len(p), seen, replace=False)]
            out[t].append(p)
    return [np.concatenate(o) if o else np.zeros((0, dims)) for o in out]


# --------------------------------------------------------------- distances

def nn_distances(a: np.ndarray, b: np.ndarray, k: int, same: bool) -> np.ndarray:
    """(len(a), k): the distance from each point of ``a`` to its 1st to k-th
    nearest neighbour in ``b``, a point not its own (``same``); inf where
    ``b`` has fewer (``get_NN_dist``)."""
    from scipy.spatial import cKDTree
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not len(a) or not len(b):
        return np.zeros((0, k))
    # an unbalanced tree builds in a third of the time, and these trees are
    # built once and queried once
    tree = cKDTree(b, balanced_tree=False, compact_nodes=False)
    d, _ = tree.query(a, k=k + int(same))
    d = np.asarray(d).reshape(len(a), -1)
    return d[:, int(same):]


def distances_of(points: Sequence[np.ndarray], orders) -> List[np.ndarray]:
    """For each order ``(t1, t2, k)``, the (n, k) distances of its pair."""
    return [nn_distances(points[t1], points[t2], k, t1 == t2) for t1, t2, k in orders]


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """The two-sample Kolmogorov-Smirnov statistic: the largest gap between
    the two cumulative distributions (``scipy.stats.ks_2samp``'s, without
    its p-value).  NaN when either has no finite value."""
    a = np.sort(np.asarray(a, dtype=float)[np.isfinite(a)])
    b = np.sort(np.asarray(b, dtype=float)[np.isfinite(b)])
    if not len(a) or not len(b):
        return float("nan")
    both = np.concatenate([a, b])
    fa = np.searchsorted(a, both, side="right") / len(a)
    fb = np.searchsorted(b, both, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def nnd_score(simulated: Sequence[np.ndarray], data: Sequence[np.ndarray]) -> float:
    """The KS statistic of every pair of targets and every order, averaged
    (``NND_score``): 0 for the same distributions, 1 for the worst, and 1 when
    nothing can be compared."""
    scores = []
    for d1, d2 in zip(simulated, data):
        if not len(d1) or not len(d2):
            continue
        for j in range(min(d1.shape[1], d2.shape[1])):
            s = ks_statistic(d1[:, j], d2[:, j])
            if np.isfinite(s):
                scores.append(s)
    return float(np.mean(scores)) if scores else 1.0


# ------------------------------------------------------------- the model

@dataclass
class Field:
    """One file's observation: its window, and the observed molecules of each
    target, (n, 2), or (n, 3) with ``z_range`` the depth they fill."""
    window: object
    points: List[np.ndarray]
    z_range: Optional[Tuple[float, float]] = None

    @property
    def n(self) -> int:
        return int(sum(len(p) for p in self.points))


@dataclass
class Model:
    """What a simulation needs besides the counts."""
    structures: List[Structure]
    efficiency: np.ndarray          # per target
    label_unc_nm: np.ndarray        # per target
    rotation: str = "plane"

    def __post_init__(self):
        self.efficiency = np.asarray(self.efficiency, dtype=float).reshape(-1)
        self.label_unc_nm = np.asarray(self.label_unc_nm, dtype=float).reshape(-1)
        if np.any(self.efficiency <= 0) or np.any(self.efficiency > 1):
            raise ValueError("a labelling efficiency is above 0 and at most 1")
        if np.any(self.label_unc_nm < 0):
            raise ValueError("a label uncertainty is not negative")

    @property
    def n_targets(self) -> int:
        return len(self.efficiency)

    def orders(self):
        return neighbour_orders(self.structures, self.n_targets)


def per_target(value, n_targets: int, what: str) -> np.ndarray:
    """One number for every target, or one each: ``0.5`` or ``"0.5, 0.4"``."""
    if isinstance(value, str):
        parts = [p for p in re.split(r"[,;\s]+", value.strip()) if p]
        try:
            values = [float(p) for p in parts]
        except ValueError:
            raise ValueError(f"{what}: {value!r} is not a number or a list of "
                             "them") from None
    else:
        values = list(np.atleast_1d(np.asarray(value, dtype=float)))
    if len(values) == 1:
        values = values * n_targets
    if len(values) != n_targets:
        raise ValueError(f"{what}: one value, or one for each of the {n_targets} "
                         f"targets; there are {len(values)}")
    return np.array(values, dtype=float)


def totals_of(model: Model, fields: Sequence[Field]) -> np.ndarray:
    """The molecules of each target there are, labelled or not: observed over
    the labelling efficiency (``int(n / le)``, as Picasso)."""
    seen = np.array([sum(len(f.points[t]) for f in fields) for t in range(model.n_targets)],
                    dtype=float)
    return np.floor(seen / model.efficiency)


def data_distances(fields: Sequence[Field], orders) -> List[np.ndarray]:
    """The observed distances of each order, the files' pooled."""
    per = [distances_of(f.points, orders) for f in fields]
    return [np.concatenate([p[i] for p in per]) for i in range(len(orders))]


def simulated_distances(model: Model, fields: Sequence[Field], counts: np.ndarray,
                        orders, n_sims: int, rng) -> List[np.ndarray]:
    """The distances of each order over ``n_sims`` simulations of the
    candidate, each file given its share of the structures
    (``get_NN_dist_simulated``)."""
    shares = np.array([f.n for f in fields], dtype=float)
    shares = shares / max(shares.sum(), 1.0)
    out: List[List[np.ndarray]] = [[] for _ in orders]
    counts = np.asarray(counts, dtype=float)
    for _ in range(max(int(n_sims), 1)):
        for f, share in zip(fields, shares):
            pts = simulate(model.structures, np.round(counts * share).astype(np.int64),
                           f.window, model.efficiency, model.label_unc_nm, rng,
                           f.z_range, model.rotation)
            for i, d in enumerate(distances_of(pts, orders)):
                out[i].append(d)
    return [np.concatenate(o) if o else np.zeros((0, orders[i][2]))
            for i, o in enumerate(out)]


@dataclass
class Fit:
    """What `fit` found: the best mixture, its score, what was tried and,
    with a bootstrap, the errors."""
    names: List[str]
    counts: np.ndarray               # (structures,) of the best
    proportions: np.ndarray          # (structures,) %, of the molecules
    score: float
    space: np.ndarray                # every candidate, (n, structures)
    tested: np.ndarray               # the candidates scored, (m, structures)
    scores: np.ndarray               # their scores, (m,)
    passes: np.ndarray               # 0 coarse, 1 fine (or every one), per tested
    tested_proportions: np.ndarray   # (m, structures), %
    errors: Optional[np.ndarray] = None       # % std over the bootstrap
    score_error: Optional[float] = None
    boot_proportions: Optional[np.ndarray] = None
    orders: List[Tuple[int, int, int]] = field(default_factory=list)


def plan(n_space: int, props: np.ndarray, search: str, bootstraps: int) -> int:
    """How many candidates `fit` scores, before it knows the winner: the
    neighbourhoods taken around the grid's middle."""
    if n_space <= 1:
        return 1 + bootstraps
    centre = props[int(np.argmin(np.abs(props - props.mean(axis=0)).sum(axis=1)))]
    near = len(around(props, centre))
    if search == "brute_force":
        n = n_space
    else:
        n = len(farthest_points(props, max(2, int(n_space * COARSE_FRACTION)),
                                NEIGHBOURHOOD_PCT / 2)) + near
    return n + bootstraps * near


def fit(model: Model, fields: Sequence[Field], granularity: int = GRANULARITY,
        search: str = "coarse_to_fine", n_sims: int = N_SIMS,
        bootstraps: int = N_BOOTSTRAPS, seed: int = 0,
        progress: Optional[Callable[[float], None]] = None) -> Fit:
    """The proportions of the model's structures that make the fields'
    nearest-neighbour distances best (``SPINNA.fit_stoichiometry``):
    ``search`` "brute_force" scores every candidate, "coarse_to_fine" a
    spread sample and then the neighbourhood of its winner; ``bootstraps``
    rounds (0 for none) give the errors.  ``progress(fraction)`` is called
    after each candidate, and may raise to stop."""
    orders = model.orders()
    if not orders:
        raise ValueError("no structure has two molecules: nearest neighbours cannot "
                         "tell the structures apart")
    if not any(f.n for f in fields):
        raise ValueError("no molecules to fit")
    rng = np.random.default_rng(seed)
    data = data_distances(fields, orders)
    space = search_space(model.structures, totals_of(model, fields), granularity)
    props = proportions(space, model.structures)
    planned = max(plan(len(space), props, search, bootstraps), 1)
    done = [0]

    def score_all(rows: np.ndarray, against) -> np.ndarray:
        out = np.empty(len(rows))
        for i, counts in enumerate(rows):
            sim = simulated_distances(model, fields, counts, orders, n_sims, rng)
            out[i] = nnd_score(sim, against)
            done[0] += 1
            if progress is not None:
                progress(min(done[0] / planned, 1.0))
        return out

    if len(space) == 1 or search == "brute_force":
        tested = np.arange(len(space))
        scores = score_all(space, data)
        passes = np.ones(len(space), dtype=np.int64)
        best = tested[int(np.argmin(scores))]
    elif search == "coarse_to_fine":
        coarse = farthest_points(props, max(2, int(len(space) * COARSE_FRACTION)),
                                 NEIGHBOURHOOD_PCT / 2)
        coarse_scores = score_all(space[coarse], data)
        winner = coarse[int(np.argmin(coarse_scores))]
        fine = around(props, props[winner])
        # the plan guessed the neighbourhoods' size; now one is known
        planned = done[0] + len(fine) * (1 + bootstraps)
        fine_scores = score_all(space[fine], data)
        best = fine[int(np.argmin(fine_scores))]
        tested = np.concatenate([coarse, fine])
        scores = np.concatenate([coarse_scores, fine_scores])
        passes = np.concatenate([np.zeros(len(coarse), np.int64),
                                 np.ones(len(fine), np.int64)])
    else:
        raise ValueError(f"no search {search!r}: brute_force or coarse_to_fine")
    best_score = float(scores[np.flatnonzero(tested == best)[-1]])
    result = Fit(names=[s.name for s in model.structures], counts=space[best].copy(),
                 proportions=props[best].copy(), score=best_score, space=space,
                 tested=space[tested], scores=scores, passes=passes,
                 tested_proportions=props[tested], orders=orders)
    if bootstraps > 0:
        near = around(props, props[best])
        planned = done[0] + bootstraps * len(near)
        found, found_scores = [], []
        for _ in range(int(bootstraps)):
            fake = simulated_distances(model, fields, space[best], orders, 1, rng)
            s = score_all(space[near], fake)
            i = int(np.argmin(s))
            found.append(props[near[i]])
            found_scores.append(s[i])
        result.boot_proportions = np.array(found)
        result.errors = np.std(result.boot_proportions, axis=0)
        result.score_error = float(np.std(found_scores))
    return result


# ---------------------------------------------------- labelling efficiency

def efficiency_structures(distance_nm: float) -> List[Structure]:
    """A alone, B alone, and A with B at ``distance_nm`` (``fit_le``)."""
    d = float(distance_nm) / 2
    return [Structure("A alone", [[0, 0, 0]], [0]), Structure("B alone", [[0, 0, 0]], [1]),
            Structure(f"A-B {distance_nm:g} nm", [[-d, 0, 0], [d, 0, 0]], [0, 1])]


def efficiency_from_proportions(props: np.ndarray) -> np.ndarray:
    """The labelling efficiencies of A and B from the proportions (%) of A
    alone, B alone and A-B (``get_le_from_props``): A's is the fraction of
    the B molecules that have an A partner, B's the other way round.  Works
    on (3,) or (n, 3)."""
    p = np.asarray(props, dtype=float)
    a, b, pair = p[..., 0], p[..., 1], p[..., 2] / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([np.where(b + pair > 0, pair / (b + pair), np.nan),
                         np.where(a + pair > 0, pair / (a + pair), np.nan)], axis=-1)


@dataclass
class EfficiencyFit:
    """What `fit_efficiency` found: the efficiencies of A and B, the tag
    distance that fitted best, and that distance's `Fit`."""
    efficiency: np.ndarray                 # (2,)
    errors: Optional[np.ndarray]           # (2,) or None
    distance_nm: float
    fit: Fit
    scores_by_distance: List[Tuple[float, float]]
    model: Model


def fit_efficiency(fields: Sequence[Field], distances_nm: Sequence[float],
                   label_unc_nm, granularity: int = GRANULARITY,
                   search: str = "coarse_to_fine", n_sims: int = N_SIMS,
                   bootstraps: int = N_BOOTSTRAPS, seed: int = 0, rotation: str = "plane",
                   progress: Optional[Callable[[float], None]] = None) -> EfficiencyFit:
    """The labelling efficiency of two tags A and B on one protein (Hellmeier
    et al. 2024, ``fit_le``): fitted with every label detected, as A alone,
    B alone and A-B at each of ``distances_nm`` -- the one that fits best is
    kept -- and the proportions read as efficiencies.  The bootstrap's
    proportions give the efficiencies' errors."""
    distances_nm = [float(d) for d in distances_nm]
    if not distances_nm:
        raise ValueError("a tag distance to try")
    best: Optional[Tuple[Fit, float, Model]] = None
    tried = []
    for i, d in enumerate(distances_nm):
        model = Model(efficiency_structures(d), [1.0, 1.0],
                      per_target(label_unc_nm, 2, "label uncertainty"), rotation)

        def step(fraction, i=i):
            if progress is not None:
                progress((i + fraction) / len(distances_nm))
        f = fit(model, fields, granularity, search, n_sims, bootstraps, seed + i, step)
        tried.append((d, f.score))
        if best is None or f.score < best[0].score:
            best = (f, d, model)
    f, d, model = best
    le = efficiency_from_proportions(f.proportions)
    errors = None
    if f.boot_proportions is not None:
        errors = np.nanstd(efficiency_from_proportions(f.boot_proportions), axis=0)
    return EfficiencyFit(le, errors, d, f, tried, model)


# ----------------------------------------------------------------- figures

def ordinal(k: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")


def draw_histograms(ax, data: np.ndarray, simulated: Optional[np.ndarray] = None,
                    title: str = "", bin_nm: float = HIST_BIN_NM,
                    upto: Optional[float] = None) -> None:
    """One pair of targets: the data's 1st to k-th nearest-neighbour
    distances as bars, the simulation's as lines in the same colours
    (``plot_NN``), each normalised to its area."""
    data = np.asarray(data, dtype=float).reshape(len(data), -1)
    finite = data[np.isfinite(data)]
    if upto is None:
        upto = float(np.percentile(finite, 99)) if len(finite) else 200.0
    edges = np.arange(0.0, max(upto, bin_nm) + bin_nm, bin_nm)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for j in range(data.shape[1]):
        colour = NN_COLOURS[j % len(NN_COLOURS)]
        d = data[:, j][np.isfinite(data[:, j])]
        if len(d):
            h, _ = np.histogram(d, edges, density=False)
            ax.bar(mid, h / (len(d) * bin_nm), width=bin_nm, color=colour, alpha=0.55,
                   edgecolor="none", label=f"data, {ordinal(j + 1)}")
        if simulated is not None and simulated.shape[1] > j:
            s = simulated[:, j][np.isfinite(simulated[:, j])]
            if len(s):
                h, _ = np.histogram(s, edges)
                ax.plot(mid, h / (len(s) * bin_nm), color=colour, linewidth=1.4,
                        label=f"simulated, {ordinal(j + 1)}")
    ax.set_xlim(0, edges[-1])
    ax.set_xlabel("distance (nm)")
    ax.set_ylabel("frequency (1/nm)")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)


def draw_scores(ax, result: Fit) -> None:
    """The score of every candidate tried against its proportions: against
    the second structure's share for two structures, against each one's
    (a colour each) for more; the best ringed, and its bootstrap error as a
    bar."""
    shown = result.tested_proportions
    n_s = shown.shape[1]
    which = [1] if n_s == 2 else list(range(n_s))
    two_passes = bool((result.passes == 0).any())
    for j in which:
        colour = NN_COLOURS[j % len(NN_COLOURS)] if n_s > 2 else "0.3"
        for p, marker, label in ((0, "x", "coarse"), (1, "o", "fine" if two_passes else "")):
            sel = result.passes == p
            if not sel.any():
                continue
            name = f"{result.names[j]}, {label}" if n_s > 2 and label else (
                result.names[j] if n_s > 2 else label)
            ax.plot(shown[sel, j], result.scores[sel], marker, color=colour, markersize=4,
                    alpha=0.8, linestyle="none", label=name or None)
        best = result.proportions[j]
        ax.plot([best], [result.score], "o", markersize=10, markerfacecolor="none",
                markeredgecolor="#d62728")
        if result.errors is not None:
            ax.errorbar([best], [result.score], xerr=[result.errors[j]], color="#d62728",
                        capsize=3)
    ax.set_xlabel(f"{result.names[1]} (% of the molecules)" if n_s == 2
                  else "share of the molecules (%)")
    ax.set_ylabel("KS statistic (lower fits better)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, frameon=False)
