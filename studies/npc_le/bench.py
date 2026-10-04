"""Test bench for the NPC labelling efficiency: which way of telling a gap
from a corner is least biased by imprecise and repeated localizations?

Simulates Nup96 pores (tilted up to 10 degrees), links them (always grouped,
as on real data), segments them with `ROIManager/Segment/NPC`, and scores
several corner counters against the truth of the same pores:

* ``labelled``: the corners with at least one labelled copy (the recipe);
* ``detected``: the corners with at least one copy that left a localization
  in the grouped table -- what a perfect counter could see.

The efficiency is fitted the way the plugin fits it (likelihood, 3..8).

    python studies/npc_le/bench.py            # the grid, writes results.csv
    python studies/npc_le/bench.py --quick    # one seed, fewer conditions
"""
from __future__ import annotations

import argparse
import csv
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from smappy.group import group
from smappy.plugins.npc import (LabelingEfficiencySettings, NPCSegmentSettings,
                                PRECISION_OF_ARC, fit_circle, labeling_efficiency,
                                segment_npcs)
from smappy.simulate import ground_truth, simulate
from smappy.simulate.settings import (BlinkingSettings, LabellingSettings,
                                      SimulationSettings, StructureSettings)

HERE = Path(__file__).resolve().parent
STRUCTURE = HERE / "npc_tilt.yaml"
CORNERS = 8
STEP = 2 * np.pi / CORNERS
R, DR = 50.0, 20.0                 # NPC Corners' defaults
WINDOW = 150.0                     # the localizations of a site, nm
SPOKE = 5.6 / 53.7                 # half the angle between a spoke's two copies
FRAMES_PER_BLINK = 1000


# --------------------------------------------------------------- the data

@dataclass
class Pore:
    theta: np.ndarray              # angle of each ring localization, fitted centre
    s: np.ndarray                  # its angular precision, sigma / rho
    kept: np.ndarray               # passes the precision filter
    blink: np.ndarray              # which grouped row (for subsampling)
    phase_fit: float
    phase_true: float
    sigma: np.ndarray              # precision of each ring localization, nm
    sigma_window: np.ndarray       # precision of every localization near the pore
    rho_near: np.ndarray           # distance from the fitted centre, within WINDOW
    sigma_near: np.ndarray         # precision of the same
    theta_true_centre: np.ndarray  # same localizations, around the true centre
    s_true_centre: np.ndarray
    kept_true_centre: np.ndarray
    labelled: int
    detected: int


def settings_for(efficiency, photons, blinks, seed) -> SimulationSettings:
    # As the experiments are run: imaged until every fluorophore has bleached
    # ("every blink": a geometric number of blinks, mean `blinks`, bleaching
    # 1 / blinks), and the activation raised so that the blinks are spread
    # evenly -- with the stack as long as the blinks need, so the density of
    # emitters in a frame is the same for every condition.
    return SimulationSettings(
        n_frames=FRAMES_PER_BLINK * int(np.ceil(blinks)), seed=seed,
        structure=StructureSettings(file=str(STRUCTURE)),
        labelling=LabellingSettings(efficiency=efficiency),
        blinking=BlinkingSettings(blinks=blinks, photons=photons,
                                  photons_std=photons / 2, activation="all"))


def circular_phase(theta, weight) -> float:
    return float(np.angle(np.sum(weight * np.exp(1j * CORNERS * theta))) / CORNERS)


def ring(x, y, sigma, centre):
    rho = np.hypot(x - centre[0], y - centre[1])
    theta = np.arctan2(y - centre[1], x - centre[0])
    inside = (rho > R - DR) & (rho < R + DR)
    s = sigma / np.maximum(rho, 1e-9)
    kept = sigma < PRECISION_OF_ARC * STEP * R
    return theta[inside], s[inside], kept[inside], inside


def corner_of(theta, phase) -> np.ndarray:
    return np.floor(np.mod(theta - phase + STEP / 2, 2 * np.pi) / STEP).astype(int)


GROUPING = "precision"            # or "smappy": the fixed 50 nm of smappy.group
LINK_SIGMAS, LINK_MAX_NM = 3.0, 200.0


def group_by_precision(locs):
    """Link localizations in consecutive frames when they are closer than
    3 sqrt(s1^2 + s2^2) (and 200 nm): a dim frame at the end of a blink has
    a precision of 100 nm and more and lands beyond smappy's fixed 50 nm, so
    one blink became 1.5 rows at 500 photons.  Positions are combined with
    inverse-variance weights, precisions as 1 / sqrt(sum 1 / s^2); the truth
    columns come from the most precise localization of the group."""
    from scipy.spatial import cKDTree
    frame = np.asarray(locs["frame"], np.int64)
    x = np.asarray(locs["x_nm"], float)
    y = np.asarray(locs["y_nm"], float)
    s = np.asarray(locs["xy_err_nm"], float)
    order = np.argsort(frame, kind="stable")
    parent = np.arange(len(x))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    starts = np.searchsorted(frame[order], np.unique(frame))
    bounds = dict(zip(np.unique(frame), zip(starts, list(starts[1:]) + [len(x)])))
    for f, (a, b) in bounds.items():
        if f + 1 not in bounds:
            continue
        here = order[a:b]
        c, d = bounds[f + 1]
        there = order[c:d]
        tree = cKDTree(np.column_stack((x[there], y[there])))
        for i in here:
            for j in tree.query_ball_point((x[i], y[i]), LINK_MAX_NM):
                k = there[j]
                dist = np.hypot(x[i] - x[k], y[i] - y[k])
                if dist < LINK_SIGMAS * np.hypot(s[i], s[k]):
                    parent[root(k)] = root(i)
    ids = np.array([root(i) for i in range(len(x))])
    _, gid = np.unique(ids, return_inverse=True)
    w = 1 / s ** 2
    wsum = np.bincount(gid, w)
    columns = {"x_nm": np.bincount(gid, w * x) / wsum,
               "y_nm": np.bincount(gid, w * y) / wsum,
               "xy_err_nm": 1 / np.sqrt(wsum),
               "photons": np.bincount(gid, np.asarray(locs["photons"], float)),
               "frame": np.minimum.reduceat(frame[np.argsort(gid, kind="stable")],
                                            np.r_[0, np.cumsum(np.bincount(gid))[:-1]])}
    best = np.full(gid.max() + 1, -1)
    rank = np.argsort(-w, kind="stable")             # most precise first
    for i in rank[::-1]:
        best[gid[i]] = i
    for name in ("emitter", "copy"):
        columns[name] = np.asarray(locs[name])[best]
    from smappy.locs import Localizations
    return Localizations(columns, dict(locs.metadata)), gid + 1


def pores_of(settings: SimulationSettings, segment: NPCSegmentSettings = None,
             truth_only: bool = True) -> List[Pore]:
    """The pores the segmenter finds.  By default with *min localizations* 3
    and only the sites on a true pore (the counting under study, not the
    segmentation); ``segment=NPCSegmentSettings(), truth_only=False`` is the
    pipeline as a user runs it, junk sites included (their truth fields are
    those of the nearest pore and mean nothing)."""
    locs = simulate(settings)
    grouped, _ = group_by_precision(locs) if GROUPING == "precision" else group(locs)
    truth = ground_truth(settings)
    poses = locs.metadata["copies"]
    centres = np.column_stack((poses["x_nm"], poses["y_nm"]))
    # in-plane turn of each pore; the tilt (beta) is up to 10 degrees, small
    turn = np.radians(np.asarray(poses["alpha_deg"]) + np.asarray(poses["gamma_deg"]))
    f = truth.fluorophores
    f_corner = np.full(len(f), -1)
    for c in range(len(centres)):
        mine = f.copy == c
        th = np.arctan2(f.xyz[mine, 1] - centres[c, 1], f.xyz[mine, 0] - centres[c, 0])
        f_corner[mine] = corner_of(th, turn[c])

    # the precision of every localization in the field, which is what a
    # cutoff's pass fraction q is measured on
    pores_of.sigma_all = np.asarray(grouped["xy_err_nm"], float)
    x = np.asarray(grouped["x_nm"], float)
    y = np.asarray(grouped["y_nm"], float)
    sigma = np.asarray(grouped["xy_err_nm"], float)
    emitter = np.asarray(grouped["emitter"]).round().astype(int)
    copy = np.asarray(grouped["copy"]).round().astype(int)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack((x, y)))
    out = []
    # 3 on the ring, not 10: one row per blink, and at one blink a pore at
    # low efficiency has only a dozen.  The segmenter is not under study, so a
    # site counts only when it is on a true pore (within 20 nm, one per pore):
    # at 500 photons it also proposes as many sites again on scattered
    # imprecise localizations, which `pores_of.junk` counts
    pores_of.junk = 0
    taken = set()
    from scipy.spatial import cKDTree as _Tree
    true_tree = _Tree(centres)
    if segment is None:
        segment = NPCSegmentSettings(min_locs=3)
    pores_of.found = 0
    for site in segment_npcs(grouped, segment):
        if not site["use"]:
            continue
        distance, c = true_tree.query(site["center"])
        if distance > 20 or c in taken:
            pores_of.junk += 1
            if truth_only:
                continue
        else:
            taken.add(c)
            pores_of.found += 1
        near = np.asarray(tree.query_ball_point(site["center"], WINDOW))
        xs, ys, ss = x[near], y[near], sigma[near]
        centre = fit_circle(xs, ys, site["center"], radius=R, scale=DR)[:2]
        theta, s, kept, inside = ring(xs, ys, ss, centre)
        close = np.hypot(xs - centre[0], ys - centre[1]) < R + 3 * DR
        theta_t, s_t, kept_t, _ = ring(xs, ys, ss, centres[c])
        if not kept.any():
            continue
        w = 1.0 / np.maximum(s[kept], 1e-6) ** 2
        mine = f.copy == c
        own = near[copy[near] == c]
        out.append(Pore(
            theta=theta, s=s, kept=kept, blink=near[inside],
            sigma=ss[inside], sigma_window=ss[close],
            rho_near=np.hypot(xs - centre[0], ys - centre[1]), sigma_near=ss,
            phase_fit=circular_phase(theta[kept], w),
            phase_true=float(np.mod(turn[c] + STEP / 2, STEP) - STEP / 2),
            theta_true_centre=theta_t, s_true_centre=s_t, kept_true_centre=kept_t,
            labelled=len(np.unique(f_corner[mine])),
            detected=len(np.unique(f_corner[emitter[own]]))))
    return out


# ------------------------------------------------------------- counters

def hard(theta, s, kept, phase) -> int:
    """The plugin's rule: a segment with a kept localization is a corner."""
    return len(np.unique(corner_of(theta[kept], phase)))


def hard_all(theta, s, kept, phase) -> int:
    """The plugin's rule without its precision filter."""
    return len(np.unique(corner_of(theta, phase)))


def dead_zone(k: float, filtered: bool = True) -> Callable:
    """Ignore a localization within k of its precision of a segment border --
    itself a filter that adapts to the precision, so it may go without the
    fixed one."""
    def count(theta, s, kept, phase):
        offset = np.mod(theta - phase + STEP / 2, STEP) - STEP / 2
        sure = (kept if filtered else True) & (np.abs(offset) < STEP / 2 - k * s)
        return len(np.unique(corner_of(theta[sure], phase)))
    return count


def evidence(theta, s, phase) -> np.ndarray:
    """Each localization shared among the corners by its probability of
    coming from each, its precision widened by a spoke's spread; summed per
    corner."""
    width = np.sqrt(s ** 2 + SPOKE ** 2)[:, None]
    centres = phase + STEP * np.arange(CORNERS)
    d = np.mod(theta[:, None] - centres[None, :] + np.pi, 2 * np.pi) - np.pi
    m = np.exp(-0.5 * (d / width) ** 2)
    m /= np.maximum(m.sum(axis=1, keepdims=True), 1e-300)
    return m.sum(axis=0)


def soft(t: float, filtered: bool = True) -> Callable:
    def count(theta, s, kept, phase):
        use = kept if filtered else np.ones_like(kept)
        return int(np.sum(evidence(theta[use], s[use], phase) >= t))
    return count


def relative(f: float, floor: float = 0.5, filtered: bool = True) -> Callable:
    """A corner is seen when its evidence is at least f of the pore's typical
    occupied corner: the threshold grows with the localizations per corner,
    which is what a stray does not."""
    def count(theta, s, kept, phase):
        use = kept if filtered else np.ones_like(kept)
        e = evidence(theta[use], s[use], phase)
        occupied = e[e >= floor]
        if not len(occupied):
            return 0
        return int(np.sum(e >= max(floor, f * np.median(occupied))))
    return count


METHODS: Dict[str, Callable] = {
    "hard": hard,
    "hard all": hard_all,
    "dead k=0.5": dead_zone(0.5),
    "dead k=1": dead_zone(1.0),
    "dead k=2": dead_zone(2.0),
    "dead k=0.5 all": dead_zone(0.5, filtered=False),
    "dead k=1 all": dead_zone(1.0, filtered=False),
    "soft t=0.5": soft(0.5),
    "soft t=1": soft(1.0),
    "relative f=0.2": relative(0.2),
    "relative f=0.3": relative(0.3),
    "relative f=0.3 all": relative(0.3, filtered=False),
}


def counts(pores: List[Pore], method: Callable, centre: str = "fit",
           phase: str = "fit") -> np.ndarray:
    out = []
    for p in pores:
        theta, s, kept = ((p.theta, p.s, p.kept) if centre == "fit" else
                          (p.theta_true_centre, p.s_true_centre, p.kept_true_centre))
        out.append(method(theta, s, kept, p.phase_fit if phase == "fit" else p.phase_true))
    return np.asarray(out)


def ele(n) -> float:
    try:
        return labeling_efficiency(n, LabelingEfficiencySettings(bootstrap=0))["efficiency"]
    except ValueError:
        return float("nan")


# ------------------------------------------------------------------- grid

GRID = {"efficiency": (0.2, 0.35, 0.5, 0.7), "photons": (500, 5000),
        "blinks": (1, 3, 10)}
SEEDS = (1, 2, 3)


def run(grid=GRID, seeds=SEEDS, out=HERE / "results.csv") -> None:
    rows = []
    for efficiency, photons, blinks in itertools.product(*grid.values()):
        for seed in seeds:
            start = time.time()
            pores = pores_of(settings_for(efficiency, photons, blinks, seed))
            labelled = np.array([p.labelled for p in pores])
            detected = np.array([p.detected for p in pores])
            base = {"efficiency": efficiency, "photons": photons, "blinks": blinks,
                    "seed": seed, "pores": len(pores), "junk": pores_of.junk,
                    "ele_labelled": ele(labelled), "ele_detected": ele(detected)}
            variants = [(name, m, "fit", "fit") for name, m in METHODS.items()]
            variants += [("hard true phase", hard, "fit", "true"),
                         ("hard true centre+phase", hard, "true", "true")]
            for name, method, centre, phase in variants:
                n = counts(pores, method, centre, phase)
                rows.append({**base, "method": name, "ele": ele(n),
                             "corners_minus_detected": float(np.mean(n - detected)),
                             "over": int(np.sum(n > detected)),
                             "under": int(np.sum(n < detected))})
            print(f"LE {efficiency} photons {photons} blinks {blinks} seed {seed}: "
                  f"{len(pores)} pores, {time.time() - start:.1f} s", flush=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        run({"efficiency": (0.35,), "photons": (500, 5000), "blinks": (1, 10)},
            seeds=(1,), out=HERE / "quick.csv")
    else:
        run()
