"""MINFLUX attempts as Imspector's m2410 records: every iteration of every
attempt, for testing and explaining what is read from them (*Sequence QC*,
`minflux_qc`) with a known truth.

Not a physical simulation of the beam pattern.  The scouting iteration
finds a molecule with probability ``passed[0]``; each catching iteration
passes with its ``passed[k]`` and otherwise ends for a reason drawn from
``reasons[k]`` (dark, cfr, out of range); an attempt that reaches the
final iteration gives a burst of localizations whose number is geometric
with mean ``locs_per_burst``, ending dark (or out of range).  A catching
iteration's position scatters about the molecule by ``catching_nm[k]``, a
localization's by ``precision_nm``; photons are Poisson about
``photons[k]``; times run on at ``dwell_s`` per row.  The records go through
`io.minflux.table` like a file's.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

#: the m2410 record, the fields `io.minflux` and `minflux_qc` read
DTYPE = [("vld", "?"), ("fnl", "?"), ("bot", "?"), ("eot", "?"), ("sta", "u1"),
         ("tim", "f8"), ("tid", "u4"), ("gri", "u4"), ("thi", "u1"), ("sqi", "u1"),
         ("itr", "i4"), ("loc", "f8", (3,)), ("lnc", "f8", (3,)), ("eco", "u4"),
         ("ecc", "u4"), ("efo", "f4"), ("efc", "f4"), ("fbg", "f4"), ("cfr", "f2"),
         ("dcr", "f2", (2,))]
#: m2410's codes (abberior's wiki): valid, valid final, cfr, no signal, out of range
STA = {"valid": 2, "final": 1, "cfr": 6, "dark": 8, "out of range": 9}


def simulate_attempts(n_events: int = 2000, passed: Sequence[float] = (0.05, 0.9, 0.6),
                      reasons: Optional[Sequence[Dict[str, float]]] = None,
                      locs_per_burst: float = 15.0,
                      photons: Sequence[float] = (2.0, 60.0, 60.0, 25.0),
                      catching_nm: Sequence[float] = (60.0, 20.0, 8.0),
                      precision_nm: float = 4.0, field_nm: float = 5000.0,
                      dwell_s: float = 1e-3, cfr: float = 0.3,
                      seed: Optional[int] = 0) -> np.ndarray:
    """``n_events`` scouting events as m2410 records, sorted by time (see the
    module docstring).  ``passed`` holds one probability per iteration before
    the final one; ``reasons[k]`` the share of each reason an attempt ends in
    iteration k (default: dark in scouting, then 80 % dark and 20 % cfr in
    the last catching iteration, dark otherwise).  cfr is measured in the
    last catching iteration, as in abberior's 2D sequence."""
    rng = np.random.default_rng(seed)
    n_catch = len(passed)
    final_itr = n_catch
    if reasons is None:
        reasons = [{"dark": 1.0}] * (n_catch - 1) + [{"dark": 0.8, "cfr": 0.2}]
    rows = []
    t = 0.0
    for tid in range(1, n_events + 1):
        where = rng.uniform(0, field_nm, 2)
        for k in range(n_catch):
            ok = rng.random() < passed[k]
            eco = int(rng.poisson(photons[k]))
            if ok:
                xy = where + rng.normal(0, catching_nm[k], 2)
                c = cfr + rng.normal(0, 0.05) if k == n_catch - 1 else 0.0
                rows.append((True, False, k == 0, False, STA["valid"], t, tid, k, xy,
                             eco, c))
                t += dwell_s
                continue
            names = list(reasons[k])
            reason = rng.choice(names, p=np.asarray([reasons[k][r] for r in names]) /
                                sum(reasons[k].values()))
            rows.append((False, False, k == 0, True, STA[reason], t, tid, k, None, eco, 0.0))
            t += dwell_s
            break
        else:
            n = int(rng.geometric(1.0 / max(locs_per_burst, 1.0)))
            for j in range(n):
                xy = where + rng.normal(0, precision_nm, 2)
                rows.append((True, True, False, False, STA["final"], t, tid, final_itr, xy,
                             int(rng.poisson(photons[final_itr])), 0.0))
                t += dwell_s
            rows.append((False, False, False, True, STA["dark"], t, tid, final_itr, None,
                         int(rng.poisson(1.0)), 0.0))
            t += dwell_s
    out = np.zeros(len(rows), dtype=DTYPE)
    for i, (vld, fnl, bot, eot, sta, tim, tid, itr, xy, eco, c) in enumerate(rows):
        pos = (np.nan, np.nan, np.nan) if xy is None else (xy[0] * 1e-9, xy[1] * 1e-9, 0.0)
        out[i] = (vld, fnl, bot, eot, sta, tim, tid, 0, 0, 0, itr, pos, pos, eco, 0,
                  eco / 1e-3 if vld else 0.0, 0.0, np.nan, c, (np.nan, np.nan))
    return out
