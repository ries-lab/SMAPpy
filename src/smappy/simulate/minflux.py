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

On request the molecule walks during its burst, as a motor does: steps of
``step_nm`` along a direction drawn per attempt, after dwells drawn from a
gamma distribution of mean ``step_dwell_s`` and shape ``step_shape`` (2 for
kinesin's two events per step).  A share ``two_molecules`` of the attempts
catch two molecules at once: twice the photons in the final iteration and a
cfr about ``two_cfr`` -- the pair's centre, where one beam pattern sees
both -- which the instrument aborts above ``cfr_limit`` (the sequence's
``ccrLimit``).

`write_msr` writes measurements of these records, with a sequence in the
metadata, as an `.msr` file (`io.msr_write`), so that a simulation opens
the way a measurement does: the MINFLUX tutorial's data.
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
                      step_nm: float = 0.0, step_dwell_s: float = 0.03,
                      step_shape: float = 2.0, two_molecules: float = 0.0,
                      two_cfr: float = 0.65, cfr_limit: Optional[float] = None,
                      t0: float = 0.0, seed: Optional[int] = 0) -> np.ndarray:
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
    t = t0
    for tid in range(1, n_events + 1):
        where = rng.uniform(0, field_nm, 2)
        two = two_molecules > 0 and rng.random() < two_molecules
        for k in range(n_catch):
            ok = rng.random() < passed[k]
            eco = int(rng.poisson(photons[k]))
            if ok:
                xy = where + rng.normal(0, catching_nm[k], 2)
                c = ((two_cfr if two else cfr) + rng.normal(0, 0.05)
                     if k == n_catch - 1 else 0.0)
                if cfr_limit is not None and c > cfr_limit:
                    rows.append((False, False, k == 0, True, STA["cfr"], t, tid, k, None,
                                 eco, c))
                    t += dwell_s
                    break
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
            gain = 2.0 if two else 1.0
            if step_nm:
                angle = rng.uniform(0, 2 * np.pi)
                heading = np.array([np.cos(angle), np.sin(angle)])
                walked = 0.0
                next_step = t + rng.gamma(step_shape, step_dwell_s / step_shape)
            for j in range(n):
                if step_nm:
                    while t >= next_step:
                        walked += step_nm
                        next_step += rng.gamma(step_shape, step_dwell_s / step_shape)
                    centre = where + walked * heading
                else:
                    centre = where
                xy = centre + rng.normal(0, precision_nm, 2)
                rows.append((True, True, False, False, STA["final"], t, tid, final_itr, xy,
                             int(rng.poisson(gain * photons[final_itr])), 0.0))
                t += dwell_s
            rows.append((False, False, False, True, STA["dark"], t, tid, final_itr, None,
                         int(rng.poisson(1.0)), 0.0))
            t += dwell_s
    out = np.zeros(len(rows), dtype=DTYPE)
    for i, (vld, fnl, bot, eot, sta, tim, tid, itr, xy, eco, c) in enumerate(rows):
        pos = (np.nan, np.nan, np.nan) if xy is None else (xy[0] * 1e-9, xy[1] * 1e-9, 0.0)
        out[i] = (vld, fnl, bot, eot, sta, tim, tid, 0, 0, 0, itr, pos, pos, eco, 0,
                  eco / dwell_s if vld else 0.0, 0.0, np.nan, c, (np.nan, np.nan))
    return out


# ------------------------------------------------------------------ .msr

#: a 2D tracking sequence: the scouting iteration, two catching iterations
#: (cfr measured in the second, ``ccrLimit``), the final one -- the
#: parameters `io.minflux`, *Sequence table* and the tutorial read, with
#: values like those of abberior's 2D tracking sequence
SEQUENCE_2D = [
    {"Mode": {"id": "prbt", "pattern": "hexagon", "dim": 2}, "patGeoFactor": 0.79,
     "phtLimit": 40, "ccrLimit": -1, "bgcThreshold": 15000, "power": 2},
    {"Mode": {"id": "mflx", "pattern": "hexagon", "dim": 2}, "patGeoFactor": 0.84,
     "phtLimit": 20, "ccrLimit": -1, "bgcThreshold": 30000, "power": 2},
    {"Mode": {"id": "mflx", "pattern": "hexagon", "dim": 2}, "patGeoFactor": 0.42,
     "phtLimit": 20, "ccrLimit": 0.8, "bgcThreshold": 30000, "power": 4},
    {"Mode": {"id": "mflx", "pattern": "hexagon", "dim": 2}, "patGeoFactor": 0.28,
     "phtLimit": 20, "ccrLimit": -1, "bgcThreshold": 50000, "power": 6},
]


def sequence_attrs(final_power: float, date: str, name: str = "Tracking_2D") -> dict:
    """A measurement's ``zarr/mfx/.zattrs``: one thread, `SEQUENCE_2D` with
    the final iteration's laser power ``final_power``, the 640 nm MINFLUX
    laser and the 405 nm activation laser."""
    iterations = []
    for k, it in enumerate(SEQUENCE_2D):
        power = final_power if k == len(SEQUENCE_2D) - 1 else it["power"]
        iterations.append({
            "Mode": it["Mode"], "patGeoFactor": it["patGeoFactor"], "patDwellTime": 1e-4,
            "patRepeat": 1, "phtLimit": it["phtLimit"], "ccrLimit": it["ccrLimit"],
            "bgcThreshold": it["bgcThreshold"], "maxOffTime": 3e-3,
            "_excitation": {"laser": "MINFLUX640", "power": power, "wavelength": 6.4e-7},
            "_activation": {"laser": "ACT405"}, "_detection": {"channels": ["APD1", "APD2"]}})
    lasers = [{"id": "ACT405", "calibration": "ACT 405", "wavelength": 4.05e-7, "purpose": 0},
              {"id": "MINFLUX640", "calibration": "MF 640", "wavelength": 6.4e-7,
               "purpose": 1}]
    detectors = [{"id": "APD1", "expcontrol_name": "Cy5 near"},
                 {"id": "APD2", "expcontrol_name": "Cy5 far"}]
    return {"acquisition_date": date, "version": "2.1",
            "measurement": {"dimensionality": 2, "threads": [{
                "sequences": [{"id": name, "Itr": iterations, "locLimit": -1,
                               "stickiness": 4}],
                "lasers": lasers, "detectors": detectors}]}}


def write_msr(path, measurements) -> None:
    """An `.msr` of MINFLUX measurements, each ``(label, records, attrs)``:
    m2410 records (`simulate_attempts`) and their ``.zattrs``
    (`sequence_attrs`)."""
    from ..io import msr_write
    write_obf = msr_write.write_obf
    stacks = []
    for label, rec, attrs in measurements:
        xy = rec["loc"][:, :2][np.isfinite(rec["loc"][:, 0])]
        corners = ([xy.min(0).tolist(), xy.max(0).tolist()] if len(xy)
                   else [[0.0, 0.0], [1e-5, 1e-5]])
        stacks.append(msr_write.measurement_stack(label, rec, attrs, roi_m=corners,
                                                  chunk=4096))
    write_obf(path, stacks)


def motor_msr(path, conditions: Sequence = ((1000, 0.03, 2.0), (20, 0.1, 1.0)),
              n_events: int = 2500, seed: int = 0):
    """The MINFLUX tutorial's file: kinesin-like motors stepping 8 nm, tracked
    by the 2D sequence, a measurement per ``(ATP in µM, mean dwell in s,
    dwell shape)`` of ``conditions``, a minute apart, labelled ``kinesin
    <ATP>uM ATP``.  At saturating ATP two events of similar rate set the
    pace of a step (shape 2); where ATP is scarce its binding alone does
    (shape 1), and the dwell is longer.  A tenth of the attempts catch two
    molecules."""
    measurements = []
    for k, (atp, dwell, shape) in enumerate(conditions):
        rec = simulate_attempts(
            n_events, passed=(0.05, 0.9, 0.7), locs_per_burst=1000.0,
            photons=(3.0, 40.0, 40.0, 50.0), catching_nm=(60.0, 20.0, 8.0),
            precision_nm=3.5, field_nm=8000.0, dwell_s=5e-4, cfr=0.25,
            step_nm=8.0, step_dwell_s=dwell, step_shape=shape,
            two_molecules=0.1, two_cfr=0.6, cfr_limit=0.8,
            seed=None if seed is None else seed + k)
        date = f"2026-06-11T15:{8 + k:02d}:27+02:00"
        measurements.append((f"kinesin {atp:g}uM ATP", rec, sequence_attrs(6.0, date)))
    write_msr(path, measurements)
    return path
