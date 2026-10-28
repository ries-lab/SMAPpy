"""COMET's precision weights, its width in z and its error bar, on simulations.

    python studies/drift_weights/study.py weights     # the per-axis options against each other
    python studies/drift_weights/study.py target      # the same at three target sigmas
    python studies/drift_weights/study.py driftless   # the curve where there is no drift

Each prints the rms error of the drift curve against the one put in (both
made zero mean), per axis, over the seeds, and the noise the error bar
reports.  The results are in docs/drift.md, "Precision weights and the width
in z" and "Wavy drift where there is none"; README.md says how the data are
made.
"""
import collections
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from smappy.drift import DriftSettings, estimate_drift
from smappy.locs import Localizations
from smappy.simulate import SimulationSettings, simulate
from smappy.simulate.settings import BlinkingSettings, StructureSettings

HERE = Path(__file__).parent


def truth_curve(n_frames, amplitude=1.0, seed=0):
    """Slow sinusoids and a creep, 90-150 nm over the acquisition."""
    t = np.arange(n_frames) / n_frames
    phase = np.random.default_rng(seed).uniform(0, 2 * np.pi, 3)
    return amplitude * np.column_stack([
        60 * np.sin(2 * np.pi * 0.8 * t + phase[0]) + 50 * t,
        45 * np.sin(2 * np.pi * 1.3 * t + phase[1]) - 30 * t,
        40 * np.sin(2 * np.pi * 1.1 * t + phase[2]) + 20 * t])


def data(seed=1, n_frames=20000, density=0.02, amplitude=1.0, photons=1500.0,
         background=40.0, tmp=None):
    """Tilted 3D filaments (filaments3d.yaml at ``density`` labels per nm),
    photons and background set for clathrin-like precisions (median 14 nm
    laterally, 41 nm in z), moved by `truth_curve`."""
    spec = yaml.safe_load((HERE / "filaments3d.yaml").read_text())
    spec["elements"][0]["density"] = density
    path = Path(tmp or HERE) / f"_filaments_{seed}_{density}.yaml"
    path.write_text(yaml.safe_dump(spec))
    try:
        locs = simulate(SimulationSettings(
            n_frames=n_frames, seed=seed, background=background,
            structure=StructureSettings(preset="from file", file=str(path)),
            blinking=BlinkingSettings(blinks=4.0, photons=photons,
                                      photons_std=photons / 2)))
    finally:
        path.unlink()
    drift = truth_curve(n_frames, amplitude, seed)
    frame = np.asarray(locs["frame"], int)
    columns = dict(locs.columns)
    for axis, name in enumerate(("x_nm", "y_nm", "z_nm")):
        columns[name] = np.asarray(columns[name], float) + drift[frame, axis]
    return Localizations(columns, dict(locs.metadata)), drift


def _columns(locs):
    columns = {k: np.array(v) for k, v in locs.columns.items()}
    for name in ("x_nm", "y_nm", "z_nm", "xy_err_nm", "z_err_nm"):
        columns[name] = columns[name].astype(float)
    return columns


def with_junk(locs, fraction=0.2, seed=0):
    """A fraction made imprecise: 20-40 nm laterally, 150-300 nm in z,
    recorded so and moved by that much at random."""
    rng = np.random.default_rng(seed + 100)
    c = _columns(locs)
    bad = rng.random(len(locs)) < fraction
    sxy, sz = rng.uniform(20, 40, bad.sum()), rng.uniform(150, 300, bad.sum())
    c["xy_err_nm"][bad], c["z_err_nm"][bad] = sxy, sz
    for name, s in (("x_nm", sxy), ("y_nm", sxy), ("z_nm", sz)):
        c[name][bad] += rng.normal(0, s)
    return Localizations(c, dict(locs.metadata))


def with_pinned(locs, drift, fraction=0.2, seed=0):
    """A fraction of out-of-focus fits held at z = +-400 nm (on the side of
    their true z), which do not follow the z drift, with the precision such
    fits are given (20-40 nm, 150-300 nm in z)."""
    rng = np.random.default_rng(seed + 200)
    c = _columns(locs)
    bad = rng.random(len(locs)) < fraction
    k = bad.sum()
    frame = np.asarray(c["frame"], int)[bad]
    sxy = rng.uniform(20, 40, k)
    c["xy_err_nm"][bad], c["z_err_nm"][bad] = sxy, rng.uniform(150, 300, k)
    c["x_nm"][bad] += rng.normal(0, sxy)
    c["y_nm"][bad] += rng.normal(0, sxy)
    true_z = c["z_nm"][bad] - drift[frame, 2]
    c["z_nm"][bad] = np.where(true_z >= 0, 400.0, -400.0) + rng.normal(0, 15, k)
    return Localizations(c, dict(locs.metadata))


def with_median_precision(locs):
    """Every localization given the median precision: the precision weights
    then widen z (and x, y) by the same amount for every pair."""
    c = _columns(locs)
    for name in ("xy_err_nm", "z_err_nm"):
        c[name][:] = np.median(c[name])
    return Localizations(c, dict(locs.metadata))


def error(found, truth):
    truth = truth[:len(found)]
    e = (found - found.mean(0)) - (truth - truth.mean(0))
    return np.sqrt(np.mean(e ** 2, axis=0))


# the options compared: settings, and what is done to the table first
OPTIONS = {
    "COMET": (dict(precision_weights=False), None),
    "z width 3": (dict(precision_weights=False, sigma_z_factor=3.0), None),
    "median precision": (dict(), with_median_precision),
    "precision weights": (dict(), None),
}


def _run(rows, name, option, locs, truth, select=None, **extra):
    settings, prepare = OPTIONS[option]
    table = prepare(locs) if prepare else locs
    drift = estimate_drift(table, replace(DriftSettings(), **settings, **extra),
                           select=select)
    rows[(name, option)].append((error(drift.drift, truth), drift.noise_nm))


def report(rows):
    for (name, option), values in rows.items():
        e = np.sqrt(np.mean([v[0] ** 2 for v in values], axis=0))
        n = np.sqrt(np.mean([v[1] ** 2 for v in values], axis=0))
        print(f"{name:24s} {option:18s} error {np.round(e, 2)}  noise {np.round(n, 2)}")


def weights(seeds=(1, 2, 3, 4)):
    rows = collections.defaultdict(list)
    for seed in seeds:
        dense, truth = data(seed, density=0.06)
        sparse, truth = data(seed, density=0.02)
        large, big = data(seed, density=0.02, amplitude=2.0)
        cases = [
            ("dense, xy_err < 15", dense, dense["xy_err_nm"] < 15, truth, OPTIONS),
            ("dense", dense, None, truth, OPTIONS),
            ("dense, 20% imprecise", with_junk(dense, 0.2, seed), None, truth, OPTIONS),
            ("sparse", sparse, None, truth, OPTIONS),
            ("sparse, 20% pinned", with_pinned(sparse, truth, 0.2, seed), None, truth,
             OPTIONS),
            ("drift x2, 40% pinned", with_pinned(large, big, 0.4, seed), None, big,
             ("COMET", "z width 3", "precision weights")),
        ]
        for name, locs, select, curve, options in cases:
            for option in options:
                _run(rows, name, option, locs, curve, select)
            if "pinned" in name:      # and the hand-set cut on z precision
                cut = np.asarray(locs["z_err_nm"]) < 100
                for option in ("COMET", "precision weights"):
                    _run(rows, name + ", z_err < 100", option, locs, curve, cut)
    report(rows)


def target(seeds=(1, 2, 3, 4)):
    rows = collections.defaultdict(list)
    for seed in seeds:
        sparse, truth = data(seed, density=0.02)
        for name, locs in (("sparse", sparse),
                           ("sparse, 20% pinned", with_pinned(sparse, truth, 0.2, seed))):
            for sigma in (30, 15, 8):
                for option in ("COMET", "z width 3", "precision weights"):
                    _run(rows, f"{name}, target {sigma}", option, locs, truth,
                         target_sigma_nm=float(sigma))
    report(rows)


def driftless(seeds=(1, 2, 3)):
    """The curve's rms about its mean where none was put in, against the
    noise the error bar reports, by density and knot spacing."""
    rows = collections.defaultdict(list)
    for seed in seeds:
        for density in (0.01, 0.02, 0.04, 0.08):
            locs, truth = data(seed, density=density, amplitude=0.0)
            for knots in (500, 1000, 2000, 4000):
                drift = estimate_drift(locs, DriftSettings(spline_knot_frames=knots))
                rows[(f"density {density}, {len(locs)} locs", f"knots {knots}")].append(
                    (error(drift.drift, truth), drift.noise_nm))
            for weighted in (False, True):
                drift = estimate_drift(locs, DriftSettings(precision_weights=weighted))
                rows[(f"density {density}", f"weights {weighted}")].append(
                    (error(drift.drift, truth), drift.noise_nm))
    report(rows)


if __name__ == "__main__":
    {"weights": weights, "target": target, "driftless": driftless}[sys.argv[1]]()
