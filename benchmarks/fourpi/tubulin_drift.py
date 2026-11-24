"""(docs/4pi.md §31)  PYTHONPATH=src python benchmarks/fourpi/tubulin_drift.py authors|TABLE [what ...]

P6 on the tubulin data: the sample's 3D drift over Cell01, measured by *COMET* and *RCC* on the phase z after
*4Pi z*, against the authors' (``Cell01_DCresult.mat``'s ``shifts``, a curve per frame over all 360 000).

``authors`` builds a 4Pi table of the authors' own raw fits of the whole movie
(``Cell01_642_tmpresult_20200912.mat``, 11.8 M): x and y in nm, the frame, their astigmatic z as the envelope
(turned to our direction and scale, ``-0.8975 zf + 473.7``, fitted to our P1 envelope on the 283 802 pairs of the
first 6000 frames, 53 nm scatter about it) and their phase.  Their CRLB has x and y but neither the phase's error
nor the envelope's, so those are modelled from what they have, fitted to ours on the same pairs:
``log phase_err = 3.010 - 0.721 log I - 0.066 log mod + 0.369 log bg_var`` (R² 0.94) and
``log z_env_err = 7.233 - 0.243 log I - 0.087 log c4 + 0.447 log c5 + 0.031 log zast_err`` (R² 0.79), with
``I`` photons, ``mod`` their modulation (``zangctrresult``), ``bg_var`` and ``c4``, ``c5`` their CRLB columns 3, 4,
5.  So the localizations are theirs, and *4Pi z* and the drift ours: the drift methods compared on the same data.
A ``TABLE`` instead is one of ours (`tubulin_fit.py`), after *4Pi z*.

``what``: any of ``comet`` (from the sure fringes or all, z on the lateral 2000-frame knots or, the default for
a 4Pi table, on 250), ``rcc`` (one window per 3000
frames, the authors' cycle) and ``check`` (the cavity *4Pi z* follows in 250-frame windows against COMET's
fine z).  Outputs go to ``smappy_p6/`` beside the data.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tubulin_common import DATA, PIXEL_UM

from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.fourpi_z import FourPiZ, FourPiZSettings

OUT = DATA / "smappy_p6"
OUT.mkdir(exist_ok=True)
PX = PIXEL_UM * 1000
K = -0.021357736947536773          # P1's pupil model, 294.2 nm


def authors_table() -> Localizations:
    cache = OUT / "authors_table.npz"
    if not cache.exists():
        import scipy.io as sio
        raw = sio.loadmat(DATA / "Cell01_642_tmpresult_20200912.mat")
        col = {k: np.asarray(raw[k]).ravel().astype(np.float64) for k in
               ("xresult", "yresult", "tresult", "zfresult", "zangresult", "Iresult", "zangctrresult",
                "zast_err_result")}
        crlb = np.asarray(raw["CRLBresult"], np.float64)
        del raw
        lg = lambda v: np.log(np.maximum(v, 1e-9))
        phase_err = np.exp(3.010 - 0.721 * lg(col["Iresult"]) - 0.066 * lg(col["zangctrresult"])
                           + 0.369 * lg(crlb[:, 3]))
        env_err = np.exp(7.233 - 0.243 * lg(col["Iresult"]) - 0.087 * lg(crlb[:, 4])
                         + 0.447 * lg(crlb[:, 5]) + 0.031 * lg(col["zast_err_result"]))
        np.savez(cache, frame=col["tresult"].astype(np.int64),
                 x_nm=((col["xresult"] + 150) * PX).astype(np.float32), y_nm=(col["yresult"] * PX).astype(np.float32),
                 xy_err_nm=(np.sqrt(0.5 * (crlb[:, 0] + crlb[:, 1])) * PX).astype(np.float32),
                 photons=col["Iresult"].astype(np.float32),
                 z_env_nm=(-0.8975 * col["zfresult"] + 473.7).astype(np.float32),
                 z_env_err_nm=env_err.astype(np.float32), phase_rad=col["zangresult"].astype(np.float32),
                 phase_err_rad=phase_err.astype(np.float32))
    d = dict(np.load(cache))
    ok = np.all([np.isfinite(v) for v in d.values()], axis=0)
    columns = {k: v[ok] for k, v in d.items()}
    columns["x_err_nm"] = columns["y_err_nm"] = columns["xy_err_nm"]
    return Localizations(columns, {"units": "nm", "fourpi": {"phase_per_nm": K}})


def with_z(name: str) -> Localizations:
    path = OUT / f"{name}_4piz.npz"
    if path.exists():
        d = dict(np.load(path, allow_pickle=True))
        meta = d.pop("_metadata").item()
        return Localizations(d, meta)
    if name == "authors":
        locs = authors_table()
    else:
        from smappy.io.formats import load
        loaded = load(name, columns="all")
        locs = loaded[0] if isinstance(loaded, tuple) else loaded
    t0 = time.time()
    result = FourPiZ().run(Context(locs=locs), FourPiZSettings())
    print(f"4Pi z on {len(locs):,} in {time.time() - t0:.0f} s: {result.text}")
    out = result.locs
    np.savez(path, _metadata=np.array(dict(out.metadata), dtype=object),
             **{k: np.asarray(out[k]) for k in out.keys()})
    return out


def authors_drift(n_frames: int) -> np.ndarray:
    """Theirs per frame, in our directions: their z runs the other way."""
    path = DATA / "authors_shifts.npy"
    if not path.exists():
        import scipy.io as sio
        np.save(path, sio.loadmat(DATA / "Cell01_DCresult.mat", simplify_cells=True)["shifts"].astype(float))
    d = np.load(path)[:n_frames].copy()
    d[:, 2] *= -1
    return d


def compare(name: str, ours: np.ndarray, theirs: np.ndarray, file):
    a, b = ours - ours.mean(0), theirs - theirs.mean(0)
    d = a - b
    lines = [f"{name}: ours minus theirs, rms x {np.sqrt(np.mean(d[:, 0] ** 2)):.1f}, y "
             f"{np.sqrt(np.mean(d[:, 1] ** 2)):.1f}, z {np.sqrt(np.mean(d[:, 2] ** 2)):.1f} nm; "
             f"ptp ours {np.ptp(ours, 0).round(1)}, theirs {np.ptp(theirs, 0).round(1)}; correlation "
             + ", ".join(f"{np.corrcoef(a[:, k], b[:, k])[0, 1]:.3f}" for k in range(3))]
    # their z is per 3000-frame cycle: our curve's average over each cycle against theirs at the cycle's centre
    cycles = len(ours) // 3000
    if cycles > 1:
        mean = a[:cycles * 3000].reshape(cycles, 3000, 3).mean(1)
        centre = b[1500:cycles * 3000:3000]
        dd = mean - centre
        dd -= dd.mean(0)
        lines.append(f"  per cycle (our mean against theirs at the centre): rms x {np.sqrt(np.mean(dd[:, 0] ** 2)):.1f}, "
                     f"y {np.sqrt(np.mean(dd[:, 1] ** 2)):.1f}, z {np.sqrt(np.mean(dd[:, 2] ** 2)):.1f} nm; "
                     f"largest |z| {np.abs(dd[:, 2]).max():.1f} nm at cycle {np.abs(dd[:, 2]).argmax()}")
    text = "\n".join(lines)
    print(text)
    file.write(text + "\n")


def main():
    which = sys.argv[1]
    what = sys.argv[2:] or ["comet", "rcc", "check"]
    name = "authors" if which == "authors" else Path(which).stem
    locs = with_z(which)
    frame = np.asarray(locs["frame"], np.int64)
    n_frames = int(frame.max()) + 1
    p = np.asarray(locs["fringe_probability"], float)
    sure = p >= 0.9
    print(f"{name}: {len(locs):,} localizations over {n_frames} frames; fringe 90 % sure: {sure.mean():.1%}; "
          f"expected wrong {np.nanmean(1 - p):.1%}")
    theirs = authors_drift(n_frames)
    log = open(OUT / f"drift_{name}.txt", "a")
    log.write(f"\n== {time.ctime()} {name}: {len(locs):,} localizations, {n_frames} frames\n")
    import pickle

    if "comet" in what:
        from smappy.drift import DriftSettings, estimate_cost, estimate_drift
        # the whole hour's 4.5 M blinks within COMET's 300 nm are 19 billion pairs, 457 GB: RCC first over
        # one window per cycle, then COMET within 50 nm, on a random 30 % of the localizations (about 3 GB)
        part = np.random.default_rng(0).random(len(locs)) < (0.3 if len(locs) > 2e6 else 1.0)
        fine = dict(split_half=False, rcc_prepass=True, rcc_prepass_windows=max(n_frames // 3000, 2),
                    max_drift_nm=50.0, initial_sigma_nm=None)
        for label, select, settings in (
                ("COMET, sure fringes, z knots 2000", part, DriftSettings(**fine, z_knot_frames=2000)),
                ("COMET, all, z knots 2000", part, DriftSettings(**fine, z_knot_frames=2000, sure_fringe=0)),
                ("COMET, sure fringes, z knots 250", part, DriftSettings(**fine))):
            print(estimate_cost(locs, settings, select=select))
            t0 = time.time()
            d = estimate_drift(locs, settings, select=select, progress=lambda s: None)
            seconds = time.time() - t0
            compare(f"{label} ({seconds:.0f} s, {d.n_used:,} used)", d.drift, theirs, log)
            with open(OUT / f"drift_{name}_{label.replace(', ', '_').replace(' ', '')}.pkl", "wb") as f:
                pickle.dump(d.to_dict(), f)
    if "rcc" in what:
        from smappy.rcc import RCCSettings, estimate_drift_rcc
        windows = max(n_frames // 3000, 2)
        for label, p_sure in (("RCC, sure fringes", 0.9), ("RCC, all", 0.0)):
            t0 = time.time()
            d = estimate_drift_rcc(locs, RCCSettings(split_half=False, n_timepoints=windows, sure_fringe=p_sure))
            compare(f"{label}, {windows} windows ({time.time() - t0:.0f} s)", d.drift, theirs, log)
            with open(OUT / f"drift_{name}_{label.replace(', ', '_').replace(' ', '')}.pkl", "wb") as f:
                pickle.dump(d.to_dict(), f)
    if "check" in what:
        check(locs, sure, n_frames, log)
    log.close()


def check(locs, sure, n_frames, log, short=100, knots=250):
    """What *4Pi z* missed of the cavity, seen two ways.  *4Pi z* took out the cavity's phase in 1000-frame
    windows; in ``short``-frame windows it follows the cavity closer (to about 1.5 nm as z here), so the
    difference of the two, as z, is the wander the 1000-frame windows left in z.  COMET with z on fine knots
    sees it as drift: its fine z less its coarse z should be that difference, smoothed to its knots."""
    from smappy.drift import DriftSettings, estimate_drift
    from smappy.fourpi import cavity_phase
    k = float(locs.metadata["fourpi"]["phase_per_nm"])
    a = {c: np.asarray(locs[c], float) for c in ("phase_rad", "phase_err_rad", "z_env_nm", "z_env_err_nm",
                                                 "frame")}
    args = (a["phase_rad"], a["phase_err_rad"], a["z_env_nm"], a["z_env_err_nm"], a["frame"], k)
    frames = np.arange(n_frames)
    long, fine = (cavity_phase(*args, window_frames=w) for w in (1000, short))
    missed = (np.interp(frames, fine.centres, fine.phase0) - np.interp(frames, long.centres, long.phase0)) / k
    part = np.random.default_rng(0).random(len(locs)) < (0.3 if len(locs) > 2e6 else 1.0)
    common = dict(split_half=False, max_drift_nm=50.0)
    if len(locs) > 2e6:
        common.update(rcc_prepass=True, rcc_prepass_windows=max(n_frames // 3000, 2))
    coarse = estimate_drift(locs, DriftSettings(**common, z_knot_frames=2000), select=part)
    finer = estimate_drift(locs, DriftSettings(**common, z_knot_frames=knots), select=part)
    seen = finer.drift[:, 2] - coarse.drift[:, 2]
    kernel = np.ones(knots) / knots
    smooth = np.convolve(missed - missed.mean(), kernel, mode="same")
    seen = seen - seen.mean()
    r = np.corrcoef(seen, smooth)[0, 1]
    slope = np.polyfit(smooth, seen, 1)[0]
    text = (f"check: cavity missed by 1000-frame windows, as z: rms {missed.std():.2f} nm ({short}-frame windows, "
            f"median error {np.median(fine.error) / abs(k):.2f} nm); COMET's z on {knots}-frame knots less its z on "
            f"2000: rms {seen.std():.2f} nm; correlation {r:.3f}, slope {slope:.2f}; rms of the difference "
            f"{np.std(seen - smooth):.2f} nm")
    print(text)
    log.write(text + "\n")
    np.savez(OUT / "check.npz", missed=missed, smooth=smooth, seen=seen)


if __name__ == "__main__":
    main()
