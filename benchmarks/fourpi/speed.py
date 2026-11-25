"""Where *Spline 4Pi*'s time goes, end to end, on real 4Pi data (docs/4pi.md §28).

    nice python benchmarks/fourpi/speed.py [--frames N] [--drift | --kept] [--repeat 3] [--save DIR] [--compare DIR]

Two data sets:

* **tubulin**: Gabin's 4Pi-SMS DCIMG (2017-11-28, `Cell01_642_000_000.dcimg`, strips of 2048 x 168, the
  four channels at the 4Pi-SMS (Yale) corners), read from ``~/.cache/uipsf/4pi_tubulin_2017-11-28``, with
  P1's model of that microscope (§26: ``smappy_p1/pupil_680_100_held_4pi.h5``, the channels registered from
  the bead fields, ``registration_cell06-12.json``) and its camera (0.5 e-/ADU, 128 nm).  ``--stand-in``
  fits with the Ries-lab bead model instead, unregistered, as §28's first numbers were.
* **nup96**: the Ries-lab Nup96-mMaple quadrant ``.mat`` series (zenodo 10823308) with v1's in-situ model
  and the maps kept with it.

For each: the reader alone (frames/s through `channels.ChannelSource`), the engine alone on blocks already
in memory (detection and the C++ fit, fits/s), the C++ fit alone on the engine's own ROIs, and the whole
plugin, `FourPiFit().run`, with its parts timed -- the channel-drift prefit (``--drift``), the fit loop,
writing, and *after the fit* (4Pi z).  ``--save DIR`` keeps each set's localizations; ``--compare DIR``
compares this run's with them molecule by molecule (positions, z, phase, logL), which is how every speed
change was checked to keep the numbers.
"""
import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np

HOME = Path.home() / ".cache/uipsf"
ZENODO = Path(os.environ.get("UIPSF_DATA", HOME / "zenodo"))


def cases(stand_in=False):
    from smappy.plugins.fit import CameraSettings, SourceSettings
    from smappy.plugins.fit_4pi import FourPiFitSettings, FourPiModelSettings

    def tubulin():
        tubulin = HOME / "4pi_tubulin_2017-11-28"
        if stand_in:
            s = FourPiFitSettings(
                source=SourceSettings(path=str(tubulin / "Cell01/Cell01_642_000_000.dcimg")),
                camera=CameraSettings(conversion=0.45, offset=100.0, pixelsize_um=0.12),
                model=FourPiModelSettings(
                    calibration=str(ZENODO / "4pi_40nm_bead/261027_4pi_40nm_bead_3Dcal_4pi.h5")))
            s.model.registration.source = "none"
        else:                            # P1's model and registration (benchmarks/fourpi/tubulin_fit.py)
            s = FourPiFitSettings(
                source=SourceSettings(path=str(tubulin / "Cell01/Cell01_642_000_000.dcimg")),
                camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.128),
                model=FourPiModelSettings(calibration=str(tubulin / "smappy_p1/pupil_680_100_held_4pi.h5")))
            s.model.registration.source = "file"
            s.model.registration.path = str(tubulin / "smappy_p1/registration_cell06-12.json")
        s.channels.layout = "yale"
        return s

    def nup96():
        return FourPiFitSettings(
            source=SourceSettings(path=str(ZENODO / "4pi_smlm_nup96/cell3_nup96_mMaple__560_00043_00000.mat")),
            camera=CameraSettings(conversion=0.44, offset=100.0, pixelsize_um=0.129),
            model=FourPiModelSettings(
                calibration=str(ZENODO / "results/4pi_smlm_nup96/psfmodel_iter1_insitu_zernike_4pi.h5")))

    return {"tubulin": tubulin, "nup96": nup96}


class Timer:
    """Wraps methods to add up their wall time under a name."""

    def __init__(self):
        self.seconds, self._undo = {}, []

    def wrap(self, owner, name, label):
        original = getattr(owner, name)

        def timed(*args, **kwargs):
            t = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.seconds[label] = self.seconds.get(label, 0.0) + time.perf_counter() - t
        setattr(owner, name, timed)
        self._undo.append((owner, name, original))

    def undo(self):
        for owner, name, original in reversed(self._undo):
            setattr(owner, name, original)
        self._undo.clear()


def read_alone(plugin, settings, frames):
    source = plugin.open_source(settings)
    t = time.perf_counter()
    n = sum(len(b) for _, b in source.frames(chunk=settings.source.chunk, stop=frames))
    return n / (time.perf_counter() - t)


def engine_alone(plugin, settings, frames):
    """Detection and fit on blocks already read: the engine's own rate, and its ROIs."""
    from smappy.pipeline import drive
    source = plugin.open_source(settings)
    blocks = list(source.frames(chunk=settings.source.chunk, stop=frames))
    camera = settings.camera.resolve(source)
    finder = plugin.region_finder(settings, camera)
    model = plugin.model(settings, camera)
    engine = plugin.engine(settings, camera, finder, model)
    kept = []
    flush = engine.flush

    def keeping():                       # the ROIs the engine fits, for the C++ alone
        if engine._pairs:
            from smappy.multichannel import _concat_pairs
            kept.append(_concat_pairs(engine._pairs))
        return flush()
    engine.flush = keeping
    t = time.perf_counter()
    drive(engine, iter(blocks), read_ahead=0)
    wall = time.perf_counter() - t
    return engine.stats, wall, kept, model, camera


def fit_alone(model, camera, pairs, n_threads=0):
    from smappy.pipeline import with_readout
    t = time.perf_counter()
    n = 0
    for p in pairs:
        model.fit(with_readout(p.images, camera.readout_variance), p.link, n_threads=n_threads)
        n += len(p.images)
    return n / (time.perf_counter() - t), n


def _forget_drift(movie):
    try:
        from smappy.multichannel import drift_cache_path
    except ImportError:                  # a version before the cache
        return
    drift_cache_path(movie).unlink(missing_ok=True)


def whole(plugin, settings, frames, drift, fresh=True):
    from smappy.io.hdf5 import LocalizationWriter
    from smappy.plugins import Context
    from smappy.plugins.fit_4pi import FourPiFit
    settings.source.stop = frames
    settings.model.channel_drift = drift
    out = Path(tempfile.mkdtemp()) / "locs.hdf5"
    if drift and fresh:                  # measured, not read from a run before
        _forget_drift(settings.source.path)
    settings.output.path = str(out)
    timer = Timer()
    timer.wrap(FourPiFit, "channel_drift", "drift prefit")
    timer.wrap(LocalizationWriter, "append", "writing")
    timer.wrap(FourPiFit, "finish", "after the fit")
    t = time.perf_counter()
    try:
        result = plugin.run(Context(), settings)
    finally:
        timer.undo()
    wall = time.perf_counter() - t
    return result, wall, timer.seconds


def compare(locs, path):
    """This run's localizations against a saved run's, matched by frame and peak."""
    from smappy.io.hdf5 import load_localizations
    old = load_localizations(path)

    def keyed(t):
        return {(int(f), round(float(x), 1), round(float(y), 1)): i
                for i, (f, x, y) in enumerate(zip(t["frame"], t["peak_x_pix"] if "peak_x_pix" in t
                                                   else t["x_nm"], t["peak_y_pix"] if "peak_y_pix" in t
                                                   else t["y_nm"]))}
    a, b = keyed(locs), keyed(old)
    common = sorted(set(a) & set(b))
    ia, ib = np.array([a[k] for k in common]), np.array([b[k] for k in common])
    print(f"    compared: {len(common)} of {len(locs)} now / {len(old)} saved matched by frame and peak")
    for col, unit in (("x_nm", "nm"), ("y_nm", "nm"), ("z_env_nm", "nm"), ("phase_rad", "rad"),
                      ("photons", ""), ("logl", "")):
        if col not in locs or col not in old:
            continue
        d = np.asarray(locs[col], float)[ia] - np.asarray(old[col], float)[ib]
        if col == "phase_rad":
            d = np.angle(np.exp(1j * d))
        d = d[np.isfinite(d)]
        if col == "logl":
            print(f"      another optimum (|d logL| > 0.5): {np.mean(d > 0.5):.4f} better now, "
                  f"{np.mean(d < -0.5):.4f} worse now")
        print(f"      {col:10}: max |d| {np.max(np.abs(d)) if len(d) else 0:.3g} {unit}, "
              f"99th pct {np.percentile(np.abs(d), 99) if len(d) else 0:.3g}, "
              f"share > 1e-3: {np.mean(np.abs(d) > 1e-3) if len(d) else 0:.4f}")


def spread(values, fmt="{:,.0f}"):
    """The median of the repeats and their range, as the tables give them."""
    v = np.asarray(values, float)
    if len(v) == 1:
        return fmt.format(v[0])
    return f"{fmt.format(np.median(v))} ({fmt.format(v.min())}-{fmt.format(v.max())})"


def load():
    return os.getloadavg()[0]


def main():
    from smappy.plugins.fit_4pi import FourPiFit
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--drift", action="store_true", help="with the channel-drift prefit")
    parser.add_argument("--kept", action="store_true",
                        help="the whole plugin with the channel drift read from beside the movie (measured "
                             "first if it is not there), timed --repeat times")
    parser.add_argument("--repeat", type=int, default=1, help="each number this often: median (range)")
    parser.add_argument("--one-thread-fits", type=int, default=20000,
                        help="the C++ fit on one thread on this many of the engine's ROIs")
    parser.add_argument("--stand-in", action="store_true", help="tubulin with the Ries-lab bead model (§28)")
    parser.add_argument("--only", choices=["tubulin", "nup96"])
    parser.add_argument("--save")
    parser.add_argument("--compare")
    args = parser.parse_args()
    r = args.repeat
    for name, make in cases(args.stand_in).items():
        if args.only and name != args.only:
            continue
        plugin = FourPiFit()
        settings = make()
        print(f"{name}: {args.frames} frames, {os.cpu_count()} cores, load {load():.1f}", flush=True)
        rates = [read_alone(plugin, settings, args.frames) for _ in range(r)]
        print(f"  reading alone: {spread(rates)} frames/s", flush=True)
        engine = [engine_alone(plugin, settings, args.frames) for _ in range(r)]
        stats, _, pairs, model, camera = engine[0]
        n_fits = stats["pairs"]
        fps = [e[0]["frames"] / e[1] for e in engine]
        det = [e[0]["detect_seconds"] / e[0]["frames"] * 1e3 for e in engine]
        fit = [e[0]["pairs"] / e[0]["fit_seconds"] for e in engine]
        print(f"  engine alone: {spread(fps)} frames/s; detection {spread(det, '{:.2f}')} ms/frame, "
              f"fit {spread(fit)} fits/s; {n_fits / stats['frames']:.1f} fits per frame", flush=True)
        all_cores = [fit_alone(model, camera, pairs)[0] for _ in range(r)]
        n = pairs[0].images.shape[-1]
        print(f"  C++ fit alone, all cores: {spread(all_cores)} fits/s ({n_fits} ROIs of {n} px)", flush=True)
        few, k = [], 0
        for p in pairs:                  # the first ROIs, for one thread
            if k >= args.one_thread_fits:
                break
            few.append(p)
            k += len(p.images)
        one = [fit_alone(model, camera, few, n_threads=1)[0] for _ in range(r)]
        print(f"  C++ fit alone, one thread: {spread(one)} fits/s ({k} ROIs); "
              f"{np.median(all_cores) / np.median(one):.1f}x on all cores", flush=True)
        if args.kept:
            # once first, untimed in the table: it measures the drift if what is kept beside the movie
            # is for other settings, and leaves it there for the runs that are timed
            _, wall, parts = whole(plugin, make(), args.frames, True, fresh=False)
            print(f"  whole plugin, first run: {args.frames / wall:,.1f} frames/s, {wall:.1f} s; "
                  + ", ".join(f"{k} {v:.1f} s" for k, v in parts.items()), flush=True)
            runs = [whole(plugin, make(), args.frames, True, fresh=False) for _ in range(r)]
            result = runs[0][0]
            print(f"  whole plugin, drift kept: {spread([args.frames / w for _, w, _ in runs], '{:,.1f}')} "
                  f"frames/s; {len(result.locs)} localizations; load {load():.1f}", flush=True)
        else:
            result, wall, parts = whole(plugin, make(), args.frames, args.drift)
            s = result.locs.metadata.get("stats", {})
            loop = wall - sum(parts.values())
            print(f"  whole plugin: {args.frames / wall:,.0f} frames/s, {wall:.1f} s; "
                  + ", ".join(f"{k} {v:.1f} s" for k, v in parts.items())
                  + f"; fit loop otherwise {loop:.1f} s (detection {s.get('detect_seconds', 0):.1f}, "
                    f"fit {s.get('fit_seconds', 0):.1f}); {len(result.locs)} localizations")
            if args.drift:                   # again: the drift read from beside the movie, where it now is
                _, wall2, parts2 = whole(plugin, make(), args.frames, True, fresh=False)
                print(f"  again, the drift kept: {args.frames / wall2:,.0f} frames/s, {wall2:.1f} s; "
                      + ", ".join(f"{k} {v:.1f} s" for k, v in parts2.items()))
                _forget_drift(settings.source.path)
        if args.save:
            from smappy.io.hdf5 import save_localizations
            Path(args.save).mkdir(parents=True, exist_ok=True)
            save_localizations(Path(args.save) / f"{name}.hdf5", result.locs)
        if args.compare:
            compare(result.locs, Path(args.compare) / f"{name}.hdf5")


if __name__ == "__main__":
    main()
