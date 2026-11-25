"""How fast the single-channel fitters and the two-colour spline fit are, and where their time goes
(docs/performance.md, "The fitters' speed on a quiet machine").

    PYTHONPATH=src nice python benchmarks/fitters/speed.py simulate      # the movies, once (~3 min, ~2.5 GB)
    PYTHONPATH=src nice python benchmarks/fitters/speed.py rois [--repeat 3]
    PYTHONPATH=src nice python benchmarks/fitters/speed.py plugin [--repeat 3]
    PYTHONPATH=src nice python benchmarks/fitters/speed.py detect [--repeat 3]

The data are simulated, written to ``--data`` (``~/.cache/smappy/speed_bench``): movies of 2000 frames of
256 x 256 at 10 and 30 molecules a frame, 1000-5000 photons, 30 photons of background a pixel, z within
+-500 nm, an sCMOS-like camera (offset 100, 1 e-/ADU, no EM).  The spots are drawn with a real bead
calibration (``~/.cache/decode_smlm/experimental_data_workflow/spline_calibration_3dcal.mat``, 197 planes of
26 x 26, astigmatic) for Spline 3D and the elliptical Gaussian, as round Gaussians of sigma 1.3 px for
Gaussian 2D, and for Spline 3D 2C on a chip of 512 x 256 split left / right, each half 256 x 256, the
second channel at 30 % of the photons and 0.37, -0.21 px off the map (the same spline in both).

* ``rois``: the C++ fit alone, on the 13-px boxes the fitter's own detection cuts from the 30-a-frame
  movie, on one thread and on all cores (fits/s).
* ``plugin``: the whole plugin (`run`, written to an HDF5 file, everything after the fit at its default)
  on each movie, frames/s, with the share of the wall time in photon conversion, filtering, maxima and
  cutoff, cutting the boxes, the fit, writing, and the rest; reading alone beside it (it runs in its own
  thread, `pipeline.prefetch`).  What overlapping detection with the fit would give is
  rest + max(detection, fit).
* ``detect``: peak finding alone per frame, by step, on one thread and all: 256 x 256, 2048 x 2048 (the
  256 frames tiled), and the 4Pi tubulin's channel sum (Spline 4Pi's detection image, real data, P1's
  layout and registration).

Each number is repeated ``--repeat`` times and given as the median (range).  Run it alone: a loaded
machine moves these by 20-40 %.
"""
import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("SMAPPY_SPEED_DATA", Path.home() / ".cache/smappy/speed_bench"))
CALIBRATION = Path.home() / ".cache/decode_smlm/experimental_data_workflow/spline_calibration_3dcal.mat"
FRAMES, SIZE, ROI = 2000, 256, 13
DENSITIES = (10, 30)
PHOTONS, BACKGROUND, Z_NM = (1000.0, 5000.0), 30.0, 500.0
OFFSET, CONVERSION, PIXEL_UM = 100.0, 1.0, 0.11
SHIFT, RATIO = (0.37, -0.21), 0.3
SIGMA = 1.3


def load():
    return os.getloadavg()[0]


def spread(values, fmt="{:,.0f}"):
    """The median of the repeats and their range."""
    v = np.asarray(values, float)
    if len(v) == 1:
        return fmt.format(v[0])
    return f"{fmt.format(np.median(v))} ({fmt.format(v.min())}-{fmt.format(v.max())})"


# ------------------------------------------------------------------------------------------ the data
def movie_path(kind, density):
    return DATA / f"{kind}_{density}.tif"


def dual_calibration_path():
    return DATA / "dual_3dcal.h5"


def _spline():
    from smappy.io.calibration import load_spline_calibration
    return load_spline_calibration(CALIBRATION)


def _transformation():
    """Secondary -> main, as `calibrate.dual` keeps it: the right half moved back by its width and the offset."""
    dx, dy = SHIFT
    return np.array([[1.0, 0.0, -SIZE - dx], [0.0, 1.0, -dy], [0.0, 0.0, 1.0]])


def _spots(rng, n):
    xy = rng.uniform(8, SIZE - 8, (n, 2))
    photons = rng.uniform(*PHOTONS, n)
    z = rng.uniform(-Z_NM, Z_NM, n)
    return xy, photons, z


def _spline_patch(cal, x, y, z_nm, photons, size=15):
    from smappy.io.calibration import evaluate_spline
    return evaluate_spline(cal, x, y, cal.z_nm_to_index(z_nm), size) * photons


def _gauss_patch(x, y, photons, size=15):
    from scipy.special import erf
    edges = np.arange(size + 1) - 0.5

    def axis(c):
        e = erf((edges - c) / (np.sqrt(2) * SIGMA))
        return 0.5 * np.diff(e)
    return photons * np.outer(axis(y), axis(x))


def simulate(kind, density, rng):
    """One movie, in ADU (uint16)."""
    import tifffile
    cal = _spline() if kind != "gauss" else None
    width = 2 * SIZE if kind == "dual" else SIZE
    forward = np.linalg.inv(_transformation())
    frames = np.empty((FRAMES, SIZE, width), np.uint16)
    for f in range(FRAMES):
        frame = np.full((SIZE, width), BACKGROUND)
        xy, photons, z = _spots(rng, density)
        for (x, y), p, zz in zip(xy, photons, z):
            x0, y0 = int(np.rint(x)) - 7, int(np.rint(y)) - 7
            if kind == "gauss":
                patch = _gauss_patch(x - x0, y - y0, p)
            elif kind == "spline":
                patch = _spline_patch(cal, x - x0, y - y0, zz, p)
            else:
                patch = _spline_patch(cal, x - x0, y - y0, zz, p * (1 - RATIO))
            frame[y0:y0 + 15, x0:x0 + 15] += patch
            if kind == "dual":
                sx, sy = (forward @ np.array([x, y, 1.0]))[:2]
                x1, y1 = int(np.rint(sx)) - 7, int(np.rint(sy)) - 7
                frame[y1:y1 + 15, x1:x1 + 15] += _spline_patch(cal, sx - x1, sy - y1, zz, p * RATIO)
        frames[f] = np.clip(rng.poisson(frame) / CONVERSION + OFFSET, 0, 65535)
    tifffile.imwrite(movie_path(kind, density), frames)


def write_dual_calibration():
    from dataclasses import replace

    from smappy.calibrate.dual import (
        DualColorCalibration,
        save_dual_color_calibration,
        split_geometry,
    )
    # the .mat's bead stack is not the knots a native file keeps (a stand-in: the fit reads only the
    # coefficients), nor are its parameters JSON
    cal = _spline()
    cal = replace(cal, psf=np.zeros([n + 1 for n in cal.shape], np.float32), parameters={})
    geometry = split_geometry("right-left", "left", (SIZE, 2 * SIZE), split_position=SIZE)
    dual = DualColorCalibration(cal, cal, _transformation(), geometry,
                                {"secondary_main_brightness_ratio": RATIO / (1 - RATIO)})
    save_dual_color_calibration(dual_calibration_path(), dual)


def cmd_simulate(args):
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    write_dual_calibration()
    for kind in ("spline", "gauss", "dual"):
        for density in DENSITIES:
            t = time.perf_counter()
            simulate(kind, density, rng)
            print(f"{movie_path(kind, density)}: {time.perf_counter() - t:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------ the fitters
def fitters():
    """(name, plugin class, settings maker(path, out), movie kind)."""
    from smappy.pipeline import FitSettings
    from smappy.plugins.fit import (
        CameraSettings,
        DualModelSettings,
        DualSplineFit,
        DualSplineFitSettings,
        GaussianFit,
        GaussianFitSettings,
        GaussianModelSettings,
        OutputSettings,
        SourceSettings,
        SplineFit,
        SplineFitSettings,
        SplineModelSettings,
    )

    def common(path, out):
        return {"source": SourceSettings(path=str(path)),
                    "camera": CameraSettings(conversion=CONVERSION, offset=OFFSET, pixelsize_um=PIXEL_UM),
                    "fit": FitSettings(roisize=ROI),
                    "output": OutputSettings(path=str(out)) if out else OutputSettings(save=False)}

    def gauss(path, out):
        return GaussianFitSettings(model=GaussianModelSettings(), **common(path, out))

    def elliptical(path, out):
        return GaussianFitSettings(model=GaussianModelSettings(elliptical=True), **common(path, out))

    def spline(path, out):
        return SplineFitSettings(model=SplineModelSettings(calibration=str(CALIBRATION)), **common(path, out))

    def dual(path, out):
        s = DualSplineFitSettings(model=DualModelSettings(calibration=str(dual_calibration_path())),
                                  **common(path, out))
        s.model.registration.move = "none"      # the map as it is: no measured move, a prefit of its own
        return s

    return [("Gaussian 2D", GaussianFit, gauss, "gauss"),
            ("Gaussian 2D elliptical", GaussianFit, elliptical, "spline"),
            ("Spline 3D", SplineFit, spline, "spline"),
            ("Spline 3D 2C", DualSplineFit, dual, "dual")]


def _parts(plugin, settings):
    source = plugin.open_source(settings)
    camera = settings.camera.resolve(source)
    finder = plugin.region_finder(settings, camera)
    model = plugin.model(settings, camera)
    return source, camera, finder, model


def _boxes(plugin, settings, frames=1000):
    """The boxes the fitter's own engine would fit, from the first ``frames``: (fit(n_threads), count)."""
    from smappy.pipeline import LocalizationEngine, drive, with_readout
    source, camera, finder, model = _parts(plugin, settings)
    engine = plugin.engine(settings, camera, finder, model)
    kept = []
    if isinstance(engine, LocalizationEngine):
        flush = engine.flush

        def keeping():
            if engine._rois:
                kept.append(np.concatenate(engine._rois))
            return flush()
        engine.flush = keeping
        drive(engine, source.frames(chunk=settings.source.chunk, stop=frames), read_ahead=0)
        rois = np.concatenate(kept)
        data = with_readout(rois, camera.readout_variance)

        def fit(n_threads):
            model.fit(data, iterations=settings.fit.iterations, n_threads=n_threads)
        return fit, len(rois)
    from smappy.multichannel import _concat_pairs
    flush = engine.flush

    def keeping():
        if engine._pairs:
            kept.append(_concat_pairs(engine._pairs))
        return flush()
    engine.flush = keeping
    drive(engine, source.frames(chunk=settings.source.chunk, stop=frames), read_ahead=0)
    data = [(with_readout(p.images, camera.readout_variance), p.link) for p in kept]

    def fit(n_threads):
        for images, link in data:
            model.fit(images, link, iterations=settings.fit.iterations, n_threads=n_threads)
    return fit, sum(len(p.images) for p in kept)


def cmd_rois(args):
    for name, cls, make, kind in fitters():
        plugin = cls()
        settings = make(movie_path(kind, 30), None)
        fit, n = _boxes(plugin, settings)
        out = {}
        for threads in (1, 0):
            rates = []
            for _ in range(args.repeat):
                t = time.perf_counter()
                fit(threads)
                rates.append(n / (time.perf_counter() - t))
            out[threads] = rates
        print(f"{name}: {n} boxes of {ROI} px; one thread {spread(out[1])} fits/s, all cores {spread(out[0])} "
              f"fits/s ({np.median(out[0]) / np.median(out[1]):.1f}x); load {load():.1f}", flush=True)


# ------------------------------------------------------------------------------------------ the whole plugin
class Timer:
    """Wraps attributes to add up their wall time under a name (benchmarks/fourpi/speed.py's)."""

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


def whole(cls, settings):
    import smappy.camera
    import smappy.detect
    import smappy.multichannel
    import smappy.pipeline
    from smappy.detect import PeakFinder
    from smappy.io.hdf5 import LocalizationWriter
    from smappy.plugins import Context
    timer = Timer()
    timer.wrap(smappy.pipeline, "fit_photons", "photons")     # the single-channel engine's name for it
    timer.wrap(smappy.camera, "fit_photons", "photons")       # the multi-channel engine imports it each block
    timer.wrap(PeakFinder, "__call__", "peak finding")
    timer.wrap(smappy.detect, "find_candidates", "maxima+cutoff")
    timer.wrap(smappy.pipeline, "cut_rois", "boxes")
    timer.wrap(smappy.multichannel, "cut_channel_rois", "boxes")
    timer.wrap(LocalizationWriter, "append", "writing")
    timer.wrap(smappy.pipeline, "drive", "loop")              # the fit loop, without setting up and after
    t = time.perf_counter()
    try:
        result = cls().run(Context(), settings)
    finally:
        timer.undo()
    wall = time.perf_counter() - t
    parts = dict(timer.seconds)
    parts["filter"] = parts.pop("peak finding", 0.0) - parts.get("maxima+cutoff", 0.0)
    parts["fit"] = result.locs.metadata.get("stats", {}).get("fit_seconds", 0.0)
    return wall, parts, len(result.locs)


def read_alone(cls, settings):
    source = cls().open_source(settings)
    t = time.perf_counter()
    n = sum(len(b) for _, b in source.frames(chunk=settings.source.chunk))
    return n / (time.perf_counter() - t)


ORDER = ("photons", "filter", "maxima+cutoff", "boxes", "fit", "writing")


def cmd_plugin(args):
    out_dir = Path(tempfile.mkdtemp())
    for name, cls, make, kind in fitters():
        for density in DENSITIES:
            path = movie_path(kind, density)
            settings = make(path, out_dir / "locs.hdf5")
            read = [read_alone(cls, settings) for _ in range(args.repeat)]
            whole(cls, make(path, out_dir / "locs.hdf5"))      # once untimed: the caches, the file in memory
            runs = [whole(cls, make(path, out_dir / "locs.hdf5")) for _ in range(args.repeat)]
            walls = np.array([w for w, _, _ in runs])
            loops = np.array([p["loop"] for _, p, _ in runs])
            # shares of the loop's time; the loop's own rest is turning fits into localizations, the
            # queue to the reading thread, progress.  Since the fit runs in a worker while the next
            # block is found (`LocalizationEngine.overlap`), detection and the fit overlap, and the
            # shares add up to more than the loop: the rest is then what neither covers, the
            # "overlapped" figure the gain the overlap could still give (1.00x: it has all of it)
            shares = {k: np.median([p.get(k, 0.0) / p["loop"] for _, p, _ in runs]) for k in ORDER}
            rest = max(0.0, 1 - sum(shares.values()))
            detection = sum(shares[k] for k in ("photons", "filter", "maxima+cutoff", "boxes"))
            overlapped = min(1.0, rest + shares["writing"] + max(detection, shares["fit"]))
            print(f"{name}, {density}/frame: whole {spread(FRAMES / walls, '{:,.0f}')} frames/s, loop "
                  f"{spread(FRAMES / loops, '{:,.0f}')} frames/s, setup and after {spread(walls - loops, '{:.2f}')} s "
                  f"({runs[0][2] / FRAMES:.1f} locs/frame); reading alone {spread(read)} frames/s; loop: "
                  + ", ".join(f"{k} {100 * v:.0f} %" for k, v in shares.items())
                  + f", rest {100 * rest:.0f} %; overlapped: {1 / overlapped:.2f}x; load {load():.1f}",
                  flush=True)


# ------------------------------------------------------------------------------------------ peak finding
def _steps(finder, photons_of, raw, threads, repeat):
    """ms per frame of each step, ``repeat`` times."""
    from smappy.detect import find_candidates
    from smappy.fit_rois import cut_rois
    times = {k: [] for k in ("photons", "filter", "maxima+cutoff", "boxes")}
    n = len(raw)
    for _ in range(repeat):
        t0 = time.perf_counter()
        photons = photons_of(raw, threads)
        t1 = time.perf_counter()
        filtered = finder.filter(photons, n_threads=threads)
        t2 = time.perf_counter()
        candidates = find_candidates(filtered, finder.cutoff, n_threads=threads)
        t3 = time.perf_counter()
        cut_rois(photons, candidates, ROI)
        t4 = time.perf_counter()
        for k, a, b in (("photons", t0, t1), ("filter", t1, t2), ("maxima+cutoff", t2, t3), ("boxes", t3, t4)):
            times[k].append((b - a) / n * 1e3)
    return times, len(candidates) / n


def _fourpi_channels(frames=200):
    """The tubulin's raw channels and a function from them to its detection image, as Spline 4Pi finds."""
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "fourpi"))
    from speed import cases

    from smappy.camera import fit_photons
    from smappy.channels import detection_lookup
    from smappy.multichannel import detection_image
    from smappy.plugins.fit_4pi import FourPiFit
    plugin, settings = FourPiFit(), cases()["tubulin"]()
    source = plugin.open_source(settings)
    camera = settings.camera.resolve(source)
    raw = np.concatenate([b for _, b in source.frames(chunk=frames, stop=frames)])
    layout = source.layout
    h, w = raw.shape[-2:]
    engine = plugin.engine(settings, camera, plugin.region_finder(settings, camera), plugin.model(settings, camera))
    lookups = [None] + [detection_lookup(layout, c, (h, w), engine.resampling)
                        for c in range(1, layout.n_channels)]

    def photons_of(block, threads):        # four channels to photons, summed onto channel 0
        return detection_image(fit_photons(block, camera, n_threads=threads), layout, lookups, threads)
    return raw, photons_of


def cmd_detect(args):
    import tifffile

    from smappy.camera import fit_photons
    from smappy.metadata import CameraMetadata
    from smappy.plugins.fit import DetectionSettings
    finder = DetectionSettings().finder()
    camera = CameraMetadata(conversion=CONVERSION, offset=OFFSET, pixelsize_um=PIXEL_UM, em_on=False)

    def plain(block, threads):
        return fit_photons(block, camera, n_threads=threads)
    small = tifffile.imread(movie_path("spline", 30), key=range(200))
    big = np.stack([np.block([[small[(i + 8 * r + c) % 200] for c in range(8)] for r in range(8)])
                    for i in range(10)])
    cases = [("256 x 256, 30 molecules", small, plain), ("2048 x 2048 (tiled), 1920 molecules", big, plain)]
    try:
        raw, photons_of = _fourpi_channels()
        cases.append((f"4Pi tubulin channel sum, {raw.shape[-1]} x {raw.shape[-2]} x 4 channels", raw, photons_of))
    except Exception as e:                     # noqa: BLE001  (no 4Pi data here)
        print(f"4Pi tubulin skipped: {e}")
    for label, raw, photons_of in cases:
        for threads in (1, 0):
            times, per_frame = _steps(finder, photons_of, raw, threads, args.repeat)
            total = [sum(v) for v in zip(*times.values())]
            print(f"{label}, {'one thread' if threads == 1 else 'all cores'}: {spread(total, '{:.3f}')} ms/frame; "
                  + ", ".join(f"{k} {spread(v, '{:.3f}')}" for k, v in times.items())
                  + f"; {per_frame:.0f} candidates a frame; load {load():.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["simulate", "rois", "plugin", "detect"])
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    {"simulate": cmd_simulate, "rois": cmd_rois, "plugin": cmd_plugin, "detect": cmd_detect}[args.what](args)


if __name__ == "__main__":
    main()
