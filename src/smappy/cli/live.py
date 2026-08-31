"""Fit an acquisition while the microscope writes it, and watch it build up.

  smappy-live DATA OUT --camera CONFIG.yaml [--cal FILE] [...]

DATA is the growing Micro-Manager TIFF, or the directory it is written into --
it does not have to exist yet.  The window opens as soon as the first frames
appear and updates every few seconds; the fit stops once nothing new has been
written for --timeout seconds, which is how an acquisition ends.

Everything the viewer offers offline works here: zoom, pan, the filter boxes,
contrast, grouping.  An update never changes any of them.
"""
import argparse

from ..detect import (AbsoluteCutoff, DoGFilter, DynamicCutoff,
                            GaussFilter, PeakFinder)
from ..io.calibration import load_spline_calibration, warn_on_em_mismatch
from ..io.watch import WatchSettings, open_growing_stack
from ..live import LiveSettings, live_view
from ..pipeline import FitSettings
from ..psf import GaussianPSF, SplinePSF
from ..render import RenderSettings

from .camera_args import add_camera_arguments, camera_from_args

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="growing MM TIFF, or the directory it goes into")
    ap.add_argument("out", help="HDF5 to write; this is the result")
    ap.add_argument("--cal", default=None, help="_3dcal.mat; without it, Gaussian fit")
    ap.add_argument("--chunk", type=int, default=100, help="frames read at a time")
    ap.add_argument("--update", type=float, default=3.0, help="seconds between updates")
    ap.add_argument("--flush", type=float, default=5.0,
                    help="seconds after which buffered ROIs are fitted anyway")
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between file checks")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="stop after this long with no new frame")
    ap.add_argument("--wait", type=float, default=300.0,
                    help="how long to wait for the file to appear")
    ap.add_argument("--roisize", type=int, default=13)
    ap.add_argument("--sigma", type=float, default=1.2)
    ap.add_argument("--cutoff", type=float, default=1.7)
    ap.add_argument("--filter", choices=["dog", "gauss"], default="dog")
    ap.add_argument("--max-fit-distance", type=float, default=None)
    ap.add_argument("--units", choices=["pixel", "nm", "pixel+nm"], default="nm")
    ap.add_argument("--threads", type=int, default=0, help="0 = one per core")
    ap.add_argument("--mode", choices=["hist", "gauss", "precision"], default="precision")
    add_camera_arguments(ap)
    a = ap.parse_args()

    watch = WatchSettings(poll=a.poll, timeout=a.timeout, appear_timeout=a.wait)
    live = LiveSettings(chunk=a.chunk, update_seconds=a.update,
                        flush_seconds=a.flush, watch=watch)

    print(f"waiting for frames in {a.data} ...")
    src = open_growing_stack(a.data, watch)
    cam = camera_from_args(src, a)
    if a.cal:
        cal = load_spline_calibration(a.cal)
        warn_on_em_mismatch(cal, cam.em_on)
        model = SplinePSF(cal)
    else:
        model = GaussianPSF(sigma=a.sigma)
    flt = DoGFilter(a.sigma) if a.filter == "dog" else GaussFilter(a.sigma)
    cut = DynamicCutoff(a.cutoff) if a.cutoff < 20 else AbsoluteCutoff(a.cutoff)
    finder = PeakFinder(flt, cut)
    settings = FitSettings(roisize=a.roisize, max_fit_distance=a.max_fit_distance,
                           output_unit=a.units, n_threads=a.threads)

    print(f"{cam}\n{model}\n{finder}\n{src.n_frames} frames so far in "
          f"{len(src.files)} file(s); writing {a.out}\n"
          f"close the window to stop; the fit ends {a.timeout:g} s after the last frame")

    viewer = live_view(src, cam, finder, model, settings, output=a.out, live=live,
                       render_settings=RenderSettings(mode=a.mode))

    s = viewer.fit.engine.stats
    print(f"\n{viewer.fit.n_emitted} localizations written to {a.out}")
    print(f"  {s['frames']} frames, {s['candidates']} candidates, "
          f"{s['rejected']} rejected after fit")
    print(f"  {s['detect_seconds']:.1f} s detection, {s['fit_seconds']:.1f} s fitting")
    if viewer.fit.error is not None:
        raise SystemExit(f"the fit stopped early: {viewer.fit.error!r}")


if __name__ == "__main__":
    main()
