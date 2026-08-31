"""Run the whole pipeline on a dataset and write an HDF5 of localizations.

  smappy-fit DATA OUT --camera CONFIG.yaml [--cal FILE]
             [--frames N] [--roisize N] [--sigma S]
             [--cutoff F] [--filter dog|gauss]

The camera is stated in --camera or with --pixelsize/--conversion/--offset;
--cameras CAMERAS.mat is an optional shortcut where a SMAP settings file exists.
"""
import argparse
import time

from ..detect import (AbsoluteCutoff, DoGFilter, DynamicCutoff,
                            GaussFilter, PeakFinder)
from ..io.calibration import load_spline_calibration, warn_on_em_mismatch
from ..io.hdf5 import LocalizationWriter
from ..io.tiff import open_stack
from ..pipeline import FitSettings, fit_stack, provenance
from ..psf import GaussianPSF, SplinePSF

from .camera_args import add_camera_arguments, camera_from_args

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--cal", default=None, help="_3dcal.mat; without it, Gaussian fit")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--roisize", type=int, default=13)
    ap.add_argument("--sigma", type=float, default=1.2)
    ap.add_argument("--cutoff", type=float, default=1.7)
    ap.add_argument("--filter", choices=["dog", "gauss"], default="dog")
    ap.add_argument("--max-fit-distance", type=float, default=None)
    ap.add_argument("--units", choices=["pixel", "nm", "pixel+nm"], default="pixel")
    ap.add_argument("--threads", type=int, default=0, help="0 = one per core")
    ap.add_argument("--read-ahead", type=int, default=2)
    add_camera_arguments(ap)
    a = ap.parse_args()

    src = open_stack(a.data)
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

    n_frames = min(a.frames, src.n_frames) if a.frames else src.n_frames
    print(f"{cam}\n{model}\n{finder}\nfitting {n_frames} of {src.n_frames} frames, "
          f"blocks of {settings.block_rois()} ROIs\n")

    t0 = time.time()
    last = [t0]


    def show(engine):
        if time.time() - last[0] > 2.0:
            last[0] = time.time()
            s = engine.stats
            rate = s["frames"] / (time.time() - t0)
            print(f"\r  {s['frames']}/{n_frames} frames  {rate:6.0f} fps  "
                  f"{s['localizations']} locs", end="", flush=True)


    with LocalizationWriter(a.out) as writer:
        writer.set_metadata(provenance(cam, finder, model, settings, source=a.data))
        _, engine = fit_stack(src.frames(chunk=a.chunk, stop=n_frames), cam, finder,
                              model, settings, sink=writer.append, progress=show,
                              read_ahead=a.read_ahead)
        n_written = len(writer)
    elapsed = time.time() - t0

    s = engine.stats
    print(f"\r{' ' * 70}\r{n_written} localizations written to {a.out}")
    print(f"  {elapsed:.1f} s total: {s['detect_seconds']:.1f} s detection, "
          f"{s['fit_seconds']:.1f} s fitting, "
          f"{elapsed - s['detect_seconds'] - s['fit_seconds']:.1f} s I/O")
    print(f"  {s['frames'] / elapsed:.0f} frames/s, "
          f"{s['rois'] / max(s['fit_seconds'], 1e-9):,.0f} fits/s")
    print(f"  candidates {s['candidates']}, dropped at border {s['dropped_at_border']}, "
          f"rejected after fit {s['rejected']}")


if __name__ == "__main__":
    main()
