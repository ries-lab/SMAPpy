"""Fit real ROIs with the spline PSF and report whether the fit is sane.

Also fits the same ROIs with and without the x-flip demanded by a mirrored
calibration: the correct orientation must give the better log-likelihood.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smappy.io.tiff import open_stack
from smappy.cli.camera_args import add_camera_arguments, camera_from_args
from smappy.io.calibration import load_spline_calibration, warn_on_em_mismatch
from smappy.camera import to_photons
from smappy.detect import DoGFilter, DynamicCutoff, PeakFinder
from smappy.roi import cut_rois
from smappy.psf import SplinePSF

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("data")
ap.add_argument("cal", help="_3dcal.mat")
ap.add_argument("nframes", nargs="?", type=int, default=50)
ap.add_argument("roisize", nargs="?", type=int, default=13)
add_camera_arguments(ap)
a = ap.parse_args()
nframes, roisize = a.nframes, a.roisize

src = open_stack(a.data)
cam = camera_from_args(src, a)
cal = load_spline_calibration(a.cal)
warn_on_em_mismatch(cal, cam.em_on)
model = SplinePSF(cal)
print(f"{cam}\n{model}\ncalibration mirrored: {model.mirror}\n")

start, block = next(src.frames(chunk=nframes))
photons = to_photons(block, cam) / cam.excess_noise
finder = PeakFinder(DoGFilter(1.2), DynamicCutoff(1.7))
cands, _ = finder(photons, first_frame=start)

for mirror in (False, True):
    rois = cut_rois(photons, cands, roisize, first_frame=start, mirror=mirror)
    t = time.time()
    res = model.fit(rois.images, iterations=50)
    dt = time.time() - t
    p = model.unpack(res)
    x = rois.to_image_x(p["x_roi"]); y = rois.to_image_y(p["y_roi"])
    dx = x - rois.candidates.x; dy = y - rois.candidates.y
    ok = np.isfinite(p["z_nm"]) & (np.abs(dx) < 3) & (np.abs(dy) < 3)
    tag = "MIRRORED (x-flipped)" if mirror else "as acquired"
    print(f"--- {tag}: {len(rois)} ROIs in {dt:.2f} s "
          f"({len(rois)/dt:,.0f} fits/s)")
    print(f"    logL           : median {np.median(res.logl):9.1f}   "
          f"mean {np.mean(res.logl):9.1f}")
    print(f"    photons        : median {np.median(p['photons'][ok]):7.0f} "
          f"(x{cam.excess_noise:g} EM = {np.median(p['photons'][ok])*cam.excess_noise:.0f})")
    print(f"    background     : median {np.median(p['background'][ok]):7.1f}")
    print(f"    z              : median {np.median(p['z_nm'][ok]):7.0f} nm, "
          f"10-90% [{np.percentile(p['z_nm'][ok],10):.0f}, {np.percentile(p['z_nm'][ok],90):.0f}]")
    print(f"    fit - candidate: dx {np.median(dx):+.2f} +- {np.std(dx[ok]):.2f}, "
          f"dy {np.median(dy):+.2f} +- {np.std(dy[ok]):.2f} pix")
    print(f"    precision      : x {np.median(p['x_err_pix'][ok])*cam.pixelsize_um*1000:.1f} nm, "
          f"z {np.median(p['z_err_nm'][ok]):.1f} nm")
    print(f"    iterations     : median {np.median(res.iterations):.0f}, "
          f"converged early {np.mean(res.iterations < 49)*100:.0f}%")
    print(f"    usable fits    : {ok.sum()}/{len(rois)}")
