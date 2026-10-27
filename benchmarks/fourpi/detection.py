"""Detection on the channels' sum: nearest-pixel, shift and bilinear resampling compared.

    python benchmarks/fourpi/detection.py [--frames 1000]

Needs the 4Pi-SMS example data (zenodo 4022827, `scripts/uipsf/fetch_zenodo.py --url ...`, docs/4pi.md
§10) and a 4Pi model of its beads (the PSF calibration window's *4Pi beads*, or `--model`).  Two parts:

* the movie: the time to put the channels onto channel 0 and find the molecules, and the candidates each
  mode finds against bilinear's -- matched within a pixel, extra, missed;
* a simulation with the movie's registration and known molecules: how many each mode finds (recall at the
  same threshold), and the precision of the fits from its candidates.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from smappy import fourpi
from smappy.channels import ChannelLayout, ChannelSource, detection_lookup, onto_reference
from smappy.io.tiff import open_stack
from smappy.metadata import CameraMetadata
from smappy.multichannel import MultiChannelEngine
from smappy.pipeline import FitSettings
from smappy.plugins.fit import DetectionSettings
from smappy.psf import Global4PiPSF

MODES = ("bilinear", "nearest", "shift")
DATA = Path(os.environ.get("UIPSF_DATA", Path.home() / ".cache/uipsf/zenodo")) / "4pi_sms_nup96"
CAMERA = CameraMetadata(conversion=0.5, offset=100.0, pixelsize_um=0.1)


def summed(channels, lookups):
    total = channels[:, 0].astype(np.float32)
    for c in range(1, channels.shape[1]):
        total = total + onto_reference(channels[:, c], lookups[c])
    return total


def matched(a, b, frames_a, frames_b, radius=1.0):
    """How many of ``a`` have a ``b`` in the same frame within ``radius`` pixels."""
    hits = 0
    for f in np.unique(frames_a):
        pa, pb = a[frames_a == f], b[frames_b == f]
        if len(pb):
            d, _ = cKDTree(pb).query(pa)
            hits += int((d <= radius).sum())
    return hits


def movie(cal, layout, n_frames):
    source = ChannelSource.wrap(open_stack(DATA / "Cell11/Cell11_642_000_000.dcimg"), layout)
    blocks = [b for _, b in source.frames(chunk=250, stop=n_frames)]
    finder = DetectionSettings().finder()
    out = {}
    for mode in MODES:
        lookups = [None] + [detection_lookup(layout, c, layout.size[::-1], mode) for c in range(1, 4)]
        resample = detect = 0.0
        cands = []
        first = 0
        for block in blocks:
            photons = (block.astype(np.float32) - 100) * 0.5
            t = time.perf_counter()
            image = summed(photons, lookups)
            resample += time.perf_counter() - t
            t = time.perf_counter()
            c, _ = finder(image, first_frame=first)
            detect += time.perf_counter() - t
            cands.append(np.c_[c.frame, c.x, c.y])
            first += len(block)
        out[mode] = (np.concatenate(cands), resample / n_frames * 1e3, detect / n_frames * 1e3)
    ref = out["bilinear"][0]
    print(f"movie, {n_frames} frames of 4 x {layout.size[0]} x {layout.size[1]}:")
    for mode in MODES:
        c, rt, dt = out[mode]
        hit = matched(c[:, 1:].astype(float), ref[:, 1:].astype(float), c[:, 0], ref[:, 0])
        back = matched(ref[:, 1:].astype(float), c[:, 1:].astype(float), ref[:, 0], c[:, 0])
        print(f"  {mode:8}: resampling {rt:.3f} ms/frame, finding {dt:.3f} ms/frame; {len(c)} candidates, "
              f"{hit} within 1 px of bilinear's, {len(c) - hit} extra, {len(ref) - back} of bilinear's missed")


def simulation(cal, layout, seed=1):
    rng = np.random.default_rng(seed)
    w, h = layout.size
    n_frames, per = 600, 3
    frame = np.repeat(np.arange(n_frames), per)
    x, y = rng.uniform(10, w - 10, (2, len(frame)))
    z = rng.uniform(-400, 400, len(frame))
    photons = rng.lognormal(np.log(2500), 0.6, len(frame))
    chip = (max(r[1] + r[3] for r in layout.regions), max(r[0] + r[2] for r in layout.regions))
    frames = fourpi.render_frames(cal, layout, frame, x, y, z, photons, phase0=0.3, n_frames=n_frames,
                                  background=15, seed=seed, frame_shape=chip)
    channels = layout.cut(frames)
    print(f"simulation, {len(frame)} molecules (median {np.median(photons):.0f} photons) in {n_frames} frames:")
    for mode in MODES:
        engine = MultiChannelEngine(CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=0.1),
                                    DetectionSettings().finder(), Global4PiPSF(cal), layout,
                                    FitSettings(roisize=13), resampling=mode)
        t = time.perf_counter()
        engine.push(channels, 0)
        locs = engine.flush()
        seconds = time.perf_counter() - t
        lx, ly, lf = locs["x_nm"] / 100, locs["y_nm"] / 100, np.asarray(locs["frame"])
        found, dx, dz = 0, [], []
        for f in range(n_frames):
            t_, l_ = frame == f, lf == f
            if not l_.any():
                continue
            d, i = cKDTree(np.c_[lx[l_], ly[l_]]).query(np.c_[x[t_], y[t_]])
            ok = d < 1
            found += int(ok.sum())
            dx += list((lx[l_][i[ok]] - x[t_][ok]) * 100)
            dz += list(np.asarray(locs["z_nm"])[l_][i[ok]] - z[t_][ok])
        mad = lambda v: 1.4826 * np.median(np.abs(np.asarray(v) - np.median(v)))
        print(f"  {mode:8}: found {found} of {len(frame)} ({100 * found / len(frame):.1f} %); x {mad(dx):.2f} nm, "
              f"envelope z {mad(dz):.1f} nm (robust spreads); {seconds:.1f} s in all")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--model", default=None, help="a 4Pi model of the example's beads, with its layout")
    a = ap.parse_args()
    cal = fourpi.load_fourpi_calibration(a.model)
    layout = ChannelLayout.from_dict(cal.layout).with_maps(cal.maps())
    movie(cal, layout, a.frames)
    simulation(cal, layout)


if __name__ == "__main__":
    main()
