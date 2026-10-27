"""How many starts the 4Pi fit needs: fewer starts against the most, molecule by molecule, on real data.

    python benchmarks/fourpi/starts.py

Each movie's first 1500 frames are fitted with every setting below; the molecules are matched by their
candidate (frame and peak pixel), so each is the same ROIs fitted again.  A fit is *stuck* where a
setting's likelihood is below the best setting's by more than 0.5 (a local optimum); for those, how far z
and the phase are from the best fit's.  Needs the 4Pi-SMS example (zenodo 4022827, docs/4pi.md §10) with a
model of its beads, and uiPSF's Nup96 example with v1's in-situ model.
"""
import os
import time
from pathlib import Path

import h5py
import numpy as np

from smappy import fourpi
from smappy.channels import ChannelLayout, ChannelSource, maps_from_uipsf_v1
from smappy.io.tiff import open_stack
from smappy.metadata import CameraMetadata
from smappy.multichannel import MultiChannelEngine
from smappy.pipeline import FitSettings, drive
from smappy.plugins.fit import DetectionSettings
from smappy.psf import Global4PiPSF

ROOT = Path(os.environ.get("UIPSF_DATA", Path.home() / ".cache/uipsf/zenodo"))
SETTINGS = [("1 z start", 1, 1), ("2 z starts", 2, 1), ("3 z starts", 3, 1), ("1 z, 2 phases", 1, 2),
            ("3 z, 2 phases", 3, 2)]


def cases(sms_model):
    cal = fourpi.load_fourpi_calibration(sms_model)
    layout = ChannelLayout.from_dict(cal.layout).with_maps(cal.maps())
    yield ("4Pi-SMS Nup96-AF647, its beads' pupil model", cal,
           ChannelSource.wrap(open_stack(ROOT / "4pi_sms_nup96/Cell11/Cell11_642_000_000.dcimg"), layout),
           CameraMetadata(conversion=0.5, offset=100.0, pixelsize_um=0.1))
    path = ROOT / "results/4pi_smlm_nup96/psfmodel_iter1_insitu_zernike_4pi.h5"
    with h5py.File(path, "r") as g:
        maps = maps_from_uipsf_v1(g["res/T"][...], g["res/imgcenter"][...], True)
    base = open_stack(ROOT / "4pi_smlm_nup96/cell3_nup96_mMaple__560_00043_00000.mat")
    yield ("uiPSF Nup96-mMaple, v1's in-situ model", fourpi.load_uipsf_v1(path),
           ChannelSource.wrap(base, base.layout.with_maps(maps)),
           CameraMetadata(conversion=0.44, offset=100.0, pixelsize_um=0.129))


def fit_all(cal, source, camera, n_starts, phase_starts, frames=1500):
    blocks = [(a, b) for a, b in source.frames(chunk=500, stop=frames)]
    model = Global4PiPSF(cal, n_starts=n_starts, phase_starts=phase_starts)
    engine = MultiChannelEngine(camera, DetectionSettings().finder(), model, source.layout, FitSettings(roisize=13))
    parts = []
    t = time.perf_counter()
    drive(engine, iter(blocks), sink=parts.append)
    from smappy.locs import concat
    return concat([p for p in parts if p is not None]), engine.stats["fit_seconds"], time.perf_counter() - t


def main(sms_model):
    for name, cal, source, camera in cases(sms_model):
        print(name)
        runs = {label: fit_all(cal, source, camera, z, p) for label, z, p in SETTINGS}
        keys = {label: {(int(f), int(x), int(y)): i for i, (f, x, y) in
                        enumerate(zip(r[0]["frame"], r[0]["peak_x_nm"], r[0]["peak_y_nm"]))}
                for label, r in runs.items()}
        common = set.intersection(*(set(k) for k in keys.values()))
        index = {label: np.array([keys[label][c] for c in common]) for label in runs}
        ll = {label: np.asarray(runs[label][0]["logl"], float)[index[label]] for label in runs}
        best = np.max(np.stack(list(ll.values())), axis=0)
        ref = "3 z, 2 phases"
        z_ref = np.asarray(runs[ref][0]["z_nm"], float)[index[ref]]
        ph_ref = np.asarray(runs[ref][0]["phase_rad"], float)[index[ref]]
        bright = np.asarray(runs[ref][0]["photons"], float)[index[ref]] > 1000
        for label, (locs, fit_s, all_s) in runs.items():
            stuck = ll[label] < best - 0.5
            dz = np.abs(np.asarray(locs["z_nm"], float)[index[label]] - z_ref)
            dph = np.abs(np.angle(np.exp(1j * (np.asarray(locs["phase_rad"], float)[index[label]] - ph_ref))))
            far = stuck & ((dz > 50) | (dph > 0.5))
            print(f"  {label:14}: fit {fit_s:5.1f} s; stuck {100 * stuck.mean():5.2f} % "
                  f"({100 * (stuck & bright).sum() / max(bright.sum(), 1):5.2f} % of > 1000 photons), "
                  f"of them far off (z > 50 nm or phase > 0.5 rad) {100 * far.mean():5.2f} %; "
                  f"median dlogl when stuck {np.median((best - ll[label])[stuck]) if stuck.any() else 0:.1f}")
        print(f"  ({len(common)} molecules fitted by every setting)")
        # against 3 z starts, the setting fewer starts would replace: how often they differ by a lot
        z3 = np.asarray(runs["3 z starts"][0]["z_nm"], float)[index["3 z starts"]]
        p3 = np.asarray(runs["3 z starts"][0]["phase_rad"], float)[index["3 z starts"]]
        for label in ("1 z start", "2 z starts"):
            locs = runs[label][0]
            dz = np.abs(np.asarray(locs["z_nm"], float)[index[label]] - z3)
            dph = np.abs(np.angle(np.exp(1j * (np.asarray(locs["phase_rad"], float)[index[label]] - p3))))
            print(f"  {label} against 3 z starts: z off by > 20 nm {100 * (dz > 20).mean():.2f} %, > 50 nm "
                  f"{100 * (dz > 50).mean():.2f} %; phase off by > 0.5 rad {100 * (dph > 0.5).mean():.2f} %")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
