"""A 4Pi netfit model on a real 4Pi movie, read exactly as Spline 4Pi reads it (docs/4pi.md §30).

The movie, its layout and its channel maps come from Spline 4Pi's own source (`FourPiFit.open_source`) with
the settings its benchmark runs used -- ``nup96``: the Ries-lab Nup96 ``.mat`` series, 0.44 e-/ADU, offset 100,
129 nm, the model's maps (P2's `insitu/fit_movie.py`); ``tubulin``: Gabin's Cell01 DCIMG, the 4Pi-SMS layout,
the registration of the bead fields (P1's `tubulin_fit.py`).  The frames are converted to photons as the
fitter converts them, put on the network's grid (`netfit.fourpi.FourChannels`, from the source's layout) and
localized in blocks, each with the window's context from the frames around it (`netfit.fourpi.localize`).

The columns are Spline 4Pi's names: frame, x_nm, y_nm (where channel 0 saw the molecule on the camera
frame, as Spline 4Pi puts them), z_nm and z_env_nm (both the envelope's z; *4Pi z* replaces z_nm),
phase_rad, their errors, photons, prob and each channel's background; then *4Pi z* with its defaults, as
after a Spline 4Pi fit, and the table saved (``.hdf5``) with the model's ``phase_per_nm`` in its metadata.
``NUP96_DIR`` and ``TUBULIN_DIR`` move the data (on a GPU box: a copy of the Mac's cache).

    python real_movie.py MODEL.safetensors CALIBRATION.h5 nup96|tubulin OUT.hdf5 [--start 0] [--stop N] [--device cuda]
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

from smappy.camera import to_photons
from smappy.metadata import CameraMetadata
from smappy.netfit import fourpi as NF
from smappy.netfit.predict import Predictor

NUP96 = sorted(glob.glob(os.path.join(os.environ.get("NUP96_DIR", os.path.expanduser(
    "~/.cache/uipsf/zenodo/4pi_smlm_nup96")), "*.mat")))
TUBULIN = Path(os.environ.get("TUBULIN_DIR", Path.home() / ".cache/uipsf/4pi_tubulin_2017-11-28"))


def settings_for(data: str, calibration: str):
    from smappy.channels import FourPiChannelsSettings
    from smappy.plugins.fit import CameraSettings, SourceSettings
    from smappy.plugins.fit_4pi import (
        FourPiFitSettings,
        FourPiModelSettings,
        FourPiRegistrationSettings,
    )
    if data == "nup96":
        return FourPiFitSettings(source=SourceSettings(path=NUP96[0]),
                                 camera=CameraSettings(conversion=0.44, offset=100.0, pixelsize_um=0.129),
                                 model=FourPiModelSettings(calibration=calibration))
    if data == "tubulin":
        reg = TUBULIN / "smappy_p1" / "registration_cell06-12.json"
        return FourPiFitSettings(source=SourceSettings(path=str(TUBULIN / "Cell01/Cell01_642_000_000.dcimg")),
                                 camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.128),
                                 channels=FourPiChannelsSettings(layout="yale"),
                                 model=FourPiModelSettings(calibration=calibration, registration=FourPiRegistrationSettings(
                                     source="file", path=str(reg))))
    raise ValueError(data)


def run(model, calibration, data, start=0, stop=None, device="cpu", block=500, log=print):
    from smappy.plugins.fit_4pi import FourPiFit
    s = settings_for(data, calibration)
    source = FourPiFit().open_source(s)
    camera = CameraMetadata(conversion=s.camera.conversion, offset=s.camera.offset,
                            pixelsize_um=s.camera.pixelsize_um)
    roi = np.asarray(getattr(source.base, "roi_offset", None) or (0, 0), float)
    four = NF.FourChannels.from_layout(source.layout)
    pred = Predictor(model, device)
    k = pred.window
    n = len(source.base) if stop is None else stop
    parts, t0 = [], time.time()
    for a in range(start, n, block):
        b = min(a + block, n)
        lo, hi = max(a - k, 0), min(b + k, n)
        chunk = np.concatenate([c for _, c in source.frames(chunk=hi - lo, start=lo, stop=hi)])
        photons = to_photons(chunk, camera)
        out = NF.localize(pred, four, photons, first_frame=lo, block=hi - lo)
        keep = (out["frame"] >= a) & (out["frame"] < b)
        parts.append({key: v[keep] for key, v in out.items()})
        log(f"{b - start} frames, {sum(len(p['frame']) for p in parts)} localizations, "
            f"{(b - start) / (time.time() - t0):.0f} frames/s", file=sys.stderr, flush=True)
    c = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    px = s.camera.pixelsize_um * 1000
    cols = {"frame": c["frame"], "x_nm": (c["x_pix"] + roi[0]) * px, "y_nm": (c["y_pix"] + roi[1]) * px,
            "z_nm": c["z_nm"], "z_env_nm": c["z_nm"], "z_err_nm": c["z_err_nm"], "z_env_err_nm": c["z_err_nm"],
            "x_err_nm": c["x_err_pix"] * px, "y_err_nm": c["y_err_pix"] * px,
            "phase_rad": c["phase_rad"], "phase_err_rad": c["phase_err_rad"],
            "photons": c["photons"], "photons_err": c["photons_err"], "prob": c["prob"],
            "background": c["background"]}
    for ch in range(c["background_c"].shape[1]):
        cols[f"background_ch{ch}"] = c["background_c"][:, ch]
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("calibration")
    ap.add_argument("data", choices=("nup96", "tubulin"))
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--block", type=int, default=500)
    a = ap.parse_args()
    cols = run(a.model, a.calibration, a.data, a.start, a.stop, a.device, a.block)
    save_table(cols, a.calibration, a.out)
    print(f"{len(cols['frame'])} localizations -> {a.out}")


def save_table(cols, calibration, out):
    """The columns as a table with the model's phase slope, through *4Pi z* at its defaults, saved."""
    from smappy.fourpi import load_fourpi_calibration
    from smappy.io.formats import save
    from smappy.locs import Localizations
    from smappy.plugins import Context
    from smappy.plugins.fourpi_z import FourPiZ, FourPiZSettings
    k = float(load_fourpi_calibration(calibration).phase_per_nm)
    locs = Localizations({c: np.asarray(v) for c, v in cols.items()}, {"fourpi": {"phase_per_nm": k}})
    locs = FourPiZ().run(Context(locs=locs), FourPiZSettings()).locs
    save(locs, out)


if __name__ == "__main__":
    main()
