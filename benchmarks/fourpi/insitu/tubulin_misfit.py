"""(docs/4pi.md §34)  PYTHONPATH=src python benchmarks/fourpi/insitu/tubulin_misfit.py FIT [MODEL] [frames]

How far a 4Pi model misses the tubulin, in numbers, on the two figures of §32 -- the held-out (even) blocks of
500 frames only, so a model learnt on the odd ones is judged on molecules it never saw:

1. **The channels' split against the phase** (*4Pi z*'s figure *channels*, `fourpi.channel_shares`): in 24 phase
   bins, each channel's median share of the ROIs' photons less the model's for the same localizations.  Printed:
   per channel the rms over the bins, the largest, and its mean in the four bins nearest ±π; and the residual's
   first harmonic (amplitude and the phase where it peaks) -- a phase offset of the channel shows as a harmonic
   a quarter turn from the channel's own, a wrong modulation in phase with it, a wrong transmission as the mean.
2. **The Preview's residual** (Spline 4Pi's figure *molecule*, `fit_4pi.brightest_molecule`): the ROIs of
   ``frames`` frames spread over the even blocks fitted again and drawn by the model (`fourpi.molecule_model`);
   of those inside the z grid, above 2000 photons and no worse than the median (half the boxes: dense filaments
   put neighbours in the rest), per channel the mean Pearson chi² per pixel and the **systematic** part: the
   mean residual image over the molecules of a (z, phase) cell, n * mean² - 1 summed over its pixels and the
   cells (pure noise gives 0 per pixel; a lobe shared by the molecules gives n times its square).

FIT is a `tubulin_fit.py` table of MODEL (default: the table's own calibration); the mean residual images are
saved beside FIT as ``<FIT>_misfit.npz`` for drawing.  Takes a few minutes on the CPU.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from tubulin_common import CONVERSION, MOVIE, OFFSET, PIXEL_UM, registration_settings

from smappy.channels import FourPiChannelsSettings
from smappy.fourpi import (
    channel_shares,
    load_fourpi_calibration,
    model_shares,
    molecule_model,
)
from smappy.io.formats import load
from smappy.plugins.fit import CameraSettings, SourceSettings
from smappy.plugins.fit_4pi import FourPiFit, FourPiFitSettings, FourPiModelSettings

BLOCK = 500
Z_BINS, PHASE_BINS = 3, 4
MIN_PHOTONS = 2000.0


def held_out(frame):
    return (np.asarray(frame) // BLOCK) % 2 == 0


def share_misfit(locs, cal):
    """Part 1: the split's residual per channel (module docstring)."""
    keep = held_out(locs["frame"])
    names = [f"roi_photons_ch{c}" for c in range(cal.n_channels)]
    sums = np.stack([np.asarray(locs[n], float)[keep] for n in names], 1)
    phase = np.asarray(locs["phase_rad"], float)[keep]
    z = np.asarray(locs["z_env_nm"], float)[keep]
    ends = np.sort(cal.z_index_to_nm(np.array([1.0, cal.shape[0] - 2.0])))
    inside = (z > ends[0]) & (z < ends[1])
    roisize = int(locs.metadata["fourpi"]["roisize"])
    binned = channel_shares(sums[inside], phase[inside],
                            model=model_shares(cal, z[inside], phase[inside], roisize))
    r = binned["median"] - binned["model"]                       # (bins, C)
    c = binned["centres"]
    near_pi = np.argsort(np.abs(np.abs(c) - np.pi))[:4]
    out = {"localizations": int(inside.sum())}
    for k in range(r.shape[1]):
        ok = np.isfinite(r[:, k])
        A = np.c_[np.ones(ok.sum()), np.cos(c[ok]), np.sin(c[ok])]
        m, a, b = np.linalg.lstsq(A, r[ok, k], rcond=None)[0]
        own = np.c_[np.ones(ok.sum()), np.cos(c[ok]), np.sin(c[ok])] @ np.linalg.lstsq(
            A, binned["model"][ok, k], rcond=None)[0]
        own_peak = float(c[ok][np.argmax(own)])
        out[f"ch{k + 1}"] = {"rms": float(np.sqrt(np.nanmean(r[:, k] ** 2))),
                             "max": float(np.nanmax(np.abs(r[:, k]))),
                             "near_pi": float(np.nanmean(r[near_pi, k])),
                             "mean": float(m), "harmonic": float(np.hypot(a, b)),
                             "harmonic_peak_rad": float(np.arctan2(b, a)),
                             "model_peak_rad": own_peak}
    return out, binned


def fitter(cal_path):
    settings = FourPiFitSettings(
        source=SourceSettings(path=str(MOVIE)),
        camera=CameraSettings(conversion=CONVERSION, offset=OFFSET, pixelsize_um=PIXEL_UM),
        channels=FourPiChannelsSettings(layout="yale"),
        model=FourPiModelSettings(calibration=str(cal_path),
                                  registration=registration_settings()))
    return FourPiFit(), settings


def residual_misfit(cal_path, frames):
    """Part 2: the Preview's residual over many molecules (module docstring)."""
    from smappy.multichannel import MultiChannelEngine
    from smappy.pipeline import with_readout
    plugin, settings = fitter(cal_path)
    source = plugin.open_source(settings)
    camera = plugin._camera(settings)
    layout = source.layout
    finder = plugin.region_finder(settings, camera)
    model = plugin.model(settings, camera)
    cal = model.calibration
    n = source.n_frames
    even = [f for f in range(n) if held_out(f)]
    chosen = np.asarray(even)[np.linspace(0, len(even) - 1, frames).astype(int)]
    data_all, fit_all, theta_all = [], [], []
    for f in chosen:
        raw = source.frame(int(f))
        engine = MultiChannelEngine(camera, finder, model, layout, settings.fit,
                                    frame_shape=source.base.shape[-2:], keep_rois=True)
        engine.push(raw[None], int(f))
        engine.flush()
        rois = engine.rois
        if rois is None or not len(rois):
            continue
        images = np.asarray(rois.images, float)
        link = np.ascontiguousarray(rois.link)
        data = np.asarray(with_readout(images, camera.readout_variance), np.float32)
        theta = np.asarray(model.fit(data, link).theta, float)
        z = theta[:, model.slot(4)]
        good = np.flatnonzero(np.all(np.isfinite(theta), axis=1) & (z > 1) & (z < cal.shape[0] - 2)
                              & (theta[:, model.slot(2)] > MIN_PHOTONS))
        if not len(good):
            continue
        data_all.append(data[good])
        fit_all.append(molecule_model(cal, model, theta[good], link[good], rois.roisize))
        theta_all.append(theta[good])
    data, fitted, theta = (np.concatenate(a) for a in (data_all, fit_all, theta_all))
    r = (data - fitted) / np.sqrt(np.clip(fitted, 1e-3, None))      # (n, C, h, w)
    chi = np.mean(r ** 2, axis=(1, 2, 3))
    typical = chi <= np.median(chi)
    r, theta = r[typical], theta[typical]
    z = cal.z_index_to_nm(theta[:, model.slot(4)])
    phase = np.angle(np.exp(1j * theta[:, model.slot(5)]))
    zb = np.clip(np.digitize(z, np.quantile(z, np.linspace(0, 1, Z_BINS + 1)[1:-1])), 0, Z_BINS - 1)
    pb = np.clip(((phase + np.pi) / (2 * np.pi) * PHASE_BINS).astype(int), 0, PHASE_BINS - 1)
    C = r.shape[1]
    means = np.zeros((Z_BINS, PHASE_BINS) + r.shape[1:])
    counts = np.zeros((Z_BINS, PHASE_BINS), int)
    systematic = np.zeros(C)
    for i in range(Z_BINS):
        for j in range(PHASE_BINS):
            cell = (zb == i) & (pb == j)
            counts[i, j] = cell.sum()
            if counts[i, j] < 20:
                continue
            means[i, j] = r[cell].mean(0)
            systematic += (counts[i, j] * means[i, j] ** 2 - 1).sum(axis=(-2, -1))
    pixels = r.shape[-1] * r.shape[-2]
    out = {"molecules": len(r), "of": len(chi), "frames": len(chosen)}
    for c in range(C):
        out[f"ch{c + 1}"] = {"chi2_per_px": float(np.mean(r[:, c] ** 2)),
                             "systematic_per_px": float(systematic[c] / (pixels * (counts >= 20).sum())),
                             "systematic_total": float(systematic[c])}
    return out, {"means": means, "counts": counts, "z_edges": np.quantile(z, np.linspace(0, 1, Z_BINS + 1))}


if __name__ == "__main__":
    fit_path = Path(sys.argv[1])
    locs = load(str(fit_path))[0]
    cal_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else locs.metadata["fourpi"]["calibration"]
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    cal = load_fourpi_calibration(cal_path)
    shares, binned = share_misfit(locs, cal)
    print("shares", json.dumps(shares, indent=1), flush=True)
    residual, images = residual_misfit(cal_path, frames)
    print("residual", json.dumps(residual, indent=1))
    np.savez(str(fit_path) + "_misfit.npz", **images, share_centres=binned["centres"],
             share_median=binned["median"], share_model=binned["model"])
    json.dump({"shares": shares, "residual": residual, "model": str(cal_path)},
              open(str(fit_path) + "_misfit.json", "w"), indent=1)
