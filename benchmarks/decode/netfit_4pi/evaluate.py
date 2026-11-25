"""A 4Pi netfit model scored on fixed simulated sets (docs/4pi.md §30): the training's own drawing, at a
sparse and a dense density and mean on-times of 1 and 3 frames, each set drawn once from its seed.

Reported per set: detection (Jaccard, recall, precision); lateral and z (the envelope's) RMSE and their
error over the IAB model's CRLB (`netfit.fourpi.crlb`, equal to Spline 4Pi's) as a robust width, near
focus (|z| < 300 nm) and over all; the phase's RMSE and width over its CRLB; z from the phase with the
cavity's phase known, its wrong-fringe share and its RMSE on the right fringe.  3-frame window.

    python evaluate.py MODEL.safetensors CALIBRATION.h5 [--samples 1024] [--device cuda] [--out scores.json]
"""
import argparse
import json
from dataclasses import replace

import numpy as np
import torch

from smappy.netfit import fourpi, metrics, network, train
from smappy.training import model_file

SETS = {"d5_on1": (5.0, 1.0), "d25_on1": (25.0, 1.0), "d5_on3": (5.0, 3.0), "d25_on3": (25.0, 3.0)}


def load(path, device):
    tensors, meta = model_file.read(path)
    settings = train.settings_from_dict(meta["settings"])
    model = network.build(settings.parts, meta["window"], settings.channels)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in tensors.items()})
    return model.to(device).eval(), settings, network.Scaling(**meta["scaling"])


def score(model, settings, scaling, calibration, density, life, samples, device, seed):
    sim = replace(settings.simulation, density_per_40px=density, lifetime_frames=life, lifetime_max_frames=0.0)
    s = replace(settings, simulation=sim, calibration=calibration)
    drawer = train.Drawer(s, device)
    g = torch.Generator(device).manual_seed(seed)
    batches, outs = [], []
    with torch.no_grad():
        for a in range(0, samples, 128):
            b = drawer(min(128, samples - a), g)
            outs.append(model((b.frames - scaling.input_offset) / scaling.input_scale, 1, b.aux))
            batches.append(b)
    cam = s.camera
    found = {"frame": [], "xyz": [], "err": [], "phase": [], "phase_err": []}
    truth = {"frame": [], "xyz": [], "photons": [], "phase": [], "crlb": [], "crlb_phase": []}
    offset = 0
    for b, out in zip(batches, outs):
        f = train._found(out, scaling, settings.parts.schedule.settings["threshold"], settings.parts.extraction.name,
                         cam.pixel_nm, getattr(model, "out_scale", 1), model.layout)
        for k, kept in found.items():
            kept.append(f[k] + (offset if k == "frame" else 0))
        m = b.mask.cpu().numpy()
        frame = np.nonzero(m)[0]
        xyz = b.xyz.cpu().numpy()[m].astype(np.float64)
        ph = b.photons.cpu().numpy()[m]
        bgs = np.stack([train.local_background(b, frame, xyz, c) for c in range(settings.channels.n)], 1)
        bound = fourpi.crlb(drawer.cal, xyz[:, 0], xyz[:, 1], b.shift.cpu().numpy()[m], xyz[:, 2],
                            b.phase.cpu().numpy()[m], ph, bgs, bool(cam.em_gain))
        truth["frame"].append(frame + offset)
        truth["xyz"].append(xyz)
        truth["photons"].append(ph)
        truth["phase"].append(b.phase.cpu().numpy()[m].astype(np.float64))
        truth["crlb"].append(np.c_[bound[:, 0] * cam.pixel_nm[0], bound[:, 1] * cam.pixel_nm[1], bound[:, 2]])
        truth["crlb_phase"].append(bound[:, 3])
        offset += len(b.frames)
    found = {k: np.concatenate(v) for k, v in found.items()}
    truth = {k: np.concatenate(v) for k, v in truth.items()}
    r = metrics.evaluate(truth, found, cam.pixel_nm)
    r.update(fourpi.phase_scores(truth, found, cam.pixel_nm, drawer.cal.phase_per_nm))
    r["n_truth"] = len(truth["frame"])
    return r


KEYS = ("jaccard", "recall", "precision", "core_rmse_lat_nm", "core_rmse_ax_nm", "core_crlb_width_lat",
        "core_crlb_width_ax", "core_rmse_phase_rad", "core_crlb_width_phase", "core_wrong_fringe_share",
        "core_rmse_z_phase_nm", "rmse_lat_nm", "rmse_ax_nm", "crlb_width_lat", "crlb_width_ax", "rmse_phase_rad",
        "crlb_width_phase", "claim_width_phase", "wrong_fringe_share", "rmse_z_phase_nm", "pixel_mod")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("calibration")
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out")
    a = ap.parse_args()
    model, settings, scaling = load(a.model, a.device)
    results = {}
    for i, (name, (density, life)) in enumerate(SETS.items()):
        r = score(model, settings, scaling, a.calibration, density, life, a.samples, a.device, 777 + i)
        results[name] = r
        print(name, {k: round(r[k], 4) for k in KEYS if r.get(k) is not None}, flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()
