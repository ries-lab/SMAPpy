"""(docs/4pi.md §34)  Refining P1's tubulin model in situ, a few parameters at a time.

    PYTHONPATH=src python benchmarks/fourpi/insitu/tubulin_refine.py cut FIT OUTDIR      # on the Mac (the movie)
    PYTHONPATH=src python benchmarks/fourpi/insitu/tubulin_refine.py learn OUTDIR BEAD SET [SIGMA]   # on a GPU
    PYTHONPATH=src python benchmarks/fourpi/insitu/tubulin_refine.py judge OUTDIR BEAD MODEL.pkl ...

``cut``: from FIT (`tubulin_fit.py`'s table of the bead model over Cell01), the molecules alone within 1.5 boxes in
their frame and the frames beside it (`fourpi_insitu.alone`; §29's boxes held their neighbours' light), above the
median photons: 3000 from the odd blocks of 500 frames to learn from, 2000 from the even ones held out, cut from
the movie into ``OUTDIR/learn.npz`` and ``held.npz``.  ``learn``: in situ from BEAD with only the parameters of
``SET`` free (`SETS`), the molecules tied to the cavity's phase by SIGMA (0.2 rad), two rounds; the model as
``OUTDIR/<SET>_4pi.h5`` and its learnt values in ``<SET>.pkl``.  ``judge``: the held-out molecules localised on
the bead model and on each model, from three z starts: the objective, the deviance per pixel and per channel,
and how many land on another fringe than the beads put them, into ``OUTDIR/judge.json``.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

BLOCK = 500
ISOLATION, ISOLATION_FRAMES = 1.5, 1
TRANSFORMS = ("ch1.transform", "ch2.transform", "ch3.transform")
#: the sets tried, smallest first: the channels' maps; with the arms' misalignment; with each channel's interference
#: (coherence, balance, offset, share); with the blur; with the pupils too -- everything §29 learnt
SETS = {
    "maps": TRANSFORMS,
    "maps_offset": TRANSFORMS + ("fourpi.offset",),
    "maps_offset_channels": TRANSFORMS + ("fourpi.offset", "fourpi.coherence", "fourpi.balance", "fourpi.delta",
                                          "fourpi.share"),
    "channels_sigma": TRANSFORMS + ("fourpi.offset", "fourpi.coherence", "fourpi.balance", "fourpi.delta",
                                    "fourpi.share", "psf.sigma"),
    "all": TRANSFORMS + ("fourpi.offset", "fourpi.coherence", "fourpi.balance", "fourpi.delta", "fourpi.share",
                         "pupil.phase", "lo.pupil.phase", "psf.sigma"),
}
#: the misalignment's prior σ (nm): x, y held where they are not learnt (the default's 1 nm), loose where they are
OFFSET_FREE = (100.0, 100.0, 100.0)


def cut(fit_path, outdir):
    sys.path.insert(0, __file__.rsplit("/", 2)[0])
    from tubulin_common import MOVIE, PIXEL_UM, camera

    from smappy import fourpi
    from smappy.calibrate.fourpi_insitu import (
        FourPiInSituSettings,
        choose,
        cut,
        z_range_of,
    )
    from smappy.channels import ChannelLayout
    from smappy.io.formats import load
    from smappy.io.tiff import open_stack
    locs = load(str(fit_path))[0]
    cal = fourpi.load_fourpi_calibration(locs.metadata["fourpi"]["calibration"])
    layout = ChannelLayout.from_dict(locs.metadata["channel_layout"])
    px = (PIXEL_UM * 1000.0, PIXEL_UM * 1000.0)
    stack = open_stack(str(MOVIE))
    for name, n, parity, seed in (("learn", 3000, 1, 0), ("held", 2000, 0, 1)):
        s = FourPiInSituSettings(molecules=n, seed=seed, isolation=ISOLATION, isolation_frames=ISOLATION_FRAMES)
        index = choose(locs, z_range_of(cal), s, frames=lambda f, p=parity: (f // BLOCK) % 2 == p)
        mol = cut(stack, layout, locs, index, camera(), s.roi_size, px)
        np.savez(Path(outdir) / f"{name}.npz", data=mol.data, roi_xy=mol.roi_xy, frame=mol.frame, index=mol.index,
                 **{"start_" + k: v for k, v in mol.start.items()})
        print(name, len(mol), "photons median", float(np.median(mol.start["photons"])), flush=True)


def molecules(path):
    z = np.load(path)
    return z["data"], z["roi_xy"], z["frame"], {k[6:]: z[k] for k in z.files if k.startswith("start_")}


def device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def learn(outdir, bead_path, name, sigma=0.2):
    from smappy import fourpi
    from smappy.calibrate.fourpi_beads import model_from_calibration
    from uipsf.workflows import fourpi_insitu as fi
    outdir = Path(outdir)
    cal = fourpi.load_fourpi_calibration(bead_path)
    data, roi_xy, frame, start = molecules(outdir / "learn.npz")
    model, shared = model_from_calibration(cal, device=device())
    free = SETS[name]
    tag = name if sigma == 0.2 else f"{name}_tie{sigma:g}"          # 0: untied
    settings = fi.InSituSettings(free=free, cavity_sigma_rad=sigma or None,
                                 offset_sigma_nm=OFFSET_FREE if "fourpi.offset" in free else fi.OFFSET_SIGMA_NM)
    batch = fi.molecule_batch(data, roi_xy, frame=frame)
    t = time.time()
    res = fi.learn_insitu(model, shared, batch, start, cal.period_nm, settings=settings,
                          progress=lambda e: print(e.get("round"), e.get("stage"), e.get("iteration"),
                                                   e.get("objective"), round(e.get("seconds", 0)), flush=True)
                          if e.get("iteration", 0) % 20 == 0 else None)
    print(tag, "learnt in", round((time.time() - t) / 60, 1), "min", res.rounds, flush=True)
    from smappy.calibrate.fourpi_insitu import z_range_of
    c = fi.calibration(res, z_range_nm=z_range_of(cal), dz_nm=float(cal.dz), roi_px=cal.shape[1] + 1,
                       parameters={"free": list(free), "cavity_sigma_rad": sigma})
    c.layout = cal.layout
    fourpi.save_fourpi_calibration(str(outdir / f"{tag}_4pi.h5"), c, overwrite=True)
    pickle.dump({"shared": {k: v.cpu() for k, v in res.shared.items()}, "kept": res.kept, "rounds": res.rounds,
                 "free": free, "sigma": sigma}, open(outdir / f"{tag}.pkl", "wb"))
    for n in sorted(set(free)):
        print(n, np.round(np.asarray(shared[n]), 3).tolist(), "->", np.round(res.shared[n].cpu().numpy(), 3).tolist())
    print("period", cal.period_nm, "->", c.period_nm, flush=True)


def judge(outdir, bead_path, models):
    import torch

    from smappy import fourpi
    from smappy.calibrate.fourpi_beads import model_from_calibration
    from uipsf.noise import ShiftedPoisson
    from uipsf.solve import Problem
    from uipsf.workflows import fourpi_insitu as fi
    outdir = Path(outdir)
    cal = fourpi.load_fourpi_calibration(bead_path)
    data, roi_xy, frame, start = molecules(outdir / "held.npz")
    model, shared = model_from_calibration(cal, device=device())
    hb = fi.molecule_batch(data, roi_xy)
    mm = fi.molecule_model(model, tuple(hb.data.shape[-2:]))
    candidates = [("bead", shared, cal.period_nm)]
    for m in models:
        values = pickle.load(open(m, "rb"))["shared"]
        period = fourpi.load_fourpi_calibration(str(m).replace(".pkl", "_4pi.h5")).period_nm
        candidates.append((Path(m).stem, values, period))
    out, z = {"held": len(hb.data)}, {}
    for name, values, period in candidates:
        found = fi.localise(mm, values, hb, start, period)
        batch = fi._device_batch(hb, mm)
        p = fi._held_shared(fi.params_for(mm, batch, values, found))
        D = Problem(mm, ShiftedPoisson(), batch).deviance(p)              # [E, C, K, Y, X]
        per_channel = D.flatten(2).mean(2).mean(0) if D.dim() > 2 else torch.zeros(1)
        z[name] = found["xyz"][:, 2]
        out[name] = {"objective": float(found["objective"].sum()),
                     "mean deviance/px": float(np.mean(found["deviance"])),
                     "deviance/px per channel": [round(float(v), 4) for v in per_channel.cpu()],
                     "period": float(period)}
        if name != "bead":
            d = z[name] - z["bead"]
            out[name]["gain nats/molecule"] = (out["bead"]["objective"] - out[name]["objective"]) / len(hb.data)
            out[name]["other fringe than beads"] = float(np.mean(np.abs(d - np.median(d)) > cal.period_nm / 2))
            out[name]["z - beads: median, robust sd"] = [float(np.median(d)),
                                                         float(1.4826 * np.median(np.abs(d - np.median(d))))]
        print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(outdir / "judge.json", "w"), indent=1)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "cut":
        cut(sys.argv[2], sys.argv[3])
    elif what == "learn":
        learn(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]) if len(sys.argv) > 5 else 0.2)
    elif what == "judge":
        judge(sys.argv[2], sys.argv[3], sys.argv[4:])
