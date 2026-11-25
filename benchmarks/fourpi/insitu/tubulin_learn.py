"""(docs/4pi.md §29)  PYTHONPATH=src python benchmarks/fourpi/insitu/tubulin_learn.py FIT BEAD OUT [N] [SIGMA]

4Pi in situ on Gabin's tubulin (Cell01's first 6000 frames): FIT is `tubulin_fit.py`'s table with the bead model
BEAD, OUT the in-situ model's file.
``N`` molecules (3000) chosen from the odd blocks of 500 frames and learnt from, started from the bead model
(`tubulin_pupil.py 680 100 held`, which keeps its recipe); 2000 from the even blocks held out and localised on
both models, from three z starts each -- the held-out objective.  Saved as ``OUT`` (a Spline 4Pi model) with the
learnt values in ``OUT.pkl`` and the held-out numbers in ``OUT.heldout.json``.  ``SIGMA`` (rad) ties each
molecule's phase to the cavity's while learning (`InSituSettings.cavity_sigma_rad`); without it each has its own."""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from smappy import fourpi
from smappy.calibrate.fourpi_beads import model_from_calibration
from smappy.calibrate.fourpi_insitu import FourPiInSituSettings, InSituMolecules, choose, cut, z_range_of
from smappy.channels import ChannelLayout
from smappy.io.formats import load
from smappy.io.tiff import open_stack
from tubulin_common import MOVIE, PIXEL_UM, camera

from uipsf.workflows import fourpi_insitu as fi

BLOCK = 500
fit_path, bead_path, out = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
SIGMA = float(sys.argv[5]) if len(sys.argv) > 5 else None
dev = "mps" if torch.backends.mps.is_available() else "cpu"
locs = load(fit_path)[0]
cal = fourpi.load_fourpi_calibration(bead_path)
layout = ChannelLayout.from_dict(locs.metadata["channel_layout"])
px = (PIXEL_UM * 1000.0, PIXEL_UM * 1000.0)


def molecules(n, parity, seed):
    """The molecules cut, kept beside the fit: cutting them from the DCIMG takes minutes."""
    s = FourPiInSituSettings(molecules=n, seed=seed)
    cache = Path(fit_path).with_name(f"{Path(fit_path).stem}_molecules_{parity}_{n}_{seed}.npz")
    if cache.exists():
        z = np.load(cache)
        start = {k[6:]: z[k] for k in z.files if k.startswith("start_")}
        return InSituMolecules(z["data"], z["roi_xy"], start, z["frame"], z["index"]), s
    index = choose(locs, z_range_of(cal), s, frames=lambda f: (f // BLOCK) % 2 == parity)
    mol = cut(open_stack(str(MOVIE)), layout, locs, index, camera(), s.roi_size, px)
    np.savez(cache, data=mol.data, roi_xy=mol.roi_xy, frame=mol.frame, index=mol.index,
             **{"start_" + k: v for k, v in mol.start.items()})
    return mol, s


mol, s = molecules(N, 1, 0)
print("learning from", len(mol), "molecules; photons median", np.median(mol.start["photons"]),
      "z", np.percentile(mol.start["xyz"][:, 2], [5, 50, 95]).round(0).tolist(), flush=True)
model, shared = model_from_calibration(cal, device=dev)
batch = fi.molecule_batch(mol.data, mol.roi_xy, frame=mol.frame)
t = time.time()
res = fi.learn_insitu(model, shared, batch, mol.start, cal.period_nm,
                      settings=fi.InSituSettings(rounds=s.rounds, cavity_sigma_rad=SIGMA),
                      progress=lambda e: print(e.get("round"), e.get("stage"), e.get("iteration"), e.get("objective"),
                                               round(e.get("seconds", 0)), flush=True)
                      if e.get("iteration", 0) % 10 == 0 else None)
print("learnt in", round((time.time() - t) / 60, 1), "min", res.rounds, flush=True)
c = fi.calibration(res, z_range_nm=z_range_of(cal), dz_nm=float(cal.dz), roi_px=cal.shape[1] + 1)
c.layout = locs.metadata["channel_layout"]
fourpi.save_fourpi_calibration(out, c, overwrite=True)
pickle.dump({"shared": {k: v.cpu() for k, v in res.shared.items()}, "found": res.found, "kept": res.kept,
             "rounds": res.rounds, "index": mol.index}, open(out + ".pkl", "wb"))
for n in ("pupil.phase", "lo.pupil.phase", "fourpi.coherence", "fourpi.balance", "fourpi.share", "fourpi.delta",
          "fourpi.offset", "psf.sigma"):
    if n in shared:
        print(n, np.round(np.asarray(shared[n]), 3).tolist(), "->", np.round(res.shared[n].cpu().numpy(), 3).tolist())
print("period", cal.period_nm, "->", c.period_nm, flush=True)

held, _ = molecules(2000, 0, 1)
hb = fi.molecule_batch(held.data, held.roi_xy)
mm = fi.molecule_model(model, tuple(hb.data.shape[-2:]))
out_json = {"learnt": len(mol), "kept": int(len(res.kept)), "held": len(held)}
z = {}
for name, values in (("bead", shared), ("in situ", res.shared)):
    found = fi.localise(mm, values, hb, held.start, cal.period_nm)
    z[name] = found["xyz"][:, 2]
    out_json[name] = {"objective": float(found["objective"].sum()),
                      "mean deviance/px": float(np.mean(found["deviance"])),
                      "median deviance/px": float(np.median(found["deviance"]))}
d = z["in situ"] - z["bead"]
out_json["z in situ - bead: median, robust sd, beyond half a period"] = [
    float(np.median(d)), float(1.4826 * np.median(np.abs(d - np.median(d)))),
    float(np.mean(np.abs(d - np.median(d)) > cal.period_nm / 2))]
out_json["gain nats/molecule"] = (out_json["bead"]["objective"] - out_json["in situ"]["objective"]) / len(held)
print(json.dumps(out_json, indent=1))
json.dump(out_json, open(out + ".heldout.json", "w"), indent=1)
