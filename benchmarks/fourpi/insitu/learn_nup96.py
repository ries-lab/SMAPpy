"""P2: learn the 4Pi model in situ from the Nup96 movie's odd files.  python learn_nup96.py FIT.hdf5 BEADMODEL.h5 OUT.h5 [N]"""
import sys, os, glob, time, json, pickle, numpy as np, torch
from smappy import fourpi
from smappy.io.formats import load
from smappy.io.tiff import open_stack
from smappy.channels import ChannelLayout
from smappy.metadata import CameraMetadata
from smappy.calibrate.fourpi_beads import model_from_calibration
from smappy.calibrate.fourpi_insitu import FourPiInSituSettings, choose, cut
from uipsf.workflows import fourpi_insitu as fi
fit_path, bead_path, out = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
dev = "mps" if torch.backends.mps.is_available() else "cpu"
paths = sorted(glob.glob(os.path.expanduser("~/.cache/uipsf/zenodo/4pi_smlm_nup96/*.mat")))
locs = load(fit_path)[0]
cal = fourpi.load_fourpi_calibration(bead_path)
layout = ChannelLayout.from_dict(locs.metadata["channel_layout"])
s = FourPiInSituSettings(molecules=N)
index = choose(locs, 600.0, s, frames=lambda f: (f // 2000) % 2 == 1)
print("chosen", len(index), "of", len(locs), flush=True)
camera = CameraMetadata(conversion=0.44, offset=100.0, pixelsize_um=0.129)
t = time.time()
mol = cut(open_stack(paths[0]), layout, locs, index, camera, s.roi_size, (129.0, 129.0), progress=print)
print("cut", len(mol), "in", time.time() - t, "s; photons median", np.median(mol.start["photons"]), flush=True)
np.savez(out + ".molecules.npz", data=mol.data, roi_xy=mol.roi_xy, frame=mol.frame, index=mol.index,
         **{"start_" + k: v for k, v in mol.start.items()})
model, shared = model_from_calibration(cal, device=dev)
batch = fi.molecule_batch(mol.data, mol.roi_xy, frame=mol.frame)
t = time.time()
res = fi.learn_insitu(model, shared, batch, mol.start, cal.period_nm,
                      settings=fi.InSituSettings(rounds=s.rounds),
                      progress=lambda e: print(e.get("round"), e.get("stage"), e.get("iteration"), e.get("objective"),
                                               round(e.get("seconds", 0)), flush=True)
                      if e.get("iteration", 0) % 5 == 0 else None)
print("learnt in", (time.time() - t) / 60, "min", res.rounds, flush=True)
c = fi.calibration(res, z_range_nm=600.0, dz_nm=20.0, roi_px=21)
c.layout = locs.metadata["channel_layout"]
fourpi.save_fourpi_calibration(out, c, overwrite=True)
pickle.dump({"shared": {k: v.cpu() for k, v in res.shared.items()}, "found": res.found, "kept": res.kept,
             "rounds": res.rounds, "index": mol.index}, open(out + ".pkl", "wb"))
for n in ("pupil.phase", "lo.pupil.phase", "fourpi.coherence", "fourpi.balance", "fourpi.delta", "fourpi.offset", "psf.sigma"):
    print(n, np.round(np.asarray(shared[n]), 3).tolist(), "->", np.round(res.shared[n].cpu().numpy(), 3).tolist())
print("period", c.period_nm)
