"""P2: held-out molecules (even files) localised on each model; mean deviance per pixel.  python heldout.py NAME=model.h5[.pkl] ..."""
import sys, os, glob, json, pickle, numpy as np, torch
from smappy import fourpi
from smappy.io.formats import load
from smappy.io.tiff import open_stack
from smappy.channels import ChannelLayout
from smappy.metadata import CameraMetadata
from smappy.calibrate.fourpi_beads import model_from_calibration
from smappy.calibrate.fourpi_insitu import FourPiInSituSettings, choose, cut
from uipsf.workflows import fourpi_insitu as fi
dev = "mps" if torch.backends.mps.is_available() else "cpu"
P = os.path.expanduser("~/.cache/uipsf/p2/")
cache = P + "heldout.npz"
if not os.path.exists(cache):
    locs = load(P + "fit_bead_v2.hdf5")[0]
    layout = ChannelLayout.from_dict(locs.metadata["channel_layout"])
    s = FourPiInSituSettings(molecules=2000, seed=1)
    index = choose(locs, 600.0, s, frames=lambda f: (f // 2000) % 2 == 0)
    paths = sorted(glob.glob(os.path.expanduser("~/.cache/uipsf/zenodo/4pi_smlm_nup96/*.mat")))
    mol = cut(open_stack(paths[0]), layout, locs, index, CameraMetadata(conversion=0.44, offset=100.0, pixelsize_um=0.129),
              13, (129.0, 129.0))
    np.savez(cache, data=mol.data, roi_xy=mol.roi_xy, **{"start_" + k: v for k, v in mol.start.items()})
z = np.load(cache)
start = {k[6:]: z[k] for k in z.files if k.startswith("start_")}
bead_model, bead_shared = model_from_calibration(fourpi.load_fourpi_calibration(P + "bead_v2_4pi.h5"), device=dev)
model = fi.molecule_model(bead_model, (13, 13))
batch = fi.molecule_batch(z["data"], z["roi_xy"])
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    shared = bead_shared if path == "bead" else pickle.load(open(path, "rb"))["shared"]
    found = fi.localise(model, shared, batch, start, 265.0)
    d = found["deviance"]
    print(json.dumps({"model": name, "molecules": len(d), "mean deviance/px": float(np.mean(d)),
                      "median deviance/px": float(np.median(d)), "total objective": float(found["objective"].sum())}), flush=True)
