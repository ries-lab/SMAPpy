"""Time uiPSF v1 (TensorFlow) on Zenodo `1ch_40nm_bead`, as its demo notebook runs it (plan §9, M7).

Run in an environment with TensorFlow and v1's dependencies, against uiPSF's ``test_roi`` b367f00:

    python scripts/uipsf/time_v1_zenodo.py --uipsf path/to/uiPSF --data $UIPSF_DATA/1ch_40nm_bead

The settings are the demo's (`demo/demo_beadPSF_1ch.ipynb`, which made the Zenodo result file): M2
system file, gain 0.22, offset 400, 20 beads, 200 L-BFGS-B iterations, n_max 8, bin 1.  One JSON line:
the time of each step (loading, ROIs, learning with v1's rejection and relearning, localisation) and
L-BFGS-B's evaluation counts, so v2's time can be set beside it.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

np.product = np.prod                                    # NumPy 2 shims for the v1 code
np.string_ = np.bytes_

ap = argparse.ArgumentParser()
ap.add_argument("--uipsf", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--iterations", type=int, default=200)
args = ap.parse_args()
out = os.path.abspath(args.out) if args.out else None
sys.path.insert(0, os.path.abspath(args.uipsf))
os.chdir(os.path.join(os.path.abspath(args.uipsf), "demo"))   # config paths are relative to it

from unittest import mock  # noqa: E402

# results writing (pins an old PyYAML), dialogs, the field-dependent model (needs a matching TF Probability)
for _name in ("hdfdict", "tkinter", "tkinter.messagebox", "tensorflow_probability", "czifile"):
    sys.modules[_name] = mock.MagicMock()
import tensorflow as tf  # noqa: E402
from psflearning import io  # noqa: E402
from psflearning.learning import optimizers  # noqa: E402
from psflearning.psflearninglib import psflearninglib  # noqa: E402

evaluations = []                                       # every L_BFGS_B run's scipy status
_minimize = optimizers.L_BFGS_B.minimize


def minimize(self, *a, **k):
    out = _minimize(self, *a, **k)
    st = getattr(self, "status", None)
    if st is not None:
        evaluations.append(dict(maxiter=self.maxiter, nit=int(st.nit), nfev=int(st.nfev), message=str(st.message)))
    return out


optimizers.L_BFGS_B.minimize = minimize

L = psflearninglib()
L.param = io.param.combine("config_base", psftype="zernike", channeltype="1ch", sysfile="M2")
L.param.datapath = os.path.abspath(args.data) + "/"
L.param.savename = os.path.join(os.environ.get("TMPDIR", "/tmp"), "v1_psfmodel")
L.param.keyword = "Pos"
L.param.subfolder = "Pos"
L.param.gain = 0.22
L.param.ccd_offset = 400
L.param.FOV.z_step = 1
L.param.pixel_size.z = 0.05
L.param.roi.max_bead_number = 20
L.param.roi.bead_radius = 0.025
L.param.batch_size = 30
L.param.iteration = args.iterations
L.param.option.imaging.emission_wavelength = 0.6
L.param.plotall = False

t = {}
t0 = time.perf_counter()
images = L.load_data()
t["load_s"] = time.perf_counter() - t0
L.getpsfclass()
t0 = time.perf_counter()
dataobj = L.prep_data(images)
dataobj.rois = dataobj.rois.astype(np.float32)        # float64 under NumPy 2's promotion; v1 assumed float32
t["rois_s"] = time.perf_counter() - t0
t0 = time.perf_counter()
psfobj, fitter = L.learn_psf(dataobj, time=0)
t["learn_s"] = time.perf_counter() - t0
row = dict(program="uiPSF v1", dataset="1ch_40nm_bead", beads=int(dataobj.rois.shape[0]),
           **{k: round(v, 1) for k, v in t.items()}, lbfgs=evaluations,
           threads=tf.config.threading.get_intra_op_parallelism_threads() or os.cpu_count(),
           system=platform.system(), tensorflow=tf.__version__)
print(json.dumps(row), flush=True)
if out:
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
