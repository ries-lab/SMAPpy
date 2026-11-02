"""Spline 3D's *focus in situ* (`smappy.calibrate.depth.insitu_focus`) on M2's January movie, from a minimal bead
calibration made with the window's defaults: is the focus stable from different starts, and sensible?

    python benchmarks/uipsf/real/insitu_focus_m2.py OUT_FOLDER     # the calibration is kept there

Every aberration of the bead model held, the focus and the blur learnt, from its own estimate, 300 and 900 nm;
on the CPU.  Prints one JSON line per start (`results/insitu_minimal.jsonl`, step "focus in situ").
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import smappy.calibrate.insitu_run as ir
from smappy.calibrate import depth
from smappy.calibrate.pupil import BeadSettings, OpticsSettings, run_calibration
from smappy.io.calibration import load_spline_calibration

D = "/Users/ries/Library/Mobile Documents/com~apple~CloudDocs/Vienna/Data/smappy test"
OUT = Path(sys.argv[1])
base = ir.run_insitu
# on the CPU: the Mac is shared, one GPU job at a time
ir.run_insitu = lambda *a, **k: base(*a, **dict(k, device="cpu"))
o = OpticsSettings(NA=1.43, wavelength_nm=680.0, n_immersion=1.516, n_medium=1.33, pixel_x_nm=116.0, pixel_y_nm=127.0)
cal_path = OUT / "jan_minimal_3dcal.h5"
if not cal_path.exists():
    files = sorted(glob.glob(f"{D}/260105_AA_BeadsCalZ_638_M2/*/*.ome.tif"))
    run_calibration(files, False, o, BeadSettings(), save_to=cal_path)
cal = load_spline_calibration(str(cal_path))
MOVIE = "260105_AA_CLTA-B-SNAP_AF647_C8_638i75_UVi20_25ms_3D_1"
movie = [f"{D}/{MOVIE}/{MOVIE}_MMStack_Default.ome.tif"]
for start in [None, 300.0, 900.0]:
    t = time.time()
    r = depth.insitu_focus(cal, movie, None, focus_nm=start)
    h = np.asarray(r.heights)
    print(json.dumps(dict(step="focus in situ", start_nm=start, focus_nm=round(r.focus_nm, 1),
                          edge_sd_nm=round(float(r.edge_sd_nm), 1),
                          spike=round(float(r.spike), 3), molecules=r.n, blur_nm=r.blur_nm,
                          heights_nm=np.nanpercentile(h, [5, 25, 50, 75, 95]).round(0).tolist() if len(h) else None,
                          notes=r.notes, minutes=round((time.time() - t) / 60, 1))), flush=True)
