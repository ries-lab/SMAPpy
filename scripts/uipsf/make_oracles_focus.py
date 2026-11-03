"""Write tests/uipsf_tests/fixtures/oracle_focus.npz: focal fields from Leutenegger's library, the excitation oracle.

Leutenegger's vectorial Debye library (chirp-z, Leutenegger et al. 2006, LGPL 2.1) as SimuFLUX ships it
(``SimuFLUX/python/psfs/library``) is uipsf's oracle for focusing, never part of it (scanned-psfs.md, "Oracle
and licences"): this script imports it from a SimuFLUX checkout and writes its fields, so that the test needs
neither the library nor SimuFLUX.

    python scripts/uipsf/make_oracles_focus.py --simuflux ~/git/SimuFLUX/python

Each case is a beam at 640 nm through an NA 1.4 oil objective, the aperture filled by a Gaussian of waist 7 mm
on a 6.5 mm aperture (SimuFLUX's default), with the lab's masks.  Stored per case: ``E`` [3, X, Y, Z] complex
(E_x, E_y, E_z; the library's first lateral axis is x), the coordinates ``x``, ``y``, ``z`` in nm, and the
settings as JSON.  ``depth`` cases focus into water: the library focuses at the interface and z is the depth
below it, which is uipsf's emitter depth with the objective unmoved.

Also stored, ``lens.sin``, ``lens.tp``, ``lens.ts``: the library's objective transmission (its "air-lens-air"
model, `library/transmission.py`) tabulated against sin θ in the immersion, so that the test can weight uipsf's
fields as the library does without carrying the library's code.  What the comparison found
(`tests/uipsf_tests/test_excitation.py`):
- the library writes E_z = t_p sin θ_imm / n_s, which is t_p sin θ_s / n_m: its E_z is n_m too small;
- its pupil's x runs against uipsf's u (it transforms with exp(−i k·r)): a mask it shifts by s along x sits at
  u = −s in its own coordinates, at +s in uipsf's;
- it weights the pupil by 1/cos θ_imm and its lens model, uipsf by the aplanatic √cos θ_imm / cos θ_med.
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--simuflux", default=os.path.expanduser("~/git/SimuFLUX/python"),
                help="SimuFLUX's python folder, the one holding psfs/")
FIXTURE = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "uipsf_tests", "fixtures",
                       "oracle_focus.npz")
ap.add_argument("--out", default=FIXTURE)
args = ap.parse_args()
sys.path.insert(0, os.path.abspath(args.simuflux))
from psfs.library import effField  # noqa: E402
from psfs.library import modifiers as _modifiers  # noqa: E402
from psfs.library.math import cart2pol, pol2cart  # noqa: E402
from psfs.library.transmission import transmission  # noqa: E402


def phaseramp_after_gauss(sys, E, r, t, p):
    """SimuFLUX's `phaseramp` (Jonas Ries's) for a field that is already a map: the library's tiles E as if it
    were one row, which fails after `gauss`, so SimuFLUX never fills its vortex with a Gaussian."""
    xx, yy = pol2cart(p, r)
    p, r = cart2pol(xx + sys["maskshift"][1], yy + sys["maskshift"][0])
    return E * np.exp(1j * p)


_modifiers.phaseramp2 = phaseramp_after_gauss       # `modifiers` looks the modifiers up in its own globals

BASE = dict(Da=0.0065, Mt=100, NA=1.4, Na=200, Po=1e-5, ft=0.25, lo=640e-9, nl=1.518, nm=1.518, ns=1.518,
            wa=0.007, maskshift=[0, 0], pl=0, rx=500e-9, ry=500e-9, rz=400e-9, xo=0, yo=0, zo=0)
CASES = {
    "gauss_x": dict(Ei=["gauss", "linear"]),
    "gauss_circular": dict(Ei=["gauss", "circular"]),
    "vortex": dict(Ei=["gauss", "phaseramp2", "circular"]),
    "vortex_shifted": dict(Ei=["gauss", "phaseramp2", "circular"], maskshift=[0.0, 0.1]),
    "tophat": dict(Ei=["gauss", "pishift", "circular"]),
    "flat_x": dict(Ei=["linear"], wa=1.0),
    "depth_water": dict(Ei=["gauss", "circular"], ns=1.33, NA=1.35, zo=2000e-9),
}
STEP = dict(dr=25e-9, dz=200e-9)

out = {}
for name, change in CASES.items():
    s = dict(BASE, **change)
    r = effField(dict(s), dict(STEP), {"mem": 50000})
    out[f"{name}.E"] = r["E"].astype(np.complex64)
    out[f"{name}.x"] = np.asarray(r["x"]).ravel() * 1e9
    out[f"{name}.y"] = np.asarray(r["y"]).ravel() * 1e9
    out[f"{name}.z"] = np.asarray(r["z"]).ravel() * 1e9
    out[f"{name}.settings"] = np.array(json.dumps({k: v for k, v in s.items()}))
    print(name, r["E"].shape, flush=True)
sin = np.linspace(0.0, 0.99, 1000)
tp, ts = transmission(dict(BASE), np.arcsin(sin))
out["lens.sin"], out["lens.tp"], out["lens.ts"] = sin, np.real(tp), np.real(ts)
np.savez_compressed(args.out, **out)
print("wrote", os.path.abspath(args.out))
