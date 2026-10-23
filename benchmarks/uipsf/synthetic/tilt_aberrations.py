"""What a tilted coverslip does to the pupil: the layers' phase in Zernikes (rad, Noll-normalised RMS).

The phase a tilt adds, 2π·[(c_h(t) − c_h(0))·h + c_0(t) − c_0(0)], fitted over the aperture by the
Zernikes 5 … 37 beside piston, the two lateral shifts and the exact high-NA defocus k_z,imm, what an
emitter's own position absorbs (`uipsf.optics.layers.aberrations`); the shift, which the model takes out,
is given in nm.  For an oil objective on an aqueous sample and a water objective through a 170 µm
coverslip, at the depths and tilts asked for.

    python benchmarks/uipsf/synthetic/tilt_aberrations.py
"""
from __future__ import annotations

import math

from uipsf.optics import Optics
from uipsf.optics.layers import aberrations
from uipsf.optics.zernike import zernike_name

OBJECTIVES = {
    "oil NA 1.43 (n 1.518), sample 1.33": Optics(NA=1.43, wavelength_nm=670, n_imm=1.518, n_med=1.33,
                                                    pixel_size_nm=(110, 110)),
    "water NA 1.2 (n 1.33), coverslip 1.52": Optics(NA=1.2, wavelength_nm=670, n_imm=1.33, n_med=1.33,
                                                       n_cov=1.52, pixel_size_nm=(110, 110)),
}
SHOWN = (5, 6, 7, 8, 11, 12, 13, 16, 17)


def main():
    for name, optics in OBJECTIVES.items():
        print(f"\n{name}; tilt along x; rad (Noll RMS), shift nm")
        print("tilt°  depth nm  " + "  ".join(f"{j:>6}" for j in SHOWN) + "   shift_x")
        for deg in (0.5, 1.0, 2.0):
            for h in (0.0, 500.0, 1000.0):
                z, s = aberrations(optics, (math.radians(deg), 0.0), h)
                print(f"{deg:4.1f}  {h:7.0f}   " + "  ".join(f"{z[j]:+6.3f}" for j in SHOWN) + f"   {s[0]:+8.1f}")
    print("\n" + ", ".join(f"{j} {zernike_name(j)}" for j in SHOWN))


if __name__ == "__main__":
    main()
