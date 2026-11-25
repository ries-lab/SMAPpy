"""The astigmatic counterpart of the Nup96 v2 4Pi bead model (docs/4pi.md §33): one objective's PSF.

The v2 bead model (§27) learnt one pupil per objective; their astigmatism is opposite (Noll 6: +46 nm in the
first, -32 nm in the second), so the incoherent sum of the 4Pi channels (their I parts) is nearly round and
would be a poor astigmatic microscope.  The counterpart is the first objective's pupil alone, as it was learnt --
every phase mode, γ, the blur -- drawn by uiPSF (`export.from_description`) at the model's NA, emission, immersion
and pixel, sampled into a Spline 3D calibration as *PSF calibration* samples a pupil model
(`calibrate.pupil.spline_calibration`).  The same optics and aberrations as the 4Pi image, minus the second
objective and the interference.

    python make_astig.py [MODEL_4PI.h5] [OUT_3dcal.h5]
"""
import sys
from pathlib import Path
from types import SimpleNamespace

from smappy import fourpi
from smappy.calibrate.pupil import spline_calibration
from smappy.io.calibration import save_spline_calibration

MODEL = Path.home() / ".cache/uipsf/p2/bead_v2_4pi.h5"
OUT = Path.home() / ".cache/uipsf/p2/astig_v2_upper_3dcal.h5"
DZ_NM, Z_RANGE_NM, ROI_PX = 20.0, 800.0, 27


def main():
    from uipsf.export.model_file import from_description
    from uipsf.optics import Optics
    model = Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    cal = fourpi.load_fourpi_calibration(model)
    pupil = dict(cal.parameters["pupils"][0])
    pixel = [float(v) for v in cal.pixelsize_nm]
    optics = Optics(NA=float(pupil["NA"]), wavelength_nm=float(pupil["wavelength_nm"]),
                    n_imm=float(pupil["n_immersion"]), n_med=float(pupil["n_medium"]), pixel_size_nm=tuple(pixel))
    learned = from_description(pupil, optics, roi_px=ROI_PX)
    camera = SimpleNamespace(is_em=False, pixelsize_um=[pixel[0] / 1000, pixel[1] / 1000])
    spline = spline_calibration(learned, DZ_NM, Z_RANGE_NM, ROI_PX, camera=camera)
    spline.parameters.update(source=f"the first pupil of {model.name}, one objective (docs/4pi.md §33)")
    save_spline_calibration(out, spline, overwrite=True)
    print("wrote", out, spline.coeff.shape if hasattr(spline, "coeff") else "")


if __name__ == "__main__":
    main()
