"""(docs/4pi.md §26)  PYTHONPATH=src python benchmarks/fourpi/tubulin_average.py

Step 2a on the tubulin data: the 4Pi bead average (`fourpi.iab_from_beads`) of the three one-bead scans
Cell03 -- 05 learnt together, each scan at its own phase and, as the cavity drifted through each, at its own
rate -- against one rate for all.  The boxes cut by the registration of the bead fields Cell06 -- 12
(`tubulin_layout.py`): the scans' own three beads sit within a pixel of each other and fix a shift alone.
Each model is fitted back to the three scans (`fourpi_beads.bead_fits`): the envelope's z and the phase's
against the stage.  Saved as ``smappy_p1/average_<rates>_4pi.h5``."""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tubulin_common import OUT, REGISTRATION, SCANS, camera

from smappy import fourpi
from smappy.calibrate import fourpi_beads as fb
from smappy.channels import FourPiChannelsSettings

settings = fb.FourPiBeadSettings(channels=FourPiChannelsSettings(layout="yale"), mirror_steps="",
                                 registration=fb.ScanRegistrationSettings(source="file", path=str(REGISTRATION)))
beads, layout, _ = fb.scan_beads([str(p) for p in SCANS], settings, camera())
print("beads", beads.data.shape, "at", [r["exact_xy"][0] for r in beads.records])
found = {"beads": beads, "layout": layout}
original = fourpi.iab_from_beads


def one_rate(*args, scan_of=None, **kwargs):
    return original(*args, **kwargs)


for name, patch in (("one", one_rate), ("per_scan", None)):
    fourpi.iab_from_beads = patch or original
    cal = fb.average_model(found, settings=settings)
    fourpi.iab_from_beads = original
    fourpi.save_fourpi_calibration(OUT / f"average_{name}_4pi.h5", cal, overwrite=True)
    p = cal.parameters
    fits = fb.bead_fits(beads, cal)
    print(f"\n{name} rate: period {2 * np.pi / abs(cal.phase_per_nm):.1f} nm, solved {p['channels']} "
          f"(condition {p['condition']:.2f}), rates per slice {np.round(p['w_per_bead'], 4).tolist()}, "
          f"z {cal.z_range_nm if hasattr(cal, 'z_range_nm') else ''}")
    print("  gains", np.round(p["gains"], 3).tolist(), "offsets", np.round(p["offsets_rad"], 3).tolist())
    middle = slice(len(beads.z_nm) // 4, 3 * len(beads.z_nm) // 4)
    for b in range(beads.data.shape[0]):
        phase = np.unwrap(fits["phase"][b, middle, 0])
        resid = phase - np.polyval(np.polyfit(beads.z_nm[middle], phase, 1), beads.z_nm[middle])
        print(f"  scan {b}: z slope {fits['slope'][b]:.3f}, scatter {fits['scatter'][b]:.1f} nm; "
              f"phase slope {fits['phase_slope'][b]:.3f}, phase about its line {np.std(resid):.3f} rad "
              f"({np.std(resid) / abs(cal.phase_per_nm):.1f} nm)")

# leave one scan out: a model of the other two (with two beads the channels share one PSF), the third
# fitted with it -- what a model does on data it has not seen
print("\nleave one scan out (model of the other two, shared PSF):")
for name, patch in (("one", one_rate), ("per_scan", None)):
    for held in range(3):
        rest = [i for i in range(3) if i != held]
        part = fb.FourPiBeads(beads.data[rest], beads.roi_xy[rest], beads.z_nm, [beads.records[i] for i in rest])
        fourpi.iab_from_beads = patch or original
        cal = fb.average_model({"beads": part, "layout": layout}, settings=settings)
        fourpi.iab_from_beads = original
        one = fb.FourPiBeads(beads.data[[held]], beads.roi_xy[[held]], beads.z_nm, [beads.records[held]])
        fits = fb.bead_fits(one, cal)
        middle = slice(len(beads.z_nm) // 4, 3 * len(beads.z_nm) // 4)
        phase = np.unwrap(fits["phase"][0, middle, 0])
        resid = phase - np.polyval(np.polyfit(beads.z_nm[middle], phase, 1), beads.z_nm[middle])
        print(f"  {name:8s} scan {held} held: z slope {fits['slope'][0]:.3f}, scatter {fits['scatter'][0]:.1f} nm; "
              f"phase slope {fits['phase_slope'][0]:.3f}, about its line {np.std(resid):.3f} rad")
