"""(docs/insitu-fitting-plan.md, phase 3 on real data)  Spline 3D 2C with *focus in situ* on uiPSF's Zenodo
two-colour movie, `2ch_smlm_nup96_WGA` (Nup96 and WGA in U2OS, the chip split in two on M2), as a user would
run it: a dual pupil calibration from the same microscope's `2ch_40nm_bead` stacks, then the movie's first N
frames fitted (of the 17024 the archive holds of the series)

* with the focus on *auto* (the bead model drawn at the focus found from the lowest localizations),
* in situ, *learn* = focus,
* in situ, *learn* = aberrations,
* in situ, *learn* = aberrations with *learn registration*,

each with the shipped defaults otherwise (a share of the light per molecule where *link photons* is off).

    PYTHONPATH=src python benchmarks/uipsf/real/insitu_dual_zenodo.py OUT [--frames N] [--runs ...]

Printed, one JSON line per run (and in ``benchmarks/uipsf/results/insitu_dual_zenodo.jsonl``): the focus, the
held-out deviance per model, whether it fell back and why, the learnt channel offset in z and share, and the
two measures that need no truth from `insitu_learn_m2` -- consecutive-frame *blink pairs* (robust sd over the
errors, 1 when model and errors are honest) and the share *below the coverslip*.  Set against uiPSF v1's in
situ model of the same movie (``results/2ch_smlm_nup96_WGA``) only by eye, through the learnt Zernikes printed.
"""
import argparse
import glob
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from insitu_learn_m2 import error_columns, pairs  # noqa: E402

DATA = Path(os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo")))
MOVIE = DATA / "2ch_smlm_nup96_WGA" / "01_191009_u2os_course_96_WGA_3D_ritu_1_MMStack_Pos0.ome.tif"
RUNS = {"auto": ("auto", "focus", False), "focus": ("insitu", "focus", False),
        "aberrations": ("insitu", "aberrations", False), "registration": ("insitu", "aberrations", True)}
SHOWN = ("pupil.phase", "ch1.pupil.phase", "ch1.dz", "ch1.share")


def calibration(out: Path) -> Path:
    """The dual pupil calibration of the Zenodo bead stacks, with the window's defaults (kept in OUT)."""
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, run_calibration
    from zenodo_camera import CAMERA  # the beads' camera: normal port, 4.7 e-/ADU, 116 x 127 nm pixels
    path = out / "zenodo_2ch_beads_3dcal.h5"
    if not path.exists():
        files = sorted(glob.glob(str(DATA / "2ch_40nm_bead" / "*" / "*.ome.tif")))
        o = OpticsSettings(NA=1.43, wavelength_nm=680.0, n_immersion=1.516, n_medium=1.35,
                           pixel_x_nm=116.0, pixel_y_nm=127.0)
        run_calibration(files, True, o, BeadSettings(), camera=CAMERA, save_to=path)
    return path


def rounded(v):
    v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    return np.round(v.astype(float), 3).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--frames", type=int, default=10000)
    p.add_argument("--runs", nargs="*", default=list(RUNS))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from smappy.calibrate import depth
    from smappy.plugins import Context
    from smappy.plugins.fit import (
        CameraSettings,
        DualSplineFit,
        DualSplineFitSettings,
        OutputSettings,
        SourceSettings,
    )

    from uipsf.device import default_device
    results = Path(__file__).resolve().parents[1] / "results" / "insitu_dual_zenodo.jsonl"
    t = time.time()
    cal = calibration(out)
    print(f"calibration {cal} ({(time.time() - t) / 60:.1f} min)", flush=True)

    for name in args.runs:
        focus, learn, registration = RUNS[name]
        # the camera from the movie's tags (M2's Evolve, EM gain 200, 6.7 e-/ADU) but its pixels, which the
        # database has square: 116 x 127 nm in the camera's frame, as the beads were calibrated (docs/uipsf/data.md)
        s = DualSplineFitSettings(source=SourceSettings(path=str(MOVIE), stop=args.frames),
                                  camera=CameraSettings(pixelsize_um=0.116, pixelsize_y_um=0.127),
                                  output=OutputSettings(path=str(out / f"nup96_wga_{name}_locs.hdf5")))
        s = replace(s, model=replace(s.model, calibration=str(cal), sample=replace(s.model.sample, focus=focus),
                                     learn=learn, learn_registration=registration))
        plugin = DualSplineFit()        # its run learns, as the GUI's Run after the preflight
        ctx = Context(progress=lambda text: None)
        t = time.time()
        result = plugin.run(ctx, s)
        minutes = (time.time() - t) / 60
        found = depth.found(plugin._insitu_key(s)) if focus == "insitu" else None
        locs = result.locs
        z = np.asarray(locs["z_nm"], float)
        row = dict(run=name, device=str(default_device()), frames=args.frames, localizations=int(len(locs)),
                   used_focus=result.settings.model.sample.focus,
                   used_calibration=result.settings.model.calibration,
                   focus_nm=None if found is None else round(found.focus_nm, 1),
                   fallback=None if found is None else found.fallback,
                   heldout=None if found is None else found.heldout,
                   messages=None if found is None else [str(m) for m in found.messages],
                   blink_pairs=pairs(locs, error_columns(locs.columns)),
                   below_coverslip=round(float(np.mean(z < -50.0)), 4),
                   z_percentiles=np.nanpercentile(z, [5, 25, 50, 75, 95]).round(0).tolist(),
                   channels={str(c): int(n) for c, n in zip(*np.unique(np.asarray(locs["channel"]),
                                                                        return_counts=True), strict=True)}
                   if "channel" in locs.columns else None,
                   warnings=[str(w) for w in getattr(ctx, "warnings", [])], minutes=round(minutes, 1))
        drawn = getattr(found, "drawn", None)
        if drawn is not None:
            row["learnt"] = {k: rounded(drawn.shared[k]) for k in SHOWN if k in drawn.shared}
        line = json.dumps(row, default=str)
        print(line, flush=True)
        with results.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    sys.exit(main())
