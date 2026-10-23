"""uiPSF v1 (TensorFlow) on a bead / movie pair of one microscope: does v1's in situ astigmatism agree with its beads'?

Run in v1's environment (TensorFlow, uiPSF's ``psflearning``), as its demos do (`demo_beadPSF_1ch.ipynb`,
`demo_insituPSF_1ch.ipynb`), with the same optics for both:

    python benchmarks/uipsf/real/v1_pair.py beads --uipsf ~/git/uiPSF --data BEAD_FOLDER --gain 0.268 \\
        --offset 398.6 --save OUT/beads
    python benchmarks/uipsf/real/v1_pair.py insitu --uipsf ~/git/uiPSF --data MOVIE_FOLDER --keyword NAME \\
        --gain 0.067 --offset 398.6 --frames 20000 24000 --stage-pos 0.45 --save OUT/insitu

Why (October 2026): v2's in situ learns 0.3-0.4 rad more primary astigmatism than the same day's beads on M2,
at every depth, whatever σ or the index; v1's own beads and in situ agree on Zenodo's FD pair.  If v1 shows the
gap on M2's January pair too, it is the data's or physics both share; if not, it is in v2.  ``--gain`` is
photons per ADU (v1's): SMAPpy's conversion over the EM gain.  The M2 system file is v1's own (127 x 116 um
pixels with swapxy, as Zenodo's M2 Evolve beads), so v1's coefficients are in its convention, which flips
Noll 6's sign against v2's.  Prints one JSON line: the learnt phase (Noll 2-28, v1 index i is Noll i + 1),
the magnitude, σ in pixels and the result file.
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("beads", "insitu"))
ap.add_argument("--uipsf", required=True)
ap.add_argument("--data", required=True, help="the folder v1 reads (beads: one subfolder per stack)")
ap.add_argument("--keyword", default="MMStack")
ap.add_argument("--subfolder", default="")
ap.add_argument("--gain", type=float, required=True, help="photons per ADU")
ap.add_argument("--offset", type=float, required=True)
ap.add_argument("--wavelength", type=float, default=0.68, help="um")
ap.add_argument("--na", type=float, default=1.43)
ap.add_argument("--n-max", type=int, default=6)
ap.add_argument("--frames", type=int, nargs=2, default=None)
ap.add_argument("--stage-pos", type=float, default=0.45, help="um, the focus above the coverslip's")
ap.add_argument("--start-astigmatism", type=float, default=None,
                help="v1's start (Noll 6, its index 5), rad; default the system file's (M2: +0.5)")
ap.add_argument("--bead-radius", type=float, default=0.05, help="um")
ap.add_argument("--save", required=True, help="the result file's path without extension")
args = ap.parse_args()
sys.path.insert(0, os.path.abspath(args.uipsf))
os.chdir(os.path.join(os.path.abspath(args.uipsf), "demo"))          # config paths are relative to it

from psflearning import io  # noqa: E402
from psflearning.psflearninglib import psflearninglib  # noqa: E402

L = psflearninglib()
L.param = io.param.combine("config_base", psftype="zernike" if args.mode == "beads" else "insitu",
                           channeltype="1ch", sysfile="M2")
L.param.datapath = os.path.abspath(args.data) + "/"
L.param.savename = os.path.abspath(args.save)
L.param.keyword = args.keyword
L.param.subfolder = args.subfolder
L.param.gain, L.param.ccd_offset = args.gain, args.offset
L.param.option.imaging.emission_wavelength = args.wavelength
L.param.option.imaging.NA = args.na
L.param.option.model.n_max = args.n_max
if args.mode == "beads":
    L.param.FOV.z_step = 1
    L.param.pixel_size.z = 0.02                                       # the January stacks' 20 nm steps
    L.param.roi.bead_radius = args.bead_radius
    L.param.batch_size = 30
    images = L.load_data()
    L.getpsfclass()
    dataobj = L.prep_data(images)
    psfobj, fitter = L.learn_psf(dataobj, time=0)
    resfile = L.save_result(psfobj, dataobj, fitter)
else:
    if args.frames:
        L.param.insitu.frame_range = list(args.frames)
    L.param.roi.peak_height = 0.3
    L.param.option.insitu.stage_pos = args.stage_pos
    L.param.option.insitu.var_stagepos = True                        # learnt, as v2 learns its focus
    if args.start_astigmatism is not None:
        # the start sets the sign v1 learns: M2's file says +0.5, which on January's movie gave the twin
        L.param.option.insitu.zernike_index = [5]
        L.param.option.insitu.zernike_coeff = [args.start_astigmatism]
    images = L.load_data()
    L.getpsfclass()
    dataobj = L.prep_data(images)
    resfile = L.iterlearn_psf(dataobj, time=0)

f, p = io.h5.load(resfile)
z = np.asarray(f.res.zernike_coeff)
phase, magnitude = (z[1], z[0]) if z.ndim == 2 else (z[1].mean(0), z[0].mean(0))
print(json.dumps(dict(mode=args.mode, data=args.data, resfile=str(resfile),
                      phase={str(i + 1): round(float(c), 4) for i, c in enumerate(phase)},
                      magnitude={str(i + 1): round(float(c / magnitude[0]), 4) for i, c in enumerate(magnitude)},
                      sigma_px=np.asarray(f.res.sigma).round(3).tolist(),
                      stagepos=np.asarray(f.res.stagepos).tolist() if "stagepos" in f.res else None)), flush=True)
