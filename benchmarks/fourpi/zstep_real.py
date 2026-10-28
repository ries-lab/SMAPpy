"""(docs/4pi.md §18)  python benchmarks/fourpi/zstep_real.py scalar 1,2

uiPSF's real 4Pi beads (50 nm slices, 3 mirror steps, ±500 nm): models learnt from every n-th slice of 20
beads, each judged on 10 other beads' full 50 nm scans -- their mean deviance with the model held, each bead's
own parameters free.  Subsampling also drops photons, which counts against the coarse steps; at ~3e6 photons
per bead that costs little."""
import os, glob, json, sys, time, numpy as np, torch
from smappy.calibrate.psf_methods import FOURPI
from smappy.calibrate.fourpi_beads import FourPiBeadSettings, FourPiBeads
from smappy.calibrate.pupil import RunSettings
from smappy.metadata import CameraMetadata
import uipsf
from uipsf.model import PerObservation
from uipsf.model.fourpi import FourPiModel
from uipsf.model.fourpi_start import rough_start
from uipsf.noise import ShiftedPoisson
from uipsf.params import Params
from uipsf.solve import GaussNewtonLBFGS, Problem
from uipsf.workflows.fourpi import best_phases, learn_fourpi
OUT = os.environ.get("OUT", ".").rstrip("/") + "/"
KIND = sys.argv[1] if len(sys.argv) > 1 else "scalar"
EVERY = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4,5").split(",")]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
paths = sorted(glob.glob(os.path.expanduser("~/.cache/uipsf/zenodo/4pi_40nm_bead/bead*")))[:8]
inputs = {"fourpi": FourPiBeadSettings(layout="file"), "camera": CameraMetadata(conversion=0.44, offset=100.0, pixelsize_um=0.129),
          "run": RunSettings(save_model=False), "detection": None}
beads = FOURPI["average"].detect(paths, False, inputs).found["beads"]
E = beads.data.shape[0]
order = np.random.default_rng(0).permutation(E)
train, held = order[:20], order[20:30]
print("beads", beads.data.shape, "train", len(train), "held", len(held), flush=True)
optics = uipsf.Optics(NA=1.35, wavelength_nm=600, n_imm=1.406, n_med=1.406, pixel_size_nm=(129, 129))
grid = uipsf.PupilGrid.build(optics, 64, device=dev)


def model():
    return FourPiModel(grid, beads.data.shape[-2:], n_phases=3, n_max=6, steps_rad=[2.0, 0.0, -2.0],
                       bead_radius_nm=20.0, intensity=PerObservation(), kind=KIND)


def part(idx, every):
    keep = slice(None, None, every)
    return FourPiBeads(beads.data[idx][:, :, keep], beads.roi_xy[idx], beads.z_nm[keep],
                       [beads.records[i] for i in idx]).batch()


held_batch = part(held, 1).to(dev)
noise = ShiftedPoisson()
results = []
for every in EVERY:
    t = time.time()
    m = model()
    fit = learn_fourpi(m, part(train, every).to(dev))
    shared = {k: v.detach() for k, v in fit.shared().items()}
    start = rough_start(held_batch, m)
    p = Params(m.specs(held_batch, start), len(held), device=dev, values=shared)
    p = p.fix(*[n for n, s in p.specs.items() if s.scope == "shared"])
    p = best_phases(m, held_batch, p, noise)
    r = GaussNewtonLBFGS().solve(Problem(m, noise, held_batch), p)
    dv = Problem(m, noise, held_batch).deviance(r.params).numpy()
    rec = {"kind": KIND, "step_nm": 50 * every, "slices": len(beads.z_nm[::every]),
           "train_objective": float(fit.result.history[-1]["objective"]),
           "held_mean_deviance": float(dv.mean()),
           "coherence": fit.shared()["fourpi.coherence"].detach().cpu().numpy().round(3).tolist(),
           "minutes": (time.time() - t) / 60}
    results.append(rec)
    print(json.dumps(rec), flush=True)
    json.dump(results, open(OUT + f"real_subsample_{KIND}.json", "w"), indent=1)
