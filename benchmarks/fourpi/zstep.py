"""How coarse a 4Pi bead scan's z step can be, and how long its range must be (docs/4pi.md §18).

    TOTAL=3000000 HALF_RANGE=500 python benchmarks/fourpi/zstep.py 20,50,75,100 0,1,2 12 mps

  Bead stacks simulated from a known 4Pi model (scalar, uiPSF's
example optics, opposite astigmatism in the arms, coherence ~0.8, 40 nm beads, 3 mirror steps) at several z
steps, the same photons and background per bead whatever the step; a model learnt from each, exported, and
judged on simulated single molecules fitted with it: the envelope's z and z from the phase against the truth.
Env: HALF_RANGE (nm, the scan's ±), TOTAL (photons per bead).  Args: steps, seeds, beads, device."""
import json, math, os, sys, time, numpy as np, torch
import uipsf
from uipsf.model import PerObservation
from uipsf.model.fourpi import FourPiModel
from uipsf.params import Params
from uipsf.simulate import simulate
from uipsf.workflows.fourpi import calibration, learn_fourpi
from smappy import fourpi
from smappy.psf import Global4PiPSF

OUT = os.environ.get("OUT", ".").rstrip("/") + "/"           # where the results' JSON goes
STEPS = [float(s) for s in sys.argv[1].split(",")]
SEEDS = [int(s) for s in sys.argv[2].split(",")]
E = int(sys.argv[3])
DEVICE = sys.argv[4]
C, M, R = 4, 3, 21
HALF_RANGE = float(os.environ.get("HALF_RANGE", 500))
TOTAL_PHOTONS = float(os.environ.get("TOTAL", 3.0e6))
# the background over the whole scan: 20 photons per pixel and frame at 10 nm steps over ±500 nm, 3e6 photons;
# with the bead's photons, a longer scan collects proportionally more
# one budget per bead whatever the range, as a scan of the same duration: photons and background fixed
BG_TOTAL = 20.0 * 101 * M * TOTAL_PHOTONS / 3.0e6
STEPS_RAD = [0.0, -2.09, -4.19]
optics = uipsf.Optics(NA=1.35, wavelength_nm=600, n_imm=1.406, n_med=1.406, pixel_size_nm=(129, 129))
grid = uipsf.PupilGrid.build(optics, 64, device=DEVICE)


def truth_values(model):
    up = model.pupils[0]
    ph_up, ph_lo = torch.zeros(len(up.phase_modes)), torch.zeros(len(up.phase_modes))
    for j, a, b in ((6, 0.5, -0.5), (8, 0.1, 0.0), (7, 0.0, -0.08), (11, 0.05, 0.1), (5, 0.05, 0.0)):
        ph_up[up.phase_modes.index(j)], ph_lo[up.phase_modes.index(j)] = a, b
    return {"pupil.phase": ph_up, "lo.pupil.phase": ph_lo, "psf.sigma": torch.tensor([50.0, 50.0]),
            "fourpi.offset": torch.tensor([10.0, -10.0, 40.0]), "fourpi.steps": torch.tensor(STEPS_RAD[1:]),
            "fourpi.delta": torch.tensor([1.6, 3.1, 4.7]), "fourpi.share": torch.tensor([0.26, 0.24, 0.25]),
            "fourpi.balance": torch.tensor([0.5, 0.47, 0.53, 0.5]),
            "fourpi.coherence": torch.tensor([0.8, 0.78, 0.82, 0.79])}


def model_of():
    return FourPiModel(grid, (R, R), n_phases=M, n_max=6, steps_rad=[2.0, 0.0, -2.0], bead_radius_nm=20.0,
                       intensity=PerObservation())


def scan(step, seed):
    J = int(math.floor(2 * HALF_RANGE / step)) + 1
    z = -(np.arange(J) - (J - 1) / 2) * step
    K = J * M
    rng = np.random.default_rng(seed)
    model = model_of()
    roi_xy = torch.tensor(rng.uniform(20, 180, (E, 1, 2)).repeat(C, 1), dtype=torch.float32)
    batch = uipsf.Batch(data=torch.zeros(E, C, K, R, R), z_nm=torch.tensor(np.repeat(z, M), dtype=torch.float32),
                        s=1.0, roi_xy=roi_xy).to(DEVICE)
    xyz = torch.tensor(np.c_[rng.uniform(-40, 40, (E, 2)), rng.uniform(-60, 60, E)], dtype=torch.float32)
    start = {"xyz": xyz, "photons": torch.full((E,), TOTAL_PHOTONS / K), "background": torch.full((E, C), BG_TOTAL / K)}
    p = Params(model.specs(batch, start), E, values=truth_values(model), device=DEVICE)
    values = p.physical()
    values["emitter.phase"] = torch.tensor(rng.uniform(-np.pi, np.pi, E), dtype=torch.float32, device=DEVICE)
    values["emitter.photons_per_slice"] = torch.full((E, K), TOTAL_PHOTONS / K, device=DEVICE)
    values["emitter.background"] = torch.full((E, C), BG_TOTAL / K, device=DEVICE)
    p = Params(list(p.specs.values()), E, values=values, device=DEVICE)
    sim = simulate(model, p, batch, seed=seed)
    shared = {k: v.detach().cpu() for k, v in p.physical().items() if p.specs[k].scope == "shared"}
    return sim, model, shared, J


def judge(cal, truth_cal, n=3000, seed=7, depth=400.0):
    """Molecules of 3000 photons, background 10, z within ±400 nm, drawn with the true model and fitted with
    ``cal``: the envelope's z, and z from the phase as 4Pi z makes it (the learnt slope, a constant offset
    taken out, put on the fringe the envelope points to)."""
    rois, t = fourpi.simulate_rois(truth_cal, n, roisize=13, photons=3000.0, background=10.0, z_nm=(-depth, depth), seed=seed)
    psf = Global4PiPSF(cal)
    v = psf.unpack(psf.fit(rois))
    period = abs(2 * np.pi / truth_cal.phase_per_nm)
    z_env = np.asarray(v["z_env_nm"])
    dz = z_env - t["z_nm"]
    k = cal.phase_per_nm
    raw = np.asarray(v["phase_rad"]) - t["phase0"]
    offset = np.angle(np.mean(np.exp(1j * (raw - k * t["z_nm"]))))
    z_ph = z_env + np.angle(np.exp(1j * (raw - offset - k * z_env))) / k
    dzp = z_ph - t["z_nm"]
    ok = np.abs(dzp) < period / 2
    return {"z_env_rms_nm": float(np.sqrt(np.mean(dz ** 2))), "z_env_bias_nm": float(np.median(dz)),
            "z_env_crlb_nm": float(np.median(np.asarray(v["z_err_nm"]))),
            "z_phase_rms_nm": float(np.sqrt(np.mean(dzp[ok] ** 2))),
            "z_phase_slope": float(np.polyfit(t["z_nm"][ok], z_ph[ok], 1)[0]),
            "wrong_fringe": float(1 - ok.mean()), "period_nm": float(2 * np.pi / abs(k))}


results, truth_cal = [], None
name = OUT + f"zstep_range{int(HALF_RANGE)}_total{int(TOTAL_PHOTONS)}_{'_'.join(map(str, SEEDS))}.json"
for seed in SEEDS:
    for step in STEPS:
        t0 = time.time()
        sim, tmodel, tshared, J = scan(step, seed)
        if truth_cal is None:
            truth_cal = calibration((tmodel, tshared), z_range_nm=600, dz_nm=20, roi_px=13)
            results.append({"step_nm": 0, "seed": -1, **judge(truth_cal, truth_cal),
                            **{f"{k}@600": v for k, v in judge(truth_cal, truth_cal, depth=600.0).items()}})
            print("truth", json.dumps({k: round(v, 3) for k, v in results[-1].items()}), flush=True)
        fit = learn_fourpi(model_of(), sim)
        cal = calibration(fit, z_range_nm=600, dz_nm=20, roi_px=13)
        sh = {k: v.detach().cpu() for k, v in fit.shared().items()}
        pupil = float(torch.sqrt(torch.mean((sh["pupil.phase"] - tshared["pupil.phase"]) ** 2)
                                 + torch.mean((sh["lo.pupil.phase"] - tshared["lo.pupil.phase"]) ** 2)))
        r = {"step_nm": step, "seed": seed, "slices": J, "frames": J * M, "photons_per_frame": TOTAL_PHOTONS / (J * M),
             "minutes": (time.time() - t0) / 60, "pupil_rms_rad": pupil, "half_range_nm": HALF_RANGE,
             **judge(cal, truth_cal), **{f"{k}@600": v for k, v in judge(cal, truth_cal, depth=600.0).items()}}
        results.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        json.dump(results, open(name, "w"), indent=1)
