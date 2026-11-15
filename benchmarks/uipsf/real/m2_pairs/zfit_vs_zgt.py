"""Spots drawn from AB's model propagated to known heights, localised with each in situ model: z_fit vs z_gt.

    python zfit_vs_zgt.py OUTDIR MODEL [MODEL ...]

Truth: `insitu_all_n1345_tilt_s315_ab_n1345` (AB's every aberration, γ, σ 31.5 nm, n_med 1.345, focus 607 nm)
with its tilt set to 0 -- a molecule at height h in the sample, drawn by the in situ model's own physics.  At
each height 0..1200 nm, `REP` spots at random sub-pixel positions, 4570 photons, background 98 / px (the real
movie's medians), Poisson noise.  Each model localises them (x, y, z, photons, background free) from `STARTS_NM`
and keeps the best likelihood per spot.  Prints per height the median z_fit - z_gt and its spread, the median x
and y error, and writes a figure `zfit_vs_zgt.png`.
"""
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HEIGHTS = np.arange(0.0, 1201.0, 100.0)
REP, PHOTONS, BG = 40, 4570.0, 98.0
STARTS_NM = (0.0, 300.0, 600.0, 900.0, 1200.0)


def main():
    out, names = sys.argv[1], sys.argv[2:]
    from smappy.calibrate.pupil import load_start

    from uipsf.noise import ShiftedPoisson
    from uipsf.simulate import parameters, simulate, template
    from uipsf.solve import Problem
    from uipsf.workflows.scoring import localise

    t = load_start(os.path.join(out, "insitu_all_n1345_tilt_s315_ab_n1345_3dcal.h5"))
    tm = t.model
    shared = {k: torch.as_tensor(v).detach().clone() for k, v in t.shared.items()}
    shared["geometry.tilt"] = torch.zeros_like(shared["geometry.tilt"])
    rng = np.random.default_rng(0)
    E = len(HEIGHTS) * REP
    z_gt = np.repeat(HEIGHTS, REP)
    px, py = tm.pixel_size_nm
    xy = rng.uniform(-0.5, 0.5, (E, 2)) * np.array([px, py])
    roi = tuple(int(s) for s in tm.channels[0].psf.roi) if hasattr(tm.channels[0].psf, "roi") else (17, 17)
    b0 = template(E, torch.zeros(1, dtype=torch.float64), roi, dtype=torch.float32)
    values = dict(shared)
    values.update({"emitter.xyz": torch.tensor(np.c_[xy, z_gt], dtype=torch.float64),
                   "emitter.photons": torch.full((E,), PHOTONS), "emitter.background": torch.full((E,), BG)})
    truth = parameters(tm, b0, values)
    batch = simulate(tm, truth, b0, seed=1)
    noise = ShiftedPoisson()
    results = {}
    for name in names:
        m = load_start(os.path.join(out, f"{name}_3dcal.h5"))
        sh = {k: torch.as_tensor(v) for k, v in m.shared.items()}
        best = np.full(E, np.inf)
        found = np.full((E, 3), np.nan)
        for z0 in STARTS_NM:
            start = torch.tensor(np.c_[xy * 0, np.full(E, z0)], dtype=torch.float64)
            fit, problem = localise(m.model, sh, batch, noise, start={"emitter.xyz": start})
            dev = Problem(m.model, noise, problem.batch).deviance(fit.params)
            e = 0.5 * dev.reshape(E, -1).sum(1).double().numpy()
            f = fit.params.physical()["emitter.xyz"].detach().cpu().double().numpy()
            better = e < best
            best[better], found[better] = e[better], f[better]
        results[name] = (found, best)
        dz = found[:, 2] - z_gt
        dx, dy = found[:, 0] - xy[:, 0], found[:, 1] - xy[:, 1]
        print(f"\n{name}  (½Σdeviance {best.sum():.0f}, deviance/pixel {2 * best.mean() / np.prod(roi):.3f})")
        print("  z_gt   z_fit-z_gt median  robust sd   x err  y err (nm, medians)")
        for h in HEIGHTS:
            k = z_gt == h
            mad = 1.4826 * np.median(np.abs(dz[k] - np.median(dz[k])))
            ex, ey = np.median(dx[k]), np.median(dy[k])
            print(f"  {h:5.0f}   {np.median(dz[k]):8.1f}       {mad:6.1f}    {ex:6.1f} {ey:6.1f}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, (found, _) in results.items():
        med = [np.median(found[z_gt == h, 2] - h) for h in HEIGHTS]
        axes[0].plot(HEIGHTS, [np.median(found[z_gt == h, 2]) for h in HEIGHTS], "o-", label=name[:40])
        axes[1].plot(HEIGHTS, med, "o-", label=name[:40])
        axes[2].plot(HEIGHTS, [np.median(found[z_gt == h, 0] - xy[z_gt == h, 0]) for h in HEIGHTS], "o-",
                     label=name[:40] + " x")
        axes[2].plot(HEIGHTS, [np.median(found[z_gt == h, 1] - xy[z_gt == h, 1]) for h in HEIGHTS], "s--",
                     label=name[:40] + " y")
    axes[0].plot(HEIGHTS, HEIGHTS, "k:", lw=1)
    axes[0].set(xlabel="z ground truth (nm)", ylabel="z fit (nm)", title="z fit vs z gt (medians)")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set(xlabel="z ground truth (nm)", ylabel="z fit - z gt (nm)", title="z error")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set(xlabel="z ground truth (nm)", ylabel="xy error (nm)", title="x, y error")
    for a in axes:
        a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "zfit_vs_zgt.png"), dpi=110)


if __name__ == "__main__":
    main()
