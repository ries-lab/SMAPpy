"""Why does the twin test prefer the mirrored pupil on M2's nucleus movie?  The test rerun, and where it ends.

    python twin_check.py OUTDIR MODEL [MODEL ...] [--molecules N] [--json FILE]

Every M2 in situ run's twin test preferred the mirrored pupil by 800-1500 nats, and the stage's direction is
not the cause (architecture.md, "The stage's direction": the stacks and the movie agree with the code's z).
The test (`uipsf.workflows.insitu.twin_fit`) starts from the twin of the learnt pupil, the focus kept, and then
learns *everything* again -- the pupil up to its order, σ, the focus and every molecule's height, the heights
with no bound at the coverslip -- and compares its optimum with the learnt run's.  So "prefer twin" can mean
three different things, and this script tells them apart:

* **the relearning simply goes deeper**: the learnt run stopped in a shallower place, and any restart would
  win.  Then refitting from the *learnt* pupil, the same way (`A`), wins as much as the twin (`B`) does;
* **the twin cheats on depth**: the twin's mirrored heights put molecules below the glass or far from the
  learnt focus, and a free geometry absorbs a depth model that is wrong (the nucleus' index; the score per
  height found the in situ pupils winning only deep).  Then `B` has many heights below 0, an edge far from 0,
  and loses its lead with the focus held at the learnt one (`C` against `D`);
* **the twin is better**: it keeps a physical geometry (an edge at 0, heights where the learnt ones are) and
  still wins, on the molecules it learnt from and on held-out ones.

Per MODEL (a saved in situ ``<name>_3dcal.h5`` in OUTDIR), on the movie's *used* molecules (`movie_rois`'s
defaults; ``--molecules`` takes that many of them, seeded, for a quicker run), four fits, each from the
molecules localised on its start pupil from `start_heights`, then everything the model learns free:

* ``A``: from the learnt pupil;      ``B``: from its twin (`twin`, `twin_model`), as `twin_fit` does;
* ``D``: A with the focus held;      ``C``: B with the focus held at the learnt one.

Reported per fit: the optimum (½ deviance + priors) and B − A, C − D; the low orders in nm (Z5-Z22) with the
astigmatism's sign; the focus; the heights' 5/50/95 % and the share below 0 and below −100 nm; the heights'
lower edge (`coverslip_edge`) and its spike.  Then each fit's pupil scores the *held out* and *not drawn*
molecules (`heights.localise_all`, five starts), saved as ``heights_<MODEL>__<A|B|C|D>.npz``, so that

    python score_by_height.py OUTDIR <MODEL>__A <MODEL>__A <MODEL>__B

gives the twin's score per height bin against the matched refit.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heights import MOVIE, ROOT, coverslip_edge, localise_all, path, unit_table  # noqa: E402

MODES = (5, 6, 7, 8, 11, 12, 16, 22)


def refit(model, shared: dict, batch, noise, hold_focus: bool = False):
    """Every molecule localised on ``shared`` from `start_heights`, then everything the model learns free (the
    focus held with ``hold_focus``): `twin_fit`'s procedure, from any start.  The `FitResult`."""
    from dataclasses import replace

    from uipsf.params import Params
    from uipsf.solve import GaussNewtonLBFGS, Problem
    from uipsf.workflows.insitu import localise_from_heights, start_heights

    focus = float(torch.as_tensor(shared["geometry.focus"]))
    found, _ = localise_from_heights(model, shared, batch, noise, start_heights(model, focus))
    sub = replace(batch, start=found)
    p = Params(model.specs(sub), sub.n_emitters, dtype=sub.data.dtype, device=sub.data.device, values=shared)
    if hold_focus:
        p = p.fix("geometry.focus")
    return GaussNewtonLBFGS().solve(Problem(model, noise, sub), p)


def summary(model, fit, wavelength_nm: float) -> dict:
    """What a fit ended at: its optimum, low orders in nm, focus, σ and the heights' distribution and edge."""
    phys = fit.params.physical()
    pupil = model.channels[0].psf.pupil
    phase = {int(j): float(v) for j, v in zip(pupil.phase_modes, phys["pupil.phase"].detach().cpu().reshape(-1),
                                                 strict=False)}
    phase.update({int(j): float(v) for j, v in getattr(pupil, "held_phase", {}).items()})
    h = phys["emitter.xyz"][:, 2].detach().cpu().double().numpy()
    sd = fit.std["emitter.xyz"][:, 2].detach().cpu().double().numpy() if "emitter.xyz" in fit.std else \
        np.full(len(h), 20.0)
    edge = coverslip_edge(h, sd)
    return dict(objective=float(fit.objective),
                phase_nm={f"Z{j}": round(phase.get(j, 0.0) * wavelength_nm / (2 * math.pi), 1) for j in MODES},
                focus_nm=float(phys["geometry.focus"]),
                sigma_nm=np.round(phys["psf.sigma"].detach().cpu().double().numpy().reshape(-1), 1).tolist(),
                heights_p5_50_95=np.percentile(h, [5, 50, 95]).round(0).tolist(),
                below_0=float(np.mean(h < 0)), below_minus100=float(np.mean(h < -100)),
                edge_nm=None if edge is None else round(edge.shift_nm, 1),
                edge_sd_nm=None if edge is None else round(edge.sd_nm, 1),
                edge_spike=None if edge is None else round(edge.spike, 2))


def shared_of(fit) -> dict:
    return {k: v.detach() for k, v in fit.params.physical().items() if not k.startswith("emitter.")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--molecules", type=int, help="this many of the used molecules (seeded), all by default")
    ap.add_argument("--wavelength", type=float, default=float(os.environ.get("WL", 690.0)))
    ap.add_argument("--json")
    a = ap.parse_args()
    from smappy.calibrate.insitu_run import movie_rois
    from smappy.calibrate.pupil import load_start

    from uipsf.workflows.insitu import twin, twin_model

    rois = movie_rois(sorted(glob.glob(os.path.join(ROOT, MOVIE))))
    used = rois.ids("used")
    if a.molecules and a.molecules < len(used):
        used = np.sort(np.random.default_rng(0).choice(used, a.molecules, replace=False))
    unseen = np.sort(np.concatenate([rois.ids("held out"), rois.ids("not drawn")]))
    print(f"learning from {len(used)} used molecules; scoring {len(unseen)} held out and not drawn", flush=True)
    for name in a.names:
        m = load_start(os.path.join(a.out, f"{name}_3dcal.h5"))
        model = m.model
        dtype = model.channels[0].psf.grid.rho.dtype
        batch, noise = rois.batch(ids=used)
        batch = batch.to(dtype=dtype)
        learnt = {k: torch.as_tensor(v) for k, v in m.shared.items()}
        mirrored_model, mirrored = twin_model(model), twin(model, learnt)
        fits = {}
        for label, (mdl, start, hold) in {"A": (model, learnt, False), "B": (mirrored_model, mirrored, False),
                                          "D": (model, learnt, True), "C": (mirrored_model, mirrored, True)}.items():
            print(f"{name}: fit {label} ...", flush=True)
            fit = refit(mdl, start, batch, noise, hold_focus=hold)
            fits[label] = (mdl, fit, summary(mdl, fit, a.wavelength))
        rows = {k: v[2] for k, v in fits.items()}
        print(f"\n{name}  (learnt focus {float(learnt['geometry.focus']):.0f} nm)")
        print(f"  B - A (twin, all free):    {rows['B']['objective'] - rows['A']['objective']:+10.1f} nats "
              f"(negative: the twin wins)")
        print(f"  C - D (twin, focus held):  {rows['C']['objective'] - rows['D']['objective']:+10.1f} nats")
        for label in "ABDC":
            r = rows[label]
            print(f"  {label}: objective {r['objective']:.1f}  focus {r['focus_nm']:.0f}  σ {r['sigma_nm']}  "
                  f"heights {r['heights_p5_50_95']}  below 0 {r['below_0']:.1%}, below -100 {r['below_minus100']:.1%}"
                  f"  edge {r['edge_nm']} ± {r['edge_sd_nm']} (spike {r['edge_spike']})")
            print(f"     {r['phase_nm']}")
        # each fit's pupil on the molecules none learnt from, saved for score_by_height.py
        unseen_batch, unseen_noise = rois.batch(ids=unseen)
        unseen_batch = unseen_batch.to(dtype=dtype)
        table = unit_table(rois, unseen)
        for label, (mdl, fit, r) in fits.items():
            shared = shared_of(fit)
            found = localise_all(mdl, shared, unseen_batch, unseen_noise)
            tag = f"{name}__{label}"
            np.savez(path(a.out, tag), **table, **found, focus_nm=r["focus_nm"])
            r["unseen_half_deviance_mean"] = float(np.mean(found["half_deviance"]))
            print(f"  {label}: unseen molecules ½Σdeviance {r['unseen_half_deviance_mean']:.3f} per molecule -> "
                  f"{path(a.out, tag)}", flush=True)
        print(f"  unseen, B - A: {rows['B']['unseen_half_deviance_mean'] - rows['A']['unseen_half_deviance_mean']:+.3f}"
              f", C - D: {rows['C']['unseen_half_deviance_mean'] - rows['D']['unseen_half_deviance_mean']:+.3f} "
              "nats per molecule")
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(dict(kind="twin_check", model=name, molecules=int(len(used)),
                                        learnt_focus_nm=float(learnt["geometry.focus"]), fits=rows)) + "\n")


if __name__ == "__main__":
    main()
