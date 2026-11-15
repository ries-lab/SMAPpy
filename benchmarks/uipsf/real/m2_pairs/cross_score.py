"""In situ against molecule stacks on M2 (12 November 2026): each model scores the other's held-out molecules.

    python cross_score.py OUTDIR STACK_MODEL INSITU_FOR_STACK MODEL [MODEL ...] [--json FILE]

Two held-out sets, neither seen by any model's learning:

* **stack**: the AB coverslip sweep's molecules that pass the selection (significance 5, the runs' setting)
  but were not drawn into the 4000 learnt from (`molecule_rois`'s draw is seeded, so the same split as the
  runs').  Every model scores them in the stack's geometry (`CoverslipStackGeometry`, STACK_MODEL's): its
  pupil phase, γ and σ held, the geometry -- each sweep's focus, the focal plane's slope, the molecules'
  height -- learnt again on 600 of the drawn molecules (the same 600 for every model), then the held-out
  molecules localised with x, y, photons and background free.  So a model is not charged for a focus offset
  that only belongs to the other geometry, only for its PSF's shape.
* **in situ**: the nucleus movie's molecules in the *held out* and *not drawn* states (`movie_rois`'s
  defaults, deterministic).  An in situ model scores them with its own focus; a stack model through
  INSITU_FOR_STACK, the in situ run that held every aberration of it and learnt only the focus (and σ).  Each
  molecule localised with x, y, z, photons and background free from five heights, the best kept.

Each MODEL is a saved `<name>_3dcal.h5` in OUTDIR; a stack model's name is STACK_MODEL itself.  Reported: ½ Σ
deviance per molecule on each set (lower is better; a difference is a log-likelihood ratio, nats per
molecule), relative to the first MODEL.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

ROOT = os.environ.get("M2_DATA", os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Vienna/Data/"
                                                     "smappy test/261112_new"))
STACK = "261112_MR_Intercalib2_AB_50ms_coverslip/*/*.ome.tif"
MOVIE = "261112_MR_Intercalib2_BGAAF647_50ms_30i_nucleus_1/*_MMStack_Pos0.ome.tif"
STARTS_NM = (-200.0, 200.0, 600.0, 1000.0, 1400.0)
GEOMETRY_MOLECULES = 600


def phase_by_noll(m) -> dict:
    """{Noll index: rad} of a saved model: its learnt modes and the ones it held (in situ's *hold*)."""
    pupil = m.model.channels[0].psf.pupil
    out = {}
    held = getattr(pupil, "held_phase", None)
    if isinstance(held, dict):
        out.update({int(j): float(v) for j, v in held.items()})
    values = torch.as_tensor(m.shared.get("pupil.phase", torch.zeros(0))).reshape(-1)
    out.update({int(j): float(v) for j, v in zip(pupil.phase_modes, values, strict=False)})
    return out


def gamma_of(m) -> float:
    if "pupil.gamma" in m.shared:
        return float(torch.as_tensor(m.shared["pupil.gamma"]))
    return float(m.model.channels[0].psf.pupil.held_gamma)


def half_deviance(model, noise, fit, problem) -> np.ndarray:
    from uipsf.solve import Problem
    dev = Problem(model, noise, problem.batch).deviance(fit.params)
    return 0.5 * dev.reshape(dev.shape[0], -1).sum(1).double().cpu().numpy()


def stack_scores(out, stack_name, names):
    from smappy.calibrate.molecule_stack import MoleculeStackSettings, molecule_rois
    from smappy.calibrate.pupil import load_start

    from uipsf.params import Params
    from uipsf.solve import LevenbergMarquardt, Problem
    from uipsf.workflows.scoring import localise

    rois = molecule_rois(sorted(glob.glob(os.path.join(ROOT, STACK))),
                         MoleculeStackSettings(molecules=4000, significance=5.0, height_nm=None))
    drawn, held = rois.ids("used"), rois.ids("not drawn")
    rng = np.random.default_rng(0)
    geo_ids = np.sort(rng.choice(drawn, min(GEOMETRY_MOLECULES, len(drawn)), replace=False))
    geo_batch, noise = rois.batch(ids=geo_ids)
    held_batch, _ = rois.batch(ids=held)
    print(f"stack: {len(drawn)} drawn, {len(held)} held out; geometry from {len(geo_ids)}", flush=True)
    template = load_start(os.path.join(out, f"{stack_name}_3dcal.h5"))
    model = template.model
    modes = list(model.channels[0].psf.pupil.phase_modes)
    result = {}
    for name in names:
        m = template if name == stack_name else load_start(os.path.join(out, f"{name}_3dcal.h5"))
        ph = phase_by_noll(m)
        shared = {k: torch.as_tensor(v).detach().clone() for k, v in template.shared.items()}
        shared["pupil.phase"] = torch.tensor([ph.get(j, 0.0) for j in modes], dtype=shared["pupil.phase"].dtype)
        shared["pupil.gamma"] = torch.tensor(gamma_of(m), dtype=torch.float64)
        shared["psf.sigma"] = torch.as_tensor(m.shared["psf.sigma"]).reshape(-1)[:1].clone()
        # the geometry learnt again with this pupil held: the emitters first, then everything but the PSF
        fit0, _ = localise(model, shared, geo_batch, noise)
        values = dict(shared)
        values.update({k: v.detach() for k, v in fit0.params.physical().items() if k.startswith("emitter.")})
        specs = model.specs(geo_batch)
        p = Params(specs, geo_batch.n_emitters, dtype=geo_batch.data.dtype, device=geo_batch.data.device,
                   values=values)
        p = p.fix(*[s.name for s in specs if s.scope == "shared" and not s.name.startswith("geometry.")])
        fit = LevenbergMarquardt(max_iter=60).solve(Problem(model, noise, geo_batch), p)
        learnt = {k: v.detach() for k, v in fit.params.physical().items() if k.startswith("geometry.")}
        shared.update(learnt)
        hfit, hproblem = localise(model, shared, held_batch, noise)
        e = half_deviance(model, noise, hfit, hproblem)
        result[name] = e
        print(f"  {name:44s} stack held-out: {e.mean():9.2f} per molecule; height "
              f"{float(learnt['geometry.height']):.0f} nm, plane {learnt['geometry.plane'].tolist()}", flush=True)
    return result, len(held)


def insitu_scores(out, stack_name, insitu_for_stack, names):
    from smappy.calibrate.insitu_run import movie_rois
    from smappy.calibrate.pupil import load_start

    from uipsf.workflows.scoring import localise

    rois = movie_rois(sorted(glob.glob(os.path.join(ROOT, MOVIE))))
    ids = np.sort(np.concatenate([rois.ids("held out"), rois.ids("not drawn")]))
    # the *held out* are drawn as the used are (every z bin's share); the *not drawn* are the bins' surplus, so
    # they over-represent the heights where molecules are plentiful: reported apart as well
    held_only = np.isin(ids, rois.ids("held out"))
    batch, noise = rois.batch(ids=ids)
    print(f"in situ: {len(ids)} held out ({len(rois.ids('held out'))} held out, "
          f"{len(rois.ids('not drawn'))} not drawn)", flush=True)
    result = {}
    for name in names:
        file = insitu_for_stack if name == stack_name else name
        m = load_start(os.path.join(out, f"{file}_3dcal.h5"))
        shared = {k: torch.as_tensor(v) for k, v in m.shared.items()}
        best = np.full(batch.n_emitters, np.inf)
        for z0 in STARTS_NM:
            xyz = torch.zeros(batch.n_emitters, 3, dtype=torch.float64)
            xyz[:, 2] = z0
            fit, problem = localise(m.model, shared, batch, noise, start={"emitter.xyz": xyz})
            best = np.minimum(best, half_deviance(m.model, noise, fit, problem))
        result[name] = best
        result[name + "@held out"] = best[held_only]
        result[name + "@not drawn"] = best[~held_only]
        print(f"  {name:44s} in situ held-out: {best.mean():9.2f} per molecule (scored as {file}); held out "
              f"{best[held_only].mean():.2f}, not drawn {best[~held_only].mean():.2f}", flush=True)
    return result, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("stack")
    ap.add_argument("insitu_for_stack")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--only", choices=("stack", "insitu"))
    a = ap.parse_args()
    rows = {n: {} for n in a.names}
    for kind, fn in (("stack", lambda: stack_scores(a.out, a.stack, a.names)),
                     ("insitu", lambda: insitu_scores(a.out, a.stack, a.insitu_for_stack, a.names))):
        if a.only and a.only != kind:
            continue
        res, n = fn()
        for key, e in res.items():
            name, _, part = key.partition("@")
            col = kind + ("_" + part.replace(" ", "_") if part else "")
            base = res[a.names[0] + ("@" + part if part else "")]
            rows[name][f"{col}_molecules"] = len(e)
            rows[name][f"{col}_per_molecule"] = float(e.mean())
            rows[name][f"{col}_vs_first_per_molecule"] = float((e - base).mean())
            rows[name][f"{col}_vs_first_median"] = float(np.median(e - base))
    print("\nnats per molecule against", a.names[0], "(negative: fits the held-out molecules better)")
    for name, r in rows.items():
        nan = float("nan")
        print(f"  {name:44s} stack {r.get('stack_vs_first_per_molecule', nan):+8.3f}   "
              f"in situ {r.get('insitu_vs_first_per_molecule', nan):+8.3f} "
              f"(held out {r.get('insitu_held_out_vs_first_per_molecule', nan):+.3f}, "
              f"not drawn {r.get('insitu_not_drawn_vs_first_per_molecule', nan):+.3f})")
    if a.json:
        with open(a.json, "a") as f:
            for name, r in rows.items():
                f.write(json.dumps(dict(model=name, **r)) + "\n")


if __name__ == "__main__":
    main()
