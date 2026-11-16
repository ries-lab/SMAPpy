"""Can the bead fit tell the stage's direction at all?  The beads simulated from each learnt model, learnt both ways.

    python flip_sim.py [SEEDS]          # SEEDS: how many noise draws per truth (default 2)

`m2_new.py beads` learnt the 19 beads with stage z as read and turned round (``FLIP_Z``); as read won by 27 244 nats.
That is a preference only if the model, in its own basis, can produce a gap of that size when the direction is known.
So: each learnt model (as read, flipped) draws the beads again -- the same beads, slices, photons, background, drift
and bleaching it learnt, Poisson with the camera's excess noise and read noise, as `uipsf.noise.ShiftedPoisson`
models them -- and each draw is learnt in both layouts.  The flipped layout is the as-read one with the slices in
reverse (checked here against the batch ``FLIP_Z`` reads).  If the gap with the truth known is as large as the real one
and in the truth's favour from either side, the real gap is the direction; if the as-read layout wins whatever drew
the data, it is the model's bias.
"""
import dataclasses
import functools
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_new  # noqa: E402


def flip_k(batch):
    """The batch with its slices in reverse order, z kept: stage z read the other way."""
    K = batch.data.shape[2]

    def along(t):
        if isinstance(t, torch.Tensor) and t.dim() == batch.data.dim() and t.shape[2] == K:
            return t.flip(2)
        return t
    return dataclasses.replace(batch, data=batch.data.flip(2), mask=along(batch.mask), s=along(batch.s),
                               frame=None if batch.frame is None else batch.frame.flip(-1))


def draw(cal, rois, seed):
    """The learning batch of ``rois`` with data drawn from ``cal``'s model at its learnt parameters."""
    from uipsf.device import on_model_device, to_cpu64
    from uipsf.workflows.calibrate import learning_batch
    lb = learning_batch(on_model_device(rois.batch, cal.model), cal.options)
    with torch.no_grad():
        p = {k: v.to(lb.data.device) for k, v in cal.result.params.physical().items()}
        mu = to_cpu64(cal.model(p, lb))
    F = float(rois.noise.excess)
    s = to_cpu64(torch.as_tensor(lb.s)) if isinstance(lb.s, torch.Tensor) else float(lb.s)
    g = torch.Generator().manual_seed(int(seed))
    d = F * (torch.poisson((mu / F + s).clamp(min=0), generator=g) - s)
    return dataclasses.replace(lb, data=d.to(rois.batch.data.dtype).cpu())


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.6)
    import uipsf.workflows.calibrate as _c
    _c.learn_beads = functools.partial(_c.learn_beads, chunk_size=int(os.environ.get("CHUNK", 4)))
    import smappy.calibrate.input as cal_input
    from smappy.calibrate.pupil import (
        BeadSettings,
        OpticsSettings,
        PSFModelSettings,
        RunSettings,
        bead_rois,
        run_calibration,
    )
    files = sorted(glob.glob(os.path.join(m2_new.ROOT, m2_new.SETS["beads"])))
    optics = OpticsSettings(NA=1.43, wavelength_nm=m2_new.WAVELENGTH, n_immersion=1.516, n_medium=1.33,
                            pixel_x_nm=m2_new.PIXEL[0], pixel_y_nm=m2_new.PIXEL[1])
    model = BeadSettings(psf=PSFModelSettings(zernike_order=8, blur="round", blur_nm=20.0))
    model.rejection_k = 1000.0              # every bead kept, the same 19 in every learning
    model.bead_diameter_nm = 100.0
    model.drift = "linear"

    rois = {"as read": bead_rois(files)}
    _assemble = cal_input._assemble         # m2_new's FLIP_Z, for the reading alone

    def assemble(records, summary, source, dz_nm):
        records = [(i, a, dict(md, ZPositionUm=-float(md["ZPositionUm"])) if md.get("ZPositionUm") is not None
                    else md) for i, a, md in records]
        return _assemble(records, summary, source, dz_nm)
    cal_input._assemble = assemble
    rois["flipped"] = bead_rois(files)
    cal_input._assemble = _assemble
    a, f = rois["as read"].batch, rois["flipped"].batch
    # the reading's flip is the reverse one slice on (60 slices: the nominal focus is the slice past the middle,
    # on either side), a constant 50 nm that each bead's learnt z takes up; flip_k keeps z, the same up to that
    same = flip_k(a).data.roll(-1, 2)[:, :, :-1]
    print("flipped batch = as read in reverse, one slice on:", bool(torch.equal(same, f.data[:, :, :-1])), flush=True)

    def learn(r, batch=None):
        if batch is not None:
            r = dataclasses.replace(r, batch=batch)
        run = run_calibration(files, False, optics, model, run=RunSettings(save_model=False), rois=r, decisions={})
        return run.cal

    def summary(cal):
        p = cal.result.params.physical()
        ph = m2_new.nm_of(cal.model.psf.pupil, p["pupil.phase"].detach().cpu().reshape(-1)
                          [:len(cal.model.psf.pupil.phase_modes)])
        return (f"½ deviance {float(cal.result.objective):.1f} over {int(cal.result.params.n_emitters)}, "
                f"γ {float(p['pupil.gamma'].reshape(-1)[0]):.3f}, Z6 {ph['Z6(2, 2)']:+.1f}, "
                f"Z11 {ph['Z11(4, 0)']:+.1f}, Z22 {ph['Z22(6, 0)']:+.1f}")

    truth = {}
    for name in ("as read", "flipped"):
        truth[name] = learn(rois[name])
        print(f"real, learnt {name}: {summary(truth[name])}", flush=True)
    print(f"real: flipped − as read {float(truth['flipped'].result.objective - truth['as read'].result.objective):+.1f}"
          " nats", flush=True)
    for name in ("as read", "flipped"):
        for seed in range(seeds):
            sim = draw(truth[name], rois[name], seed)
            layouts = {name: sim, ("flipped" if name == "as read" else "as read"): flip_k(sim)}
            got = {}
            for learnt in ("as read", "flipped"):
                cal = learn(rois[learnt], layouts[learnt])
                got[learnt] = float(cal.result.objective)
                print(f"drawn from {name}, seed {seed}, learnt {learnt}: {summary(cal)}", flush=True)
            print(f"drawn from {name}, seed {seed}: flipped − as read {got['flipped'] - got['as read']:+.1f} nats",
                  flush=True)


if __name__ == "__main__":
    main()
