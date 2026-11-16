"""The AB sweep learnt twice on the same molecules: stage z as read, and turned round (z_direction.py's question).

    python flip_same.py

The molecules are found once and both learnings get them, with ``z_nm`` and ``defocus_nm`` negated for the second;
everything else is `m2_new.py ab` with NMED=1.345 (``ab_n1345``), one round.  Rejection is off: with it on (even
k = 1000, through the 2 px offset limit) the flipped model lost ~300 more molecules than the other, and its total
over fewer molecules came out *lower* -- a comparison of different sets.  ``FLIP_Z`` in `m2_new.py` cannot do
this: the draw bins the molecules by signed defocus, so a flipped stage z draws others.

The beads need no script of their own: ``REJECT_K=1000`` keeps all 19 found both ways (`m2_new.py beads` with and
without ``FLIP_Z``).
"""
import copy
import functools
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_new  # noqa: E402


def main():
    from smappy.calibrate.molecule_stack import MoleculeStackSettings, molecule_rois
    from smappy.calibrate.molecule_stack_run import run_molecule_stack
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, PSFModelSettings, RunSettings

    import uipsf.workflows.rejection as rejection
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.6)
    # every molecule kept: run_molecule_stack builds its own RejectionSettings, so the class is patched
    rejection.RejectionSettings = functools.partial(rejection.RejectionSettings, enabled=False)
    files = sorted(glob.glob(os.path.join(m2_new.ROOT, m2_new.SETS["ab"])))
    optics = OpticsSettings(NA=1.43, wavelength_nm=m2_new.WAVELENGTH, n_immersion=1.516, n_medium=1.345,
                            pixel_x_nm=m2_new.PIXEL[0], pixel_y_nm=m2_new.PIXEL[1])
    model = BeadSettings(psf=PSFModelSettings(zernike_order=8, blur="round", blur_nm=20.0))
    stack = MoleculeStackSettings(molecules=4000, significance=5.0, height_nm=None, max_rounds=1)
    rois = molecule_rois(files, stack, None, lambda *a: None)
    print("molecules", len(rois.ids("used")), flush=True)
    out = {}
    for name in ("as read", "flipped"):
        r = copy.copy(rois)
        if name == "flipped":
            r.z_nm, r.defocus_nm = -rois.z_nm, -rois.defocus_nm
        res = run_molecule_stack(files, optics, model, stack, run=RunSettings(save_model=False), rois=r)
        p = res.cal.result.params.physical()
        ph = m2_new.nm_of(res.cal.model.psf.pupil, p["pupil.phase"].detach().cpu().reshape(-1))
        out[name] = float(res.cal.result.objective)
        print(f"{name}: ½ deviance {out[name]:.1f} over {int(res.cal.result.params.n_emitters)} molecules, "
              f"height {res.height_nm():.1f} nm, " + ", ".join(f"{k} {ph[k]:+.2f}" for k in
                                                               ("Z5(2, -2)", "Z6(2, 2)", "Z7(3, -1)", "Z8(3, 1)",
                                                                "Z11(4, 0)", "Z22(6, 0)", "Z37(8, 0)")), flush=True)
    print(f"flipped − as read: {out['flipped'] - out['as read']:+.1f} nats")


if __name__ == "__main__":
    main()
