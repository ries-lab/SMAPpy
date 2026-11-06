"""The first confocal bead: a donut scan and a filled-beam z-stack of one 120 nm bead fitted with `ConfocalPSF`,
alone or together through one objective -- C2's first look at real data (docs/uipsf/scanned-psfs.md §9).

    python benchmarks/uipsf/real/confocal_bead.py FR_vibrations_silence.msr \\
        [--n-med 1.38] [--pinhole 1.0] [--pupil minimal] [--fill flat] [--fix-centre] [--donut-only] \\
        [--starts 3] [--out benchmarks/uipsf/results/confocal_bead.jsonl] [--figures DIR]

The file (Abberior, 2026-11-06, not in git) holds three confocal images of the same bead, 30 nm pixels: an
overview with the vortex donut (``Ch1 {0}``), one plane of the filled beam (``Ch1 {207}``) and a 7-plane stack
of it, 200 nm apart (``Ch1 {208}``).  The donut and the stack are fitted; the bead's place in each is fixed here
(a crop of 51 and 61 pixels around it), so the script is this file's, not a reader.

The optics are the user's (2026-11-06): an oil objective, NA 1.4, n_imm 1.518; the sample's medium 1.38;
640 nm excitation, ~680 nm emission; the pinhole 0.87 or 1 AU; circular light, right-handed (+1: a donut fit
of either handedness preferred +1 by 340 nats).

**The fit.**  One pupil phase, Noll 5-11 (astigmatism, coma, trefoil, spherical), shared by both scans -- one
objective, the excitation beam with the vortex in its path, the filled one without -- and each scan its own
bead position, photons and background.  ``--pupil minimal`` is `MinimalPupil` with γ = 0 (a flat magnitude),
the ring held at h = 0 and the aperture's NA learnt; ``--fill`` the input beam's Gaussian 1/e radius in
aperture radii, ``flat`` a top-hat, ``free`` learnt (it ran off to a top-hat).  The vortex's centre is learnt
unless ``--fix-centre``: free, it moved outside the aperture in two of twelve fits.  Poisson likelihood by
L-BFGS through autograd, not the solver: two `ForwardModel` s share parameters here, which `Problem` does not
take yet.  Each start but the first draws the phases at 0.3 rad rms and the donut's z at 150 nm.

Prints and appends one JSON line per start, the half deviance of each scan against the saturated model, the
parameters and the run's settings; ``--figures`` draws the best start's data against the model and its pupil.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

MODES = [5, 6, 7, 8, 9, 10, 11]
NAMES = {5: "astig 45", 6: "astig 0", 7: "coma y", 8: "coma x", 9: "trefoil y", 10: "trefoil x", 11: "spherical"}
STEP_NM = 30.0
STACK = ("Ch1 {208}", (66, 62), 30)          # name, the bead's (row, column), half-width of the crop
DONUT = ("Ch1 {0}", (131, 206), 25)
STAGE_NM = np.arange(7) * 200.0 - 600.0      # the stack's objective positions (the file's 1.4 µm over 7 planes)
POS = 100.0                                  # positions in units of 100 nm for the optimiser


def read(path):
    """The stack [7, 61, 61] and the donut [51, 51], in counts."""
    from smappy.io import obf
    stacks = {s.name: s for s in obf.stacks(path)}

    def crop(name, centre, h):
        a = np.squeeze(obf.read_stack(path, stacks[name])).astype(float)
        r, c = centre
        return a[..., r - h:r + h + 1, c - h:c + h + 1]
    return crop(*STACK), crop(*DONUT)


def model(args, batch, donut: bool):
    """(psf, ForwardModel, the parameters' physical defaults) of one scan."""
    from uipsf.model import (
        BeamSpec,
        Bleaching,
        ConfocalPSF,
        ExcitationBeam,
        ForwardModel,
        MaskSpec,
        MinimalPupil,
        ZernikePupil,
        make_mask,
        minimal_grid_optics,
    )
    from uipsf.optics import Optics
    from uipsf.optics.grid import PupilGrid
    from uipsf.params import Params
    o = Optics(NA=args.na, wavelength_nm=args.excitation, n_imm=args.n_imm, n_med=args.n_med,
               pixel_size_nm=(STEP_NM, STEP_NM))
    grid = PupilGrid.build(minimal_grid_optics(o) if args.pupil == "minimal" else o, 64, device="cpu",
                           dtype=torch.float64)
    mask = make_mask(MaskSpec(kind="vortex", charge=1, free=("centre",)), grid, aperture_na=o.NA) if donut else None
    if args.pupil == "minimal":
        pupil = MinimalPupil(grid, phase_modes=MODES, na=args.na, held_gamma=0.0, held_ring_h=0.0, learn_na=True,
                             mask=mask)
    else:
        pupil = ZernikePupil(grid, phase_modes=MODES, magnitude_n_max=0, mask=mask)
    fill = None if args.fill == "flat" else 2.15 if args.fill == "free" else float(args.fill)
    beam = ExcitationBeam(grid, pupil, BeamSpec(polarisation="circular", handedness=args.handedness, fill=fill))
    psf = ConfocalPSF(beam, tuple(batch.data.shape[-2:]), emission_nm=args.emission, pinhole_au=args.pinhole,
                      sample="bead", bead_radius_nm=args.bead_radius)
    fm = ForwardModel(psf, intensity=Bleaching(order=0))
    start = {"xyz": torch.zeros(1, 3), "photons": torch.ones(1), "background": torch.ones(1)}
    base = Params(fm.specs(batch, start), n_emitters=1, dtype=torch.float64, device="cpu").physical()
    return psf, fm, {k: v.detach() for k, v in base.items()}


def half_deviance_offset(counts) -> float:
    """Σ (n − n log n): the saturated model's Poisson negative log-likelihood, so a fit's reads as ½ deviance."""
    n = np.asarray(counts, dtype=float)
    return float(np.where(n > 0, n - n * np.log(np.where(n > 0, n, 1.0)), 0.0).sum())


def fit(args, stack, donut, seed: int):
    """One start: the result's JSON line and what the figures need."""
    from uipsf import Batch
    from uipsf.model.pupil import GRID_MARGIN
    b_s = Batch(data=torch.tensor(stack)[None, None], z_nm=Batch.z_from_objective(torch.tensor(STAGE_NM)))
    b_d = Batch(data=torch.tensor(donut)[None, None, None], z_nm=torch.zeros(1))
    ps, fs, base_s = model(args, b_s, False)
    pd, fd, base_d = model(args, b_d, True)
    na_lo, na_hi = 0.8 * args.na, args.na * GRID_MARGIN
    g = torch.Generator().manual_seed(seed)
    phase0 = np.zeros(len(MODES)) if seed == 0 else np.random.default_rng(seed).normal(0, 0.3, len(MODES))
    v = {"phase": torch.tensor(phase0),
         "xyz_s": torch.zeros(3), "lnN_s": torch.tensor(np.log(stack.sum() / 0.1)), "lnb_s": torch.tensor(np.log(0.4)),
         "xyz_d": torch.randn(3, generator=g) * torch.tensor([20.0, 20.0, 150.0]) / POS,
         "lnN_d": torch.tensor(np.log(donut.sum() / 0.05)), "lnb_d": torch.tensor(np.log(0.5)),
         "centre": torch.zeros(2)}
    if args.pupil == "minimal":        # NA = lo + (hi − lo)·sigmoid(a), started at the objective's
        v["na"] = torch.tensor(float(np.log((args.na - na_lo) / (na_hi - args.na))))
    if args.fill == "free":
        v["lnfill"] = torch.tensor(np.log(2.15))
    for name, t in v.items():
        t.requires_grad_(not (name == "centre" and args.fix_centre))

    def shared():
        out = {"pupil.phase": v["phase"]}
        if "na" in v:
            out["pupil.na"] = na_lo + (na_hi - na_lo) * torch.sigmoid(v["na"])
        if "lnfill" in v:
            out["beam.fill"] = v["lnfill"].exp()
        return out

    def dicts():
        s = dict(base_s, **shared(), **{"emitter.xyz": (v["xyz_s"] * POS)[None],
                                        "emitter.photons": v["lnN_s"].exp()[None],
                                        "emitter.background": v["lnb_s"].exp()[None]})
        d = dict(base_d, **shared(), **{"emitter.xyz": (v["xyz_d"] * POS)[None],
                                        "emitter.photons": v["lnN_d"].exp()[None],
                                        "emitter.background": v["lnb_d"].exp()[None], "mask.centre": v["centre"]})
        return s, d

    def nll():
        s, d = dicts()
        terms = []
        for fm, p, b in ((fs, s, b_s), (fd, d, b_d)):
            mu = fm(p, b).clamp(min=1e-9)
            terms.append((mu - b.data * mu.log()).sum())
        return terms

    opt = torch.optim.LBFGS([t for t in v.values() if t.requires_grad], lr=1.0, max_iter=400, history_size=20,
                            tolerance_grad=1e-6, tolerance_change=1e-10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        L_s, L_d = nll()
        L = L_d if args.donut_only else L_s + L_d
        L.backward()
        return L

    for _ in range(3):
        opt.step(closure)
    with torch.no_grad():
        s, d = dicts()
        L_s, L_d = (float(t) for t in nll())
        mu_s, mu_d = fs(s, b_s)[0, 0].numpy(), fd(d, b_d)[0, 0, 0].numpy()
        extra = {k.split(".")[1]: round(float(x), 4) for k, x in shared().items() if k != "pupil.phase"}
    dev_s, dev_d = L_s - half_deviance_offset(stack), L_d - half_deviance_offset(donut)
    row = dict(n_med=args.n_med, pinhole_au=args.pinhole, pupil=args.pupil, fill_mode=args.fill,
               fix_centre=args.fix_centre, donut_only=args.donut_only, seed=seed,
               half_deviance=round(dev_d if args.donut_only else dev_s + dev_d, 1),
               donut_half_deviance=round(dev_d, 1), stack_half_deviance=round(dev_s, 1),
               phase={NAMES[j]: round(float(c), 3) for j, c in zip(MODES, v["phase"].detach(), strict=True)},
               stack_xyz=[round(float(x) * POS, 1) for x in v["xyz_s"].detach()],
               donut_xyz=[round(float(x) * POS, 1) for x in v["xyz_d"].detach()],
               vortex_centre=[round(float(x), 3) for x in v["centre"].detach()], **extra)
    return row, (ps, s, pd, d, mu_s, mu_d)


def figures(outdir, row, stack, donut, ps, s, pd, d, mu_s, mu_d, na):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    h_s, h_d = stack.shape[-1] // 2, donut.shape[-1] // 2
    fig, axes = plt.subplots(3, 8, figsize=(24, 9))
    for k in range(len(STAGE_NM)):
        vmax = stack[k].max()
        axes[0, k].imshow(stack[k], cmap="magma", vmax=vmax)
        axes[0, k].set_title(f"stack, objective {STAGE_NM[k]:+.0f} nm")
        axes[1, k].imshow(mu_s[k], cmap="magma", vmax=vmax)
        axes[1, k].set_title("model")
        for prof_d, prof_m in ((stack[k, h_s], mu_s[k, h_s]), (stack[k, :, h_s], mu_s[k, :, h_s])):
            axes[2, k].plot(prof_d, ".", ms=3)
            axes[2, k].plot(prof_m, "-")
    axes[0, 7].imshow(donut, cmap="magma")
    axes[0, 7].set_title("donut")
    axes[1, 7].imshow(mu_d, cmap="magma", vmax=donut.max())
    axes[1, 7].set_title("model")
    for prof_d, prof_m in ((donut[h_d], mu_d[h_d]), (donut[:, h_d], mu_d[:, h_d])):
        axes[2, 7].plot(prof_d, ".", ms=3)
        axes[2, 7].plot(prof_m, "-")
    fig.suptitle(f"n_med {row['n_med']}, {row['pinhole_au']} AU, {row['pupil']}, fill {row['fill_mode']}: "
                 f"½ deviance {row['half_deviance']} over {stack.size + donut.size} pixels")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confocal_bead_fit.png"), dpi=65)

    with torch.no_grad():
        _, phase = ps.pupil(s)
        vortex = torch.angle(pd.beam.objective(d))
        amp = (ps.beam.objective(s) * ps.beam.amplitude(s)).abs().numpy()
    rho = ps.grid.rho.real if ps.grid.rho.is_complex() else ps.grid.rho
    inside = rho.numpy() * ps.grid.optics.NA <= na
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    po = np.where(inside, phase.numpy(), np.nan)
    lim = np.nanmax(np.abs(po))
    im = axes[0].imshow(po, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0].set_title("the objective's aberration (rad at the excitation)")
    fig.colorbar(im, ax=axes[0])
    im = axes[1].imshow(np.where(inside, vortex.numpy(), np.nan), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f"the donut's pupil, wrapped (vortex centre {row['vortex_centre']})")
    fig.colorbar(im, ax=axes[1])
    axes[2].barh(list(row["phase"]), list(row["phase"].values()))
    axes[2].axvline(0, color="k", lw=0.5)
    axes[2].set_xlabel("rad")
    axes[2].set_title("Zernike coefficients (Noll 5-11)")
    edge = ps.grid.optics.NA
    line = np.linspace(-edge, edge, amp.shape[-1])
    axes[3].plot(line, amp[amp.shape[0] // 2] / amp.max(), label="along x")
    axes[3].plot(line, amp[:, amp.shape[1] // 2] / amp.max(), label="along y")
    for x in (-na, na):
        axes[3].axvline(x, color="k", lw=0.5, ls=":")
    axes[3].set_xlabel("n sin θ (dotted: the objective's NA)")
    axes[3].set_title(f"the filled beam's amplitude, NA {row.get('na', na)}")
    axes[3].legend(fontsize=7)
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confocal_bead_pupil.png"), dpi=80)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("msr")
    ap.add_argument("--n-med", type=float, default=1.38)
    ap.add_argument("--n-imm", type=float, default=1.518)
    ap.add_argument("--na", type=float, default=1.4)
    ap.add_argument("--excitation", type=float, default=640.0)
    ap.add_argument("--emission", type=float, default=680.0)
    ap.add_argument("--bead-radius", type=float, default=60.0)
    ap.add_argument("--pinhole", type=float, default=1.0, help="diameter, Airy units")
    ap.add_argument("--handedness", type=int, default=1)
    ap.add_argument("--pupil", choices=("zernike", "minimal"), default="minimal")
    ap.add_argument("--fill", default="flat", help="the beam's 1/e radius in aperture radii, 'flat' or 'free'")
    ap.add_argument("--fix-centre", action="store_true", help="hold the vortex on the axis")
    ap.add_argument("--donut-only", action="store_true")
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figures", default=None, help="a folder for the best start's figures")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    stack, donut = read(args.msr)
    best = None
    for seed in range(args.starts):
        t0 = time.time()
        row, extra = fit(args, stack, donut, seed)
        row["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(row), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        if best is None or row["half_deviance"] < best[0]["half_deviance"]:
            best = (row, extra)
    if args.figures:
        row, (ps, s, pd, d, mu_s, mu_d) = best
        figures(args.figures, row, stack, donut, ps, s, pd, d, mu_s, mu_d, args.na)


if __name__ == "__main__":
    main()
