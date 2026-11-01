"""The learnt PSFs of `m5_media.py` in real space, beside the bead average, for each magnitude model.

    python benchmarks/uipsf/real/m5_media_psfs.py ACQUISITION [...] --out DIR \\
        [--variants zernike symmetric "cos bead na asym" "cos signed bead na asym"] [--dz 100] [--chunk 1]
    python benchmarks/uipsf/real/m5_media_psfs.py --draw DIR

The likelihoods say which magnitude fits better; this shows whether the PSFs look different at all.  Each
model is learnt as `m5_media.py` learns it (the same beads, windows and pixel), and the calibration window's
own views (`smappy.calibrate.model_views.pupil_views`) align the beads and the model onto common planes:
data and model in photons per photon, background removed, so they share one colour scale.  One .npz per
acquisition and model in DIR; ``--draw`` makes a figure per acquisition of five planes, the bead average on
top, each model below it and its residual against the data.
"""
from __future__ import annotations

import argparse
import functools
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "synthetic")]
from m5_media import GRID_NA, N_IMM, N_MED, ORDER, WAVELENGTH, acquisition, select, subset  # noqa: E402

PLANES_NM = (-600.0, -300.0, 0.0, 300.0, 600.0)
#: the residual's colour range, a fraction of the bead average's peak
RESIDUAL = 0.05
LABEL = {"zernike": "free Zernikes", "symmetric": "Zernikes, m = 0 only", "cos bead na asym": "cos θ^γ + m ≠ 0",
         "cos bead na": "cos θ^γ, symmetric", "cos signed bead na asym": "cos θ^γ × two-way + m ≠ 0",
         "cos steps bead na asym": "cos θ^γ × monotone + m ≠ 0",
         "minimal": "minimal: cos θ^γ, NA 1.50, ring height, 6 m ≠ 0"}
#: a variant's blur, after a "+" ("zernike+zpixel"): (σ(z), `BeadSettings.blur`); none: σx and σy learnt
BLURS = {"sigma": (False, "round"), "pixel": (False, "fixed"), "zpixel": (True, "fixed"), "z": (True, "round")}
BLUR_LABEL = {"sigma": "σ", "pixel": "σ = pixel", "zpixel": "σ(z), σ₀ = pixel", "z": "σ(z)"}


def variant_label(variant: str) -> str:
    pupil, _, blur = variant.partition("+")
    return LABEL[pupil] + (f", {BLUR_LABEL[blur]}" if blur else "")


def learn(args):
    import torch
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    from edge_magnitude import use_pupil
    from smappy.calibrate.model_views import pupil_views
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, RunSettings, bead_rois, camera_for, run_calibration
    from z_blur import use_z_blur

    import uipsf.workflows.beads as beads
    import uipsf.workflows.calibrate as calibrate
    import uipsf.workflows.scoring as scoring
    beads.MAX_DZ_NM = args.dz
    calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=args.chunk, max_dz_nm=args.dz)
    scoring.localise = functools.partial(scoring.localise, chunk_size=int(2 * args.window // args.dz + 1) * args.chunk)
    os.makedirs(args.out, exist_ok=True)
    import radial_magnitude
    for path in args.acquisitions:
        name, cond, files, bead_nm = acquisition(path)
        radial_magnitude.BEAD_NM = bead_nm             # the "… bead …" variants' ring, as `edge_pupil` builds them
        rois = bead_rois(files, camera_for(files[0]), dual=False)
        keep, focus, _ = select(rois, name)
        rois = subset(rois, keep, focus, args.window)
        optics = OpticsSettings(NA=args.grid_na, wavelength_nm=WAVELENGTH[cond["colour"]], n_immersion=N_IMM,
                                n_medium=N_MED[cond["medium"]], pixel_x_nm=104.7, pixel_y_nm=106.4)
        for variant in args.variants:
            pupil, _, blur = variant.partition("+")
            zblur, kind = BLURS.get(blur, (False, "free"))
            use_z_blur(zblur)
            if pupil == "minimal":                     # the user's minimal model, as `m5_consensus.py` learns it
                from m5_consensus import minimal
                magnitude = minimal("height", cond["medium"] == "mismatch", [2, 3, 5, 6, 8, 16], {}, {}, bead_nm)
            else:
                magnitude = use_pupil(pupil)
            model = BeadSettings(zernike_order=ORDER, magnitude=magnitude, blur=kind,
                                 blur_nm=float(np.mean((104.7, 106.4))) / np.sqrt(12) if kind == "fixed" else 30.0,
                                 bead_diameter_nm=bead_nm)
            run = run_calibration(files, False, optics, model, run=RunSettings(save_model=False), rois=rois,
                                  decisions={i: "include" for i in range(len(keep))})
            v = pupil_views(run)
            np.savez_compressed(os.path.join(args.out, f"{name}__{variant.replace(' ', '_')}.npz"),
                                z_nm=v.z_nm, data=v.data[0], model=v.model[0],
                                objective=float(run.cal.result.objective), **cond)
            print(name, variant, "done", flush=True)


def draw(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    found = sorted(glob.glob(os.path.join(folder, "*.npz")))
    for name in sorted({os.path.basename(f).split("__")[0] for f in found}):
        runs = {os.path.basename(f).split("__")[1][:-4].replace("_", " "): np.load(f) for f in found
                if os.path.basename(f).startswith(name + "__")}
        order = [v for v in runs if v.partition('+')[0] in LABEL]
        first = runs[order[0]]
        z = first["z_nm"]
        planes = [int(np.argmin(np.abs(z - p))) for p in PLANES_NM]
        data = first["data"]
        top = np.nanmax(data[planes])
        best = min(float(runs[v]["objective"]) for v in order)
        n = 1 + 2 * len(order)
        fig, axes = plt.subplots(n, len(planes), figsize=(1.7 * len(planes) + 1.6, 1.7 * n))
        rows = [("bead average", data, "magma", (0, top))]
        for v in order:
            m = runs[v]["model"]
            dn = float(runs[v]["objective"]) - best
            title = f"{variant_label(v)}\n(+{dn:,.0f} nats)" if dn else f"{variant_label(v)}\n(best)"
            rows.append((title, m, "magma", (0, top)))
            rows.append(("model − data", m - data, "RdBu_r", (-RESIDUAL * top, RESIDUAL * top)))
        for r, (label, vol, cmap, lim) in enumerate(rows):
            for c, k in enumerate(planes):
                ax = axes[r, c]
                # a cube root for the PSFs, so the rings far from focus are seen beside the bright core
                img = np.cbrt(np.clip(vol[k], 0, None) / top) * top if cmap == "magma" else vol[k]
                ax.imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if cmap != "magma":
                    rms = 100 * float(np.sqrt(np.nanmean(vol[k] ** 2))) / top
                    ax.text(0.03, 0.04, f"rms {rms:.2f} %", transform=ax.transAxes, fontsize=6.5, color="#333333")
                if r == 0:
                    ax.set_title(f"z = {z[k]:+.0f} nm", fontsize=8)
                if c == 0:
                    ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
        cond = (f"{first['colour']} nm, {'oil' if str(first['medium']) == 'match' else 'buffer'}, {first['lens']}"
                + (f", {float(first['bead_nm']):.0f} nm beads" if "bead_nm" in first else ""))
        fig.suptitle(f"{cond}: bead average and models (cube root; residual ±{100 * RESIDUAL:.0f} % of the peak)",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, f"{name}.png"), dpi=130)
        plt.close(fig)
        print(os.path.join(folder, f"{name}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisitions", nargs="*")
    ap.add_argument("--variants", nargs="+",
                    default=["zernike", "symmetric", "cos bead na asym", "cos signed bead na asym"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--draw", default=None)
    ap.add_argument("--window", type=float, default=1000.0)
    ap.add_argument("--grid-na", type=float, default=GRID_NA,
                    help="the pupil grid's edge; 1.51 with the minimal model, whose aperture is held at 1.50")
    ap.add_argument("--dz", type=float, default=100.0)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--memory", type=float, default=0.7)
    args = ap.parse_args()
    if args.draw:
        draw(args.draw)
    else:
        learn(args)


if __name__ == "__main__":
    main()
