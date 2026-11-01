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
from bead_slices import measure  # noqa: E402,I001
from m5_media import GRID_NA, N_IMM, N_MED, ORDER, WAVELENGTH, acquisition, select, subset  # noqa: E402

PLANES_NM = (-600.0, -300.0, 0.0, 300.0, 600.0)
#: the residual's colour range, a fraction of the bead average's peak
RESIDUAL = 0.05
#: the sections' residual range, % of each plane's own peak
SECTION_RESIDUAL = 10.0
LABEL = {"zernike": "free Zernikes", "symmetric": "Zernikes, m = 0 only", "cos bead na asym": "cos θ^γ + m ≠ 0",
         "cos bead na": "cos θ^γ, symmetric", "cos signed bead na asym": "cos θ^γ × two-way + m ≠ 0",
         "cos steps bead na asym": "cos θ^γ × monotone + m ≠ 0",
         "minimal": "minimal: cos θ^γ, NA 1.50, ring height, 6 m ≠ 0",
         "minimal0": "minimal, symmetric: cos θ^γ, NA 1.50, ring height"}
#: a variant's blur, after a "+" ("zernike+zpixel"): (σ(z), `BeadSettings.blur`); none: σx and σy learnt
BLURS = {"sigma": (False, "round"), "pixel": (False, "fixed"), "zpixel": (True, "fixed"), "z": (True, "round")}
BLUR_LABEL = {"sigma": "σ", "pixel": "σ = pixel", "zpixel": "σ(z), σ₀ = pixel", "z": "σ(z)"}


PHOTON_LABEL = {"frame": "photons × factor per frame", "obs": "photons free per slice"}


def variant_label(variant: str) -> str:
    rest, _, photons = variant.partition("@")
    pupil, _, blur = rest.partition("+")
    return LABEL[pupil] + (f", {BLUR_LABEL[blur]}" if blur else "") + (f", {PHOTON_LABEL[photons]}" if photons else "")


def _intensities():
    from uipsf.model import Bleaching, PerFrameFactor, PerObservation
    return {"": Bleaching, "frame": PerFrameFactor, "obs": PerObservation}


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

    from uipsf.model import ForwardModel
    INTENSITY = _intensities()
    for path in args.acquisitions:
        name, cond, files, bead_nm = acquisition(path)
        radial_magnitude.BEAD_NM = bead_nm             # the "… bead …" variants' ring, as `edge_pupil` builds them
        rois = bead_rois(files, camera_for(files[0]), dual=False)
        keep, focus, _ = select(rois, name)
        rois = subset(rois, keep, focus, args.window)
        optics = OpticsSettings(NA=args.grid_na, wavelength_nm=WAVELENGTH[cond["colour"]], n_immersion=N_IMM,
                                n_medium=N_MED[cond["medium"]], pixel_x_nm=104.7, pixel_y_nm=106.4)
        for variant in args.variants:
            rest, _, photons = variant.partition("@")
            pupil, _, blur = rest.partition("+")
            # the photons of each bead and slice: the default smooth curve per bead, that times a free factor per
            # camera frame of each stack (the laser), or a free count per bead and slice -- whether the pixels'
            # biases along z are the photons' and not the PSF's (the user's)
            calibrate.ForwardModel = functools.partial(ForwardModel, intensity=INTENSITY[photons]())
            zblur, kind = BLURS.get(blur, (False, "free"))
            use_z_blur(zblur)
            if pupil in ("minimal", "minimal0"):
                # the user's minimal model, as `m5_consensus.py` learns it; "minimal0" without the six m ≠ 0
                # modes, the physical symmetric magnitude alone
                from m5_consensus import minimal
                modes = [2, 3, 5, 6, 8, 16] if pupil == "minimal" else []
                magnitude = minimal("height", cond["medium"] == "mismatch", modes, {}, {}, bead_nm)
            else:
                magnitude = use_pupil(pupil)
            model = BeadSettings(zernike_order=ORDER, magnitude=magnitude, blur=kind,
                                 blur_nm=float(np.mean((104.7, 106.4))) / np.sqrt(12) if kind == "fixed" else 30.0,
                                 bead_diameter_nm=bead_nm)
            run = run_calibration(files, False, optics, model, run=RunSettings(save_model=False), rois=rois,
                                  decisions={i: "include" for i in range(len(keep))})
            v = pupil_views(run)
            # the bends, each slice localised alone on the model (`bead_slices.measure`), over ±600 nm
            bends = measure(run, 600.0, centre_nm=[rois.records[i]["focus_nm"] for i in run.cal.beads.used])
            np.savez_compressed(os.path.join(args.out, f"{name}__{variant.replace(' ', '_')}.npz"),
                                z_nm=v.z_nm, data=v.data[0], model=v.model[0],
                                objective=float(run.cal.result.objective), window_nm=args.window,
                                used=len(run.cal.beads.used), curve_z_nm=np.asarray(bends["curve_z_nm"], float),
                                curve_nm=np.asarray(bends["curve_nm"], float),
                                sigma_nm=run.cal.result.params.physical()["psf.sigma"].detach().cpu().numpy(),
                                **{k: bends[k] for k in ("x_systematic_rms_nm", "y_systematic_rms_nm",
                                                         "z_systematic_rms_nm", "z_scatter_over_crlb")}, **cond)
            print(name, variant, "z bend", bends["z_systematic_rms_nm"], flush=True)


def draw(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    found = sorted(glob.glob(os.path.join(folder, "*.npz")))
    for name in sorted({os.path.basename(f).split("__")[0] for f in found}):
        runs = {os.path.basename(f).split("__")[1][:-4].replace("_", " "): np.load(f) for f in found
                if os.path.basename(f).startswith(name + "__")}
        order = [v for v in runs if v.partition('@')[0].partition('+')[0] in LABEL]
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
            # each model on the bead average's planes: the planes every used bead reaches can differ by one
            at = [int(np.argmin(np.abs(runs[v]["z_nm"] - z[k]))) for k in range(len(z))]
            m = runs[v]["model"][at]
            own = runs[v]["data"][at]
            dn = float(runs[v]["objective"]) - best
            title = f"{variant_label(v)}\n(+{dn:,.0f} nats)" if dn else f"{variant_label(v)}\n(best)"
            rows.append((title, m, "magma", (0, top)))
            rows.append(("model − data", m - own, "RdBu_r", (-RESIDUAL * top, RESIDUAL * top)))
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


def draw_sections(folder):
    """The PSFs through z: XZ and YZ sections through the bead's centre, the bead average and each model on a
    cube root, the residual as a fraction of each plane's own peak (±`SECTION_RESIDUAL`) so that the defocused
    planes show their misfit beside the focus."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    found = sorted(glob.glob(os.path.join(folder, "*.npz")))
    for name in sorted({os.path.basename(f).split("__")[0] for f in found}):
        runs = {os.path.basename(f).split("__")[1][:-4].replace("_", " "): np.load(f) for f in found
                if os.path.basename(f).startswith(name + "__")}
        order = [v for v in runs if v.partition('@')[0].partition('+')[0] in LABEL]
        first = runs[order[0]]
        z, data = first["z_nm"], first["data"]
        cy, cx = np.unravel_index(np.nanargmax(np.nansum(data, 0)), data.shape[1:])
        top = float(np.nanmax(data))
        px = 105.5                                        # nm, the mean pixel: sections drawn to scale
        Y, X = data.shape[1:]
        extent_x = [-(X // 2) * px, (X - X // 2) * px, z.min(), z.max()]

        def sections(vol, cy=cy, cx=cx):
            return vol[:, cy, :], vol[:, :, cx]           # [z, x], [z, y]

        rows = [("bead average", data, None)]
        best = min(float(runs[v]["objective"]) for v in order)
        for v in order:
            at = [int(np.argmin(np.abs(runs[v]["z_nm"] - zk))) for zk in z]
            m, own = runs[v]["model"][at], runs[v]["data"][at]
            dn = float(runs[v]["objective"]) - best
            rows.append((f"{variant_label(v)}\n(+{dn:,.0f} nats)" if dn else f"{variant_label(v)}\n(best)", m, None))
            peak = np.nanmax(own, axis=(1, 2), keepdims=True)
            rows.append(("model − data,\n% of the plane's peak", (m - own) / peak * 100, "residual"))
        fig, axes = plt.subplots(len(rows), 2, figsize=(6.4, 2.0 * len(rows)), squeeze=False)
        for r, (label, vol, kind) in enumerate(rows):
            for c, (sec, axis) in enumerate(zip(sections(vol), ("x", "y"), strict=True)):
                ax = axes[r, c]
                if kind is None:
                    img = np.cbrt(np.clip(sec, 0, None) / top)
                    ax.imshow(img, cmap="magma", vmin=0, vmax=1, origin="lower", aspect="auto", extent=extent_x)
                else:
                    ax.imshow(sec, cmap="RdBu_r", vmin=-SECTION_RESIDUAL, vmax=SECTION_RESIDUAL, origin="lower",
                              aspect="auto", extent=extent_x)
                ax.set_xticks([-1000, 0, 1000])
                ax.tick_params(labelsize=6)
                if r == 0:
                    ax.set_title(f"{axis}z section", fontsize=8)
                if r == len(rows) - 1:
                    ax.set_xlabel(f"{axis} (nm)", fontsize=7)
                if c == 0:
                    ax.set_ylabel(f"{label}\nz (nm)", fontsize=6.5)
        cond = (f"{first['colour']} nm, {'oil' if str(first['medium']) == 'match' else 'buffer'}, {first['lens']}"
                + (f", {float(first['bead_nm']):.0f} nm beads" if "bead_nm" in first else ""))
        fig.suptitle(f"{cond}: through z (cube root; residual ±{SECTION_RESIDUAL:.0f} % of each plane's peak)",
                     fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, f"{name}_sections.png"), dpi=130)
        plt.close(fig)
        print(os.path.join(folder, f"{name}_sections.png"))


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
        draw_sections(args.draw)
    else:
        learn(args)


if __name__ == "__main__":
    main()
