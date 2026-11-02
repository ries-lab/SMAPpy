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
import dataclasses
import functools
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "synthetic")]
from bead_slices import measure  # noqa: E402,I001
from m5_bends import per_bead  # noqa: E402
from m5_media import GRID_NA, N_IMM, N_MED, ORDER, WAVELENGTH, acquisition, select, stage_drift, subset  # noqa: E402

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
    rest, _, extra = variant.partition("~")
    extra = {"zscale": ", z scale learnt", "na": ", NA learnt", "drift": ", lateral drift learnt", "": ""}.get(
        extra, f", n_imm {extra[1:]}")
    rest, _, photons = rest.partition("@")
    pupil, _, blur = rest.partition("+")
    return (LABEL[pupil] + (f", {BLUR_LABEL[blur]}" if blur else "") + (f", {PHOTON_LABEL[photons]}" if photons else "")
            + extra)


def _intensities():
    from uipsf.model import Bleaching, PerFrameFactor, PerObservation
    return {"": Bleaching, "frame": PerFrameFactor, "obs": PerObservation}


def fit_record(run, rois) -> dict:
    """What a fit leaves for its figures and its scorecard (`m5_scorecard.py`): the bead average and the model on
    common planes (`model_views.pupil_views`), the residual's standard error over beads (whether the averaged
    residual is noise), every bead's slice fits against its stack fit, the median bends, σ, the learnt z scale
    and aperture, and the deviance."""
    import torch
    from smappy.calibrate.model_views import pupil_views
    v = pupil_views(run)
    used = list(run.cal.beads.used)
    res = np.stack([np.asarray(v.beads[b][1][0]) - np.asarray(v.beads[b][0][0]) for b in used if b in v.beads])
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.nanstd(res, 0) / np.sqrt(np.sum(np.isfinite(res), 0))
    centres = [rois.records[i]["focus_nm"] for i in used]
    bends = measure(run, 600.0, centre_nm=centres)
    each = per_bead(run, centres)
    p = run.cal.result.params.physical()
    return dict(z_nm=v.z_nm, data=v.data[0], model=v.model[0], resid_se=se,
                objective=float(run.cal.result.objective), used=len(used),
                atlas_mean_deviance=float(run.cal.atlas.mean_deviance),
                curve_z_nm=np.asarray(bends["curve_z_nm"], float), curve_nm=np.asarray(bends["curve_nm"], float),
                bead_error=each["error"], bead_crlb=each["crlb"], bead_z=each["z"], bead_where=each["where"],
                bead_photons=each["photons"], bead_frame=each["frame"],
                bead_stack=np.array([int(rois.records[i]["stack"]) for i in used]),
                bead_x_px=np.array([rois.records[i]["x_px"] for i in used]),
                bead_y_px=np.array([rois.records[i]["y_px"] for i in used]),
                sigma_nm=p["psf.sigma"].detach().cpu().numpy(), z_scale=float(p["geometry.z_scale"]),
                edge_rho=float(p.get("pupil.edge_rho", torch.tensor(np.nan))),
                **{k: bends[k] for k in ("x_systematic_rms_nm", "y_systematic_rms_nm", "z_systematic_rms_nm",
                                         "z_scatter_over_crlb")})


def learn(args):
    import torch
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    from edge_magnitude import use_pupil
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

    from uipsf.model import BeadStackGeometry, ForwardModel
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
            # "~zscale": the stage's z scale learnt; "~n1.515": another immersion index; "~na": the minimal models'
            # aperture learnt (the user's: is the shared z bend a z scale, oil a few degrees off its index, or
            # the NA?)
            rest, _, extra = variant.partition("~")
            rest, _, photons = rest.partition("@")
            pupil, _, blur = rest.partition("+")
            # "~drift": a lateral drift per defocus (`m5_media.stage_drift`), the stage's shear kept out of coma
            calibrate.BeadStackGeometry = (functools.partial(stage_drift, "xy") if extra == "drift" else
                                           functools.partial(BeadStackGeometry, z_scale=extra == "zscale"))
            index = re.fullmatch(r"n(\d+\.\d+)", extra)            # "~n1.515"; "~na" is the aperture
            run_optics = dataclasses.replace(optics, n_immersion=float(index[1])) if index else optics
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
                # the aperture at NA 1.50 on whatever grid this is (it was held at 1.50/1.51 of the grid, so at
                # 1.49 on a grid at 1.50)
                magnitude = minimal("height", cond["medium"] == "mismatch", modes, {}, {}, bead_nm, args.grid_na,
                                    free_na=extra == "na")
            else:
                magnitude = use_pupil(pupil)
            model = BeadSettings(pixel_blur=False, zernike_order=ORDER, magnitude=magnitude, blur=kind,
                                 blur_nm=float(np.mean((104.7, 106.4))) / np.sqrt(12) if kind == "fixed" else 30.0,
                                 bead_diameter_nm=bead_nm)
            run = run_calibration(files, False, run_optics, model, run=RunSettings(save_model=False), rois=rois,
                                  decisions={i: "include" for i in range(len(keep))})
            record = fit_record(run, rois)
            np.savez_compressed(os.path.join(args.out, f"{name}__{variant.replace(' ', '_')}.npz"), **record,
                                window_nm=args.window, n_immersion=run_optics.n_immersion,
                                effective_na=record["edge_rho"] * args.grid_na, **cond)
            print(name, variant, "z bend", record["z_systematic_rms_nm"], flush=True)


def draw(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    found = sorted(glob.glob(os.path.join(folder, "*.npz")))
    for name in sorted({os.path.basename(f).split("__")[0] for f in found}):
        runs = {os.path.basename(f).split("__")[1][:-4].replace("_", " "): np.load(f) for f in found
                if os.path.basename(f).startswith(name + "__")}
        order = [v for v in runs if v.partition('~')[0].partition('@')[0].partition('+')[0] in LABEL]
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
        order = [v for v in runs if v.partition('~')[0].partition('@')[0].partition('+')[0] in LABEL]
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


def row_order(name: str) -> tuple:
    """Acquisitions in rows that pair oil with buffer: by bead size, colour and step, oil first."""
    import re
    bead = 40 if name.startswith("40nm") else 100
    colour = re.search(r"_(600|676)_", name)
    step = re.search(r"_(0\.\d+)um", name)
    return (bead, colour[1] if colour else "600", step[1] if step else "", "mismatch" in name, name)


def draw_bends(folder, axis: int = 2, lim: float = 60.0):
    """Each bead's error through z, fitted minus expected (``axis``: 0 x, 1 y, 2 z), against its z about its
    own focus: a row per acquisition, a column per model, a line per bead coloured by its x on the chip and
    the median black.  All beads bent alike is the model's z dependence; a fan ordered by colour is the field."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    found = sorted(glob.glob(os.path.join(folder, "*.npz")))
    names = sorted({os.path.basename(f).split("__")[0] for f in found}, key=row_order)
    variants = list(dict.fromkeys(os.path.basename(f).split("__")[1][:-4].replace("_", " ") for f in found))
    variants = [v for v in variants if all(os.path.exists(os.path.join(folder, f"{n}__{v.replace(' ', '_')}.npz"))
                                           for n in names)]
    if not names or not variants or "bead_z" not in np.load(found[0]).files:
        return
    fig, axes = plt.subplots(len(names), len(variants), figsize=(2.7 * len(variants), 2.3 * len(names)),
                             sharex=True, sharey=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    reach = max(np.nanmax(np.abs(np.load(f)["bead_z"])) for f in found)
    grid = np.arange(-np.ceil(reach / 100) * 100, np.ceil(reach / 100) * 100 + 1, 100.0)
    lim = lim * 2 if reach > 1200 else lim                  # whole stacks: the far slices err more
    for r, n in enumerate(names):
        for c, v in enumerate(variants):
            d = np.load(os.path.join(folder, f"{n}__{v.replace(' ', '_')}.npz"))
            z, err, x = d["bead_z"], d["bead_error"][..., axis], d["bead_x_px"]
            err = err - np.nanmedian(np.where(np.abs(z) <= 600, err, np.nan), axis=1, keepdims=True)
            span = (x - x.min()) / max(np.ptp(x), 1)
            ax = axes[r, c]
            for e in range(len(z)):
                o = np.argsort(z[e])
                ax.plot(z[e][o], err[e][o], color=cmap(span[e]), lw=0.6, alpha=0.7)
            med = [np.nanmedian(err[np.abs(z - g) <= 50]) if np.any(np.abs(z - g) <= 50) else np.nan for g in grid]
            ax.plot(grid, med, color="black", lw=1.8)
            ax.axhline(0, color="#999999", lw=0.6)
            ax.set_ylim(-lim, lim)
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(v, fontsize=7)                     # the short name: five columns of full labels overlap
            if c == 0:
                short = n.replace("20261029_", "").replace("_50ms", "").removesuffix("_1")
                ax.set_ylabel(f"{short}\n{'xyz'[axis]} error (nm)", fontsize=6.5)
            if r == len(names) - 1:
                ax.set_xlabel("z about focus (nm)", fontsize=6.5)
    fig.suptitle(f"{'xyz'[axis]}: each slice fitted alone minus the stack fit, per bead (colour: x on the chip), "
                 "median black", fontsize=8)
    fig.tight_layout()
    out = os.path.join(folder, f"bends_{'xyz'[axis]}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(out)


def draw_fields(folder):
    """The beads on the chip per acquisition (rows) and model (columns): `m5_bends.field_map`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from m5_bends import field_map
    found = sorted(glob.glob(os.path.join(folder, "*.npz")))
    names = sorted({os.path.basename(f).split("__")[0] for f in found}, key=row_order)
    variants = list(dict.fromkeys(os.path.basename(f).split("__")[1][:-4].replace("_", " ") for f in found))
    variants = [v for v in variants if all(os.path.exists(os.path.join(folder, f"{n}__{v.replace(' ', '_')}.npz"))
                                           for n in names)]
    if not names or not variants or "bead_z" not in np.load(found[0]).files:
        return
    fig, axes = plt.subplots(len(names), len(variants), figsize=(2.8 * len(variants), 2.7 * len(names)),
                             squeeze=False)
    pts = None
    for r, n in enumerate(names):
        for c, v in enumerate(variants):
            d = np.load(os.path.join(folder, f"{n}__{v.replace(' ', '_')}.npz"))
            pts = field_map(axes[r, c], d["bead_x_px"], d["bead_y_px"], d["bead_error"], d["bead_z"])
            if r == 0:
                axes[r, c].set_title(v, fontsize=7)
            if c == 0:
                axes[r, c].set_ylabel(n.replace("20261029_", "").replace("_50ms", "").removesuffix("_1"), fontsize=6.5)
    fig.colorbar(pts, ax=axes.ravel().tolist(), shrink=0.4, label="z scale − 1 (%)")
    fig.suptitle("each bead through focus on the chip: arrow, lateral slope (20 px = 20 nm/µm); colour, z scale; "
                 "size, z error about its line", fontsize=8)
    out = os.path.join(folder, "field.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(out)


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
        for axis in (2, 0, 1):
            draw_bends(args.draw, axis, 60.0 if axis == 2 else 20.0)
        draw_fields(args.draw)
    else:
        learn(args)


if __name__ == "__main__":
    main()
