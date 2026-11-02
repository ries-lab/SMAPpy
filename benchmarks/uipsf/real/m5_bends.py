"""Each bead's z error through focus: fitted minus expected z against expected z, per bead, on M5's beads.

    python benchmarks/uipsf/real/m5_bends.py ACQUISITION [...] --out DIR [--dz 100] [--memory 0.7]
    python benchmarks/uipsf/real/m5_bends.py --draw DIR

The bends so far were medians over the beads (`bead_slices.measure`); the user asks what each bead does
(1 November 2026): whether the bend is one curve all beads share -- the model's z dependence -- or a
spread of beads, each bent its own way, as aberrations that vary across the field would make them.  Per
acquisition (`m5_media.acquisition`, its consensus beads, the whole field): free Zernikes to order 8 and
σx, σy free, ±1 µm about each bead's focus; then every slice localised alone on the learnt model
(`scoring.slice_fits`), as `measure` does, but kept per bead: the slice fit minus the stack fit in x, y, z,
its CRLB, and each bead's z about its own focus and its place on the chip.  ``--draw`` makes one figure of
every acquisition in DIR, a panel each: a line per bead, coloured by its x on the chip, the median bold.
"""
from __future__ import annotations

import argparse
import functools
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from m5_media import GRID_NA, N_IMM, N_MED, ORDER, WAVELENGTH, acquisition, select, subset  # noqa: E402

PIXEL = (104.7, 106.4)


def per_bead(run, centre_nm) -> dict:
    """The slice fits against the stack fit, per bead: (found − where) [E, K, 3] nm, its CRLB, z about focus, and
    where the stack fit puts each slice in its ROI (nm), whose remainder by the pixel is the bead's phase; each
    slice's photons, and its camera frame (`Batch.frame`, one number per frame of each stack, so two beads in one
    frame share it: their errors there share the stage's and the laser's) and stack."""
    from uipsf.workflows.beads import MAX_DZ_NM
    from uipsf.workflows.scoring import slice_fits
    cal = run.cal
    used = torch.as_tensor(np.asarray(cal.beads.used))
    batch = run.rois.batch.subset(used).thinned(MAX_DZ_NM)
    found, crlb, where, photons = slice_fits(cal.model, cal.shared(), batch, run.rois.noise, cal.result.params,
                                             photons=True)
    stage = np.broadcast_to(batch.z_nm.cpu().double().numpy(), where.shape[:2])
    k = np.abs(stage - np.asarray(centre_nm, float)[:, None]).argmin(1)
    z = where[..., 2] - where[np.arange(len(k)), k, 2][:, None]        # each bead's z about its own focus
    frame = batch.frame.cpu().numpy() if batch.frame is not None else np.full(where.shape[:2], -1)
    stack = batch.stack_id.cpu().numpy() if batch.stack_id is not None else np.full(len(k), -1)
    return dict(error=np.asarray(found - where), crlb=np.asarray(crlb), z=np.asarray(z), where=np.asarray(where),
                photons=np.asarray(photons), frame=frame, stack=stack)


def learn(args):
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, RunSettings, bead_rois, camera_for, run_calibration

    import uipsf.workflows.beads as beads
    import uipsf.workflows.calibrate as calibrate
    import uipsf.workflows.scoring as scoring
    from uipsf.model.pupil import ZernikePupil
    beads.MAX_DZ_NM = args.dz
    calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=1, max_dz_nm=args.dz)
    scoring.localise = functools.partial(scoring.localise, chunk_size=int(2 * args.window // args.dz + 1))
    calibrate.ZernikePupil = ZernikePupil
    os.makedirs(args.out, exist_ok=True)
    for path in args.acquisitions:
        name, cond, files, bead_nm = acquisition(path)
        target = os.path.join(args.out, f"{name}.npz")
        if os.path.exists(target):
            continue
        rois = bead_rois(files, camera_for(files[0]), dual=False)
        keep, focus, _ = select(rois, name)
        rois = subset(rois, keep, focus, args.window)
        optics = OpticsSettings(NA=GRID_NA, wavelength_nm=WAVELENGTH[cond["colour"]], n_immersion=N_IMM,
                                n_medium=N_MED[cond["medium"]], pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])
        model = BeadSettings(zernike_order=ORDER, magnitude="free", blur="free", blur_nm=30.0, bead_diameter_nm=bead_nm)
        run = run_calibration(files, False, optics, model, run=RunSettings(save_model=False), rois=rois,
                              decisions={i: "include" for i in range(len(keep))})
        used = list(run.cal.beads.used)
        out = per_bead(run, [rois.records[i]["focus_nm"] for i in used])
        np.savez_compressed(target, **out, x_px=np.array([rois.records[i]["x_px"] for i in used]),
                            y_px=np.array([rois.records[i]["y_px"] for i in used]), **cond,
                            sigma_nm=run.cal.result.params.physical()["psf.sigma"].detach().cpu().numpy())
        print(name, len(used), "beads", flush=True)


def draw(folder, axis: int = 2, lim: float = 60.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    cols = 4
    rows = -(-len(files) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), sharex=True, sharey=True, squeeze=False)
    for ax in axes.ravel()[len(files):]:
        ax.axis("off")
    cmap = plt.get_cmap("viridis")
    for ax, f in zip(axes.ravel(), files, strict=False):
        d = np.load(f)
        z, err, x = d["z"], d["error"][..., axis], d["x_px"]
        err = err - np.nanmedian(np.where(np.abs(z) <= 600, err, np.nan), axis=1, keepdims=True)   # each bead's offset
        span = (x - x.min()) / max(np.ptp(x), 1)
        for e in range(len(z)):
            order = np.argsort(z[e])
            ax.plot(z[e][order], err[e][order], color=cmap(span[e]), lw=0.7, alpha=0.7)
        grid = np.arange(-1000, 1001, 100.0)
        med = [np.nanmedian(err[np.abs(z - g) <= 50]) if np.any(np.abs(z - g) <= 50) else np.nan for g in grid]
        ax.plot(grid, med, color="black", lw=2)
        ax.axhline(0, color="#999999", lw=0.6)
        name = os.path.basename(f)[:-4].replace("20261029_", "").replace("_50ms", "").removesuffix("_1")
        ax.set_title(f"{name} ({len(z)} beads)", fontsize=7.5)
        ax.set_ylim(-lim, lim)
        ax.tick_params(labelsize=6.5)
    label = "xyz"[axis]
    for ax in axes[-1]:
        ax.set_xlabel("expected z about each bead's focus (nm)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"fitted − expected {label} (nm)", fontsize=7)
    fig.suptitle(f"{label} error of each slice fitted alone, per bead (colour: its x on the chip, left dark), "
                 "median black", fontsize=9)
    fig.tight_layout()
    out = os.path.join(folder, f"bends_{label}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(out)


def bead_slopes(z, error, z_range: float = 600.0) -> dict:
    """Each bead's error as a straight line in z within ±``z_range``: the x, y slopes (nm per µm of defocus:
    the bead moving sideways through focus -- coma, tilt), the z slope (its own z scale less one, %), and
    the z error's rms about its line (nm).  The constant part is the bead's position, which its stack fit
    has already absorbed."""
    E = len(z)
    out = dict(dx=np.full(E, np.nan), dy=np.full(E, np.nan), scale=np.full(E, np.nan), rest=np.full(E, np.nan))
    for e in range(E):
        ok = (np.abs(z[e]) <= z_range) & np.all(np.isfinite(error[e]), -1)
        if ok.sum() < 4:
            continue
        A = np.stack([np.ones(ok.sum()), z[e][ok] / 1000.0], 1)              # per µm
        coef, *_ = np.linalg.lstsq(A, error[e][ok], rcond=None)              # [2, 3]
        out["dx"][e], out["dy"][e] = coef[1, 0], coef[1, 1]
        out["scale"][e] = coef[1, 2] / 10.0                                  # nm per µm -> %
        out["rest"][e] = float(np.sqrt(np.mean((error[e][ok, 2] - A @ coef[:, 2]) ** 2)))
    return out


def field_map(ax, x_px, y_px, error, z, shape=None, arrow_nm_per_um: float = 20.0):
    """One acquisition's beads on the chip: an arrow of each bead's lateral slope through focus, the colour of
    its z scale, the size of what its z error leaves about its line."""
    s = bead_slopes(z, error)
    # capped: a bead that misfits badly (tens of nm) would hide the others
    size = 8 + 2.5 * np.clip(np.nan_to_num(s["rest"]), 0, 25) ** 1.4
    pts = ax.scatter(x_px, y_px, c=s["scale"], s=size, cmap="RdBu_r", vmin=-5, vmax=5, edgecolors="#333333",
                     linewidths=0.4, zorder=2)
    # an arrow of `arrow_nm_per_um` nm/µm is 20 px long
    k = 20.0 / arrow_nm_per_um
    ax.quiver(x_px, y_px, s["dx"] * k, s["dy"] * k, angles="xy", scale_units="xy", scale=1, width=0.006,
              color="#222222", zorder=3)
    ax.set_aspect("equal")
    ax.invert_yaxis()                                                        # as the image: y down
    if shape is not None:
        ax.set_xlim(0, shape[1])
        ax.set_ylim(shape[0], 0)
    ax.tick_params(labelsize=6)
    return pts


def draw_field(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    cols = 4
    rows = -(-len(files) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.ravel()[len(files):]:
        ax.axis("off")
    pts = None
    for ax, f in zip(axes.ravel(), files, strict=False):
        d = np.load(f)
        pts = field_map(ax, d["x_px"], d["y_px"], d["error"], d["z"])
        name = os.path.basename(f)[:-4].replace("20261029_", "").replace("_50ms", "").removesuffix("_1")
        ax.set_title(name, fontsize=7.5)
    fig.colorbar(pts, ax=axes.ravel().tolist(), shrink=0.5, label="z scale − 1 (%)")
    fig.suptitle("each bead through focus on the chip (px): arrow, lateral slope (20 px = 20 nm per µm of "
                 "defocus); colour, its z scale; size, its z error about that line", fontsize=8)
    out = os.path.join(folder, "field.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisitions", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--draw", default=None)
    ap.add_argument("--window", type=float, default=1000.0)
    ap.add_argument("--dz", type=float, default=100.0)
    ap.add_argument("--memory", type=float, default=0.7)
    args = ap.parse_args()
    if args.draw:
        for axis in (2, 0, 1):
            draw(args.draw, axis, 60.0 if axis == 2 else 20.0)
        draw_field(args.draw)
    else:
        learn(args)


if __name__ == "__main__":
    main()
