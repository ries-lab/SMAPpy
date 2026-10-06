"""The figures uiPSF's notebooks show, drawn from its result file.

uiPSF draws nothing useful while it learns; its demo notebooks call
``psflearning.makeplots`` afterwards on the saved result.  Those functions
make their own figures and call ``plt.show()``, which a result window cannot
use -- a smappy plot is *handed* a figure, or a subfigure on the window's All
page -- so the ones worth keeping are redrawn here from the same arrays (the
dict `convert.read` gives), each into the figure it is handed:

* `draw_data_vs_model`: ``showpsfvsdata`` / ``showpsfvsdata_insitu``
* `draw_localization_bias`: ``showlocalization``
* `draw_pupil`: ``showzernike`` (``showpupil`` for a model without Zernike
  coefficients)
* `draw_emitters`: ``showcoord``
* `draw_transformation`: ``showtransform``, as residuals rather than overlays

Left out: ``showlearnedparam`` (per-emitter position, photons, background,
drift: diagnostics of the learning rather than of the model), ``showpsf``
(the model alone: it is in `draw_data_vs_model`) and the field-dependent
maps, for models the plugin does not offer.

All of it is in uiPSF's coordinates: a channel's array as uiPSF was handed it
(the secondary flipped back if the splitter mirrors it), y down, x across.
"""
from __future__ import annotations

import numpy as np

# how many planes the data-versus-model figure shows, plus the x-z section
PLANES = 6
# the terms uiPSF's showzernike names, by Noll index
NAMED = {5: "astigmatism 45", 6: "astigmatism 0", 7: "coma y", 8: "coma x",
         11: "spherical", 22: "2nd spherical"}


def channels(data):
    """``[(label, res)]``: one entry, or one per channel of a multi-channel result."""
    res = data["res"]
    if "channel0" in res:
        names = ("main", "secondary")
        return [(names[i] if i < 2 else f"channel {i}", res[f"channel{i}"])
                for i in range(sum(k.startswith("channel") for k in res))]
    return [("", res)]


def _insitu(data) -> bool:
    return "insitu" in str(data["params"].get("PSFtype", ""))


def _rois(data, k):
    """Channel ``k``'s measured and modelled ROIs."""
    rois = data["rois"]
    multi = len(channels(data)) > 1
    return ((rois["psf_data"][k], rois["psf_fit"][k]) if multi
            else (rois["psf_data"], rois["psf_fit"]))


def _binned(data, k, res):
    """In situ: the molecules averaged per z plane of the model, as uiPSF does.

    Each molecule's fitted z picks the plane of the model it belongs to; the
    planes no molecule fell into stay empty.
    """
    measured, _ = _rois(data, k)
    model = np.asarray(res["I_model"], float)
    z = np.asarray(channels(data)[0][1]["pos"])[:, 0]
    zoffset = float(np.real(np.ravel(channels(data)[0][1].get("zoffset", [0]))[0]))
    index = np.digitize(z, zoffset + np.arange(len(model) + 1)) - 1
    average = np.zeros_like(model)
    for plane in range(len(model)):
        here = index == plane
        if here.any():
            average[plane] = measured[here].mean(axis=0)
    return average, model


def typical_bead(measured, modelled) -> int:
    """The bead whose model fits it as well as most: the median relative error."""
    error = (np.sum((measured - modelled) ** 2, axis=tuple(range(1, measured.ndim)))
             / np.sum(measured ** 2, axis=tuple(range(1, measured.ndim))))
    return int(np.argsort(error)[len(error) // 2])


def data_and_model(data):
    """``[(label, data stack, model stack, what)]`` per channel."""
    out = []
    for k, (label, res) in enumerate(channels(data)):
        if _insitu(data):
            measured, modelled = _binned(data, k, res)
            what = "molecules averaged per plane"
        else:
            stacks, fits = _rois(data, k)
            bead = typical_bead(stacks, fits)
            measured, modelled = stacks[bead], fits[bead]
            what = f"bead {bead}"
        out.append((label, np.asarray(measured, float), np.asarray(modelled, float), what))
    return out


def draw_data_vs_model(figure, data) -> None:
    """Measured above, uiPSF's model below, at planes through focus and in x-z."""
    rows = data_and_model(data)
    dz_um = float(data["params"]["pixel_size"]["z"])
    axes = figure.subplots(2 * len(rows), PLANES + 1, squeeze=False)
    insitu = _insitu(data)
    for r, (label, measured, modelled, what) in enumerate(rows):
        nz, ny, nx = measured.shape
        planes = np.linspace(0, nz - 1, PLANES + 2)[1:-1].round().astype(int)
        if insitu:          # uiPSF's own axis: the height above the coverslip
            zoffset = float(np.real(np.ravel(channels(data)[0][1].get("zoffset", [0]))[0]))
            z_um, unit = (zoffset + np.arange(nz)) * dz_um, "um high"
        else:               # the stack's: where the objective was, focus at 0
            z_um, unit = (np.arange(nz) - (nz - 1) / 2) * dz_um, "um stage"
        for row, stack, name in ((2 * r, measured, "data"), (2 * r + 1, modelled, "model")):
            top = max(float(stack.max()), 1e-12)
            for c, plane in enumerate(planes):
                ax = axes[row, c]
                ax.imshow(stack[plane], cmap="magma", vmin=0, vmax=top)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 2 * r:
                    ax.set_title(f"{z_um[plane]:.2f} {unit}" if insitu
                                 else f"{z_um[plane]:+.2f} {unit}", fontsize=8)
            ax = axes[row, -1]
            ax.imshow(stack[:, ny // 2, :], cmap="magma", aspect="auto", vmin=0, vmax=top)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 2 * r:
                ax.set_title("x-z", fontsize=8)
        axes[2 * r + 1, 0].set_ylabel("model", fontsize=8)
        axes[2 * r, 0].set_ylabel(f"{(label + ' ') if label else ''}data\n({what})", fontsize=8)


def has_localization_bias(data) -> bool:
    """Bead results carry each bead refitted plane by plane; in situ ones do not."""
    loc = data.get("locres", {}).get("loc")
    return bool(loc) and np.asarray(loc["z"]).ndim == 2 and np.asarray(loc["z"]).shape[1] > 1


def localization_bias(data):
    """``(x, y, z)`` bias in nm, beads by planes, as uiPSF's ``plotlocbias``.

    uiPSF refits every bead's every plane with the learnt model (spline MLE);
    x and y are the deviation from the bead's position, z the fitted plane
    less the plane it was taken at.
    """
    loc, p = data["locres"]["loc"], data["params"]["pixel_size"]
    x = np.asarray(loc["x"], float) * 1000 * float(p["x"])
    y = np.asarray(loc["y"], float) * 1000 * float(p["y"])
    z = np.asarray(loc["z"], float)
    z = (z - np.arange(z.shape[1])) * 1000 * float(p["z"])
    return x, y, z


def draw_localization_bias(figure, data) -> None:
    """Each bead's fitted x, y, z against where it was, through the stack."""
    x, y, z = localization_bias(data)
    dz_nm = 1000 * float(data["params"]["pixel_size"]["z"])
    centre = (z.shape[1] - 1) / 2
    stage = (np.arange(z.shape[1]) - centre) * dz_nm
    axes = figure.subplots(1, 3)
    for ax, values, name in zip(axes, (x, y, z), ("x", "y", "z")):
        ax.plot(stage, values.T, color="0.2", alpha=0.15, lw=0.8)
        ax.plot(stage, np.nanmedian(values, axis=0), color="C3", lw=1.5, label="median")
        ax.axhline(0, color="C0", lw=0.8)
        ax.set(xlabel="stage z (nm)", ylabel=f"{name} bias (nm)")
        inner = values[:, 2:-2] if values.shape[1] > 4 else values
        lo, hi = np.nanquantile(inner, [0.005, 0.995])
        ax.set_ylim(max(lo, -300) - 5, min(hi, 300) + 5)
    axes[0].legend(fontsize=8)


def _wavelength_nm(data) -> float:
    return 1000 * float(data["params"]["option"]["imaging"]["emission_wavelength"])


def pupil_images(res, wavelength_nm):
    """Magnitude and aberration phase (nm of wavefront) over the pupil.

    The phase is rebuilt from the Zernike terms from Noll 5 up, as uiPSF's
    ``showzernike`` does: piston, tilt and defocus only place the emitter.
    Without Zernike coefficients (a free pupil) it is the learnt pupil's own.
    """
    pupil = np.asarray(res["pupil"])
    pupil = pupil[0] if pupil.ndim > 2 else pupil
    aperture = np.abs(pupil) > 0
    if "zernike_coeff" in res and "zernike_polynomial" in res:
        coeff = np.asarray(res["zernike_coeff"], float).reshape(2, -1)
        Zk = np.asarray(res["zernike_polynomial"], float)
        n = min(len(Zk), coeff.shape[1])
        magnitude = np.tensordot(coeff[0, :n], Zk[:n], 1)
        phase = np.tensordot(coeff[1, 4:n], Zk[4:n], 1)
    else:
        magnitude, phase = np.abs(pupil), np.angle(pupil)
    phase = phase * wavelength_nm / (2 * np.pi)
    return (np.where(aperture, magnitude, np.nan), np.where(aperture, phase, np.nan))


def draw_pupil(figure, data) -> None:
    """Per channel: the pupil's magnitude and aberration, and the Zernike terms."""
    rows = channels(data)
    wavelength = _wavelength_nm(data)
    grid = figure.add_gridspec(len(rows), 3, width_ratios=(1, 1, 2.2))
    for r, (label, res) in enumerate(rows):
        magnitude, phase = pupil_images(res, wavelength)
        ax = figure.add_subplot(grid[r, 0])
        image = ax.imshow(magnitude, cmap="viridis")
        figure.colorbar(image, ax=ax, shrink=0.8)
        ax.set_title(f"{label} magnitude".strip(), fontsize=9)
        ax.set_axis_off()
        ax = figure.add_subplot(grid[r, 1])
        limit = np.nanmax(np.abs(phase)) or 1.0
        image = ax.imshow(phase, cmap="RdBu_r", vmin=-limit, vmax=limit)
        figure.colorbar(image, ax=ax, shrink=0.8, label="nm")
        ax.set_title(f"{label} aberration".strip(), fontsize=9)
        ax.set_axis_off()
        ax = figure.add_subplot(grid[r, 2])
        if "zernike_coeff" not in res:
            ax.text(0.5, 0.5, "free pupil: no Zernike terms", ha="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue
        coeff = np.asarray(res["zernike_coeff"], float).reshape(2, -1)
        noll = np.arange(1, coeff.shape[1] + 1)
        nm = coeff[1] * wavelength / (2 * np.pi)
        ax.bar(noll[4:], nm[4:], color="C0", width=0.8, label="phase (nm rms)")
        twin = ax.twinx()
        twin.plot(noll, coeff[0], ".", color="C1", ms=4, label="magnitude")
        twin.set_ylabel("magnitude", fontsize=8, color="C1")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlabel("Noll index", fontsize=8)
        ax.set_ylabel("phase (nm rms)", fontsize=8)
        named = [f"{name} {nm[j - 1]:+.0f}" for j, name in NAMED.items() if j <= len(nm)]
        ax.text(0.99, 0.97, "\n".join(named), transform=ax.transAxes, ha="right",
                va="top", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))


def draw_emitters(figure, data) -> None:
    """Every emitter found, and those uiPSF kept for the model, per channel."""
    rows = channels(data)
    size = np.asarray(data["rois"]["image_size"]).ravel()
    axes = figure.subplots(1, len(rows), squeeze=False)[0]
    beads = not _insitu(data)
    for ax, (label, res) in zip(axes, rows):
        found, kept = np.asarray(res["cor_all"]), np.asarray(res["cor"])
        ax.plot(found[:, -1], found[:, -2], ".", color="0.6", ms=3 if beads else 1,
                label=f"found ({len(found)})")
        ax.plot(kept[:, -1], kept[:, -2], "o", mfc="none", color="C3",
                ms=7 if beads else 3, label=f"used ({len(kept)})")
        ax.set_xlim(0, size[-1])
        ax.set_ylim(size[-2], 0)
        ax.set_aspect("equal")
        ax.set(xlabel="x (px)", ylabel="y (px)", title=label or None)
        ax.legend(fontsize=7, loc="lower right")


def transformation_residuals(data):
    """Main-channel positions, and the secondary's minus the main's mapped by ``T``.

    In uiPSF's channel arrays and pixels: ``T`` maps the main channel's
    (y, x, 1) - imgcenter to the secondary's, as ``showtransform`` applies it.
    """
    res = data["res"]
    T = np.asarray(res["T"], float).reshape(-1, 3, 3)[0]
    centre = np.asarray(res["imgcenter"], float).ravel()
    main = np.asarray(res["channel0"]["pos"], float)[:, 1:]
    secondary = np.asarray(res["channel1"]["pos"], float)[:, 1:]
    mapped = (np.c_[main, np.ones(len(main))] - centre) @ T
    mapped = mapped[:, :2] + centre[:2]
    return main, secondary - mapped


def draw_transformation(figure, data) -> None:
    """Where the transformation leaves each bead pair, in nm: arrows and spread."""
    main, residual = transformation_residuals(data)
    p = data["params"]["pixel_size"]
    nm = residual * 1000 * np.array([float(p["y"]), float(p["x"])])
    left, right = figure.subplots(1, 2)
    scale = max(float(np.nanmax(np.hypot(*nm.T))), 1.0)
    size = np.asarray(data["rois"]["image_size"]).ravel()
    left.quiver(main[:, 1], main[:, 0], nm[:, 1], -nm[:, 0], angles="xy",
                scale_units="xy", scale=scale / (0.08 * size[-1]), color="C3")
    left.plot(main[:, 1], main[:, 0], ".", color="0.3", ms=3)
    left.set_xlim(0, size[-1])
    left.set_ylim(size[-2], 0)
    left.set_aspect("equal")
    left.set(xlabel="x (px)", ylabel="y (px)",
             title=f"residuals, longest arrow {scale:.1f} nm")
    right.plot(nm[:, 1], nm[:, 0], "o", ms=4, color="C3")
    right.axhline(0, color="0.5", lw=0.8)
    right.axvline(0, color="0.5", lw=0.8)
    rms = np.sqrt(np.mean(nm ** 2, axis=0))
    right.set(xlabel="x residual (nm)", ylabel="y residual (nm)",
              title=f"rms x {rms[1]:.1f}, y {rms[0]:.1f} nm")
    right.set_aspect("equal", adjustable="datalim")


def figures(data):
    """``{tab name: (draw(figure), panels, size)}`` for what this result has."""
    n = len(channels(data))
    out = {"data vs model": (lambda fig: draw_data_vs_model(fig, data),
                             2 * n * (PLANES + 1), (11, 2.6 * n))}
    if has_localization_bias(data):
        out["localization bias"] = (lambda fig: draw_localization_bias(fig, data), 3, (11, 3.5))
    out["pupil"] = (lambda fig: draw_pupil(fig, data), 3 * n, (11, 3.3 * n))
    out["emitters" if _insitu(data) else "beads"] = (lambda fig: draw_emitters(fig, data),
                                                     n, (4.5 * n, 4.5))
    if n > 1 and "T" in data["res"]:
        out["channel transformation"] = (lambda fig: draw_transformation(fig, data), 2, (10, 4.5))
    return out
