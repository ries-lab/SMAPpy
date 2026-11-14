"""The figures of a calibration: the pupil beside its Zernikes, the residual atlas, the beads (plan §7.5).

Every function draws on a matplotlib figure it is given and returns
nothing: SMAPpy's window passes a tab's figure (a `SubFigure` on its *All*
page), a notebook passes `plt.figure()`, and the same pictures come out.
Nothing here imports pyplot, sets a figure's size or its layout engine.

They read a `Calibration` (the fit just made) or a `LearnedPSF` (a model
file) through `zernikes` and `pupil_maps`, so a previous model is drawn the
same way, and `draw_zernikes` draws one against the other: the comparison
a diagnosis is about.  The atlas and the bead table belong to a fit and are
drawn from a `Calibration` only.
"""
from __future__ import annotations

import numpy as np
import torch

from ..model.channels import prefix, view
from ..optics.zernike import ZERNIKE_NAMES, zernike_family, zernike_name

STATES = {"used": "#1f77b4", "held out": "#2ca02c", "rejected": "#d62728", "excluded": "#7f7f7f"}
#: a colour per key aberration (`KEY_ABERRATIONS`), the bars and the table's rows; the other modes grey
FAMILY_COLOURS = {"astigmatism": "#1f77b4", "coma 1": "#d62728", "coma 2": "#ff9896", "spherical 1": "#2ca02c",
                  "spherical 2": "#98df8a"}
OTHER = "#b0b0b0"
#: what a phase is shown in: radians, waves (rad / 2π) or nanometres of wavefront (waves × λ).  Radians are
#: what the model fits; a wave and its nanometres are what an optician reads, and λ/14 rms is Maréchal's limit
PHASE_UNITS = ("rad", "waves", "nm")
#: The one switch for the unit every aberration is shown in -- the pupils' maps, the Zernikes' bars and
#: tables, of every PSF learning (one channel, dual colour, in situ, 4Pi) -- where a figure or a page is not
#: told otherwise, and the first of the window's *units*.  nm of wavefront: an aberration's size in nm reads
#: the same at any wavelength, which radians and waves do not.  The model and its files stay in radians.
PHASE_UNIT = "nm"


def unit_choices() -> tuple[str, ...]:
    """`PHASE_UNITS` with `PHASE_UNIT` first, as the window offers them (its first choice is the one shown)."""
    return (PHASE_UNIT,) + tuple(u for u in PHASE_UNITS if u != PHASE_UNIT)


def phase_factor(wavelength_nm: float, units: str | None = None) -> tuple[float, str]:
    """(factor, label) that turn a phase in radians into ``units`` (`PHASE_UNITS`, `PHASE_UNIT` if None) at
    ``wavelength_nm``."""
    units = units or PHASE_UNIT
    if units == "rad":
        return 1.0, "rad"
    if units == "waves":
        return 1.0 / (2 * np.pi), "waves"
    if units == "nm":
        return float(wavelength_nm) / (2 * np.pi), "nm"
    raise ValueError(f"phase units {units!r}: one of {', '.join(PHASE_UNITS)}")


def _shared(source) -> dict:
    return source.shared() if callable(getattr(source, "shared", None)) else source.shared


def zernikes(source, which: str = "phase", channel: int = 0) -> tuple[list[int], np.ndarray, np.ndarray]:
    """(Noll modes, coefficients, CRLBs) of ``which`` ("phase", rad, or "magnitude") of ``source``, a
    `Calibration` or a `LearnedPSF`; NaN CRLBs where a coefficient was not fitted."""
    pupil = source.model.channels[channel].psf.pupil
    modes = list(getattr(pupil, f"{which}_modes", []))
    name = prefix(channel) + f"pupil.{which}"
    shared = _shared(source)
    if not modes or name not in shared:
        return [], np.zeros(0), np.zeros(0)
    values = torch.as_tensor(shared[name]).detach().cpu().double().numpy()
    std = source.result.std if hasattr(source, "result") else source.crlb
    crlb = (torch.as_tensor(std[name]).detach().cpu().double().numpy() if name in std
            else np.full(len(modes), np.nan))
    return modes, values, crlb


def coverslip_tilt(source) -> tuple[np.ndarray, np.ndarray] | None:
    """(tilt, CRLB) of the coverslip, (t_x, t_y) in degrees, of ``source`` (a `Calibration` or a
    `LearnedPSF`); None for a model without one (`model.geometry`).  NaN CRLBs for a tilt held, not learnt.
    Degrees, not the radians the model fits: a coverslip's tilt is read and set in degrees."""
    shared = _shared(source)
    if "geometry.tilt" not in shared:
        return None
    values = torch.as_tensor(shared["geometry.tilt"]).detach().cpu().double().numpy()
    std = source.result.std if hasattr(source, "result") else source.crlb
    crlb = (torch.as_tensor(std["geometry.tilt"]).detach().cpu().double().numpy() if "geometry.tilt" in std
            else np.full(2, np.nan))
    return np.degrees(values), np.degrees(crlb)


def tilt_text(source) -> str:
    """The coverslip's tilt in a line, *coverslip tilt x 0.52° ± 0.03°, y −0.10° ± 0.03°*; empty without one."""
    found = coverslip_tilt(source)
    if found is None:
        return ""
    (tx, ty), (ex, ey) = found

    def one(v, e):
        return f"{v:.2f}°" + ("" if not np.isfinite(e) else f" ± {e:.2f}°")
    return f"coverslip tilt x {one(tx, ex)}, y {one(ty, ey)}"


def _learnt(source, name):
    """(values, CRLBs) of the shared parameter ``name``, as float arrays; None when the model has none, NaN
    CRLBs where it was held."""
    shared = _shared(source)
    if name not in shared:
        return None
    values = torch.as_tensor(shared[name]).detach().cpu().double().numpy().reshape(-1)
    std = source.result.std if hasattr(source, "result") else source.crlb
    crlb = (torch.as_tensor(std[name]).detach().cpu().double().numpy().reshape(-1) if name in std
            else np.full(values.shape, np.nan))
    return values, crlb


def _pm(v, e, fmt) -> str:
    return fmt.format(v) + ("" if not np.isfinite(e) else " ± " + fmt.format(e))


def drift_text(source) -> str:
    """The stage's drift in a line where the model learnt one (`model.geometry`): a linear one's rate per
    stack, nm per µm of the stage's travel, or a free one's rms over the slices; empty without."""
    g = source.model.geometry
    kind = getattr(g, "drift", "none")
    if kind == "linear":
        found = _learnt(source, "geometry.drift_rate")
        if found is None:
            return ""
        rate, err = (a.reshape(-1, 2) for a in found)
        stacks = "; ".join(f"x {_pm(r[0], e[0], '{:.0f}')}, y {_pm(r[1], e[1], '{:.0f}')}"
                           for r, e in zip(rate, err, strict=True))
        if len(rate) > 4:                   # many stacks: their spread, not each
            stacks = (f"x {rate[:, 0].mean():.0f} ({rate[:, 0].min():.0f} … {rate[:, 0].max():.0f}), "
                      f"y {rate[:, 1].mean():.0f} ({rate[:, 1].min():.0f} … {rate[:, 1].max():.0f}) "
                      f"over {len(rate)} stacks")
        return f"drift nm/µm {stacks}"
    if kind == "free":
        found = _learnt(source, "geometry.drift")
        if found is None:
            return ""
        d = found[0].reshape(-1, 3)
        d = d - d.mean(0)
        axes = [a for a in "xyz" if a in getattr(g, "drift_axes", "xyz")]
        rms = ", ".join(f"{a} {np.sqrt(np.mean(d[:, 'xyz'.index(a)] ** 2)):.1f}" for a in axes)
        return f"drift rms nm {rms} over {len(d)} slices"
    return ""


#: every parameter a model can learn besides the Zernike modes and the emitters' own: (label, unit, scale, the
#: names of its elements).  `model_values` reports these; one it does not know comes under its own name, so a new
#: parameter is never left out of the table, only less nicely named
VALUES = {
    "pupil.gamma": ("γ (apodisation)", "", 1.0, None),
    "pupil.ring_h": ("SAF ring height h", "nm", 1.0, None),
    "pupil.na": ("NA", "", 1.0, None),
    "psf.sigma": ("blur σ", "nm", 1.0, ("x", "y")),
    "optics.n_med": ("sample index", "", 1.0, None),
    "geometry.z_scale": ("z scale", "", 1.0, None),
    "geometry.tilt": ("coverslip tilt", "°", 180.0 / np.pi, ("x", "y")),
    "geometry.height": ("molecules' height", "nm", 1.0, None),
    "geometry.plane": ("focal plane slope", "nm/µm", 1.0, ("x", "y")),
    "geometry.focus": ("focus", "nm", 1.0, "sweep"),
    "geometry.drift_rate": ("drift rate", "nm/µm", 1.0, "stack xy"),
    "transform": ("registration", "", 1.0, ("h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32")),
    "share": ("photon share", "", 1.0, None),
    "dz": ("z offset", "nm", 1.0, None),
    "field.phase": ("field-dependent phase term", "rad", 1.0, "term"),
}
#: per slice or frame, many numbers each: summarised by their rms about the mean (drift) or spread (factors)
SUMMARISED = {"geometry.drift": ("drift rms", "nm"), "intensity.slice_factor": ("brightness factor per slice", ""),
              "intensity.frame_factor": ("brightness factor per frame", "")}
#: the pupil's own maps and modes, which the pupil's page and the Zernike rows show
_NOT_VALUES = ("pupil.phase", "pupil.magnitude", "pupil.phase_map", "pupil.magnitude_map")


def model_values(source) -> list[dict]:
    """Every number the model learnt or held besides the Zernike modes and the emitters' own, one dict per
    element: ``channel`` (1-based), ``parameter`` (its label), ``element`` ("x", "sweep 3", ""), ``value``,
    ``error`` (the CRLB; NaN where held), ``unit`` and ``how`` ("learnt" or "held").  A parameter held at a
    value the model does not keep among its parameters -- in situ's γ, a ring height held -- is a row too.
    Many-valued per-slice parameters (the free drift, brightness factors) come as a summary row each."""
    shared = _shared(source)
    std = source.result.std if hasattr(source, "result") else getattr(source, "crlb", {})
    rows = []

    def add(c, label, element, value, error, unit, how):
        rows.append({"channel": c + 1, "parameter": label, "element": element, "value": float(value),
                     "error": float(error), "unit": unit, "how": how})

    n_channels = source.model.n_channels
    for c in range(n_channels):
        pre = prefix(c)
        pupil = source.model.channels[c].psf.pupil
        for name in sorted(k for k in shared if (k.startswith(pre) if pre else not k.startswith("ch"))):
            base = name[len(pre):]
            if base.startswith("emitter.") or base in _NOT_VALUES:
                continue
            values = torch.as_tensor(shared[name]).detach().cpu().double().numpy()
            learnt = name in std
            how = "learnt" if learnt else "held"
            if base in SUMMARISED:
                label, unit = SUMMARISED[base]
                if base == "geometry.drift":
                    d = values.reshape(-1, 3) - values.reshape(-1, 3).mean(0)
                    axes = getattr(source.model.geometry, "drift_axes", "xyz")
                    for i, a in enumerate("xyz"):
                        if a in axes:
                            add(c, label, f"{a}, {len(d)} slices", np.sqrt(np.mean(d[:, i] ** 2)), np.nan, unit, how)
                else:
                    v = values.reshape(-1)
                    add(c, label, f"mean of {len(v)}, ± their sd", v.mean(), v.std(), unit, how)
                continue
            label, unit, scale, elements = VALUES.get(base, (base, "", 1.0, None))
            errors = (torch.as_tensor(std[name]).detach().cpu().double().numpy() if learnt
                      else np.full(values.shape, np.nan))
            if elements == "stack xy":
                v, e = values.reshape(-1, 2), errors.reshape(-1, 2)
                for k in range(len(v)):
                    for i, a in enumerate("xy"):
                        add(c, label, f"stack {k + 1} {a}", v[k, i], e[k, i], unit, how)
                continue
            v, e = values.reshape(-1) * scale, errors.reshape(-1) * scale
            for k in range(len(v)):
                if len(v) == 1:
                    element = ""
                elif isinstance(elements, tuple) and len(elements) == len(v):
                    element = elements[k]
                elif isinstance(elements, str):
                    element = f"{elements} {k + 1}"
                else:
                    element = str(k + 1)
                add(c, label, element, v[k], e[k], unit, how)
        # held at values outside the parameters: in situ's γ from the beads, a ring height held
        if pre + "pupil.gamma" not in shared and getattr(pupil, "held_gamma", None) is not None:
            add(c, VALUES["pupil.gamma"][0], "", pupil.held_gamma, np.nan, "", "held")
        if getattr(pupil, "ring", False) and getattr(pupil, "held_ring_h", None) is not None:
            add(c, VALUES["pupil.ring_h"][0], "", pupil.held_ring_h, np.nan, "nm", "held")
    return rows


def _others_text(source, channel: int) -> list[str]:
    """`model_text`'s parts for what it does not name itself: the molecules' height, the focal plane's slope and
    the sweeps' foci, the z scale, a dual channel's share, offset and registration, field terms."""
    own = {VALUES[k][0] for k in ("pupil.gamma", "pupil.ring_h", "pupil.na", "psf.sigma", "optics.n_med",
                                  "geometry.tilt", "geometry.drift_rate")} | {SUMMARISED["geometry.drift"][0]}
    groups: dict = {}
    for row in model_values(source):
        if row["channel"] != channel + 1 or row["parameter"] in own:
            continue
        groups.setdefault(row["parameter"], []).append(row)
    parts = []
    for label, rows in groups.items():
        if all(r["how"] == "held" for r in rows) and label not in ("molecules' height",):
            continue                         # held at its start (a z scale of 1, a registration): the table says it
        unit = rows[0]["unit"]
        unit = ("" if unit in ("", "°") else " ") + unit
        if len(rows) <= 3:
            text = ", ".join((f"{r['element']} " if r["element"] else "") + _pm(r["value"], r["error"], "{:.3g}")
                             for r in rows)
            held = " held" if all(r["how"] == "held" for r in rows) else ""
            parts.append(f"{label} {text}{unit}{held}")
        else:
            v = np.array([r["value"] for r in rows])
            parts.append(f"{label} {v.min():.3g} … {v.max():.3g}{unit} over {len(v)}")
    return parts


def model_text(source, channel: int = 0) -> str:
    """The model's physical numbers in a line, with their CRLBs: the minimal pupil's γ and ring height,
    the blur (beyond the pixel's where the model has it under σ), a learnt sample index, the stage's drift and the
    coverslip's tilt, each where the model has it, then every other value it learnt (`model_values`; the
    molecules' height also when held).  Every value, held ones too, is in `zernike_table`'s last rows."""
    psf = source.model.channels[channel].psf
    pre = prefix(channel)
    parts = []
    gamma = _learnt(source, pre + "pupil.gamma")
    if gamma is not None:
        parts.append("γ " + _pm(gamma[0][0], gamma[1][0], "{:.3f}"))
        ring = _learnt(source, pre + "pupil.ring_h")
        if ring is not None:
            parts.append("ring h " + _pm(ring[0][0], ring[1][0], "{:.0f}") + " nm")
        na = _learnt(source, pre + "pupil.na")
        if na is not None:
            parts.append("NA " + _pm(na[0][0], na[1][0], "{:.3f}"))
    elif getattr(psf.pupil, "held_gamma", None) is not None:          # in situ: the bead model's γ
        parts.append(f"γ {psf.pupil.held_gamma:.3f} held")
    if getattr(psf.pupil, "ring", False) and getattr(psf.pupil, "held_ring_h", None) is not None:
        parts.append(f"ring h {psf.pupil.held_ring_h:.0f} nm held")
    sigma = _learnt(source, pre + "psf.sigma")
    if sigma is not None:
        parts.append("blur σ " + "/".join(_pm(v, e, "{:.1f}") for v, e in zip(*sigma, strict=True)) + " nm"
                     + (" beyond the pixel" if getattr(psf, "pixel_blur", False) else ""))
    n_med = _learnt(source, "optics.n_med")                         # one for every channel (`channels.SHARED`)
    if n_med is not None:
        parts.append("sample index " + _pm(n_med[0][0], n_med[1][0], "{:.4f}"))
    parts += [t for t in (drift_text(source), tilt_text(source)) if t]
    parts += _others_text(source, channel)
    return "; ".join(parts)


def phase_scale(source, channel: int = 0, units: str | None = None) -> tuple[float, str]:
    """`phase_factor` at the channel's own λ."""
    return phase_factor(source.model.channels[channel].psf.grid.optics.wavelength_nm, units)


def pupil_maps(source, channel: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(magnitude, phase) of channel ``channel``'s pupil, [P, P], NaN outside the aperture."""
    psf = source.model.channels[channel].psf
    with torch.no_grad():
        p = view({k: torch.as_tensor(v) for k, v in _shared(source).items()}, channel)
        magnitude, phase = psf.pupil({k: v.to(psf.grid.rho.device) for k, v in p.items()})
    inside = psf.grid.aperture.detach().cpu().numpy() > 0
    out = []
    for m in (magnitude, phase):
        m = m.detach().cpu().double().numpy()
        out.append(np.where(inside, m, np.nan))
    return out[0], out[1]


def _label(j: int) -> str:
    return f"{zernike_name(j)}\n{j}"


def draw_pupil(fig, source, channel: int = 0, vertical: bool = False, units: str | None = None) -> None:
    """The pupil's phase (in ``units``, `PHASE_UNITS`; `PHASE_UNIT` if None) and magnitude over the
    aperture, side by side or, ``vertical``, one above the other."""
    magnitude, phase = pupil_maps(source, channel)
    factor, label = phase_scale(source, channel, units)
    phase = phase * factor
    a, b = fig.subplots(2, 1) if vertical else fig.subplots(1, 2)
    lim = np.nanmax(np.abs(phase)) or 1.0
    im = a.imshow(phase, cmap="RdBu_r", vmin=-lim, vmax=lim)
    a.set_title(f"phase ({label})")
    fig.colorbar(im, ax=a, shrink=0.8)
    im = b.imshow(magnitude, cmap="viridis")
    b.set_title("magnitude")
    fig.colorbar(im, ax=b, shrink=0.8)
    for ax in (a, b):
        ax.set_xticks([])
        ax.set_yticks([])


def half_values(halves, source, which: str = "phase", channel: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(the full fit's sd from the halves [M], every half's value [2K, M]) of ``which`` of ``channel``, in
    the order of `zernikes`; NaN, and no values, where ``halves`` (a `HalfSplits`) has none."""
    modes = zernikes(source, which, channel)[0]
    name = prefix(channel) + f"pupil.{which}"
    if halves is None or name not in halves.std or not modes:      # no modes: an empty parameter
        return np.full(len(modes), np.nan), np.zeros((0, len(modes)))
    return np.asarray(halves.std[name], float), np.asarray(halves.values[name], float).reshape(-1, len(modes))


def draw_zernikes(fig, source, channel: int = 0, reference=None, reference_label: str = "start",
                  units: str | None = None, halves=None) -> None:
    """The phase (in ``units``, `PHASE_UNITS`; `PHASE_UNIT` if None) and, where there are any, the magnitude
    Zernikes with their CRLBs as error bars; ``reference`` (another `Calibration` or `LearnedPSF`) drawn
    beside them, mode by mode.  The key aberrations' legend goes below the bars, out of their way.

    With ``halves`` (`halves.HalfSplits`), a second, wider whisker on each bar is the uncertainty from the
    half-split fits, and every half's own value is a dot beside the bar: the whisker says how large, the
    dots whether one half went its own way."""
    which = [w for w in ("phase", "magnitude") if zernikes(source, w, channel)[0]]
    axes = np.atleast_1d(fig.subplots(len(which), 1, squeeze=False)[:, 0])
    handles = []
    for ax, w in zip(axes, which, strict=True):
        modes, values, crlb = zernikes(source, w, channel)
        factor, label = phase_scale(source, channel, units) if w == "phase" else (1.0, "relative")
        values, crlb = values * factor, crlb * factor
        x = np.arange(len(modes))
        width = 0.4 if reference is not None else 0.8
        families = [zernike_family(j) for j in modes]
        colours = [FAMILY_COLOURS.get(f, OTHER) for f in families]
        ax.bar(x, values, width, yerr=crlb, capsize=2, color=colours)
        if halves is not None:
            spread, each = half_values(halves, source, w, channel)
            if np.isfinite(spread).any():
                ax.errorbar(x, values, yerr=spread * factor, fmt="none", ecolor="#444444", elinewidth=2.5,
                            capsize=6, capthick=1.5, alpha=0.6)
                for row in each * factor:
                    ax.plot(x - width / 3, row, ".", color="k", markersize=3, alpha=0.7)
        if w == "phase":
            decimals = {"rad": 2, "waves": 3, "nm": 1}[label]
            # the key aberrations' values on their bars: the numbers a calibration is read by
            span = float(np.nanmax(np.abs(values))) if len(values) else 1.0
            for xi, v, f in zip(x, values, families, strict=True):
                if f is not None and np.isfinite(v):
                    # black: in the bar's own colour the pale ones (coma 2, spherical 2) could not be read
                    ax.annotate(f"{v:.{decimals}f}", (xi, v), xytext=(0, 2 if v >= 0 else -2),
                                textcoords="offset points", ha="center", va="bottom" if v >= 0 else "top",
                                fontsize="x-small", color="k")
            ax.margins(y=0.15 if span else 0.05)
            from matplotlib.patches import Patch
            handles += [Patch(color=c, label=f) for f, c in FAMILY_COLOURS.items() if f in families]
        if reference is not None:
            rmodes, rvalues, rcrlb = zernikes(reference, w, channel)
            rfactor = phase_scale(reference, channel, units)[0] if w == "phase" else 1.0
            rvalues, rcrlb = rvalues * rfactor, rcrlb * rfactor
            r = {j: (v, e) for j, v, e in zip(rmodes, rvalues, rcrlb, strict=True)}
            rv = np.array([r.get(j, (np.nan, np.nan))[0] for j in modes])
            re = np.array([r.get(j, (np.nan, np.nan))[1] for j in modes])
            ax.bar(x + width, rv, width, yerr=re, capsize=2, color="#555555", label=reference_label)
        if w == "phase" and reference is not None:
            from matplotlib.patches import Patch
            handles.append(Patch(color="#555555", label=reference_label))
        if w == "phase" and halves is not None:
            from matplotlib.lines import Line2D
            handles += [Line2D([], [], color="k", lw=1, label="± CRLB"),
                        Line2D([], [], color="#444444", lw=2.5, alpha=0.6,
                               label=f"± halves ({halves.n_splits} splits)"),
                        Line2D([], [], color="k", marker=".", lw=0, label="each half")]
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x + (width / 2 if reference is not None else 0))
        ax.set_xticklabels([_label(j) for j in modes], fontsize="small")
        ax.set_ylabel(label)
        ax.set_title(f"{w}: name and Noll index", fontsize="small")
    if handles:
        # under the phase bars, below their names: inside, it covered the bars it names
        axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize="small",
                       ncols=min(len(handles), 6), frameon=False)
    fig.supxlabel(ZERNIKE_NAMES, fontsize="x-small", wrap=True)


def draw_pupil_and_zernikes(fig, source, channel: int = 0, reference=None, reference_label: str = "start",
                            units: str | None = None, halves=None) -> None:
    """The pupil (`draw_pupil`, one map above the other) on the left, its Zernikes (`draw_zernikes`) on the
    right: the picture and the numbers it is made of, read together, the phase in ``units``."""
    left, right = fig.subfigures(1, 2, width_ratios=[1, 2.4])
    draw_pupil(left, source, channel, vertical=True, units=units)
    draw_zernikes(right, source, channel, reference, reference_label, units=units, halves=halves)
    # the numbers beside the Zernikes: γ, the ring, σ, the drift, the coverslip's tilt (whose coma is not theirs)
    text = model_text(source, channel)
    if text:
        fig.suptitle(text, fontsize="small", wrap=True)


def zernike_table(source, reference=None, halves=None) -> tuple[list[str], list[tuple], list]:
    """(columns, rows, colours) of every channel's Zernike coefficients with their CRLBs, and
    ``reference``'s beside them: the numbers of `draw_zernikes`, to read or copy, each key aberration's
    row in its bars' colour (`FAMILY_COLOURS`, None for the others).  NaN where a mode was not fitted.
    Below the modes, every other value the model learnt or held (`model_values`), in four columns of their
    own: so that the table is the whole model's numbers, not its Zernikes alone."""
    from ..optics.zernike import noll_to_nm
    # the phase in `PHASE_UNIT` first, the others after the magnitude
    first, *others = unit_choices()
    columns = ["channel", "Noll", "name", "aberration", "n", "m", f"phase ({first})", f"± ({first})", "magnitude",
               "±"]
    for u in others:
        columns += [f"phase ({u})", f"± ({u})"]
    if halves is not None:
        # the half-split uncertainty beside the CRLB, and how many CRLBs it is: above 1, the CRLB is optimistic
        columns += [f"± halves ({first})", "halves / CRLB", "magnitude ± halves"]
    if reference is not None:
        columns += [f"start phase ({first})", "start magnitude"]
    rows = []
    for c in range(source.model.n_channels):
        phase = {j: (v, e) for j, v, e in zip(*zernikes(source, "phase", c), strict=True)}
        magnitude = {j: (v, e) for j, v, e in zip(*zernikes(source, "magnitude", c), strict=True)}
        start_phase = start_magnitude = {}
        half_phase = dict(zip(zernikes(source, "phase", c)[0], half_values(halves, source, "phase", c)[0],
                              strict=True))
        half_magnitude = dict(zip(zernikes(source, "magnitude", c)[0],
                                  half_values(halves, source, "magnitude", c)[0], strict=True))
        if reference is not None:
            start_phase = {j: v for j, v, _ in zip(*zernikes(reference, "phase", c), strict=True)}
            start_magnitude = {j: v for j, v, _ in zip(*zernikes(reference, "magnitude", c), strict=True)}
        nan = (float("nan"), float("nan"))
        f = {u: phase_scale(source, c, u)[0] for u in PHASE_UNITS}
        for j in sorted(set(phase) | set(magnitude)):
            n, m = noll_to_nm(j)
            radians = phase.get(j, nan)
            row = (c + 1, j, zernike_name(j), zernike_family(j) or "", n, m, *(v * f[first] for v in radians),
                   *magnitude.get(j, nan))
            for u in others:
                row += tuple(v * f[u] for v in radians)
            if halves is not None:
                spread = half_phase.get(j, float("nan"))
                row += (spread * f[first], spread / radians[1], half_magnitude.get(j, float("nan")))
            if reference is not None:
                row += (start_phase.get(j, float("nan")) * phase_scale(reference, c, first)[0],
                        start_magnitude.get(j, float("nan")))
            rows.append(tuple(float(v) if isinstance(v, (np.floating, float)) else v for v in row))
    # every other value the model learnt or held (`model_values`), below the modes: the parameter in *name*,
    # its element in *aberration*, and its own columns at the end, NaN in the phase's
    values = model_values(source)
    if values:
        width = len(columns)
        columns += ["value", "± (value)", "unit", "learnt or held"]
        rows = [row + (float("nan"), float("nan"), "", "") for row in rows]
        for v in values:
            row = [float("nan")] * width
            row[:6] = [v["channel"], "", v["parameter"], v["element"], "", ""]
            rows.append(tuple(row) + (v["value"], v["error"], v["unit"], v["how"]))
    colours = [FAMILY_COLOURS.get(row[3]) for row in rows]
    return columns, rows, colours


def draw_atlas(fig, atlas, groups=("slice", "|z|", "radius", "field")) -> None:
    """The residual atlas: the mean deviance per group (≈ 1 where the model is right), each panel
    titled with its heterogeneity chi²/dof and in red when it is flagged; the findings as the figure's
    title."""
    shown = [g for g in groups if g in atlas.groups]
    axes = np.atleast_1d(fig.subplots(1, len(shown), squeeze=False)[0])
    for ax, name in zip(axes, shown, strict=True):
        g = atlas.groups[name]
        ax.plot(np.asarray(g.keys, float), g.deviance, "o-", ms=3)
        ax.axhline(1, color="k", lw=0.5)
        threshold = atlas.checks.get(name, (None, None))[1]
        flagged = name in atlas.flagged()
        ax.set_title(f"{name}: χ²/dof {g.chi2:.2g}" + ("" if threshold is None else f" (≤ {threshold:.2g})"),
                     color="#d62728" if flagged else "k", fontsize="small")
        ax.set_xlabel(name)
    axes[0].set_ylabel("deviance per pixel")
    text = "; ".join(f.suggests for f in atlas.findings) or f"nothing flagged (mean deviance {atlas.mean_deviance:.3g})"
    fig.suptitle(text, fontsize="small")


def draw_beads(fig, cal, selected=()) -> None:
    """Every bead where it is on the chip, by its state in the bead table; the ``selected`` beads ringed.
    Each point carries its bead (``pair_ids``, with the pick's index), so a window can select it."""
    ax = fig.subplots()
    selected = set(selected)
    for state, colour in STATES.items():
        rows = [r for r in cal.beads.rows if r.state == state and r.field_xy is not None]
        if rows:
            xy = np.array([r.field_xy for r in rows], float)
            points = ax.scatter(xy[:, 0], xy[:, 1], s=20, color=colour, label=f"{state} ({len(rows)})", picker=5)
            points.pair_ids = np.array([r.bead for r in rows])
            for r, (x, y) in zip(rows, xy, strict=True):
                ax.annotate(str(r.bead), (x, y), fontsize="xx-small", xytext=(2, 2), textcoords="offset points")
    ring = [r.field_xy for r in cal.beads.rows if r.bead in selected and r.field_xy is not None]
    if ring:
        ring = np.array(ring, float)
        ax.scatter(ring[:, 0], ring[:, 1], s=140, facecolors="none", edgecolors="k", linewidths=1.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.legend(fontsize="small")


def figures(cal, reference=None, halves=None) -> dict:
    """{title: draw(fig)} for every figure of ``cal``, each a closure that draws when asked, so tabs
    draw lazily; ``reference`` is the model it started from, if any."""
    out = {}
    for c in range(cal.model.n_channels):
        tag = "" if cal.model.n_channels == 1 else f" {c + 1}"
        out[f"Pupil and Zernikes{tag}"] = lambda fig, c=c, units=None: draw_pupil_and_zernikes(
            fig, cal, c, reference, units=units, halves=halves)
    if cal.atlas is not None:
        out["Residual atlas"] = lambda fig: draw_atlas(fig, cal.atlas)
    if any(r.field_xy is not None for r in cal.beads.rows):
        out["Beads"] = lambda fig: draw_beads(fig, cal)
    return out
