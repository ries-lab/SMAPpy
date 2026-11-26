"""Register layers: the shift of one layer against another, by cross-correlation.

(Not ``register.py``: see `plugins/registration.py` for why a plugin module
must never be called that.)

Wave 5, step 5b (`docs/smap_porting/wave5_design.md`).  A target layer is
measured against a reference layer -- localization layers as they show,
binned; image layers looked up on the same grid -- in x and y, and in z when
asked (SMAP's XYZdisplacement).  The work is `smappy.layer_registration`.

What it measured is a transformation like any other: it is added to the
session's named transformations (`Result.transformations`) under *name*,
and *Transform* applies it.  This plugin changes no coordinates: the form's
notice offers **Apply with Transform**, which opens Transform set to the new
name and to the target layer's file (or image) -- the one place a
transformation is applied (`transform.settings_for`).  To measure and apply
in one step, Transform's source *measured from the layers* calls the same
`measure_layers` and draws the same figure (`plot_registration`); this
plugin stays for measuring alone, and for keeping a shift to apply to other
files later.

Departures from SMAP: a shift only (no rotation or scale: later); one
correlation for both kinds of layer, where SMAP's RegisterLocs rendered
localizations and treated images apart; the shift found within *max shift*
(SMAP's 2500 nm), or anywhere when it is on auto.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import Context, FormState, Notice, Plot, Plugin, Result, param, register


def _layers():
    return [(i, f"layer {i}") for i in range(1, 7)]


@dataclass
class RegisterLayersSettings:
    reference: int = param(1, label="reference", choices=_layers,
                           help="the layer that stays")
    target: int = param(2, label="target", choices=_layers,
                        help="the layer whose shift is measured")
    bin_nm: float = param(20.0, label="bin", unit="nm", min=1,
                          help="the pixel of the pictures that are correlated")
    max_shift_nm: Optional[float] = param(2500.0, label="max shift", unit="nm",
                                          min=1,
                                          help="the farthest the peak is looked "
                                               "for; auto: anywhere")
    use_z: bool = param(False, label="z too",
                        help="also the shift in z; localizations with z only")
    z_bin_nm: float = param(10.0, label="z bin", unit="nm", min=0.1,
                            help="the bin of the z histograms", advanced=True)
    max_z_shift_nm: float = param(300.0, label="max z shift", unit="nm", min=1,
                                  help="the farthest the z peak is looked for",
                                  advanced=True)
    name: str = param("", label="name",
                      help="what the transformation is kept as; auto: the "
                           "target onto the reference")


def default_name(session, settings: RegisterLayersSettings) -> str:
    def called(n: int) -> str:
        layers = getattr(session, "layers", None) or []
        if 1 <= n <= len(layers):
            return layers[n - 1].name
        return f"layer {n}"
    return f"{called(settings.target)} onto {called(settings.reference)}"


def transform_target(session, layer: int) -> dict:
    """Transform's *target*, *file* and *channel* that move what layer
    ``layer`` (from 1) shows (`transform.layer_target`); where that is not
    one file, all files, for the user to narrow in Transform's form."""
    from .transform import layer_target
    found, _ = layer_target(session, layer)
    return found


@register("Analysis/Register/Register layers")
class RegisterLayers(Plugin):
    """Measure how far one layer is shifted against another, and keep the
    shift as a named transformation for Transform."""

    description = ("Measure the shift of a target layer against a reference "
                   "layer by cross-correlation -- localizations as they show, "
                   "or images -- in x and y, optionally z, and keep it as a "
                   "named transformation that Transform applies.")
    Settings = RegisterLayersSettings
    changes_locs = False
    version = "1"

    def describe(self, ctx: Optional[Context], settings) -> FormState:
        state = FormState(shown={"z_bin_nm": settings.use_z,
                                 "max_z_shift_nm": settings.use_z})
        session = getattr(ctx, "session", None)
        if session is None:
            return state
        choices = [(i + 1, f"{i + 1}: {layer.name}" + (" (image)" if layer.is_image else ""))
                   for i, layer in enumerate(session.layers)]
        state.choices["reference"] = choices
        state.choices["target"] = choices
        name = settings.name.strip() or default_name(session, settings)
        state.hints["name"] = default_name(session, settings)
        kept = session.transformations.get(name)
        if kept is not None and 1 <= settings.target <= len(session.layers):
            from .transform import settings_for
            state.notice = Notice(
                f"{name}: {kept}", action="Apply with Transform", window="transform",
                settings=settings_for(name, **transform_target(session, settings.target)),
                kind="info")
        return state

    def run(self, ctx: Context, settings: RegisterLayersSettings) -> Result:
        found = measure_layers(ctx, settings.reference, settings.target,
                               settings.bin_nm, settings.max_shift_nm,
                               settings.use_z, settings.z_bin_nm,
                               settings.max_z_shift_nm)
        name = settings.name.strip() or default_name(ctx.session, settings)
        t = found.transformation(name, {"reference": settings.reference,
                                        "target": settings.target,
                                        "bin_nm": settings.bin_nm})
        text = f"{name}: {found}; kept as {name!r}, applied with Transform"
        return Result(text=text, plot=plot_registration(found), settings=settings,
                      data={"registration": found, "transformation": t,
                            "shift_nm": found.shift_nm},
                      transformations={name: t})


def measure_layers(ctx: Context, reference: int, target: int, bin_nm: float = 20.0,
                   max_shift_nm: Optional[float] = 2500.0, use_z: bool = False,
                   z_bin_nm: float = 10.0, max_z_shift_nm: float = 300.0):
    """The `layer_registration.Registration` of layer ``target`` against layer
    ``reference`` (from 1), each as it shows: what this plugin measures and
    what Transform's source *measured from the layers* applies."""
    from ..layer_registration import register as measure
    if reference == target:
        raise ValueError("the reference and the target are the same layer")
    a = layer_input(ctx, reference, use_z)
    b = layer_input(ctx, target, use_z)
    ctx.report("correlating")
    return measure(a, b, bin_nm, max_shift_nm, use_z, z_bin_nm, max_z_shift_nm)


def layer_input(ctx: Context, number: int, use_z: bool) -> dict:
    """Layer ``number`` (from 1) as `layer_registration.register` takes it:
    the localizations it shows, or its image."""
    session = ctx.session
    if session is None:
        if number != 1:
            raise ValueError("other layers need a session")
        locs, selection = ctx.locs, ctx.selection
    else:
        if not 1 <= number <= len(session.layers):
            raise ValueError(f"there is no layer {number}")
        layer = session.layers[number - 1]
        if layer.is_image:
            if layer.image is None or layer.image.is_empty:
                raise ValueError(f"layer {number} shows no image yet")
            return {"image": layer.image, "frame": layer.frame}
        locs, selection = ctx.table(number - 1)
    shown = selection.apply(locs)
    if not len(shown):
        raise ValueError(f"layer {number} shows no localizations")
    out = {"x": np.asarray(shown["x_nm"], float), "y": np.asarray(shown["y_nm"], float)}
    if use_z and "z_nm" in shown:
        out["z"] = np.asarray(shown["z_nm"], float)
    return out


def plot_registration(found) -> Plot:
    """The two pictures over each other, and the correlation with its peak."""
    def draw(figure) -> None:
        both, corr = figure.subplots(1, 2)
        g = found.grid
        extent = (g.x0, g.x0 + g.nx * g.bin_nm, g.y0 + g.ny * g.bin_nm, g.y0)

        def scaled(p):
            hi = np.percentile(p[p > 0], 99) if np.any(p > 0) else 1.0
            return np.clip(p / (hi or 1.0), 0, 1)
        rgb = np.zeros(found.reference.shape + (3,))
        rgb[..., 0] = scaled(found.reference)
        rgb[..., 1] = scaled(found.target)
        rgb[..., 2] = scaled(found.target)
        both.imshow(rgb, extent=extent, interpolation="nearest")
        both.set(title="reference (red), target (cyan)", xlabel="x (nm)",
                 ylabel="y (nm)")
        lags = found.correlation.shape[0] // 2
        reach = lags * found.bin_nm
        corr.imshow(found.correlation, cmap="viridis", origin="upper",
                    extent=(-reach, reach, reach, -reach))
        corr.plot(found.dx_nm, found.dy_nm, "+", color="#d62728", ms=14, mew=1.5)
        corr.axhline(0, color="w", lw=0.5)
        corr.axvline(0, color="w", lw=0.5)
        corr.set(title=f"correlation: peak at ({found.dx_nm:.1f}, {found.dy_nm:.1f}) nm",
                 xlabel="shift x (nm)", ylabel="shift y (nm)")
        for ax in (both, corr):
            ax.title.set_fontsize(9)
            ax.tick_params(labelsize=7)
    return Plot(draw, panels=2, size=(10, 4.5))
