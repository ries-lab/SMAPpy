"""SPINNA: which fraction of the molecules sits in monomers, dimers, ... --
from their nearest-neighbour distances against simulations.

The algorithm is `smappy.spinna`, ported from Picasso (MIT); this module
only reads the session and draws.  The **targets** are the visible
localization layers in order -- A the first, B the second -- each as it
shows (its file and channel, its filter), so two proteins imaged in two
rounds or channels are two layers; a layer is meant to show molecules (the
centres of *Molecular mapping*), one row each.  The **window** is the
region's shape, else the drawn ROI, else the area the molecules occupy
(`spatial_statistics.window_of_points`, at a coarser pixel than *Spatial
statistics*' because molecules are sparser than localizations) or their
bounding box -- SMAPpy's window, not Picasso's density masks.  The
**structures** are a list of parts, each a line of molecules
(``"A -10 0; A 10 0"``), filled from a YAML preset in
``data/spinna_structures`` or the config folder's ``spinna_structures``.

``per_region``: a region's run hands back the proportions and their errors
as values, so the oligomeric state can be compared cell by cell (one ROI
each) -- the way to deal with a density that changes across a field, which
Picasso's masks handle instead.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import (Context, FormState, Notice, Plot, Plugin, PreflightChoice, PreflightQuestion,
               Result, Stopped, param, register)

SHIPPED = Path(__file__).resolve().parent.parent / "data" / "spinna_structures"
# above this many seconds a run asks first
ASK_ABOVE_S = 10.0
# the occupied area's pixel: molecules of tens per um^2 leave most 200 nm
# pixels empty, which the closing does not bridge
OCCUPIED_BIN_NM = 500.0
FITS = (("proportions", "proportions of structures"),
        ("efficiency", "labelling efficiency"))


def structure_folders() -> List[Path]:
    """Where models are: the user's, then the shipped ones."""
    from ..config import config_dir
    return [config_dir() / "spinna_structures", SHIPPED]


@dataclass
class StructureSettings:
    name: str = param("dimer", label="name",
                      help="what the results call it")
    # not `on`: YAML reads a bare `on:` as true
    enabled: bool = param(True, label="on", help="untick to leave it out of the model")
    molecules: str = param("A -10 0; A 10 0", label="molecules", unit="nm", kind="text",
                           help="x y (z) of each molecule about the centre, ';' "
                                "between them; a letter first names its target: A "
                                "the first visible layer, B the second")


def default_structures() -> List[StructureSettings]:
    return [StructureSettings("monomer", True, "A 0 0"),
            StructureSettings("dimer", True, "A -10 0; A 10 0")]


@dataclass
class SpinnaSettings:
    fit: str = param("proportions", label="fit", choices=FITS,
                     help="the share of each structure, or the labelling efficiency "
                          "of two tags on one protein (two visible layers)")
    preset: str = param("", label="model", kind="preset", store=structure_folders,
                        help="a set of structures to start from: it fills the list "
                             "below")
    structures: List[StructureSettings] = param(
        default_factory=default_structures, label="structures", item_on="enabled",
        help="the complexes the molecules may be in; select one to set it")
    distance_nm: str = param("10", label="tag distance", unit="nm",
                             help="between the two tags of one protein; several, "
                                  "comma-separated, are each tried")
    efficiency: str = param("0.5", label="labelling efficiency",
                            help="the fraction of molecules detected: one value, or "
                                 "one per target, '0.5, 0.4'")
    label_unc_nm: str = param("5", label="label uncertainty", unit="nm",
                              help="the spread of a detected molecule about the "
                                   "protein: one value, or one per target")
    dimensions: str = param("2D", label="dimensions", choices=(("2D", "2D"), ("3D", "3D")),
                            help="3D: the distances in x, y and z, structures placed "
                                 "through the molecules' z range")
    rotation: str = param("plane", label="rotation", advanced=True,
                          choices=(("plane", "in the plane"), ("3D", "in 3D"),
                                   ("none", "none")),
                          help="how each structure is turned at random; in 3D for a "
                               "2D fit is the turned structure seen from above")
    window: str = param("occupied", label="without an ROI",
                        choices=(("occupied", "the occupied area"),
                                 ("box", "the bounding box")),
                        help="the window when neither a region nor a drawn ROI "
                             "gives one")
    occupied_bin_nm: float = param(OCCUPIED_BIN_NM, label="occupied bin", unit="nm",
                                   min=1.0, advanced=True,
                                   help="the pixel of the occupied area's mask")
    search: str = param("coarse_to_fine", label="search",
                        choices=(("coarse_to_fine", "coarse to fine"),
                                 ("brute_force", "every candidate")),
                        help="how the candidate mixtures are tried")
    granularity: int = param(21, label="grid steps", min=2,
                             help="values of each free structure's number, from none "
                                  "to all")
    simulations: int = param(10, label="simulations", min=1,
                             help="per candidate, their distances pooled")
    bootstrap: bool = param(True, label="bootstrap errors",
                            help="fit simulations of the answer again, for the "
                                 "errors")
    bootstraps: int = param(20, label="rounds", min=2, beside=True,
                            help="simulated data sets fitted")
    seed: int = param(0, label="seed", min=0, advanced=True,
                      help="the same seed and settings give the same answer")


def preset_values(name: str) -> Dict[str, Any]:
    """A model's settings, spelt as the form spells them; only the ones the
    file names, so a model of structures leaves the efficiency alone."""
    from .. import presets as store
    from . import settings_from, settings_values
    found = store.read(structure_folders(), name)
    values = settings_values(settings_from(SpinnaSettings, found))
    named = {k.split(".")[0] for k in found}
    out = {k: v for k, v in values.items() if k.split(".")[0] in named and k != "preset"}
    out["preset"] = name
    return out


def value_key(name: str) -> str:
    """A structure's name as part of a value's name: ``dimer (20 nm)`` ->
    ``dimer_20_nm``."""
    key = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()
    return key or "structure"


def _structures(settings: SpinnaSettings):
    from .. import spinna
    on = [s for s in settings.structures if s.enabled]
    if not on:
        raise ValueError("tick a structure")
    names = [s.name.strip() or f"structure {i + 1}" for i, s in enumerate(on)]
    if len({value_key(n) for n in names}) < len(names):
        raise ValueError("two structures have the same name: " + ", ".join(names))
    return [spinna.parse_structure(n, s.molecules) for n, s in zip(names, on)]


def _n_targets(settings: SpinnaSettings) -> int:
    from .. import spinna
    if settings.fit == "efficiency":
        return 2
    return spinna.n_targets_of(_structures(settings))


def _distances(text: str) -> List[float]:
    try:
        out = [float(p) for p in re.split(r"[,;\s]+", str(text).strip()) if p]
    except ValueError:
        raise ValueError(f"tag distance: {text!r} is not a list of numbers") from None
    if not out or any(d < 0 for d in out):
        raise ValueError("tag distance: one or more distances of 0 nm or more")
    return out


@register("Analysis/DNA-PAINT/SPINNA")
class Spinna(Plugin):
    """The share of the molecules in each kind of complex (monomers, dimers,
    ...) from their nearest-neighbour distances, against simulations of
    every mixture; or the labelling efficiency of two tags."""

    Settings = SpinnaSettings
    changes_locs = False
    per_region = True
    version = "1"

    # -- the form

    def describe(self, ctx: Optional[Context], settings: SpinnaSettings) -> FormState:
        proportions = settings.fit == "proportions"
        state = FormState(shown={
            "preset": proportions, "structures": proportions, "efficiency": proportions,
            "distance_nm": not proportions,
            "occupied_bin_nm": settings.window == "occupied",
            "bootstraps": settings.bootstrap})
        try:
            n = _n_targets(settings)
            if not proportions:
                _distances(settings.distance_nm)
        except ValueError as error:
            state.notice = Notice(str(error), kind="warning")
            return state
        from ..spinna import letter
        targets = ", ".join(f"{letter(t)} the {_nth(t)} visible layer" for t in range(n))
        state.notice = Notice(f"targets: {targets}", kind="info")
        return state

    def react(self, changed: str, settings: SpinnaSettings) -> Optional[Dict[str, Any]]:
        """Choosing a model fills the structures from it."""
        if changed == "preset" and settings.preset:
            return preset_values(settings.preset)
        return None

    # -- the data

    def _layers(self, ctx: Context):
        from .histogram import _layers
        return _layers(ctx)

    def _drawn_roi(self, ctx: Context, selection):
        from ..regions import Region
        if ctx.region is not None:
            return ctx.region if isinstance(ctx.region, Region) else None
        if ctx.per_region:
            return None
        roi = getattr(ctx.session, "roi", None)
        if isinstance(roi, Region) and getattr(selection, "roi", None) is not None:
            return roi
        return None

    def _gather(self, ctx: Context, settings: SpinnaSettings, n_targets: int):
        """``(fields, names, colours, shape, unused)``: a `spinna.Field` per
        file, the targets' layer names and colours, the window's shape, and
        the visible layers left over."""
        from ..spinna import Field, letter
        from .spatial_statistics import window_of_points, window_of_region
        found = self._layers(ctx)
        if len(found) < n_targets:
            raise ValueError(
                f"the model has {n_targets} targets ("
                + ", ".join(letter(t) for t in range(n_targets))
                + f"), one per visible layer, and {len(found)} layer"
                + ("s are" if len(found) != 1 else " is") + " visible")
        three = settings.dimensions == "3D"
        names, colours, columns, shape = [], [], [], None
        for name, colour, (table, selection) in found[:n_targets]:
            needed = ("x_nm", "y_nm") + (("z_nm",) if three else ())
            for c in needed:
                if c not in table:
                    raise ValueError(f"layer {name} has no attribute {c!r}; it has "
                                     + ", ".join(sorted(table.keys())))
            xyz = np.column_stack([np.asarray(table[c], dtype=float) for c in needed])
            # a new array: the selection's own mask is not ours to change
            mask = np.asarray(selection.mask, dtype=bool) & np.all(np.isfinite(xyz), axis=1)
            files = (np.asarray(table["filenumber"])[mask] if "filenumber" in table
                     else np.zeros(int(mask.sum()), dtype=np.int64))
            names.append(name)
            colours.append(colour)
            columns.append((xyz[mask], files))
            if shape is None:
                shape = self._drawn_roi(ctx, selection)
        fields = []
        all_files = (np.unique(np.concatenate([f for _, f in columns]))
                     if columns else [])
        for f in all_files:
            points = [xyz[files == f] for xyz, files in columns]
            if not any(len(p) for p in points):
                continue
            both = np.concatenate(points)
            if shape is not None:
                window = window_of_region(shape)
            else:
                window = window_of_points(both[:, 0], both[:, 1], settings.window,
                                          settings.occupied_bin_nm)
            z_range = (float(both[:, 2].min()), float(both[:, 2].max())) if three else None
            fields.append(Field(window, points, z_range))
        return fields, names, colours, shape, [n for n, _, _ in found[n_targets:]]

    def _model(self, settings: SpinnaSettings, n_targets: int):
        from .. import spinna
        unc = spinna.per_target(settings.label_unc_nm, n_targets, "label uncertainty")
        if settings.fit == "efficiency":
            return spinna.Model(spinna.efficiency_structures(_distances(settings.distance_nm)[0]),
                                [1.0, 1.0], unc, settings.rotation)
        eff = spinna.per_target(settings.efficiency, n_targets, "labelling efficiency")
        return spinna.Model(_structures(settings), eff, unc, settings.rotation)

    # -- the cost

    def preflight(self, ctx: Context, settings: SpinnaSettings):
        if ctx.per_region:
            return None
        try:
            n_t = _n_targets(settings)
            fields, *_ = self._gather(ctx, settings, n_t)
            model = self._model(settings, n_t)
            seconds = self.expected_seconds(model, fields, settings)
        except Exception:                                     # noqa: BLE001
            return None             # the run says what is wrong
        if seconds < ASK_ABOVE_S:
            return None
        took = f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} minutes"
        choices = []
        if settings.bootstrap:
            choices.append(PreflightChoice(
                label="without the bootstrap",
                settings=replace(settings, bootstrap=False),
                help=f"no errors, about "
                     f"{self.expected_seconds(model, fields, replace(settings, bootstrap=False)):.0f} s"))
        if settings.simulations > 3:
            fewer = replace(settings, simulations=3)
            choices.append(PreflightChoice(
                label="3 simulations",
                settings=fewer,
                help=f"noisier scores, about {self.expected_seconds(model, fields, fewer):.0f} s"))
        if settings.granularity > 11:
            coarser = replace(settings, granularity=11)
            choices.append(PreflightChoice(
                label="11 grid steps",
                settings=coarser,
                help=f"steps twice as coarse, about "
                     f"{self.expected_seconds(model, fields, coarser):.0f} s"))
        n = sum(f.n for f in fields)
        return PreflightQuestion(
            text=f"{n:,} molecules, {settings.simulations} simulations per candidate"
                 + (f", {settings.bootstraps} bootstrap rounds" if settings.bootstrap else "")
                 + f": about {took}.", choices=choices)

    @staticmethod
    def expected_seconds(model, fields, settings: SpinnaSettings) -> float:
        """The candidates the fit will score times one simulation of the
        middle candidate, timed."""
        from .. import spinna
        totals = spinna.totals_of(model, fields)
        space = spinna.search_space(model.structures, totals, settings.granularity)
        props = spinna.proportions(space, model.structures)
        boots = settings.bootstraps if settings.bootstrap else 0
        n = spinna.plan(len(space), props, settings.search, boots)
        if settings.fit == "efficiency":
            n *= len(_distances(settings.distance_nm))
        middle = space[int(np.argmin(np.abs(props - props.mean(axis=0)).sum(axis=1)))]
        rng = np.random.default_rng(0)
        orders = model.orders()
        # the best of three: the first call pays for imports and caches
        one = np.inf
        for _ in range(3):
            started = time.perf_counter()
            spinna.simulated_distances(model, fields, middle, orders, 1, rng)
            one = min(one, time.perf_counter() - started)
        return n * settings.simulations * one

    # -- the run

    def run(self, ctx: Context, settings: SpinnaSettings) -> Result:
        from .. import spinna
        n_t = _n_targets(settings)
        model = self._model(settings, n_t)
        fields, names, colours, shape, unused = self._gather(ctx, settings, n_t)
        if not fields or not any(f.n for f in fields):
            if ctx.per_region:
                return Result(values={}, settings=settings)
            raise ValueError("no molecules to fit")
        started = time.monotonic()

        def progress(fraction: float) -> None:
            if ctx.stopped:
                raise Stopped("SPINNA stopped: nothing fitted")
            if not ctx.per_region:
                ctx.report(f"SPINNA: {100 * fraction:.0f}%")

        boots = settings.bootstraps if settings.bootstrap else 0
        if settings.fit == "efficiency":
            found = spinna.fit_efficiency(
                fields, _distances(settings.distance_nm), model.label_unc_nm,
                settings.granularity, settings.search, settings.simulations, boots,
                settings.seed, settings.rotation, progress)
            model, best = found.model, found.fit
        else:
            found = None
            best = spinna.fit(model, fields, settings.granularity, settings.search,
                              settings.simulations, boots, settings.seed, progress)
        took = time.monotonic() - started
        orders = best.orders
        data = spinna.data_distances(fields, orders)
        rng = np.random.default_rng(settings.seed + 7919)
        simulated = spinna.simulated_distances(model, fields, best.counts, orders,
                                               settings.simulations, rng)
        values = self._values(best, found, fields, n_t)
        text = self._text(settings, best, found, fields, names, shape, unused, n_t, took)
        letters = [spinna.letter(t) for t in range(n_t)]
        label_names = [f"{letters[t]} ({names[t]})" for t in range(n_t)]

        def histograms(where, orders=orders):
            # one pair of targets is one axis (`Plot.draw_into`), more a row
            axes = ([where] if len(orders) == 1 else
                    list(np.atleast_1d(where.subplots(1, len(orders)))))
            upto = max((float(np.percentile(d[np.isfinite(d)], 99))
                        for d in data if np.isfinite(d).any()), default=200.0)
            for ax, order, d, s in zip(axes, orders, data, simulated):
                spinna.draw_histograms(ax, d, s, spinna.order_name(order, label_names), upto=upto)
        first = Plot(histograms, name="nearest neighbours", panels=len(orders),
                     size=(4.4 * len(orders), 3.4))

        def scores(ax):
            spinna.draw_scores(ax, best)
            ax.set_title("the candidates tried", fontsize=9)

        from .spatial_statistics import draw_window
        shown = fields[0]

        def window_plot(ax):
            draw_window(ax, shown.window, [(p[:, :2], c) for p, c in zip(shown.points, colours)])
            ax.set_title("the window" + (" (first file)" if len(fields) > 1 else ""),
                         fontsize=9)
        plots = {"scores": Plot(scores, name="scores", size=(4.6, 3.4)),
                 "window": Plot(window_plot, name="window", size=(4.2, 4.0))}
        return Result(text=text, plot=first, plots=plots, values=values, settings=settings,
                      data={"fit": best, "efficiency": found, "data": data,
                            "simulated": simulated, "orders": orders,
                            "area_nm2": float(sum(f.window.area for f in fields))},
                      title="SPINNA")

    @staticmethod
    def _values(best, found, fields, n_t) -> Dict[str, float]:
        from ..spinna import letter
        values: Dict[str, float] = {}
        area_um2 = sum(f.window.area for f in fields) / 1e6
        for t in range(n_t):
            n = sum(len(f.points[t]) for f in fields)
            values[f"n_{letter(t)}"] = float(n)
            values[f"density_{letter(t)}_um2"] = float(n / area_um2) if area_um2 > 0 else float("nan")
        if found is not None:
            for t in range(2):
                values[f"efficiency_{letter(t)}"] = float(found.efficiency[t])
                if found.errors is not None:
                    values[f"efficiency_err_{letter(t)}"] = float(found.errors[t])
            values["distance_nm"] = float(found.distance_nm)
        else:
            for j, name in enumerate(best.names):
                key = value_key(name)
                values[f"{key}_pct"] = float(best.proportions[j])
                if best.errors is not None:
                    values[f"{key}_err_pct"] = float(best.errors[j])
        values["ks"] = float(best.score)
        if best.score_error is not None:
            values["ks_err"] = float(best.score_error)
        return values

    @staticmethod
    def _text(settings, best, found, fields, names, shape, unused, n_t, took) -> str:
        from ..spinna import letter, ordinal
        area = sum(f.window.area for f in fields)
        where = (f"in the {shape}" if shape is not None else
                 f"in the {'occupied area' if settings.window == 'occupied' else 'bounding box'}")
        seen = []
        for t in range(n_t):
            n = sum(len(f.points[t]) for f in fields)
            seen.append(f"{n:,} of {letter(t)} ({names[t]}, {n / max(area, 1e-9) * 1e6:.3g}/µm²)")
        lines = [f"{', '.join(seen)} {where}, {area / 1e6:.4g} µm² over {len(fields)} file"
                 + ("s" if len(fields) != 1 else "")]
        if unused:
            lines.append("not used (the model has fewer targets): " + ", ".join(unused))
        neighbours = "; ".join(
            f"{letter(a)} to {letter(b)} 1st" + (f" to {ordinal(k)}" if k > 1 else "")
            for a, b, k in best.orders)
        n_coarse = int((best.passes == 0).sum())
        tried = (f"{n_coarse} coarse + {len(best.passes) - n_coarse} fine"
                 if n_coarse else f"{len(best.passes)}")
        lines.append(f"nearest neighbours {neighbours}; {tried} of {len(best.space)} "
                     f"candidates, {settings.simulations} simulations each"
                     + (f"; bootstrap {settings.bootstraps}" if settings.bootstrap else "")
                     + f"; {took:.1f} s")
        if found is not None:
            lines.append(f"tag distance {found.distance_nm:g} nm"
                         + (f" (best of {', '.join(f'{d:g}' for d, _ in found.scores_by_distance)})"
                            if len(found.scores_by_distance) > 1 else ""))
            for t in range(2):
                err = (f" ± {100 * found.errors[t]:.1f}" if found.errors is not None else "")
                lines.append(f"  labelling efficiency of {letter(t)}: "
                             f"{100 * found.efficiency[t]:.1f}{err} %")
        width = max(len(n) for n in best.names)
        for j, name in enumerate(best.names):
            err = f" ± {best.errors[j]:.1f}" if best.errors is not None else ""
            lines.append(f"  {name:<{width}}  {best.proportions[j]:5.1f}{err} % of the "
                         f"molecules ({int(best.counts[j])} structures)")
        err = f" ± {best.score_error:.3f}" if best.score_error is not None else ""
        lines.append(f"KS statistic {best.score:.3f}{err}")
        steps = grid_steps(best.tested_proportions)
        if best.errors is not None and np.any(best.errors < steps / 2):
            lines.append(f"an error below half a grid step ({np.nanmin(steps):.2g} % here) "
                         "means the bootstrap seldom left the best candidate: more grid "
                         "steps resolve it")
        return "\n".join(lines)


def grid_steps(tested_props: np.ndarray) -> np.ndarray:
    """Per structure, the smallest change of its proportion between two
    candidates tried: what an error is measured against."""
    out = []
    for j in range(tested_props.shape[1]):
        v = np.unique(np.round(tested_props[:, j], 6))
        d = np.diff(v)
        out.append(d.min() if len(d) else np.nan)
    return np.array(out, dtype=float)


def _nth(t: int) -> str:
    from ..spinna import ordinal
    return ordinal(t + 1)
