"""Set up the layers -- bounds and grouping per layer -- as a chain step.

The Render tab is where a person filters: a precision below 20 nm, a
relative log-likelihood above -2, one layer grouped and one not.  A chain has
no hands on the Render tab, and a batch has no Render tab at all, so this
step does the same work from settings: one block per layer, each with its
grouping and its bounds.  Without it a chain filters as a freshly opened file
does (`session.DEFAULT_BOUNDS`), which is the right default and the wrong
thing to leave implicit in a methods section -- the chain's log entry says
what the bounds were either way.

Fields are not known before a file is open, so the bounds are rows rather
than one settings field per column: ``{field, lo, hi, quantile, required}``.
A quantile row takes `lo` and `hi` as fractions and resolves them per file
(`filter.quantile_range`), because an absolute photon count means different
things in different files.  A row for a field the file does not have is
skipped and said so, unless it is `required`, which stops the file.

`remove` is SMAP's "save visible only", which there lived in the writer.
Here it is a step, so it is in the log: localizations that no layer keeps
are dropped from the table, and from the file that is saved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import Context, Plugin, Result, param, register

STARTS = (("defaults", "the default bounds, then these rows"),
          ("empty", "only these rows"),
          ("keep", "the layer's current bounds, then these rows"))


def default_layers() -> List[Dict[str, Any]]:
    """One layer as a freshly opened file has it."""
    from ..session import GROUPED_BY_DEFAULT
    return [{"grouped": GROUPED_BY_DEFAULT, "start": "defaults", "bounds": []}]


@dataclass
class LayersSettings:
    layers: list = param(default_factory=default_layers, label="layers", kind="layers",
                         help="one block per layer: its grouping, where its bounds "
                              "start from, and rows of field, lo, hi (either may be "
                              "empty); a quantile row takes lo and hi as fractions, "
                              "resolved per file; a required row stops a file that "
                              "lacks the field")
    remove: bool = param(False, label="remove filtered",
                         help="drop the localizations no layer keeps from the table, "
                              "and so from the saved file")


def _number(value) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def default_bounds(locs) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """What a new layer starts with on this table (`Layer.apply_defaults`)."""
    from ..session import DEFAULT_BOUNDS, DEFAULT_BOUNDS_2D
    bounds = dict(DEFAULT_BOUNDS)
    if "z_nm" not in locs:
        bounds.update(DEFAULT_BOUNDS_2D)
    return {f: b for f, b in bounds.items() if f in locs}


def resolve_layer(locs, layer: Dict[str, Any],
                  current: Optional[Dict[str, Tuple]] = None
                  ) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], List[str]]:
    """One layer's bounds on this table, and what was skipped: ``(bounds, notes)``.

    Raises for a `required` row whose field the table does not have, naming
    the fields it does have -- the message someone reading a batch report
    needs to fix the chain.
    """
    start = layer.get("start", "defaults")
    if start not in {s for s, _ in STARTS}:
        raise ValueError(f"start {start!r} is not one of "
                         f"{', '.join(s for s, _ in STARTS)}")
    bounds = (default_bounds(locs) if start == "defaults"
              else dict(current or {}) if start == "keep" else {})
    notes = []
    for row in layer.get("bounds") or []:
        name = str(row.get("field", "")).strip()
        if not name:
            continue
        if name not in locs:
            if row.get("required"):
                raise ValueError(f"the filter needs {name!r}, which this table does "
                                 f"not have; it has {', '.join(sorted(locs.keys()))}")
            notes.append(f"no {name!r}: that bound skipped")
            continue
        lo, hi = _number(row.get("lo")), _number(row.get("hi"))
        if row.get("quantile"):
            from ..filter import quantile_range
            q_lo, q_hi = quantile_range(locs, name, lo if lo is not None else 0.0,
                                        hi if hi is not None else 1.0)
            lo = q_lo if lo is not None else None
            hi = q_hi if hi is not None else None
        bounds[name] = (lo, hi)
    return bounds, notes


def kept_by_any(locs, bounds_per_layer: Sequence[Dict[str, Tuple]]) -> np.ndarray:
    """The localizations at least one layer's bounds keep."""
    from ..filter import LocFilter
    keep = np.zeros(len(locs), dtype=bool)
    for bounds in bounds_per_layer:
        keep |= LocFilter(locs, **{f: b for f, b in bounds.items() if f in locs}).mask
    return keep


def _describe(bounds: Dict[str, Tuple]) -> str:
    parts = []
    for name, (lo, hi) in bounds.items():
        if lo is not None and hi is not None:
            parts.append(f"{lo:.4g} <= {name} <= {hi:.4g}")
        elif lo is not None:
            parts.append(f"{name} >= {lo:.4g}")
        elif hi is not None:
            parts.append(f"{name} <= {hi:.4g}")
    return ", ".join(parts) or "no bounds"


@register("Chain/Layers")
class ChainLayers(Plugin):
    """The layers' bounds and grouping, as the Render tab would set them."""
    Settings = LayersSettings
    favorite = False
    version = "1"

    def run(self, ctx: Context, settings: LayersSettings) -> Result:
        locs = ctx.locs
        layers = list(settings.layers or [])
        if not layers:
            raise ValueError("no layers: give at least one")
        current: List[Dict] = []
        if ctx.session is not None:
            current = ctx.session.layer_configs()
        configs, lines, resolved = [], [], []
        for n, layer in enumerate(layers):
            had = ({f: tuple(b) for f, b in current[n]["bounds"].items()}
                   if n < len(current) else None)
            bounds, notes = resolve_layer(locs, layer, had)
            resolved.append(bounds)
            config: Dict[str, Any] = {"bounds": {f: [lo, hi] for f, (lo, hi)
                                                 in bounds.items()}}
            if layer.get("grouped") is not None:
                config["grouped"] = bool(layer["grouped"])
            configs.append(config)
            grouped = {True: ", grouped", False: ", ungrouped"}.get(config.get("grouped"), "")
            lines.append(f"layer {n + 1}{grouped}: {_describe(bounds)}"
                         + "".join(f"; {note}" for note in notes))
        out = None
        if settings.remove and len(locs):
            keep = kept_by_any(locs, resolved)
            out = locs[keep]
            lines.append(f"removed {int((~keep).sum())} of {len(locs)} localizations "
                         "that no layer keeps")
        return Result(locs=out, text="\n".join(lines), data={"layers": configs},
                      settings=settings)
