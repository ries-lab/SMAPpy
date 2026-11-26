"""*Compare groups*: what was measured per region, compared between groups
of regions -- files, conditions, ROI keywords -- as a table of medians and
a grid of plots, one panel per measure and a box (or violin) per group.

The MFX scripts' plot grids split traces by a condition -- a laser power, a
sample, an acquisition -- and draw the same measure for each side by side.
This is that, for any region table: of ROIs or of an id column such as
``tid``.  The groups are `smappy.region_groups`', which *Histogram* (on
region results) and *Curves* offer as well.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from . import Context, FormState, Plot, Result, param, register
from .roi import SiteAnalysisPlugin, auto_only, grid_shape, measure_title
# the grouping every summary of a region table shares, re-exported here where
# it was first written
from ..region_groups import (ATTR, MAX_GROUPS, attribute_columns,  # noqa: F401
                             attribute_labels, group_choices, group_columns,
                             group_labels, numeric_fields, ordered)


@dataclass
class CompareGroupsSettings:
    evaluation: str = param("", label="results from", choices=auto_only,
                            help="the measurement compared, on the ROIs or per region of "
                                 "an id attribute")
    group_by: str = param("file", label="group by",
                          help="file; a column of the results with few values (an ROI "
                               "keyword); or, for an id attribute's regions, an attribute "
                               "of their localizations with few values (a file condition, "
                               "thi)")
    fields: str = param("", label="measures",
                        help="comma separated; empty: every numeric column")
    style: str = param("box", label="draw as", choices=(("box", "boxes"),
                                                        ("violin", "violins")),
                       help="the distribution per group")


@register("Analysis/Regions/Summarize/Compare groups")
class CompareGroups(SiteAnalysisPlugin):
    """The measures of a region table compared between groups of regions --
    files, conditions, keywords: a table of medians and a box per group."""

    Settings = CompareGroupsSettings
    changes_locs = False
    version = "1"

    def describe(self, ctx, settings) -> FormState:
        state = super().describe(ctx, settings)
        try:
            rows, ev = self.rows(ctx, settings) if ctx is not None else ([], None)
        except Exception:                                  # noqa: BLE001
            return state
        if rows:
            # one group is no comparison: the shared list without its "-"
            state.choices["group_by"] = group_choices(ctx, rows, ev)[1:]
        return state

    def run(self, ctx: Context, settings: CompareGroupsSettings) -> Result:
        rows, ev = self.rows(ctx, settings)
        if not rows:
            raise ValueError("nothing measured in the regions yet: run a per-region "
                             "measurement first")
        groups = group_labels(ctx, rows, ev, settings.group_by)
        fields = [f.strip() for f in settings.fields.split(",") if f.strip()] or \
            numeric_fields(rows, exclude=(settings.group_by,))
        missing = [f for f in fields if f not in rows[0]]
        if missing:
            raise ValueError(f"no measure {', '.join(missing)}: the results have "
                             + ", ".join(sorted(rows[0])))
        names = ordered(groups)
        table = compare(rows, groups, fields, names)
        lines = [f"{len(rows)} regions in {len(names)} groups by {settings.group_by}: "
                 + ", ".join(f"{g} ({table['n'][g]})" for g in names)]
        width = max(len(f) for f in fields)
        lines.append(" " * width + "  " + "  ".join(f"{g:>10.10}" for g in names))
        for f in fields:
            lines.append(f"{f:<{width}}  " + "  ".join(
                f"{table['median'][f][g]:>10.4g}" for g in names))
        n_rows, across = grid_shape(len(fields))

        def draw(figure):
            axes = figure.subplots(n_rows, across, squeeze=False).ravel()
            for ax, f in zip(axes, fields):
                data = [table["values"][f][g] for g in names]
                ok = [len(d) > 0 for d in data]
                pos = [i for i, k in enumerate(ok) if k]
                if not pos:
                    ax.set_visible(False)
                    continue
                vals = [data[i] for i in pos]
                if settings.style == "violin" and all(len(v) > 1 for v in vals):
                    ax.violinplot(vals, positions=pos, showmedians=True)
                else:
                    ax.boxplot(vals, positions=pos, widths=0.6, showfliers=False)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels([str(g) for g in names], rotation=30, fontsize=7)
                ax.set_title(measure_title(f), fontsize=9)
                ax.tick_params(labelsize=7)
            for ax in axes[len(fields):]:
                ax.set_visible(False)

        return Result(text="\n".join(lines),
                      plot=Plot(draw, panels=len(fields), size=(3.2 * across, 2.8 * n_rows)),
                      data={"table": table, "groups": names}, settings=settings,
                      title="compare groups")


def compare(rows, groups, fields, names) -> Dict[str, Any]:
    """Per measure and group: the values, their median, mean and count."""
    g = np.asarray(groups, dtype=object)
    out = {"values": {}, "median": {}, "mean": {}, "n": {}}
    for name in names:
        out["n"][name] = int((g == name).sum())
    for f in fields:
        v = np.asarray([r.get(f, np.nan) for r in rows], dtype=float)
        out["values"][f], out["median"][f], out["mean"][f] = {}, {}, {}
        for name in names:
            x = v[(g == name) & np.isfinite(v)]
            out["values"][f][name] = x
            out["median"][f][name] = float(np.median(x)) if len(x) else np.nan
            out["mean"][f][name] = float(x.mean()) if len(x) else np.nan
    return out
