"""*Molecular mapping*: from clusters to molecules, RESI's centres or G5M.

The work is in :mod:`smappy.molecules` (its docstring has the methods, the
constants and where they depart from Picasso, whose `clusterer` and `g5m`
modules they port; jungmannlab/picasso, MIT).  The plan it is part of is
`docs/dna-paint.md`.

It reads the localizations, not the blinks a layer may show
(``grouping = "ungrouped"``): G5M models each localization's scatter about
its molecule, and the centres' precision is ``std / sqrt(n)`` over
localizations.  The clusters are an id column -- *Find clusters*'
``cluster_id`` on *auto* -- and a cluster is fitted with its localizations
the layer's filter and ROI keep.

**Output**, one of two (as *Average ROIs* has), because a result either
replaces the table or adds files, not both -- the session adds the files
first and the table replaced after them would lose their rows:

* *a file of molecules* (the default): per file of the session, a new file
  whose rows are the molecules, as localizations -- ``x_nm``, ``y_nm``,
  ``xy_err_nm`` the precision of the position, the frame its mean -- so
  that everything that reads localizations reads molecules: rendering,
  *Spatial statistics*, *SPINNA*, *Register layers* between imaging rounds
  (RESI), a filter on ``n_events`` or ``p_val``;
* *molecule_id on the localizations*: each localization's molecule, 0 for
  none, for *Binding kinetics* or *Id statistics* per molecule.

Both are seeded, so the two runs find the same molecules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .. import molecules as mol
from ..locs import Localizations
from . import (Context, FormState, Plot, Plugin, Result, Stopped, param, register,
               settings_values)

METHODS = (("centres", "centres (RESI)"), ("g5m", "G5M"))
OUTPUTS = (("file", "a file of molecules"),
           ("column", "molecule_id on the localizations"))
#: the clusters on *auto*: *Find clusters*'
AUTO_IDS = ("cluster_id",)
#: above this many seconds a G5M run asks first (`MolecularMapping.preflight`)
ASK_ABOVE_S = 20.0


@dataclass
class G5MSettings:
    min_locs: int = param(mol.MIN_LOCS, label="minimum size", min=2,
                          help="localizations a molecule has at least")
    sigma_min: float = param(mol.SIGMA_BOUNDS[0], label="width", min=0.05,
                             help="a molecule's width at least this many times "
                                  "its localizations' precision")
    sigma_max: float = param(mol.SIGMA_BOUNDS[1], label="to", min=0.05, beside=True,
                             help="and at most this many times")
    max_rounds: int = param(mol.MAX_ROUNDS_WITHOUT_BEST, label="rounds without "
                            "improvement", min=1, advanced=True,
                            help="molecules added one by one stop after this "
                                 "many without a better BIC")
    max_locs: Optional[int] = param(None, label="largest cluster", min=1, advanced=True,
                                    help="a cluster with more localizations is "
                                         "passed over (a bead); empty: none")
    filter: bool = param(True, label="Picasso's filter",
                         help="drop sticky molecules, bad fits and those with "
                              "few binding events")
    min_std_frame: float = param(mol.MIN_STD_FRAME_FRACTION, label="frame spread",
                                 min=0.0, max=1.0, advanced=True,
                                 help="a molecule's frames' standard deviation at "
                                      "least this fraction of the acquisition")
    min_p: float = param(mol.MIN_P_VALUE, label="p-value", min=0.0, max=1.0,
                         advanced=True, help="a molecule's p_val above this")
    min_events: int = param(mol.MIN_EVENTS, label="binding events", min=0,
                            advanced=True, help="a molecule's n_events above this")


@dataclass
class MolecularMappingSettings:
    method: str = param("centres", label="method", choices=METHODS,
                        help="centres: one molecule per cluster; G5M: the "
                             "molecules in each cluster, by a Gaussian mixture")
    id_column: str = param("", label="clusters",
                           choices=lambda: [("", "auto")] + [(c, c) for c in AUTO_IDS],
                           help="the id attribute naming the clusters; auto: "
                                "cluster_id")
    output: str = param("file", label="output", choices=OUTPUTS,
                        help="the molecules as a new file, or each "
                             "localization's molecule")
    use_z: bool = param(True, label="in 3D", help="x, y and z when the table has "
                                                  "z_nm")
    g5m: G5MSettings = param(default_factory=G5MSettings, label="G5M")


def cluster_column(locs, chosen: str = "") -> str:
    """The clusters' column: ``chosen``, or on *auto* the first of
    `AUTO_IDS` the table has.  ValueError naming the integer columns there
    are when there is none."""
    from ..tracking.columns import integer_columns
    column = (chosen or "").strip() or next((c for c in AUTO_IDS if c in locs), "")
    if column and column in locs:
        return column
    have = integer_columns(locs)
    raise ValueError((f"no attribute {chosen!r}" if chosen else
                      "no clusters: run Find clusters first, or choose the id "
                      "attribute") + "; the integer attributes are "
                     + (", ".join(have) if have else "none"))


@register("Analysis/DNA-PAINT/Molecular mapping")
class MolecularMapping(Plugin):
    """Turns clusters into molecules: one per cluster at its centre (RESI),
    or the molecules a Gaussian mixture finds in each (G5M)."""

    Settings = MolecularMappingSettings
    changes_locs = True
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx: Optional[Context], settings: MolecularMappingSettings
                 ) -> FormState:
        state = FormState(shown={"g5m": settings.method == "g5m"})
        if ctx is None or ctx.session is None or not len(ctx.locs):
            return state
        from ..tracking.columns import integer_columns
        locs = ctx.locs
        found = next((c for c in AUTO_IDS if c in locs), "")
        state.choices["id_column"] = ([("", f"auto ({found})" if found else "auto")]
                                      + [(c, c) for c in integer_columns(locs)])
        state.active["use_z"] = "z_nm" in locs
        g = settings.g5m
        for name in ("min_std_frame", "min_p", "min_events"):
            state.active[f"g5m.{name}"] = g.filter
        return state

    def preflight(self, ctx: Context, settings: MolecularMappingSettings):
        """A question before a G5M run that will be long.  Its cost grows
        with a cluster's localizations times the molecules tried in it,
        ``n^2 / min_locs``: about 1e-5 s per unit on 4 cores, so a 12-site
        origami of 600 localizations takes 0.3 s.  Centres ask nothing."""
        if settings.method != "g5m":
            return None
        try:
            column = cluster_column(ctx.locs, settings.id_column)
            labels = np.asarray(ctx.locs[column], np.int64)
            labels = labels[np.asarray(ctx.selection.mask, bool) & (labels > 0)]
            n = np.bincount(labels).astype(float)
        except Exception:                                  # noqa: BLE001
            return None             # the run says what is wrong
        if settings.g5m.max_locs:
            n = n[n <= settings.g5m.max_locs]
        seconds = float(1e-5 * (n ** 2).sum() / max(settings.g5m.min_locs, 1))
        if seconds < ASK_ABOVE_S:
            return None
        took = (f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} minutes"
                if seconds < 5400 else f"{seconds / 3600:.1f} hours")
        return (f"G5M in {int((n > 0).sum()):,} clusters will take about {took}.  "
                "Clusters of one site each (Find clusters' local maxima) are "
                "fastest; a largest cluster leaves out beads.  Run anyway?")

    def run(self, ctx: Context, settings: MolecularMappingSettings) -> Result:
        locs = ctx.locs
        column = cluster_column(locs, settings.id_column)
        for c in ("x_nm", "y_nm", "frame"):
            if c not in locs:
                raise ValueError(f"no attribute {c}: the table has "
                                 + ", ".join(sorted(locs.keys())))
        use_z = settings.use_z and "z_nm" in locs
        g5m = settings.method == "g5m"
        if g5m and "xy_err_nm" not in locs:
            raise ValueError("G5M holds a molecule's width to the localization "
                             "precision, and the table has no xy_err_nm")
        if g5m and use_z and "z_err_nm" not in locs:
            raise ValueError("G5M in 3D needs z_err_nm: untick in 3D, or fit z "
                             "with a precision")
        axes = ["x_nm", "y_nm"] + (["z_nm"] if use_z else [])
        points = np.column_stack([np.asarray(locs[c], float) for c in axes])
        labels = np.asarray(locs[column], np.int64).copy()
        use = np.asarray(ctx.selection.mask, bool) & np.isfinite(points).all(1)
        labels[~use] = 0
        files = (np.asarray(locs["filenumber"], np.int64) if "filenumber" in locs
                 else np.zeros(len(locs), np.int64))
        frame = np.asarray(locs["frame"], np.int64)
        if not (labels > 0).any():
            raise ValueError(f"no clusters: no localization the layer shows has "
                             f"{column} above 0")
        spans = _spans(ctx, locs)

        def progress(fraction):
            if ctx.stopped:
                raise Stopped("Molecular mapping stopped: nothing written")
            ctx.report(f"molecular mapping: {100 * fraction:.0f}%")

        if g5m:
            values, molecule = self._g5m(locs, points, frame, labels, files, spans,
                                         settings.g5m, use_z, progress)
        else:
            values, molecule = _centres(points, frame, labels)
        n = len(values["label"])
        if not n:
            raise ValueError("no molecules: " + (
                "no cluster gave one G5M accepts" if g5m else "no clusters"))
        cluster_file = _first_of(labels, files)
        values["file"] = np.array([cluster_file[int(c)] for c in values["label"]],
                                  np.int64)
        channels = _first_of(labels, np.asarray(locs["channel"], np.int64)) \
            if "channel" in locs else None
        values["channel"] = (np.array([channels[int(c)] for c in values["label"]],
                                      np.int64) if channels is not None else None)
        n_clusters = int(len(np.unique(labels[labels > 0])))
        text = (f"{n} molecules in {n_clusters} clusters of {column} "
                f"({'G5M' if g5m else 'centres'})"
                + (f"; {values.pop('filtered')} dropped by Picasso's filter"
                   if "filtered" in values else ""))
        result = Result(text=text, settings=settings,
                        values={"molecules": n, "clusters": n_clusters},
                        data={"molecules": values, "column": column},
                        title=f"{n} molecules")
        per_cluster = np.bincount(np.unique(values["label"], return_inverse=True)[1])
        xy = points[:, :2]
        result.plot = Plot(lambda fig: draw_molecules(fig, xy, molecule,
                                                      values["centre"], per_cluster),
                           panels=2, size=(8, 3.6))
        if settings.output == "column":
            result.locs = with_molecules(locs, molecule)
            result.text += "; molecule_id written"
        else:
            if ctx.session is not None and not ctx.session.files:
                raise ValueError("the table is in no file to add the molecules "
                                 "beside: choose molecule_id on the localizations")
            result.files = list(_molecule_files(ctx, locs, values, axes, column,
                                                settings, spans))
            result.append = True
        return result

    def _g5m(self, locs, points, frame, labels, files, spans, s: G5MSettings,
             use_z: bool, progress):
        lp = np.asarray(locs["xy_err_nm"], float)
        lp = (np.column_stack([lp, lp, np.asarray(locs["z_err_nm"], float)])
              if use_z else lp)
        # a localization without a precision cannot bound a width: left out
        finite = np.isfinite(lp).all(1) if lp.ndim == 2 else np.isfinite(lp)
        labels = np.where(finite, labels, 0)
        if not (labels > 0).any():
            raise ValueError("G5M holds a molecule's width to the localization "
                             "precision, and no localization of the clusters has "
                             "one (xy_err_nm" + (", z_err_nm" if use_z else "") + ")")
        values, molecule = mol.g5m_clusters(
            points, lp, frame, labels, s.min_locs, (s.sigma_min, s.sigma_max),
            s.max_rounds, max_locs=s.max_locs, progress=progress)
        if s.filter and len(values["label"]):
            cluster_file = _first_of(labels, files)
            frames = np.array([spans[cluster_file[int(c)]][1]
                               - spans[cluster_file[int(c)]][0] + 1
                               for c in values["label"]], float)
            kept = mol.keep(values, frames, s.min_std_frame, s.min_p, s.min_events)
            values = {k: v[kept] for k, v in values.items()}
            molecule = mol.renumber(molecule, kept)
            values["filtered"] = int((~kept).sum())
        return values, molecule


def _centres(points, frame, labels):
    """RESI's centres in the shape G5M's values have."""
    c = mol.centres(points, frame, labels)
    values = {"centre": c["centre"], "sem": c["sem"], "sigma": c["std"],
              "n_locs": c["n_locs"], "n_events": c["n_events"], "frame": c["frame"],
              "std_frame": c["std_frame"], "label": c["label"]}
    rank = np.zeros(int(labels.max(initial=0)) + 1, np.int64)
    rank[c["label"]] = np.arange(1, len(c["label"]) + 1)
    return values, np.where(labels > 0, rank[np.maximum(labels, 0)], 0)


def _first_of(labels: np.ndarray, column: np.ndarray) -> Dict[int, int]:
    """Each cluster's value of ``column`` at its first localization."""
    use = np.flatnonzero(labels > 0)
    ids, first = np.unique(labels[use], return_index=True)
    return {int(i): int(column[use[f]]) for i, f in zip(ids, first)}


def _spans(ctx: Context, locs) -> Dict[int, tuple]:
    from .clusters import _spans as spans
    return spans(ctx, locs)


def with_molecules(locs: Localizations, molecule: np.ndarray) -> Localizations:
    """The table with ``molecule_id``, a recipe that keeps the largest value
    on the grouped table, as *Find clusters*' ``cluster_id``."""
    from ..mathparse import remember
    out = Localizations({**locs.columns, "molecule_id": np.asarray(molecule, np.int64)},
                        dict(locs.metadata))
    remember(out, "molecule_id", "", grouped="max")
    return out


def molecule_table(values: Dict[str, Any], rows: np.ndarray, axes: List[str],
                   cluster_column: str) -> Dict[str, np.ndarray]:
    """The columns of a table of molecules, ``rows`` of ``values``."""
    centre, sem = values["centre"][rows], values["sem"][rows]
    out = {"frame": np.round(values["frame"][rows]).astype(np.int64),
           "x_nm": centre[:, 0].astype(np.float32),
           "y_nm": centre[:, 1].astype(np.float32),
           # the precision of the position in the plane: the mean of x and y's
           "xy_err_nm": (0.5 * (sem[:, 0] + sem[:, 1])).astype(np.float32),
           "sigma_nm": (0.5 * (values["sigma"][rows][:, 0]
                               + values["sigma"][rows][:, 1])).astype(np.float32),
           "n_locs": np.asarray(values["n_locs"][rows], np.int64),
           "n_events": np.asarray(values["n_events"][rows], np.int64),
           "mean_frame": values["frame"][rows].astype(np.float32),
           "std_frame": values["std_frame"][rows].astype(np.float32),
           cluster_column: np.asarray(values["label"][rows], np.int64),
           "molecule_id": (np.flatnonzero(rows) + 1 if rows.dtype == bool
                           else np.asarray(rows) + 1).astype(np.int64)}
    if "z_nm" in axes:
        out["z_nm"] = centre[:, 2].astype(np.float32)
        out["z_err_nm"] = sem[:, 2].astype(np.float32)
        out["sigma_z_nm"] = values["sigma"][rows][:, 2].astype(np.float32)
    if "p_val" in values:
        out["p_val"] = values["p_val"][rows].astype(np.float32)
        out["sigma_rel"] = values["sigma_rel"][rows][:, 0].astype(np.float32)
    if values.get("channel") is not None:
        out["channel"] = np.asarray(values["channel"][rows], np.int64)
    return out


def _molecule_files(ctx: Context, locs, values, axes, column, settings, spans):
    """One file of molecules per file of the session that has any."""
    from ..io.formats import FileInfo
    from ..metadata import file_metadata
    names = ctx.session.file_names() if ctx.session is not None else []
    for f in np.unique(values["file"]):
        rows = values["file"] == f
        columns = molecule_table(values, rows, axes, column)
        source = names[int(f)] if int(f) < len(names) else f"file {int(f)}"
        lo, hi = spans.get(int(f), (0, int(columns["frame"].max(initial=0))))
        meta = {"units": "nm", "n_frames": int(hi - lo + 1),
                "molecular_mapping": {"method": settings.method, "source": source,
                                      "clusters": column,
                                      "settings": settings_values(settings)}}
        camera = file_metadata(locs.metadata, int(f)).get("camera")
        if camera:
            meta["camera"] = camera
        table = Localizations(columns, meta)
        name = f"molecules of {source}"
        yield table, FileInfo(name=name, path=name, format="molecules",
                              n=len(table)), None


def draw_molecules(figure, xy: np.ndarray, molecule: np.ndarray, centre: np.ndarray,
                   per_cluster: np.ndarray) -> None:
    """The localizations coloured by molecule with the molecules on top; and
    how many molecules the clusters hold."""
    left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.3, 1]})
    n = len(xy)
    pick = (np.random.default_rng(0).choice(n, 50_000, replace=False)
            if n > 50_000 else np.arange(n))
    pick = pick[molecule[pick] > 0]
    left.scatter(xy[pick, 0], xy[pick, 1], s=1, c=molecule[pick] % 20, cmap="tab20",
                 linewidths=0)
    left.scatter(centre[:, 0], centre[:, 1], s=6, c="k", marker="+", linewidths=0.6)
    left.set_aspect("equal")
    left.set_xlabel("x (nm)")
    left.set_ylabel("y (nm)")
    left.set_title(f"{len(centre)} molecules", fontsize=9)
    if len(per_cluster):
        top = int(per_cluster.max())
        right.hist(per_cluster, bins=np.arange(0.5, top + 1.5), color="0.5")
    right.set_xlabel("molecules per cluster")
    right.set_ylabel("clusters")
