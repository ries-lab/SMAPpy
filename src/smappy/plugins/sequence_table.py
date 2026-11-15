"""*Sequence table*: the MINFLUX sequence of every open measurement, a
table per measurement and thread -- one line per iteration, the parameters
abberior's wiki explains ("MINFLUX Sequence") -- with, first, the
parameters that differ between the measurements.

The sequences are those the reader kept with each file
(``FileInfo.metadata["minflux"]["measurements"][i]["sequence"]``, and
``"sequences"`` for a measurement of several threads).  Nothing is
computed: it says what the instrument was told to do, side by side, and
hands the rows back for a script to write as csv.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import Context, Plugin, Result, param, register

#: the iteration parameters shown, in this order, where a sequence has them
KEYS = ("Mode.pattern", "Mode.id", "patGeoFactor", "patDwellTime", "patRepeat",
        "phtLimit", "ccrLimit", "bgcThreshold", "pwrFactor", "maxOffTime",
        "_excitation.laser", "_excitation.power", "_excitation.wavelength")


#: internals left out of the differences: the estimator's residuals, PSF
#: files, the DAC setting a power maps to, what the live view shows
INTERNALS = ("_lemd", "_lemn", "psf", "dac", "liveview", "dpsf", "epsf", "estCoeff")


def _same(v) -> str:
    """A value as compared: floats to 6 significant digits, so a rounding
    difference (0.4750000000000001) is no difference."""
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_same(x) for x in v) + "]"
    return repr(v)


@dataclass
class SequenceTableSettings:
    keys: str = param("", label="parameters",
                      help="iteration parameters, comma separated; empty: pattern, "
                           "patGeoFactor, dwell, photon limit, cfr limit, background, power, "
                           "laser")


def measured_sequences(files) -> List[Dict[str, Any]]:
    """One entry per measurement and thread of the files: ``file``,
    ``measurement``, ``thread``, ``sequence``."""
    out = []
    for f in files:
        metas = ((f.metadata or {}).get("minflux") or {}).get("measurements") or []
        for m in metas:
            seqs = m.get("sequences") or ([m["sequence"]] if m.get("sequence") else [])
            for thread, seq in enumerate(seqs):
                out.append({"file": f.name, "measurement": m.get("label", ""),
                            "thread": thread, "sequence": seq})
    return out


def rows_of(entries, keys) -> List[Dict[str, Any]]:
    """A row per iteration of every entry: its file, measurement, thread,
    iteration and the ``keys``."""
    rows = []
    for e in entries:
        for k, it in enumerate(e["sequence"].get("iterations") or []):
            rows.append({"file": e["file"], "measurement": e["measurement"],
                         "thread": e["thread"], "iteration": k,
                         **{key: it.get(key) for key in keys}})
    return rows


def differing(entries) -> Dict[str, List[Any]]:
    """The global parameters, and the iteration parameters, whose values are
    not the same in every entry: ``{name: [value per entry]}``."""
    if len(entries) < 2:
        return {}
    out: Dict[str, List[Any]] = {}
    names = set()
    for e in entries:
        names |= {f"global {k}" for k in (e["sequence"].get("globals") or {})}
        for k, it in enumerate(e["sequence"].get("iterations") or []):
            names |= {f"itr {k} {p}" for p in it}

    def value(e, name):
        if name.startswith("global "):
            return (e["sequence"].get("globals") or {}).get(name[7:])
        _, k, p = name.split(" ", 2)
        its = e["sequence"].get("iterations") or []
        return its[int(k)].get(p) if int(k) < len(its) else None

    for name in sorted(names):
        if any(part in name for part in INTERNALS):
            continue
        vals = [value(e, name) for e in entries]
        if len({_same(v) for v in vals}) > 1:
            out[name] = vals
    return out


@register("Analysis/MINFLUX/Sequence table")
class SequenceTable(Plugin):
    """The MINFLUX sequence of every open measurement, an iteration a line,
    per thread; the parameters that differ between them first."""

    Settings = SequenceTableSettings
    changes_locs = False
    version = "1"

    def run(self, ctx: Context, settings: SequenceTableSettings) -> Result:
        from ..io.minflux import iteration_table
        files = ctx.session.files if ctx.session is not None else []
        entries = measured_sequences(files)
        if not entries:
            raise ValueError("no MINFLUX sequence: open an Abberior .msr file (its exports "
                             "carry no sequence)")
        keys = [k.strip() for k in settings.keys.split(",") if k.strip()] or list(KEYS)
        lines = []
        diff = differing(entries)
        if diff:
            lines.append(f"differ between the {len(entries)} sequences:")
            for name, vals in list(diff.items())[:60]:
                lines.append(f"  {name}: " + " | ".join(str(v) for v in vals))
            if len(diff) > 60:
                lines.append(f"  ... and {len(diff) - 60} more")
            lines.append("")
        for e in entries:
            title = f"{e['file']}" + (f", {e['measurement']}" if e["measurement"] else "") + \
                (f", thread {e['thread']}" if len(entries) > 1 or e["thread"] else "")
            lines += [title, iteration_table(e["sequence"], keys), ""]
        return Result(text="\n".join(lines).rstrip(),
                      data={"rows": rows_of(entries, keys), "differing": diff},
                      settings=settings, title="sequence table")
