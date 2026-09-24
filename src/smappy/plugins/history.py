"""What has been done to these localizations, and with which numbers.

Every run is logged -- `Session.apply` records the plugin, the line it
reported and the settings it actually used -- and the log is written into the
localization file and read back when the file is reopened.  It is the answer
to the question that comes up months later in front of a figure: *is this the
drift-corrected table, and what was the segmentation set to?*

This shows it.  Nothing here changes anything; it reads `session.history` and
prints it, and writes it to a file when asked.  Two things it is deliberately
not: it is not the undo (that goes one step, through the session), and it is
not the saved *result* of a tool (`Plugin.keep`, which is a figure and its
numbers).  This is the plain record of what happened, in order.

The export exists because the record is most useful outside the program -- in
a methods section, a lab notebook, an issue about a file that looks wrong.  A
``.csv`` opens in anything, and a ``.yaml`` keeps the settings as the nested
values they are rather than flattening them into a cell.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import Context, Plugin, Result, param, register


def entries(session, changes_only: bool = False) -> List[Dict]:
    """The log, oldest first.  ``changes_only`` keeps what moved the numbers.

    A session is not required to have one: a script builds a `Context` around
    a table and there is nothing to read, which is an empty log and not an
    error.
    """
    found = list(getattr(session, "history", None) or []) if session else []
    if changes_only:
        found = [e for e in found if e.get("changed")]
    return found


def _settings_text(settings) -> str:
    """The settings on one line, in the order the plugin declared them.

    Nested parts (`fit.roisize`) are flattened with dots, because the point of
    the line is to be read and searched, not to be parsed back.
    """
    if not isinstance(settings, dict):
        return "" if settings is None else str(settings)
    flat = []
    for key, value in settings.items():
        if isinstance(value, dict):
            flat += [f"{key}.{k} = {v}" for k, v in value.items()]
        else:
            flat.append(f"{key} = {value}")
    return ", ".join(flat)


def as_text(found: Sequence[Dict], settings: bool = True) -> str:
    """The log as it is read on screen: one run per block, in order."""
    if not found:
        return "nothing has been done to this table in this session, and the "\
               "file carries no record of anything done to it before"
    lines = []
    for entry in found:
        mark = "*" if entry.get("changed") else " "
        lines.append(f"{mark} {entry.get('time', '')}  {entry.get('what', '')}")
        if entry.get("text"):
            for line in str(entry["text"]).splitlines():
                lines.append(f"      {line}")
        steps = entry.get("steps")
        if isinstance(steps, list):
            # a chain: its steps are the record, each with its own settings,
            # rather than the chain's settings as one very long line
            for step in steps:
                if not isinstance(step, dict):
                    continue
                mark = "*" if step.get("changed") else " "
                first = str(step.get("text") or "").splitlines()[:1]
                lines.append(f"    {mark} {step.get('label', '')} ({step.get('plugin', '')})"
                             + (f": {first[0]}" if first else ""))
                if settings:
                    written = _settings_text(step.get("values"))
                    if written:
                        lines.append(f"          [{written}]")
        elif settings:
            written = _settings_text(entry.get("settings"))
            if written:
                lines.append(f"      [{written}]")
    lines.append("")
    lines.append(f"{sum(1 for e in found if e.get('changed'))} of {len(found)} "
                 "entries changed the localizations (*)")
    return "\n".join(lines)


def as_csv(found: Sequence[Dict]) -> str:
    """One row per entry, the settings as one JSON cell.

    JSON rather than a column per setting: the settings of a fit and of a
    drift correction have nothing in common, and a table with sixty mostly
    empty columns is not readable by a person or by a script.
    """
    import csv
    import io
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["time", "what", "changed", "text", "settings"])
    for entry in found:
        writer.writerow([entry.get("time", ""), entry.get("what", ""),
                         int(bool(entry.get("changed"))),
                         str(entry.get("text", "")).replace("\n", " "),
                         json.dumps(entry.get("settings"), default=str)])
    return out.getvalue()


def write(found: Sequence[Dict], path) -> Path:
    """Write the log; the extension chooses the form.

    ``.csv`` for a spreadsheet, ``.yaml`` or ``.json`` to keep the settings as
    values, anything else as the text that is on screen.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        text = as_csv(found)
    elif suffix in (".yaml", ".yml"):
        import yaml
        text = yaml.safe_dump([dict(e) for e in found], sort_keys=False,
                              default_flow_style=False)
    elif suffix == ".json":
        text = json.dumps([dict(e) for e in found], indent=1, default=str)
    else:
        text = as_text(found)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@dataclass
class HistorySettings:
    changes_only: bool = param(
        False, label="only what changed the localizations",
        help="leave out the measurements and the loads, keeping the runs that "
             "the numbers in this table depend on")
    show_settings: bool = param(True, label="with the settings",
                                help="the parameters each run was given")
    path: str = param("", label="export to", kind="save_file",
                      file_filter="CSV (*.csv);;YAML (*.yaml);;JSON (*.json);;"
                                  "Text (*.txt)",
                      help="optional: also write the log to a file, in the "
                           "form the extension asks for")


@register("Analysis/Process/History")
class History(Plugin):
    """What has been done to these localizations, with the settings used."""

    Settings = HistorySettings
    text_window = True        # the log is the result, and it is long

    def run(self, ctx: Context, settings: HistorySettings) -> Result:
        found = entries(ctx.session, settings.changes_only)
        text = as_text(found, settings.show_settings)
        written: Optional[Path] = None
        if settings.path:
            written = write(found, settings.path)
            text = f"{text}\n\nwritten to {written}"
        # `data` and not `locs`: nothing here changes the table, and the log
        # is worth handing to a script that wants to check what a file has
        # been through before it trusts it
        return Result(text=text, settings=settings,
                      data={"history": found,
                            **({"path": written} if written else {})})
