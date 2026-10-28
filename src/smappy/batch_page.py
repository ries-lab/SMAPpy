"""The batch's summary page: ``index.html`` in the output folder.

A batch of fifty cells writes fifty folders of figures and a CSV, which is
everything and nothing to look at.  The page is one row per file -- its
status, its figures as thumbnails linked to the full ones, its numbers -- and
above them the numbers chosen (`Output.trends`) plotted across the files in
the order they were taken, which is how a drift in the instrument shows: the
localization precision creeping up over a month of acquisitions, as Picasso
Server's history page shows it (Strauss 2022).

It is written again after every file (`batch.publish`), so a watched folder's
page is current, and it refreshes itself in the browser while the watch runs.
It is static HTML with relative links and no script, so the output folder can
be copied, zipped or put on a server as it is.  No PDF: a page of thumbnails
is what the browser is for, and `trends.png` is the figure to put in a
report.
"""
from __future__ import annotations

import fnmatch
import html
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

THUMB_WIDTH = 220              # pixels: four or five figures to a row on a laptop
MAX_TRENDS = 12                # chosen automatically; named ones are all drawn
REFRESH_SECONDS = 20           # while a watch runs


def write_page(job, report: Dict[str, Any]) -> Path:
    """Write ``index.html`` (and ``trends.png``) for the report as it stands."""
    from .batch import _read_json, summary_rows
    out = job.output_folder()
    rows = summary_rows(job, report)
    records = [_read_json(out / entry["name"] / "results.json") or {}
               for entry in report["files"]]
    when = [_when(r, entry) for r, entry in zip(records, report["files"])]
    columns = trend_columns(rows, job.output.trends)
    trends = out / "trends.png"
    if columns and draw_trends(rows, when, columns, trends):
        trend_html = (f'<p><a href="trends.png"><img class="trends" src="trends.png?'
                      f'{_stamp(trends)}" alt="the chosen numbers across the files"></a></p>')
    else:
        trends.unlink(missing_ok=True)
        trend_html = ("<p class=note>No numbers to plot across the files yet"
                      + (" (the trend columns match none)" if job.output.trends else "")
                      + ".</p>")
    body = [f"<h1>{html.escape(report_title(job))}</h1>",
            f"<p class=note>{_counts(report)} &middot; {html.escape(report['status'])}"
            f" &middot; started {html.escape(str(report.get('started', '')))}"
            + (f" &middot; finished {html.escape(report['finished'])}"
               if report.get("finished") else "")
            + ' &middot; <a href="summary.csv">summary.csv</a>'
              ' &middot; <a href="batch_report.json">batch_report.json</a></p>',
            "<h2>Across the files</h2>", trend_html,
            "<h2>Files</h2>", "<table><thead><tr><th>#</th><th>file</th><th>status</th>"
            "<th>taken</th><th>figures</th><th>numbers</th></tr></thead><tbody>"]
    for n, (entry, record, row, t) in enumerate(zip(report["files"], records, rows, when), 1):
        body.append(_file_row(n, out, entry, record, row, t))
    body.append("</tbody></table>")
    refresh = (f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">'
               if report.get("status") in ("watching", "running") else "")
    page = (f"<!doctype html><html><head><meta charset=utf-8>{refresh}"
            f"<meta name=viewport content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(report_title(job))}</title><style>{STYLE}</style>"
            f"</head><body>{''.join(body)}</body></html>")
    path = out / "index.html"
    temporary = path.with_suffix(".html.part")
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(path)
    return path


def report_title(job) -> str:
    if job.name:
        return job.name
    if job.origin is not None:
        return job.origin.name.split(".")[0]
    return "SMAPpy batch"


def trend_columns(rows: Sequence[Dict[str, Any]], wanted: Sequence[str]) -> List[str]:
    """The `summary.csv` columns to plot across the files.

    Named ones (patterns allowed: ``stats.*``) in the order named, if they
    hold numbers.  None named: every numeric column whose value differs
    between the files -- a constant is a setting echoed back, not a
    measurement -- and does not repeat one before it, up to `MAX_TRENDS`,
    with the time each file took last.
    """
    numeric = [c for c in _columns(rows) if c not in ("file", "status", "message")
               and len(_values(rows, c)) > 0]
    if wanted:
        out: List[str] = []
        for pattern in wanted:
            out += [c for c in numeric if fnmatch.fnmatchcase(c, pattern) and c not in out]
        return out
    varying, shapes = [], set()
    for c in numeric:
        values = tuple(_number(r.get(c)) for r in rows)
        # a column that repeats another (the same n under three names) once
        if len(set(_values(rows, c))) > 1 and values not in shapes:
            varying.append(c)
            shapes.add(values)
    first = [c for c in varying if c != "seconds"][:MAX_TRENDS - 1]
    return first + (["seconds"] if "seconds" in varying else [])


def draw_trends(rows, when, columns: Sequence[str], path: Path) -> bool:
    """One panel per column against the time each file was taken (else the
    input's modification time), the files in that order; a failed file has
    no point.  False if there is nothing to draw."""
    from matplotlib.figure import Figure
    times = list(when)
    order = sorted(range(len(rows)), key=lambda i: (times[i] is None, times[i] or datetime.min))
    use_time = all(times[i] is not None for i in order) and len({times[i] for i in order}) > 1
    n = len(columns)
    if not n or not rows:
        return False
    across = 2 if n > 1 else 1
    down = math.ceil(n / across)
    figure = Figure(figsize=(5.0 * across, 2.4 * down), layout="constrained")
    axes = figure.subplots(down, across, squeeze=False).ravel()
    for ax, column in zip(axes, columns):
        xs, ys = [], []
        for rank, i in enumerate(order):
            y = _number(rows[i].get(column))
            if y is None or rows[i].get("status") not in ("done", "unchanged"):
                continue
            xs.append(times[i] if use_time else rank + 1)
            ys.append(y)
        ax.plot(xs, ys, "o-", ms=4, lw=1)
        ax.set_title(column, fontsize=9)
        ax.tick_params(labelsize=8)
        if len(ys) > 2:
            median = sorted(ys)[len(ys) // 2]
            ax.axhline(median, color="0.6", lw=0.8, ls="--")
    for ax in axes[n:]:
        ax.set_visible(False)
    if use_time:
        for ax in axes[:n]:
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_horizontalalignment("right")
    else:
        from matplotlib.ticker import MaxNLocator
        for ax in axes[:n]:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for ax in axes[max(n - across, 0):n]:
            ax.set_xlabel("file, in order of acquisition", fontsize=8)
    figure.savefig(path, dpi=100)
    return True


# ------------------------------------------------------------------- a row

def _file_row(n: int, out: Path, entry, record, row, when) -> str:
    name = entry["name"]
    status = entry["status"]
    message = entry.get("message") or record.get("message") or ""
    link = _href(f"{name}/results.json")
    cells = [f"<td>{n}</td>",
             f'<td><a href="{link}" title="{html.escape(str(entry["input"]))}">'
             f"{html.escape(name)}</a></td>",
             f'<td class="status {html.escape(status)}" title="{html.escape(message)}">'
             f"{html.escape(status)}</td>",
             f"<td>{html.escape(when.isoformat(sep=' ') if when else '')}</td>",
             f"<td>{_thumbnails(out, name, record)}</td>",
             f"<td>{_numbers(row)}</td>"]
    return f"<tr>{''.join(cells)}</tr>"


def _thumbnails(out: Path, name: str, record) -> str:
    pieces = []
    for figure in (record.get("output") or {}).get("figures") or []:
        full = Path(figure)
        if not full.exists():
            continue
        relative = _href(full.relative_to(out).as_posix()) if out in full.parents \
            else full.as_uri()
        small = _thumbnail(out, name, full)
        title = html.escape(full.stem)
        if small is not None:
            pieces.append(f'<a href="{relative}"><img src="{_href(small)}" title="{title}" '
                          f'alt="{title}"></a>')
        elif full.suffix.lower() == ".svg":
            pieces.append(f'<a href="{relative}"><img src="{relative}" width={THUMB_WIDTH} '
                          f'title="{title}" alt="{title}"></a>')
        else:
            pieces.append(f'<a href="{relative}">{title}</a>')
    return " ".join(pieces)


def _thumbnail(out: Path, name: str, full: Path) -> Optional[str]:
    """A small copy of a PNG figure, made once and again when the figure is
    newer; its path relative to the page.  None for any other format."""
    if full.suffix.lower() != ".png":
        return None
    target = out / name / "thumbnails" / full.name
    try:
        if not target.exists() or target.stat().st_mtime < full.stat().st_mtime:
            from PIL import Image          # matplotlib's own dependency
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(full) as picture:
                # twice the width shown, for a sharp thumbnail on a retina screen
                picture.thumbnail((2 * THUMB_WIDTH, 4 * THUMB_WIDTH))
                picture.save(target)
        return target.relative_to(out).as_posix()
    except Exception:
        return None


def _numbers(row: Dict[str, Any]) -> str:
    items = [(k, v) for k, v in row.items()
             if k not in ("file", "status", "message") and v not in ("", None)]
    if not items:
        return ""
    lines = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(_format(v))}</td></tr>"
                    for k, v in items)
    table = f"<table class=numbers>{lines}</table>"
    if len(items) <= 10:
        return table
    # fifty numbers per row would make one file a page long
    return f"<details><summary>{len(items)} numbers</summary>{table}</details>"


# ------------------------------------------------------------------ helpers

def _when(record, entry) -> Optional[datetime]:
    for text in (record.get("acquired"), record.get("input_mtime")):
        if text:
            try:
                return datetime.fromisoformat(str(text))
            except ValueError:
                pass
    try:
        return datetime.fromtimestamp(Path(entry["input"]).stat().st_mtime)
    except OSError:
        return None


def _columns(rows) -> List[str]:
    out: List[str] = []
    for row in rows:
        out += [c for c in row if c not in out]
    return out


def _number(value) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _values(rows, column: str) -> List[float]:
    found = [_number(r.get(column)) for r in rows]
    return [v for v in found if v is not None]


def _format(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _counts(report) -> str:
    counts: Dict[str, int] = {}
    for entry in report["files"]:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    files = len(report["files"])
    return f"{files} file{'s' if files != 1 else ''}" + (
        ": " + ", ".join(f"{n} {s}" for s, n in counts.items()) if counts else "")


def _href(relative: str) -> str:
    return quote(relative)


def _stamp(path: Path) -> str:
    """A query that changes with the file, so a refreshed page shows the new
    plot rather than the browser's cached one."""
    try:
        return str(int(path.stat().st_mtime))
    except OSError:
        return "0"


STYLE = """
:root { --fg: #1d1d1f; --bg: #ffffff; --muted: #6e6e73; --line: #d2d2d7;
        --done: #2e7d32; --failed: #e0453a; --skipped: #c06000; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e8e8ed; --bg: #1c1c1e; --muted: #98989d; --line: #3a3a3c; }
  img { background: #ffffff; }
}
body { font: 14px/1.4 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       color: var(--fg); background: var(--bg); margin: 16px; }
h1 { font-size: 20px; margin-bottom: 4px; }
h2 { font-size: 16px; margin-top: 24px; }
a { color: inherit; }
.note { color: var(--muted); }
table { border-collapse: collapse; }
thead th { text-align: left; border-bottom: 1px solid var(--line); padding: 4px 8px; }
tbody > tr > td { vertical-align: top; border-bottom: 1px solid var(--line); padding: 6px 8px; }
td img { width: %dpx; margin: 0 4px 4px 0; border: 1px solid var(--line); }
img.trends { max-width: 100%%; }
.status.done, .status.unchanged { color: var(--done); }
.status.failed, .status.aborted { color: var(--failed); }
.status.skipped { color: var(--skipped); }
table.numbers th, table.numbers td { font-size: 12px; padding: 0 8px 0 0; border: none;
                                     text-align: left; font-weight: normal; }
table.numbers th { color: var(--muted); }
""" % THUMB_WIDTH
