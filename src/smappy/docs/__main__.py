"""Write every plugin's page as a static site: ``python -m smappy.docs -o DIR``.

The same pages the Help window shows, from the same `render`, with the maths
(SVG) and figures (PNG) written out beside them -- so the site needs no
MathJax, no network and no build tool, and says what the program says.

    python -m smappy.docs -o build/docs                   # every plugin
    python -m smappy.docs -o build/docs "Analysis/Drift/RCC"
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote

from . import STYLE, page_file, panels, render

WEB_STYLE = """
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       max-width: 860px; margin: 2em auto; padding: 0 16px; color: #1d1d1f;
       background: #fff; }
a { color: #0b62c4; }
table { border-collapse: collapse; }
pre { background: #f6f8fa; padding: 8px; overflow-x: auto; }
img { max-width: 100%; height: auto; }
nav { font-size: small; margin-bottom: 1.5em; }
ul.index li { margin: 2px 0; }
"""


def slug(path: str) -> str:
    """A file name for a plugin path, stable and readable."""
    return re.sub(r"[^A-Za-z0-9]+", "-", path).strip("-").lower()


def _page(title: str, body: str, home: bool = False) -> str:
    nav = "" if home else '<nav><a href="index.html">&larr; all plugins</a></nav>'
    return (f"<!doctype html><html><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(title)}</title>"
            f"<style>{STYLE}{WEB_STYLE}</style></head><body>{nav}{body}</body></html>")


def export(out: Path, paths: Optional[List[str]] = None,
           report=print) -> Dict[str, List[str]]:
    """Write the site to ``out``.  Returns the errors per plugin, if any."""
    from .. import plugins

    refs = plugins.refs()
    shown = panels()
    wanted = sorted(paths or list(refs) + list(shown))
    out.mkdir(parents=True, exist_ok=True)
    errors: Dict[str, List[str]] = {}

    def link(url: str) -> str:
        if url.startswith("plugin:"):
            return slug(unquote(url[len("plugin:"):])) + ".html"
        return url

    for path in wanted:
        try:
            cls = shown[path] if path in shown else plugins.get(path)
        except Exception as error:
            errors[path] = [f"could not load: {error}"]
            continue
        ref = refs.get(path)
        report(f"{path}")
        rendered = render(cls, link=link, origin=ref.origin if ref else None)
        name = slug(path)
        assets = out / name
        text = rendered.html
        if rendered.images:
            assets.mkdir(exist_ok=True)
            for file, png in rendered.images.items():
                (assets / file).write_bytes(png)
                text = text.replace(f'src="{file}"', f'src="{name}/{file}"')
        (out / f"{name}.html").write_text(_page(cls.name or path, text), encoding="utf-8")
        if rendered.errors:
            errors[path] = rendered.errors

    # the index: the menu tree, a written page marked as such
    lines = ["<h1>SMAPpy plugins</h1>",
             "<p>What each plugin does, how it works, and every setting.  "
             "The settings are read off the plugins themselves.</p>"]
    group = None
    for path in wanted:
        head = path.rpartition("/")[0]
        if head != group:
            if group is not None:
                lines.append("</ul>")
            lines.append(f"<h3>{html.escape(head.replace('/', ' › '))}</h3>"
                         '<ul class="index">')
            group = head
        ref = refs.get(path)
        panel = shown.get(path)
        written = panel is not None or page_file(path, ref.origin if ref else None) is not None
        text = panel.description if panel is not None else (ref.description if ref else "")
        description = html.escape(text) if text else ""
        lines.append(f'<li><a href="{slug(path)}.html">{html.escape(path.rsplit("/", 1)[-1])}'
                     f'</a>{"" if written else " <i>(settings only)</i>"}'
                     f'{" &ndash; " + description if description else ""}</li>')
    if group is not None:
        lines.append("</ul>")
    (out / "index.html").write_text(_page("SMAPpy plugins", "\n".join(lines), home=True),
                                    encoding="utf-8")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m smappy.docs",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", help="plugin paths; default: all")
    parser.add_argument("-o", "--out", default="build/docs", type=Path)
    args = parser.parse_args(argv)
    errors = export(args.out, args.paths or None)
    for path, messages in errors.items():
        for message in messages:
            print(f"{path}: {message}", file=sys.stderr)
    print(f"wrote {args.out / 'index.html'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
