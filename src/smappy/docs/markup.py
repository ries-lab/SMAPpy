"""The Markdown a plugin's page is written in, turned into HTML.

Why not Qt's own `setMarkdown`: it drops the text around any inline HTML (an
image in the middle of a sentence loses the sentence), and a page of equations
is nothing but images in the middle of sentences.  Why not the ``markdown``
package: it is a dependency for a few hundred lines of pages, and the subset
the pages are written in is small enough to convert here -- with the benefit
that the same HTML goes to the Help window and to the exported site.

The subset:

* ``#`` to ``####`` headings, paragraphs, ``-`` / ``1.`` lists (nested by
  indentation), ``>`` notes, ``| a | b |`` tables, fenced code;
* inline ``**bold**``, ``*italic*``, ```code```, ``[text](url)``,
  ``![alt](src)``;
* ``$tex$`` inline and ``$$tex$$`` displayed maths, handed to a ``math``
  callback that returns the HTML for it (an image, in practice);
* a fenced block whose language is ``figure`` is handed, with the rest of its
  info line as the caption, to a ``figure`` callback;
* a line that starts with ``<`` is raw HTML and passes through, which is how
  the generated parameter table gets in;
* ``<!-- comments -->`` vanish.

Anything else is text.  It is deliberately forgiving: a page that renders
slightly wrong is better than a Help window that raises.
"""
from __future__ import annotations

import html
import re
from typing import Callable, List, Optional

MathHook = Callable[[str, bool], str]            # (tex, displayed) -> html
FigureHook = Callable[[str, str], str]           # (code, caption) -> html
LinkHook = Callable[[str], str]                  # url -> url


def _plain_math(tex: str, display: bool) -> str:
    """No renderer: show the TeX itself, which is still readable."""
    text = html.escape(tex)
    return (f'<p class="math"><code>{text}</code></p>' if display
            else f"<code>{text}</code>")


def _plain_figure(code: str, caption: str) -> str:
    return f'<p class="caption">[figure] {inline(caption)}</p>'


class Converter:
    """Markdown -> HTML, with the maths, figures and links left to callbacks."""

    def __init__(self, math: Optional[MathHook] = None,
                 figure: Optional[FigureHook] = None,
                 link: Optional[LinkHook] = None):
        self.math = math or _plain_math
        self.figure = figure or _plain_figure
        self.link = link or (lambda url: url)

    # ------------------------------------------------------------------ inline
    def inline(self, text: str) -> str:
        """One paragraph's worth of text, formatted."""
        stash: List[str] = []

        def keep(fragment: str) -> str:
            stash.append(fragment)
            return f"\x00{len(stash) - 1}\x00"

        # code and maths first: nothing inside them is markup
        text = re.sub(r"`([^`]+)`",
                      lambda m: keep(f"<code>{html.escape(m.group(1))}</code>"), text)
        text = re.sub(r"(?<![\\$])\$(?!\s)([^$]+?)(?<!\s)\$(?!\$)",
                      lambda m: keep(self.math(m.group(1), False)), text)
        text = html.escape(text, quote=False).replace("\\$", "$")
        text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)",
                      lambda m: keep(f'<img src="{html.escape(m.group(2))}" '
                                     f'alt="{m.group(1)}">'), text)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                      lambda m: keep(f'<a href="{html.escape(self.link(m.group(2)))}">'
                                     f'{m.group(1)}</a>'), text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
        text = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"<i>\1</i>", text)
        text = re.sub(r"(^|\s)--(?=\s)", r"\1&ndash;", text)
        while "\x00" in text:
            text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
        return text

    # ------------------------------------------------------------------ blocks
    def convert(self, source: str) -> str:
        source = re.sub(r"<!--.*?-->", "", source, flags=re.S)
        lines = source.expandtabs(4).splitlines()
        return "\n".join(self._blocks(lines))

    def _blocks(self, lines: List[str]) -> List[str]:
        out: List[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                i += 1
                continue
            if stripped.startswith("```"):
                info = stripped[3:].strip()
                body: List[str] = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    body.append(lines[i])
                    i += 1
                i += 1
                language, _, caption = info.partition(" ")
                code = "\n".join(body)
                if language == "figure":
                    out.append(self.figure(code, caption.strip()))
                else:
                    out.append(f"<pre>{html.escape(code)}</pre>")
                continue
            if stripped.startswith("$$"):
                tex = stripped[2:]
                if tex.rstrip().endswith("$$") and len(tex.strip()) > 2:
                    tex = tex.rstrip()[:-2]
                    i += 1
                else:
                    parts = [tex]
                    i += 1
                    while i < len(lines) and "$$" not in lines[i]:
                        parts.append(lines[i].strip())
                        i += 1
                    if i < len(lines):
                        parts.append(lines[i].strip().replace("$$", ""))
                        i += 1
                    tex = " ".join(p for p in parts if p)
                out.append(self.math(tex.strip(), True))
                continue
            heading = re.match(r"(#{1,4})\s+(.*)", stripped)
            if heading:
                level = len(heading.group(1))
                out.append(f"<h{level}>{self.inline(heading.group(2))}</h{level}>")
                i += 1
                continue
            if stripped.startswith("<"):
                block = []
                while i < len(lines) and lines[i].strip():
                    block.append(lines[i])
                    i += 1
                out.append("\n".join(block))
                continue
            if stripped.startswith("|") and i + 1 < len(lines) and \
                    re.match(r"\s*\|?\s*:?-{2,}", lines[i + 1]):
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append(lines[i])
                    i += 1
                out.append(self._table(rows))
                continue
            if stripped.startswith(">"):
                quoted = []
                while i < len(lines) and lines[i].strip().startswith(">"):
                    quoted.append(lines[i].strip()[1:].lstrip())
                    i += 1
                out.append('<table class="note" width="100%" cellpadding="8">'
                           '<tr><td>' + "\n".join(self._blocks(quoted)) +
                           "</td></tr></table>")
                continue
            if _item(line):
                i, fragment = self._list(lines, i)
                out.append(fragment)
                continue
            paragraph = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip() and not _starts_block(lines[i]):
                paragraph.append(lines[i].strip())
                i += 1
            out.append(f"<p>{self.inline(' '.join(paragraph))}</p>")
        return out

    def _table(self, rows: List[str]) -> str:
        def cells(row: str) -> List[str]:
            row = row.strip()
            if row.startswith("|"):
                row = row[1:]
            if row.endswith("|"):
                row = row[:-1]
            return [c.strip() for c in row.split("|")]

        head = cells(rows[0])
        out = ['<table class="grid" cellspacing="0" cellpadding="4">', "<tr>"]
        out += [f"<th align=\"left\">{self.inline(c)}</th>" for c in head]
        out.append("</tr>")
        for row in rows[2:]:
            out.append("<tr>" + "".join(f"<td>{self.inline(c)}</td>"
                                        for c in cells(row)) + "</tr>")
        out.append("</table>")
        return "\n".join(out)

    def _list(self, lines: List[str], i: int):
        indent = _indent(lines[i])
        ordered = bool(re.match(r"\s*\d+\.\s", lines[i]))
        tag = "ol" if ordered else "ul"
        items: List[List[str]] = []
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                # a blank line ends the list unless the next line carries on
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                if nxt.strip() and (_indent(nxt) > indent or
                                    (_item(nxt) and _indent(nxt) == indent)):
                    i += 1
                    continue
                break
            if _item(line) and _indent(line) == indent:
                items.append([_item(line)])
            elif _indent(line) > indent and items:
                items[-1].append(line[indent + 2:] if _item(line) else line.strip())
            else:
                break
            i += 1
        body = []
        for item in items:
            first, rest = item[0], item[1:]
            nested = [r for r in rest if _item(r)]
            text = [r for r in rest if not _item(r)]
            if nested:
                # continuation text before the sub-list, then the sub-list
                split = next(k for k, r in enumerate(rest) if _item(r))
                inner = self.inline(" ".join([first] + rest[:split]))
                inner += "".join(self._blocks(rest[split:]))
            else:
                inner = self.inline(" ".join([first] + text))
            body.append(f"<li>{inner}</li>")
        return i, f"<{tag}>" + "".join(body) + f"</{tag}>"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _item(line: str) -> Optional[str]:
    """The text of a list item, or None when the line is not one."""
    m = re.match(r"\s*(?:[-*]|\d+\.)\s+(.*)", line)
    return m.group(1) if m else None


def _starts_block(line: str) -> bool:
    s = line.strip()
    return (s.startswith(("#", "```", "$$", "|", ">")) or bool(_item(line)))


def to_html(source: str, **hooks) -> str:
    return Converter(**hooks).convert(source)


def inline(text: str, **hooks) -> str:
    return Converter(**hooks).inline(text)
