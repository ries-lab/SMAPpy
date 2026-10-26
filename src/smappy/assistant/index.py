"""The plugin index: which page to read for a question.

Generated from what is already written -- each plugin's path, name and
summary, and the opening of its page's *What it does*, where a page says what
the plugin is for and when not to use it -- so it cannot drift from the
plugins.  It is a map, not a reference: an assistant finds two or three
candidates here and reads their pages (`read_doc`) before choosing.

Searching is word overlap, weighted by where the word was found.  Sixty-odd
plugins and the words people use for them ("drift", "clusters", "precision")
do not need more; an assistant rephrases a query that finds nothing far more
cheaply than an embedding would be maintained.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

# how much a word counts by where it was found
_WEIGHTS = (("name", 4.0), ("path", 3.0), ("summary", 2.0), ("purpose", 1.0))
_WORD = re.compile(r"[a-z0-9]+")
# words that say nothing about which plugin is meant
_STOP = frozenset("""a an and are as at be by can do does for from how i in into is it
its my of on or the this to what when which with you your want find use""".split())


@dataclass(frozen=True)
class Entry:
    path: str
    name: str
    summary: str
    purpose: str          # the opening of *What it does*, plain text
    kind: str             # "plugin", "chain" or "panel"

    def as_dict(self) -> Dict[str, str]:
        return {"path": self.path, "name": self.name, "summary": self.summary,
                "purpose": self.purpose, "kind": self.kind}


def plain(markdown: str) -> str:
    """Markdown as one line of text: links to their words, no emphasis."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", markdown)
    text = re.sub(r"[*_`]", "", text)
    return " ".join(text.split())


def purpose(body: str, paragraphs: int = 2) -> str:
    """The first paragraphs of a page's *What it does*, as plain text.

    Two, because the first usually says what the problem is and the second
    what the plugin does about it; when to use it and when not often follows,
    and is what `read_doc` is for."""
    match = re.search(r"^## What it does\s*\n(.*?)(?=^## |\Z)", body, flags=re.S | re.M)
    if not match:
        return ""
    found = [p for p in re.split(r"\n\s*\n", match.group(1).strip())
             if p.strip() and not p.lstrip().startswith(("```", "|", "#"))]
    return plain(" ".join(found[:paragraphs]))


@lru_cache(maxsize=1)
def entries() -> Tuple[Entry, ...]:
    """Every plugin, chain and panel page, once per process.

    Reading the pages costs about a tenth of a second; the plugins are not
    imported (`plugins.refs` parses them)."""
    from .. import docs, plugins
    out: List[Entry] = []
    for path, ref in plugins.refs().items():
        page = docs.read_page(docs.page_file(path, ref.origin))
        out.append(Entry(path, ref.name, ref.description, purpose(page.body), ref.kind))
    for path, panel in docs.panels().items():
        page = docs.read_page(panel.file)
        out.append(Entry(path, panel.name, panel.description,
                         purpose(page.body) or plain(page.body[:600]), "panel"))
    return tuple(out)


def words(text: str) -> List[str]:
    found = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
    # "clusters" finds "cluster", "drifts" finds "drift": the plural is
    # the one difference between a question and a page that matters here
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            for w in found]


def search(query: str, limit: int = 8) -> List[Dict[str, str]]:
    """The entries that share most words with ``query``, best first."""
    wanted = set(words(query))
    if not wanted:
        return []
    scored = []
    for entry in entries():
        score = 0.0
        for attr, weight in _WEIGHTS:
            score += weight * len(wanted & set(words(getattr(entry, attr))))
        if score:
            scored.append((score, entry.path, entry))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [entry.as_dict() for _, _, entry in scored[:limit]]
