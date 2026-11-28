"""The plugin index: which page to read for a question.

Generated from what is already written -- the plugins' paths, names and
summaries, every section of their pages and every setting's label, help and
choices -- so it cannot drift from the plugins.  It is a map, not a
reference: an assistant finds candidates here and reads their pages
(`read_doc`) before choosing.

**Every section, not the opening.**  The first index held each page's first
two paragraphs, and a plugin's modes are further down: Picasso's *Undrift
from picked* is the *markers* mode of Fiducials, said in its page's fourth
paragraph, and "undrift from picked" found nothing (2026-11).  The pages
are about 1.3 million characters in some 1200 sections; ranked by BM25 they
take a tenth of a second.

**BM25 per section, a plugin by its best section.**  A page that lacks one
of the question's words still ranks by the others, the rare words counting
most.  A plugin scores its best section and a share of its next best
(`_REST`), so the words of a question spread over two sections still find
it.  Its name, path and summary are a section that counts more
(`_HEADER`).

**Several phrasings, and the Glossary's other names.**  Words are what a
search cannot see past: "drift from selected origamis" and the *markers*
mode share only "drift".  The model asking knows the field's synonyms, so
`search` takes several queries and says how many found each plugin; and a
query that names a Glossary entry also searches its other names
(`synonyms`), at half weight.  Embeddings would close more of the gap and
were left out: a model of a hundred megabytes to ship and keep, against a
model at the other end of the call that rephrases for free.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple, Union

_WORD = re.compile(r"[a-z0-9]+")
# words that say nothing about which plugin is meant
_STOP = frozenset("""a an and are as at be by can do does for from how i in into is it
its my of on or the this to what when which with you your want find use""".split())
# suffixes taken off, longest first, where at least `_STEM` letters remain:
# "correction", "corrected" and "correcting" are one word to a question
_SUFFIXES = ("ations", "ation", "ings", "ing", "ions", "ion", "ers", "er", "ed",
             "es", "ly")
_STEM = 4
# BM25's constants, the usual ones
_K1, _B = 1.2, 0.75
_HEADER = 1.5          # a plugin's name, path and summary against a section
_REST = 0.3            # the share of its next best sections a plugin adds
_RESTN = 2             # ... how many
_SYNONYM = 0.5         # a Glossary name the query did not say itself
# a setting against a page's section: the fitters share dozens of settings,
# and at 1 their help texts put five fitters above the plugin a page is about
_SETTING = 0.7
_SNIPPET = 280         # characters of a matching sentence handed back


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


@dataclass(frozen=True)
class Section:
    path: str             # the entry it belongs to
    heading: str          # "What it does", "setting: time windows", "" for the header
    text: str             # plain text, for the snippet
    words: Tuple[str, ...]
    weight: float = 1.0


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


def stem(word: str) -> str:
    """A word without its ending, crudely and the same way for page and query.

    Not English morphology -- "cluster" and "clusters" both become "clust",
    which is all a match needs."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _STEM:
            return word[:-len(suffix)]
    # the plural: "drifts" finds "drift"; not "process" or "focus"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def words(text: str) -> List[str]:
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


# a word after one of these is what the user does not want: "drift
# correction without beads" counted "beads" for the bead plugins and put
# COMET, which needs none, fourth
_NOT = frozenset("without no not except excluding".split())
# ... until one of these ends the phrase, or `_NEGATED` words have gone by
_PHRASE_END = frozenset("and but with in on for from by using via".split())
_NEGATED = 2


def wanted_words(query: str) -> List[str]:
    """``query``'s words that should match, those negated left out."""
    kept: List[str] = []
    negated = 0
    for raw in _WORD.findall(query.lower()):
        if raw in _NOT:
            negated = _NEGATED
            continue
        if raw in _PHRASE_END:
            negated = 0
            continue
        found = words(raw)
        if negated and found:
            negated -= 1
            continue
        kept += found
    return kept


# ---------------------------------------------------------------- the pages
_HEADING = re.compile(r"^#{2,3}\s+(.*)$", re.M)
_FENCE = re.compile(r"^```.*?^```", re.S | re.M)


def page_sections(path: str, body: str) -> List[Section]:
    """A page's sections, each under its heading; code and figures left out."""
    body = _FENCE.sub("", body)
    out: List[Section] = []
    marks = list(_HEADING.finditer(body))
    starts = [(0, "")] + [(m.end(), m.group(1).strip()) for m in marks]
    ends = [m.start() for m in marks] + [len(body)]
    for (start, heading), end in zip(starts, ends):
        text = plain(body[start:end])
        if text:
            out.append(Section(path, heading or "What it does", text,
                               tuple(words(heading + " " + text))))
    return out


def _choice_words(choices) -> str:
    if not choices or callable(choices):
        return ""
    return ", ".join(str(c[1] if isinstance(c, (tuple, list)) else c) for c in choices)


def setting_sections(path: str) -> List[Section]:
    """A section per setting: its label, help and choices -- where a
    plugin's modes are named (*Segment regions*' *like examples*).  Imports
    the plugin; none where it cannot be imported, so the search still
    answers."""
    from .. import plugins
    from ..batch import _flat_specs
    try:
        fields = _flat_specs(plugins.get(path).specs())
    except Exception:
        return []
    out = []
    for name, spec in fields.items():
        info = spec.info
        if info.hidden:
            continue
        label = info.label or name.rsplit(".", 1)[-1].replace("_", " ")
        text = " ".join(p for p in (label + ":", info.help or "",
                                    _choice_words(info.choices)) if p)
        out.append(Section(path, f"setting: {label}", plain(text), tuple(words(text)),
                           _SETTING))
    return out


@lru_cache(maxsize=1)
def entries() -> Tuple[Entry, ...]:
    """Every plugin, chain and panel page, once per process.

    Reading the pages costs about a tenth of a second; the plugins are not
    imported (`plugins.refs` parses them)."""
    return tuple(entry for entry, _ in _pages())


@lru_cache(maxsize=1)
def _pages() -> Tuple[Tuple[Entry, str], ...]:
    from .. import docs, plugins
    out: List[Tuple[Entry, str]] = []
    for path, ref in plugins.refs().items():
        page = docs.read_page(docs.page_file(path, ref.origin))
        out.append((Entry(path, ref.name, ref.description, purpose(page.body), ref.kind),
                    page.body))
    for path, panel in docs.panels().items():
        page = docs.read_page(panel.file)
        out.append((Entry(path, panel.name, panel.description,
                          purpose(page.body) or plain(page.body[:600]), "panel"),
                    page.body))
    return tuple(out)


@lru_cache(maxsize=1)
def sections() -> Tuple[Section, ...]:
    """Every section of every page and every plugin's settings, once per
    process: about two seconds, most of it importing the plugins for their
    settings (no torch: the plugins import theirs inside functions)."""
    out: List[Section] = []
    for entry, body in _pages():
        header = f"{entry.name} {entry.path.replace('/', ' ')} {entry.summary}"
        out.append(Section(entry.path, "", entry.summary, tuple(words(header)), _HEADER))
        out += page_sections(entry.path, body)
        if entry.kind == "plugin":
            out += setting_sections(entry.path)
    return tuple(out)


@lru_cache(maxsize=1)
def _statistics() -> Tuple[Dict[str, float], float, Tuple[Counter, ...]]:
    found = sections()
    frequency: Counter = Counter()
    for s in found:
        frequency.update(set(s.words))
    n = len(found)
    idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in frequency.items()}
    mean = sum(len(s.words) for s in found) / max(n, 1)
    return idf, mean, tuple(Counter(s.words) for s in found)


# ------------------------------------------------------------ the Glossary
_TERM = re.compile(r"^\* \*\*(.+?)\*\*\s*(?:\(([^)]*)\))?", re.M)


@lru_cache(maxsize=1)
def synonyms() -> Tuple[Tuple[Tuple[str, ...], ...], ...]:
    """The Glossary's names for one thing, as groups of word tuples:
    ``* **fiducial** (also called a fiducial marker, or a bead)`` is one
    group of three.  A name inside the brackets that says where it is used
    ("in Picasso, a linked localization") keeps its last part."""
    from .. import docs
    panel = docs.panels().get("Panels/Glossary")
    if panel is None:
        return ()
    groups = []
    for match in _TERM.finditer(docs.read_page(panel.file).body):
        names = [match.group(1)]
        for part in re.split(r",|\bor\b", match.group(2) or ""):
            part = re.sub(r"^\s*(also called|also|in \w+)\s*", "", part.strip())
            part = re.sub(r"^(a|an|the)\s+", "", part.strip())
            if part:
                names.append(part)
        group = tuple(t for t in (tuple(words(n)) for n in names) if t)
        if len(set(group)) > 1:
            groups.append(group)
    return tuple(groups)


def _weights(query: str) -> Dict[str, float]:
    """The words to look for and how much each counts: the query's own at
    1, the other Glossary names of what it names at `_SYNONYM`."""
    said = wanted_words(query)
    weights = {w: 1.0 for w in said}
    present = set(said)
    for group in synonyms():
        if any(set(name) <= present for name in group):
            for name in group:
                for w in name:
                    weights.setdefault(w, _SYNONYM)
    return weights


def _score(query: str) -> Tuple[Dict[str, float], Dict[str, List[Tuple[float, int]]]]:
    """Each entry's score for ``query``, and its sections' (score, index)."""
    weights = _weights(query)
    if not weights:
        return {}, {}
    idf, mean, counts = _statistics()
    found = sections()
    per: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for i, (section, count) in enumerate(zip(found, counts)):
        norm = _K1 * (1 - _B + _B * len(section.words) / mean)
        score = 0.0
        for w, weight in weights.items():
            tf = count.get(w)
            if tf:
                score += weight * idf[w] * tf * (_K1 + 1) / (tf + norm)
        if score:
            per[section.path].append((score * section.weight, i))
    totals = {}
    for path, scored in per.items():
        scored.sort(reverse=True)
        totals[path] = scored[0][0] + _REST * sum(s for s, _ in scored[1:1 + _RESTN])
    return totals, per


def _sentence(text: str, wanted: Iterable[str]) -> str:
    """The sentence of ``text`` that holds most of the words wanted."""
    wanted = set(wanted)
    best, most = "", 0
    for sentence in re.split(r"(?<=[.;!?])\s+", text):
        n = len(wanted & set(words(sentence)))
        if n > most:
            best, most = sentence, n
    if len(best) > _SNIPPET:
        best = best[:_SNIPPET].rsplit(" ", 1)[0] + " ..."
    return best


def search(query: Union[str, Sequence[str]], limit: int = 8) -> List[Dict]:
    """The entries that answer ``query`` best, best first.

    ``query`` may be several phrasings of one question: each is ranked on
    its own, an entry's scores (each a share of that phrasing's best) are
    summed, and ``found_by`` says how many phrasings had it among their
    first ``limit``.  Each entry comes with the sentences that matched,
    by section (``matches``), so the one asking sees why without reading
    the page."""
    queries = [query] if isinstance(query, str) else [q for q in query if q.strip()]
    combined: Dict[str, float] = defaultdict(float)
    found_by: Counter = Counter()
    best_sections: Dict[str, List[Tuple[float, int, str]]] = defaultdict(list)
    for q in queries:
        totals, per = _score(q)
        if not totals:
            continue
        top = max(totals.values())
        for rank, (path, score) in enumerate(sorted(totals.items(), key=lambda t: -t[1])):
            combined[path] += score / top
            if rank < limit:
                found_by[path] += 1
            for s, i in per[path][:2]:
                best_sections[path].append((s / top, i, q))
    if not combined:
        return []
    by_path = {entry.path: entry for entry in entries()}
    found = sections()
    out = []
    for path in sorted(combined, key=lambda p: (-combined[p], p))[:limit]:
        entry = by_path[path]
        matches, seen = [], set()
        for _, i, q in sorted(best_sections[path], reverse=True):
            section = found[i]
            if i in seen or not section.heading:
                continue
            seen.add(i)
            text = _sentence(section.text, _weights(q))
            if text:
                matches.append({"section": section.heading, "text": text})
            if len(matches) == 2:
                break
        item = {"path": path, "name": entry.name, "kind": entry.kind,
                "summary": entry.summary, "matches": matches}
        if len(queries) > 1:
            item["found_by"] = found_by[path]
        out.append(item)
    return out


# --------------------------------------------------------------- the catalog
def _modes(path: str) -> str:
    """A plugin's main choice, the one that picks what it does -- *method*,
    *algorithm*, *source* -- as its options: where a plugin holds several
    methods, the catalog line names them."""
    from .. import plugins
    from ..batch import _flat_specs
    try:
        fields = _flat_specs(plugins.get(path).specs())
    except Exception:
        return ""
    for name, spec in fields.items():
        if "." in name or spec.info.advanced or spec.info.hidden:
            continue
        if name in _MODE_FIELDS:
            options = _choice_words(spec.info.choices)
            if options:
                label = spec.info.label or name.replace("_", " ")
                return f"{label}: " + options.replace(", ", " · ")
    return ""


# the settings whose choice is what a plugin does, not how much of it
_MODE_FIELDS = ("method", "algorithm", "approach", "source", "model", "kind",
                "format")


def catalog(prefix: str = "") -> str:
    """Every plugin and chain, one line each, by menu: its path (the name
    the GUI shows in brackets where it differs), its summary and, where it
    has one, the choice of what it does.  The overview a question is
    matched against before `search` is asked: 88 plugins in about 5000
    tokens.  The whole path on every line, as `run_plugin` takes it."""
    from .. import plugins
    lines = []
    for path, ref in sorted(plugins.refs(prefix).items()):
        leaf = path.rsplit("/", 1)[-1]
        line = path
        if ref.name and ref.name != leaf:
            line += f" [{ref.name}]"
        if ref.kind == "chain":
            line += " (chain)"
        line += f": {ref.description}"
        modes = _modes(path) if ref.kind == "plugin" else ""
        if modes:
            line += f" -- {modes}"
        lines.append(line)
    return "\n".join(lines)
