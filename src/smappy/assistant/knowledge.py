"""What the user asked the assistant to remember: decisions, answers, values.

A folder of Markdown notes, one per thing remembered, in the configuration
directory (``<config dir>/assistant/knowledge``) -- plain files, so a user
reads, edits, deletes or shares them without the assistant, and a lab puts
its notes in a shared folder by pointing ``--knowledge`` at it.

Kept only on the user's word.  The tool that writes a note says so in its
description, and a client asks before calling a tool unless told not to; the
GUI's pane will ask itself.  A note the user did not agree to is worse than
none: the next session takes it for the lab's practice.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .index import words


def default_folder() -> Path:
    from .. import config
    return config.config_dir() / "assistant" / "knowledge"


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:60] or "note"


def remember(folder: Path, title: str, text: str,
             tags: Sequence[str] = ()) -> Path:
    """Write a note; a title already used gets a number, never overwritten."""
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y-%m-%d}-{_slug(title)}"
    path, n = folder / f"{stem}.md", 2
    while path.exists():
        path, n = folder / f"{stem}-{n}.md", n + 1
    import yaml
    head = yaml.safe_dump({"title": title, "date": f"{datetime.now():%Y-%m-%d %H:%M}",
                           "tags": list(tags)}, allow_unicode=True, sort_keys=False)
    path.write_text(f"---\n{head}---\n\n{text.strip()}\n", encoding="utf-8")
    return path


def read(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    meta: Dict[str, Any] = {}
    match = re.match(r"---\s*\n(.*?)\n---\s*\n", raw, flags=re.S)
    if match:
        import yaml
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        raw = raw[match.end():]
    return {"name": path.stem, "title": str(meta.get("title") or path.stem),
            "date": str(meta.get("date") or ""), "tags": list(meta.get("tags") or []),
            "text": raw.strip()}


def recall(folder: Path, query: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    """The notes that share words with ``query``, best first; all, newest
    first, without one."""
    notes = [read(p) for p in sorted(folder.glob("*.md"), reverse=True)] \
        if folder.is_dir() else []
    wanted = set(words(query))
    if not wanted:
        return notes[:limit]
    scored = []
    for i, note in enumerate(notes):
        score = (3 * len(wanted & set(words(note["title"] + " " + " ".join(note["tags"]))))
                 + len(wanted & set(words(note["text"]))))
        if score:
            scored.append((-score, i, note))
    return [note for _, _, note in sorted(scored)[:limit]]


def forget(folder: Path, name: str) -> Optional[Path]:
    """Delete the note called ``name`` (its file's stem); None if none is."""
    path = folder / f"{Path(name).stem}.md"
    if path.parent != folder or not path.is_file():
        return None
    path.unlink()
    return path


# ------------------------------------------------------------ missing tools
# What the assistant could not do directly, noted for SMAPpy's developers:
# one file, appended to, beside the knowledge folder rather than in it -- a
# lab's shared notes are its practice, this is a wish list for the program.
# `diagnostics.bug_report` takes it along under the same name.
MISSING_TOOLS = "missing-tools.md"


def missing_tools_file() -> Path:
    from .. import config
    return config.config_dir() / "assistant" / MISSING_TOOLS


def note_missing_tool(path: Path, asked: str, workaround: str, tool: str) -> Path:
    """Append one gap to ``path``: what was asked, the detour, the tool."""
    from .. import __version__
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8") as out:
        if new:
            out.write("# Tools the assistant was missing\n\n"
                      "Noted in conversations, with the user's agreement, for "
                      "SMAPpy's developers.\n")
        out.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} -- {tool.strip()}\n\n"
                  f"- asked: {asked.strip()}\n"
                  f"- instead: {workaround.strip()}\n"
                  f"- SMAPpy {__version__}\n")
    return path


# ------------------------------------------------------------- own code
# What the assistant computed with code of its own rather than SMAPpy's
# plugins.  Its own code is welcome -- exploration, a figure, glue between
# plugins' results, a method SMAPpy lacks -- but an analysis done half outside
# the plugins looked, in the GUI's history and in a chain, as if it were
# whole, and nobody learnt which plugin would have saved the detour.  So each
# such step is written down: in the session's history, where the user sees
# it beside the plugins' runs, and here, appended to without asking, because
# it is a record of what was done (as the history is), not a wish list
# (`note_missing_tool`, which asks first).  The script is copied beside it:
# the assistant's folder is scratch, and the record should outlive it.
OWN_CODE = "own-code.md"
SCRIPT_LIMIT = 1_000_000        # bytes of a script copied; a larger one is named only


def own_code_file() -> Path:
    from .. import config
    return config.config_dir() / "assistant" / OWN_CODE


def record_own_code(path: Path, what: str, why: str, used: Sequence[str] = (),
                    script: Optional[Path] = None) -> Dict[str, Any]:
    """Append one step done with the assistant's own code to ``path``, and
    copy ``script`` into ``<path's folder>/own-code/``.  Returns the log's
    path and the copy's (None where nothing was copied, with why)."""
    from .. import __version__
    now = datetime.now()
    copied: Optional[Path] = None
    note = ""
    if script is not None:
        script = Path(script).expanduser()
        if not script.is_file():
            note = f"not copied: {script} is not a file"
        elif script.stat().st_size > SCRIPT_LIMIT:
            note = f"not copied: larger than {SCRIPT_LIMIT // 1000} kB"
        else:
            folder = path.parent / "own-code"
            folder.mkdir(parents=True, exist_ok=True)
            copied = folder / f"{now:%Y%m%d-%H%M%S}-{script.name}"
            copied.write_bytes(script.read_bytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8") as out:
        if new:
            out.write("# What the assistant did with its own code\n\n"
                      "Steps of an analysis computed outside SMAPpy's plugins: "
                      "what each did, why no plugin did it, and the plugins it "
                      "built on.  A step that recurs is a plugin to write.\n")
        out.write(f"\n## {now:%Y-%m-%d %H:%M} -- {what.strip()}\n\n"
                  f"- why not a plugin: {why.strip()}\n"
                  f"- built on: {', '.join(used) if used else 'no plugin'}\n")
        if script is not None:
            out.write(f"- script: {script}"
                      + (f" (copy: {copied.name})" if copied else f" ({note})") + "\n")
        out.write(f"- SMAPpy {__version__}\n")
    return {"file": path, "copy": copied, "note": note}
