"""A log of how SMAPpy is used, sent to its developers on the user's word.

What it is for: learning, from the people who use SMAPpy, what is hard to
find, what is missing and what the help leaves unanswered.  An AI assistant
working in SMAPpy shows this most clearly -- it searches for a plugin and
finds none, reads the source because the page did not say, or writes numpy
of its own for what a plugin does -- and so do the help pages a person opens.
Collected over weeks from a few colleagues and read by a model, it says which
features to build and which to make easier to find.

Off unless the user turns it on (Preferences > Usage log), and nothing leaves
the computer unless they also give a link to send it to.  When on:

* **events** -- every tool call an assistant makes through the MCP server
  (`assistant.protocol`): the tool, its arguments, whether it failed, how
  long it took and the start of its answer; and each help page opened.
  Appended as JSON lines to ``<config dir>/usage-log/events-YYYY-MM.jsonl``.
* gathered when a bundle is made, not recorded twice: the plugin runs of
  ``smappy.log`` (`diagnostics`) with the program's console output and its
  uncaught errors, which that log copies, the native crashes of
  ``crash.log``, the assistant's notes of its own code and
  of the tools it missed (`assistant.knowledge`, with the copied scripts),
  and the assistant client's own transcripts of the conversations held in
  its folder -- Claude Code's under ``~/.claude/projects``, Codex's under
  ``~/.codex/sessions`` -- and Claude Code's conversations in any other
  folder that called SMAPpy's tools (`claude_smappy_transcripts`), which hold
  what was asked and what was tried, and which SMAPpy cannot see any other
  way.

**Sent as a zip to a cloud folder's upload-only link** -- a Nextcloud "file
drop", which the University of Vienna's u:cloud is -- by one WebDAV ``PUT``
(`upload`).  Such a link lets anyone holding it add files and nobody read
them, so it can be handed to a lab without exposing what others sent, and it
needs no account or token on the sender's side.  The link is kept in the
configuration until the user clears it.  Only what is new since the last
send goes (`bundle`'s ``since``): the logs' lines by their time stamps, the
other files by when they last changed.

**What is left out of a bundle**: pictures (a tool's figure, a screenshot in a
transcript) and any text longer than `TEXT_LIMIT`, cut with a note -- the
localizations and images are not what is studied, and they would make a week's
bundle hundreds of megabytes.  File names and folders stay in: what kind of
data someone worked on is part of what is learnt.

Qt-free, as `diagnostics` is; the window is `gui/preferences.UsageLogPage`.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import uuid
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

KEY = "usage_log"                 # its section of config.yaml
TEXT_LIMIT = 8000                 # characters of one string kept in a bundle
ANSWER_LIMIT = 600                # characters of a tool's answer kept in an event
ARGUMENT_LIMIT = 2000             # characters of a tool's arguments kept in an event
AUTOMATIC_DAYS = 7                # how often an automatic send goes
SCRIPT_LIMIT = 1_000_000          # bytes of a copied script taken along
TIMEOUT = 120                     # seconds an upload may take
DEFAULTS = {"enabled": False, "link": "", "password": "", "name": "",
            "automatic": False, "last_sent": "", "last_error": "", "id": ""}

_lock = threading.Lock()


# ----------------------------------------------------------------- settings
def settings() -> Dict[str, Any]:
    from . import config
    saved = config.get(KEY) or {}
    return {**DEFAULTS, **(saved if isinstance(saved, dict) else {})}


def save_settings(**values) -> Dict[str, Any]:
    """Change some of the settings; the rest, the link above all, stay."""
    from . import config
    merged = {**settings(), **values}
    config.set(KEY, merged)
    return merged


def enabled() -> bool:
    try:
        return bool(settings()["enabled"])
    except Exception:          # a broken config must never stop a tool call
        return False


def installation_id() -> str:
    """A random name for this installation, made once: bundles from one
    computer are told apart without the user giving a name."""
    ident = settings()["id"]
    if not ident:
        ident = uuid.uuid4().hex[:8]
        save_settings(id=ident)
    return ident


def folder() -> Path:
    from . import config
    return config.config_dir() / "usage-log"


# ------------------------------------------------------------------- events
def _short(value: Any, limit: int) -> Any:
    """``value`` as JSON can hold it, no string longer than ``limit``."""
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        text = repr(value)
    if len(text) <= limit:
        try:
            return json.loads(text)
        except Exception:
            return text
    return text[:limit] + f" ... [{len(text) - limit} more characters]"


def event(kind: str, **fields) -> None:
    """Append one event, when the log is on.  Never raises: a log that broke
    a tool call or the help window would cost more than it tells."""
    if not enabled():
        return
    try:
        record = {"t": datetime.now().isoformat(timespec="seconds"),
                  "kind": kind, "pid": os.getpid(), **fields}
        line = json.dumps(record, default=str, ensure_ascii=False)
        target = folder() / f"events-{datetime.now():%Y-%m}.jsonl"
        with _lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as out:
                out.write(line + "\n")
    except Exception as error:      # noqa: BLE001 -- see the docstring
        print(f"usage log: could not write an event: {error}")


def answer_summary(value: Any) -> Any:
    """The start of a tool's answer, as an event keeps it: a picture is named,
    not kept."""
    if isinstance(getattr(value, "png", None), bytes):
        return f"<picture: {getattr(value, 'caption', '')}>"
    return _short(value, ANSWER_LIMIT)


def tool_call(tool: str, arguments: Dict[str, Any], seconds: float,
              answer: Any = None, error: str = "", client: str = "") -> None:
    event("tool", tool=tool, arguments=_short(arguments, ARGUMENT_LIMIT),
          seconds=round(seconds, 3), ok=not error, error=error[:ANSWER_LIMIT],
          answer=None if error else answer_summary(answer), client=client)


# ------------------------------------------------- what a bundle gathers
def _stamp(text: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def _kind(line: str) -> str:
    try:
        return str(json.loads(line).get("kind", ""))
    except Exception:       # noqa: BLE001 -- a broken line is just not a kind
        return ""


def _events_since(since: Optional[datetime]) -> List[str]:
    lines: List[str] = []
    for path in sorted(folder().glob("events-*.jsonl")):
        if since and datetime.fromtimestamp(path.stat().st_mtime) <= since:
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                when = _stamp(json.loads(line).get("t", ""))
            except Exception:
                continue
            if since is None or (when is not None and when > since):
                lines.append(line)
    return lines


_LOG_LINE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def _log_since(path: Path, since: Optional[datetime]) -> str:
    """The lines of a `diagnostics` log written after ``since``; a line with
    no time of its own (a traceback's) goes with the line it continues."""
    out: List[str] = []
    keep = since is None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _LOG_LINE.match(line)
        if match:
            when = _stamp(match.group(1))
            keep = since is None or (when is not None and when > since)
        if keep:
            out.append(line)
    return "\n".join(out) + ("\n" if out else "")


_CRASH_HEAD = re.compile(r"^=== .* started (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def _crashes_since(path: Path, since: Optional[datetime]) -> str:
    """The sections of `diagnostics`' ``crash.log`` -- one per start, headed
    by when it started -- that hold a crash and may be new: those started
    after ``since``, and the last, which a process still running or killed
    since may have written into.  A start that never crashed is its header
    alone, and is left out."""
    sections: List[List[str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if _CRASH_HEAD.match(line) or not sections:
            sections.append([])
        sections[-1].append(line)
    out = []
    for i, section in enumerate(sections):
        match = _CRASH_HEAD.match(section[0])
        started = _stamp(match.group(1).replace(" ", "T")) if match else None
        recent = since is None or started is None or started > since \
            or (i == len(sections) - 1 and _newer(path, since))
        if recent and any(line.strip() for line in section[1:]):
            out += section
    return "\n".join(out) + ("\n" if out else "")


def _newer(path: Path, since: Optional[datetime]) -> bool:
    try:
        return since is None or datetime.fromtimestamp(path.stat().st_mtime) > since
    except OSError:
        return False


def assistant_folder() -> Path:
    """Where the assistant works: the folder the settings name, else
    `launch.default_folder` -- the folder its client keys its transcripts by."""
    from . import config
    chosen = (config.get("assistant") or {}).get("folder") if isinstance(
        config.get("assistant"), dict) else ""
    if chosen:
        return Path(chosen).expanduser()
    from .assistant.launch import default_folder
    return default_folder()


def _claude_home() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude").expanduser()


def claude_transcripts(work: Path) -> List[Path]:
    """Claude Code's conversations held in ``work`` (the terminal's and the
    Claude app's Code tab alike): ``~/.claude/projects/<folder>``, the folder
    named by the path with every character but letters and digits a dash."""
    wanted = str(work.expanduser().resolve())
    projects = _claude_home() / "projects"
    project = projects / re.sub(r"[^A-Za-z0-9]", "-", wanted)
    if project.is_dir():
        return sorted(project.glob("*.jsonl"))
    # the naming is Claude Code's, not a promise: if it changed, the folder
    # is found by the ``cwd`` its conversations record
    for candidate in sorted(projects.glob("*")) if projects.is_dir() else []:
        first = next(iter(sorted(candidate.glob("*.jsonl"))), None)
        if first is not None and _records_cwd(first, wanted):
            return sorted(candidate.glob("*.jsonl"))
    return []


SMAPPY_CALL = re.compile(r'"name"\s*:\s*"mcp__smappy')


def calls_smappy(path: Path) -> bool:
    """Whether a Claude Code conversation called one of SMAPpy's tools
    (``mcp__smappy...``) -- a tool call of the assistant's, not a line that
    merely names one: a developer's session on SMAPpy's own code talks about
    those tools all day and is none of the log's business."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not SMAPPY_CALL.search(line):       # cheap: most lines are not
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if record.get("type") != "assistant":
                    continue
                content = (record.get("message") or {}).get("content")
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "tool_use" \
                            and str(block.get("name", "")).startswith("mcp__smappy"):
                        return True
    except OSError:
        pass
    return False


def claude_smappy_transcripts(since: Optional[datetime] = None) -> List[Path]:
    """Claude Code's conversations held anywhere else that called SMAPpy's
    tools: an assistant started in a data folder rather than through *Start
    assistant* left its tool calls in the events and its notes, but no
    conversation (the first logs, 2026-11).  Only those that called SMAPpy:
    the rest of what someone does with Claude Code is not SMAPpy's."""
    projects = _claude_home() / "projects"
    if not projects.is_dir():
        return []
    return [path for path in sorted(projects.glob("*/*.jsonl"))
            if _newer(path, since) and calls_smappy(path)]


def _records_cwd(path: Path, wanted: str, lines: int = 20) -> bool:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for _, line in zip(range(lines), handle):
                try:
                    if json.loads(line).get("cwd") == wanted:
                        return True
                except Exception:
                    continue
    except OSError:
        pass
    return False


def codex_transcripts(work: Path, since: Optional[datetime] = None) -> List[Path]:
    """Codex's conversations held in ``work``: each session file under
    ``~/.codex/sessions`` names its folder in its first line."""
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
    root = home / "sessions"
    if not root.is_dir():
        return []
    wanted = str(work.expanduser().resolve())
    out = []
    for path in sorted(root.rglob("*.jsonl")):
        if not _newer(path, since):
            continue
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                first = json.loads(handle.readline() or "{}")
        except Exception:
            continue
        payload = first.get("payload") if isinstance(first.get("payload"), dict) else first
        if str(payload.get("cwd") or "") == wanted:
            out.append(path)
    return out


def _scrub(value: Any) -> Any:
    """A transcript's record without pictures or very long texts."""
    if isinstance(value, dict):
        if value.get("type") in ("image", "input_image"):
            return {"type": value.get("type"), "removed": "picture"}
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        if value.startswith("data:image") or (len(value) > 2000 and _looks_base64(value)):
            return "<picture removed>"
        if len(value) > TEXT_LIMIT:
            return value[:TEXT_LIMIT] + f" ... [{len(value) - TEXT_LIMIT} more characters]"
    return value


def _looks_base64(text: str) -> bool:
    sample = text[:400]
    return " " not in sample and re.fullmatch(r"[A-Za-z0-9+/=\s]+", sample) is not None


def scrub_transcript(path: Path) -> str:
    """A JSON-lines transcript with pictures and long texts cut (`_scrub`);
    a line that is not JSON is kept as it is, cut to `TEXT_LIMIT`."""
    out = io.StringIO()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                out.write(json.dumps(_scrub(json.loads(line)), ensure_ascii=False))
            except Exception:
                out.write(line[:TEXT_LIMIT])
            out.write("\n")
    return out.getvalue()


# ------------------------------------------------------------------ bundle
def bundle(path: Path, since: Optional[datetime] = None) -> Dict[str, Any]:
    """Write a zip of everything new since ``since`` (all of it for None);
    returns what went in, by part, and the file.  Empty parts are left out,
    and ``count`` is 0 when nothing at all was new."""
    from . import __version__, diagnostics
    from .config import config_dir
    path = Path(path)
    parts: Dict[str, int] = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        events = _events_since(since)
        if events:
            z.writestr("events.jsonl", "\n".join(events) + "\n")
            parts["events"] = len(events)
        logs = diagnostics.log_dir()
        for log in sorted(logs.glob("smappy*.log*")):
            if not _newer(log, since):
                continue
            text = _log_since(log, since)
            if text:
                z.writestr(f"logs/{log.name}", text)
                parts["logs"] = parts.get("logs", 0) + 1
        crash = logs / diagnostics.CRASH_NAME
        if crash.is_file() and _newer(crash, since):
            text = _crashes_since(crash, since)
            if text:
                z.writestr(f"logs/{crash.name}", text)
                parts["crashes"] = text.count("\n=== ") + text.startswith("=== ")
        assistant = config_dir() / "assistant"
        for name in ("missing-tools.md", "own-code.md"):
            if (assistant / name).is_file() and _newer(assistant / name, since):
                z.write(assistant / name, f"assistant/{name}")
                parts["assistant notes"] = parts.get("assistant notes", 0) + 1
        for script in sorted((assistant / "own-code").glob("*")) \
                if (assistant / "own-code").is_dir() else []:
            if script.is_file() and _newer(script, since) \
                    and script.stat().st_size <= SCRIPT_LIMIT:
                z.write(script, f"assistant/own-code/{script.name}")
                parts["scripts"] = parts.get("scripts", 0) + 1
        work = assistant_folder()
        claude = [t for t in claude_transcripts(work) if _newer(t, since)]
        claude += [t for t in claude_smappy_transcripts(since) if t not in claude]
        for transcript in claude:
            z.writestr(f"transcripts/claude/{transcript.name}",
                       scrub_transcript(transcript))
            parts["transcripts"] = parts.get("transcripts", 0) + 1
        for transcript in codex_transcripts(work, since):
            z.writestr(f"transcripts/codex/{transcript.name}", scrub_transcript(transcript))
            parts["transcripts"] = parts.get("transcripts", 0) + 1
        # a send's own record ("sent", written after the time it records) goes
        # with the next bundle but is no reason for one: a send that crossed a
        # second stamped it after `last_sent`, and the next send uploaded a
        # bundle of that record alone (a flaky test of the release, 2026-11-20)
        count = sum(parts.values()) - sum(1 for line in events if _kind(line) == "sent")
        z.writestr("about.json", json.dumps({
            "smappy": __version__, "installation": installation_id(),
            "name": settings()["name"], "made": datetime.now().isoformat(timespec="seconds"),
            "since": since.isoformat(timespec="seconds") if since else None,
            "assistant folder": str(work), "parts": parts,
            "environment": diagnostics.environment()}, indent=1, ensure_ascii=False))
    return {"file": path, "parts": parts, "count": count}


def bundle_name(when: Optional[datetime] = None) -> str:
    """``<name or installation>-<date time>.zip``: unique, and sorted by
    sender, then time, in the folder it lands in."""
    who = re.sub(r"[^A-Za-z0-9_-]+", "-", settings()["name"].strip()).strip("-") \
        or installation_id()
    return f"smappy-usage-{who}-{(when or datetime.now()):%Y%m%d-%H%M%S}.zip"


# ------------------------------------------------------------------ upload
def parse_link(link: str) -> Tuple[str, str]:
    """A Nextcloud share link's server and token:
    ``https://host[/sub]/index.php/s/TOKEN`` or ``.../s/TOKEN``."""
    link = link.strip()
    match = re.match(r"(https?://[^?#]+?)(?:/index\.php)?/s/([A-Za-z0-9]+)/?(?:[?#].*)?$",
                     link)
    if not match:
        raise ValueError("not a share link: expected something like "
                         "https://ucloud.univie.ac.at/index.php/s/AbC123xYz")
    return match.group(1), match.group(2)


def _basic(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def upload(file: Path, link: str, password: str = "", name: str = "",
           opener=None) -> str:
    """Put ``file`` into the folder a Nextcloud share link names; returns the
    address it went to.  Tries the WebDAV of public shares Nextcloud has had
    since version 29 first, then the older one, which ownCloud has too.
    Raises `OSError` with the server's answer when neither takes it."""
    import urllib.error
    import urllib.parse
    import urllib.request
    host, token = parse_link(link)
    target = urllib.parse.quote(name or Path(file).name)
    data = Path(file).read_bytes()
    tries = [(f"{host}/public.php/dav/files/{token}/{target}", _basic("anonymous", password)),
             (f"{host}/public.php/webdav/{target}", _basic(token, password))]
    opener = opener or urllib.request.urlopen
    problems = []
    for url, auth in tries:
        request = urllib.request.Request(url, data=data, method="PUT", headers={
            "Authorization": auth, "Content-Type": "application/zip",
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": "smappy-usage-log"})
        try:
            with opener(request, timeout=TIMEOUT) as response:
                if 200 <= getattr(response, "status", 200) < 300:
                    return url
                problems.append(f"{url}: HTTP {response.status}")
        except urllib.error.HTTPError as error:
            problems.append(f"HTTP {error.code} {error.reason}")
        except (urllib.error.URLError, OSError) as error:
            raise OSError(f"could not reach {host}: {getattr(error, 'reason', error)}") from error
    hint = "" if password else " (does the link need a password?)"
    if not any(code in p for p in problems for code in ("401", "403")):
        hint = ""
    raise OSError("the cloud did not take the file: " + "; ".join(problems) + hint)


def send(now: Optional[datetime] = None, opener=None, everything: bool = False
         ) -> Dict[str, Any]:
    """Bundle what is new since the last send and upload it.  Returns what was
    sent (``count`` 0 and nothing uploaded when there was nothing new);
    raises `OSError` or `ValueError`, noting the error in the settings, when
    it could not go.  The time of a send is recorded only once it went."""
    import tempfile
    state = settings()
    if not state["enabled"]:
        raise ValueError("the usage log is off (Preferences > Usage log)")
    if not state["link"].strip():
        raise ValueError("no link to send to (Preferences > Usage log)")
    now = now or datetime.now()
    since = None if everything else _stamp(state["last_sent"])
    with tempfile.TemporaryDirectory(prefix="smappy-usage-") as scratch:
        path = Path(scratch) / bundle_name(now)
        made = bundle(path, since)
        made["since"] = since
        if made["count"] == 0:
            save_settings(last_error="")
            return made
        try:
            made["url"] = upload(path, state["link"], state["password"], opener=opener)
        except Exception as error:
            save_settings(last_error=f"{now:%Y-%m-%d %H:%M}: {error}")
            raise
        made["bytes"] = path.stat().st_size
    save_settings(last_sent=now.isoformat(timespec="seconds"), last_error="")
    event("sent", parts=made["parts"], bytes=made["bytes"])
    return made


def due(now: Optional[datetime] = None) -> bool:
    """Whether an automatic send is due: on, with a link, and `AUTOMATIC_DAYS`
    since the last (or never sent)."""
    state = settings()
    if not (state["enabled"] and state["automatic"] and state["link"].strip()):
        return False
    last = _stamp(state["last_sent"])
    return last is None or (now or datetime.now()) - last >= timedelta(days=AUTOMATIC_DAYS)


def send_in_background(done=None) -> Optional[threading.Thread]:
    """`send` on a thread of its own, when `due`; ``done(result, error)`` is
    called on that thread.  An upload over a slow network must not hold the
    window, and its failure is kept in the settings for the page to show."""
    if not due():
        return None

    def work():
        try:
            result, error = send(), None
        except Exception as problem:      # noqa: BLE001 -- noted in the settings
            result, error = None, problem
        if done is not None:
            try:
                done(result, error)
            except Exception as problem:  # noqa: BLE001
                print(f"usage log: {problem}")

    thread = threading.Thread(target=work, name="usage-log-send", daemon=True)
    thread.start()
    return thread


def describe_since(state: Optional[Dict[str, Any]] = None) -> str:
    """A line for the settings page: when it last went, and any failure."""
    state = state or settings()
    last = _stamp(state["last_sent"])
    text = f"last sent {last:%Y-%m-%d %H:%M}" if last else "not sent yet"
    if state["last_error"]:
        text += f"; the last try failed -- {state['last_error']}"
    return text


def iter_parts(path: Path) -> Iterator[str]:
    """The names in a bundle, for a test or a look before sending."""
    with zipfile.ZipFile(path) as z:
        yield from z.namelist()

