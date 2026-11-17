"""Turn SMAPpy usage-log bundles into digests a model can read.

    python .claude/skills/usage-logs/digest.py BUNDLES OUT [--all]

``BUNDLES`` is the folder the bundles land in (`smappy/usage_log.py` sends
``smappy-usage-<who>-<date>.zip``; searched recursively); ``OUT`` gets

* ``overview.md`` -- per sender, then pooled: the tools the assistant called
  and which failed, every `find_plugins` query with what it found, the
  source it read, the help pages opened, the plugin runs and their failures,
  the assistant's own-code and missing-tools notes, and the conversations
  ranked by how much friction they show;
* ``sessions/<sender>/<session>.md`` -- each conversation as a timeline: what
  the user asked, what the assistant said and called (shell commands, files
  written, SMAPpy's tools with their arguments), and the errors -- no
  thinking, no successful outputs but SMAPpy's searches;
* ``digested.json`` -- the bundles read, so a later run reads only new ones
  unless ``--all``.  The overview is always made from every bundle seen.

Standard library only, so it runs with any Python, and nothing in it is
SMAPpy's: the bundles are data.  A transcript grows between sends and comes
again in the next bundle, so each conversation is kept once, its longest copy.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import zipfile
from pathlib import Path

USER_LIMIT = 3000          # characters of a user's message kept
SAY_LIMIT = 800            # of the assistant's text
CALL_LIMIT = 400           # of a tool call's input
ERROR_LIMIT = 500          # of an error
SCRIPT_LINES = 60          # of a Python file the assistant wrote
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
# Claude Code names a server's tools mcp__<server>__<tool>; Codex, by version,
# with or without the mcp__
SMAPPY_TOOL = re.compile(r"^(?:mcp__)?smappy[^_]*__(.+)$")


def cut(text, limit: int) -> str:
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit] + f" [...{len(text) - limit} more]"


def one_line(value, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return cut(re.sub(r"\s+", " ", text), limit)


def block_text(content) -> str:
    """The text of a message's content: a string or a list of blocks."""
    if isinstance(content, str):
        return content
    out = []
    for block in content or []:
        if isinstance(block, dict):
            if block.get("type") in ("text", "input_text", "output_text"):
                out.append(block.get("text", ""))
            elif block.get("type") == "tool_result":
                out.append(block_text(block.get("content")))
        elif isinstance(block, str):
            out.append(block)
    return "\n".join(t for t in out if t)


def smappy_tool(name: str) -> str:
    """``find_plugins`` from ``mcp__smappy__find_plugins``; "" for another tool."""
    match = SMAPPY_TOOL.match(name or "")
    return match.group(1) if match else ""


# ------------------------------------------------------------ transcripts
def describe_call(name: str, args) -> list:
    """Lines for one tool call: what was run or written, not all of it."""
    if not isinstance(args, dict):
        try:
            args = json.loads(args)
        except Exception:
            return [f"- **{name}** {one_line(args, CALL_LIMIT)}"]
    tool = smappy_tool(name)
    if tool:
        return [f"- **smappy.{tool}** {one_line(args, CALL_LIMIT)}"]
    if name in ("Bash", "shell", "exec_command", "local_shell"):
        command = args.get("command") or args.get("cmd") or args
        if isinstance(command, list):
            command = " ".join(map(str, command))
        return [f"- **shell** `{one_line(command, CALL_LIMIT)}`"]
    if name in ("Write", "Edit", "MultiEdit", "NotebookEdit", "apply_patch"):
        path = args.get("file_path") or args.get("path") or args.get("notebook_path") or ""
        lines = [f"- **{name}** {path}"]
        body = args.get("content") or ""
        if body and str(path).endswith(".py") and name == "Write":
            shown = body.splitlines()[:SCRIPT_LINES]
            more = len(body.splitlines()) - len(shown)
            lines += ["  ```python", *("  " + s for s in shown),
                      *([f"  # ... {more} more lines"] if more > 0 else []), "  ```"]
        elif name == "apply_patch":
            lines[0] += " " + one_line(args.get("input") or args, CALL_LIMIT)
        return lines
    if name in ("Read", "Glob", "Grep", "WebFetch", "WebSearch"):
        return [f"- **{name}** {one_line(args, 200)}"]
    return [f"- **{name}** {one_line(args, CALL_LIMIT)}"]


def claude_timeline(records) -> dict:
    """A Claude Code conversation: its lines, and the counts the overview ranks by."""
    lines, stats = [], collections.Counter()
    names = {}                                   # tool_use id -> name
    first_ask, when = "", ""
    for r in records:
        kind = r.get("type")
        if kind not in ("user", "assistant") or r.get("isMeta"):
            continue
        when = when or r.get("timestamp", "")
        content = (r.get("message") or {}).get("content")
        if kind == "user":
            if isinstance(content, str) or not any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content or []):
                text = REMINDER.sub("", block_text(content)).strip()
                if not text or text.startswith(("<command-", "<local-command", "Caveat:")):
                    continue
                stats["asks"] += 1
                first_ask = first_ask or text
                lines += ["", f"### User ({r.get('timestamp', '')[:16]})", "", cut(text, USER_LIMIT), ""]
                continue
            for block in content:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                name = names.get(block.get("tool_use_id"), "")
                text = block_text(block.get("content"))
                if block.get("is_error"):
                    stats["errors"] += 1
                    lines.append(f"  - ERROR from {name}: {one_line(text, ERROR_LIMIT)}")
                elif smappy_tool(name) in ("find_plugins", "list_plugins", "read_doc"):
                    lines.append(f"  - got: {one_line(text, CALL_LIMIT)}")
            continue
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text", "").strip():
                lines.append(f"> {one_line(block['text'], SAY_LIMIT)}")
            elif block.get("type") == "tool_use":
                name = block.get("name", "")
                names[block.get("id")] = name
                count_call(stats, name, block.get("input"))
                lines += describe_call(name, block.get("input"))
    return {"lines": lines, "stats": stats, "first": first_ask, "when": when}


def count_call(stats, name: str, args) -> None:
    stats["calls"] += 1
    tool = smappy_tool(name)
    if tool:
        stats["smappy"] += 1
        stats[f"smappy.{tool}"] += 1
        return
    args = args if isinstance(args, dict) else {}
    path = str(args.get("file_path") or args.get("path") or "")
    if name in ("Write", "apply_patch") and (path.endswith(".py") or name == "apply_patch"):
        stats["scripts"] += 1
    if name in ("Bash", "shell", "exec_command", "local_shell"):
        stats["shell"] += 1
        command = str(args.get("command") or args.get("cmd") or "")
        if re.search(r"\bpython", command):
            stats["python runs"] += 1


def codex_timeline(records) -> dict:
    """A Codex conversation, from its rollout's response items."""
    lines, stats = [], collections.Counter()
    first_ask, when = "", ""
    for r in records:
        payload = r.get("payload") if isinstance(r.get("payload"), dict) else r
        kind = payload.get("type")
        when = when or r.get("timestamp", "")
        if kind == "message":
            text = block_text(payload.get("content")).strip()
            # what Codex puts in front of a conversation: the folder's AGENTS.md
            # (SMAPpy writes one there) and the environment, not the user's words
            if not text or text.startswith(("<environment_context>", "# AGENTS.md")) \
                    or "<user_instructions>" in text or "<INSTRUCTIONS>" in text:
                continue
            if payload.get("role") == "user":
                stats["asks"] += 1
                first_ask = first_ask or text
                lines += ["", "### User", "", cut(text, USER_LIMIT), ""]
            elif payload.get("role") == "assistant":
                lines.append(f"> {one_line(text, SAY_LIMIT)}")
        elif kind in ("function_call", "custom_tool_call", "local_shell_call"):
            name = payload.get("name") or kind
            args = payload.get("arguments") or payload.get("input") or payload.get("action") or {}
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except Exception:
                pass
            count_call(stats, name, args)
            lines += describe_call(name, args)
        elif kind in ("function_call_output", "custom_tool_call_output"):
            output = payload.get("output")
            text = output if isinstance(output, str) else json.dumps(output)
            if re.search(r'"exit_code":\s*[1-9]|Traceback|Error:', text or ""):
                stats["errors"] += 1
                lines.append(f"  - ERROR: {one_line(text, ERROR_LIMIT)}")
    return {"lines": lines, "stats": stats, "first": first_ask, "when": when}


def friction(stats) -> float:
    """How much a conversation struggled: errors, scripts of its own, Python
    run in a shell, and SMAPpy's source read -- per question asked."""
    raw = (2 * stats["errors"] + 3 * stats["scripts"] + stats["python runs"]
           + 2 * stats["smappy.read_source"])
    return raw / max(1, stats["asks"])


# --------------------------------------------------------------- bundles
def read_jsonl(text: str) -> list:
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


RUN = re.compile(r"^(\S+ \S+) \w+\s+\[[^\]]*\] (\w+) (\S+) \(v([^)]*)\)")
FAILED = re.compile(r"^(\S+ \S+) \w+\s+\[[^\]]*\] (\w+) (\S+) failed after")


def plugin_runs(log: str):
    """``(path, failed, first error line)`` for each plugin run of a log."""
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if RUN.match(line):
            yield RUN.match(line).group(3), False, ""
        elif FAILED.match(line):
            tail = next((l for l in lines[i + 1:i + 60] if re.match(r"^\w+(Error|Exception)\b", l)), "")
            yield FAILED.match(line).group(3), True, tail


def notes(markdown: str) -> dict:
    """The ``## `` entries of own-code.md or missing-tools.md, by heading."""
    out, head, body = {}, None, []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if head:
                out[head] = "\n".join(body).strip()
            head, body = line[3:].strip(), []
        elif head:
            body.append(line)
    if head:
        out[head] = "\n".join(body).strip()
    return out


class Sender:
    def __init__(self, name):
        self.name = name
        self.bundles, self.versions = [], set()
        self.tools = collections.Counter()
        self.failed_tools = []
        self.searches = []
        self.sources = collections.Counter()
        self.docs = collections.Counter()
        self.help = collections.Counter()
        self.runs = collections.Counter()
        self.failed_runs = []
        self.own_code, self.missing = {}, {}
        self.sessions = {}            # key -> (size, timeline)


def read_bundle(path: Path, senders: dict) -> None:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        about = json.loads(z.read("about.json")) if "about.json" in names else {}
        who = (about.get("name") or about.get("installation") or path.stem).strip() or path.stem
        s = senders.setdefault(who, Sender(who))
        s.bundles.append(path.name)
        s.versions.add(str(about.get("smappy", "?")))
        for name in names:
            data = z.read(name).decode("utf-8", errors="replace")
            if name == "events.jsonl":
                for e in read_jsonl(data):
                    if e.get("kind") == "help":
                        s.help[e.get("page", "")] += 1
                    elif e.get("kind") == "tool":
                        tool = e.get("tool", "")
                        s.tools[tool] += 1
                        args = e.get("arguments") or {}
                        if not e.get("ok"):
                            s.failed_tools.append((e.get("t", ""), tool, one_line(args, 200),
                                                   one_line(e.get("error", ""), ERROR_LIMIT)))
                        if tool == "find_plugins":
                            s.searches.append((args.get("query", "") if isinstance(args, dict) else args,
                                               one_line(e.get("answer", ""), 300)))
                        elif tool == "read_source":
                            s.sources[str(args.get("name", "")) if isinstance(args, dict) else str(args)] += 1
                        elif tool == "read_doc":
                            s.docs[str(args.get("path", "")) if isinstance(args, dict) else str(args)] += 1
            elif name.startswith("logs/"):
                for plugin, failed, error in plugin_runs(data):
                    s.runs[plugin] += 1
                    if failed:
                        s.failed_runs.append((plugin, error))
            elif name == "assistant/own-code.md":
                s.own_code.update(notes(data))
            elif name == "assistant/missing-tools.md":
                s.missing.update(notes(data))
            elif name.startswith("transcripts/"):
                client = name.split("/")[1]
                key = (client, Path(name).stem)
                previous = s.sessions.get(key)
                if previous is None or len(data) > previous[0]:
                    records = read_jsonl(data)
                    timeline = (claude_timeline if client == "claude" else codex_timeline)(records)
                    s.sessions[key] = (len(data), timeline)


# ---------------------------------------------------------------- writing
def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-") or "x"


def write_sessions(s: Sender, out: Path) -> list:
    rows = []
    for (client, session), (_, t) in s.sessions.items():
        if not t["stats"]["asks"] and not t["stats"]["calls"]:
            continue
        path = out / "sessions" / slug(s.name) / f"{client}-{session}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        st = t["stats"]
        head = [f"# {s.name}: {client} conversation {session}", "",
                f"started {t['when'][:16]}; {st['asks']} questions, {st['calls']} tool calls "
                f"({st['smappy']} SMAPpy), {st['scripts']} scripts written, "
                f"{st['python runs']} Python runs, {st['errors']} errors", ""]
        path.write_text("\n".join(head + t["lines"]) + "\n", encoding="utf-8")
        rows.append((friction(st), s.name, path.relative_to(out), one_line(t["first"], 160), st))
    return rows


def table(counter, limit=30) -> list:
    return [f"| {n} | {k or '-'} |" for k, n in counter.most_common(limit)]


def overview(senders: dict, out: Path) -> str:
    md = ["# SMAPpy usage logs: overview", ""]
    all_rows = []
    pooled = collections.defaultdict(collections.Counter)
    for s in senders.values():
        rows = write_sessions(s, out)
        all_rows += rows
        for field in ("tools", "sources", "docs", "help", "runs"):
            pooled[field].update(getattr(s, field))
        md += [f"## {s.name}", "",
               f"{len(s.bundles)} bundles ({', '.join(sorted(s.bundles)[:1] + sorted(s.bundles)[-1:])}); "
               f"SMAPpy {', '.join(sorted(s.versions))}; {len(rows)} conversations", ""]
        if s.searches:
            md += ["### find_plugins queries", ""]
            md += [f"- `{q}` -> {a}" for q, a in s.searches[-60:]] + [""]
        if s.failed_tools:
            md += ["### Tool calls that failed", ""]
            md += [f"- {t[:16]} **{tool}** {args}: {err}" for t, tool, args, err in s.failed_tools[-40:]] + [""]
        if s.failed_runs:
            md += ["### Plugin runs that failed", ""]
            md += [f"- **{p}**: {e}" for p, e in s.failed_runs[-40:]] + [""]
        for title, notes_ in (("Own code (the assistant's record)", s.own_code),
                              ("Missing tools (the assistant's record)", s.missing)):
            if notes_:
                md += [f"### {title}", ""]
                for head, body in notes_.items():
                    md += [f"- **{head}**", *("  " + l for l in body.splitlines() if l.strip())]
                md += [""]
    md += ["## Pooled", ""]
    for field, title in (("tools", "SMAPpy tools called"), ("sources", "Source read (read_source)"),
                         ("docs", "Pages read by the assistant (read_doc)"),
                         ("help", "Help pages opened by people"), ("runs", "Plugin runs")):
        if pooled[field]:
            md += [f"### {title}", "", "| n | what |", "| --- | --- |", *table(pooled[field]), ""]
    md += ["## Conversations, most friction first", "",
           "friction = (2 errors + 3 scripts + Python runs + 2 source reads) per question", "",
           "| friction | who | first question | file |", "| --- | --- | --- | --- |"]
    for score, who, path, first, _ in sorted(all_rows, key=lambda r: -r[0]):
        md.append(f"| {score:.1f} | {who} | {first.replace('|', '/')} | {path} |")
    return "\n".join(md) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bundles", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--all", action="store_true", help="read every bundle, not only new ones")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    seen_file = args.out / "digested.json"
    seen = set() if args.all or not seen_file.exists() else set(json.loads(seen_file.read_text()))
    bundles = sorted(args.bundles.expanduser().rglob("smappy-usage-*.zip"))
    new = [b for b in bundles if b.name not in seen]
    senders: dict = {}
    for bundle in bundles:              # the overview is of everything seen
        try:
            read_bundle(bundle, senders)
        except (zipfile.BadZipFile, OSError, KeyError) as error:
            print(f"skipped {bundle.name}: {error}", file=sys.stderr)
    (args.out / "overview.md").write_text(overview(senders, args.out), encoding="utf-8")
    seen_file.write_text(json.dumps(sorted(seen | {b.name for b in bundles}), indent=1))
    print(f"{len(bundles)} bundles ({len(new)} new) from {len(senders)} senders -> "
          f"{args.out / 'overview.md'}")
    for b in new:
        print(f"new: {b.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
