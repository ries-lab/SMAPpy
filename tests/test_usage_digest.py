"""The usage-log skill's digest: bundles in, an overview and timelines out."""
import importlib.util
import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from smappy import config, usage_log

DIGEST = Path(__file__).parents[1] / ".claude" / "skills" / "usage-logs" / "digest.py"


@pytest.fixture
def digest():
    spec = importlib.util.spec_from_file_location("usage_digest", DIGEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    config.load(reload=True)
    work = tmp_path / "SMAPpy assistant"
    work.mkdir()
    config.set("assistant", {"folder": str(work)})
    usage_log.save_settings(enabled=True, name="Ada")
    yield tmp_path
    config.load(reload=True)


def record(kind, content, **more):
    return json.dumps({"type": kind, "timestamp": "2026-11-17T10:00:00Z",
                       "message": {"role": kind, "content": content}, **more})


def a_conversation(home):
    """A Claude Code conversation in the shapes its transcripts have: a
    question, SMAPpy's search, a script of its own run with Python, an error."""
    work = usage_log.assistant_folder()
    folder = home / "claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(work.resolve()))
    folder.mkdir(parents=True)
    lines = [
        json.dumps({"type": "permission-mode", "mode": "default"}),
        record("user", "<system-reminder>ignore</system-reminder>How bright are my blinks per frame?"),
        record("assistant", [{"type": "thinking", "thinking": "secret"},
                             {"type": "tool_use", "id": "t1", "name": "mcp__smappy-gui__find_plugins",
                              "input": {"query": "photons per frame histogram"}}]),
        record("user", [{"type": "tool_result", "tool_use_id": "t1",
                         "content": [{"type": "text", "text": "[]"}]}]),
        record("assistant", [{"type": "text", "text": "No plugin; I will compute it."},
                             {"type": "tool_use", "id": "t2", "name": "Write",
                              "input": {"file_path": "/w/hist.py", "content": "import numpy\nprint(1)\n"}}]),
        record("user", [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}]),
        record("assistant", [{"type": "tool_use", "id": "t3", "name": "Bash",
                              "input": {"command": "python hist.py"}}]),
        record("user", [{"type": "tool_result", "tool_use_id": "t3", "is_error": True,
                         "content": "KeyError: 'phot'"}]),
    ]
    (folder / "s1.jsonl").write_text("\n".join(lines) + "\n")


def test_a_digest_ranks_the_conversation_and_shows_what_was_tried(home, digest):
    a_conversation(home)
    usage_log.tool_call("find_plugins", {"query": "photons per frame histogram"}, 0.1, [])
    usage_log.tool_call("read_source", {"name": "smappy.locs"}, 0.1, "...")
    usage_log.event("help", page="Analysis/Measure/Statistics")
    logs = config.config_dir() / "logs"
    logs.mkdir(parents=True)
    now = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    (logs / "smappy.log").write_text(
        f"{now},000 INFO    [MainThread] run Analysis/Drift/RCC (v3) on 10 localizations: -\n"
        f"{now},000 ERROR   [MainThread] run Analysis/Drift/RCC failed after 0.10 s\n"
        "Traceback (most recent call last):\n  File x\nValueError: too few frames\n")
    bundles = home / "drop"
    bundles.mkdir()
    usage_log.bundle(bundles / usage_log.bundle_name())

    out = home / "digest"
    assert digest.main([str(bundles), str(out)]) == 0
    overview = (out / "overview.md").read_text()
    assert "## Ada" in overview
    assert "`photons per frame histogram`" in overview
    assert "Analysis/Drift/RCC**: ValueError: too few frames" in overview
    assert "| 1 | smappy.locs |" in overview
    assert "| 1 | Analysis/Measure/Statistics |" in overview
    assert "How bright are my blinks per frame?" in overview
    session = next((out / "sessions" / "Ada").glob("*.md")).read_text()
    assert "**smappy.find_plugins**" in session and "got: []" in session
    assert "import numpy" in session and "`python hist.py`" in session
    assert "ERROR from Bash: KeyError: 'phot'" in session
    assert "secret" not in session and "ignore" not in session
    assert "1 scripts written, 1 Python runs, 1 errors" in session


def test_a_conversation_sent_twice_is_kept_once_and_only_new_bundles_are_named(home, digest, capsys):
    a_conversation(home)
    bundles = home / "drop"
    bundles.mkdir()
    usage_log.bundle(bundles / "smappy-usage-Ada-20261117-100000.zip")
    out = home / "digest"
    digest.main([str(bundles), str(out)])
    usage_log.bundle(bundles / "smappy-usage-Ada-20261124-100000.zip")
    digest.main([str(bundles), str(out)])
    printed = capsys.readouterr().out.splitlines()
    assert printed[-2:] == ["2 bundles (1 new) from 1 senders -> " + str(out / "overview.md"),
                            "new: smappy-usage-Ada-20261124-100000.zip"]
    assert len(list((out / "sessions" / "Ada").glob("*.md"))) == 1
    assert (out / "overview.md").read_text().count("How bright are my blinks") == 1
