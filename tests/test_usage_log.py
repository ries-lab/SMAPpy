"""The usage log: what is written, what a bundle gathers, and the upload."""
import json
import os
import re
import urllib.error
import zipfile
from datetime import datetime, timedelta

import pytest

from smappy import config, usage_log


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A configuration, a home and an assistant folder of the test's own."""
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    config.load(reload=True)
    work = tmp_path / "SMAPpy assistant"
    work.mkdir()
    config.set("assistant", {"folder": str(work)})
    yield tmp_path
    config.load(reload=True)


def events():
    lines = []
    for path in sorted(usage_log.folder().glob("events-*.jsonl")):
        lines += [json.loads(line) for line in path.read_text().splitlines()]
    return lines


def test_nothing_is_written_while_the_log_is_off(home):
    usage_log.event("help", page="Analysis/Drift/RCC")
    usage_log.tool_call("find_plugins", {"query": "drift"}, 0.1, [])
    assert not usage_log.folder().exists()


def test_the_assistants_tool_calls_are_logged_with_their_outcome(home, tmp_path):
    from smappy.assistant.protocol import Server
    from smappy.assistant.tools import Assistant
    usage_log.save_settings(enabled=True)
    server = Server(Assistant(output=tmp_path))
    server.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                   "params": {"clientInfo": {"name": "claude-code"}}})
    server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "find_plugins", "arguments": {"query": "drift beads"}}})
    server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                   "params": {"name": "describe_plugin",
                              "arguments": {"path": "No/Such/Plugin"}}})
    found, failed = events()
    assert found["tool"] == "find_plugins" and found["ok"]
    assert found["arguments"] == {"query": "drift beads"} and found["client"] == "claude-code"
    assert "Drift" in json.dumps(found["answer"])       # what the search came back with
    assert failed["tool"] == "describe_plugin" and not failed["ok"] and failed["error"]


def test_a_long_answer_is_cut_and_a_picture_named_not_kept(home):
    from smappy.assistant.tools import Image
    usage_log.save_settings(enabled=True)
    usage_log.tool_call("read_values", {"columns": ["x_nm"]}, 1.0, {"x_nm": list(range(10_000))})
    usage_log.tool_call("figure", {"result": "r1"}, 1.0, Image(b"\x89PNG" * 10_000, "drift"))
    values, picture = events()
    assert len(values["answer"]) < usage_log.ANSWER_LIMIT + 100
    assert picture["answer"] == "<picture: drift>"


def test_a_claude_folder_named_otherwise_is_found_by_the_folder_its_conversations_record(home):
    work = usage_log.assistant_folder()
    write_log(home / "claude" / "projects" / "renamed" / "s.jsonl",
              [json.dumps({"type": "summary"}), json.dumps({"cwd": str(work.resolve())})])
    assert [p.name for p in usage_log.claude_transcripts(work)] == ["s.jsonl"]


def write_log(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def test_a_bundle_gathers_the_events_logs_notes_and_the_clients_transcripts(home):
    usage_log.save_settings(enabled=True)
    usage_log.event("help", page="Analysis/Drift/RCC")
    write_log(config.config_dir() / "logs" / "smappy.log",
              ["2020-01-01 10:00:00,000 INFO    [Main] run Analysis/Drift/RCC (v1): -",
               f"{datetime.now():%Y-%m-%d %H:%M:%S},000 ERROR   [Main] run X failed",
               "Traceback (most recent call last):", "ValueError: no"])
    assistant = config.config_dir() / "assistant"
    write_log(assistant / "own-code.md", ["# own code", "## a histogram"])
    write_log(assistant / "own-code" / "20261117-hist.py", ["import numpy"])
    work = usage_log.assistant_folder()
    project = home / "claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(work.resolve()))
    picture = {"type": "image", "source": {"type": "base64", "data": "A" * 50_000}}
    write_log(project / "s1.jsonl", [
        json.dumps({"type": "user", "message": {"content": "find the drift of my beads"}}),
        json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "content": [picture, {"type": "text", "text": "B" * 20_000}]}]}})])
    codex = home / "codex" / "sessions" / "2026" / "11" / "17"
    write_log(codex / "rollout-1.jsonl", [json.dumps({"type": "session_meta",
                                                      "payload": {"cwd": str(work.resolve())}})])
    write_log(codex / "rollout-2.jsonl", [json.dumps({"type": "session_meta",
                                                      "payload": {"cwd": "/elsewhere"}})])

    since = datetime.now() - timedelta(hours=1)
    made = usage_log.bundle(home / "b.zip", since)
    names = set(usage_log.iter_parts(home / "b.zip"))
    assert {"events.jsonl", "logs/smappy.log", "assistant/own-code.md",
            "assistant/own-code/20261117-hist.py", "transcripts/claude/s1.jsonl",
            "transcripts/codex/rollout-1.jsonl", "about.json"} == names
    with zipfile.ZipFile(home / "b.zip") as z:
        log = z.read("logs/smappy.log").decode()
        transcript = z.read("transcripts/claude/s1.jsonl").decode()
        about = json.loads(z.read("about.json"))
    # only the new lines of the log, a traceback with the line it belongs to
    assert "Drift/RCC" not in log and "ValueError: no" in log
    assert "find the drift of my beads" in transcript
    assert "A" * 1000 not in transcript and "B" * (usage_log.TEXT_LIMIT + 1) not in transcript
    assert about["parts"]["transcripts"] == 2 and made["count"] == sum(about["parts"].values())


def test_claude_conversations_elsewhere_go_only_if_they_called_smappys_tools(home):
    usage_log.save_settings(enabled=True)
    projects = home / "claude" / "projects"
    call = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "mcp__smappy__find_plugins", "input": {"query": "drift"}}]}}
    write_log(projects / "-data-cells" / "used.jsonl",
              [json.dumps({"type": "user", "message": {"content": "drift?"}}), json.dumps(call)])
    # a conversation that only talks about the tools, as a developer's does
    talk = {"type": "user", "message": {"content": 'it calls "name": "mcp__smappy__run_plugin"'}}
    write_log(projects / "-code-SMAPpy" / "talk.jsonl", [json.dumps(talk)])
    write_log(projects / "-other" / "other.jsonl", [json.dumps(
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}]}})])
    usage_log.bundle(home / "b.zip", datetime.now() - timedelta(hours=1))
    names = set(usage_log.iter_parts(home / "b.zip"))
    assert "transcripts/claude/used.jsonl" in names
    assert not {"transcripts/claude/talk.jsonl", "transcripts/claude/other.jsonl"} & names


class Answer:
    status = 201

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_share_link_is_read_for_its_server_and_token():
    assert usage_log.parse_link("https://ucloud.univie.ac.at/index.php/s/AbC123xYz") == \
        ("https://ucloud.univie.ac.at", "AbC123xYz")
    assert usage_log.parse_link(" https://cloud.example.org/nc/s/Tok3n/ ") == \
        ("https://cloud.example.org/nc", "Tok3n")
    with pytest.raises(ValueError):
        usage_log.parse_link("https://ucloud.univie.ac.at/apps/files/")


def test_the_upload_puts_the_file_into_the_share_falling_back_to_the_older_webdav(tmp_path):
    file = tmp_path / "b.zip"
    file.write_bytes(b"zip")
    asked = []

    def opener(request, timeout):
        asked.append(request)
        if "/dav/files/" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
        return Answer()

    url = usage_log.upload(file, "https://cloud.example.org/index.php/s/Tok3n", "pw",
                           opener=opener)
    assert [r.full_url for r in asked] == [
        "https://cloud.example.org/public.php/dav/files/Tok3n/b.zip",
        "https://cloud.example.org/public.php/webdav/b.zip"]
    assert url == asked[1].full_url and asked[1].get_method() == "PUT"
    assert asked[1].data == b"zip"


def test_a_send_goes_once_records_when_and_sends_only_what_is_new_after(home):
    usage_log.save_settings(enabled=True, link="https://cloud.example.org/index.php/s/Tok3n",
                            name="Ada L")
    usage_log.event("help", page="Panels/Glossary")
    sent = []

    def opener(request, timeout):
        sent.append(request)
        return Answer()

    first = usage_log.send(opener=opener)
    assert first["parts"] == {"events": 1} and len(sent) == 1
    assert "/smappy-usage-Ada-L-" in sent[0].full_url
    assert usage_log.settings()["last_sent"] and not usage_log.due()
    again = usage_log.send(now=datetime.now() + timedelta(seconds=2), opener=opener)
    assert again["count"] == 0 and len(sent) == 1           # nothing new, nothing sent


def test_a_sends_own_record_alone_is_not_sent_but_goes_with_the_next(home):
    """A send that crossed a second: its "sent" record is stamped after the
    `last_sent` it saved, which is the start of the send cut to the second."""
    start = (datetime.now() - timedelta(minutes=1)).replace(microsecond=0)
    usage_log.save_settings(enabled=True, link="https://cloud.example.org/index.php/s/Tok3n",
                            last_sent=start.isoformat())
    usage_log.folder().mkdir(parents=True, exist_ok=True)
    later = (start + timedelta(seconds=1)).isoformat()
    (usage_log.folder() / f"events-{start:%Y-%m}.jsonl").write_text(
        f'{{"t": "{later}", "kind": "sent", "parts": {{"events": 1}}}}\n')
    sent = []

    def opener(request, timeout):
        sent.append(request)
        return Answer()

    assert usage_log.send(opener=opener)["count"] == 0 and not sent
    usage_log.event("help", page="Panels/Glossary")
    made = usage_log.send(opener=opener)
    assert made["count"] == 1 and made["parts"] == {"events": 2} and len(sent) == 1


def test_a_failed_send_is_noted_and_the_next_one_sends_the_same_again(home):
    usage_log.save_settings(enabled=True, link="https://cloud.example.org/s/Tok3n")
    usage_log.event("help", page="Panels/Glossary")

    def refused(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    with pytest.raises(OSError, match="password"):
        usage_log.send(opener=refused)
    state = usage_log.settings()
    assert state["last_sent"] == "" and "401" in state["last_error"]
    assert "failed" in usage_log.describe_since(state)


def test_nothing_is_sent_while_off_or_without_a_link(home):
    with pytest.raises(ValueError, match="off"):
        usage_log.send()
    usage_log.save_settings(enabled=True)
    with pytest.raises(ValueError, match="link"):
        usage_log.send()


def test_an_automatic_send_is_due_only_when_asked_for_and_a_week_has_passed(home):
    usage_log.save_settings(enabled=True, link="https://c.org/s/T", automatic=False)
    assert not usage_log.due()
    usage_log.save_settings(automatic=True)
    assert usage_log.due()                                  # never sent
    usage_log.save_settings(last_sent=(datetime.now() - timedelta(days=2)).isoformat())
    assert not usage_log.due()
    assert usage_log.due(now=datetime.now() + timedelta(days=6))


def test_the_link_stays_when_the_other_settings_change(home):
    usage_log.save_settings(link="https://c.org/s/T")
    usage_log.save_settings(enabled=True, name="x")
    config.load(reload=True)
    assert usage_log.settings()["link"] == "https://c.org/s/T"
    assert os.path.exists(config.config_file())


def test_the_preferences_page_keeps_what_is_typed(home):
    from PySide6.QtWidgets import QApplication
    from smappy.gui.preferences import UsageLogPage
    QApplication.instance() or QApplication([])
    page = UsageLogPage()
    assert not page.link.isEnabled()
    page.enabled.setChecked(True)
    assert page.link.isEnabled()
    page.link.setText("https://c.org/s/T")
    page.link.editingFinished.emit()
    assert usage_log.settings()["link"] == "https://c.org/s/T"
    assert UsageLogPage().link.text() == "https://c.org/s/T"

