"""The usage-log skill's digest: bundles in, an overview and timelines out."""
import base64
import importlib.util
import json
import re
import urllib.error
from datetime import datetime, timedelta
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


def test_a_bundle_sent_to_try_the_link_is_not_read(home, digest, capsys):
    a_conversation(home)
    bundles = home / "drop"
    bundles.mkdir()
    usage_log.bundle(bundles / "smappy-usage-test.zip")
    usage_log.bundle(bundles / "smappy-usage-Ada-test.zip")
    digest.main([str(bundles), str(home / "digest")])
    assert capsys.readouterr().out.startswith("0 bundles (0 new) from 0 senders")


LISTING = b"""<?xml version="1.0"?>
<d:multistatus xmlns:d="DAV:">
 <d:response><d:href>/remote.php/dav/files/ada/SMAPpy%20usage/</d:href></d:response>
 <d:response><d:href>/remote.php/dav/files/ada/SMAPpy%20usage/smappy-usage-Bo-20261120-100000.zip</d:href></d:response>
 <d:response><d:href>/remote.php/dav/files/ada/SMAPpy%20usage/smappy-usage-Cy-20261121-100000.zip</d:href></d:response>
 <d:response><d:href>/remote.php/dav/files/ada/SMAPpy%20usage/smappy-usage-test.zip</d:href></d:response>
 <d:response><d:href>/remote.php/dav/files/ada/SMAPpy%20usage/notes.txt</d:href></d:response>
</d:multistatus>"""
FOLDER = "https://cloud.example.org/remote.php/dav/files/ada/SMAPpy%20usage/"


class Reply:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def server(asked, refuse=None):
    def opener(request, timeout):
        asked.append(request)
        if refuse:
            raise urllib.error.HTTPError(request.full_url, refuse, "Unauthorized", {}, None)
        if request.get_method() == "PROPFIND":
            return Reply(LISTING, 207)
        return Reply(b"zip of " + request.full_url.rsplit("/", 1)[-1].encode())
    return opener


def test_the_listing_is_read_for_the_bundles_and_only_new_ones_are_fetched(digest, tmp_path):
    target = tmp_path / "bundles"
    target.mkdir()
    (target / "smappy-usage-Bo-20261120-100000.zip").write_bytes(b"had it")
    asked = []
    got = digest.fetch(FOLDER, "ada", target, password="pw", opener=server(asked))
    assert got == ["smappy-usage-Cy-20261121-100000.zip"]
    propfind, get = asked
    assert propfind.get_method() == "PROPFIND" and propfind.get_header("Depth") == "1"
    assert propfind.get_header("Authorization") == "Basic " + base64.b64encode(b"ada:pw").decode()
    assert get.get_method() == "GET" and get.full_url == FOLDER + "smappy-usage-Cy-20261121-100000.zip"
    assert (target / got[0]).read_bytes() == b"zip of smappy-usage-Cy-20261121-100000.zip"
    assert (target / "smappy-usage-Bo-20261120-100000.zip").read_bytes() == b"had it"
    assert not list(target.glob("*.part"))


def test_a_refused_login_is_reported_plainly(digest, tmp_path, monkeypatch, capsys):
    with pytest.raises(digest.FetchError, match="refused the login for 'ada'.*Keychain"):
        digest.fetch(FOLDER, "ada", tmp_path, password="wrong", opener=server([], refuse=401))
    # from the command line: a line on stderr and a code, no traceback
    monkeypatch.setattr(digest, "keychain_password", lambda user="": "wrong")
    monkeypatch.setattr(digest.urllib.request, "urlopen", server([], refuse=401))
    out = tmp_path / "out"
    assert digest.main(["--fetch", str(out), "--url", FOLDER, "--user", "ada"]) == 2
    assert "refused the login" in capsys.readouterr().err
    assert json.loads((out / "fetch.json").read_text()) == {"url": FOLDER, "user": "ada"}


def test_a_fetch_remembers_the_folder_and_digests_what_it_fetched(digest, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(digest, "keychain_password", lambda user="": "pw")
    monkeypatch.setattr(digest.urllib.request, "urlopen", server([]))
    monkeypatch.setattr(digest, "read_bundle", lambda path, senders: None)
    out = tmp_path / "out"
    assert digest.main(["--fetch", str(out), "--url", FOLDER, "--user", "ada"]) == 0
    assert digest.main(["--fetch", str(out)]) == 0                 # fetch.json kept them
    printed = capsys.readouterr().out
    assert "fetched 2 new bundles" in printed and "fetched 0 new bundles" in printed
    assert "password" not in (out / "fetch.json").read_text()


def test_the_password_is_found_by_the_service_when_stored_under_another_account(digest, monkeypatch):
    class Done:
        def __init__(self, code, out):
            self.returncode, self.stdout = code, out

    asked = []

    def run(command, **kw):
        asked.append(command)
        return Done(0, "pw\n") if "-a" not in command else Done(44, "")

    monkeypatch.setattr(digest.subprocess, "run", run)
    assert digest.keychain_password("ada@example.org") == "pw"
    assert asked[0][3:5] == ["-a", "ada@example.org"] and "-a" not in asked[1]


def test_the_logs_runs_with_spaces_uncaught_errors_console_lines_and_crashes_are_read(home, digest):
    logs = config.config_dir() / "logs"
    logs.mkdir(parents=True)
    now = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    (logs / "smappy.log").write_text(
        f"{now},000 INFO    [Thread-2 (reader)] run Localize/Spline 3D (v11) on 0 localizations: -\n"
        f"{now},000 ERROR   [Dummy-3] run Localize/Spline 3D failed after 0.05 s\n"
        "Traceback (most recent call last):\nValueError: a focus on auto needs the frames\n"
        f"{now},000 CRITICAL [MainThread] uncaught exception\n"
        f"{now},000 WARNING [MainThread] (err) Traceback (most recent call last):\n"
        f"{now},000 WARNING [MainThread] (err) ValueError: math domain error\n"
        f"{now},000 WARNING [MainThread] (err) <TiffTag.fromfile> raised TiffFileError('@463515 offset 0')\n"
        f"{now},000 WARNING [Dummy-1] (err) <TiffTag.fromfile> raised TiffFileError('@463532 offset 0')\n")
    (logs / "crash.log").write_text(
        "\n=== smappy started 2020-01-01 10:00:00, pid 1 ===\nFatal Python error: old\n"
        f"\n=== smappy started {now}, pid 2 ===\n"
        f"\n=== smappy started {now}, pid 3 ===\nFatal Python error: Segmentation fault\n")
    bundles = home / "drop"
    bundles.mkdir()
    made = usage_log.bundle(bundles / usage_log.bundle_name(),
                            datetime.now() - timedelta(hours=1))
    assert made["parts"]["crashes"] == 1          # the new crash; neither the old nor a quiet start
    out = home / "digest"
    digest.main([str(bundles), str(out)])
    overview = (out / "overview.md").read_text()
    assert "**Localize/Spline 3D**: ValueError: a focus on auto" in overview
    assert "| 1 | Localize/Spline 3D |" in overview
    assert "| 1 | ValueError: math domain error |" in overview
    assert "| 2 | <TiffTag.fromfile> raised TiffFileError('@# offset #') |" in overview
    assert "Segmentation fault" in overview and "old" not in overview.split("### Crashes")[1][:300]
