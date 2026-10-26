"""The assistant's tools and the MCP server that serves them."""
import json
import os
import sys
from pathlib import Path

import pytest

from smappy import plugins
from smappy.assistant import index, knowledge
from smappy.assistant.protocol import Server
from smappy.assistant.tools import INSTRUCTIONS, Assistant, Image, ToolError

# a small simulation: a second or so for everything a test does with it
SIMULATION = {"n_frames": 5000}
STATISTICS = "Analysis/Measure/Localization Statistics"


def simulated(tmp_path, mode="act"):
    assistant = Assistant(output=tmp_path / "out", knowledge_folder=tmp_path / "kb")
    assistant.call("run_plugin", {"path": "File/Simulate/Simulate",
                                  "settings": SIMULATION, "apply": True})
    assistant.mode = mode
    return assistant


# ------------------------------------------------------------------ the tools

def test_every_tool_has_a_description_and_a_schema_its_signature_agrees_with():
    specs = Assistant.specs()
    assert {"session_state", "find_plugins", "read_doc", "read_source", "run_plugin",
            "scan", "remember", "recall"} <= set(specs)
    for spec in specs.values():
        assert spec.description, spec.name
        schema = spec.schema()
        assert set(schema["required"]) <= set(schema["properties"]), spec.name
        json.dumps(schema)


def test_trying_a_plugin_changes_nothing_and_applying_it_does(tmp_path):
    a = simulated(tmp_path)
    table = a.session.locs
    tried = a.call("run_plugin", {"path": "Analysis/Drift/RCC",
                                  "settings": {"n_timepoints": 5}})
    assert not tried["applied"] and tried["changed_table"]
    assert a.session.locs is table and not a.session.can_undo
    applied = a.call("run_plugin", {"path": "Analysis/Drift/RCC",
                                    "settings": {"n_timepoints": 5}, "apply": True})
    assert applied["applied"] and a.session.locs is not table and a.session.can_undo
    a.call("undo")
    assert a.session.locs is table


def test_in_suggest_mode_a_change_is_a_proposal_and_the_session_stays(tmp_path):
    a = simulated(tmp_path, mode="suggest")
    table, ranges = a.session.locs, dict(a.session.layers[0].ranges)
    answer = a.call("run_plugin", {"path": "Analysis/Drift/RCC", "apply": True,
                                   "reason": "the image is smeared"})
    assert answer["proposed"]["reason"] == "the image is smeared"
    a.call("set_filter", {"column": "photons", "lo": 500})
    assert a.session.locs is table and a.session.layers[0].ranges == ranges
    assert [p["tool"] for p in a.call("list_proposals")] == ["run_plugin", "set_filter"]
    # trying still works: that is how a suggestion is made
    assert "result" in a.call("run_plugin", {"path": STATISTICS})


def test_a_setting_the_plugin_does_not_have_is_refused_with_the_ones_it_has(tmp_path):
    a = simulated(tmp_path)
    with pytest.raises(ToolError, match="n_timepoints"):
        a.call("run_plugin", {"path": "Analysis/Drift/RCC", "settings": {"windows": 5}})
    with pytest.raises(ToolError, match="no plugin"):
        a.call("run_plugin", {"path": "Analysis/Drift/RCX"})
    with pytest.raises(ToolError, match="takes"):
        a.call("session_state", {"verbose": True})


def test_files_are_written_under_the_output_folder_and_nowhere_else(tmp_path):
    a = simulated(tmp_path)
    with pytest.raises(ToolError, match="outside the output folder"):
        a.call("run_plugin", {"path": "File/Export/Image",
                              "settings": {"path": str(tmp_path / "elsewhere.png")}})
    with pytest.raises(ToolError, match="writes a file"):
        a.call("run_plugin", {"path": "File/Export/Image"})
    a.call("run_plugin", {"path": "File/Export/Image", "settings": {"path": "view.png"}})
    assert (tmp_path / "out" / "view.png").is_file()
    with pytest.raises(ToolError, match="outside"):
        a.call("save", {"path": "../escaped.h5"})
    exporting = ("steps:\n  - plugin: File/Export/Image\n    values: {path: %s}\n")
    with pytest.raises(ToolError, match="outside"):
        a.call("run_chain", {"chain": exporting % (tmp_path / "chain.png")})
    with pytest.raises(ToolError, match="absolute"):
        a.call("run_chain", {"chain": exporting % "chain.png"})
    a.call("run_chain", {"chain": exporting % (tmp_path / "out" / "chain.png")})
    assert (tmp_path / "out" / "chain.png").is_file()
    assert Path(a.call("save", {"path": "table.h5"})["saved"]).is_file()


def test_a_plugins_question_comes_back_before_it_runs_and_confirm_runs_it(tmp_path,
                                                                          monkeypatch):
    a = simulated(tmp_path)
    cls = plugins.get(STATISTICS)
    asking = type("Asking", (cls,), {"preflight": lambda self, ctx, s: "this takes hours"})
    asking.path = STATISTICS
    monkeypatch.setattr(Assistant, "_plugin", lambda self, path: asking)
    answer = a.call("run_plugin", {"path": STATISTICS})
    assert answer["question"] == "this takes hours" and "result" not in answer
    assert "result" in a.call("run_plugin", {"path": STATISTICS, "confirm": True})


def test_session_state_counts_the_blinks_a_grouped_layer_shows(tmp_path):
    a = simulated(tmp_path)
    layer = a.call("session_state")["layers"][0]
    assert layer["grouped"] and layer["rows_are"] == "blinks"
    assert 0 < layer["shown"] <= layer["rows"] < len(a.session.locs)
    tighter = a.call("set_filter", {"column": "photons", "lo": 2000})
    assert tighter["shown"] < layer["shown"]
    a.call("set_grouped", {"on": False})
    assert a.call("session_state")["layers"][0]["rows"] == len(a.session.locs)


def test_a_scan_tries_each_value_and_sets_their_numbers_side_by_side(tmp_path):
    a = simulated(tmp_path)
    scan = a.call("scan", {"path": STATISTICS, "field": "bins", "values": [20, 80]})
    assert [row["bins"] for row in scan["rows"]] == [20, 80]
    assert all("stats.photons.n0" in row for row in scan["rows"])
    assert not a.session.can_undo


def test_figures_and_the_picture_come_back_as_png(tmp_path):
    a = simulated(tmp_path)
    run = a.call("run_plugin", {"path": STATISTICS})
    assert run["figures"]
    made = a.call("figure", {"result": run["result"], "index": 0})
    assert isinstance(made, Image) and made.png.startswith(b"\x89PNG")
    assert a.call("picture", {"pixelsize_nm": 40}).png.startswith(b"\x89PNG")
    with pytest.raises(ToolError, match="no result"):
        a.call("figure", {"result": "r999"})


def test_a_chain_is_validated_tried_and_saved_where_the_tree_finds_it(tmp_path,
                                                                    monkeypatch):
    a = simulated(tmp_path)
    chain = ("steps:\n  - plugin: Analysis/Measure/Localization Statistics\n"
             "  - plugin: Analysis/Drift/RCC\n    values: {n_timepoints: 5}\n")
    assert a.call("validate_chain", {"chain": chain})["ok"]
    bad = a.call("validate_chain", {"chain": chain.replace("n_timepoints", "windows")})
    assert not bad["ok"] and "n_timepoints" in " ".join(bad["problems"])
    ran = a.call("run_chain", {"chain": chain})
    assert set(ran["steps"]) == {"localization_statistics", "rcc"}
    assert not a.session.can_undo
    monkeypatch.setattr(plugins, "chains_dir", lambda: tmp_path / "chains")
    try:
        a.call("save_chain", {"chain": chain, "name": "Stats and drift"})
        assert "Analysis/Chains/Stats And Drift" in plugins.refs("Analysis/Chains")
    finally:
        monkeypatch.undo()
        plugins.discover(force=True)


def test_a_preset_fills_the_settings_and_the_values_given_beside_it_win(tmp_path):
    """As in the form: the plugin's react fills every setting from the preset.
    It used to be ignored, and Simulate made its demo scene instead."""
    a = Assistant(output=tmp_path / "out", knowledge_folder=tmp_path / "kb")
    a.call("run_plugin", {"path": "File/Simulate/Simulate", "apply": True,
                          "settings": {"preset": "Nuclear pores (Nup96)",
                                       "n_frames": 2000}})
    assert a.session.locs.metadata["structure"] == ["npc"]
    assert a.session.locs.metadata["n_frames"] == 2000


PORES_IN_BLOCK = """
steps:
  - plugin: Analysis/Regions/Segment regions
    values: {circle.shape: ring, circle.min_radius_nm: 55.0, circle.max_radius_nm: 55.0,
             keep: "n_ring >= 4"}
  - block: each pore
    steps:
      - plugin: Analysis/Measure/Circle fit
        values: {scale_nm: 20.0}
"""


def test_a_chain_with_a_block_is_validated_into_the_blocks_steps_and_runs(tmp_path):
    a = Assistant(output=tmp_path / "out", knowledge_folder=tmp_path / "kb")
    a.call("run_plugin", {"path": "File/Simulate/Simulate", "apply": True,
                          "settings": {"preset": "Nuclear pores (Nup96)",
                                       "n_frames": 3000}})
    assert a.call("validate_chain", {"chain": PORES_IN_BLOCK})["ok"]
    bad = a.call("validate_chain", {"chain": PORES_IN_BLOCK.replace("scale_nm", "scale")})
    assert not bad["ok"]
    assert "'each pore'" in " ".join(bad["problems"]) and "scale_nm" in " ".join(bad["problems"])
    ran = a.call("run_chain", {"chain": PORES_IN_BLOCK})
    assert set(ran["steps"]) == {"segment_regions", "each_pore"}
    assert ran["steps"]["each_pore"]["data"]["regions"] > 20


def test_describe_gives_the_fields_by_their_flat_names_and_what_the_form_says(tmp_path):
    a = simulated(tmp_path)
    described = a.call("describe_plugin", {"path": "Analysis/Drift/RCC"})
    assert "n_timepoints: int" in described["fields"]
    assert described["changes_table"] and described["reads"] == "ungrouped"
    assert isinstance(described["form"], dict)
    assert not a.call("describe_plugin", {"path": STATISTICS})["changes_table"]


def test_pages_and_source_are_read_by_path_or_by_dotted_name(tmp_path):
    a = Assistant(output=tmp_path)
    assert "## What it does" in a.call("read_doc", {"path": "Analysis/Drift/RCC"})
    assert "blink" in a.call("read_doc", {"path": "Panels/Glossary"})
    module = a.call("read_source", {"name": "Analysis/Drift/RCC"})
    assert module["file"].endswith("drift_rcc.py")
    function = a.call("read_source", {"name": "smappy.rcc.estimate_drift_rcc"})
    assert function["text"].startswith("def estimate_drift_rcc")
    long = a.call("read_source", {"name": "smappy.session"})
    assert long["next_offset"] == len(long["text"]) < long["length"]
    with pytest.raises(ToolError):
        a.call("read_source", {"name": "os.path"})


# ------------------------------------------------------------------ the index

def test_the_index_holds_every_plugin_and_finds_one_by_what_it_is_for():
    paths = {e.path for e in index.entries()}
    assert set(plugins.refs()) <= paths and "Panels/Glossary" in paths
    assert "Analysis/Drift/RCC" in [e["path"] for e in index.search("drift correction")[:4]]
    assert index.search("drift from beads")[0]["path"] == "Analysis/Drift/Fiducials"
    assert index.search("the and of") == []


def test_purpose_is_the_opening_of_what_it_does_without_markup():
    body = ("## What it does\n\nFinds [drift](https://x.org) in *time*.\n\n"
            "Second.\n\nThird.\n\n## How it works\n\nNo.")
    assert index.purpose(body) == "Finds drift in time. Second."


# -------------------------------------------------------------- the knowledge

def test_notes_are_kept_found_by_their_words_and_never_overwritten(tmp_path):
    first = knowledge.remember(tmp_path, "RCC windows", "Use 10 windows for 5 min.",
                               ["drift"])
    second = knowledge.remember(tmp_path, "RCC windows", "Or 20.")
    knowledge.remember(tmp_path, "Cluster radius", "DBSCAN at 30 nm.")
    assert first != second and first.read_text().count("Use 10") == 1
    found = knowledge.recall(tmp_path, "drift windows")
    assert {n["name"] for n in found[:2]} == {first.stem, second.stem}
    assert knowledge.recall(tmp_path, "dbscan")[0]["title"] == "Cluster radius"
    assert len(knowledge.recall(tmp_path)) == 3
    assert knowledge.forget(tmp_path, first.stem) == first and not first.exists()
    assert knowledge.forget(tmp_path, "../elsewhere") is None


# ---------------------------------------------------------------- the protocol

def _ask(server, method, params=None, ident=1):
    line = json.dumps({"jsonrpc": "2.0", "id": ident, "method": method,
                       "params": params or {}})
    return json.loads(server.handle_line(line))


def test_the_server_answers_initialize_list_and_call_as_mcp_says(tmp_path):
    server = Server(Assistant(output=tmp_path), instructions=INSTRUCTIONS)
    init = _ask(server, "initialize", {"protocolVersion": "2025-06-18",
                                       "clientInfo": {"name": "test"}})["result"]
    assert init["protocolVersion"] == "2025-06-18"
    assert "tools" in init["capabilities"] and init["instructions"]
    assert _ask(server, "initialize", {"protocolVersion": "1999-01-01"}
                )["result"]["protocolVersion"] == "2025-11-25"
    assert server.handle_line(json.dumps(
        {"jsonrpc": "2.0", "method": "notifications/initialized"})) is None
    listed = _ask(server, "tools/list")["result"]["tools"]
    # headless there is no view to move and no window: `show` and
    # `open_plugin` are the GUI's
    assert {t["name"] for t in listed} == set(Assistant.specs()) - {"show", "open_plugin"}
    called = _ask(server, "tools/call", {"name": "list_plugins",
                                         "arguments": {"prefix": "Analysis/Drift"}})
    assert not called["result"]["isError"]
    assert "Analysis/Drift/RCC" in called["result"]["content"][0]["text"]


def test_a_refused_call_is_a_tool_error_the_model_reads_not_a_protocol_error(tmp_path):
    server = Server(Assistant(output=tmp_path))
    answer = _ask(server, "tools/call", {"name": "read_doc",
                                         "arguments": {"path": "Analysis/Drift/RCX"}})
    assert answer["result"]["isError"]
    assert "no plugin" in answer["result"]["content"][0]["text"]
    assert _ask(server, "nonsense")["error"]["code"] == -32601
    assert json.loads(server.handle_line("{not json"))["error"]["code"] == -32700


def test_a_real_mcp_client_lists_the_tools_and_gets_a_picture(tmp_path):
    pytest.importorskip("mcp")
    anyio = pytest.importorskip("anyio")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def session():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "smappy.assistant.server", "--output", str(tmp_path),
                  "--knowledge", str(tmp_path / "kb")],
            env={**os.environ, "SMAPPY_CONFIG_DIR": str(tmp_path / "config")})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = await client.list_tools()
                await client.call_tool("run_plugin", {"path": "File/Simulate/Simulate",
                                                      "settings": SIMULATION,
                                                      "apply": True})
                picture = await client.call_tool("picture", {"pixelsize_nm": 40})
                return {t.name for t in tools.tools}, picture

    names, picture = anyio.run(session)
    assert names == set(Assistant.specs())
    assert not picture.is_error and picture.content[0].type == "image"


def test_a_missing_tool_is_noted_for_the_developers_and_goes_with_a_bug_report(tmp_path):
    import zipfile
    from smappy import diagnostics
    from smappy.assistant import knowledge
    a = Assistant(output=tmp_path, knowledge_folder=tmp_path / "kb")
    assert a.missing_tools == knowledge.missing_tools_file()     # in the config folder
    noted = a.call("note_missing_tool", {
        "asked": "open the analysis plugin",
        "workaround": "proposed a run, whose button opened it",
        "tool": "open a plugin's window filled in"})
    text = Path(noted["file"]).read_text()
    assert "## " in text and "open a plugin's window filled in" in text
    assert "- asked: open the analysis plugin" in text and "- SMAPpy " in text
    a.call("note_missing_tool", {"asked": "b", "workaround": "c", "tool": "d"})
    assert Path(noted["file"]).read_text().count("\n## ") == 2   # appended
    with zipfile.ZipFile(diagnostics.bug_report(tmp_path / "report.zip")) as z:
        assert "open the analysis plugin" in z.read("assistant-missing-tools.md").decode()
    Path(noted["file"]).unlink()
