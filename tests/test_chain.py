"""A chain of plugins runs as one plugin: its file, its settings, its run,
its record, and the rules that decide which table each step reads."""
import numpy as np
import pytest

from smappy import chain, plugins
from smappy.chain import ChainError, ChainSkipped, ChainSpec, Step, chain_class
from smappy.locs import Localizations
from smappy.plugins import Context, Plugin, Result, register
from smappy.session import Session

from test_batch_foundations import blinks


def with_sigma(locs):
    columns = dict(locs.columns)
    columns["sigma_nm"] = np.full(len(locs), 120.0)
    return Localizations(columns, dict(locs.metadata))


# Plugins that report what they were handed, registered for these tests only.

@register("Test/Chain/Count")
class Count(Plugin):
    """How many rows the table it reads has: grouped or not."""
    def run(self, ctx, settings):
        locs, sel = ctx.table()
        return Result(text=f"{len(locs)} rows, {len(sel)} selected",
                      data={"rows": len(locs), "selected": len(sel)})


@register("Test/Chain/Count Grouped")
class CountGrouped(Count):
    grouping = "grouped"


@register("Test/Chain/Rewrite")
class Rewrite(Plugin):
    """Hands back a table from whatever it read -- wrong on a grouped one."""
    def run(self, ctx, settings):
        locs, _ = ctx.table()
        return Result(locs=locs[: len(locs) // 2], text="halved")


@register("Test/Chain/Expensive")
class Expensive(Plugin):
    def preflight(self, ctx, settings):
        return "this will take an afternoon"

    def run(self, ctx, settings):
        return Result(text="ran")


def spec(*steps, grouping="layer"):
    return ChainSpec(steps=[s if isinstance(s, Step) else Step(**s) for s in steps],
                     grouping=grouping)


def test_a_chain_file_reads_back_as_it_was_written(tmp_path):
    written = spec({"plugin": "Chain/Layers", "label": "filter",
                    "values": {"remove": True}, "version": "1"},
                   {"plugin": "Analysis/Drift/RCC", "grouping": "ungrouped",
                    "enabled": False}, grouping="grouped")
    written.path = "Analysis/Chains/Mine"
    path = chain.write(written, tmp_path / "mine.chain.yaml")
    back = chain.read(path)
    assert back.to_dict() == written.to_dict()
    assert back.keys() == ["filter", "rcc"]


def test_a_chain_is_a_plugin_whose_settings_are_its_steps():
    cls = chain_class(spec({"plugin": "Analysis/Process/Math Parser", "label": "flag",
                            "values": {"field": "good", "expression": "photons > 10"}},
                           {"plugin": "Analysis/Measure/Localization Statistics",
                            "values": {"bins": 40}}))
    settings = cls.Settings()
    assert settings.flag.field == "good" and settings.flag.expression == "photons > 10"
    assert settings.localization_statistics.bins == 40
    assert settings.flag.use_grouping == "auto" and settings.grouping == "layer"
    specs = cls.specs()
    assert specs["flag"].info.collapsed and specs["flag"].children["expression"].info.kind == "text"
    # what Save chain writes is what the form holds
    settings.flag.expression = "photons > 20"
    saved = cls.current_spec(settings)
    assert saved.steps[0].values["expression"] == "photons > 20"
    from smappy.plugins.statistics import LocalizationStatistics
    assert saved.steps[1].version == LocalizationStatistics.version   # recorded as it runs


def test_a_chain_naming_a_missing_plugin_says_which():
    with pytest.raises(ChainError, match="Analysis/Nope.*not installed"):
        chain_class(spec({"plugin": "Analysis/Nope", "label": "gone"}))


def test_a_chain_runs_its_steps_in_order_as_one_undo_step_and_one_log_entry():
    session = Session(with_sigma(blinks()))
    before = len(session.history)
    cls = chain_class(spec(
        {"plugin": "Chain/Layers", "label": "filter",
         "values": {"layers": [{"grouped": False, "start": "empty",
                                "bounds": [{"field": "xy_err_nm", "hi": 20}]}],
                    "remove": True}},
        {"plugin": "Analysis/Process/Math Parser", "label": "flag",
         "values": {"field": "bright", "expression": "photons > 500"}},
        {"plugin": "Test/Chain/Count"}))
    result = session.run(cls(), cls.Settings())
    kept = int(np.sum(blinks()["xy_err_nm"] <= 20))
    assert len(session.locs) == kept and "bright" in session.locs
    assert result.data["results"]["count"].data["rows"] == kept
    # the layer is as the step set it up, which is what Render then shows
    assert not session.layers[0].grouped
    assert session.layers[0].filter.ranges == {"xy_err_nm": (None, 20.0)}
    # one step back undoes the whole chain
    assert session.undo_stack.can_undo and len(session.undo_stack) == 1
    session.undo()
    assert len(session.locs) == 160
    # and the log has one entry for it, with every step in it
    entries = session.history[before:]
    assert [e["what"] for e in entries if e["what"] != "undo"] == [cls.path]
    steps = entries[0]["steps"]
    assert [s["plugin"] for s in steps] == ["Chain/Layers", "Analysis/Process/Math Parser",
                                            "Test/Chain/Count"]
    assert [s["changed"] for s in steps] == [True, True, False]


def test_the_grouping_is_the_layers_then_the_chains_then_the_steps_then_the_plugins():
    def rows(chain_grouping, step_grouping, plugin="Test/Chain/Count"):
        session = Session(blinks())           # the layer shows grouped
        cls = chain_class(spec({"plugin": plugin, "label": "n",
                                "grouping": step_grouping}, grouping=chain_grouping))
        return session.run(cls(), cls.Settings()).data["results"]["n"].data["rows"]

    assert rows("layer", "auto") == 40                      # the layer's own
    assert rows("ungrouped", "auto") == 160                 # the chain's
    assert rows("ungrouped", "grouped") == 40               # the step's
    assert rows("ungrouped", "ungrouped", "Test/Chain/Count Grouped") == 40  # the plugin's


def test_a_step_that_rewrites_the_grouped_table_is_refused():
    session = Session(blinks())
    cls = chain_class(spec({"plugin": "Test/Chain/Rewrite", "grouping": "grouped"}))
    with pytest.raises(ChainError, match="grouped table"):
        session.run(cls(), cls.Settings())
    ungrouped = chain_class(spec({"plugin": "Test/Chain/Rewrite", "grouping": "ungrouped"}))
    session.run(ungrouped(), ungrouped.Settings())
    assert len(session.locs) == 80


def test_a_preflight_question_skips_the_run_when_nobody_can_answer():
    cls = chain_class(spec({"plugin": "Test/Chain/Expensive"}))
    runner = cls()
    assert "afternoon" in runner.preflight(Context(locs=blinks()), cls.Settings())
    runner.preflight_policy = "skip"
    with pytest.raises(ChainSkipped, match="afternoon"):
        runner.run(Context(locs=blinks()), cls.Settings())
    runner.preflight_policy = "proceed"
    assert runner.run(Context(locs=blinks()), cls.Settings()).text == "Expensive: ran"


def test_a_failing_step_names_itself():
    session = Session(blinks())
    cls = chain_class(spec({"plugin": "Analysis/Process/Math Parser", "label": "bad",
                            "values": {"expression": "no_such_column > 1"}}))
    with pytest.raises(ChainError, match="step 'bad'"):
        session.run(cls(), cls.Settings())
    assert len(session.locs) == 160 and not session.undo_stack.can_undo


def test_the_layers_step_resolves_quantiles_and_skips_what_the_file_lacks():
    from smappy.plugins.chain_layers import resolve_layer
    locs = blinks()
    bounds, notes = resolve_layer(locs, {"start": "empty", "bounds": [
        {"field": "xy_err_nm", "lo": 0.25, "hi": 0.75, "quantile": True},
        {"field": "z_nm", "lo": -300, "hi": 300}]})
    lo, hi = bounds["xy_err_nm"]
    assert lo == pytest.approx(np.quantile(locs["xy_err_nm"], 0.25))
    assert hi == pytest.approx(np.quantile(locs["xy_err_nm"], 0.75))
    assert "z_nm" not in bounds and "z_nm" in notes[0]
    with pytest.raises(ValueError, match="needs 'z_nm'.*it has"):
        resolve_layer(locs, {"bounds": [{"field": "z_nm", "hi": 1, "required": True}]})
    # "defaults" starts from what a freshly opened file gets
    assert resolve_layer(locs, {"start": "defaults"})[0] == {"xy_err_nm": (None, 25.0)}


def test_a_second_layer_named_by_the_step_is_made():
    session = Session(blinks())
    cls = chain_class(spec({"plugin": "Chain/Layers", "values": {"layers": [
        {"grouped": True, "start": "empty"},
        {"grouped": False, "start": "empty",
         "bounds": [{"field": "photons", "lo": 100}]}]}}))
    session.run(cls(), cls.Settings())
    assert len(session.layers) == 2
    assert session.layers[0].grouped and not session.layers[1].grouped
    assert session.layers[1].filter.ranges == {"photons": (100.0, None)}


def test_discovery_finds_a_saved_chain_and_opens_it_as_a_plugin(tmp_path):
    saved = spec({"plugin": "Test/Chain/Count", "label": "n"})
    target = plugins.chains_dir() / "my_count.chain.yaml"
    chain.write(saved, target)
    try:
        plugins.discover(force=True)
        ref = plugins.refs()["Analysis/Chains/My Count"]
        assert ref.kind == "chain" and ref.description == "n"
        cls = plugins.get("Analysis/Chains/My Count")
        assert cls.path == "Analysis/Chains/My Count"
        result = cls()(blinks())
        assert result.data["results"]["n"].data["rows"] == 40
    finally:
        target.unlink()
        plugins.discover(force=True)


def test_a_chain_keeps_each_steps_result_and_draws_it_again():
    session = Session(blinks())
    cls = chain_class(spec({"plugin": "Analysis/Measure/Localization Statistics",
                            "label": "stats", "values": {"source": "all"}}))
    result = session.run(cls(), cls.Settings())
    assert any(name.startswith("stats") for name in result.plots)


def test_the_history_shows_a_chain_as_its_steps():
    from smappy.plugins.history import as_text, entries
    session = Session(with_sigma(blinks()))
    cls = chain_class(spec(
        {"plugin": "Analysis/Process/Math Parser", "label": "flag",
         "values": {"field": "bright", "expression": "photons > 500"}},
        {"plugin": "Test/Chain/Count"}))
    session.run(cls(), cls.Settings())
    text = as_text(entries(session))
    assert "* flag (Analysis/Process/Math Parser)" in text
    assert "expression = photons > 500" in text
    assert "  Count (Test/Chain/Count): 40 rows" in text   # the layer shows grouped
