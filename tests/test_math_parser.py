"""The math parser: what an expression means, and what a derived field is."""
import numpy as np
import pytest

from smappy.group import GroupSettings, group
from smappy.locs import Localizations
from smappy.mathparse import (ExpressionError, apply_recipes, evaluate,
                              names_in, recipes, remember)
from smappy.plugins import Context, Selection
from smappy.plugins.math_parser import (MathParser, MathSettings, check_field,
                                        history, remember_expression)


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """A settings directory of this test's own, so the history is not the
    developer's real one."""
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    return tmp_path / "config"


def table(n=12):
    """Three emitters, four frames each: grouping links them into three."""
    return Localizations({
        "x_nm": np.repeat([100.0, 500.0, 900.0], n // 3).astype(np.float32),
        "y_nm": np.zeros(n, np.float32),
        "frame": np.arange(n, dtype=np.int64) % (n // 3),
        "photons": np.linspace(100, 1200, n).astype(np.float32),
        "xy_err_nm": np.full(n, 10.0, np.float32),
    }, {"units": "nm"})


# ------------------------------------------------------------- the numbers

def test_an_expression_is_arithmetic_over_the_columns():
    locs = table()
    assert np.allclose(evaluate(locs, "photons * 2 + 1"),
                       np.asarray(locs["photons"]) * 2 + 1)
    assert np.allclose(evaluate(locs, "sqrt(photons) / 2"),
                       np.sqrt(locs["photons"]) / 2)
    # a reduction is one number over the column, and MATLAB gives it for free
    assert np.allclose(evaluate(locs, "photons - median(photons)"),
                       np.asarray(locs["photons"]) - np.median(locs["photons"]))


def test_the_rounding_and_modulus_functions_are_there():
    locs = Localizations({"v": np.array([-1.6, 0.4, 2.5, 7.0], np.float32)})
    assert np.allclose(evaluate(locs, "floor(v)"), [-2, 0, 2, 7])
    assert np.allclose(evaluate(locs, "ceil(v)"), [-1, 1, 3, 7])
    assert np.allclose(evaluate(locs, "round(v)"), np.round([-1.6, 0.4, 2.5, 7.0]))
    assert np.allclose(evaluate(locs, "mod(v, 2)"), np.mod([-1.6, 0.4, 2.5, 7.0], 2))
    assert np.allclose(evaluate(locs, "v % 2"), np.mod([-1.6, 0.4, 2.5, 7.0], 2))


def test_a_flag_is_a_column_of_zeros_and_ones():
    """Everything downstream treats a column as numbers; a bool is not one."""
    locs = table()
    flag = evaluate(locs, "(photons > 500) & (xy_err_nm < 25)")
    assert flag.dtype == np.float32
    assert np.array_equal(flag, (np.asarray(locs["photons"]) > 500).astype(np.float32))


def test_one_number_becomes_a_column():
    locs = table()
    assert np.array_equal(evaluate(locs, "0"), np.zeros(len(locs)))
    assert len(evaluate(locs, "median(photons)")) == len(locs)


def test_names_in_are_the_columns_and_not_the_functions():
    assert names_in("(xy_err_nm < 25) & (sigma_nm > 100)") == \
        {"xy_err_nm", "sigma_nm"}
    assert names_in("round(x_nm) + pi") == {"x_nm"}


# -------------------------------------------------------------- the refusals

def test_the_numpy_traps_are_named_rather_than_answered_wrongly():
    locs = table()
    with pytest.raises(ExpressionError, match="bind tighter"):
        evaluate(locs, "photons < 500 & x_nm > 100")
    with pytest.raises(ExpressionError, match="chained comparison"):
        evaluate(locs, "100 < photons < 500")
    with pytest.raises(ExpressionError, match="and.*or"):
        evaluate(locs, "(photons > 1) and (x_nm > 1)")
    # the same thing written properly is not refused
    assert evaluate(locs, "(photons < 500) & (x_nm >= 100)").sum() > 0


def test_a_flag_column_has_to_be_compared_before_it_is_combined():
    """Every column is numbers, flags included, so `&` on one is a mistake --
    and one with an obvious fix, which is what the message gives."""
    locs = table()
    locs.columns["bright"] = (np.asarray(locs["photons"]) > 500).astype(np.float32)
    locs.columns["near"] = (np.asarray(locs["x_nm"]) < 600).astype(np.float32)
    with pytest.raises(ExpressionError, match=r"flag column is 0/1"):
        evaluate(locs, "bright & near")
    assert np.array_equal(evaluate(locs, "(bright > 0) & (near > 0)"),
                          [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])


def test_a_missing_column_says_what_the_table_has():
    with pytest.raises(ExpressionError, match="no column `z_nm`.*photons"):
        evaluate(table(), "z_nm * 2")


def test_an_expression_cannot_reach_out_of_the_arithmetic():
    """An expression travels in a workspace file, so it is not a program."""
    locs = table()
    for attack in ("__import__('os').system('true')",
                   "photons.__class__",
                   "open('/etc/passwd')",
                   "[x for x in photons]",
                   "photons[0]",
                   "(lambda: 1)()"):
        with pytest.raises(ExpressionError):
            evaluate(locs, attack)


def test_a_field_name_has_to_be_usable_in_a_later_expression():
    locs = table()
    assert check_field("  within ", locs) == "within"
    with pytest.raises(ValueError, match="letters, digits"):
        check_field("two words", locs)
    with pytest.raises(ValueError, match="function"):
        check_field("mean", locs)
    with pytest.raises(ValueError, match="grouping"):
        check_field("n_in_group", locs)


# --------------------------------------------------------------- the plugin

def test_the_field_is_written_and_the_input_table_is_left_alone(config_dir):
    """`set_locs` keeps the old table as the undo, so it must stay as it was."""
    locs = table()
    result = MathParser()(locs, settings=MathSettings(
        field="bright", expression="photons > 500"))
    assert "bright" not in locs
    assert np.array_equal(result.locs["bright"],
                          (np.asarray(locs["photons"]) > 500).astype(np.float32))
    assert result.locs["photons"] is locs["photons"]      # the arrays are shared
    assert "bright" in result.text


def test_a_selection_leaves_the_rest_of_the_field_empty(config_dir):
    locs = table()
    mask = np.zeros(len(locs), bool)
    mask[:4] = True
    ctx = Context(locs=locs, selection=Selection(mask))
    result = MathParser().run(ctx, MathSettings(field="v", expression="photons * 2",
                                                where="selection"))
    values = np.asarray(result.locs["v"])
    assert np.allclose(values[:4], np.asarray(locs["photons"])[:4] * 2)
    assert np.isnan(values[4:]).all()
    # the expression alone does not define the field any more, so nothing
    # recomputes it -- but the grouped table can still reduce what is there
    recipe = recipes(result.locs)[0]
    assert recipe["where"] == "selection" and recipe["grouped"] == "mean"
    result.locs.columns["v"] = np.zeros(len(locs), np.float32)
    assert apply_recipes(result.locs) == []


def test_the_preview_says_what_would_be_written_without_writing_it(config_dir):
    locs = table()
    result = MathParser().preview(Context(locs=locs),
                                  MathSettings(field="v", expression="photons / 2"))
    assert "would write" in result.text and result.locs is None


# ------------------------------------------------------- grouping and recipes

def test_grouping_writes_the_group_id_and_the_on_time_back():
    locs = table()
    grouped, index = group(locs, GroupSettings(dx=50.0, dt=1))
    assert np.array_equal(locs["group_id"], index)
    assert np.array_equal(grouped["group_id"], np.arange(1, len(grouped) + 1))
    # the on-time of the group each localization went into
    assert np.array_equal(locs["n_in_group"],
                          np.asarray(grouped["n_in_group"])[index - 1])
    # and the ids are what a later combination needs: no relinking
    assert np.allclose(np.bincount(locs["group_id"])[1:], grouped["n_in_group"])


def test_a_recomputed_field_means_the_same_thing_on_both_tables():
    """`n_in_group * 20` is a statement about a blink, not a per-localization
    number to be averaged."""
    locs = table()
    group(locs, GroupSettings(dx=50.0, dt=1))          # gives it `n_in_group`
    remember(locs, "on_time_ms", "n_in_group * 20", grouped="recompute")
    apply_recipes(locs)
    assert np.allclose(locs["on_time_ms"], 80)
    grouped, _ = group(locs, GroupSettings(dx=50.0, dt=1))
    assert np.allclose(grouped["on_time_ms"], 80)


def test_a_combined_field_is_reduced_by_the_rule_that_was_chosen():
    locs = table()
    values = np.asarray(locs["photons"], np.float64)
    for rule, expected in (("sum", [values[:4].sum(), values[4:8].sum(), values[8:].sum()]),
                           ("mean", [values[:4].mean(), values[4:8].mean(), values[8:].mean()]),
                           ("max", [values[:4].max(), values[4:8].max(), values[8:].max()]),
                           ("min", [values[:4].min(), values[4:8].min(), values[8:].min()])):
        locs.columns["v"] = values.astype(np.float32)
        locs.metadata.pop("derived", None)
        remember(locs, "v", "photons", grouped=rule)
        grouped, _ = group(locs, GroupSettings(dx=50.0, dt=1))
        assert np.allclose(grouped["v"], expected, rtol=1e-5), rule


def test_a_flag_can_be_combined_as_any_or_all():
    locs = table()
    locs.columns["bright"] = (np.asarray(locs["photons"]) > 500).astype(np.float32)
    remember(locs, "bright", "photons > 500", grouped="any")
    grouped, _ = group(locs, GroupSettings(dx=50.0, dt=1))
    assert np.array_equal(grouped["bright"], [0, 1, 1])
    remember(locs, "bright", "photons > 500", grouped="all")
    grouped, _ = group(locs, GroupSettings(dx=50.0, dt=1))
    assert np.array_equal(grouped["bright"], [0, 0, 1])


def test_a_field_can_be_left_off_the_grouped_table():
    locs = table()
    locs.columns["v"] = np.ones(len(locs), np.float32)
    remember(locs, "v", "0 * photons + 1", grouped="none")
    grouped, _ = group(locs, GroupSettings(dx=50.0, dt=1))
    assert "v" not in grouped


def test_a_recipe_that_cannot_run_yet_is_skipped_not_raised():
    """`n_in_group` exists only once something has been linked."""
    locs = table()
    remember(locs, "on_time_ms", "n_in_group * 20")
    said = []
    assert apply_recipes(locs, report=said.append) == []
    assert "on_time_ms" not in locs and "n_in_group" in said[0]
    # the first grouping defines it on both tables
    grouped, _ = group(locs, GroupSettings(dx=50.0, dt=1))
    assert np.allclose(grouped["on_time_ms"], 80)
    assert np.allclose(locs["on_time_ms"], 80)


def test_a_recipe_survives_a_save_and_a_reload(tmp_path):
    from smappy.io import formats
    locs = table()
    remember(locs, "double", "photons * 2")
    apply_recipes(locs)
    path = formats.save(locs, tmp_path / "t.h5", metadata=locs.metadata)
    back, _ = formats.load(path)
    assert recipes(back)[0]["expression"] == "photons * 2"
    assert np.allclose(back["double"], np.asarray(locs["photons"]) * 2)


def test_a_redefinition_keeps_its_place_in_the_order():
    locs = table()
    remember(locs, "a", "photons")
    remember(locs, "b", "a * 2")
    remember(locs, "a", "photons * 10")
    assert [r["field"] for r in recipes(locs)] == ["a", "b"]
    apply_recipes(locs)
    assert np.allclose(locs["b"], np.asarray(locs["photons"]) * 20)


# --------------------------------------------------------------- the history

def test_the_history_keeps_the_last_expressions_most_recent_first(config_dir):
    remember_expression("a", "photons * 2")
    remember_expression("b", "x_nm + 1")
    assert [(e["field"], e["expression"]) for e in history()] == [
        ("b", "x_nm + 1"), ("a", "photons * 2")]
    remember_expression("a", "photons * 2")         # used again: it moves up
    assert [e["field"] for e in history()] == ["a", "b"]


def test_the_history_is_capped_and_a_broken_file_is_an_empty_one(config_dir):
    from smappy.plugins.math_parser import MAX_HISTORY, history_file
    for i in range(MAX_HISTORY + 5):
        remember_expression(f"f{i}", f"photons * {i}")
    assert len(history()) == MAX_HISTORY
    history_file().write_text("{[not yaml")
    assert history() == []


def test_choosing_from_the_history_fills_the_fields(config_dir):
    remember_expression("on_time_ms", "n_in_group * 20", "recompute")
    plugin = MathParser()
    label = "on_time_ms = n_in_group * 20"
    assert plugin.react("recall", MathSettings(recall=label)) == {
        "field": "on_time_ms", "expression": "n_in_group * 20",
        "grouped": "recompute"}
    assert plugin.react("field", MathSettings(recall=label)) is None


# ------------------------------------------------- the log the file carries

def test_what_a_plugin_did_is_logged_with_its_settings_and_survives_a_save(
        tmp_path, config_dir):
    """Provenance: a saved table says what was done to it, and reopening it
    continues that record rather than starting a new one."""
    from smappy.session import Session

    session = Session(table())
    session.run(MathParser(), MathSettings(field="double", expression="photons * 2"))
    entry = session.history[-1]
    assert entry["what"] == "Analysis/Process/Math Parser"
    assert entry["settings"]["expression"] == "photons * 2"

    path = session.save(tmp_path / "t.h5")
    reopened = Session()
    reopened.load(path)
    done = [e for e in reopened.history if e["what"] == "Analysis/Process/Math Parser"]
    assert done and done[-1]["settings"]["field"] == "double"
    assert reopened.history[-1]["what"] == "load"       # and the log goes on

    # a second round trip keeps the first run rather than overwriting it
    reopened.run(MathParser(), MathSettings(field="half", expression="photons / 2"))
    again = Session()
    again.load(reopened.save(tmp_path / "t2.h5"))
    fields = [e["settings"]["field"] for e in again.history
              if e["what"] == "Analysis/Process/Math Parser"]
    assert fields == ["double", "half"]
