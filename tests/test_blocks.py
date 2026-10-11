"""Blocks: a chain's steps run once in each region, and what comes back."""
import numpy as np
import pytest

from smappy import chain, plugins
from smappy.chain import Block, ChainError, ChainSpec, Step, chain_class, step_from_dict
from smappy.locs import Localizations
from smappy.plugins import Plugin, Result, register
from smappy.session import Session


# test plugins, registered for these tests only

@register("Test/Block/Count")
class Count(Plugin):
    """What a region's table has: all rows, and the selected ones."""
    per_region = True
    changes_locs = False

    def run(self, ctx, settings):
        return Result(values={"n": len(ctx.selection), "all": len(ctx.locs)})


@register("Test/Block/Tag")
class Tag(Plugin):
    """Writes the region's size onto every row of it."""
    per_region = True
    changes_locs = True

    def run(self, ctx, settings):
        n = len(ctx.locs)
        return Result(locs=Localizations({**ctx.locs.columns,
                                          "n_here": np.full(n, float(n))},
                                         dict(ctx.locs.metadata)),
                      values={"tagged": n})


@register("Test/Block/Drop")
class Drop(Plugin):
    """Keeps the bright rows only."""
    per_region = True
    changes_locs = True

    def run(self, ctx, settings):
        return Result(locs=ctx.locs[np.asarray(ctx.locs["photons"]) >= 1000],
                      values={"kept": int((np.asarray(ctx.locs["photons"]) >= 1000).sum())})


@register("Test/Block/Halves")
class Halves(Plugin):
    """Splits a region in two by x: ``cluster_id`` 1 left, 2 right."""
    per_region = True
    changes_locs = True

    def run(self, ctx, settings):
        x = np.asarray(ctx.locs["x_nm"], dtype=float)
        ids = np.where(x < np.median(x), 1, 2)
        return Result(locs=Localizations({**ctx.locs.columns, "cluster_id": ids},
                                         dict(ctx.locs.metadata)))


def clusters(sizes=(30, 50, 20), noise=10, seed=0):
    rng = np.random.default_rng(seed)
    parts, ids = [], []
    for k, n in enumerate(sizes, start=1):
        parts.append(rng.normal(k * 5000, 30, (n, 2)))
        ids.append(np.full(n, k))
    parts.append(rng.uniform(0, 20000, (noise, 2)))
    ids.append(np.zeros(noise, int))
    xy = np.vstack(parts)
    m = len(xy)
    return Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1],
                          "frame": np.arange(m, dtype=np.int64),
                          "photons": rng.gamma(4, 400, m),
                          "xy_err_nm": rng.uniform(5, 15, m),
                          "cluster_id": np.concatenate(ids).astype(np.int64)}, {})


def a_session(**kw):
    session = Session(clusters(**kw))
    session.show_grouped(0, False)
    return session


def block(*steps, regions="cluster_id", label="each", **kw):
    return Block(steps=[s if isinstance(s, (Step, Block)) else Step(**s) for s in steps],
                 label=label, regions=regions, **kw)


def run(session, *steps):
    cls = chain_class(ChainSpec(steps=list(steps)))
    return session.run(cls(), cls.Settings()), cls


# ------------------------------------------------------------------ the file

def test_a_block_reads_back_from_its_file_as_it_was_written(tmp_path):
    written = ChainSpec(steps=[
        Step(plugin="Test/Block/Count", label="first"),
        block({"plugin": "Test/Block/Tag"},
              block({"plugin": "Test/Block/Count"}, label="inner"),
              write_back=["n_here"], merge="mean", keep_figures="no")])
    path = chain.write(written, tmp_path / "b.chain.yaml")
    back = chain.read(path)
    outer = back.steps[1]
    assert isinstance(outer, Block) and outer.title() == "each"
    assert outer.write_back == ["n_here"] and outer.merge == "mean"
    assert outer.keep_figures == "no" and outer.regions == "cluster_id"
    assert isinstance(outer.steps[1], Block) and outer.steps[1].steps[0].plugin == \
        "Test/Block/Count"
    assert back.to_dict() == written.to_dict()


def test_a_block_without_a_label_says_what_it_repeats_over():
    assert Block(regions="rois").title() == "for each ROI"
    assert Block(regions="cluster_id").title() == "for each cluster_id"
    assert isinstance(step_from_dict({"block": "", "steps": []}), Block)


def test_a_segmenter_cannot_run_in_a_block():
    with pytest.raises(ChainError, match="finds ROIs"):
        chain_class(ChainSpec(steps=[block({"plugin": "Analysis/Regions/Find/Density Peaks"})]))


def test_a_step_added_to_a_block_starts_with_its_slow_options_off():
    cls = chain_class(ChainSpec(steps=[
        block({"plugin": "Analysis/Measure/Localization Statistics"}),
        Step(plugin="Analysis/Measure/Localization Statistics", label="view")]))
    settings = cls.Settings()
    assert settings.each.localization_statistics.fit_precision is False
    assert settings.view.fit_precision is True          # on the view, as ever


# ------------------------------------------------------------------- running

def test_a_block_over_an_id_column_measures_each_cluster_and_keeps_the_numbers():
    session = a_session()
    result, cls = run(session, block({"plugin": "Test/Block/Count", "label": "count"}))
    assert result.data["results"]["each"].text == "3 cluster_id regions, 1 step(s)"
    table = session.id_regions["cluster_id"]["count"]
    assert table.columns()["all"].tolist() == [30, 50, 20]
    assert result.locs is None                          # nothing written back


def test_a_step_measures_what_the_step_before_it_left_in_the_region():
    session = a_session()
    run(session, block({"plugin": "Test/Block/Drop"}, {"plugin": "Test/Block/Count"}))
    photons = np.asarray(session.locs["photons"])
    ids = np.asarray(session.locs["cluster_id"])
    table = session.id_regions["cluster_id"]["Count"]
    assert table.columns()["all"].tolist() == [
        int(((ids == k) & (photons >= 1000)).sum()) for k in (1, 2, 3)]
    assert len(session.locs) == len(ids)                # the real table is whole


def test_a_written_back_column_lands_on_the_rows_it_came_from():
    session = a_session()
    result, _ = run(session, block({"plugin": "Test/Block/Tag"}, write_back=["n_here"]))
    n_here = np.asarray(session.locs["n_here"])
    ids = np.asarray(session.locs["cluster_id"])
    for k, n in zip((1, 2, 3), (30, 50, 20)):
        assert (n_here[ids == k] == n).all()
    assert np.isnan(n_here[ids == 0]).all()             # in no region: no value
    assert "wrote back n_here" in result.data["results"]["each"].text
    assert session.can_undo                             # one table change


def test_a_column_a_step_made_but_the_block_does_not_write_back_is_warned_about():
    session = a_session()
    said = []
    cls = chain_class(ChainSpec(steps=[block({"plugin": "Test/Block/Tag"})]))
    ctx = session.context(progress=said.append)
    cls().run(ctx, cls.Settings())
    assert any("n_here" in w and "does not write back" in w for w in ctx.warnings)
    assert "n_here" not in session.locs


def test_rows_a_region_dropped_are_flagged_when_use_is_written_back():
    session = a_session()
    run(session, block({"plugin": "Test/Block/Drop"}, write_back=["use"]))
    use = np.asarray(session.locs["use"])
    photons = np.asarray(session.locs["photons"])
    ids = np.asarray(session.locs["cluster_id"])
    inside = ids > 0
    assert (use[inside] == (photons[inside] >= 1000)).all()
    assert use[~inside].all()                           # in no region: kept
    assert len(session.locs) == len(ids)                # flagged, not deleted


def test_overlapping_rois_merge_by_the_rule():
    session = a_session(sizes=(40,), noise=0)
    project = session.rois
    file_id = next(iter(project.sources))
    project.set_geometry(400)
    big = project.add_roi(file_id, [5000.0, 5000.0])          # all 40
    project.set_geometry(400)
    small = project.add_roi(file_id, [5000.0, 5000.0],
                            polygon=[[4990, 4900], [5100, 4900], [5100, 5100],
                                     [4990, 5100]])           # part of them
    x = np.asarray(session.locs["x_nm"])
    y = np.asarray(session.locs["y_nm"])
    in_small = project.region(small).mask(x, y)
    n_small = int(in_small.sum())
    assert 0 < n_small < 40
    for rule, expect_small in (("last", n_small), ("first", 40),
                               ("mean", (40 + n_small) / 2)):
        session.locs = Localizations({k: v for k, v in session.locs.columns.items()
                                      if k != "n_here"}, dict(session.locs.metadata))
        run(session, block({"plugin": "Test/Block/Tag"}, regions="rois",
                           write_back=["n_here"], merge=rule))
        n_here = np.asarray(session.locs["n_here"])
        assert np.allclose(n_here[in_small], expect_small)
        assert np.allclose(n_here[~in_small], 40)


def test_a_block_in_a_block_makes_a_table_of_the_inner_regions():
    session = a_session(sizes=(40, 60), noise=0)
    project = session.rois
    file_id = next(iter(project.sources))
    project.set_geometry(400)
    a = project.add_roi(file_id, [5000.0, 5000.0])
    b = project.add_roi(file_id, [10000.0, 10000.0])
    result, _ = run(session, block({"plugin": "Test/Block/Halves"},
                                   block({"plugin": "Test/Block/Count"}, label="halves"),
                                   regions="rois"))
    inner = project.tables["halves/Count"]
    assert sorted(inner.ids) == sorted([f"{a.id}:1", f"{a.id}:2", f"{b.id}:1", f"{b.id}:2"])
    counts = {r: inner.entry(r)["values"]["all"] for r in inner.ids}
    assert counts[f"{a.id}:1"] + counts[f"{a.id}:2"] == 40
    assert counts[f"{b.id}:1"] + counts[f"{b.id}:2"] == 60
    assert "cluster_id" not in session.locs or session.locs["cluster_id"].max() == 2


def test_the_rois_block_reports_its_status_and_stops_when_asked():
    import threading
    session = a_session(sizes=(20,) * 5, noise=0)
    project = session.rois
    file_id = next(iter(project.sources))
    for k in range(1, 6):
        project.add_roi(file_id, [k * 5000.0, k * 5000.0])
    said = []
    stop = threading.Event()
    cls = chain_class(ChainSpec(steps=[block({"plugin": "Test/Block/Count"},
                                             regions="rois")]))
    # the first status line comes after the first ROI; stop there
    ctx = session.context(progress=lambda text: (said.append(text),
                                                 "of 5 ROIs" in text and stop.set()))
    ctx.stop = stop
    result = cls().run(ctx, cls.Settings())
    sub = result.data["results"]["each"]
    assert sub.data["regions"] == 1 and any("(1 of 5 ROIs)" in t for t in said)


# --------------------------------------------------------------- the loops

def test_a_chain_that_contains_itself_is_refused_naming_the_loop():
    one = plugins.chains_dir() / "loop_one.chain.yaml"
    two = plugins.chains_dir() / "loop_two.chain.yaml"
    chain.write(ChainSpec(steps=[Step(plugin="Analysis/Chains/Loop Two")]), one)
    chain.write(ChainSpec(steps=[block({"plugin": "Analysis/Chains/Loop One"})]), two)
    try:
        plugins.discover(force=True)
        with pytest.raises(ChainError, match="cannot contain itself: Loop One -> "
                                             "Loop Two -> Loop One"):
            plugins.get("Analysis/Chains/Loop One")
    finally:
        one.unlink()
        two.unlink()
        plugins.discover(force=True)


def test_a_chain_can_be_a_step_in_a_block():
    saved = ChainSpec(steps=[Step(plugin="Test/Block/Count", label="n")])
    target = plugins.chains_dir() / "region_count.chain.yaml"
    chain.write(saved, target)
    try:
        plugins.discover(force=True)
        session = a_session()
        result, _ = run(session, block({"plugin": "Analysis/Chains/Region Count",
                                        "label": "inner chain"}))
        assert result.data["results"]["each"].text.startswith("3 cluster_id regions")
    finally:
        target.unlink()
        plugins.discover(force=True)


# ------------------------------------------------------------------ timing

@register("Test/Block/Slow")
class Slow(Plugin):
    """Takes its time, more of it with its slow option on."""
    per_region = True
    changes_locs = False

    from dataclasses import dataclass as _dc

    @_dc
    class Settings:
        from smappy.plugins import param as _p
        thorough: bool = _p(True, label="thorough", slow=True, help="slower")

    def run(self, ctx, settings):
        import time
        time.sleep(0.02 if settings.thorough else 0.002)
        return Result(values={"n": len(ctx.locs)})


def test_a_blocks_time_is_estimated_from_a_few_of_its_regions():
    from smappy.blocks import block_class
    session = a_session(sizes=(10,) * 40, noise=0)
    cls = block_class(block({"plugin": "Test/Block/Slow", "values": {"thorough": True}}))
    estimate = cls().estimate(session.context(), cls.Settings())
    assert estimate.regions == 40 and estimate.sampled == 5
    assert 0.015 < estimate.per_region["Slow"] < 0.1
    assert 0.6 < estimate.total < 4
    assert "over 40 regions" in estimate.text() and "timed on 5" in estimate.text()


def test_a_long_block_asks_first_and_offers_its_slow_options_off(monkeypatch):
    import smappy.blocks as blocks
    monkeypatch.setattr(blocks, "ASK_ABOVE_SECONDS", 0.1)
    session = a_session(sizes=(10,) * 20, noise=0)
    cls = chain_class(ChainSpec(steps=[block({"plugin": "Test/Block/Slow",
                                              "values": {"thorough": True}})]))
    settings = cls.Settings()
    asked = cls().preflight(session.context(), settings)
    assert "each will take" in asked.text and "over 20 regions" in asked.text
    choice, = asked.choices
    assert choice.label == "Slow options off"
    assert choice.settings.each.slow.thorough is False
    assert settings.each.slow.thorough is True          # the form is untouched
    monkeypatch.setattr(blocks, "ASK_ABOVE_SECONDS", 1e9)
    assert cls().preflight(session.context(), settings) is None
