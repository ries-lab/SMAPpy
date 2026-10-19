"""*Math parser* on regions (`smappy.region_maths`): expressions over what was
measured per region, kept as recipes in the source's ``maths`` table."""
import h5py
import numpy as np
import pytest

from smappy import plugins
from smappy.mathparse import ExpressionError
from smappy.region_maths import alias, aliases, evaluate
from smappy.region_results import Evaluation, is_current, rows
from smappy.region_table import RegionTable
from smappy.session import Session

from test_region_filter import clusters, overlapping_rois

MATH = "Analysis/Process/Math Parser"
IDSTATS = "Analysis/Measure/Id statistics"
EV = Evaluation("cluster_id", "maths")


def run(session, **values):
    cls = plugins.get(MATH)
    return session.run(cls(), cls.Settings(**values))


def values_of(session, name):
    return {r["roi_id"]: r[name] for r in rows(session.context(), EV)}


def test_an_evaluation_is_named_by_its_label_made_a_word():
    assert alias("NPC Corners") == "npc_corners"
    assert alias("Id statistics") == "id_statistics"
    assert alias("3D fit") == "e_3d_fit"


def test_an_expression_over_the_clusters_reads_their_values_by_alias_or_bare():
    session, _ = clusters()                        # 60, 40 and 25 localizations
    assert aliases(session, "cluster_id") == {"id_statistics": "Id statistics"}
    ids, got = evaluate(session, "cluster_id", "id_statistics.n_locs / median(n_locs)")
    assert ids == ["1", "2", "3"] and np.allclose(got, [1.5, 1.0, 0.625])
    with pytest.raises(ExpressionError, match="no value id_statistics.nope"):
        evaluate(session, "cluster_id", "id_statistics.nope + 1")


def test_the_result_is_a_current_region_table_and_a_later_recipe_reads_it():
    session, truth = clusters()
    run(session, source="cluster_id", field="rel",
        expression="id_statistics.n_locs / median(n_locs)")
    assert is_current(session.context(), EV)
    assert values_of(session, "rel") == {"1": 1.5, "2": 1.0, "3": 0.625}
    run(session, source="cluster_id", field="big", expression="maths.rel > 1",
        to_locs=True)
    assert values_of(session, "big") == {"1": 1.0, "2": 0.0, "3": 0.0}
    written = np.asarray(session.locs["big"])
    assert np.array_equal(written[truth == 1], np.ones((truth == 1).sum()))
    assert np.isnan(written[truth == 0]).all()                 # noise: no region
    table = session.id_regions["cluster_id"]["maths"]
    assert list(table.recipes) == ["rel", "big"]


def test_a_recipe_follows_its_inputs_measured_again():
    session, truth = clusters()
    run(session, source="cluster_id", field="half", expression="id_statistics.n_locs / 2")
    session.layers[0].set_bound("photons", 1000, None)
    assert not is_current(session.context(), EV)               # its inputs moved
    cls = plugins.get(IDSTATS)
    session.run(cls(), cls.Settings(id_column="cluster_id"))   # measured again
    assert is_current(session.context(), EV)
    n = values_of(session, "half")
    counted = {r["roi_id"]: r["n_locs"] for r in
               rows(session.context(), Evaluation("cluster_id", "Id statistics"))}
    assert n == {k: v / 2 for k, v in counted.items()}


def test_the_region_filter_reads_a_maths_value():
    from smappy.region_filter import Condition
    session, truth = clusters()
    run(session, source="cluster_id", field="rel",
        expression="id_statistics.n_locs / median(n_locs)")
    session.set_region_filter([Condition("cluster_id", "maths", "rel", lo=1.0)])
    assert np.array_equal(np.asarray(session.selection(0).mask, bool),
                          np.isin(truth, [1, 2]))


def test_on_rois_the_annotations_and_results_meet_by_roi():
    session, project, (a, b, c), _ = overlapping_rois()
    run(session, source="rois", field="per_nm",
        expression="made_up.corners / made_up.diameter_nm * annotations.use")
    table = project.tables["maths"]
    got = {r: table.entry(r)["values"]["per_nm"] for r in (a, b, c)}
    assert got == pytest.approx({a: 8 / 50, b: 3 / 110, c: 8 / 108})


def test_the_recipes_are_saved_with_their_table(tmp_path):
    table = RegionTable.from_columns("maths", [1, 2], {"x": np.array([1.0, 2.0])})
    table.recipes = {"x": "a.b * 2"}
    with h5py.File(tmp_path / "t.h5", "w") as f:
        table.to_hdf5(f.create_group("g"))
    with h5py.File(tmp_path / "t.h5") as f:
        assert RegionTable.from_hdf5(f["g"]).recipes == {"x": "a.b * 2"}
    assert RegionTable.from_dict(table.to_dict()).recipes == {"x": "a.b * 2"}


def test_on_regions_it_changes_the_table_only_when_asked_to():
    cls = plugins.get(MATH)
    plugin = cls()
    assert plugin.changes_locs_for(cls.Settings())                     # localizations
    assert not plugin.changes_locs_for(cls.Settings(source="cluster_id"))
    assert plugin.changes_locs_for(cls.Settings(source="cluster_id", to_locs=True))
    session, _ = clusters()
    result = run(session, source="cluster_id", field="rel", expression="n_locs * 1")
    assert result.locs is None and "rel" not in session.locs
