"""A layer coloured by what was measured per region (`smappy.region_colour`):
the values drawn as a column of the layer's snapshot, never of the table."""
import dataclasses

import numpy as np
import pytest

from smappy.region_colour import name, numeric_values, parse, region_values_onto
from smappy.region_filter import ANNOTATIONS
from smappy.render import FieldOfView

from test_region_filter import clusters, overlapping_rois

N_LOCS = "per cluster_id › Id statistics › n_locs"


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from smappy.gui import make_app
    return make_app()


def colour_by(session, field, layer=0, **more):
    state = session.layers[layer].state
    state.settings = dataclasses.replace(state.settings, color_field=field, **more)
    return state


def test_a_name_reads_back_and_no_column_is_taken_for_one():
    assert name("cluster_id", "Id statistics", "n_locs") == N_LOCS
    assert parse(N_LOCS) == ("cluster_id", "Id statistics", "n_locs")
    assert parse(name("rois", "NPC Corners", "n_corners")) == ("rois", "NPC Corners",
                                                                "n_corners")
    assert parse("photons") is None and parse(None) is None


def test_each_localization_takes_its_clusters_value_and_noise_has_none():
    session, truth = clusters()
    values = region_values_onto(session, "cluster_id", "Id statistics", "n_locs")
    for k, n in ((1, 60), (2, 40), (3, 25)):
        assert np.all(values[truth == k] == n)
    assert np.isnan(values[truth == -1]).all()


def test_the_snapshot_has_the_column_and_the_table_does_not():
    session, truth = clusters()
    state = colour_by(session, N_LOCS)
    snap = state.snapshot()
    assert N_LOCS in snap.locs and N_LOCS not in session.locs
    assert np.array_equal(np.asarray(snap.locs[N_LOCS])[truth == 1], np.full(60, 60.0))
    rendered = snap.render(FieldOfView(0.0, 0.0, 100.0, 120, 120))
    assert rendered.color is not None                   # drawn coloured, not plain


def test_hide_no_value_leaves_out_the_localizations_in_no_region():
    session, truth = clusters()
    snap = colour_by(session, N_LOCS, hide_missing=True).snapshot()
    assert np.array_equal(np.asarray(snap.mask, bool), truth >= 0)
    snap = colour_by(session, N_LOCS, hide_missing=False).snapshot()
    assert np.asarray(snap.mask, bool).all()              # grey, but there


def test_a_merged_blink_takes_its_clusters_value():
    session, truth = clusters()
    session.show_grouped(0, True)
    state = colour_by(session, N_LOCS)
    snap = state.snapshot()
    grouped = snap.locs
    ids = np.asarray(grouped["cluster_id"])
    values = np.asarray(grouped[N_LOCS])
    assert np.all(values[ids == 1] == 60) and np.isnan(values[ids == -1]).all()


def test_overlapping_rois_give_their_mean_and_an_unused_roi_gives_nothing():
    session, project, (a, b, c), xy = overlapping_rois()
    in_a, in_b, in_c = (np.isin(np.arange(len(xy)), project.session_rows(project.rois[r]))
                        for r in (a, b, c))
    values = region_values_onto(session, "rois", "Made up", "diameter_nm")
    assert np.all(values[in_a & ~in_b] == 50)
    assert np.all(values[in_a & in_b] == 80)            # (50 + 110) / 2
    assert np.all(values[in_c] == 108)
    assert np.isnan(values[~(in_a | in_b | in_c)]).all()
    project.rois[b].use = False
    values = region_values_onto(session, "rois", "Made up", "diameter_nm")
    assert np.all(values[in_a] == 50)


def test_measured_again_the_colours_follow_without_a_click():
    session, project, (a, b, c), xy = overlapping_rois()
    field = name("rois", "Made up", "corners")
    state = colour_by(session, field)
    in_c = np.isin(np.arange(len(xy)), project.session_rows(project.rois[c]))
    assert np.all(np.asarray(state.snapshot().locs[field])[in_c] == 8)
    project.tables["Made up"].set(c, {"values": {"corners": 5, "diameter_nm": 108},
                                      "signature": "again"})
    session.changed("rois")
    assert np.all(np.asarray(state.snapshot().locs[field])[in_c] == 5)


def test_only_numbers_are_offered():
    session, project, (a, b, c), xy = overlapping_rois()
    project.rois[a].annotations["kind"] = "pore"          # a keyword: no colour
    offered = numeric_values(session)
    assert offered["rois"]["Made up"] == ["corners", "diameter_nm"]
    assert "use" in offered["rois"][ANNOTATIONS]
    assert "kind" not in offered["rois"][ANNOTATIONS]


def test_a_layer_coloured_by_a_value_that_is_gone_draws_plainly():
    session, _ = clusters()
    snap = colour_by(session, "per cluster_id › Nothing › here").snapshot()
    rendered = snap.render(FieldOfView(0.0, 0.0, 100.0, 120, 120))
    assert rendered.color is None or np.all(np.isfinite(rendered.color))


def test_the_render_tab_offers_region_values_and_colours_by_one(qapp):
    from smappy.gui.render_tab import RenderTab
    session, truth = clusters()
    tab = RenderTab(session)
    tab.color.setCurrentIndex(tab.color.findData("region"))
    offered = [tab.color_field.itemData(i) for i in range(tab.color_field.count())]
    assert N_LOCS in offered and "photons" not in offered
    # shown by its value first, the source after it
    assert tab.color_field.itemText(offered.index(N_LOCS)).startswith("n_locs")
    tab.color_field.setCurrentIndex(offered.index(N_LOCS))
    settings = session.layers[0].state.settings
    assert settings.color_field == N_LOCS
    assert settings.color_range is not None              # the values' range, not None
    tab.hide_missing.setChecked(True)
    assert session.layers[0].state.settings.hide_missing
    # back to a column of the localizations
    tab.color.setCurrentIndex(tab.color.findData("field"))
    assert session.layers[0].state.settings.color_field in session.locs
    # and from elsewhere -- the results window's header
    tab.colour_by_region(N_LOCS)
    assert tab.color.currentData() == "region"
    assert session.layers[0].state.settings.color_field == N_LOCS


def test_the_results_window_lists_every_region_sorts_copies_and_colours(qapp, tmp_path):
    from smappy.gui.region_results_window import RegionResultsWindow
    from PySide6.QtCore import Qt
    session, truth = clusters()
    coloured, filtered, centred = [], [], []
    window = RegionResultsWindow(session, colour_by=coloured.append,
                                 filter_on=lambda *a: filtered.append(a),
                                 center_on=lambda x, y: centred.append((x, y)))
    window.show_source("cluster_id")
    model = window.model
    assert model.names == ["1", "2", "3"]                  # a row per cluster
    n = next(k for k, (label, value) in enumerate(model.columns, 1) if value == "n_locs")
    assert [row[n - 1] for row in model.values] == [60, 40, 25]
    window.table.sortByColumn(n, Qt.AscendingOrder)        # numbers sorted as numbers
    header, rows = window.rows_as_text()
    assert [r[0] for r in rows] == ["3", "2", "1"]
    window.save_csv(tmp_path / "regions.csv")
    assert (tmp_path / "regions.csv").read_text().splitlines()[1].startswith("3,")
    menu = window.header_menu(n)
    actions = {a.text(): a for a in menu.actions()}
    actions["Colour localizations by this"].trigger()
    assert coloured == [N_LOCS]
    actions["Filter regions on this..."].trigger()
    assert filtered == [("cluster_id", "Id statistics", "n_locs")]
    window._on_row(window.proxy.index(0, 0))               # cluster 3, centred on it
    assert np.allclose(centred[0], 9000, atol=30)
    window.close()


def test_the_results_window_marks_an_rois_old_values_and_shows_it_in_the_manager(qapp):
    from smappy.gui.region_results_window import RegionResultsWindow, results_table
    session, project, (a, b, c), xy = overlapping_rois()
    shown = []
    window = RegionResultsWindow(session, show_roi=shown.append)
    window.show_source("rois")
    regions, names, columns, values, stale = results_table(session, "rois")
    assert regions == list(project.ordered_ids())
    assert ("Made up", "diameter_nm") in columns
    window._on_row(window.proxy.index(0, 0))
    assert shown and shown[0] in (a, b, c)
    window.close()


def test_the_regions_tab_opens_the_table_and_its_filter_on_a_value(qapp):
    from smappy.gui.roi_tab import ROIHeader
    session, _ = clusters()
    coloured = []
    header = ROIHeader(session, colour_by=coloured.append)
    window = header.open_results("cluster_id")
    assert window.source.currentData() == "cluster_id"
    header.filter_on("cluster_id", "Id statistics", "n_locs")
    assert header.region_filter.chosen == ("cluster_id", "Id statistics", "n_locs")
    assert window.colour_by is not None
    window.close()
