"""The region filter's panel in the Regions tab and its button in the Render
tab: conditions chosen from the tree, kept, switched off per layer."""
import numpy as np
import pytest

from smappy.region_runs import REGION_FILTER

from test_region_filter import clusters, overlapping_rois


@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from smappy.gui import make_app
    return make_app()


def leaves(item):
    out = []
    for i in range(item.childCount()):
        child = item.child(i)
        out += [child] if child.childCount() == 0 else leaves(child)
    return out


def leaf(panel, name):
    from smappy.gui.region_filter_panel import ROLE
    return next(l for l in leaves(panel.tree.invisibleRootItem())
                if l.data(0, ROLE)[2] == name)


def test_a_value_chosen_in_the_tree_with_a_range_becomes_a_condition(app):
    from smappy.gui.render_tab import RenderTab
    from smappy.gui.roi_tab import ROIHeader
    session, truth = clusters()
    header, render = ROIHeader(session), RenderTab(session)
    panel = header.region_filter
    panel.tree.setCurrentItem(leaf(panel, "n_locs"))
    assert "3 regions" in panel.what.text() and panel.add.text() == "add"
    panel.lo.set(30)
    panel._on_add()
    assert [c.value for c in session.region_filter] == ["n_locs"]
    assert np.array_equal(np.asarray(session.selection(0).mask, bool),
                          np.isin(truth, [1, 2]))
    # the same value again is the same condition, updated
    panel.tree.setCurrentItem(leaf(panel, "n_locs"))
    assert panel.add.text() == "update" and panel.lo.value() == 30
    # the Render tab's button: the filter off in this layer, and on again
    button = render.filter.regions
    assert not button.isHidden() and button.isChecked()
    button.setChecked(False)
    assert REGION_FILTER not in session.layers[0].ranges
    assert np.asarray(session.selection(0).mask).all()
    button.setChecked(True)
    assert REGION_FILTER in session.layers[0].ranges
    panel._remove(0)
    assert session.region_filter == [] and render.filter.regions.isHidden()


def test_the_search_finds_a_value_inside_its_group(app):
    from smappy.gui.roi_tab import ROIHeader
    session, _ = clusters()
    panel = ROIHeader(session).region_filter
    panel.search.setText("std")            # a group: all of it
    shown = [l for l in leaves(panel.tree.invisibleRootItem()) if not l.isHidden()]
    assert {l.parent().text(0) for l in shown} == {"std"}
    assert [l.text(0) for l in shown] == ["x", "y", "xy"]
    panel.search.setText("median")         # a member: in every group that has one
    shown = [l for l in leaves(panel.tree.invisibleRootItem()) if not l.isHidden()]
    assert {l.text(0) for l in shown} == {"median"}
    assert {l.parent().text(0) for l in shown} == {"photons", "xy_err_nm"}


def test_an_annotation_is_filtered_by_the_texts_ticked(app):
    from PySide6.QtCore import Qt
    from smappy.gui.roi_tab import ROIHeader
    session, project, (a, b, c), _ = overlapping_rois()
    project.rois[b].use = False
    session.changed("rois")
    panel = ROIHeader(session).region_filter
    panel.tree.setCurrentItem(leaf(panel, "use"))
    texts = {panel.texts.item(i).text(): panel.texts.item(i)
             for i in range(panel.texts.count())}
    assert set(texts) == {"true", "false"}
    texts["true"].setCheckState(Qt.Checked)
    panel._on_add()
    assert session.region_filter_outcome().passing["rois"] == {a, c}
