"""The fit region: shapes drawn on the preview's frame decide which
candidates are fitted."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt    # noqa: E402

from smappy.plugins import Context                                   # noqa: E402
from smappy.plugins.fit import GaussianFit, in_region                # noqa: E402
from smappy.session import Session                                   # noqa: E402
from test_preview import settings_for, stack                         # noqa: E402,F401

# the stack's four emitters, (x, y) in frame pixels
EMITTERS = ((12, 10), (40, 10), (20, 34), (48, 34))


def rect(kind, x0, y0, x1, y1):
    return {"kind": kind, "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def test_include_shapes_add_up_and_exclude_shapes_cut_out():
    x = np.array([5.0, 15.0, 25.0, 35.0])
    y = np.zeros(4)
    assert in_region(x, y, []).all()                               # no shapes: all
    left = rect("include", 0, -1, 20, 1)
    right = rect("include", 30, -1, 40, 1)
    assert in_region(x, y, [left]).tolist() == [True, True, False, False]
    assert in_region(x, y, [left, right]).tolist() == [True, True, False, True]
    hole = rect("exclude", 10, -1, 20, 1)
    assert in_region(x, y, [left, right, hole]).tolist() == [True, False, False, True]
    assert in_region(x, y, [hole]).tolist() == [True, False, True, True]  # only out


def fitted_xy(stack, shapes, use=True):
    settings = settings_for(GaussianFit, stack)
    settings.region.shapes = shapes
    settings.region.use = use
    settings.output.path = str(stack.with_suffix(".h5"))
    locs = GaussianFit().run(Context(), settings).locs
    return np.column_stack([np.asarray(locs["x_nm"]) / 100, np.asarray(locs["y_nm"]) / 100])


def near(xy, emitter):
    return bool((np.hypot(xy[:, 0] - emitter[0], xy[:, 1] - emitter[1]) < 2).any())


def test_only_the_emitters_in_the_region_are_fitted(stack):
    top = rect("include", 0, 0, 60, 22)
    xy = fitted_xy(stack, [top])
    assert near(xy, EMITTERS[0]) and near(xy, EMITTERS[1])
    assert not near(xy, EMITTERS[2]) and not near(xy, EMITTERS[3])
    left_out = fitted_xy(stack, [rect("exclude", 30, 0, 60, 22)])
    assert not near(left_out, EMITTERS[1])
    assert all(near(left_out, e) for e in (EMITTERS[0], EMITTERS[2], EMITTERS[3]))
    # unticked: the shapes are kept and the whole frame is fitted
    assert all(near(fitted_xy(stack, [top], use=False), e) for e in EMITTERS)


def test_the_preview_declares_its_frame_drawable_and_finds_only_inside(stack):
    settings = settings_for(GaussianFit, stack)
    settings.region.shapes = [rect("include", 0, 0, 30, 22)]
    result = GaussianFit().preview(Context(), settings, frame=0)
    assert result.data["candidates"] == 1
    edit = result.plot.edit
    assert edit.field == "region.shapes" and edit.panel == 0
    assert edit.shapes == settings.region.shapes
    plt.close("all")


def test_a_shape_drawn_on_the_preview_goes_into_the_form(stack):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.figures import FigurePane
    from smappy.gui.plugin_panel import PluginPanel

    session = Session()
    panel = PluginPanel(GaussianFit, session)
    panel.form.set_values({"source.path": str(stack), "camera.conversion": 1.0,
                           "camera.offset": 100.0, "camera.pixelsize_um": 0.1,
                           "camera.em_on": False})
    result = panel.plugin.preview(session.context(), panel.form.value(), frame=0)
    panel._show_figures = False
    panel._job = "preview"
    panel._on_done(result)
    pane = FigurePane()
    pane.offer(result.plot)
    pane.draw()
    tools = pane.shape_tools
    assert not tools.isHidden() and tools.ax is pane.figure.axes[0]
    tools._add("exclude", [[30, 0], [60, 0], [60, 22], [30, 22]])
    value = panel.form.value()
    assert value.region.shapes == [rect("exclude", 30, 0, 60, 22)]
    assert "1 left out" in panel.form.fields["region"].fields["shapes"].label.text()
    tools.undo()
    assert panel.form.value().region.shapes == []
    plt.close("all")
