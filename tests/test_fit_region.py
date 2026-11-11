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


# ------------------------------------------------------------ a split chip: copied through the channels
def halves(mirrored=False):
    from smappy.channels import ChannelLayout
    return ChannelLayout([(30, 0, 30, 48), (0, 0, 30, 48)],
                         flips=[(False, False), (mirrored, False)])


def test_a_shape_on_one_channel_is_copied_where_that_sample_lies_in_the_other():
    from smappy.plugins.fit import region_copies
    shape = rect("include", 8, 5, 16, 15)                       # on the left channel, channel 2
    copy, = region_copies([shape], halves())
    np.testing.assert_allclose(np.sort(np.asarray(copy["points"])[:, 0]), [38, 38, 46, 46])
    assert copy["kind"] == "include" and copy["copied_from"] == 1
    # mirrored: the left channel's x runs the other way, and so does the copy
    copy, = region_copies([shape], halves(mirrored=True))
    np.testing.assert_allclose(np.sort(np.asarray(copy["points"])[:, 0]), [43, 43, 51, 51])


def test_a_strip_along_the_seam_crops_both_channels_inner_edges():
    """The usual *leave out*: the DoG's artefacts and scattered light along the boundary.  Each
    side's part is copied onto the other: mirrored, inner edge onto inner edge."""
    from smappy.plugins.fit import channel_region
    strip = rect("exclude", 27, 0, 33, 48)
    judged = channel_region([strip], halves(mirrored=True))
    x = np.array([24.0, 28.0, 31.0, 36.0])
    assert in_region(x, np.full(4, 10.0), judged).tolist() == [True, False, False, True]
    # not mirrored, the inner edge of the left is the outer edge of the right
    judged = channel_region([strip], halves())
    assert not in_region(np.array([2.0]), np.array([10.0]), judged)[0]


def test_a_two_channel_fit_fits_both_channels_in_a_region_drawn_on_one(stack):
    """A *fit in* rectangle on one half of a split chip fitted that half alone (2026-11)."""
    settings = settings_for(GaussianFit, stack)
    settings.channels.split = "two channels"
    settings.channels.layout, settings.channels.main_channel = "right-left", "right"
    settings.region.shapes = [rect("include", 4, 2, 24, 20)]   # the left half's top: (12, 10)
    settings.output.save = False
    locs = GaussianFit().run(Context(), settings).locs
    xy = np.column_stack([np.asarray(locs["x_nm"]) / 100, np.asarray(locs["y_nm"]) / 100])
    assert near(xy, EMITTERS[0]) and near(xy, EMITTERS[1])     # and its copy on the right: (40, 10)
    assert not near(xy, EMITTERS[2]) and not near(xy, EMITTERS[3])
    assert set(np.unique(locs["camera_channel"])) == {0, 1}


def test_a_region_that_leaves_a_channel_nothing_is_said():
    from smappy.plugins.fit import region_problem
    assert region_problem([rect("include", 4, 2, 24, 20)], halves()) is None
    assert region_problem([rect("exclude", 0, 0, 30, 48)], halves()) \
        .startswith("the fit region leaves nothing of channels 1 and 2")
    off = rect("include", 70, 0, 80, 10)                       # beside the frame: no channel
    assert "channels 1 and 2" in region_problem([off], halves())
    assert region_problem([off], None) is None                 # a whole-chip fit: not asked


def test_the_form_says_when_the_region_leaves_a_channel_nothing(stack):
    settings = settings_for(GaussianFit, stack)
    settings.channels.split = "two channels"
    settings.channels.layout, settings.channels.main_channel = "right-left", "right"
    settings.region.shapes = [rect("exclude", 0, 0, 60, 48)]
    notice = GaussianFit().describe(None, settings).notice
    assert notice is not None and "leaves nothing" in notice.text


def _edited_pane(stack, shapes):
    """A Preview's figure with the drawing tools, its shapes written into a list as the form would."""
    pytest.importorskip("PySide6")
    from types import SimpleNamespace
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.figures import FigurePane
    settings = settings_for(GaussianFit, stack)
    settings.region.shapes = shapes
    result = GaussianFit().preview(Context(), settings, frame=0)
    kept = []
    result.plot.edit.on_change = kept.append
    pane = FigurePane()
    pane.offer(result.plot)
    pane.draw()
    tools = pane.shape_tools
    tools.edit_button.setChecked(True)
    tools._choose(len(tools.buttons))
    click = (lambda x, y: tools._pressed(SimpleNamespace(inaxes=tools.ax, xdata=x, ydata=y)))
    return tools, kept, click


def test_a_drawn_rectangle_is_taken_up_and_resized(stack):
    from matplotlib.widgets import RectangleSelector
    tools, kept, click = _edited_pane(stack, [rect("include", 0, 0, 30, 22)])
    click(50, 40)                                              # on no shape: nothing taken up
    assert tools._editing is None and tools._selector is None
    click(15, 10)
    assert tools._editing == 0 and isinstance(tools._selector, RectangleSelector)
    tools._selector.extents = (0, 40, 0, 22)                   # its right edge dragged out
    tools._selector.onselect(None, None)
    assert kept[-1] == [rect("include", 0, 0, 40, 22)]
    # drawn again, the shape stays in the selector, at its new size
    assert tools._editing == 0 and tools._selector.extents == pytest.approx((0, 40, 0, 22))
    click(55, 45)                                              # let go
    assert tools._selector is None
    plt.close("all")


def test_a_drawn_polygon_is_taken_up_by_its_vertices(stack):
    from matplotlib.widgets import PolygonSelector
    triangle = {"kind": "exclude", "points": [[5.0, 5.0], [25.0, 5.0], [15.0, 20.0]]}
    tools, kept, click = _edited_pane(stack, [triangle])
    click(15, 9)
    assert isinstance(tools._selector, PolygonSelector)
    tools._selector.verts = [(5, 5), (25, 5), (15, 30)]
    tools._selector.onselect(tools._selector.verts)
    assert kept[-1] == [{"kind": "exclude", "points": [[5, 5], [25, 5], [15, 30]]}]
    plt.close("all")
