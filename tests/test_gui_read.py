"""The assistant reads and sets what the GUI shows (`gui/gui_read.py`): asked about the open
Spline 3D form it could not tell which calibration the form named, and searched the disk for
one (a usage log, 2026-11)."""
import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def gui(app, tmp_path):
    """A control window with simulated data, its assistant in suggest mode."""
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.session import Session
    session = Session()
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    control.show()
    a = control._assistant().server.assistant
    a.knowledge = tmp_path / "kb"
    a.output = tmp_path
    a.mode, a.runner = "act", None
    a.call("run_plugin", {"path": "File/Simulate/Simulate", "settings": {"n_frames": 2000},
                          "apply": True})
    a.mode = "suggest"
    yield control, a
    manager = getattr(control.roi_tab, "manager", None)
    for window in list(control._plugin_windows.values()) + [manager, render, control]:
        if window is not None:
            window.close()


# ------------------------------------------------------------ the walk itself
def test_a_window_reads_each_field_by_its_label_under_its_section(app):
    """The generic reader names a control by what the user reads beside it: a form's label,
    the label before it in its row (every control of the row after it), its own text; a unit
    after it is no name; a folded section is read and said folded; a hidden widget and a tab
    widget's other pages are not read."""
    from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                                   QGridLayout, QHBoxLayout, QLabel, QLineEdit, QSpinBox,
                                   QTabWidget, QVBoxLayout, QWidget)
    from smappy.gui.gui_read import read_widget
    from smappy.gui.widgets import CollapsibleSection
    window = QWidget()
    column = QVBoxLayout(window)
    form = QFormLayout()
    size = QSpinBox()
    size.setValue(13)
    form.addRow("box size", size)
    column.addLayout(form)
    row = QHBoxLayout()
    row.addWidget(QLabel("range"))
    row.addWidget(QLineEdit("10"))
    row.addWidget(QLineEdit("90"))
    row.addWidget(QLabel("nm"))
    column.addLayout(row)
    grid = QGridLayout()
    grid.addWidget(QLabel("LUT"), 0, 0)
    lut = QComboBox()
    lut.addItems(["gray", "red hot"])
    lut.setCurrentIndex(1)
    grid.addWidget(lut, 0, 1)
    grid.addWidget(QCheckBox("invert"), 0, 2)
    column.addLayout(grid)
    inner = QWidget()
    QFormLayout(inner).addRow("gamma", QDoubleSpinBox(value=0.5))
    column.addWidget(CollapsibleSection("more", inner, expanded=False))
    hidden = QLineEdit("secret")
    hidden.hide()
    column.addWidget(hidden)
    pages = QTabWidget()
    first, second = QWidget(), QWidget()
    QVBoxLayout(first).addWidget(QCheckBox("on the first page"))
    QVBoxLayout(second).addWidget(QCheckBox("on the second page"))
    pages.addTab(first, "one")
    pages.addTab(second, "two")
    column.addWidget(pages)
    reading = read_widget(window)
    assert reading.fields == {"box size": 13, "range": "10", "range #2": "90",
                              "LUT": "red hot", "LUT: invert": False, "more/gamma": 0.5,
                              "page": "one", "one/on the first page": False}
    assert reading.folded == ["more"] and reading.choices["page"] == ["one", "two"]
    assert "nm" not in reading.notes and "range" not in reading.notes


def test_an_untitled_form_is_named_by_its_settings_in_words(app):
    """Not "settings", "settings #2": the PSF calibration window has five untitled forms."""
    from dataclasses import dataclass
    from PySide6.QtWidgets import QVBoxLayout, QWidget
    from smappy.gui.gui_read import read_widget
    from smappy.gui.params import SettingsForm
    from smappy.plugins import param

    @dataclass
    class PupilModelSettings:
        zernike_order: int = param(6, label="Zernike order")

    @dataclass
    class BeadSettings:
        roi_size: int = param(27, label="ROI size")

    window = QWidget()
    column = QVBoxLayout(window)
    column.addWidget(SettingsForm(PupilModelSettings))
    column.addWidget(SettingsForm(BeadSettings))
    assert read_widget(window).forms == {"pupil model": {"zernike_order": 6},
                                         "bead": {"roi_size": 27}}


def test_a_field_that_opens_a_dialog_is_set_and_the_dialog_closed_unanswered(app):
    """Every call runs on the GUI thread: a modal dialog would hold the assistant's queue."""
    from PySide6.QtWidgets import QComboBox, QMessageBox
    from smappy.gui.gui_read import _NoDialogs, set_control
    box = QComboBox()
    box.addItems(["one", "two"])
    asked = []
    box.activated.connect(lambda i: asked.append(QMessageBox(
        QMessageBox.Question, "Really?", "sure?", QMessageBox.Yes | QMessageBox.No).exec()))
    with _NoDialogs() as guard:
        set_control(box, "two")
    assert box.currentText() == "two" and asked and guard.closed[0] in ("Really?", "sure?")


# --------------------------------------------------------------- the tools
def test_read_gui_lists_the_tabs_their_plugins_the_windows_and_what_is_in_front(gui):
    control, a = gui
    listed = a.call("read_gui", {})
    tabs = {t["name"]: t for t in listed["tabs"]}
    assert {"File", "Localize", "Render", "Regions"} <= set(tabs)
    assert {p["plugin"] for p in tabs["Localize"]["plugins"]} >= {"Localize/Spline 3D"}
    assert any(w["name"] == "control window" for w in listed["windows"])
    assert listed["in_front"]
    # a plugin's window, opened by open_plugin, is listed beside the tabs
    a.call("open_plugin", {"path": "Analysis/Drift/RCC"})
    names = [w["name"] for w in a.call("read_gui", {})["windows"]]
    assert "RCC drift correction (window)" in names


def test_a_plugin_panel_reads_by_run_plugins_names_with_what_differs_from_the_defaults(
        gui, tmp_path):
    """"Which calibration does my fit use?" -- read from the form, not guessed."""
    control, a = gui
    calibration = str(tmp_path / "beads_3dcal.mat")
    done = a.call("set_gui", {"name": "Localize > Spline 3D",
                              "values": {"model.calibration": calibration, "fit.roisize": 15}})
    assert done["set"] == ["model.calibration", "fit.roisize"] and not a.proposals
    assert done["now"]["settings"]["fit.roisize"] == 15
    panel = control._assistant_panel("Localize/Spline 3D")
    assert panel.form.value().model.calibration == calibration     # in the user's form
    read = a.call("read_gui", {"name": "Localize/Spline 3D"})       # by its path, too
    assert read["kind"] == "plugin" and read["plugin"] == "Localize/Spline 3D"
    assert read["settings"]["model.calibration"] == calibration
    assert read["defaults_where_different"] == {"model.calibration": "", "fit.roisize": 13}
    narrowed = a.call("read_gui", {"name": "Localize > Spline 3D", "fields": ["fit"]})
    assert set(narrowed["settings"]) == {k for k in read["settings"] if k.startswith("fit.")}
    with pytest.raises(Exception, match="no field 'fit.box'.*fit.roisize"):
        a.call("set_gui", {"name": "Localize > Spline 3D", "values": {"fit.box": 3}})


def test_set_gui_types_into_a_form_so_a_preset_fills_the_rest(gui):
    control, a = gui
    a.call("open_plugin", {"path": "File/Simulate/Simulate"})
    a.call("set_gui", {"name": "Simulate (window)", "values": {"preset": "nuclear pores"}})
    form = control._plugin_windows["File/Simulate/Simulate"].panel.form.value()
    assert form.preset == "Nuclear pores (Nup96)"
    assert form.components[0].structure.preset == "npc"      # filled by the preset's react


def test_the_render_tab_is_set_in_suggest_mode_and_the_roi_manager_is_proposed(gui):
    """The Render tab changes what is shown, as set_layer does; the ROI manager the ROIs."""
    control, a = gui
    render = a.call("read_gui", {"name": "Render"})
    assert render["kind"] == "tab" and render["fields"]["display/LUT"]
    a.call("set_gui", {"name": "Render", "values": {"display/LUT": "green"}})
    assert a.call("read_gui", {"name": "Render", "fields": ["display/LUT"]})[
        "fields"]["display/LUT"] == "green"
    assert control.session.layers[0].get_display().lut == "green" and not a.proposals
    with pytest.raises(Exception, match="no 'mauve'.*green"):
        a.call("set_gui", {"name": "Render", "values": {"display/LUT": "mauve"}})
    control._open_manager()
    manager = a.call("read_gui", {"name": "ROI manager"})
    assert manager["fields"]["shape"] and manager["fields"]["size"] > 0
    proposed = a.call("set_gui", {"name": "ROI manager", "values": {"size": 500},
                                  "reason": "the pores are larger"})
    assert proposed["proposed"]["tool"] == "set_gui"
    assert a.call("read_gui", {"name": "ROI manager"})["fields"]["size"] != 500
    a.carry_out(proposed["proposed"]["id"])                   # the user's yes
    assert a.call("read_gui", {"name": "ROI manager"})["fields"]["size"] == 500


def test_focus_gui_brings_the_section_forward_unfolded_with_the_field_outlined(gui):
    control, a = gui
    from smappy.gui.plugin_tab import PluginTab
    assert a.call("focus_gui", {"name": "Localize > Spline 3D",
                                "fields": ["fit.roisize", "detection.sigma"]}) == {
        "shown": "Localize > Spline 3D", "fields": ["fit.roisize", "detection.sigma"]}
    page = control.tab_page(control.tabs.currentIndex())
    assert isinstance(page, PluginTab) and page.tab.name == "Localize"
    panel = control._assistant_panel("Localize/Spline 3D")
    leaf = panel.form.fields["fit"].fields["roisize"]
    assert "border" in leaf.styleSheet()
    assert "border" in panel.form.fields["detection"].fields["sigma"].styleSheet()
    assert "Localize > Spline 3D" in a.call("read_gui", {})["in_front"]
    with pytest.raises(Exception, match="no field 'fit.nothing'"):
        a.call("focus_gui", {"name": "Localize > Spline 3D", "fields": ["fit.nothing"]})


def test_a_panels_last_run_is_read_and_kept_for_figure(gui):
    """"Do these numbers make sense?" about a run the user made in the panel."""
    from smappy.plugins import Result
    control, a = gui
    a.call("open_plugin", {"path": "Analysis/Drift/RCC"})
    panel = control._plugin_windows["Analysis/Drift/RCC"].panel
    panel.result = Result(text="drift 12 nm", plot=lambda ax: ax.plot([0, 1], [0, 1]))
    read = a.call("read_gui", {"name": "Analysis/Drift/RCC"})
    run = read["last_run"]
    assert run["text"] == "drift 12 nm" and run["figures"] == ["main"]
    assert a.call("read_gui", {"name": "Analysis/Drift/RCC"})["last_run"]["result"] \
        == run["result"]                                       # kept once
    assert a.call("figure", {"result": run["result"]})


def test_every_window_the_assistant_opens_and_every_tab_can_be_read(gui):
    """read_gui is generic: each window `open_window` opens, and each tab, reads without a
    line of its own -- and names its fields, so a new window is not left unread."""
    control, a = gui
    for name in control._assistant_windows():
        if name == "help":
            continue                       # read_doc reads its pages
        a.call("open_window", {"name": name})
    listed = a.call("read_gui", {})
    names = [t["name"] for t in listed["tabs"]] + [w["name"] for w in listed["windows"]]
    for name in set(control._assistant_windows()) - {"help"}:
        assert name in names
    for name in names:
        read = a.call("read_gui", {"name": name})
        assert read["name"] == name
        assert read.get("fields") or read.get("forms") or read.get("plugins") \
            or read.get("buttons") or read.get("panels"), name
