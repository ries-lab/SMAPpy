"""What the GUI shows, read and set for the assistant: every tab, panel and window.

The assistant could open a plugin's window filled in and press the menus' items
(`assistant_actions.py`), but not see what the windows held: asked about the
open Spline 3D form it could not tell which calibration the form named, and
searched the disk for one (a usage log, 2026-11).  Here is the reading, and
the setting of fields, for the tools `read_gui`, `set_gui` and `focus_gui`
(`assistant.tools`).

**Two readers, one for forms and one for everything else.**  A plugin's panel
is a settings dataclass (`params.SettingsForm`), so it is read as one: its
values by the dotted names `run_plugin` takes, which differ from the
defaults, what the plugin says about them now (`Plugin.describe`: what *auto*
resolves to, the fields it does not read, its notice), the panel's status
line and its last run.  Every other window and tab -- the Render tab, the
Regions tab, the ROI manager, the batch and training windows -- is hand-made
Qt, and is read by walking its widgets in their layouts' order: each control
named by its label (a form row's label, the label before it in its row, its
own text, else its tooltip) under the titles of the sections, groups and
pages it sits in, with its value as the user sees it.  A `SettingsForm` met
on the way is read as a form; a `PluginPanel` as a panel.  Generic on
purpose: a window written next month is read without a line here, and a
control renamed in the GUI is renamed here with it -- the cost is that a
name is only as good as the label the window gives it.

**What is read is what the user sees**: a widget its window hides is left
out, as is every page of a tab widget but the shown one (its value says
which; setting it shows another), while a folded section's content is read,
marked folded, since what is folded still runs.

**Setting is typing.**  A field is set through the method a user's edit goes
through and its edit signal is sent as an edit sends it -- a form's
``field_changed``, which runs the plugin's ``react`` (a preset fills the
rest); a line's ``editingFinished``; a combo box's ``activated`` -- so the
window does what it does on an edit.  Buttons are not pressed here: what a
click does without asking is `do_action`'s, curated with the reason for each
left out.  A field that opens a dialog when set is answered by closing the
dialog and saying so (`_NoDialogs`): every call runs on the GUI thread, and a
modal dialog would hold the assistant's queue until someone closed it.

**Focusing** brings a panel or window forward with a field outlined for a few
seconds, its sections unfolded and scrolled to: what a mode that walks the
user through steps builds on (read it, point at it, let the user act).
Nothing takes the keyboard: the user is typing to the assistant.
"""
from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractButton, QAbstractItemView, QAbstractSlider, QAbstractSpinBox, QApplication,
    QBoxLayout, QComboBox, QDockWidget, QDoubleSpinBox, QFormLayout, QGraphicsView,
    QGroupBox, QLabel, QLineEdit, QMainWindow, QMenu, QPlainTextEdit, QScrollArea,
    QScrollBar, QSpinBox, QSplitter, QStackedWidget, QTabWidget, QTextEdit, QToolBar,
    QWidget)

from ..assistant.tools import ToolError, _pick, _plain

#: Rows of a table or list a read gives; a table of thousands of ROIs is read by the tools
#: that read the data, not row by row here.
MAX_ROWS = 30
#: Entries of a combo box listed with it, and characters of a text box.
MAX_CHOICES = 40
MAX_TEXT = 2000
#: How long `bring_into_view` outlines a field, in ms.
FLASH_MS = 3000
#: A label after a control this short is its unit ("nm"), not the next control's name.
UNIT_LENGTH = 8

#: Windows and tabs whose fields change the data or the ROIs as they are set: there, set_gui is
#: a proposal in suggest mode.  Everywhere else a field changes a form (run later) or what is
#: shown, which suggest mode makes as it makes `open_plugin`.
CHANGES_SESSION = ("ROI manager", "Regions")


# ------------------------------------------------------------------- reading
@dataclass
class Reading:
    """One window or tab read: what a tool answers, and the widgets behind its names."""
    fields: Dict[str, Any] = field(default_factory=dict)
    greyed: List[str] = field(default_factory=list)
    choices: Dict[str, List[str]] = field(default_factory=dict)
    hints: Dict[str, str] = field(default_factory=dict)       # an empty line's placeholder
    folded: List[str] = field(default_factory=list)           # sections folded shut
    notes: List[str] = field(default_factory=list)            # what labels say
    texts: Dict[str, str] = field(default_factory=dict)       # read-only text boxes
    buttons: List[str] = field(default_factory=list)
    tables: Dict[str, Any] = field(default_factory=dict)
    forms: Dict[str, Any] = field(default_factory=dict)       # a SettingsForm met, by its name
    panels: Dict[str, Any] = field(default_factory=dict)      # a PluginPanel met
    figures: List[str] = field(default_factory=list)
    # the widget behind each name, for setting and focusing: not answered
    widgets: Dict[str, Any] = field(default_factory=dict)
    form_widgets: Dict[str, Any] = field(default_factory=dict)

    def name(self, path: List[str], label: str) -> str:
        """``label`` under ``path``, numbered when the window has it twice."""
        name = "/".join([p for p in path if p] + [label or "unnamed"])
        taken = lambda n: n in self.widgets or n in self.form_widgets or n in self.tables \
            or n in self.panels
        if taken(name):
            k = 2
            while taken(f"{name} #{k}"):
                k += 1
            name = f"{name} #{k}"
        return name

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in ("fields", "greyed", "choices", "hints", "folded", "forms", "panels",
                    "tables", "texts", "notes", "buttons", "figures"):
            value = getattr(self, key)
            if value:
                out[key] = value
        return _plain(out)


def _clean(text: str) -> str:
    """A label's text as a name: no markup, no accelerator, no colon, one line."""
    text = re.sub(r"<[^>]+>", " ", text or "").replace("&nbsp;", " ").replace("&", "")
    text = " ".join(text.split())
    return text.rstrip(":").strip()


def _own_name(widget: QWidget) -> str:
    """A control's name when no label names it: its text, its tooltip, its accessible name."""
    if isinstance(widget, QAbstractButton) and _clean(widget.text()):
        return _clean(widget.text())
    if isinstance(widget, QGroupBox) and widget.title():
        return _clean(widget.title())
    tip = _clean(widget.toolTip().splitlines()[0]) if widget.toolTip() else ""
    if tip and len(tip) <= 50:
        return tip
    return _clean(widget.accessibleName()) or widget.objectName() or ""


def _is_control(widget: QWidget) -> bool:
    return isinstance(widget, (QAbstractButton, QComboBox, QAbstractSpinBox, QLineEdit,
                               QAbstractSlider, QPlainTextEdit, QTextEdit,
                               QAbstractItemView)) and not isinstance(widget, QScrollBar)


def _key(widget) -> int:
    """A widget's identity: its C++ address.  PySide may hand back a new Python wrapper for the
    same widget, whose ``id`` differs, and a label met in its layout was read again as a note."""
    import shiboken6
    return shiboken6.getCppPointer(widget)[0]


def _kind_word(widget: QWidget) -> str:
    """What an unlabelled control is, for its name when nothing else names it."""
    for kind, word in ((QComboBox, "choice"), (QAbstractSpinBox, "number"),
                       (QLineEdit, "text"), (QAbstractButton, "tick"),
                       (QAbstractSlider, "slider")):
        if isinstance(widget, kind):
            return word
    return "field"


def _rows(layout) -> List[Tuple[list, bool]]:
    """A layout's items as rows, each in reading order, and whether it runs across: a grid
    row by row, a horizontal box as one row across, anything else as one row down."""
    from PySide6.QtWidgets import QGridLayout
    if isinstance(layout, QGridLayout):
        rows = []
        for r in range(layout.rowCount()):
            items, seen = [], set()
            for c in range(layout.columnCount()):
                item = layout.itemAtPosition(r, c)
                if item is not None and _key(item) not in seen:
                    seen.add(_key(item))
                    items.append(item)
            if items:
                rows.append((items, True))
        return rows
    items = [layout.itemAt(k) for k in range(layout.count())]
    across = isinstance(layout, QBoxLayout) and layout.direction() in (
        QBoxLayout.LeftToRight, QBoxLayout.RightToLeft)
    return [(items, across)]


def _is_unit(widget: QWidget) -> bool:
    return (isinstance(widget, QLabel) and 0 < len(_clean(widget.text())) <= UNIT_LENGTH
            and not widget.text().rstrip().endswith(":"))


class _Walker:
    """One walk over a window's widgets into a `Reading` (the module's docstring)."""

    def __init__(self, session=None):
        self.reading = Reading()
        self.session = session
        self.seen: set = set()

    # ------------------------------------------------------------ the walk
    def walk(self, widget: QWidget, path: List[str], label: str = "", root: bool = False) -> None:
        from .params import SettingsForm
        from .plugin_panel import PluginPanel
        from .widgets import CollapsibleSection
        if widget is None or _key(widget) in self.seen:
            return
        if not root and (widget.isHidden() or isinstance(widget, QMenu)):
            return
        self.seen.add(_key(widget))
        r = self.reading
        if isinstance(widget, PluginPanel):
            name = r.name(path, label or widget.plugin.name or widget.plugin.path)
            r.panels[name] = panel_summary(widget, self.session)
            return
        if isinstance(widget, SettingsForm):
            name = r.name(path, label or "settings")
            r.forms[name] = form_values(widget)
            r.form_widgets[name] = widget
            return
        if isinstance(widget, CollapsibleSection):
            title = _clean(widget.button.text())
            inner = path + [title]
            if not widget.button.isChecked():
                r.folded.append("/".join(p for p in inner if p))
            self.seen.add(_key(widget.button))
            for w in widget.title_widgets:
                self.walk(w, inner)
            self.seen.add(_key(widget.content))
            self._children(widget.content, inner)
            # the title bar's own buttons (help, detach) are the GUI's, not the window's
            self.seen.update(_key(w) for w in widget.findChildren(
                QWidget, options=Qt.FindDirectChildrenOnly))
            return
        if isinstance(widget, QTabWidget):
            pages = [_clean(widget.tabText(i)) for i in range(widget.count())]
            name = r.name(path, label or "page")
            shown = pages[widget.currentIndex()] if pages and widget.currentIndex() >= 0 else ""
            r.fields[name] = shown
            r.choices[name] = pages
            r.widgets[name] = widget
            if widget.currentWidget() is not None:
                self.walk(widget.currentWidget(), path + [shown], root=True)
            return
        if isinstance(widget, QStackedWidget):
            self.walk(widget.currentWidget(), path, label, root=True)
            return
        if isinstance(widget, QScrollArea):
            self.walk(widget.widget(), path, label, root=True)
            return
        if isinstance(widget, QSplitter):
            for i in range(widget.count()):
                self.walk(widget.widget(i), path)
            return
        if isinstance(widget, QMainWindow):
            for bar in widget.findChildren(QToolBar, options=Qt.FindDirectChildrenOnly):
                if not bar.isHidden():
                    self._toolbar(bar, path)
            self.walk(widget.centralWidget(), path, root=True)
            for dock in widget.findChildren(QDockWidget, options=Qt.FindDirectChildrenOnly):
                if not dock.isHidden():
                    self.walk(dock.widget(), path + [_clean(dock.windowTitle())], root=True)
            from PySide6.QtWidgets import QStatusBar
            bar = widget.findChild(QStatusBar, options=Qt.FindDirectChildrenOnly)
            if bar is not None and bar.currentMessage():
                r.notes.append(_clean(bar.currentMessage()))
            return
        if isinstance(widget, QGroupBox):
            title = _clean(widget.title())
            if widget.isCheckable():
                self._control(widget, path, label or title)
            self._children(widget, path + [title] if title else path)
            return
        if isinstance(widget, QLabel):
            text = _clean(widget.text())
            if text and len(r.notes) < 60:
                r.notes.append(text)
            return
        if isinstance(widget, QGraphicsView) or type(widget).__name__ in (
                "FigureCanvasQTAgg", "Canvas", "QOpenGLWidget", "GLViewWidget"):
            r.figures.append("/".join(p for p in path + [label] if p) or "figure")
            return
        if _is_control(widget):
            self._control(widget, path, label)
            return
        self._children(widget, path, label)

    def _children(self, widget: QWidget, path: List[str], label: str = "") -> None:
        layout = widget.layout()
        if layout is not None:
            self._layout(layout, path, label)
        # widgets in no layout (placed by hand), after the laid-out ones
        for child in widget.findChildren(QWidget, options=Qt.FindDirectChildrenOnly):
            if _key(child) not in self.seen and not child.isWindow():
                self.walk(child, path)

    def _layout(self, layout, path: List[str], label: str = "") -> None:
        if isinstance(layout, QFormLayout):
            for row in range(layout.rowCount()):
                head = layout.itemAt(row, QFormLayout.LabelRole)
                body = (layout.itemAt(row, QFormLayout.FieldRole)
                        or layout.itemAt(row, QFormLayout.SpanningRole))
                text = ""
                if head is not None and isinstance(head.widget(), QLabel):
                    self.seen.add(_key(head.widget()))
                    if head.widget().isHidden():
                        continue                      # a row hidden with its label
                    text = _clean(head.widget().text())
                if body is not None:
                    self._item(body, path, text)
            return
        for items, across in _rows(layout):
            self._row(items, across, path, label)

    def _row(self, items, across: bool, path: List[str], label: str) -> None:
        """One row of a layout: a label names the control after it -- in a row across, every
        control after it until the next label ("colour range", "colour range #2")."""
        pending, used, last_control = label, False, False
        for item in items:
            widget = item.widget()
            if isinstance(widget, QLabel):
                self.seen.add(_key(widget))
                if widget.isHidden():
                    continue
                if last_control and _is_unit(widget):
                    continue                          # the unit of the control before
                if pending and not used and pending != label:
                    self.reading.notes.append(pending)     # two labels: the first a note
                pending, used, last_control = _clean(widget.text()), False, False
                continue
            if widget is None and item.layout() is None:
                continue                              # a spacer
            if widget is not None and widget.isHidden():
                continue
            before = self._count()
            self._item(item, path, pending)
            last_control = self._count() > before
            used = used or last_control
            if last_control and not across:
                pending, used = "", False
        if pending and not used and pending != label:
            self.reading.notes.append(pending)

    def _count(self) -> int:
        r = self.reading
        return len(r.widgets) + len(r.form_widgets) + len(r.tables) + len(r.buttons) \
            + len(r.panels)

    def _item(self, item, path: List[str], label: str) -> None:
        if item.widget() is not None:
            self.walk(item.widget(), path, label)
        elif item.layout() is not None:
            self._layout(item.layout(), path, label)

    def _toolbar(self, bar: QToolBar, path: List[str]) -> None:
        r = self.reading
        for action in bar.actions():
            text = _clean(action.text())
            if not text or action.isSeparator() or not action.isVisible():
                continue
            if action.isCheckable():
                name = r.name(path, text)
                r.fields[name] = action.isChecked()
                r.widgets[name] = action                     # set by `trigger`
                if not action.isEnabled():
                    r.greyed.append(name)
            else:
                r.buttons.append(text + ("" if action.isEnabled() else " (greyed)"))
        for w in bar.findChildren(QWidget, options=Qt.FindDirectChildrenOnly):
            if _is_control(w) and not isinstance(w, QAbstractButton) and not w.isHidden():
                self.walk(w, path)
            self.seen.add(_key(w))

    # --------------------------------------------------------------- controls
    def _control(self, widget: QWidget, path: List[str], label: str) -> None:
        r = self.reading
        if isinstance(widget, QAbstractButton) and not widget.isCheckable():
            text = _own_name(widget) or label
            if text:
                r.buttons.append(text + ("" if widget.isEnabled() else " (greyed)"))
            return
        if isinstance(widget, QAbstractItemView):
            name = r.name(path, label or _own_name(widget) or "table")
            r.tables[name] = _model_rows(widget)
            r.widgets[name] = widget
            return
        if isinstance(widget, (QPlainTextEdit, QTextEdit)) and widget.isReadOnly():
            text = widget.toPlainText().strip()
            if text:
                name = r.name(path, label or _own_name(widget) or "text")
                r.texts[name] = text[-MAX_TEXT:]
            return
        own = _own_name(widget)
        if isinstance(widget, QAbstractButton) and label and own and own != label:
            label = f"{label}: {own}"                  # a row's label and its tick's text
        name = r.name(path, label or own or _kind_word(widget))
        r.widgets[name] = widget
        r.fields[name] = control_value(widget)
        if not widget.isEnabled():
            r.greyed.append(name)
        if isinstance(widget, QComboBox) and widget.count() <= MAX_CHOICES:
            r.choices[name] = [widget.itemText(i) for i in range(widget.count())]
        if isinstance(widget, QLineEdit) and not widget.text() and widget.placeholderText():
            r.hints[name] = widget.placeholderText()


def control_value(widget) -> Any:
    """A control's value as the user reads it."""
    if isinstance(widget, QAbstractButton):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        if widget.specialValueText() and widget.value() == widget.minimum():
            return widget.specialValueText()
        return widget.value()
    if isinstance(widget, QAbstractSpinBox):
        return widget.text()
    if isinstance(widget, QLineEdit):
        return widget.text()
    if isinstance(widget, QAbstractSlider):
        return widget.value()
    if isinstance(widget, (QPlainTextEdit, QTextEdit)):
        return widget.toPlainText()[:MAX_TEXT]
    return None


def _model_rows(view: QAbstractItemView) -> Dict[str, Any]:
    """A table's or list's columns, its first rows and which are selected."""
    model = view.model()
    if model is None:
        return {"rows": []}
    from PySide6.QtCore import QAbstractListModel, QModelIndex
    n = model.rowCount(QModelIndex())
    # a list model's columnCount is private in PySide: one column
    m = 1 if isinstance(model, QAbstractListModel) else model.columnCount(QModelIndex())
    hidden = getattr(view, "isColumnHidden", lambda c: False)
    columns = [c for c in range(m) if not hidden(c)]
    head = [str(model.headerData(c, Qt.Horizontal) or "") for c in columns]
    rows = []
    for r in range(min(n, MAX_ROWS)):
        row = []
        for c in columns:
            index = model.index(r, c)
            text = model.data(index, Qt.DisplayRole)
            check = model.data(index, Qt.CheckStateRole)
            text = "" if text is None else str(text)
            if check is not None:
                on = check == Qt.Checked or check == 2 or getattr(check, "value", None) == 2
                text = ("[x] " if on else "[ ] ") + text
            row.append(text)
        rows.append(row if len(row) != 1 else row[0])
    out: Dict[str, Any] = {"rows": rows}
    if len(columns) > 1 and any(head):
        out["columns"] = head
    if n > MAX_ROWS:
        out["not shown"] = f"{n - MAX_ROWS} more rows"
    selection = view.selectionModel()
    if selection is not None:
        chosen = sorted({i.row() for i in selection.selectedIndexes()})
        if chosen:
            out["selected rows"] = chosen[:MAX_ROWS]
    return out


def read_widget(widget: QWidget, session=None) -> Reading:
    """``widget`` -- a window, a tab's page -- read whole (the module's docstring)."""
    walker = _Walker(session)
    walker.walk(widget, [], root=True)
    return walker.reading


# --------------------------------------------------------------------- forms
def form_values(form) -> Dict[str, Any]:
    """A `SettingsForm`'s fields by dotted name, as the form holds them."""
    return _plain(form.values())


def _hidden(name: str, flags: Dict[str, bool]) -> bool:
    """Whether ``name`` is hidden by `FormState.shown`, itself or a part it is in."""
    parts = name.split(".")
    return any(flags.get(".".join(parts[:k])) is False for k in range(1, len(parts) + 1))


def plugin_reading(cls, settings, session, panel=None) -> Dict[str, Any]:
    """A plugin's form read as the assistant needs it: the values it shows, by the names
    run_plugin takes; which differ from the defaults; what the plugin says about them now
    (`Plugin.describe`) -- what *auto* resolves to, what it does not read, its notice -- and,
    with its ``panel``, the panel's status, output and last run (``"_result"``, which the
    tool keeps for `figure` and takes out)."""
    from ..plugins import settings_values
    values = settings_values(settings) if settings is not None else {}
    try:
        defaults = settings_values(cls.Settings()) if cls.Settings is not None else {}
    except Exception:                       # a settings class with a required field
        defaults = {}
    state, trouble = None, ""
    try:
        context = session.context(grouping=cls.grouping) if session is not None else None
        state = cls().describe(context, settings)
    except Exception as error:
        trouble = f"could not describe: {type(error).__name__}: {error}"
    shown = getattr(state, "shown", None) or {}
    visible = {k: v for k, v in values.items() if not _hidden(k, shown)}
    out: Dict[str, Any] = {"plugin": cls.path, "settings": visible,
                           "defaults_where_different": {
                               k: defaults[k] for k, v in visible.items()
                               if k in defaults and _plain(defaults[k]) != _plain(v)}}
    if len(visible) < len(values):
        out["hidden_by_the_form"] = len(values) - len(visible)
    if state is not None:
        if state.hints:
            out["auto_resolves_to"] = {k: v for k, v in state.hints.items()
                                       if not _hidden(k, shown)}
        unused = [k for k, on in state.active.items() if not on and not _hidden(k, shown)]
        if unused:
            out["not_read_now"] = unused
        if state.notice is not None:
            out["notice"] = state.notice.text
    elif trouble:
        out["notice"] = trouble
    if panel is not None:
        status = _clean(panel.status.text())
        if status:
            out["status"] = status
        out["running"] = bool(panel.busy)
        if panel.run_button.text() not in ("Run", ""):
            out["run_button"] = panel.run_button.text()
        lines = panel.output.toPlainText().strip()
        if lines:
            out["output"] = lines[-MAX_TEXT:]
    out = _plain(out)
    if panel is not None and panel.result is not None:
        out["_result"] = panel.result
    return out


def panel_summary(panel, session) -> Dict[str, Any]:
    """A panel met inside a window: which plugin, and what differs from its defaults."""
    try:
        whole = plugin_reading(type(panel.plugin), panel.form.value(), session, None)
    except ValueError as error:
        return {"plugin": panel.plugin.path, "problem": str(error)}
    return {"plugin": whole["plugin"], "differs_from_default": {
        k: whole["settings"].get(k) for k in whole["defaults_where_different"]}}


# ------------------------------------------------------------------- setting
class _NoDialogs(QObject):
    """While a field is set, a modal dialog that opens is closed and remembered: the call runs
    on the GUI thread, and a dialog would hold every call after it until someone closed it."""

    def __init__(self):
        super().__init__()
        self.closed: List[str] = []
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._check)

    def _check(self) -> None:
        try:
            widget = QApplication.activeModalWidget()
            if widget is not None:
                # a message box's title is empty on macOS: its text says what it asked
                said = widget.text() if hasattr(widget, "text") and callable(widget.text) else ""
                self.closed.append(_clean(widget.windowTitle()) or _clean(str(said))[:120]
                                   or type(widget).__name__)
                if hasattr(widget, "reject"):
                    widget.reject()
                else:
                    widget.close()
        except Exception:
            traceback.print_exc()

    def __enter__(self):
        self.timer.start()
        return self

    def __exit__(self, *exc):
        self.timer.stop()
        return False


def set_control(widget, value) -> None:
    """``value`` into one control, as a user's edit puts it there (the module's docstring)."""
    from PySide6.QtGui import QAction
    if isinstance(widget, QAction):
        if not widget.isEnabled():
            raise ToolError("it is greyed out: another setting decides it")
        if bool(value) != widget.isChecked():
            widget.trigger()
        return
    if not widget.isEnabled():
        raise ToolError("it is greyed out: another setting decides it")
    if isinstance(widget, QTabWidget):
        pages = [_clean(widget.tabText(i)) for i in range(widget.count())]
        widget.setCurrentIndex(pages.index(_pick("page", str(value), [(p, p) for p in pages])))
    elif isinstance(widget, QAbstractButton):
        if not isinstance(value, bool):
            raise ToolError("it is a tick: true or false")
        if widget.isChecked() != value:
            widget.click()
    elif isinstance(widget, QComboBox):
        items = [widget.itemText(i) for i in range(widget.count())]
        if widget.isEditable() and str(value) not in items:
            widget.setEditText(str(value))
            widget.lineEdit().editingFinished.emit()
            return
        index = items.index(_pick("value", str(value), [(t, t) for t in items]))
        widget.setCurrentIndex(index)
        widget.activated.emit(index)
        widget.textActivated.emit(items[index])
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        if isinstance(value, str) and widget.specialValueText() \
                and value.strip().lower() == widget.specialValueText().strip().lower():
            value = widget.minimum()
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ToolError("it is a number") from None
        if not widget.minimum() <= number <= widget.maximum():
            raise ToolError(f"it is between {widget.minimum():g} and {widget.maximum():g}")
        widget.setValue(int(round(number)) if isinstance(widget, QSpinBox) else number)
        widget.editingFinished.emit()
    elif isinstance(widget, QLineEdit):
        if widget.isReadOnly():
            raise ToolError("it is read-only")
        widget.setText("" if value is None else f"{value:g}" if isinstance(value, float)
                       else str(value))
        widget.editingFinished.emit()
    elif isinstance(widget, QAbstractSlider):
        widget.setValue(int(value))
    elif isinstance(widget, (QPlainTextEdit, QTextEdit)):
        if widget.isReadOnly():
            raise ToolError("it is read-only")
        widget.setPlainText(str(value))
    else:
        raise ToolError("it is not a field that takes a value")


def set_form(form, values: Dict[str, Any], what: str) -> List[str]:
    """``values`` (dotted names) into a `SettingsForm`, each as an edit: set, then
    ``field_changed``, which the form's owner answers as it answers typing (a plugin's
    ``react``).  Returns the names set; a `ToolError` names the fields for one it lacks."""
    known = form.values()
    done = []
    for name, value in values.items():
        if name not in known and form.spec_at(name) is None:
            # the fields of the part it names, else the parts: a fitter has two hundred
            near = sorted(k for k in known if k.split(".")[0] == name.split(".")[0]) \
                or sorted({k.split(".")[0] for k in known})
            raise ToolError(f"{what} has no field {name!r}; it has " + ", ".join(near[:60]))
        leaf = _leaf(form, name)
        if leaf is not None and getattr(leaf, "_values", None) and isinstance(value, str) \
                and value not in leaf._values:
            # a choice by the words the form shows
            labels = [(str(v), leaf.widget.itemText(i)) for i, v in enumerate(leaf._values)]
            picked = _pick(name, value, labels)
            value = next(v for v in leaf._values if str(v) == picked)
        if isinstance(value, list) and leaf is not None and not hasattr(leaf, "items"):
            value = tuple(value) if isinstance(known.get(name), tuple) else value
        try:
            form.set_values({name: value})
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
            raise ToolError(f"{what}: {name}: {error}") from None
        form.field_changed.emit(name)
        done.append(name)
    return done


def _leaf(form, name: str):
    """The field widget at a dotted name, or None (a list's item, a part)."""
    from .params import SettingsForm
    target = form
    for part in name.split("."):
        if not isinstance(target, SettingsForm) or part not in target.fields:
            return None
        target = target.fields[part]
    return target


# ------------------------------------------------------------------ focusing
class _Flash(QObject):
    """A field outlined for `FLASH_MS`, then as it was: a timer's connected signal, never a
    `singleShot` callable (CLAUDE.md)."""

    def __init__(self, widget: QWidget):
        super().__init__(widget)
        self.widget = widget
        self.style = widget.styleSheet()
        widget.setStyleSheet(self.style + " QWidget { border: 2px solid #e0453a; }")
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._end)
        self.timer.start(FLASH_MS)

    def _end(self) -> None:
        try:
            self.widget.setStyleSheet(self.style)
            self.deleteLater()
        except Exception:
            traceback.print_exc()


def bring_into_view(widget: QWidget, flash: bool = True) -> None:
    """``widget`` unfolded, scrolled to and outlined, its window raised -- no keyboard taken."""
    from .widgets import CollapsibleSection
    parent = widget.parentWidget()
    child = widget
    while parent is not None:
        if isinstance(parent, CollapsibleSection) and not parent.button.isChecked():
            parent.set_expanded(True)
        if isinstance(parent, QTabWidget):
            index = parent.indexOf(child)
            if index < 0:
                index = next((i for i in range(parent.count())
                              if parent.widget(i).isAncestorOf(widget)), -1)
            if index >= 0:
                parent.setCurrentIndex(index)
        if isinstance(parent, QStackedWidget) and parent.indexOf(child) >= 0:
            parent.setCurrentWidget(child)
        child, parent = parent, parent.parentWidget()
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QScrollArea):
            parent.ensureWidgetVisible(widget)
        parent = parent.parentWidget()
    window = widget.window()
    if window.isMinimized():
        window.showNormal()
    window.show()
    window.raise_()
    if flash:
        _Flash(widget)


# ----------------------------------------------------------- the whole GUI
@dataclass
class Entry:
    """One thing read_gui reads: a tab, a plugin's panel or a window."""
    name: str
    kind: str                       # "tab", "plugin", "window"
    widget: Callable[[], Any]       # made (a panel built) when asked
    plugin: str = ""                # a plugin's path
    front: bool = False             # what the user has in front
    extra: Dict[str, Any] = field(default_factory=dict)

    def listed(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind,
                **({"plugin": self.plugin} if self.plugin else {}),
                **({"in_front": True} if self.front else {}), **self.extra}


class GuiAccess:
    """The control window's tabs, panels and windows, read and set by name (`read_gui`)."""

    def __init__(self, control):
        self.control = control                  # the `app.ControlWindow`

    # ------------------------------------------------------------- the list
    def _windows(self) -> Dict[str, QWidget]:
        """The windows by the names the assistant knows them by (`open_window`'s)."""
        c = self.control
        roi_tab = getattr(c, "roi_tab", None)
        named = {"control window": c, "2D view": getattr(c, "render_window", None),
                 "3D view": getattr(c, "view3d_window", None),
                 "ROI manager": getattr(roi_tab, "manager", None),
                 "PSF calibration": getattr(c, "psf_calibration_window", None),
                 "batch window": getattr(c, "batch_window", None),
                 "DECODE training": getattr(c, "decode_training_window", None),
                 "microscope manager": getattr(c, "_microscope_manager", None),
                 "camera editor": getattr(c, "_camera_editor", None),
                 "region results": getattr(roi_tab, "results_window", None)}
        return {k: w for k, w in named.items() if isinstance(w, QWidget)}

    def _plugin_windows(self) -> Dict[str, QWidget]:
        out = dict(getattr(self.control, "_plugin_windows", {}) or {})
        controls = getattr(self.control, "assistant_controls", None)
        for path, window in (getattr(controls, "_windows", None) or {}).items():
            out.setdefault(path, window)
        return out

    def entries(self) -> List[Entry]:
        from .plugin_tab import PluginTab
        c = self.control
        out: List[Entry] = []
        active = QApplication.activeWindow()
        current = c.tabs.currentIndex()
        for index, (tab, page) in enumerate(zip(c.workspace.tabs, c._tab_widgets())):
            shown = not tab.hidden and (tab.detached or index == current)
            out.append(Entry(tab.name, "tab", lambda page=page: page, front=shown,
                             extra={k: True for k, on in (("hidden", tab.hidden),
                                                          ("detached", tab.detached)) if on}))
            if not isinstance(page, PluginTab):
                continue
            for section, instance in zip(page.sections, tab.instances):
                slot = page.slots.get(instance.id)
                title = _clean(section.button.text()) or instance.title()
                out.append(Entry(f"{tab.name} > {title}", "plugin",
                                 lambda slot=slot: slot.ensure() if slot is not None else None,
                                 plugin=instance.plugin,
                                 front=shown and section.button.isChecked(),
                                 extra={} if section.button.isChecked() else {"folded": True}))
        windows = self._plugin_windows()
        for path, window in windows.items():
            if window is not None and window.isVisible():
                title = _clean(getattr(window.panel.plugin, "name", "") or path)
                out.append(Entry(f"{title} (window)", "plugin",
                                 lambda window=window: window.panel, plugin=path,
                                 front=window is active))
        known = {id(w) for w in windows.values()}
        for name, window in self._windows().items():
            known.add(id(window))
            if window.isVisible() or name == "control window":
                out.append(Entry(name, "window", lambda window=window: window,
                                 front=window is active))
        for window in QApplication.topLevelWidgets():
            if (id(window) in known or not window.isVisible() or not window.isWindow()
                    or isinstance(window, QMenu)
                    or window.windowType() in (Qt.ToolTip, Qt.Popup, Qt.SplashScreen)):
                continue
            title = _clean(window.property("smappy_base_title") or window.windowTitle())
            if title and not any(e.name == title for e in out):
                out.append(Entry(title, "window", lambda window=window: window,
                                 front=window is active))
        return out

    def overview(self) -> Dict[str, Any]:
        entries = self.entries()
        tabs = []
        for e in entries:
            if e.kind == "tab":
                tabs.append({**e.listed(), "plugins": []})
            elif e.kind == "plugin" and " > " in e.name and tabs:
                tabs[-1]["plugins"].append({k: v for k, v in e.listed().items()
                                            if k != "kind"})
        for t in tabs:
            del t["kind"]
            if not t["plugins"]:
                del t["plugins"]
        windows = [e.listed() for e in entries if e.kind == "window"
                   or (e.kind == "plugin" and " > " not in e.name)]
        return {"tabs": tabs, "windows": windows,
                "in_front": [e.name for e in entries if e.front]}

    def find(self, name: str) -> Entry:
        """The entry ``name`` means: its exact name; a plugin's path (the panel the user sees
        first: an open section, a window, then any pinned); else the one name with it in."""
        entries = self.entries()
        for e in entries:
            if e.name == name:
                return e
        by_path = [e for e in entries if e.plugin and e.plugin == name]
        if by_path:
            by_path.sort(key=lambda e: (not e.front, "folded" in e.extra))
            return by_path[0]
        chosen = _pick("name", name, [(e.name, e.name) for e in entries])
        return next(e for e in entries if e.name == chosen)

    def changes(self, entry: Entry) -> str:
        """What setting a field of ``entry`` changes: "form" (run later), "view" or
        "session" (`CHANGES_SESSION`)."""
        if entry.kind == "plugin":
            return "form"
        if entry.name in CHANGES_SESSION:
            return "session"
        from .render_tab import RenderTab
        from .roi_tab import ROIHeader
        widget = entry.widget()
        if isinstance(widget, RenderTab) or entry.name in ("2D view", "3D view"):
            return "view"
        if isinstance(widget, QWidget) and widget.findChild(ROIHeader) is not None:
            return "session"
        return "form"

    # ------------------------------------------------------------------ read
    def read(self, name: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        entry = self.find(name)
        widget = entry.widget()
        if widget is None:
            raise ToolError(f"{entry.name} could not be built; its section says why")
        head = {"name": entry.name, "kind": entry.kind,
                **({"in_front": True} if entry.front else {})}
        if entry.kind == "plugin":
            out = self._read_panel(widget)
        else:
            out = read_widget(widget, self.control.session).as_dict()
            if entry.kind == "tab":
                out.update(self._tab_plugins(entry))
        if fields:
            out = _narrowed(out, fields)
        return {**head, **out}

    def _read_panel(self, panel) -> Dict[str, Any]:
        from .plugin_panel import PluginPanel
        if not isinstance(panel, PluginPanel):         # a chain's panel: read as a window
            return read_widget(panel, self.control.session).as_dict()
        try:
            settings = panel.form.value()
        except ValueError as error:                      # a field typed that does not read
            return {"plugin": panel.plugin.path, "problem": str(error),
                    "settings": form_values(panel.form)}
        return plugin_reading(type(panel.plugin), settings, self.control.session, panel)

    def _tab_plugins(self, entry: Entry) -> Dict[str, Any]:
        listed = [e for e in self.entries() if e.name.startswith(entry.name + " > ")]
        if not listed:
            return {}
        return {"plugins": [{"name": e.name, "plugin": e.plugin,
                             **({"open": True} if "folded" not in e.extra else {})}
                            for e in listed]}

    # ------------------------------------------------------------------- set
    def plugin_class(self, name: str):
        """The plugin class of the panel ``name`` means, or None for a window or tab."""
        entry = self.find(name)
        if entry.kind != "plugin":
            return None
        widget = entry.widget()
        plugin = getattr(widget, "plugin", None)
        return type(plugin) if plugin is not None and hasattr(widget, "form") else None

    def write(self, name: str, values: Dict[str, Any]) -> Dict[str, Any]:
        entry = self.find(name)
        widget = entry.widget()
        if widget is None:
            raise ToolError(f"{entry.name} could not be built")
        with _NoDialogs() as guard:
            if entry.kind == "plugin" and hasattr(widget, "form") and hasattr(widget, "plugin"):
                done = set_form(widget.form, values, widget.plugin.name or widget.plugin.path)
            else:
                done = self._write_widgets(widget, values, entry.name)
        out: Dict[str, Any] = {"name": entry.name, "set": done}
        if guard.closed:
            out["dialogs_closed"] = (f"setting it opened {', '.join(guard.closed)}, which "
                                     "was closed unanswered: ask the user to do it there")
        return out

    def _write_widgets(self, widget, values: Dict[str, Any], what: str) -> List[str]:
        reading = read_widget(widget, self.control.session)
        done = []
        for key, value in values.items():
            target = reading.widgets.get(key)
            if target is not None and key not in reading.tables:
                try:
                    set_control(target, value)
                except ToolError as error:
                    raise ToolError(f"{key}: {error}") from None
                done.append(key)
                continue
            form, rest = _form_of(reading, key)
            if form is not None:
                set_form(form, {rest: value}, key.rsplit("/", 1)[0] if "/" in key else what)
                done.append(key)
                continue
            names = [n for n in reading.widgets if n not in reading.tables] \
                + [f"{f}/<dotted name>" for f in reading.form_widgets]
            raise ToolError(f"{what} has no field {key!r}; it has " + ", ".join(names[:60]))
        return done

    # ------------------------------------------------------------------- focus
    def focus(self, name: str, fields=()) -> Dict[str, Any]:
        """``name`` forward; each of ``fields`` unfolded and outlined, scrolled to the first:
        what a guided step's *Do next* shows (docs/ai-integration.md)."""
        entry = self.find(name)
        widget = entry.widget()
        if widget is None:
            raise ToolError(f"{entry.name} could not be built")
        fields = [fields] if isinstance(fields, str) else [f for f in fields or () if f]
        targets = [self._field_widget(entry, widget, f) for f in fields]
        self._bring_forward(entry)
        if not targets:
            bring_into_view(widget, flash=False)
        for target in reversed(targets):            # the first scrolled to last: in view
            bring_into_view(target)
        return {"shown": entry.name, **({"fields": fields} if fields else {})}

    def _bring_forward(self, entry: Entry) -> None:
        c = self.control
        tab_name = entry.name.split(" > ")[0]
        names = [t.name for t in c.workspace.tabs]
        if entry.kind == "window" or tab_name not in names:
            return
        index = names.index(tab_name)
        c.show_tab(index)
        if " > " in entry.name:
            page = c.tab_page(index)
            for section, instance in zip(page.sections, page.tab.instances):
                title = _clean(section.button.text()) or instance.title()
                if f"{tab_name} > {title}" == entry.name:
                    page.open_section(instance.id)
        c.show()
        c.raise_()

    def _field_widget(self, entry: Entry, widget, field_name: str) -> QWidget:
        if entry.kind == "plugin" and hasattr(widget, "form"):
            leaf = _leaf(widget.form, field_name)
            if leaf is None:
                raise ToolError(f"{entry.name} has no field {field_name!r}; read_gui lists them")
            return leaf
        reading = read_widget(widget, self.control.session)
        target = reading.widgets.get(field_name)
        if isinstance(target, QWidget):
            return target
        form, rest = _form_of(reading, field_name)
        leaf = _leaf(form, rest) if form is not None else None
        if leaf is None:
            raise ToolError(f"{entry.name} has no field {field_name!r}; read_gui names them")
        return leaf


def _form_of(reading: Reading, key: str) -> Tuple[Any, str]:
    """The form a name like ``"Prepare/psf.kind"`` is in, and the field's dotted name."""
    for name, form in reading.form_widgets.items():
        if key.startswith(name + "/"):
            return form, key[len(name) + 1:]
    if len(reading.form_widgets) == 1 and "/" not in key:
        return next(iter(reading.form_widgets.values())), key
    return None, ""


def _narrowed(out: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """Only the names that start with one of ``fields`` (or have it after a / or a .), in
    every map and list that is by name."""
    wanted = [f.strip().lower() for f in fields if f.strip()]

    def keep(name: str) -> bool:
        low = name.lower()
        return any(low.startswith(w) or f"/{w}" in low or f".{w}" in low for w in wanted)

    narrowed: Dict[str, Any] = {}
    for key, value in out.items():
        if isinstance(value, dict) and key in ("settings", "fields", "choices", "hints",
                                                "defaults_where_different",
                                                "auto_resolves_to", "tables", "texts",
                                                "panels"):
            narrowed[key] = {k: v for k, v in value.items() if keep(k)}
        elif isinstance(value, list) and key in ("greyed", "not_read_now", "folded"):
            narrowed[key] = [k for k in value if keep(k)]
        elif key == "forms":
            narrowed[key] = {f: {k: v for k, v in form.items() if keep(k) or keep(f)}
                             for f, form in value.items()}
        elif key in ("notes", "buttons", "figures"):
            continue                                  # not by name: left out when narrowed
        else:
            narrowed[key] = value
    return {k: v for k, v in narrowed.items() if v not in ({}, [])}
