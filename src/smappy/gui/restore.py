"""A saved file's session brought back: *File > Open and restore*.

A localization file keeps its tabs and their values (`/gui/state`, what
*Workspace > From data file* has always put back) and, beside them, a
session block (`/gui/session`, `handoff.session_block`): the layers and how
each is drawn, the merge settings, the 2D view's range, the 3D view and its
slab, which windows were open -- the 3D view, the ROI manager, plugins opened
from the Plugins menu with their settings -- and where every window was.

**Restoring is a command of its own, never what Open does.**  Opening a
colleague's file should not rearrange one's tabs and windows, and plain
loading stays one click; the restore is asked for, for a file being opened
or for the one already open.

**Windows come back onto the screens there are.**  A file saved on a desk
with two monitors and opened on a laptop would put half its windows where
nothing is drawn, so each saved rectangle is put on the screen it overlaps
most -- the nearest when it overlaps none -- and shrunk to fit it
(`clamp_rect`).  Plain rectangles, not Qt's `saveGeometry` bytes: those are
one platform's and cannot be clamped before they are applied.
"""
from __future__ import annotations

import traceback
from typing import Dict, List, Optional, Sequence, Tuple

Rect = Tuple[int, int, int, int]          # x, y, width, height

#: what a window is called in the block; a plugin window is "plugin:" + path
CONTROL, RENDER, VIEW3D, MANAGER, PLUGIN = "control", "render", "3d", "roi manager", "plugin:"

# a title bar's height, kept free above a window's client area: the
# rectangles are the client area (`geometry`), which Qt sets without one
TITLE = 28


def clamp_rect(rect: Sequence[int], screens: Sequence[Sequence[int]],
               top: int = TITLE) -> Rect:
    """``rect`` moved onto one of ``screens`` (their free areas, x, y, w, h)
    and shrunk to fit it, with ``top`` pixels kept above it for the title
    bar.  The screen is the one ``rect`` overlaps most, or the nearest to
    its centre when it overlaps none."""
    x, y, w, h = (int(v) for v in rect)
    if not screens:
        return x, y, w, h

    def overlap(s) -> int:
        sx, sy, sw, sh = s
        return max(0, min(x + w, sx + sw) - max(x, sx)) * \
            max(0, min(y + h, sy + sh) - max(y, sy))

    def distance(s) -> float:
        sx, sy, sw, sh = s
        cx, cy = x + w / 2, y + h / 2
        dx = max(sx - cx, 0, cx - (sx + sw))
        dy = max(sy - cy, 0, cy - (sy + sh))
        return dx * dx + dy * dy

    best = max(screens, key=overlap)
    if overlap(best) == 0:
        best = min(screens, key=distance)
    sx, sy, sw, sh = (int(v) for v in best)
    w, h = max(1, min(w, sw)), max(1, min(h, sh - top))
    x = min(max(x, sx), sx + sw - w)
    y = min(max(y, sy + top), sy + sh - h)
    return x, y, w, h


def _screens() -> List[Rect]:
    from PySide6.QtGui import QGuiApplication
    out = []
    for screen in QGuiApplication.screens():
        g = screen.availableGeometry()
        out.append((g.x(), g.y(), g.width(), g.height()))
    return out


def _window_state(window) -> Dict:
    g = window.normalGeometry() if window.isMaximized() else window.geometry()
    if not g.isValid():
        g = window.geometry()
    return {"rect": [g.x(), g.y(), g.width(), g.height()],
            "maximized": bool(window.isMaximized()),
            "visible": bool(window.isVisible())}


def windows(control) -> Dict[str, Dict]:
    """Where each window is and whether it is open; a plugin window's
    settings with it, and the tab in front with the control window."""
    from .. import plugins as registry
    out = {CONTROL: _window_state(control), RENDER: _window_state(control.render_window)}
    out[CONTROL]["tab"] = int(control.tabs.currentIndex())
    if control.view3d_window is not None:
        out[VIEW3D] = _window_state(control.view3d_window)
    manager = getattr(control.roi_tab, "manager", None) if control.roi_tab else None
    if manager is not None:
        out[MANAGER] = _window_state(manager)
    for path, window in control._plugin_windows.items():
        if not window.isVisible():
            continue
        state = _window_state(window)
        try:
            state["settings"] = registry.settings_values(window.panel.form.value())
        except Exception:          # a form that cannot say: the window alone
            traceback.print_exc()
        out[PLUGIN + path] = state
    return out


def session_block(control) -> Dict:
    """The block `Session.save` writes for the GUI: the session's own half
    (`handoff.session_block`) with the 2D view and the windows."""
    from .. import handoff
    view3d = control.view3d_window
    return handoff.session_block(
        control.session, view=control.render_window.view.view.viewRange(),
        projection=view3d.view.projection if view3d is not None else None,
        windows=windows(control))


def place(window, state: Optional[Dict], screens: Optional[List[Rect]] = None) -> None:
    """Put ``window`` where ``state`` had it, on a screen there is."""
    if not state or not state.get("rect"):
        return
    x, y, w, h = clamp_rect(state["rect"], _screens() if screens is None else screens)
    if window.isMaximized():
        window.showNormal()
    window.setGeometry(x, y, w, h)
    if state.get("maximized"):
        window.showMaximized()


def restore(control, path, gui_state: Optional[Dict], block: Optional[Dict]) -> List[str]:
    """Bring back what ``path`` was saved with onto the session the control
    window shows, which holds that file's table.  Returns what could not be
    restored (plugins not installed here), for the caller to say."""
    from PySide6.QtCore import QRectF
    from .. import handoff
    from .. import plugins as registry
    missing: List[str] = []
    if gui_state:
        missing += control.apply_workspace_state(gui_state, label=f"from {path.name}")
    if not block:
        return missing
    session = control.session
    handoff.apply_session_block(session, block)
    view = block.get("view")
    if view:
        (x0, x1), (y0, y1) = view
        canvas = control.render_window.view
        canvas.view.setRange(QRectF(x0, y0, x1 - x0, y1 - y0), padding=0)
        canvas.schedule()
    screens = _screens()
    states = block.get("windows") or {}
    place(control, states.get(CONTROL), screens)
    tab = (states.get(CONTROL) or {}).get("tab")
    if tab is not None and 0 <= int(tab) < control.tabs.count():
        control.tabs.setCurrentIndex(int(tab))
    place(control.render_window, states.get(RENDER), screens)
    view3d = states.get(VIEW3D)
    if view3d and view3d.get("visible"):
        control.show_3d()
        window = control.view3d_window
        # the window keeps its own copy of the projection, taken when it
        # was made -- an older window is told the whole of it here
        import copy
        window.view.projection = copy.deepcopy(session.projection)
        window.view.render()
        place(window, view3d, screens)
    elif control.view3d_window is not None and control.view3d_window.isVisible():
        control.view3d_window.close()
    manager = states.get(MANAGER)
    if manager and manager.get("visible") and control.roi_tab is not None:
        control.roi_tab.open_manager()
        place(control.roi_tab.manager, manager, screens)
    for name, state in states.items():
        if not name.startswith(PLUGIN) or not state.get("visible", True):
            continue
        plugin = name[len(PLUGIN):]
        if plugin not in registry.refs():
            missing.append(plugin)
            continue
        window = control.open_plugin_window(plugin)
        if window is None:
            continue
        if state.get("settings") is not None:
            try:
                cls = registry.get(plugin)
                window.panel.form.set(registry.settings_from(cls.Settings, state["settings"]))
            except Exception:       # the window opens with what it had
                traceback.print_exc()
        place(window, state, screens)
    return missing
