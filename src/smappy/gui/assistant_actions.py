"""What the control window's menus and buttons do, as the assistant's actions.

The assistant does not click: it calls (`assistant.tools.Action`).  Each
entry here is one click of the GUI -- a menu item, the Load plugin's *Add*,
the Render tab's *+ > image layer* -- done by the method the click calls,
with what the click would have asked for (a file, a layer) as an argument,
so that no dialog holds the call: every call runs on the GUI thread, one at
a time (`assistant_server`), and a modal dialog would stop them all until
someone closed it.

A menu item is either here or in `LEFT_OUT` with the reason, and
`tests/test_assistant_gui.py` walks the real menus to keep it so: a new item
that is neither fails there rather than being out of the assistant's reach
without anyone deciding it.  The Plugins menu is `open_plugin`'s and
`run_plugin`'s, and the windows are `open_window`'s too.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..assistant.tools import Action, ToolError

PATH = {"type": "string", "description": "the file"}
PATHS = {"type": ["string", "array"], "items": {"type": "string"},
         "description": "a file, or several"}
LAYER = {"type": "integer", "description": "a layer's index, as session_state lists it"}

#: The menu items that are not actions, and why: a modal dialog with no
#: version that asks nothing, a file written outside the output folder, the
#: assistant's own connection.
LEFT_OUT = {
    "File/Save": "writes over the open file: the save tool writes under the output folder",
    "File/Save as": "the save tool, which writes under the output folder",
    "File/Workspace/Save": "writes a file outside the output folder",
    "File/Workspace/Save as": "writes a file outside the output folder",
    "File/Workspace/Recent": "a list: File/Workspace/Open takes the file",
    "File/Preferences": "a modal dialog: the user's own settings",
    "File/Quit": "ends the program, and the assistant's connection with it",
    "Edit/Undo steps": "Edit/Undo takes how many",
    "Edit/Redo steps": "Edit/Redo takes how many",
    "Plugins": "open_plugin and run_plugin",
    "Assistant": "the assistant's own connection and settings",
    "Help/Tutorials": "videos in the user's browser",
    "Help/Save a bug report": "a dialog asking where to save it",
    "Help/Check for updates": "a dialog that may install and restart",
    "Help/About SMAPpy": "a modal dialog",
}


#: The plugins' own window buttons (`Plugin.window`, `window_buttons`), by
#: the opener they name in `Session.window_openers`: the action or the
#: tool that does the same.
WINDOW_BUTTONS = {
    "open files": "File/Open",
    "add files": "File/Add",
    "bead calibration": "open_window",
    "psf calibration": "open_window",
}


def menu_name(text: str) -> str:
    """A menu item's text as an action's name: no "...", no "&"."""
    return text.replace("&", "").removesuffix("...").strip()


def menu_items(window) -> List[str]:
    """Every item of the window's menus, by path ("File/Workspace/Open"):
    what the test checks the actions and `LEFT_OUT` against."""
    out: List[str] = []

    def walk(menu, prefix):
        for action in menu.actions():
            if action.isSeparator() or not action.text():
                continue
            name = f"{prefix}/{menu_name(action.text())}"
            if name in LEFT_OUT:                    # a whole submenu left out
                out.append(name)
            elif action.menu() is not None and action.menu().actions():
                walk(action.menu(), name)
            else:
                out.append(name)

    for action in window.menuBar().actions():
        top = menu_name(action.text())
        if top in LEFT_OUT:
            out.append(top)
        elif action.menu() is not None:
            walk(action.menu(), top)
    return out


def _paths(path) -> List[str]:
    paths = [path] if isinstance(path, str) else list(path or [])
    missing = [p for p in paths if not Path(p).expanduser().exists()]
    if not paths or missing:
        raise ToolError(f"no file {missing[0] if missing else '(none given)'}")
    return [str(Path(p).expanduser()) for p in paths]


def _render_tab(window):
    from .render_tab import RenderTab
    for i in range(window.tabs.count()):
        if isinstance(window.tabs.widget(i), RenderTab):
            return i, window.tabs.widget(i)
    raise ToolError("the workspace has no Render tab")


def actions_of(window) -> Dict[str, Action]:
    """The actions of one control window (`ControlWindow`), by name."""
    session = window.session

    def opened(paths, append):
        window.open_dropped(_paths(paths), append=bool(append))
        return ("tables load in the background: session_state shows them when "
                "they are in.  An image is opened as a source, shown by no "
                "layer until Render/Add image layer or set_layer; a long stack "
                "goes to a fitter's file on the Localize tab")

    def restore_open():
        path = session.path
        if not path or Path(path).suffix.lower() not in (".h5", ".hdf5"):
            raise ToolError("the open file is not a smappy HDF5 file: only those "
                            "carry a session")
        window.restore_from(Path(path))

    def workspace_from(path):
        missing = window.restore_state_from_file(_paths(path)[0])
        return {"not installed": missing} if missing else None

    def undo(steps=1):
        if not session.can_undo:
            raise ToolError("nothing to undo")
        session.undo(int(steps))

    def redo(steps=1):
        if not session.can_redo:
            raise ToolError("nothing to redo")
        session.redo(int(steps))

    def frame_time(frame_time_ms, file=0):
        if not 0 <= int(file) < len(session.files):
            raise ToolError(f"no file {file}; there are {len(session.files)}")
        session.set_frame_time(int(file), None if frame_time_ms is None
                               else float(frame_time_ms))

    def show_tab(name):
        names = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        found = [i for i, n in enumerate(names) if n.lower() == str(name).strip().lower()]
        if not found:
            raise ToolError(f"no tab {name!r}; there are {', '.join(names)}")
        window.tabs.setCurrentIndex(found[0])
        window.show()
        window.raise_()

    def render_tab():
        index, tab = _render_tab(window)
        window.tabs.setCurrentIndex(index)
        return tab

    def select_layer(layer):
        if not 0 <= int(layer) < len(session.layers):
            raise ToolError(f"no layer {layer}; there are {len(session.layers)}")
        render_tab().strip.select(int(layer))

    def add_locs_layer():
        tab = render_tab()
        tab.strip._add()
        return {"layer": len(session.layers) - 1}

    def add_image_layer(image=None):
        tab = render_tab()
        chosen = None
        if image is not None:
            chosen = next((im for i, im in enumerate(session.images)
                           if im.name == image or str(i) == str(image)), None)
            if chosen is None:
                names = ", ".join(repr(im.name) for im in session.images) or "none"
                raise ToolError(f"no image {image!r}; there are {names}")
        tab._add_image()
        layer = session.layers[-1]
        if chosen is not None:
            layer.show_image(chosen, 0)
            tab.strip.rebuild()
            tab.strip.select(len(session.layers) - 1)
            session.changed("layer")
        return {"layer": len(session.layers) - 1,
                "image": None if layer.image.is_empty else layer.image.name}

    def remove_layer(layer):
        if not 0 <= int(layer) < len(session.layers):
            raise ToolError(f"no layer {layer}; there are {len(session.layers)}")
        if len(session.layers) == 1:
            raise ToolError("the last layer stays")
        session.remove_layer(int(layer))

    def fit_file(path):
        if not window.fit_file(Path(_paths(path)[0])):
            raise ToolError("no fitter on a tab takes it: "
                            + window.statusBar().currentMessage())

    def copied(done):
        if not done:
            raise ToolError("nothing to copy")

    a = [
        # ---- File
        Action("File/Open", "File > Open with these files: localization tables "
               "replace what is open (append=true adds them); an image of a few "
               "frames becomes a source no layer shows yet; a long stack goes to "
               "a fitter's file.  The Load plugin's Open.", opened,
               {"path": PATHS, "append": {"type": "boolean"}}, ("path",)),
        Action("File/Add", "The Load plugin's Add: these files added to what is "
               "open -- tables as more files, images as sources.",
               lambda path: opened(path, True), {"path": PATHS}, ("path",)),
        Action("File/Open and restore", "Open a smappy HDF5 file in place of what "
               "is open and bring back the session it was saved with: tabs, "
               "layers, views, windows.", lambda path: window.open_and_restore(
                   _paths(path)[0]), {"path": PATH}, ("path",)),
        Action("File/Restore saved session", "The session the open smappy HDF5 "
               "file was saved with: tabs, layers, views, windows.", restore_open),
        Action("File/Session name", "Name the session, and colour it (a name of "
               "Session.COLOURS, '' for none): every window's title says it.",
               lambda name, colour=None: session.rename(name, colour),
               {"name": {"type": "string"}, "colour": {"type": "string"}},
               ("name",), shows=True),
        Action("File/File properties", "A file's frame time, which turns frames "
               "into seconds; null: not known.", frame_time,
               {"frame_time_ms": {"type": ["number", "null"]},
                "file": {"type": "integer", "description": "the file's number, default 0"}},
               ("frame_time_ms",)),
        Action("File/Open in a new window", "The session as it is now in a new "
               "instance of the program.", lambda: window.open_in_new_window()),
        Action("File/Workspace/Open", "A saved workspace: its tabs and their "
               "values replace these.", lambda path: window.load_gui_from(
                   _paths(path)[0]), {"path": PATH}, ("path",)),
        Action("File/Workspace/Reset", "The shipped workspace, now and at the "
               "next start.", window.load_default_gui),
        Action("File/Workspace/From data file", "The workspace a localization "
               "file was saved with.", workspace_from, {"path": PATH}, ("path",)),
        Action("File/Microscope manager", "The microscope entries' window.",
               window.open_microscope_manager, shows=True),
        # ---- Edit
        Action("Edit/Undo", "Undo changes to the table.", undo,
               {"steps": {"type": "integer", "description": "default 1"}}),
        Action("Edit/Redo", "Redo what was undone.", redo,
               {"steps": {"type": "integer", "description": "default 1"}}),
        Action("Edit/Copy", "Copy the selected text, else the picture of the "
               "window in front, to the clipboard.",
               lambda: copied(window.copy()), shows=True),
        Action("Edit/Copy rendered image", "The render to the clipboard.",
               lambda: copied(window.copy(window.render_window)), shows=True),
        Action("Edit/Copy window as image", "The window last in front, as a "
               "picture, to the clipboard.",
               lambda: copied(window.copy_window_image()), shows=True),
        # ---- View
        Action("View/2D view", "The render window, in front.", window.show_2d,
               shows=True),
        Action("View/3D view", "The 3D view, shown or hidden.", window.toggle_3d,
               shows=True),
        Action("View/ROI manager", "The ROI manager.", window._open_manager, shows=True),
        Action("View/Tab", "A tab of the control window in front: File, "
               "Localize, Render, Analysis, ...", show_tab,
               {"name": {"type": "string"}}, ("name",), shows=True),
        # ---- the Render tab's buttons
        Action("Render/Select layer", "The Render tab, editing this layer.",
               select_layer, {"layer": LAYER}, ("layer",), shows=True),
        Action("Render/Add localization layer", "+ > localizations: a copy of the "
               "layer the tab edits, which it then edits.", add_locs_layer, shows=True),
        Action("Render/Add image layer", "+ > image layer: the image given (a name "
               "or index of session_state's images), else the fit's kept frames, "
               "else the first image opened.  set_layer sets its colour.",
               add_image_layer, {"image": {"type": ["string", "integer"]}}, shows=True),
        Action("Render/Remove layer", "- : the layer and its settings removed.",
               remove_layer, {"layer": LAYER}, ("layer",)),
        Action("Render/Draw line", "The render window, drawing a line ROI.",
               window.draw_line, shows=True),
        # ---- the Localize tab
        Action("Localize/Fit file", "A raw image stack into the pinned fitter's "
               "file, its tab in front, as a dropped stack goes: run_plugin "
               "fits it.", fit_file, {"path": PATH}, ("path",), shows=True),
        # ---- Help
        Action("Help/Plugin documentation", "The help window.",
               window.show_plugin_docs, shows=True),
        Action("Help/Show the log folder", "The log folder, in the file browser.",
               window.show_log_folder, shows=True),
    ]
    out = {action.name: action for action in a}
    # the plugins' own window buttons (`Plugin.window`, `window_buttons`), by
    # the plugin and the button: the Load plugin's two are File/Open and Add
    out["File/Load/Load: Open"] = out["File/Open"]
    out["File/Load/Load: Add"] = out["File/Add"]
    return out
