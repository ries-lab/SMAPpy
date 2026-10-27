"""The assistant's tools: plain functions over a `Session`, answering in JSON.

Every tool is a method of `Assistant` marked with `tool`, which records its
description and its arguments' JSON schema -- what an MCP client is shown --
so the protocol layer only lists them and calls them.  Nothing here imports
Qt or the protocol: a test, a script and the GUI's server
(`gui/assistant_server.py`) call the same methods.

**Reading, trying, changing.**  Most tools only read.  `run_plugin` with
``apply=false`` *tries*: it runs on a copy of the session (`Session.scratch`,
which shares the arrays and copies nothing) and hands back what the plugin
said, its numbers and its figures, while the user's session stays as it was.
A *change* -- a run applied, a file loaded, a filter set, an undo -- is made
only in *act* mode; in *suggest* mode it is recorded as a proposal, which the
GUI offers the user to run.  The assistant cannot switch the mode: the
user sets it, here with ``--mode`` and in the GUI with its pane's switch.

**View changes in the running GUI.**  With the GUI attached (a ``viewer``),
the changes to what is shown rather than to the data -- a filter, grouping,
the 2D view moved by `show` -- are made in *suggest* mode too: the user sees
them at once, and a proposal to look somewhere would be a click to make a
click.  Headless there is nobody looking, and they stay proposals.

**Files are written only under the output folder.**  A setting that names a
file to write (``kind="save_file"``) must name one under it, in either mode
and on a copy too, since a plugin run on a copy still writes its file; an
export (``File/Export/...``) must name its file.  A plugin given an empty
path elsewhere writes where it always does -- a fit beside its images --
which is what a user who started it would expect.
"""
from __future__ import annotations

import base64
import dataclasses
import inspect
import io
import itertools
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import index, knowledge

MODES = ("suggest", "act")
# a figure is drawn at this size and resolution: legible to a model, a few
# tens of kilobytes rather than the hundreds a print-quality PNG takes
FIGURE_DPI = 90
MAX_SOURCE = 60_000          # characters of source in one answer
MAX_RESULTS = 50             # results kept for `figure`; the oldest go


@dataclass
class Image:
    """A PNG to show the model, not to describe in JSON."""
    png: bytes
    caption: str = ""

    def base64(self) -> str:
        return base64.b64encode(self.png).decode("ascii")


@dataclass
class ToolSpec:
    name: str
    description: str
    params: Dict[str, Dict[str, Any]]      # JSON schema of each argument
    required: List[str]
    function: Callable

    def schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": self.params,
                "required": self.required, "additionalProperties": False}


def tool(description: str, needs: str = "", **params: Dict[str, Any]):
    """Mark a method as a tool.  Each keyword is an argument's JSON schema;
    one with a ``default`` in the method's signature is optional.
    ``needs="viewer"`` (or ``"panels"``, ``"windows"``) lists the tool only
    where the assistant has that -- the running GUI's view, its plugin
    windows, its other windows: a
    model shown a tool that always refuses would keep trying it."""
    def mark(function):
        function._tool = (description, params)
        function._needs = needs
        return function
    return mark


class Later:
    """An answer that comes later: a run the GUI makes through the plugin's
    own panel, on its thread, ends after the call has returned
    (`Assistant.runner`).  ``then(callback)`` is told ``(value, None)`` or
    ``(None, why)`` once, whenever that is -- at once if it has been."""

    def __init__(self):
        self._outcome: Optional[tuple] = None
        self._callbacks: List[Callable] = []

    def then(self, callback: Callable) -> None:
        if self._outcome is not None:
            callback(*self._outcome)
        else:
            self._callbacks.append(callback)

    def resolve(self, value) -> None:
        self._end((value, None))

    def fail(self, why: str) -> None:
        self._end((None, why))

    def _end(self, outcome: tuple) -> None:
        if self._outcome is not None:
            return                      # once: a run ends one way
        self._outcome = outcome
        callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback(*outcome)


class ToolError(Exception):
    """A call that cannot be done, with what to do instead.  The client gets
    the message as the tool's answer, marked as an error, and not a
    traceback: the model should read *why* and correct the call."""


@dataclass
class Proposal:
    id: str
    tool: str
    arguments: Dict[str, Any]
    reason: str
    time: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    # what the user did with it: "proposed", "opened" (its panel, filled in),
    # "done" or "dismissed" -- so the model can ask what became of it
    status: str = "proposed"
    # a run the user made from the panel it opened: its kept result (for
    # `figure`) and the settings it ran with, which the user may have changed
    result: str = ""
    ran_with: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Kept:
    """A run's result, kept so its figures can be asked for afterwards."""
    id: str
    path: str
    applied: bool
    result: Any


def _plain(value):
    from ..batch import json_safe
    return json_safe(value)


def _pick(name: str, value: str, options: Sequence) -> str:
    """The entry of ``options`` -- ``(value, label)`` pairs -- that ``value``
    means (`Assistant._named`)."""
    low = value.strip().lower()
    for exact, label in options:
        if value == exact:
            return exact
    for exact, label in options:
        if low in (exact.lower(), label.lower()):
            return exact
    found = {exact for exact, label in options
             if low in exact.lower() or low in label.lower()}
    if len(found) == 1:
        return found.pop()
    listed = ", ".join(repr(exact) if exact == label else f"{exact!r} ({label})"
                       for exact, label in options)
    raise ToolError(f"{name}: no {value!r}" + (" that is one entry" if found else "")
                    + f"; it is one of {listed}")


def source_folder() -> Path:
    """The installed package's folder -- a checkout's ``src/smappy``, or
    ``site-packages/smappy`` after pip or uv: the Python code that runs, with
    the plugins' pages beside it.  The C++ is there compiled only."""
    from .. import __file__ as package
    return Path(package).resolve().parent


def _version() -> str:
    from .. import __version__
    return str(__version__)


def source_note() -> str:
    """What a client is told about the code, after `INSTRUCTIONS`: where it
    is.  A coding agent was not told, and grepping it is what it is best at."""
    return (f"\nSMAPpy {_version()}'s own code -- the version that runs -- is in "
            f"{source_folder()}.  read_source shows a module or a plugin; an agent "
            "with file access may also read and grep that folder.  The C++ "
            "(fitting, rendering, neighbour search) is installed compiled only: "
            f"its source is on GitHub, ries-lab/SMAPpy at tag v{_version()}.\n")


def _within(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


class Assistant:
    """The tools, over one session.

    ``output`` is the folder files may be written to; ``knowledge`` the
    folder of remembered notes.  ``viewer`` is the running GUI's 2D view, or
    None headless: anything with ``frame(xrange, yrange)``, ``reset()`` and
    ``current()`` -> ``((x0, x1), (y0, y1), pixel_nm)``.  ``panels`` opens a
    plugin's window in the GUI filled in, ``panels(path, settings)``; None
    headless.  ``windows`` the GUI's own windows that are not plugins, a
    mapping of name to ``open(roi_id=None)``; None headless.  ``runner``
    makes an applied run in the GUI through the plugin's own panel --
    shown, filled in, its Run pressed -- ``runner(path, cls, settings)`` ->
    a `Later` of the result, or None where it cannot; None headless.
    """

    def __init__(self, session=None, mode: str = "act", output=None,
                 knowledge_folder=None, viewer=None, panels=None,
                 missing_tools=None, windows=None):
        if mode not in MODES:
            raise ValueError(f"mode is one of {', '.join(MODES)}, not {mode!r}")
        if session is None:
            from ..session import Session
            session = Session()
        self.session = session
        self.mode = mode
        self.output = Path(output or Path.cwd()).expanduser()
        self.knowledge = Path(knowledge_folder or knowledge.default_folder()).expanduser()
        self.viewer = viewer
        self.panels = panels
        self.windows = windows
        self.runner = None
        self.missing_tools = Path(missing_tools or knowledge.missing_tools_file()).expanduser()
        self.proposals: List[Proposal] = []
        self.results: List[Kept] = []
        self._ids = itertools.count(1)

    # ------------------------------------------------------------- the registry
    @classmethod
    def specs(cls) -> Dict[str, ToolSpec]:
        out: Dict[str, ToolSpec] = {}
        for name, function in inspect.getmembers(cls, inspect.isfunction):
            marked = getattr(function, "_tool", None)
            if marked is None:
                continue
            description, params = marked
            signature = inspect.signature(function)
            required = [p for p in params
                        if signature.parameters[p].default is inspect.Parameter.empty]
            out[name] = ToolSpec(name, description, params, required, function)
        return out

    def tools(self) -> Dict[str, ToolSpec]:
        """The tools this assistant offers: `specs` less those that need
        what it does not have."""
        return {name: spec for name, spec in self.specs().items()
                if not spec.function._needs
                or getattr(self, spec.function._needs, None) is not None}

    def call(self, name: str, arguments: Optional[Dict[str, Any]] = None):
        """Call a tool by name; a `ToolError` for a call that cannot be made."""
        tools = self.tools()
        spec = tools.get(name)
        if spec is None:
            raise ToolError(f"no tool {name!r}; there are {', '.join(sorted(tools))}")
        arguments = dict(arguments or {})
        unknown = sorted(set(arguments) - set(spec.params))
        if unknown:
            raise ToolError(f"{name} has no argument {', '.join(unknown)}; "
                            f"it takes {', '.join(spec.params) or 'none'}")
        missing = [p for p in spec.required if p not in arguments]
        if missing:
            raise ToolError(f"{name} needs {', '.join(missing)}")
        return spec.function(self, **arguments)

    # ------------------------------------------------------------------ helpers
    def _propose(self, tool_name: str, arguments: Dict[str, Any], reason: str) -> Dict:
        proposal = Proposal(f"p{next(self._ids)}", tool_name, arguments, reason)
        self.proposals.append(proposal)
        return {"proposed": proposal.as_dict(),
                "note": "suggest mode: nothing was changed.  The user decides; "
                        "say what you propose and why."}

    def _changing(self) -> bool:
        return self.mode == "act"

    def _viewing(self) -> bool:
        """Whether a change to what is shown is made: in act mode, and in
        suggest mode with the GUI attached (the module's docstring)."""
        return self._changing() or self.viewer is not None

    def proposal(self, ident: str) -> Proposal:
        found = next((p for p in self.proposals if p.id == ident), None)
        if found is None:
            raise ToolError(f"no proposal {ident!r}")
        return found

    def settings_for(self, path: str, values: Optional[Dict[str, Any]] = None):
        """A plugin's settings from a call's flat values, files to write
        under the output folder as `run_plugin` makes them: what the GUI
        fills a proposal's panel with."""
        cls = self._plugin(path)
        return self._settings(cls, self._in_output(cls, values))

    def ran(self, ident: str, result, path: str = "") -> str:
        """The user ran a proposal from its panel: it is done, and its result
        is kept as any run's is.  Returns the result's id."""
        from ..plugins import settings_values
        proposal = self.proposal(ident)
        kept = self._keep(path or proposal.arguments.get("path") or "chain", True, result)
        proposal.status, proposal.result = "done", kept.id
        if getattr(result, "settings", None) is not None:
            try:
                proposal.ran_with = _plain(settings_values(result.settings))
            except Exception:           # a chain's settings are not flat values
                proposal.ran_with = None
        return kept.id

    def carry_out(self, ident: str):
        """Do what a proposal proposed, as the user's decision: the call is
        made in act mode whatever the switch says, and the proposal is done."""
        proposal = self.proposal(ident)
        mode, self.mode = self.mode, "act"
        try:
            answer = self.call(proposal.tool, proposal.arguments)
        finally:
            self.mode = mode
        proposal.status = "done"
        return answer

    def _plugin(self, path: str):
        from .. import plugins
        try:
            return plugins.get(path)
        except KeyError:
            near = index.search(path.replace("/", " "), limit=5)
            hint = ", ".join(e["path"] for e in near) or "find_plugins lists them"
            raise ToolError(f"no plugin {path!r}; perhaps {hint}") from None

    def _options(self, cls) -> Dict[str, List[tuple]]:
        """Each field's list of names, as ``(value, label)``: a preset's
        saved names, a field's choices.  The one list `describe_plugin`
        shows and `_named` checks against."""
        from .. import presets as store
        from ..batch import _flat_specs
        out: Dict[str, List[tuple]] = {}
        for name, spec in _flat_specs(cls.specs()).items():
            info = spec.info
            try:
                if info.kind == "preset" and info.store is not None:
                    options = [(n, n) for n in store.names(info.store())]
                elif info.choices is not None:
                    raw = info.choices() if callable(info.choices) else info.choices
                    options = [(str(c[0]), str(c[1])) if isinstance(c, (tuple, list))
                               else (str(c), str(c)) for c in raw]
                else:
                    continue
            except Exception:           # a list that needs a session to be made
                continue
            if options:
                out[name] = options
        return out

    def _named(self, cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """``values`` with each name from a list -- a preset, one of a
        field's choices -- made the entry it means.  An unknown name used to
        pass and be ignored: "Nuclear pores" for the preset "Nuclear pores
        (Nup96)" filled nothing, and Simulate made its demo scene.  So: the
        name exactly, or a choice's label as the form shows it; else the one
        entry that contains it, case aside; else a refusal listing them."""
        options = self._options(cls)
        out = dict(values)
        for name, value in values.items():
            if name in options and isinstance(value, str) and value:
                out[name] = _pick(name, value, options[name])
        return out

    def _settings(self, cls, values: Optional[Dict[str, Any]]):
        """Settings from flat dotted values, refusing a name the plugin does
        not have -- `settings_from` drops it silently, which is right for a
        file saved by an older version and wrong for a model that guessed."""
        from ..batch import Problem, _check_values, _flat_specs
        from ..plugins import settings_from
        values = self._named(cls, dict(values or {}))
        problems: List[Problem] = []
        _check_values(values, _flat_specs(cls.specs()), cls.path, problems)
        errors = [p.message for p in problems if p.level == "error"]
        if errors:
            raise ToolError("; ".join(errors) + ".  describe_plugin lists the fields.")
        settings = settings_from(cls.Settings, values)
        # what the form does on an edit: the plugin's `react` fills other
        # fields -- a preset fills every setting.  Without it a preset named
        # here was silently ignored, and Simulate made its demo scene.  The
        # values given win over what was filled, as an edit after the preset.
        filled: Dict[str, Any] = {}
        plugin = cls()
        for name in values:
            try:
                filled.update(plugin.react(name, settings) or {})
            except (ValueError, TypeError):
                pass                       # as the panel's `_react`
        if filled:
            settings = settings_from(cls.Settings, {**filled, **values})
        return settings

    def _in_output(self, cls, values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """``values`` with each file to write that is given relative made
        absolute under the output folder: the plugin would otherwise write it
        relative to wherever the server happened to start."""
        from ..batch import _flat_specs
        out = dict(values or {})
        for name, spec in _flat_specs(cls.specs()).items():
            value = out.get(name)
            if spec.info.kind == "save_file" and isinstance(value, str) and value \
                    and not Path(value).expanduser().is_absolute():
                out[name] = str(self.output / value)
        return out

    def _writes(self, cls, settings, relative: bool = True) -> List[Path]:
        """The files ``settings`` will write, refused outside the output
        folder or, for an export, left empty.  ``relative=False`` refuses a
        relative path too: a chain's steps are read as written, and a
        relative path there is relative to wherever the server started."""
        from ..plugins import settings_values
        from ..batch import _flat_specs
        specs = _flat_specs(cls.specs())
        values = settings_values(settings)
        out: List[Path] = []
        for name, spec in specs.items():
            if spec.info.kind != "save_file":
                continue
            value = values.get(name)
            if not value:
                if cls.path.startswith("File/Export/"):
                    raise ToolError(f"{cls.path} writes a file: set {name} to a "
                                    f"path under {self.output}")
                continue
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                if not relative:
                    raise ToolError(f"{name}: give the file as an absolute path "
                                    f"under {self.output}")
                path = self.output / path
            if not _within(path, self.output):
                raise ToolError(f"{name}: {path} is outside the output folder, "
                                f"{self.output}; files are written only under it")
            out.append(path)
        return out

    def _keep(self, path: str, applied: bool, result) -> Kept:
        kept = Kept(f"r{next(self._ids)}", path, applied, result)
        self.results.append(kept)
        del self.results[:-MAX_RESULTS]
        return kept

    def _answer(self, kept: Kept, seconds: float) -> Dict[str, Any]:
        result = kept.result
        plots = result.figures()
        return {"result": kept.id, "plugin": kept.path,
                "applied": kept.applied, "seconds": round(seconds, 2),
                "text": result.text or "", "warnings": list(getattr(result, "warnings", []) or []),
                "data": _plain(result.data or {}),
                "values": _plain(getattr(result, "values", None) or {}),
                "changed_table": result.locs is not None or bool(result.files),
                "written": str(result.written) if result.written else None,
                "figures": [{"index": i, "name": p.name or "main"}
                            for i, p in enumerate(plots)]}

    def _shown(self, layer: int) -> Dict[str, Any]:
        """How many rows the layer's table has and how many pass its filter:
        blinks when it is grouped, as the plugins that read it see them."""
        try:
            ctx = self.session.context(layer)
            return {"rows": len(ctx.locs), "shown": int(ctx.selection.mask.sum()),
                    "rows_are": "blinks" if self.session.layers[layer].grouped
                    else "localizations"}
        except Exception as error:
            return {"shown": None, "why": f"{type(error).__name__}: {error}"}

    def _layer(self, layer: int):
        layers = self.session.layers
        if not 0 <= layer < len(layers):
            raise ToolError(f"no layer {layer}; there are {len(layers)} (0..{len(layers) - 1})")
        return layers[layer]

    # ------------------------------------------------------------------ reading
    @tool("What is open: the files, the table's columns, where the data lie "
          "(extent_nm) and, in the GUI, what the 2D view shows, each layer's "
          "filter and grouping and how many localizations it shows, the ROIs, "
          "what can be undone, the results kept, the mode and the output "
          "folder.  Call it first, and after anything that changes the session.")
    def session_state(self) -> Dict[str, Any]:
        s = self.session
        locs = s.locs
        layers = []
        for i, layer in enumerate(s.layers):
            if layer.is_image:
                layers.append({"index": i, "name": layer.name, "image": True})
                continue
            layers.append({"index": i, "name": layer.name, "grouped": layer.grouped,
                           "files": layer.files, "filter": {
                               k: list(v) for k, v in layer.ranges.items()},
                           **self._shown(i)})
        rois = getattr(s, "_rois", None)
        extent = None
        if len(locs):
            (x0, x1), (y0, y1) = s.full_view(margin_fraction=0)
            extent = {"x": [round(float(x0), 1), round(float(x1), 1)],
                      "y": [round(float(y0), 1), round(float(y1), 1)]}
        view = None
        if self.viewer is not None:
            (x0, x1), (y0, y1), pixel = self.viewer.current()
            view = {"x_nm": [round(x0, 1), round(x1, 1)],
                    "y_nm": [round(y0, 1), round(y1, 1)], "pixel_nm": round(pixel, 3)}
        return {
            "mode": self.mode, "output_folder": str(self.output),
            # the code that runs, wherever it was installed: read and grep it
            "smappy": {"version": _version(), "source": str(source_folder())},
            "files": [{"number": n, "name": f.name, "path": f.path, "n": f.n,
                       "frame_time_ms": f.frame_time_ms} for n, f in enumerate(s.files)],
            "n_localizations": len(locs),
            "columns": list(locs.columns) if len(locs) else [],
            "extent_nm": extent,
            **({"view": view} if view is not None else {}),
            "layers": layers,
            "rois": len(rois.rois) if rois is not None and hasattr(rois, "rois") else 0,
            "undo": s.undo_labels()[-5:],
            "results": [{"result": k.id, "plugin": k.path, "applied": k.applied}
                        for k in self.results[-10:]],
            "proposals": len(self.proposals),
            **({"windows": list(self.windows)} if self.windows is not None else {}),
        }

    @tool("Every plugin and saved chain by path, with its one-line summary.  "
          "A prefix narrows it: 'Analysis/Drift', 'File'.",
          prefix={"type": "string", "description": "the start of a path"})
    def list_plugins(self, prefix: str = "") -> List[Dict[str, str]]:
        from .. import plugins
        return [{"path": path, "name": ref.name, "summary": ref.description,
                 "kind": ref.kind, "per_region": ref.per_region}
                for path, ref in plugins.refs(prefix).items()]

    @tool("The plugins and help pages that match a question in words, best "
          "first, each with what its page says it is for.  A map, not the "
          "reference: read the candidates' pages (read_doc) before choosing.",
          query={"type": "string", "description": "e.g. 'drift without beads'"},
          limit={"type": "integer", "description": "how many, default 8"})
    def find_plugins(self, query: str, limit: int = 8) -> List[Dict[str, str]]:
        return index.search(query, limit)

    @tool("A plugin's settings as run_plugin takes them -- flat dotted names "
          "with type, default, unit, range, choices and help; under 'options' "
          "the names a field may take (a preset's saved names, a choice's "
          "values) -- whether it "
          "changes the table, which table it reads, and what its form says "
          "about these settings on this session (values on auto, a warning).",
          path={"type": "string", "description": "the plugin's path"},
          settings={"type": "object", "description": "values to describe it with"})
    def describe_plugin(self, path: str, settings: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
        from ..batch import describe_plugin
        cls = self._plugin(path)
        values = self._settings(cls, settings)
        out: Dict[str, Any] = {"fields": describe_plugin(path),
                               # what a field may be: a preset's names, its
                               # choices (value: label) -- never guess one
                               "options": {
                                   name: {v: l for v, l in options} if any(
                                       v != l for v, l in options) else [v for v, _ in options]
                                   for name, options in self._options(cls).items()},
                               "changes_table": bool(cls().changes_locs_for(values)),
                               "per_region": bool(getattr(cls, "per_region", False)),
                               "reads": cls.grouping or "the table the layer shows"}
        try:
            state = cls().describe(self.session.context(0), values)
        except Exception as error:      # a form's hook failing is not the call's
            state = None
            out["form"] = f"could not describe: {type(error).__name__}: {error}"
        if state is not None:
            out["form"] = {k: _plain(v) for k, v in (
                ("auto", state.hints), ("unused", [n for n, on in state.active.items() if not on]),
                ("hidden", [n for n, on in state.shown.items() if not on]),
                ("choices", {k: list(v) for k, v in state.choices.items()}))}
            if state.notice is not None:
                out["form"]["notice"] = state.notice.text
        return out

    @tool("A plugin's or a panel's help page, as Markdown: what it does, how "
          "it works, the maths, notes on choosing each setting, what a good "
          "and a bad result look like, references.  Panels are 'Panels/<name>'; "
          "'Panels/Glossary' explains the field's words.",
          path={"type": "string", "description": "a plugin's path or 'Panels/...'"})
    def read_doc(self, path: str) -> str:
        from .. import docs
        panels = docs.panels()
        if path in panels:
            return docs.read_page(panels[path].file).body
        cls = self._plugin(path)
        page = docs.page_for(cls)
        if not page.written:
            raise ToolError(f"{path} has no page; read_source shows its code")
        head = f"# {cls.name or path}\n\n`{path}`, version {cls.version}.  {cls.description}\n"
        return head + "\n" + page.body

    @tool("The source code: a plugin's module (by its path), or any module, "
          "class or function of smappy by its dotted name "
          "('smappy.rcc', 'smappy.rcc.estimate_drift_rcc').  The code is "
          "the authority on what is computed.  Long modules come in parts: "
          "pass the offset the answer gives.",
          name={"type": "string", "description": "a plugin path or a dotted name"},
          offset={"type": "integer", "description": "the character to start at"})
    def read_source(self, name: str, offset: int = 0) -> Dict[str, Any]:
        import importlib
        target = None
        if "/" in name:
            target = inspect.getmodule(self._plugin(name))
        else:
            if not name.startswith("smappy"):
                raise ToolError("a dotted name of smappy's, starting 'smappy.'")
            from .. import docs
            try:
                target = importlib.import_module(name)
            except ImportError:
                try:
                    target = docs.resolve(name)
                except Exception:
                    raise ToolError(f"no module, class or function {name!r}") from None
        try:
            text = inspect.getsource(target)
            file = inspect.getsourcefile(target)
        except (TypeError, OSError):
            raise ToolError(f"{name} has no Python source (a compiled extension?)") from None
        part = text[offset:offset + MAX_SOURCE]
        out = {"file": file, "text": part, "length": len(text)}
        if offset + MAX_SOURCE < len(text):
            out["next_offset"] = offset + MAX_SOURCE
        return out

    @tool("A figure of a result, as a picture.",
          result={"type": "string", "description": "the result's id, e.g. 'r3'"},
          index={"type": "integer", "description": "which figure, 0 the main one"})
    def figure(self, result: str, index: int = 0) -> Image:
        kept = next((k for k in self.results if k.id == result), None)
        if kept is None:
            raise ToolError(f"no result {result!r}; the kept ones are "
                            f"{', '.join(k.id for k in self.results) or 'none'}")
        plots = kept.result.figures()
        if not 0 <= index < len(plots):
            raise ToolError(f"{result} has {len(plots)} figure(s)")
        from .. import figures
        import matplotlib.pyplot as plt
        made = figures.render(plots[index])
        buffer = io.BytesIO()
        try:
            made.savefig(buffer, format="png", dpi=FIGURE_DPI)
        finally:
            plt.close(made)
        return Image(buffer.getvalue(), f"{kept.path}: {plots[index].name or 'main'}")

    @tool("A picture of the localizations as the view would render them: "
          "the visible layers with their filters, a scale bar.",
          pixelsize_nm={"type": "number", "description": "the picture's pixel"},
          region={"type": "string", "enum": ["view", "roi", "all"],
                  "description": "default 'all'"})
    def picture(self, pixelsize_nm: Optional[float] = None, region: str = "all") -> Image:
        from .. import plugins
        if not len(self.session.locs):
            raise ToolError("no localizations: load a file or simulate first")
        cls = plugins.get("File/Export/Image")
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "view.png"
            settings = cls.Settings(path=str(target), region=region,
                                    pixelsize_nm=pixelsize_nm)
            self.session.scratch().run(cls(), settings)
            written = sorted(Path(folder).glob("*.png"))
            if not written:
                raise ToolError("the picture was not written")
            return Image(written[0].read_bytes(), "the view")

    # ------------------------------------------------------------------- trying
    @tool("Run a plugin.  apply=false (the default) tries it on a copy of the "
          "session: you get its text, numbers and figures and nothing "
          "changes.  apply=true runs it on the session -- in act mode; in "
          "suggest mode it becomes a proposal for the user.  If the plugin "
          "has a question before it runs (a long run, say) you get the "
          "question: put it to the user, then call again with confirm=true.",
          path={"type": "string", "description": "the plugin's path"},
          settings={"type": "object",
                    "description": "flat dotted names, as describe_plugin lists them"},
          layer={"type": "integer", "description": "the layer it reads, default 0"},
          apply={"type": "boolean", "description": "change the session (act mode)"},
          confirm={"type": "boolean", "description": "the user answered the question"},
          reason={"type": "string", "description": "why, for the user (a proposal)"})
    def run_plugin(self, path: str, settings: Optional[Dict[str, Any]] = None,
                   layer: int = 0, apply: bool = False, confirm: bool = False,
                   reason: str = "") -> Dict[str, Any]:
        cls = self._plugin(path)
        values = self._settings(cls, self._in_output(cls, settings))
        self._layer(layer)
        writes = self._writes(cls, values)
        if (apply or writes or path.startswith("File/Export/")) and not self._changing():
            return self._propose("run_plugin", {"path": path, "settings": settings or {},
                                                "layer": layer, "apply": apply}, reason)
        for written in writes:
            written.parent.mkdir(parents=True, exist_ok=True)
        plugin = cls()
        target = self.session if apply else self.session.scratch()
        if not confirm:
            question = plugin.preflight(target.context(layer, grouping=plugin.grouping),
                                        values)
            if question is not None:
                text = question if isinstance(question, str) else question.text
                choices = [] if isinstance(question, str) else [
                    {"label": c.label, "help": c.help,
                     "settings": _plain(c.settings) if c.settings is not None else None}
                    for c in question.choices]
                return {"question": text, "alternatives": choices,
                        "note": "nothing ran: ask the user, then call again with "
                                "confirm=true (and an alternative's settings if chosen)"}
        if apply and layer == 0:
            later = self._through_panel(path, cls, values)
            if later is not None:
                return later
        started = time.perf_counter()
        try:
            result = target.run(plugin, values, layer=layer)
        except ToolError:
            raise
        except Exception as error:
            raise ToolError(f"{path} failed: {type(error).__name__}: {error}\n"
                            + traceback.format_exc(limit=4)) from None
        return self._answer(self._keep(path, apply, result), time.perf_counter() - started)

    def _through_panel(self, path: str, cls, values, answer=None) -> Optional[Later]:
        """An applied run made as the user makes it, in the GUI: its section
        or window shown, the settings in its form, its Run pressed, on its
        thread -- so the run is seen, can be stopped, and holds nothing.
        The answer comes when the run ends.  None headless, or where the
        GUI cannot (`runner`)."""
        if self.runner is None:
            return None
        started = time.perf_counter()
        pending = self.runner(path, cls, values)
        if pending is None:
            return None
        out = Later()

        def ended(result, why) -> None:
            if why is not None:
                out.fail(f"{path}: {why}")
                return
            kept = self._answer(self._keep(path, True, result), time.perf_counter() - started)
            out.resolve(answer(kept, result) if answer is not None else kept)
        pending.then(ended)
        return out

    @tool("Try a plugin over several values of one setting, on copies of the "
          "session: each value's numbers side by side.  Nothing changes.",
          path={"type": "string", "description": "the plugin's path"},
          field={"type": "string", "description": "the setting to vary, dotted"},
          values={"type": "array", "description": "the values to try"},
          settings={"type": "object", "description": "the other settings"},
          layer={"type": "integer", "description": "the layer it reads, default 0"})
    def scan(self, path: str, field: str, values: List[Any],
             settings: Optional[Dict[str, Any]] = None, layer: int = 0) -> Dict[str, Any]:
        from ..batch import scalars
        if not values:
            raise ToolError("no values to try")
        cls = self._plugin(path)
        rows = []
        for value in values:
            each = self._settings(cls, {**(settings or {}), field: value})
            if self._writes(cls, each):
                raise ToolError("a scan writes no files: leave the file settings empty")
            started = time.perf_counter()
            try:
                result = self.session.scratch().run(cls(), each, layer=layer)
            except Exception as error:
                rows.append({field: value, "error": f"{type(error).__name__}: {error}"})
                continue
            kept = self._keep(path, False, result)
            rows.append({field: value, "result": kept.id,
                         "seconds": round(time.perf_counter() - started, 2),
                         "text": (result.text or "").split("\n")[0],
                         **{k: _plain(v) for k, v in scalars(result.data or {}).items()}})
        return {"plugin": path, "field": field, "rows": rows}

    @tool("Check a chain (YAML or JSON text, the format of docs/batch.md) "
          "without running it: the errors name the fields that do exist.",
          chain={"type": "string", "description": "the chain's text"})
    def validate_chain(self, chain: str) -> Dict[str, Any]:
        from ..batch import validate_chain
        spec = self._chain(chain)
        _, problems = validate_chain(spec)
        return {"ok": not any(p.level == "error" for p in problems),
                "problems": [str(p) for p in problems]}

    def _chain(self, text: str):
        import yaml
        from ..chain import ChainSpec
        try:
            raw = yaml.safe_load(text)
            return ChainSpec.from_dict(raw)
        except Exception as error:
            raise ToolError(f"not a chain: {type(error).__name__}: {error}") from None

    @tool("Run a chain of plugins as one, like run_plugin: apply=false tries "
          "it on a copy, apply=true runs it on the session (act mode).",
          chain={"type": "string", "description": "the chain's text"},
          apply={"type": "boolean", "description": "change the session (act mode)"},
          reason={"type": "string", "description": "why, for the user (a proposal)"})
    def run_chain(self, chain: str, apply: bool = False, reason: str = "") -> Dict[str, Any]:
        from ..batch import validate_chain
        from ..chain import chain_class
        spec = self._chain(chain)
        _, problems = validate_chain(spec)
        errors = [str(p) for p in problems if p.level == "error"]
        if errors:
            raise ToolError("the chain does not validate:\n" + "\n".join(errors))
        cls = chain_class(spec)
        writes = self._writes(cls, cls.Settings(), relative=False)
        if (apply or writes) and not self._changing():
            return self._propose("run_chain", {"chain": chain, "apply": apply}, reason)
        for written in writes:
            written.parent.mkdir(parents=True, exist_ok=True)
        def with_steps(answer, result):
            steps = (result.data or {}).get("results", {})
            answer["steps"] = {key: {"text": r.text or "", "data": _plain(r.data or {})}
                               for key, r in steps.items()}
            answer.pop("data", None)
            return answer

        if apply:
            later = self._through_panel(cls.path or "", cls, cls.Settings(), with_steps)
            if later is not None:
                return later
        target = self.session if apply else self.session.scratch()
        started = time.perf_counter()
        try:
            result = target.run(cls(), cls.Settings())
        except Exception as error:
            raise ToolError(f"the chain failed: {type(error).__name__}: {error}") from None
        return with_steps(self._answer(self._keep(cls.path or "chain", apply, result),
                                       time.perf_counter() - started), result)

    # ----------------------------------------------------------------- changing
    @tool("Open a localization file (or images) into the session.  "
          "append=true adds it to what is open.",
          path={"type": "string", "description": "the file"},
          append={"type": "boolean", "description": "add rather than replace"},
          reason={"type": "string", "description": "why, for the user (a proposal)"})
    def load(self, path: str, append: bool = False, reason: str = "") -> Dict[str, Any]:
        if not Path(path).expanduser().exists():
            raise ToolError(f"no file {path}")
        if not self._changing():
            return self._propose("load", {"path": path, "append": append}, reason)
        try:
            info = self.session.load(Path(path).expanduser(), append=append)
        except Exception as error:
            raise ToolError(f"could not open {path}: {type(error).__name__}: {error}") from None
        return {"loaded": info.name, "n": info.n, "state": self.session_state()}

    @tool("Set a layer's filter on a column: keep lo <= value <= hi; null for "
          "no bound.  remove=true takes the column's filter away.",
          column={"type": "string", "description": "e.g. 'xy_err_nm'"},
          lo={"type": ["number", "null"]}, hi={"type": ["number", "null"]},
          layer={"type": "integer", "description": "default 0"},
          remove={"type": "boolean"},
          reason={"type": "string", "description": "why, for the user (a proposal)"})
    def set_filter(self, column: str, lo: Optional[float] = None,
                   hi: Optional[float] = None, layer: int = 0, remove: bool = False,
                   reason: str = "") -> Dict[str, Any]:
        target = self._layer(layer)
        columns = list(self.session.locs.columns)
        if column not in columns and not remove:
            raise ToolError(f"no column {column!r}; the table has {', '.join(columns)}")
        if not self._viewing():
            return self._propose("set_filter", {"column": column, "lo": lo, "hi": hi,
                                                "layer": layer, "remove": remove}, reason)
        if remove:
            target.remove_bound(column)
        else:
            target.set_bound(column, lo, hi)
        self.session.changed("filter")
        return {"layer": layer, "filter": {k: list(v) for k, v in target.ranges.items()},
                **self._shown(layer)}

    @tool("Show a layer grouped (one row per blink: localizations in "
          "consecutive frames merged) or not.",
          on={"type": "boolean"}, layer={"type": "integer", "description": "default 0"},
          reason={"type": "string", "description": "why, for the user (a proposal)"})
    def set_grouped(self, on: bool, layer: int = 0, reason: str = "") -> Dict[str, Any]:
        self._layer(layer)
        if not self._viewing():
            return self._propose("set_grouped", {"on": on, "layer": layer}, reason)
        self.session.show_grouped(layer, on)
        return {"layer": layer, "grouped": on, **self._shown(layer)}

    @tool("Move the user's 2D view: onto a point (x_nm, y_nm, with width_nm "
          "across), onto an ROI of the ROI manager by its number, or out to "
          "everything.  Changes no data, so it is made in suggest mode too.  "
          "picture(region='view') then shows what the user sees.",
          needs="viewer",
          x_nm={"type": "number"}, y_nm={"type": "number"},
          width_nm={"type": "number", "description": "how much to show across; "
                                                       "default: as now"},
          roi={"type": "integer", "description": "an ROI's number, from the ROI manager"},
          everything={"type": "boolean", "description": "show all the data"})
    def show(self, x_nm: Optional[float] = None, y_nm: Optional[float] = None,
             width_nm: Optional[float] = None, roi: Optional[int] = None,
             everything: bool = False) -> Dict[str, Any]:
        viewer = self.viewer
        given = sum((x_nm is not None or y_nm is not None, roi is not None, bool(everything)))
        if given != 1:
            raise ToolError("give one of: x_nm and y_nm, roi, everything=true")
        if everything:
            viewer.reset()
        elif roi is not None:
            project = self.session.rois
            found = next((r for r in project.rois.values() if r.number == roi), None)
            if found is None:
                numbers = sorted(r.number for r in project.rois.values())
                raise ToolError(f"no ROI {roi}; there are "
                                f"{', '.join(map(str, numbers)) or 'none'}")
            x0, y0, x1, y1 = project.region(found).bounds
            # a margin, so the ROI's outline is seen with what is around it
            mx, my = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
            viewer.frame((x0 - mx, x1 + mx), (y0 - my, y1 + my))
        else:
            if x_nm is None or y_nm is None:
                raise ToolError("a point needs both x_nm and y_nm")
            (x0, x1), (y0, y1), _ = viewer.current()
            width = float(width_nm) if width_nm else x1 - x0
            if not width > 0:
                raise ToolError("width_nm must be positive")
            height = width * (y1 - y0) / (x1 - x0) if x1 > x0 else width
            viewer.frame((x_nm - width / 2, x_nm + width / 2),
                         (y_nm - height / 2, y_nm + height / 2))
        (x0, x1), (y0, y1), pixel = viewer.current()
        return {"view": {"x_nm": [round(x0, 1), round(x1, 1)],
                         "y_nm": [round(y0, 1), round(y1, 1)],
                         "pixel_nm": round(pixel, 3)}}

    @tool("Open a plugin in its own window in SMAPpy, its settings filled in -- "
          "those given, the rest at their defaults -- for the user to look at "
          "and run with its Run.  Runs nothing and changes no data, so it works "
          "in suggest mode too.  When the user asks to open, show or set up a "
          "plugin, this is the call; say in a few words what you opened.  A "
          "preset or a choice is one of the names describe_plugin lists under "
          "'options'.",
          needs="panels",
          path={"type": "string", "description": "the plugin's path"},
          settings={"type": "object",
                    "description": "flat dotted names, as describe_plugin lists them"})
    def open_plugin(self, path: str, settings: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
        cls = self._plugin(path)
        self.panels(path, self.settings_for(path, settings))
        return {"opened": cls.name or path, "settings": settings or {}}

    @tool("Open one of SMAPpy's own windows that is not a plugin: the ROI "
          "manager (on an ROI, given its number), the 3D view, the batch "
          "window, the bead or PSF calibration, the microscope manager, the "
          "camera editor, the help window.  session_state lists them under "
          "'windows'.  Changes nothing, so it works in suggest mode too.",
          needs="windows",
          name={"type": "string", "description": "e.g. 'ROI manager', '3D view'"},
          roi={"type": "integer",
               "description": "for the ROI manager: the ROI to show, by its number"})
    def open_window(self, name: str, roi: Optional[int] = None) -> Dict[str, Any]:
        chosen = _pick("name", name, [(n, n) for n in self.windows])
        roi_id = None
        if roi is not None:
            if chosen != "ROI manager":
                raise ToolError("roi is for the ROI manager")
            found = next((r for r in self.session.rois.rois.values()
                          if r.number == roi), None)
            if found is None:
                numbers = sorted(r.number for r in self.session.rois.rois.values())
                raise ToolError(f"no ROI {roi}; there are "
                                f"{', '.join(map(str, numbers)) or 'none'}")
            roi_id = found.id
        self.windows[chosen](roi_id)
        return {"opened": chosen, **({"roi": roi} if roi is not None else {})}

    @tool("Undo the last changes to the table.",
          steps={"type": "integer", "description": "how many, default 1"},
          reason={"type": "string", "description": "why, for the user (a proposal)"})
    def undo(self, steps: int = 1, reason: str = "") -> Dict[str, Any]:
        if not self.session.can_undo:
            raise ToolError("nothing to undo")
        if not self._changing():
            return self._propose("undo", {"steps": steps}, reason)
        self.session.undo(steps)
        return {"undo": self.session.undo_labels()[-5:]}

    @tool("Save the session's localizations (HDF5) under the output folder.",
          path={"type": "string", "description": "relative to the output folder"},
          reason={"type": "string", "description": "why, for the user (a proposal)"})
    def save(self, path: str, reason: str = "") -> Dict[str, Any]:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = self.output / target
        if not _within(target, self.output):
            raise ToolError(f"{target} is outside the output folder, {self.output}")
        if not self._changing():
            return self._propose("save", {"path": path}, reason)
        target.parent.mkdir(parents=True, exist_ok=True)
        written = self.session.save(target)
        return {"saved": str(written or target)}

    @tool("Save a chain where SMAPpy finds it: it appears in the plugin tree "
          "under its path, in the GUI and to smappy-batch.  Only when the "
          "user asked for the workflow to be kept.",
          chain={"type": "string", "description": "the chain's text"},
          name={"type": "string", "description": "its file name, without .chain.yaml"},
          reason={"type": "string", "description": "why, for the user (a proposal)"})
    def save_chain(self, chain: str, name: str, reason: str = "") -> Dict[str, Any]:
        from ..batch import validate_chain
        from ..chain import write
        from .. import plugins
        spec = self._chain(chain)
        _, problems = validate_chain(spec)
        errors = [str(p) for p in problems if p.level == "error"]
        if errors:
            raise ToolError("the chain does not validate:\n" + "\n".join(errors))
        if not self._changing():
            return self._propose("save_chain", {"chain": chain, "name": name}, reason)
        stem = knowledge._slug(name)
        target = plugins.chains_dir() / f"{stem}.chain.yaml"
        if target.exists():
            raise ToolError(f"{target} exists; choose another name")
        target.parent.mkdir(parents=True, exist_ok=True)
        write(spec, target)
        plugins.discover(force=True)
        return {"saved": str(target)}

    # ---------------------------------------------------------------- knowledge
    @tool("Keep a note in the user's local knowledge base: a decision, an "
          "answer, a value, how a problem was solved.  ONLY after the user "
          "has said, in this conversation, that this should be kept -- ask "
          "first, quoting what you would write.",
          title={"type": "string"}, text={"type": "string", "description": "Markdown"},
          tags={"type": "array", "items": {"type": "string"}})
    def remember(self, title: str, text: str, tags: Sequence[str] = ()) -> Dict[str, Any]:
        path = knowledge.remember(self.knowledge, title, text, tags)
        return {"kept": path.stem, "file": str(path)}

    @tool("Search the user's knowledge base: what earlier sessions were asked "
          "to keep.  Without a query, the newest notes.  Look here before "
          "deciding something the user may have decided already.",
          query={"type": "string"}, limit={"type": "integer", "description": "default 20"})
    def recall(self, query: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        return knowledge.recall(self.knowledge, query, limit)

    @tool("Delete a note from the knowledge base, when the user asks.",
          name={"type": "string", "description": "the note's name, from recall"})
    def forget(self, name: str) -> Dict[str, Any]:
        gone = knowledge.forget(self.knowledge, name)
        if gone is None:
            raise ToolError(f"no note {name!r}")
        return {"forgotten": gone.stem}

    @tool("Note a tool SMAPpy lacks, for its developers.  When you had to take a "
          "detour because no tool does what the user asked directly, say so in "
          "one sentence after helping and ask whether to note it; call this "
          "ONLY after the user said yes.",
          asked={"type": "string", "description": "what the user wanted, in their words"},
          workaround={"type": "string", "description": "how you got there instead"},
          tool={"type": "string",
                "description": "the tool that would have done it, in a line"})
    def note_missing_tool(self, asked: str, workaround: str, tool: str) -> Dict[str, Any]:
        path = knowledge.note_missing_tool(self.missing_tools, asked, workaround, tool)
        return {"noted": tool, "file": str(path)}

    @tool("What was proposed in suggest mode, and what the user did with "
          "each: its status is proposed, opened (its panel shown to the user, "
          "filled in), done or dismissed.  A run the user made from the panel "
          "has its result's id (for figure) and the settings it ran with, "
          "which the user may have changed.")
    def list_proposals(self) -> List[Dict[str, Any]]:
        return [p.as_dict() for p in self.proposals]


# What a client is told when it connects (MCP's `instructions`): how to work
# with these tools.  Agent-neutral, short; the pages hold the rest.
INSTRUCTIONS = """\
SMAPpy analyses single-molecule localization microscopy data.  These tools
work on one open session.

How to work:
* A user often says what they need and not how.  Help as they ask: explain
  how to do it in SMAPpy (the plugin, its settings, what the numbers mean);
  suggest it (run_plugin with apply=true, a proposal in suggest mode; show
  the place you mean); or do it (act mode).  When unclear, explain and offer
  to suggest.
* session_state first.  find_plugins for a question, then read_doc of the
  candidates before choosing: each page says what the method is for, when not
  to use it, how to choose its settings and what a good and a bad result look
  like.  read_source when a page leaves a question open; the code is the
  authority on what is computed.
* describe_plugin for the settings' exact names.  Try with run_plugin
  (apply=false, the default), look at the numbers and the figures (figure),
  and judge them against the page before applying anything.
* Check a method on simulated data with a known truth (File/Simulate/Simulate)
  when you are unsure it suits the data.  scan to see how a result depends on
  a setting.
* Say what you did as a chain (validate_chain, run_chain): it can be run
  again in the GUI and over many files with smappy-batch.
* recall before deciding what the user may have decided already; remember
  and save_chain only when the user has agreed to keep it.
* In suggest mode changes become proposals: explain each one to the user.
* A detour because no tool does what was asked: after helping, offer in one
  sentence to note the missing tool (note_missing_tool), and only on a yes.
* Without tools of your own to read files and run commands (a chat client),
  and a task needs them -- writing a plugin, reading SMAPpy's code across
  files, a figure or a calculation the tools do not make -- say once that
  Claude Code can do it: the Claude app's Code tab, or Assistant > Start
  assistant in SMAPpy.
"""
