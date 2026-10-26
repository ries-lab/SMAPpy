"""The assistant's tools: plain functions over a `Session`, answering in JSON.

Every tool is a method of `Assistant` marked with `tool`, which records its
description and its arguments' JSON schema -- what an MCP client is shown --
so the protocol layer only lists them and calls them.  Nothing here imports
Qt or the protocol: a test, a script and the GUI's server (to come) call the
same methods.

**Reading, trying, changing.**  Most tools only read.  `run_plugin` with
``apply=false`` *tries*: it runs on a copy of the session (`Session.scratch`,
which shares the arrays and copies nothing) and hands back what the plugin
said, its numbers and its figures, while the user's session stays as it was.
A *change* -- a run applied, a file loaded, a filter set, an undo -- is made
only in *act* mode; in *suggest* mode it is recorded as a proposal, which the
GUI will offer the user to run.  The assistant cannot switch the mode: the
user sets it, here with ``--mode``.

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


def tool(description: str, **params: Dict[str, Any]):
    """Mark a method as a tool.  Each keyword is an argument's JSON schema;
    one with a ``default`` in the method's signature is optional."""
    def mark(function):
        function._tool = (description, params)
        return function
    return mark


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


def _within(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


class Assistant:
    """The tools, over one session.

    ``output`` is the folder files may be written to; ``knowledge`` the
    folder of remembered notes.
    """

    def __init__(self, session=None, mode: str = "act", output=None,
                 knowledge_folder=None):
        if mode not in MODES:
            raise ValueError(f"mode is one of {', '.join(MODES)}, not {mode!r}")
        if session is None:
            from ..session import Session
            session = Session()
        self.session = session
        self.mode = mode
        self.output = Path(output or Path.cwd()).expanduser()
        self.knowledge = Path(knowledge_folder or knowledge.default_folder()).expanduser()
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

    def call(self, name: str, arguments: Optional[Dict[str, Any]] = None):
        """Call a tool by name; a `ToolError` for a call that cannot be made."""
        spec = self.specs().get(name)
        if spec is None:
            raise ToolError(f"no tool {name!r}; there are {', '.join(sorted(self.specs()))}")
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

    def _plugin(self, path: str):
        from .. import plugins
        try:
            return plugins.get(path)
        except KeyError:
            near = index.search(path.replace("/", " "), limit=5)
            hint = ", ".join(e["path"] for e in near) or "find_plugins lists them"
            raise ToolError(f"no plugin {path!r}; perhaps {hint}") from None

    def _settings(self, cls, values: Optional[Dict[str, Any]]):
        """Settings from flat dotted values, refusing a name the plugin does
        not have -- `settings_from` drops it silently, which is right for a
        file saved by an older version and wrong for a model that guessed."""
        from ..batch import Problem, _check_values, _flat_specs
        from ..plugins import settings_from
        values = dict(values or {})
        problems: List[Problem] = []
        _check_values(values, _flat_specs(cls.specs()), cls.path, problems)
        errors = [p.message for p in problems if p.level == "error"]
        if errors:
            raise ToolError("; ".join(errors) + ".  describe_plugin lists the fields.")
        return settings_from(cls.Settings, values)

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
    @tool("What is open: the files, the table's columns, each layer's filter "
          "and grouping and how many localizations it shows, the ROIs, what "
          "can be undone, the results kept, the mode and the output folder.  "
          "Call it first, and after anything that changes the session.")
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
        return {
            "mode": self.mode, "output_folder": str(self.output),
            "files": [{"number": n, "name": f.name, "path": f.path, "n": f.n,
                       "frame_time_ms": f.frame_time_ms} for n, f in enumerate(s.files)],
            "n_localizations": len(locs),
            "columns": list(locs.columns) if len(locs) else [],
            "layers": layers,
            "rois": len(rois.rois) if rois is not None and hasattr(rois, "rois") else 0,
            "undo": s.undo_labels()[-5:],
            "results": [{"result": k.id, "plugin": k.path, "applied": k.applied}
                        for k in self.results[-10:]],
            "proposals": len(self.proposals),
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
          "with type, default, unit, range, choices and help -- whether it "
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
        started = time.perf_counter()
        try:
            result = target.run(plugin, values, layer=layer)
        except ToolError:
            raise
        except Exception as error:
            raise ToolError(f"{path} failed: {type(error).__name__}: {error}\n"
                            + traceback.format_exc(limit=4)) from None
        return self._answer(self._keep(path, apply, result), time.perf_counter() - started)

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
        target = self.session if apply else self.session.scratch()
        started = time.perf_counter()
        try:
            result = target.run(cls(), cls.Settings())
        except Exception as error:
            raise ToolError(f"the chain failed: {type(error).__name__}: {error}") from None
        answer = self._answer(self._keep(cls.path or "chain", apply, result),
                              time.perf_counter() - started)
        steps = (result.data or {}).get("results", {})
        answer["steps"] = {key: {"text": r.text or "", "data": _plain(r.data or {})}
                           for key, r in steps.items()}
        answer.pop("data", None)
        return answer

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
        if not self._changing():
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
        if not self._changing():
            return self._propose("set_grouped", {"on": on, "layer": layer}, reason)
        self.session.show_grouped(layer, on)
        return {"layer": layer, "grouped": on, **self._shown(layer)}

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

    @tool("What was proposed in suggest mode and not yet acted on.")
    def list_proposals(self) -> List[Dict[str, Any]]:
        return [p.as_dict() for p in self.proposals]


# What a client is told when it connects (MCP's `instructions`): how to work
# with these tools.  Agent-neutral, short; the pages hold the rest.
INSTRUCTIONS = """\
SMAPpy analyses single-molecule localization microscopy data.  These tools
work on one open session.

How to work:
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
"""
