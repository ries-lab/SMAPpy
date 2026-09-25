"""A chain of plugins that runs as one: its file, its settings, its run.

A chain is a list of steps -- a plugin path, a label, the values its settings
were saved with -- and `chain_class` turns one into an ordinary `Plugin`
subclass, so the GUI's panels, the workspace, a script and the batch runner
all meet it as the plugin it looks like.  `docs/batch.md` has the decisions;
the ones that shape this module:

* **Settings are generated.**  The chain's `Settings` is a dataclass with one
  field per step, typed as a copy of that step's own settings with one field
  more -- the step's grouping -- so the form the GUI already builds from
  nested dataclasses draws a chain as its steps, folded.  A copy rather than
  the plugin's own class, because the grouping choice belongs *in* the step's
  section and a plugin's settings cannot grow a field for it.

* **It runs on a scratch session.**  Each step's result is applied to a
  headless copy of the session (`Session.scratch`), so step three sees what
  step two did, and nothing reaches the real session until the chain's one
  result does: one undo step, one log entry, and a chain that can run in the
  GUI's worker thread like any plugin.

* **Linear.**  SMAP's workflows were graphs; `NOTES.md` ("No module chain")
  records why the fitting pipeline is not, and the same reasons hold here --
  a chain does what a person clicking through the tabs would do, in order.

The file format is deliberately the flat dotted map every other saved form
uses (`settings_values`), so a chain, a workspace and a pipeline file say the
same thing the same way.
"""
from __future__ import annotations

import copy
import dataclasses
import keyword
import re
import time
import typing
from dataclasses import dataclass, field, fields, make_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .columns import current
from .plugins import (Context, ParamInfo, Plugin, PreflightQuestion, Result,
                      _hint_from_text, param, settings_from, settings_values)

SCHEMA = "smappy-chain-v1"
SUFFIX = ".chain.yaml"
# the chain's own choice of table for every step, and a step's
STEP_GROUPINGS = ("auto", "grouped", "ungrouped")
CHAIN_GROUPINGS = ("layer", "grouped", "ungrouped")
# the field a step's settings gain; a plugin with a field of this name cannot
# be a step, and says so rather than having its field shadowed
GROUPING_FIELD = "use_grouping"
# where a chain lands in the tree when its file does not say
DEFAULT_GROUP = "Analysis/Chains"


class ChainError(ValueError):
    """A chain that cannot be built or cannot go on: the message says why."""


class ChainSkipped(Exception):
    """A step's preflight asked a question, and the answer is to skip the file."""


class ChainAborted(Exception):
    """A step's preflight asked a question, and the answer is to stop."""


# ---------------------------------------------------------------- the file

@dataclass
class Step:
    """One plugin in a chain, as it is written in the file."""
    plugin: str                                  # "Analysis/Drift/RCC"
    label: str = ""                              # "" : the plugin's name
    values: Dict[str, Any] = field(default_factory=dict)   # by dotted name
    version: str = ""                            # the plugin's, when saved
    grouping: str = "auto"                       # auto | grouped | ungrouped
    enabled: bool = True

    def title(self) -> str:
        return self.label or self.plugin.rsplit("/", 1)[-1]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"plugin": self.plugin}
        if self.label:
            out["label"] = self.label
        if self.version:
            out["version"] = self.version
        if self.grouping != "auto":
            out["grouping"] = self.grouping
        if not self.enabled:
            out["enabled"] = False
        out["values"] = dict(self.values)
        return out

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Step":
        if not isinstance(raw, dict) or not raw.get("plugin"):
            raise ChainError(f"a step needs a `plugin`: {raw!r}")
        grouping = str(raw.get("grouping", "auto"))
        if grouping not in STEP_GROUPINGS:
            raise ChainError(f"step {raw['plugin']!r}: grouping {grouping!r} is not "
                             f"one of {', '.join(STEP_GROUPINGS)}")
        values = raw.get("values") or {}
        if not isinstance(values, dict):
            raise ChainError(f"step {raw['plugin']!r}: `values` must be a mapping")
        return cls(plugin=str(raw["plugin"]), label=str(raw.get("label") or ""),
                   values=dict(values), version=str(raw.get("version") or ""),
                   grouping=grouping, enabled=bool(raw.get("enabled", True)))


@dataclass
class ChainSpec:
    """A whole chain: its steps, where it sits in the tree, its grouping."""
    steps: List[Step] = field(default_factory=list)
    path: str = ""                    # "" : from the file's name and folder
    description: str = ""
    grouping: str = "layer"           # layer | grouped | ungrouped
    origin: Optional[Path] = None     # the file it was read from; not written

    @property
    def name(self) -> str:
        if self.path:
            return self.path.rsplit("/", 1)[-1]
        if self.origin is not None:
            return chain_title(self.origin)
        return "Unsaved chain"

    def keys(self) -> List[str]:
        """Each step's key, the name its settings go by: unique identifiers."""
        return unique_keys([s.title() for s in self.steps])

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"schema": SCHEMA}
        if self.path:
            out["path"] = self.path
        if self.description:
            out["description"] = self.description
        out["grouping"] = self.grouping
        out["steps"] = [s.to_dict() for s in self.steps]
        return out

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], origin: Optional[Path] = None) -> "ChainSpec":
        if not isinstance(raw, dict):
            raise ChainError(f"{origin or 'a chain'} is not a mapping")
        schema = raw.get("schema", SCHEMA)
        if schema != SCHEMA:
            raise ChainError(f"{origin or 'this'} is a {schema!r} file, not a chain "
                             f"({SCHEMA})")
        grouping = str(raw.get("grouping", "layer"))
        if grouping not in CHAIN_GROUPINGS:
            raise ChainError(f"grouping {grouping!r} is not one of "
                             f"{', '.join(CHAIN_GROUPINGS)}")
        steps = raw.get("steps") or []
        if not isinstance(steps, list):
            raise ChainError("`steps` must be a list")
        return cls(steps=[Step.from_dict(s) for s in steps],
                   path=str(raw.get("path") or ""),
                   description=str(raw.get("description") or ""),
                   grouping=grouping, origin=origin)


def identifier(text: str) -> str:
    """A label as a settings field name: ``"NPC radius"`` -> ``npc_radius``."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower() or "step"
    if name[0].isdigit():
        name = "s_" + name
    if keyword.iskeyword(name) or name == "grouping":
        name += "_"
    return name


def unique_keys(titles: List[str]) -> List[str]:
    """Identifiers for ``titles``, numbered where two would collide."""
    keys: List[str] = []
    for title in titles:
        key, n = identifier(title), 2
        base = key
        while key in keys:
            key, n = f"{base}_{n}", n + 1
        keys.append(key)
    return keys


def chain_title(origin: Path) -> str:
    """``npc_standard.chain.yaml`` -> ``Npc Standard``, as discovery titles a
    ``.py``."""
    from .plugins.discovery import title
    name = Path(origin).name
    for suffix in (SUFFIX, ".chain.yml", ".yaml", ".yml", ".json"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return title(name)


def read(path) -> ChainSpec:
    """A chain from its file, ``.yaml`` or ``.json``."""
    path = Path(path).expanduser()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json
        raw = json.loads(text)
    else:
        import yaml
        raw = yaml.safe_load(text)
    # a chain saved before a column was renamed names the old one, in a
    # layer's bounds or a step's settings
    return ChainSpec.from_dict(current(raw), origin=path)


def write(spec: ChainSpec, path) -> Path:
    import yaml
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_plain(spec.to_dict()), sort_keys=False,
                                   default_flow_style=False), encoding="utf-8")
    return path


def _plain(value):
    """Values as YAML can write them: tuples as lists, numpy as Python."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def spec_from(plugin_cls, settings) -> Step:
    """A step recording a plugin and the settings it would run with."""
    return Step(plugin=plugin_cls.path, values=settings_values(settings),
                version=getattr(plugin_cls, "version", ""))


# ------------------------------------------------------------ the settings

def _hints(settings_cls) -> Dict[str, Any]:
    try:
        return typing.get_type_hints(settings_cls)
    except (TypeError, NameError):
        return {f.name: _hint_from_text(f.type) for f in fields(settings_cls)}


def _flat_infos(specs, prefix: str) -> Dict[str, ParamInfo]:
    """Every spec's presentation by dotted name, copied: `Plugin.specs` writes
    `advanced` onto the infos it returns, which may be a field's own."""
    out: Dict[str, ParamInfo] = {}
    for name, spec in specs.items():
        out[prefix + name] = dataclasses.replace(spec.info)
        if spec.children:
            out.update(_flat_infos(spec.children, f"{prefix}{name}."))
    return out


def step_settings_class(key: str, plugin_cls, step: Step) -> type:
    """The step's settings: the plugin's fields, defaulting to the step's
    values, plus the step's grouping."""
    base = plugin_cls.Settings
    columns: List[tuple] = []
    if base is not None:
        defaults = settings_from(base, step.values)
        hints = _hints(base)
        for f in fields(base):
            if not f.init:
                continue
            if f.name == GROUPING_FIELD:
                raise ChainError(f"{plugin_cls.path} has a field called "
                                 f"{GROUPING_FIELD!r}, which a chain step needs")
            value = getattr(defaults, f.name)
            if isinstance(value, (list, dict, set)) or dataclasses.is_dataclass(value):
                made = field(default_factory=lambda v=value: copy.deepcopy(v),
                             metadata=f.metadata)
            else:
                made = field(default=value, metadata=f.metadata)
            columns.append((f.name, hints.get(f.name, Any), made))
    required = getattr(plugin_cls, "grouping", None)
    columns.append((GROUPING_FIELD, str, param(
        required or step.grouping, label="grouping", choices=STEP_GROUPINGS,
        advanced=True,
        help=(f"this plugin only works {required}" if required else
              "which table this step reads: auto takes the chain's choice, "
              "or each layer's own"))))
    name = "".join(w.capitalize() for w in key.split("_")) + "Step"
    return make_dataclass(name, columns, namespace={"__module__": __name__})


def plugin_settings(plugin_cls, step_settings):
    """The step's settings as the plugin's own class, which is what it runs on."""
    base = plugin_cls.Settings
    if base is None:
        return None
    return base(**{f.name: copy.deepcopy(getattr(step_settings, f.name))
                   for f in fields(base) if f.init})


def effective_grouping(required: Optional[str], step: str, chain: str) -> Optional[str]:
    """Which table a step reads: the plugin's requirement, else the step's
    choice, else the chain's, else None -- each layer's own."""
    if required in ("grouped", "ungrouped"):
        return required
    if step in ("grouped", "ungrouped"):
        return step
    if chain in ("grouped", "ungrouped"):
        return chain
    return None


def is_source(plugin_cls) -> bool:
    """Does this plugin make its own table, from a file or from nothing?

    What decides a batch's input kind: a chain whose first step is a fit reads
    images, and anything else is handed localization files.
    """
    return plugin_cls.path.startswith(("Localize/", "File/Load/", "File/Simulate/"))


# ------------------------------------------------------------- the plugin

class ChainPlugin(Plugin):
    """A chain, as a plugin.  Made by `chain_class`; not registered itself."""
    spec: ChainSpec = None
    steps: List[tuple] = []           # (key, Step, plugin class) of enabled steps
    logged = True                     # a chain's record is the point of it
    # What a step's preflight question means here.  "proceed": a person has
    # already agreed (the GUI asks every step up front, in `preflight`), or a
    # script asked for the run.  "skip" and "abort" are for a batch, where
    # nobody is there to ask.
    preflight_policy = "proceed"

    @classmethod
    def input_kind(cls) -> str:
        """"images" for a chain that starts with a fit, else "localizations"."""
        first = cls.steps[0][2] if cls.steps else None
        if first is not None and first.path.startswith("Localize/"):
            return "images"
        return "localizations"

    def preflight(self, ctx: Context, settings):
        """Every step's question, asked once, before any of them runs.

        Asked of the table as it is now, so a step late in the chain is asked
        about a table its predecessors have not changed yet -- the cost of
        asking once instead of stopping in the middle for an answer.
        """
        questions = []
        for key, step, cls in self.steps:
            sub = getattr(settings, key)
            try:
                asked = cls().preflight(ctx, plugin_settings(cls, sub))
            except Exception:
                asked = None           # an estimate is a courtesy (see PluginPanel)
            if asked:
                text = asked.text if isinstance(asked, PreflightQuestion) else str(asked)
                questions.append(f"{step.title()}: {text}")
        return "\n\n".join(questions) or None

    def run(self, ctx: Context, settings) -> Result:
        from .session import Session
        session = (ctx.session.scratch() if ctx.session is not None
                   else Session(ctx.locs).scratch())
        records: List[Dict[str, Any]] = []
        results: Dict[str, Result] = {}
        plots: Dict[str, Any] = {}
        kept: Dict[str, Any] = {}
        lines: List[str] = []
        layers = None
        path = None
        for key, step, cls in self.steps:
            label = step.title()
            sub = getattr(settings, key)
            plugin = cls()
            own = plugin_settings(cls, sub)
            grouping = effective_grouping(cls.grouping, getattr(sub, GROUPING_FIELD),
                                          settings.grouping)
            sctx = session.context(min(ctx.layer, len(session.layers) - 1),
                                   progress=lambda text, l=label: ctx.report(f"{l}: {text}"),
                                   grouping=grouping)
            if self.preflight_policy != "proceed":
                asked = plugin.preflight(sctx, own)
                if asked:
                    text = asked.text if isinstance(asked, PreflightQuestion) else str(asked)
                    stop = ChainAborted if self.preflight_policy == "abort" else ChainSkipped
                    raise stop(f"{label}: {text}")
            ctx.report(f"{label}: starting")
            started = time.perf_counter()
            try:
                result = plugin.run(sctx, own)
            except Exception as error:
                raise ChainError(f"step {label!r} ({cls.path}) failed -- "
                                 f"{type(error).__name__}: {error}") from error
            if result.locs is not None and sctx.served_grouped:
                raise ChainError(
                    f"step {label!r} ran on the grouped table and handed back a "
                    "new table: there is no way from one row per blink back to "
                    "the localizations, so run it ungrouped")
            session.apply(plugin, result)
            seconds = time.perf_counter() - started
            changed = result.locs is not None or bool(result.files)
            records.append({"key": key, "label": label, "plugin": cls.path,
                            "version": cls.version, "grouping": grouping or "layer",
                            "changed": changed, "seconds": round(seconds, 3),
                            "text": result.text,
                            "values": _plain(settings_values(own))})
            results[key] = result
            for figure in result.figures():
                name = f"{label}: {figure.name}" if figure.name else label
                plots[name] = dataclasses.replace(figure, name=name)
            try:
                saved = plugin.keep(result)
            except Exception:
                saved = None           # the run stands; only the replot is lost
            if saved:
                kept[key] = {"plugin": cls.path, "label": label, "data": saved}
            if "layers" in (result.data or {}) or "bounds" in (result.data or {}):
                layers = session.layer_configs()
            path = (result.data or {}).get("path") or path
            first = result.text.splitlines()[0] if result.text else "done"
            lines.append(f"{label}: {first}")
        changed = any(r["changed"] for r in records)
        data: Dict[str, Any] = {"steps": records, "results": results, "kept": kept}
        if layers is not None:
            data["layers"] = layers
        if path:
            data["path"] = path
        return Result(locs=session.locs if changed else None,
                      text="\n".join(lines) or "no steps", plots=plots, data=data,
                      settings=settings,
                      log={"steps": [{k: v for k, v in r.items() if k != "key"}
                                     for r in records]})

    def keep(self, result: Result) -> Optional[Dict[str, Any]]:
        """Each step's own kept result, by key: two drift steps in one chain
        keep two curves."""
        return (result.data or {}).get("kept") or None

    def restore(self, saved: Dict[str, Any]) -> Optional[Result]:
        from . import plugins
        found: Dict[str, Any] = {}
        for entry in (saved or {}).values():
            try:
                cls = plugins.get(entry["plugin"])
                again = cls().restore(entry.get("data") or {})
            except Exception:
                continue
            if again is None:
                continue
            label = entry.get("label") or cls.name
            for figure in again.figures():
                name = f"{label}: {figure.name}" if figure.name else label
                found[name] = dataclasses.replace(figure, name=name)
        return Result(plots=found) if found else None

    # the GUI asks these of the chain; each question is the step's own
    def _step(self, dotted: str):
        key, _, rest = dotted.partition(".")
        for k, step, cls in self.steps:
            if k == key:
                return k, cls, rest
        return None, None, rest

    def react(self, changed: str, settings):
        key, cls, rest = self._step(changed)
        if cls is None or not rest or rest == GROUPING_FIELD:
            return None
        replies = cls().react(rest, plugin_settings(cls, getattr(settings, key)))
        return {f"{key}.{k}": v for k, v in (replies or {}).items()} or None

    def hints(self, settings):
        out = {}
        for key, step, cls in self.steps:
            try:
                found = cls().hints(plugin_settings(cls, getattr(settings, key)))
            except Exception:
                found = None
            out.update({f"{key}.{k}": v for k, v in (found or {}).items()})
        return out or None

    def active(self, settings):
        out = {}
        for key, step, cls in self.steps:
            try:
                found = cls().active(plugin_settings(cls, getattr(settings, key)))
            except Exception:
                found = None
            out.update({f"{key}.{k}": v for k, v in (found or {}).items()})
            if cls.grouping:
                out[f"{key}.{GROUPING_FIELD}"] = False
        return out or None

    @classmethod
    def current_spec(cls, settings=None) -> ChainSpec:
        """The chain as a file would say it, with ``settings`` as its values.

        What **Save chain** writes: the steps as they are, each with the
        values in the form and the version of the plugin that is installed.
        """
        steps = []
        by_key = {k: (step, plugin) for k, step, plugin in cls.steps}
        keys = cls.spec.keys()
        for key, step in zip(keys, cls.spec.steps):
            step = copy.deepcopy(step)
            if key in by_key and settings is not None:
                sub = getattr(settings, key)
                plugin = by_key[key][1]
                step.values = settings_values(plugin_settings(plugin, sub))
                step.grouping = ("auto" if plugin.grouping
                                 else getattr(sub, GROUPING_FIELD))
                step.version = plugin.version
            steps.append(step)
        grouping = settings.grouping if settings is not None else cls.spec.grouping
        return ChainSpec(steps=steps, path=cls.spec.path,
                         description=cls.spec.description, grouping=grouping,
                         origin=cls.spec.origin)


def chain_class(spec: ChainSpec, path: Optional[str] = None) -> type:
    """A `Plugin` subclass that runs ``spec``.

    Every step's plugin is imported here, so a chain naming one that is not
    installed fails now, with its name, rather than halfway through a run.
    """
    from . import plugins
    keys = spec.keys()
    steps: List[tuple] = []
    columns: List[tuple] = [("grouping", str, param(
        spec.grouping, label="grouping", choices=CHAIN_GROUPINGS, advanced=True,
        help="which table every step reads, unless the step says otherwise: "
             "layer leaves it to each layer, as the Render tab has it"))]
    extras: Dict[str, ParamInfo] = {}
    for key, step in zip(keys, spec.steps):
        if not step.enabled:
            continue
        try:
            cls = plugins.get(step.plugin)
        except KeyError:
            raise ChainError(f"step {step.title()!r} needs the plugin "
                             f"{step.plugin!r}, which is not installed") from None
        if getattr(cls, "scope", "locs") == "site":
            raise ChainError(f"step {step.title()!r}: {step.plugin} runs once per "
                             "ROI and belongs in the ROI manager's pipeline")
        step_cls = step_settings_class(key, cls, step)
        columns.append((key, step_cls, param(
            default_factory=step_cls, label=step.title(), collapsed=True,
            help=cls.description or step.plugin)))
        extras.update(_flat_infos(cls.specs(), f"{key}."))
        steps.append((key, step, cls))
    settings_cls = make_dataclass("ChainSettings", columns,
                                  namespace={"__module__": __name__})
    name = spec.name
    attrs = {
        "spec": spec, "steps": steps, "Settings": settings_cls, "params": extras,
        "name": name, "path": path or spec.path or f"{DEFAULT_GROUP}/{name}",
        "description": spec.description or " -> ".join(s.title() for _, s, _ in steps),
        "version": "1",
    }
    return type(f"Chain_{identifier(name)}", (ChainPlugin,), attrs)


def load_class(origin, tree_path: Optional[str] = None) -> type:
    """The plugin class for a chain file, at ``tree_path`` if discovery placed
    it.  Built afresh each time it is asked for: the file is a few lines and
    may have been saved a moment ago."""
    return chain_class(read(origin), path=tree_path)
