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

* **An evaluator is a step over every site.**  A plugin with
  ``per_region = True`` can measure one ROI; as a step set to *run on* each
  ROI it is run over every
  included ROI of the ROI manager's project (`ROIProject.evaluate`) before
  the next step starts, so a chain can segment, evaluate and analyse the
  site table as a person does in the ROI manager.  Its results are stored
  with the ROIs like any evaluation's, under the step's name, and the
  evaluation window's pipeline is left as the user set it up: a result
  belongs to the ROI and says how it was made (`ROIProject.evaluations`),
  so nothing downstream needs to know which pipeline ran it.

* **Stop ends it between steps.**  A step is handed the chain's
  `Context.stop`; once it is set, no further step starts, and the steps
  done -- the one that was running, if it kept a part (a fit's frames so
  far) -- are the chain's result, which says ``stopped``.  A step that
  raises `Stopped` keeps nothing and the chain ends before it.

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

from .plugins import (Context, FormState, ParamInfo, Plugin, PreflightChoice,
                      PreflightQuestion, Result, Stopped, _hint_from_text, param,
                      settings_from, SOURCES, refuse_grouped_table, settings_values)

SCHEMA = "smappy-chain-v1"
SUFFIX = ".chain.yaml"
# the chain's own choice of table for every step, and a step's
STEP_GROUPINGS = ("auto", "grouped", "ungrouped")
CHAIN_GROUPINGS = ("layer", "grouped", "ungrouped")
# as the GUI says them: which table, in the words of the Render tab's
# *merge blinks*
MERGED = {"auto": "auto", "layer": "as each layer", "grouped": "merged",
          "ungrouped": "not merged"}
# the field a step's settings gain; a plugin with a field of this name cannot
# be a step, and says so rather than having its field shadowed
GROUPING_FIELD = "use_grouping"
# a block's choices: what it repeats over, and how it merges overlaps
BLOCK_ROIS = "rois"
KEEP_FIGURES = ("auto", "yes", "no")
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
class Block:
    """A list of steps run once in each region (`docs/regions.md`).

    ``regions`` is what it repeats over: ``rois``, every included ROI of the
    ROI manager, or the name of an id column (``cluster_id``).  In each
    region the steps run on that region's tables, a table one hands back
    handed to the next, all of it thrown away afterwards but the numbers
    each step returned -- and the columns named in ``write_back``, merged
    into the real table by ``merge`` where regions overlap.  The numbers
    named in ``write_values`` go onto the table too, each region's onto its
    rows (`region_runs.values_onto`).  A step may be a
    block itself, or a chain.
    """
    steps: List[Any] = field(default_factory=list)   # Step or Block
    label: str = ""
    regions: str = BLOCK_ROIS
    write_back: List[str] = field(default_factory=list)
    write_values: List[str] = field(default_factory=list)
    merge: str = "last"
    keep_figures: str = "auto"       # auto: on for ROIs, off for an id column
    enabled: bool = True
    plugin: str = ""                 # a step's field, empty: a block has none
    grouping: str = "auto"

    def title(self) -> str:
        if self.label:
            return self.label
        return "for each ROI" if self.regions == BLOCK_ROIS else f"for each {self.regions}"

    def keys(self) -> List[str]:
        return unique_keys([s.title() for s in self.steps])

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"block": self.label, "regions": self.regions}
        if self.write_back:
            out["write_back"] = list(self.write_back)
        if self.write_values:
            out["write_values"] = list(self.write_values)
        if self.merge != "last":
            out["merge"] = self.merge
        if self.keep_figures != "auto":
            out["keep_figures"] = self.keep_figures
        if not self.enabled:
            out["enabled"] = False
        out["steps"] = [s.to_dict() for s in self.steps]
        return out

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Block":
        from .region_runs import WriteBack
        steps = raw.get("steps") or []
        if not isinstance(steps, list):
            raise ChainError("a block's `steps` must be a list")
        merge = str(raw.get("merge", "last"))
        if merge not in WriteBack.RULES:
            raise ChainError(f"block merge {merge!r} is not one of "
                             f"{', '.join(WriteBack.RULES)}")
        keep = str(raw.get("keep_figures", "auto"))
        if keep not in KEEP_FIGURES:
            raise ChainError(f"block keep_figures {keep!r} is not one of "
                             f"{', '.join(KEEP_FIGURES)}")
        def names(key):
            found = raw.get(key) or []
            if isinstance(found, str):
                found = [c.strip() for c in found.split(",") if c.strip()]
            return [str(c) for c in found]
        return cls(steps=[step_from_dict(s) for s in steps],
                   label=str(raw.get("block") or raw.get("label") or ""),
                   regions=str(raw.get("regions") or BLOCK_ROIS),
                   write_back=names("write_back"), write_values=names("write_values"),
                   merge=merge,
                   keep_figures=keep, enabled=bool(raw.get("enabled", True)))


def step_from_dict(raw: Dict[str, Any]):
    """A step of a chain's file: a plugin step, or a block (``block:``)."""
    if isinstance(raw, dict) and ("block" in raw or ("steps" in raw
                                                     and "plugin" not in raw)):
        return Block.from_dict(raw)
    return Step.from_dict(raw)


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
        return cls(steps=[step_from_dict(s) for s in steps],
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
    return ChainSpec.from_dict(raw, origin=path)


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
                                 f"{f.name!r}, which a chain step needs")
            value = getattr(defaults, f.name)
            if isinstance(value, (list, dict, set)) or dataclasses.is_dataclass(value):
                made = field(default_factory=lambda v=value: copy.deepcopy(v),
                             metadata=f.metadata)
            else:
                made = field(default=value, metadata=f.metadata)
            columns.append((f.name, hints.get(f.name, Any), made))
    required = getattr(plugin_cls, "grouping", None)
    if required:
        help = f"this plugin only works with blinks {MERGED[required]}"
    else:
        help = ("whether this step reads merged blinks: auto takes the chain's "
                "choice, or each layer's own; in each ROI, each layer's own")
    columns.append((GROUPING_FIELD, str, param(
        required or step.grouping, label="merge blinks",
        choices=[(g, MERGED[g]) for g in STEP_GROUPINGS],
        advanced=True, help=help)))
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
    return plugin_cls.path.startswith(SOURCES)


def is_evaluator(plugin_cls) -> bool:
    """Can this plugin measure one region at a time (`Plugin.per_region`)?"""
    return bool(getattr(plugin_cls, "per_region", False))


def made_by(plugin_cls, step_settings) -> List[Tuple[str, str]]:
    """The evaluations a step leaves, as ``(name, plugin path)``: a block
    leaves one per step, on the ROIs or per region of its id column
    (`BlockPlugin.evaluations`); anything else none."""
    from .blocks import BlockPlugin
    if isinstance(plugin_cls, type) and issubclass(plugin_cls, BlockPlugin):
        return plugin_cls.evaluations(step_settings)
    return []


def _with_step(settings, key: str, plugin_cls, own):
    """The chain's settings with one step's replaced by ``own``, the plugin's
    own settings: what a step's preflight choice runs the chain with."""
    sub = getattr(settings, key)
    if type(own) is type(sub):            # a block: its settings are the step's
        return dataclasses.replace(settings, **{key: copy.deepcopy(own)})
    sub = dataclasses.replace(sub, **{f.name: copy.deepcopy(getattr(own, f.name))
                                      for f in fields(plugin_cls.Settings) if f.init})
    return dataclasses.replace(settings, **{key: sub})


def _roi_signature(session) -> Optional[str]:
    """The ROI project as text, to tell whether a chain changed it."""
    from .roi_manager.core import json_text
    state = session.roi_state()
    return None if state is None else json_text(state)


# ------------------------------------------------------------- the plugin

class ChainPlugin(Plugin):
    """A chain, as a plugin.  Made by `chain_class`; not registered itself."""
    spec: ChainSpec = None
    steps: List[tuple] = []           # (key, Step, plugin class) of enabled steps
    logged = True                     # a chain's record is the point of it
    # whether a step changes the table: `chain_class` answers for the
    # chain's own values, `changes_locs_for` for the form's
    changes_locs = True
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

    def changes_locs_for(self, settings) -> bool:
        """Whether any step changes the table with the settings it has."""
        for key, step, cls in self.steps:
            try:
                answer = cls().changes_locs_for(
                    plugin_settings(cls, getattr(settings, key)))
            except Exception:
                answer = cls.changes_locs
            if answer:
                return True
        return False

    def preflight(self, ctx: Context, settings):
        """Every step's question, asked once, before any of them runs.

        Asked of the table as it is now, so a step late in the chain is asked
        about a table its predecessors have not changed yet -- the cost of
        asking once instead of stopping in the middle for an answer.  A step
        is told which evaluations the steps before it will have made, so an
        ROI analysis after the chain's own evaluator does not ask which.

        A step that asks to *choose* (a `PreflightQuestion` without a plain
        run, such as which of several evaluations to analyse) is put first,
        with its choices as the chain's: taking one sets that step's field,
        and the GUI asks again until nothing is left to choose.
        """
        questions = []
        offered: List[PreflightChoice] = []   # a step's other ways to run
        made: List[Tuple[str, str]] = []
        for key, step, cls in self.steps:
            sub = getattr(settings, key)
            own = plugin_settings(cls, sub)
            sctx = copy.copy(ctx)
            sctx.evaluations = list(made)
            made.extend(made_by(cls, sub))
            try:
                asked = cls().preflight(sctx, own)
            except Exception:
                asked = None           # an estimate is a courtesy (see PluginPanel)
            if not asked:
                continue
            if isinstance(asked, PreflightQuestion) and asked.choices \
                    and not asked.run_label:
                choices = [PreflightChoice(
                    label=c.label, help=c.help,
                    settings=None if c.settings is None
                    else _with_step(settings, key, cls, c.settings))
                    for c in asked.choices]
                return PreflightQuestion(text=f"{step.title()}: {asked.text}",
                                         choices=choices, run_label="")
            text = asked.text if isinstance(asked, PreflightQuestion) else str(asked)
            questions.append(f"{step.title()}: {text}")
            if isinstance(asked, PreflightQuestion) and not offered:
                offered = [PreflightChoice(
                    label=c.label, help=c.help,
                    settings=None if c.settings is None
                    else _with_step(settings, key, cls, c.settings))
                    for c in asked.choices]
        if offered:
            return PreflightQuestion(text="\n\n".join(questions), choices=offered)
        return "\n\n".join(questions) or None

    def run(self, ctx: Context, settings) -> Result:
        from .session import Session
        # the whole table, whatever the chain's own context shows: each step
        # gets the table its grouping says, from the copy
        locs, _ = ctx.ungrouped()
        ctx.served_grouped = False
        session = (ctx.session.scratch(locs) if ctx.session is not None
                   else Session(locs).scratch())
        records: List[Dict[str, Any]] = []
        results: Dict[str, Result] = {}
        plots: Dict[str, Any] = {}
        kept: Dict[str, Any] = {}
        lines: List[str] = []
        layers = None
        written = None
        images = None
        rois_before = _roi_signature(session)
        made: List[Tuple[str, str]] = []  # (name, evaluator) of the chain's evaluations
        stopped = False
        for key, step, cls in self.steps:
            label = step.title()
            if ctx.stopped:
                lines.append(f"stopped before {label}")
                stopped = True
                break
            sub = getattr(settings, key)
            plugin = cls()
            own = plugin_settings(cls, sub)
            grouping = (None if not hasattr(sub, GROUPING_FIELD) else effective_grouping(
                cls.grouping, getattr(sub, GROUPING_FIELD), settings.grouping))
            sctx = session.context(min(ctx.layer, len(session.layers) - 1),
                                   progress=lambda text, l=label: ctx.report(f"{l}: {text}"),
                                   grouping=grouping, evaluations=list(made))
            if self.preflight_policy != "proceed":
                asked = plugin.preflight(sctx, own)
                if asked:
                    text = asked.text if isinstance(asked, PreflightQuestion) else str(asked)
                    stop = ChainAborted if self.preflight_policy == "abort" else ChainSkipped
                    raise stop(f"{label}: {text}")
            ctx.report(f"{label}: starting")
            started = time.perf_counter()
            sctx.stop = ctx.stop
            try:
                result = plugin.run(sctx, own)
            except Stopped as error:
                lines.append(f"{label}: {error}")
                stopped = True
                break
            except Exception as error:
                raise ChainError(f"step {label!r} ({cls.path}) failed -- "
                                 f"{type(error).__name__}: {error}") from error
            try:
                refuse_grouped_table(sctx, result, plugin, f"step {label!r}")
            except ValueError as error:
                raise ChainError(str(error)) from None
            session.apply(plugin, result)
            made.extend(result.evaluations or ())
            for warning in sctx.warnings:
                ctx.warn(warning)
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
            if result.layers is not None or result.bounds:
                layers = session.layer_configs()
            written = result.written or written
            # a fit's kept frames go on with the table, as its path does
            images = list(result.images or ()) or images
            first = result.text.splitlines()[0] if result.text else "done"
            lines.append(f"{label}: {first}")
            stopped = stopped or result.stopped
        changed = any(r["changed"] for r in records)
        return Result(locs=session.locs if changed else None,
                      text="\n".join(lines) or "no steps", plots=plots,
                      data={"steps": records, "results": results, "kept": kept},
                      stopped=stopped, layers=layers, written=written,
                      images=images or (),
                      # the ROIs a segmenter found and the runs an evaluator
                      # made, which `Session.apply` puts into the session's
                      # project, and what blocks measured per region of an
                      # id column
                      roi_project=(session.roi_state()
                                   if _roi_signature(session) != rois_before else None),
                      id_regions=session.id_regions if session._id_regions else {},
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

    def describe(self, ctx: Optional[Context], settings) -> FormState:
        """Each step's own, under the step's key: its hints, its fields read,
        its lists -- told, as in `preflight`, which evaluations the steps
        before it will have made -- and its notice, saying whose it is and
        opening its window on the step's settings.  A step whose grouping
        is fixed by its plugin has none to choose."""
        state = FormState()
        made: List[Tuple[str, str]] = []
        for key, step, cls in self.steps:
            sub = getattr(settings, key)
            sctx = None
            if ctx is not None:
                sctx = copy.copy(ctx)
                sctx.evaluations = list(made)
            made.extend(made_by(cls, sub))
            try:
                own = plugin_settings(cls, sub)
                state.update(cls().describe(sctx, own).under(key, step.title(), own))
            except Exception:
                pass                       # best effort, like the panel's
            if cls.grouping and hasattr(sub, GROUPING_FIELD):
                state.active[f"{key}.{GROUPING_FIELD}"] = False
        return state

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
                if isinstance(step, Block):
                    step = plugin().block_from(sub)
                else:
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
        spec.grouping, label="merge blinks",
        choices=[(g, MERGED[g]) for g in CHAIN_GROUPINGS], advanced=True,
        help="whether every step reads merged blinks, unless the step says "
             "otherwise: as each layer has it in the Render tab, or not"))]
    extras: Dict[str, ParamInfo] = {}
    for key, step in zip(keys, spec.steps):
        if not step.enabled:
            continue
        if isinstance(step, Block):
            from .blocks import block_class
            cls = block_class(step)
            columns.append((key, cls.Settings, param(
                default_factory=cls.Settings, label=step.title(), collapsed=True,
                help=cls.description)))
            extras.update(_flat_infos(cls.specs(), f"{key}."))
            steps.append((key, step, cls))
            continue
        cls = step_plugin(step)
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
    chain = type(f"Chain_{identifier(name)}", (ChainPlugin,), attrs)
    # the answer for the chain's own values, as a plugin's class says it for
    # its defaults; `changes_locs_for` answers for the form's
    try:
        chain.changes_locs = bool(chain().changes_locs_for(settings_cls()))
    except Exception:
        chain.changes_locs = any(cls.changes_locs for _, _, cls in steps)
    return chain


# the chain files being built, innermost last: a chain that contains itself,
# directly or through another, would otherwise build forever
_BUILDING: List[str] = []


def load_class(origin, tree_path: Optional[str] = None) -> type:
    """The plugin class for a chain file, at ``tree_path`` if discovery placed
    it.  Built afresh each time it is asked for: the file is a few lines and
    may have been saved a moment ago.  A chain that contains itself is
    refused, naming the loop."""
    key = str(Path(origin).resolve())
    if key in _BUILDING:
        loop = _BUILDING[_BUILDING.index(key):] + [key]
        raise ChainError("a chain cannot contain itself: "
                         + " -> ".join(chain_title(Path(k)) for k in loop))
    _BUILDING.append(key)
    try:
        return chain_class(read(origin), path=tree_path)
    finally:
        _BUILDING.pop()


def step_plugin(step: Step) -> type:
    """The plugin class a step names, or a `ChainError` that says which."""
    from . import plugins
    try:
        return plugins.get(step.plugin)
    except KeyError:
        raise ChainError(f"step {step.title()!r} needs the plugin "
                         f"{step.plugin!r}, which is not installed") from None
