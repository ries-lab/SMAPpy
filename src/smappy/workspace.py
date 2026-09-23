"""The saveable GUI state: tabs, the plugins pinned to them, and their values.

There is one plugin tree, and a tab is a *curated list over it* rather than a
branch of it.  So a tab does not own a path prefix, no plugin is unreachable
because no tab claims it, and a tab the user invents costs nothing.

A pin is an `Instance`, not a plugin path: it has its own label and its own
parameter values, so `Spline 3D (beads)` and `Spline 3D (data)` can sit side by
side.  SMAP's evaluation list needs the same thing -- its `addmodule` renames on
collision, `modulename_2`, for exactly this reason -- so the ROI pipeline is a
`list[Instance]` too and gets ordering, renaming and saving for free.

Nothing here imports Qt: the ROI pipeline is provenance and has to be readable
without a GUI, and window geometry is carried as opaque text the GUI encodes.
"""
from __future__ import annotations

import copy
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from . import config

VERSION = 1
WORKSPACE_NAME = "workspace.yaml"

# What a fresh install shows.  `seed` is used *once*, to generate the workspace
# from whatever plugins are actually installed; afterwards the workspace is
# data and a tab is whatever the user made of it.
DEFAULT_TABS = (
    {"name": "File", "kind": "plugins",
     "seed": ("File/Load/", "File/Save/", "File/Export/", "File/Simulate/")},
    {"name": "Localize", "kind": "plugins", "header": "localize", "seed": "Localize/"},
    {"name": "Render", "kind": "render"},
    {"name": "Analysis", "kind": "plugins", "seed": "Analysis/"},
    # a tuple seeds in the order given: finding sites comes before summarising them
    {"name": "ROI", "kind": "plugins", "header": "roi",
     "seed": ("ROIManager/Segment/", "ROIManager/Analyze/")},
)


@dataclass
class Instance:
    """One pinned plugin: which, called what, set how."""
    plugin: str                                     # "Analysis/Drift/COMET"
    id: str = ""                                    # stable across renames
    label: str = ""                                 # "" : use the plugin's name
    values: Dict[str, Any] = field(default_factory=dict)   # by dotted name
    enabled: bool = True                            # the pipeline's checkbox
    # A chain being built or edited, as its file would say it (`chain.ChainSpec`).
    # None for a plugin, and for a chain that is used as saved: then `plugin`
    # is its path and the file is the chain.
    chain: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:12]

    def title(self, fallback: str = "") -> str:
        return self.label or fallback or self.plugin.rsplit("/", 1)[-1]

    def copy(self) -> "Instance":
        """A duplicate with its own identity, so the values can diverge."""
        return replace(self, id=uuid.uuid4().hex[:12], values=dict(self.values),
                       chain=copy.deepcopy(self.chain))


@dataclass
class Tab:
    """A named list of instances, or one of the bespoke widgets."""
    name: str
    kind: str = "plugins"          # "plugins", or a widget: "render", "roi"
    header: str = ""               # a header widget above a plugin list
    instances: List[Instance] = field(default_factory=list)

    def index_of(self, instance_id: str) -> int:
        return next(i for i, x in enumerate(self.instances) if x.id == instance_id)

    def move(self, instance_id: str, by: int) -> bool:
        """Shift one instance up or down.  False if it is already at the end."""
        i = self.index_of(instance_id)
        j = i + by
        if not 0 <= j < len(self.instances):
            return False
        order = self.instances
        order[i], order[j] = order[j], order[i]
        return True


@dataclass
class Workspace:
    """Everything the GUI restores: the tabs, and where the window was."""
    tabs: List[Tab] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    version: int = VERSION
    path: Optional[Path] = None            # where it was read from, if anywhere

    # ------------------------------------------------------------ defaults
    @classmethod
    def default(cls, refs: Optional[Dict[str, Any]] = None) -> "Workspace":
        """The shipped workspace, seeded from the plugins actually installed.

        A blank window on a fresh install would make every new user's first
        task the chooser dialog, so the tabs start with the plugins that ask to
        be pinned.  A tab the *user* adds still starts empty.
        """
        if refs is None:
            from . import plugins
            refs = plugins.refs()
        tabs = []
        for spec in DEFAULT_TABS:
            seed = spec.get("seed") or ()
            prefixes = (seed,) if isinstance(seed, str) else tuple(seed)
            # scope="site" is skipped: an evaluator measures one ROI and has no
            # meaning outside the evaluation pipeline, so a tab's Run button
            # could only fail.
            instances = [Instance(plugin=path)
                         for prefix in prefixes
                         for path, ref in sorted(refs.items())
                         if path.startswith(prefix) and ref.favorite
                         and ref.scope == "locs"]
            tabs.append(Tab(name=spec["name"], kind=spec.get("kind", "plugins"),
                            header=spec.get("header", ""), instances=instances))
        return cls(tabs=tabs)

    # ----------------------------------------------------------- as data
    def to_dict(self) -> Dict[str, Any]:
        tabs = [asdict(t) for t in self.tabs]
        for tab in tabs:                  # a plugin has no chain: say nothing
            for instance in tab["instances"]:
                if instance.get("chain") is None:
                    instance.pop("chain", None)
        return {"version": self.version, "tabs": tabs, "layout": dict(self.layout)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: Optional[Path] = None) -> "Workspace":
        """Read tolerantly: this file outlives the code that wrote it.

        Anything unrecognised is dropped rather than raising -- a workspace
        that will not load is a workspace that loses every tab the user built.
        """
        tabs = []
        for raw in data.get("tabs") or []:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            instances = []
            for item in raw.get("instances") or []:
                if not isinstance(item, dict) or not item.get("plugin"):
                    continue
                values = item.get("values")
                chain = item.get("chain")
                instances.append(Instance(
                    plugin=str(item["plugin"]), id=str(item.get("id") or ""),
                    label=str(item.get("label") or ""),
                    values=values if isinstance(values, dict) else {},
                    enabled=bool(item.get("enabled", True)),
                    chain=chain if isinstance(chain, dict) else None))
            tabs.append(Tab(name=str(raw["name"]), kind=str(raw.get("kind") or "plugins"),
                            header=str(raw.get("header") or ""), instances=instances))
        layout = data.get("layout")
        return cls(tabs=tabs, layout=layout if isinstance(layout, dict) else {},
                   version=int(data.get("version") or VERSION), path=path)

    # ------------------------------------------------------------ the file
    def save(self, path=None) -> Path:
        path = Path(path) if path else (self.path or default_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        self.path = path
        return path

    def prune(self, known: Sequence[str]) -> List[str]:
        """Drop instances whose plugin is no longer installed; say which.

        An uninstalled plugin should not cost the user the rest of the tab, and
        a silent disappearance is worse than a line in the log.
        """
        gone = []
        for tab in self.tabs:
            keep = []
            for instance in tab.instances:
                # a chain being built carries itself, installed or not
                (keep if instance.plugin in known or instance.chain
                 else gone).append(instance)
            tab.instances = keep
        return [i.plugin for i in gone]


def default_path() -> Path:
    return config.config_dir() / WORKSPACE_NAME


def load(path=None) -> Workspace:
    """The workspace at ``path``, or the auto-saved one, or the shipped default."""
    path = Path(path) if path else default_path()
    if not path.exists():
        return Workspace.default()
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return Workspace.default()
    if not isinstance(data, dict) or not data.get("tabs"):
        return Workspace.default()
    return Workspace.from_dict(data, path=path)
