"""Finding plugins by reading their files, not by importing them.

A plugin's place in the tree comes from where its file sits::

    <root>/Analysis/Drift/comet.py   ->  "Analysis/Drift/COMET"
    <root>/MyLab/Cluster/dbscan.py   ->  "MyLab/Cluster/DBSCAN"

as in SMAP's ``plugins/+Analyze/+cluster/Foo.m``, so adding a plugin is a file
copy.  An explicit ``@register("A/B/C")`` in the file wins over the folder: one
file may declare several plugins, and the built-ins should not have to
rearrange themselves to keep the paths they already have.

Scanning *parses* each file with `ast` and never executes it, so the tree is
complete a few milliseconds after startup and a plugin that imports torch costs
nothing until someone opens it.  `PluginRef.load` does the import, once.  This
is what SMAP needs its generated ``plugin.m`` cache for; reading the files is
fast enough in Python that there is no cache to go stale.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# attributes we read off the class body when they are plain literals
_READ = ("name", "description", "per_region")


@dataclass
class Diagnostic:
    """A file that should have held a plugin and did not.

    Collected rather than raised: one bad plugin must not stop the others from
    loading, and a plugin that vanishes silently is worse than one that
    complains.
    """
    origin: Path
    message: str

    def __str__(self) -> str:
        return f"{self.origin}: {self.message}"


@dataclass(frozen=True)
class PluginRef:
    """What the scanner knows about a plugin before anything is imported."""
    path: str                       # "Analysis/Drift/COMET"
    name: str                       # "COMET"
    description: str
    origin: Path                    # the file it lives in
    root: str                       # the root it came from, for the UI and errors
    attr: Optional[str] = None      # the class; None for a chain
    per_region: bool = False        # hands back Result.values: can run per region
    kind: str = "plugin"            # "plugin": a class in a .py.  "chain": a .chain.yaml

    @property
    def group(self) -> str:
        """Everything above the leaf: ``"Analysis/Drift"``."""
        return self.path.rpartition("/")[0]

    @property
    def parts(self) -> Tuple[str, ...]:
        return tuple(self.path.split("/"))

    def load(self) -> type:
        """Import the module and return the plugin class.  Cached by the loader."""
        if self.kind == "chain":
            from ..chain import load_class
            return load_class(self.origin, self.path)
        module = load_module(self.origin)
        cls = getattr(module, self.attr, None)
        if cls is None:
            raise ImportError(f"{self.origin} has no {self.attr!r}")
        # The folder named this plugin, so tell the class where it ended up.
        # Its own __dict__, not getattr: a plugin subclassing a registered one
        # would otherwise inherit that one's path and keep it.
        if not cls.__dict__.get("path"):
            cls.path = self.path
        if not cls.__dict__.get("name"):
            cls.name = self.name
        return cls


# --------------------------------------------------------------- importing

_MODULES: Dict[Path, Any] = {}
# when each standalone file was imported, by its modification time
_STAMPS: Dict[Path, float] = {}


def load_module(origin: Path):
    """Import a plugin file, once per path -- or once per change, outside the
    package.

    A file inside the ``smappy.plugins`` package is imported under its real
    dotted name, so its relative imports work; anything else is imported
    standalone under a name derived from its location, and imported again
    when the file has changed since: a user's plugin is written a few lines
    at a time, often by an assistant, and a restart between each try and the
    next was most of the cost of trying.  What is already open keeps the
    code it was made with; a panel or window opened after the change gets
    the new one.
    """
    origin = Path(origin).resolve()
    dotted = _dotted_name(origin)
    if origin in _MODULES:
        if dotted or _stamp(origin) == _STAMPS.get(origin):
            return _MODULES[origin]
    if dotted:
        module = importlib.import_module(dotted)
    else:
        # the full path is hashed in: two roots can hold Drift/comet.py, and
        # giving both the same module name would make the second silently be
        # the first
        readable = "_".join(origin.with_suffix("").parts[-3:])
        readable = "".join(c if c.isalnum() or c == "_" else "_" for c in readable)
        digest = hashlib.sha1(str(origin).encode()).hexdigest()[:8]
        unique = f"smappy_plugin_{readable}_{digest}"
        spec = importlib.util.spec_from_file_location(unique, origin)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {origin}")
        module = importlib.util.module_from_spec(spec)
        # in sys.modules before exec, so a dataclass in it can be pickled and
        # `typing.get_type_hints` can resolve names against the module
        sys.modules[unique] = module
        stamp = _stamp(origin)
        spec.loader.exec_module(module)
        _STAMPS[origin] = stamp
    _MODULES[origin] = module
    return module


def _stamp(origin: Path) -> float:
    try:
        return origin.stat().st_mtime
    except OSError:
        return -1.0


def _dotted_name(origin: Path) -> Optional[str]:
    """``smappy.plugins.fit`` for a file inside the package, else None."""
    package_dir = Path(__file__).resolve().parent
    try:
        relative = origin.relative_to(package_dir)
    except ValueError:
        return None
    return ".".join(("smappy", "plugins") + relative.with_suffix("").parts)


# ----------------------------------------------------------------- parsing

@dataclass
class _Class:
    name: str
    bases: List[str]
    registered: List[str] = field(default_factory=list)   # @register paths
    attrs: Dict[str, Any] = field(default_factory=dict)
    doc: str = ""


def _base_names(node: ast.ClassDef) -> List[str]:
    out = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            out.append(base.id)
        elif isinstance(base, ast.Attribute):
            out.append(base.attr)
    return out


def _register_paths(node: ast.ClassDef) -> List[str]:
    """The string argument of every ``@register(...)`` on this class."""
    paths = []
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        named = (func.id if isinstance(func, ast.Name)
                 else func.attr if isinstance(func, ast.Attribute) else "")
        if named != "register" or not decorator.args:
            continue
        first = decorator.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            paths.append(first.value)
    return paths


def _literal_attrs(node: ast.ClassDef) -> Dict[str, Any]:
    """``name = "COMET"`` and friends, when the value is a plain literal.

    Anything computed is skipped rather than guessed at; the import will settle
    it later, and a wrong guess in the tree is worse than a missing one.
    """
    out: Dict[str, Any] = {}
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            targets, value = [statement.target.id], statement.value
        elif isinstance(statement, ast.Assign):
            targets = [t.id for t in statement.targets if isinstance(t, ast.Name)]
            value = statement.value
        else:
            continue
        if value is None:
            continue
        for target in targets:
            if target not in _READ:
                continue
            try:
                out[target] = ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError):
                pass
    return out


def _looks_like_plugin(cls: _Class) -> bool:
    """A base called ``Plugin``, or one whose name ends in it.

    Deliberately by name: resolving the real base would mean importing, which
    is the one thing the scanner does not do.
    """
    return any(b == "Plugin" or b.endswith("Plugin") for b in cls.bases)


# ----------------------------------------------------------------- scanning

def title(stem: str) -> str:
    """``my_drift`` -> ``My Drift``, leaving ``NPC3D`` alone.

    A word that is already mixed case was spelled that way on purpose.
    """
    words = stem.replace("-", " ").replace("_", " ").split()
    return " ".join(w.capitalize() if w.islower() else w for w in words) or stem


def _skip(name: str) -> bool:
    return name.startswith((".", "_")) or name == "__pycache__"


def _refs_in_file(origin: Path, path_prefix: str, root: str
                  ) -> Tuple[List[PluginRef], List[Diagnostic]]:
    try:
        tree = ast.parse(origin.read_text(encoding="utf-8"), filename=str(origin))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        return [], [Diagnostic(origin, f"cannot be read: {exc}")]

    classes: List[_Class] = []
    used_as_base = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = _base_names(node)
        used_as_base.update(bases)
        classes.append(_Class(node.name, bases, _register_paths(node),
                              _literal_attrs(node), ast.get_docstring(node) or ""))

    def make(path: str, cls: _Class, default_name: str) -> PluginRef:
        attrs = cls.attrs
        leaf = path.rsplit("/", 1)[-1]
        doc = cls.doc.strip().splitlines()
        return PluginRef(
            path=path,
            name=str(attrs.get("name") or leaf or default_name),
            description=str(attrs.get("description") or (doc[0] if doc else "")),
            origin=origin, root=root,
            attr=cls.name,
            per_region=attrs.get("per_region") is True,
        )

    # 1. explicit @register wins, and is the only way to get several from one file
    explicit = [(cls, p) for cls in classes for p in cls.registered]
    if explicit:
        return [make(p, cls, cls.name) for cls, p in explicit], []

    # 2. otherwise the folder names it, and the file holds exactly one plugin
    candidates = [c for c in classes if _looks_like_plugin(c)
                  and not c.name.startswith("_") and c.name not in used_as_base]
    stem = origin.stem
    derived = f"{path_prefix}/{title(stem)}" if path_prefix else title(stem)
    if len(candidates) == 1:
        return [make(derived, candidates[0], title(stem))], []
    if not any(_looks_like_plugin(c) for c in classes):
        return [], []           # a helper module living beside the plugins
    if not candidates:
        return [], [Diagnostic(origin, "every plugin class here is private or is "
                                       "used as a base; add @register to the one "
                                       "that should appear")]
    names = ", ".join(c.name for c in candidates)
    return [], [Diagnostic(origin, f"several plugins ({names}); give each an "
                                   "@register path, or split the file")]


# The user's own folder (`config.user_plugin_dir`): its label among the
# roots, and where a file at its top level lands.  A plugin saved straight
# into it sits under "User" -- one place to find what one wrote, and no
# "Thing" beside "Analysis" and "File" at the top of the tree -- while one in
# a subfolder takes its place from the folders as in any root, so
# ``Analysis/Measure/thing.py`` puts it beside the shipped measurements.
USER_ROOT = "user"
USER_GROUP = "User"

CHAIN_SUFFIXES = (".chain.yaml", ".chain.yml")
# where a chain in the chains folder lands when its file does not say
CHAIN_GROUP = "Analysis/Chains"


def _chain_ref(origin: Path, path_prefix: str, root: str
               ) -> Tuple[List[PluginRef], List[Diagnostic]]:
    """A chain file as a ref: its ``path:`` key, else where it sits.

    Read with `yaml.safe_load` -- a few lines of data, nothing executed -- and
    only for the path and the description: whether its steps' plugins exist
    is found out when it is opened, with their names, not while the tree is
    being built.
    """
    try:
        import yaml
        raw = yaml.safe_load(origin.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return [], [Diagnostic(origin, f"cannot be read: {exc}")]
    if not isinstance(raw, dict) or not isinstance(raw.get("steps", []), list):
        return [], [Diagnostic(origin, "is not a chain: no list of `steps`")]
    name = origin.name
    for suffix in CHAIN_SUFFIXES:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    path = str(raw.get("path") or f"{path_prefix or CHAIN_GROUP}/{title(name)}")
    steps = [s.get("label") or str(s.get("plugin", "")).rsplit("/", 1)[-1]
             for s in raw.get("steps") or [] if isinstance(s, dict)]
    return [PluginRef(path=path, name=path.rsplit("/", 1)[-1],
                      description=str(raw.get("description") or " -> ".join(steps)),
                      origin=origin, root=root, kind="chain")], []


def scan(roots: Sequence[Tuple[str, Path]],
         chain_roots: Sequence[Tuple[str, Path]] = ()
         ) -> Tuple[Dict[str, PluginRef], List[Diagnostic]]:
    """Every plugin under ``roots``, without importing any of them.

    ``roots`` are ``(label, directory)`` in priority order: a later root
    shadowing an earlier path wins, so a user can override a shipped plugin by
    putting their own at the same place.  Chains (``*.chain.yaml``) are found
    in every root, and in ``chain_roots``, which hold chains only and are
    skipped quietly when they do not exist yet.
    """
    refs: Dict[str, PluginRef] = {}
    problems: List[Diagnostic] = []
    walk = [(label, directory, False) for label, directory in roots]
    walk += [(label, directory, True) for label, directory in chain_roots]
    for label, directory, chains_only in walk:
        directory = Path(directory).expanduser()
        if not directory.is_dir():
            # the user's folder is not made until something is put in it
            if not chains_only and label != USER_ROOT:
                problems.append(Diagnostic(directory, "plugin folder does not exist"))
            continue
        files = [] if chains_only else sorted(directory.rglob("*.py"))
        files += sorted(o for suffix in CHAIN_SUFFIXES
                        for o in directory.rglob(f"*{suffix}"))
        for origin in files:
            relative = origin.relative_to(directory)
            if any(_skip(part) for part in relative.parts[:-1]) or _skip(origin.name):
                continue
            prefix = "/".join(relative.parts[:-1])
            if not prefix and label == USER_ROOT:
                prefix = USER_GROUP
            if origin.name.endswith(CHAIN_SUFFIXES):
                found, trouble = _chain_ref(origin, prefix, label)
            else:
                found, trouble = _refs_in_file(origin, prefix, label)
            problems.extend(trouble)
            for ref in found:
                previous = refs.get(ref.path)
                if previous is not None and previous.origin != ref.origin:
                    problems.append(Diagnostic(
                        ref.origin, f"{ref.path!r} shadows the one in "
                                    f"{previous.origin} (from {previous.root})"))
                refs[ref.path] = ref
    return dict(sorted(refs.items())), problems
