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
_READ = ("name", "description", "favorite", "scope")
_SCOPES = ("locs", "site")


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
    attr: Optional[str] = None      # the class; None: declared, resolve on import
    scope: str = "locs"             # "locs": run once.  "site": run per ROI
    favorite: bool = True

    @property
    def group(self) -> str:
        """Everything above the leaf: ``"Analysis/Drift"``."""
        return self.path.rpartition("/")[0]

    @property
    def parts(self) -> Tuple[str, ...]:
        return tuple(self.path.split("/"))

    def load(self) -> type:
        """Import the module and return the plugin class.  Cached by the loader."""
        module = load_module(self.origin)
        if self.attr is not None:
            cls = getattr(module, self.attr, None)
            if cls is None:
                raise ImportError(f"{self.origin} has no {self.attr!r}")
        else:
            # declared through __smappy_plugins__: importing ran @register
            from . import _REGISTRY
            cls = _REGISTRY.get(self.path)
            if cls is None:
                raise ImportError(
                    f"{self.origin} declares {self.path!r} in __smappy_plugins__ "
                    "but importing it registered no such plugin")
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


def load_module(origin: Path):
    """Import a plugin file, once per path.

    A file inside the ``smappy.plugins`` package is imported under its real
    dotted name, so its relative imports work; anything else is imported
    standalone under a name derived from its location.
    """
    origin = Path(origin).resolve()
    if origin in _MODULES:
        return _MODULES[origin]
    dotted = _dotted_name(origin)
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
        spec.loader.exec_module(module)
    _MODULES[origin] = module
    return module


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


def _module_declared(tree: ast.Module) -> List[str]:
    """``__smappy_plugins__ = [...]`` at module level, for dynamic registration."""
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        names = [t.id for t in statement.targets if isinstance(t, ast.Name)]
        if "__smappy_plugins__" not in names:
            continue
        try:
            value = ast.literal_eval(statement.value)
        except (ValueError, TypeError, SyntaxError):
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [v for v in value if isinstance(v, str)]
    return []


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

    def make(path: str, cls: Optional[_Class], default_name: str) -> PluginRef:
        attrs = cls.attrs if cls else {}
        leaf = path.rsplit("/", 1)[-1]
        doc = (cls.doc if cls else "").strip().splitlines()
        scope = attrs.get("scope", "locs")
        return PluginRef(
            path=path,
            name=str(attrs.get("name") or leaf or default_name),
            description=str(attrs.get("description") or (doc[0] if doc else "")),
            origin=origin, root=root,
            attr=cls.name if cls else None,
            scope=scope if scope in _SCOPES else "locs",
            favorite=bool(attrs.get("favorite", True)),
        )

    # 1. explicit @register wins, and is the only way to get several from one file
    explicit = [(cls, p) for cls in classes for p in cls.registered]
    if explicit:
        return [make(p, cls, cls.name) for cls, p in explicit], []

    # 2. a module that builds its plugins dynamically says so
    declared = _module_declared(tree)
    if declared:
        return [make(p, None, p.rsplit("/", 1)[-1]) for p in declared], []

    # 3. otherwise the folder names it, and the file holds exactly one plugin
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


def scan(roots: Sequence[Tuple[str, Path]]
         ) -> Tuple[Dict[str, PluginRef], List[Diagnostic]]:
    """Every plugin under ``roots``, without importing any of them.

    ``roots`` are ``(label, directory)`` in priority order: a later root
    shadowing an earlier path wins, so a user can override a shipped plugin by
    putting their own at the same place.
    """
    refs: Dict[str, PluginRef] = {}
    problems: List[Diagnostic] = []
    for label, directory in roots:
        directory = Path(directory).expanduser()
        if not directory.is_dir():
            problems.append(Diagnostic(directory, "plugin folder does not exist"))
            continue
        for origin in sorted(directory.rglob("*.py")):
            relative = origin.relative_to(directory)
            if any(_skip(part) for part in relative.parts[:-1]) or _skip(origin.name):
                continue
            prefix = "/".join(relative.parts[:-1])
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
