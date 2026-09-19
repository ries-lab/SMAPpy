"""The evaluation pipeline: an ordered list of evaluators run on every ROI.

A tab's pins are run one at a time on demand; an evaluation is a *pipeline*
run over every site at once, each step contributing columns to that site's row.
That is why it is not a tab, and why it has a window of its own.

A step is a `workspace.Instance` -- the same type a tab pins -- so ordering,
renaming, duplicating and saving are one implementation, and the same evaluator
can appear twice with different parameters.  SMAP needs that too: its
`addmodule` renames on collision, `modulename_2`.

The pipeline is provenance.  The columns of a site table mean nothing without
knowing which evaluators produced them, in what order and with what parameters,
so it is written into the localization file beside the results, and can also be
saved under a name and sent to a colleague.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from ..plugins import settings_from, settings_values
from ..workspace import Instance

SCOPE = "site"


@dataclass
class Step:
    """One resolved instance: the plugin imported and its settings built."""
    label: str
    path: str
    plugin: Any
    settings: Any
    version: str = "1"

    def as_record(self) -> Dict[str, Any]:
        """What goes into the run's provenance."""
        return {"label": self.label, "plugin": self.path, "version": self.version,
                "parameters": settings_values(self.settings)}

    def identity(self) -> Dict[str, Any]:
        """What makes this step's numbers what they are -- all but its name.

        The label is what the columns are called and not part of the
        measurement, so renaming a step must not make every site's result
        look out of date.  The plugin, its version and its parameters are
        exactly what does.
        """
        return {k: v for k, v in self.as_record().items() if k != "label"}


def default_instances() -> List[Instance]:
    """What a pipeline starts as: every installed evaluator, once."""
    from .. import plugins
    return [Instance(plugin=path)
            for path, ref in sorted(plugins.refs().items()) if ref.scope == SCOPE]


def instances_from_run(recorded: Sequence[Dict[str, Any]]) -> List[Instance]:
    """The instances a recorded run's pipeline describes.

    A project can carry runs and no pipeline of its own -- a script that
    passed its steps straight to `evaluate`, or a file from before the
    pipeline travelled with the data -- and the run says exactly what ran,
    down to the parameters.  Reading it back is what lets the stored numbers
    still be checked and reported.
    """
    return [Instance(plugin=str(step.get("plugin") or ""),
                     label=str(step.get("label") or ""),
                     values=dict(step.get("parameters") or {}))
            for step in recorded if isinstance(step, dict) and step.get("plugin")]


def evaluators() -> Dict[str, Any]:
    """Every plugin that measures one ROI, by path."""
    from .. import plugins
    return {p: r for p, r in plugins.refs().items() if r.scope == SCOPE}


def resolve(instances: Sequence[Instance], skip_disabled: bool = True) -> List[Step]:
    """Import each instance's plugin and build its settings.

    A step whose plugin is not installed is left out rather than raising: an
    uninstalled evaluator should cost its own columns, not the whole run.
    """
    from .. import plugins
    steps: List[Step] = []
    labels: Dict[str, int] = {}
    for instance in instances:
        if skip_disabled and not instance.enabled:
            continue
        try:
            cls = plugins.get(instance.plugin)
        except (KeyError, ImportError):
            continue
        label = instance.title(getattr(cls, "name", ""))
        # two steps must not write into one another's columns
        if label in labels:
            labels[label] += 1
            label = f"{label} {labels[label]}"
        else:
            labels[label] = 1
        steps.append(Step(label=label, path=instance.plugin, plugin=cls(),
                          settings=settings_from(cls.Settings, instance.values),
                          version=str(getattr(cls, "version", "1"))))
    return steps


def merged_values(record: Dict[str, Any]) -> Dict[str, Any]:
    """One row from a record's steps.

    A column keeps its plain name while only one step produces it, and is
    qualified with the step's label when two would collide -- so the common
    case reads as it always did and the ambiguous one is never silently lost.
    """
    steps = record.get("steps") or {}
    seen: Dict[str, int] = {}
    for values in steps.values():
        for name in (values.get("values") or {}):
            seen[name] = seen.get(name, 0) + 1
    row: Dict[str, Any] = {}
    for label, values in steps.items():
        for name, value in (values.get("values") or {}).items():
            row[name if seen[name] == 1 else f"{label}.{name}"] = value
    return row


def errors(record: Dict[str, Any]) -> Dict[str, str]:
    """Which steps failed on this ROI, and why."""
    return {label: values["error"]
            for label, values in (record.get("steps") or {}).items()
            if "error" in values}


# ------------------------------------------------------------- as a file

def to_dict(instances: Sequence[Instance]) -> Dict[str, Any]:
    from dataclasses import asdict
    return {"version": 1, "pipeline": [asdict(i) for i in instances]}


def from_dict(doc: Optional[Dict[str, Any]]) -> List[Instance]:
    """Read tolerantly; a pipeline file outlives the code that wrote it."""
    out: List[Instance] = []
    for raw in ((doc or {}).get("pipeline") or []):
        if not isinstance(raw, dict) or not raw.get("plugin"):
            continue
        values = raw.get("values")
        out.append(Instance(plugin=str(raw["plugin"]), id=str(raw.get("id") or ""),
                            label=str(raw.get("label") or ""),
                            values=values if isinstance(values, dict) else {},
                            enabled=bool(raw.get("enabled", True))))
    return out


def save(instances: Sequence[Instance], path) -> Path:
    path = Path(path)
    path.write_text(yaml.safe_dump(to_dict(instances), sort_keys=False))
    return path


def load(path) -> List[Instance]:
    return from_dict(yaml.safe_load(Path(path).read_text()) or {})
