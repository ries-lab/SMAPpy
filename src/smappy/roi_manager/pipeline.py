"""The ROI project's steps: what a block run in every ROI is made of.

The ROI project holds a `chain.Block` (`ROIProject.block`): the per-region
steps window edits it, a chain file can hold it, and it is saved into the
localization file beside the results it produced, since the columns of a
results table mean nothing without knowing which steps produced them, in
what order and with what parameters.  This module turns a block into the
`Step`s that run, and reads a run's record back.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Sequence

from ..plugins import settings_values


@dataclass
class Step:
    """One resolved step: the plugin imported and its settings built."""
    label: str
    path: str
    plugin: Any
    settings: Any
    version: str = "1"
    # whether a region keeps the data its figure is drawn from (`Plugin.keep`)
    # beside its numbers: on for ROIs, which are looked at one by one, off
    # for the regions of an id column (docs/regions.md, "Figures").  Not part
    # of the step's identity: it changes what is kept, not what is measured.
    keep: bool = True

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


def default_block():
    """What a project's block starts as: every installed per-region plugin
    that is a `favorite`, once, with its own defaults.  A specialised one --
    NPC Corners -- says it is not, and is added when the data call for it
    rather than run on every site."""
    from .. import plugins
    from ..chain import Block, Step as ChainStep
    return Block(steps=[ChainStep(plugin=path)
                        for path, ref in sorted(plugins.refs().items())
                        if ref.per_region and ref.favorite])


def block_from_run(recorded: Sequence[Dict[str, Any]]):
    """The block a recorded run's steps describe.

    A project can carry runs and no block of its own -- a script that passed
    its steps straight to `evaluate` -- and the run says exactly what ran,
    down to the parameters.  Reading it back is what lets the stored numbers
    still be checked and reported.
    """
    from ..chain import Block, Step as ChainStep
    return Block(steps=[ChainStep(plugin=str(step.get("plugin") or ""),
                                  label=str(step.get("label") or ""),
                                  values=dict(step.get("parameters") or {}))
                        for step in recorded
                        if isinstance(step, dict) and step.get("plugin")])


def block_steps(block) -> List[Step]:
    """The enabled steps of ``block``, ready to run (`BlockPlugin.pipeline`).

    A step whose plugin is not installed, or that cannot run in a region, is
    left out rather than failing the rest: an uninstalled plugin should cost
    its own columns, not the whole run.
    """
    from ..blocks import block_class
    from ..chain import Block

    def usable(step) -> bool:
        try:
            block_class(Block(steps=[step]))
        except (KeyError, ImportError, ValueError):
            return False
        return True
    kept = [s for s in block.steps if s.enabled and usable(s)]
    if not kept:
        return []
    cls = block_class(replace(block, steps=kept))
    return cls().pipeline(cls.Settings())


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
