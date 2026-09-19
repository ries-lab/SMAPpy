"""What the two drift plugins keep in the file, and how they draw it again.

COMET and RCC differ in how they find the curve and in nothing else afterwards,
so what is worth saving -- the curve itself -- and how it is drawn belong to
neither of them.  The mixin is what a plugin adds to say "my result is a
drift", and is the model for any other tool with a figure worth keeping.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..drift import Drift
from . import Result


class KeepsDrift:
    """`keep` / `restore` for a plugin whose result carries a `Drift`."""

    def keep(self, result: Result) -> Optional[Dict[str, Any]]:
        drift = (result.data or {}).get("drift")
        return None if drift is None else {"drift": drift.to_dict()}

    def restore(self, saved: Dict[str, Any]) -> Optional[Result]:
        try:
            drift = Drift.from_dict(saved.get("drift") or {})
        except (TypeError, ValueError):
            return None
        if not len(drift):
            return None
        return Result(text=str(drift), plot=drift.plot, data={"drift": drift},
                      settings=drift.settings)
