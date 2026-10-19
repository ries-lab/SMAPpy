"""Data as the model and the solver see it.  Reading, the camera, bead finding and pairing are SMAPpy's
(`smappy.calibrate.pupil`, docs/uipsf/plan.md §7.5)."""
from .dataset import Batch

__all__ = ["Batch"]
