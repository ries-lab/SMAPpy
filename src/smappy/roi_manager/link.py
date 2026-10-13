"""The name `Session.rois` imports the ROI project by.

The project and the session's ROIs were two classes while a project could own
files of its own; they are one now, `core.ROIProject`.  This module goes when
`session.py` imports from `core` (it is being reworked elsewhere as this is
written, so it was left alone).
"""
from .core import ROIProject as SessionROIs, block_from  # noqa: F401
