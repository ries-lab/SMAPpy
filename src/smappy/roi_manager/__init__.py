"""Analysis ROI projects: the model, the plugins and their storage.

The interface lives in the GUI (`smappy.gui.roi_window` and the control
window's Regions tab), which drives this model over the session's files.  A
script uses the same project through a session with no window::

    from smappy.session import Session

    session = Session()
    session.load("localizations.h5")
    project = session.rois
    project.set_geometry(300, "circle")
    project.find(next(iter(project.sources)))
    project.evaluate()
    rows = project.results()
    session.save("localizations.h5")         # the ROIs and results go with it

The plugins themselves live in `smappy.plugins.roi`, with everything else, so
that the folder tree stays the plugin tree; they are re-exported here because
this is where one goes looking for them.
"""
from .core import ROI, ROIProject
from ..plugins.roi import Histograms, histograms
from ..plugins.segment import SegmentRegions
from .pipeline import Step

__all__ = ['ROI', 'ROIProject', 'Step', 'SegmentRegions', 'Histograms',
           'histograms']
