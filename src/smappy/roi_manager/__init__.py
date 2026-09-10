"""Analysis ROI projects and an optional matplotlib selection interface."""
from .core import ROI, ROIProject
from .plugins import DensityPeaks, Statistics, Histograms


def show(project=None, files=(), block=True):
    """Open a project, optionally adding localization HDF5 files."""
    import matplotlib.pyplot as plt
    from .gui import ROIManager
    if isinstance(project, (str, bytes)) or hasattr(project, '__fspath__'):
        project = ROIProject.load(project)
    if project is None:
        project = ROIProject()
    for path in files:
        project.add_file(path)
    manager = ROIManager(project)
    plt.show(block=block)
    return manager


__all__ = ['ROI', 'ROIProject', 'DensityPeaks', 'Statistics', 'Histograms', 'show']
