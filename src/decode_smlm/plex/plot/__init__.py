# Imported when first used (decode_smlm.plex._lazy): DECODE-Plex imported all of these here.
from decode_smlm.plex._lazy import attach

__getattr__, __dir__ = attach(
    __name__,
    submodules=['dist'],
    names={'deviation': ('.dist', 'deviation'), 'PlotFrame': ('.plot', 'PlotFrame'), 'PlotFrameCoord': ('.plot', 'PlotFrameCoord'), 'PlotCoordinates': ('.plot', 'PlotCoordinates'), 'PlotCoordinates3D': ('.plot', 'PlotCoordinates3D'), 'crosshair': ('.utils', 'crosshair')},
)
