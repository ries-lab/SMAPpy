# Imported when first used (decode_smlm.plex._lazy): DECODE-Plex imported all of these here.
from decode_smlm.plex._lazy import attach

__getattr__, __dir__ = attach(
    __name__,
    submodules=['renderer'],
    names={'Renderer2D': ('.renderer', 'Renderer2D'), 'RendererIndividual2D': ('.renderer', 'RendererIndividual2D')},
)
