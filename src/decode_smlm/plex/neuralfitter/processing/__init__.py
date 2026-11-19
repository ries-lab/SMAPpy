# Imported when first used (decode_smlm.plex._lazy): DECODE-Plex imported all of these here.
from decode_smlm.plex._lazy import attach

__getattr__, __dir__ = attach(
    __name__,
    submodules=['background', 'coupled', 'model_input', 'multichannel', 'post', 'to_emitter', 'utils'],
)
