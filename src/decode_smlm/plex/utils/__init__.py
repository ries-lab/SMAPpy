# Imported when first used (decode_smlm.plex._lazy): DECODE-Plex imported all of these here.
from decode_smlm.plex._lazy import attach

__getattr__, __dir__ = attach(
    __name__,
    submodules=['aux_trafo', 'bookkeeping', 'dev', 'emitter_trafo', 'example_helper', 'files', 'future', 'hardware', 'system', 'torch', 'types'],
)
