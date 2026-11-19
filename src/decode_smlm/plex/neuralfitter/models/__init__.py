# Imported when first used (decode_smlm.plex._lazy): DECODE-Plex imported all of these here.
from decode_smlm.plex._lazy import attach

__getattr__, __dir__ = attach(
    __name__,
    submodules=['aggregator', 'core', 'factory', 'model_param', 'model_speced_impl', 'unet_param'],
    names={'SigmaMUNet': ('.model_speced_impl', 'SigmaMUNet')},
)
