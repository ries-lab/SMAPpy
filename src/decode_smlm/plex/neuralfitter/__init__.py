# Imported when first used (decode_smlm.plex._lazy): DECODE-Plex imported all of these here.
from decode_smlm.plex._lazy import attach

__getattr__, __dir__ = attach(
    __name__,
    submodules=['callbacks', 'coord_transform', 'data', 'de_bias', 'frame_processing', 'indicator', 'inference', 'logger', 'loss', 'model', 'models', 'process', 'processing', 'sample', 'sampler', 'scale_transform', 'scaling', 'spec', 'target_generator', 'train', 'utils'],
)
