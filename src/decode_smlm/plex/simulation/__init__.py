# Imported when first used (decode_smlm.plex._lazy): DECODE-Plex imported all of these here.
from decode_smlm.plex._lazy import attach

__getattr__, __dir__ = attach(
    __name__,
    submodules=['background', 'camera', 'code', 'microscope', 'noise', 'psf_kernel', 'roi',
                'roi_place', 'sampler', 'simulator', 'structures', 'trafo'],
    names={'Simulation': ('.simulator', 'Simulation'), 'RandomStructure': ('.structures', 'RandomStructure')},
)
