"""The camera of the Zenodo `1ch_40nm_bead` stacks, as typed into a standalone uipsf (docs/uipsf/data.md).

SMAPpy's camera database identifies it from the stacks' tags as the M2 microscope's Evolve 512 in
its normal-port mode, 5 MHz, gain 1: 4.7 e⁻/ADU, offset 398.6 ADU.  The read noise, 21 e⁻ rms, is
what the residuals show at zero signal (docs/uipsf/architecture.md, M3); the database has none.
"""
from uipsf.noise import Camera

CAMERA = Camera(conversion=4.7, offset=398.6, read_noise_e=21.0)
