# Code taken from uiPSF project
# Original code: https://github.com/ries-lab/uiPSF/tree/main
# Acknowledgments:

import numpy as np
import scipy as sp
from scipy import ndimage


def psf_to_cspline(
    psf: np.ndarray,
) -> np.ndarray:
    """
    Converts 3D voxelized psf into spline psf
    Args:
        psf: np.array of shape (X, Y, Z)

    Returns:
        np.array of shape (X, Y, Z-1, 64)
    """

    spline_matrix = np.zeros((64, 64))

    for i in range(1, 5):
        dz = (i - 1) / 3
        for j in range(1, 5):
            dy = (j - 1) / 3
            for k in range(1, 5):
                dx = (k - 1) / 3
                for l in range(1, 5):
                    for m in range(1, 5):
                        for n in range(1, 5):
                            row = (i - 1) * 16 + (j - 1) * 4 + (k - 1)
                            col = (l - 1) * 16 + (m - 1) * 4 + (n - 1)
                            spline_matrix[row, col] = (
                                dx ** (l - 1) * dy ** (m - 1) * dz ** (n - 1)
                            )

    # upsample psf with factor of 3
    psf_up = ndimage.zoom(psf, 3.0, mode="grid-constant", grid_mode=True)[
        1:-1, 1:-1, 1:-1
    ]

    spline_matrix = np.float32(spline_matrix)
    coeff = calsplinecoeff(spline_matrix, psf, psf_up)
    return coeff


def calsplinecoeff(A: np.ndarray, psf: np.ndarray, psf_up: np.ndarray) -> np.ndarray:
    """
    Compute spline coefficients from 3D voxelized psf and its 3 times upsampled equivalent.
    Args:
        A: np.array of shape (64, 64)
        psf: np.array of shape (X, Y, Z)
        psf_up: np.array of shape (3*X, 3*Y, 3*Z)

    Returns:
        np.array of shape (X, Y, Z-1, 64)

    """
    coeff = np.zeros((64, psf.shape[0] - 1, psf.shape[1] - 1, psf.shape[2] - 1))
    for i in range(coeff.shape[1]):
        for j in range(coeff.shape[2]):
            for k in range(coeff.shape[3]):
                temp = psf_up[
                    i * 3 : 3 * (i + 1) + 1,
                    j * 3 : 3 * (j + 1) + 1,
                    k * 3 : 3 * (k + 1) + 1,
                ]
                x = sp.linalg.solve(A, temp.flatten())
                coeff[:, i, j, k] = x

    return coeff
