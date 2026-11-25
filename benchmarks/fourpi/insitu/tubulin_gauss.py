"""(docs/4pi.md §29)  PYTHONPATH=src python benchmarks/fourpi/insitu/tubulin_gauss.py MODEL_4pi.h5 ...

Does a Gaussian take the 1.6x fewer photons?  The authors fit an astigmatic Gaussian to the four channels' sum
(4Pi-SMS ``fit_ast.m``); our photons are the 4Pi model's, all of the spot's light.  Here the model's own spots
-- at the movie's median brightness and background, every z and cavity phase, centred -- are summed over the
channels and fitted by an elliptical Gaussian with a flat background, by Poisson likelihood on the noise-free
image (its expectation: what an unbiased fit converges to on average).  The Gaussian's photons over the model's
are the ratio a Gaussian fit's photons would show against ours if the model were the truth."""
import sys

import numpy as np
from scipy.optimize import minimize
from smappy import fourpi

N, BG = 4477.0, 137.0                 # the movie's median photons (ours) and background (all four channels, per px)


def gauss_fit(img):
    R = img.shape[0]
    yy, xx = np.mgrid[:R, :R] - (R - 1) / 2

    def model(p):
        x, y, sx, sy, n, b = p
        return n / (2 * np.pi * sx * sy) * np.exp(-0.5 * ((xx - x) / sx) ** 2 - 0.5 * ((yy - y) / sy) ** 2) + b

    def nll(p):
        if p[2] <= 0.3 or p[3] <= 0.3 or p[4] <= 0 or p[5] <= 0:
            return 1e30
        m = model(p)
        return float(np.sum(m - img * np.log(m)))

    b0 = np.median(np.r_[img[0], img[-1], img[:, 0], img[:, -1]])
    p0 = [0, 0, 1.5, 1.5, max(img.sum() - b0 * img.size, 1), b0]
    r = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6})
    return r.x


for path in sys.argv[1:]:
    cal = fourpi.load_fourpi_calibration(path)
    for roi in (13, 16):
        rows = []
        for z_nm in (-400, -200, 0, 200, 400):
            for phase in np.linspace(0, 2 * np.pi, 4, endpoint=False):
                s = fourpi.spots(cal, [roi / 2], [roi / 2], [cal.z_nm_to_index(z_nm)], [phase], roi)[0]
                spot = s.sum(0)
                # a channel's expectation is photons x its spot (`fourpi.simulate_rois`): the channels' spots
                # together hold one photon, less what falls outside the box
                img = N * spot + BG
                p = gauss_fit(img)
                rows.append((z_nm, p[4] / N, p[5] - BG, spot.sum()))
        r = np.array(rows)
        print(f"{path.rsplit('/', 1)[-1]} box {roi}: Gaussian photons / model's median {np.median(r[:, 1]):.3f} "
              f"(range {r[:, 1].min():.2f} -- {r[:, 1].max():.2f}); Gaussian background - true, median "
              f"{np.median(r[:, 2]):+.2f}/px; the box holds {np.median(r[:, 3]):.2f} of the model's light")
        for z in (-400, 0, 400):
            print(f"   z {z:+d}: {np.median(r[r[:, 0] == z, 1]):.3f}")
