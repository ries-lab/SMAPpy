"""Write tests/uipsf_tests/fixtures/oracle_v1.npz: PSFs computed by uiPSF v1 (TensorFlow), for the oracle tests.

Run once, in an environment with TensorFlow and NumPy, against uiPSF's
``test_roi`` branch at b367f00 (the commit the review checked):

    git -C uiPSF checkout b367f00
    python scripts/uipsf/make_oracles_v1.py --uipsf path/to/uiPSF

It imports uiPSF's own model classes unchanged, with tkinter, omegaconf and
uiPSF's io package stubbed out, as the review's scripts do.  Each case
stores v1's output and the parameters in v2's units and conventions
(nm, x/y/z of `docs/uipsf/conventions.md`), so the test needs no TensorFlow.

The mapping from v1 to v2 (derived in docs/uipsf/conventions.md, and what the
test confirms):
- v1 places the image at −pos, so x_v2 = −pos_x · pixel size / bin.
- v1's z is the objective's position, increasing along the stack:
  z_v2 = −(pos_z + Zrange) · Δz.
- v1's depth (with_IMM) is the emitter's depth: depth_v2 = pos_depth · Δz.
- v1 keeps σ as π × (sd in upsampled pixels): σ_v2 = σ_v1 / π · pixel / bin.
- Zernike index i in v1 is Noll index i + 1; phase coefficients carry over.
"""
import argparse
import os
import sys
import types
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--uipsf", required=True, help="checkout of ries-lab/uiPSF at test_roi b367f00")
FIXTURE = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "uipsf_tests", "fixtures", "oracle_v1.npz")
ap.add_argument("--out", default=FIXTURE)
args = ap.parse_args()

np.product = np.prod                                    # NumPy 2 shim for the v1 code
REPO = os.path.abspath(args.uipsf)
pkg = types.ModuleType("psflearning")
pkg.__path__ = [os.path.join(REPO, "psflearning")]
sys.modules["psflearning"] = pkg
for name in ["tkinter", "tkinter.messagebox", "omegaconf", "psflearning.io", "psflearning.io.param",
             "psflearning.io.h5"]:
    sys.modules[name] = mock.MagicMock()
learning = types.ModuleType("psflearning.learning")
learning.__path__ = [os.path.join(REPO, "psflearning", "learning")]
sys.modules["psflearning.learning"] = learning
import scipy.ndimage  # noqa: E402

sys.modules.setdefault("scipy.ndimage.filters", scipy.ndimage)
import tensorflow as tf  # noqa: E402
from psflearning.learning.psfs.PSFZernikeBased_file import PSFZernikeBased  # noqa: E402
from psflearning.learning.utilities import genZern1  # noqa: E402

NA, LAM, NIMM, NMED, NCOV = 1.43, 0.68, 1.516, 1.335, 1.516
NZ, L, NMAX, PUPIL = 41, 21, 8, 64


def options(bin=1, with_imm=False):
    return NS(model=NS(with_IMM=with_imm, const_pupilmag=False, symmetric_mag=False, blur_sigma=0.5,
                       var_blur=True, var_photon=False, estimate_drift=False, bin=bin, pupilsize=PUPIL,
                       n_max=NMAX, with_apoid=True),
              imaging=NS(NA=NA, emission_wavelength=LAM, RI=NS(imm=NIMM, med=NMED, cov=NCOV)))


class Data:
    def __init__(self, n, px=(0.127, 0.116), pz=0.05, radius=0.0):
        self.rois = np.ones((n, NZ, L, L), np.float32)
        self.pixelsize_x, self.pixelsize_y, self.pixelsize_z = px[0], px[1], pz
        self.bead_radius, self.skew_const = radius, None
        self.centers = np.tile(np.array([[20, 100, 100]]), (n, 1))
        self.centers_all = self.centers

    def get_image_data(self):
        return self.rois, self.rois, self.centers, np.zeros(self.rois.shape[0])


def model(kind="vector", bin=1, n=1, px=(0.127, 0.116), radius=0.0, analyser=None, with_imm=False):
    m = PSFZernikeBased(options(bin, with_imm))
    m.psftype = kind
    m.polarization_type = analyser
    m.data = Data(n, px, radius=radius)
    m.calc_initials(m.data)
    return m


def coefficients(aberrated):
    c = np.zeros((2, (NMAX + 1) * (NMAX + 2) // 2, 1, 1), np.float32)
    c[0, 0] = 1
    if aberrated:
        for noll, v in {5: 0.6, 6: -0.3, 7: 0.25, 8: -0.2, 9: 0.15, 11: 0.4, 13: -0.1}.items():
            c[1, noll - 1] = v
        for noll, v in {4: -0.15, 6: 0.1, 11: 0.05}.items():
            c[0, noll - 1] = v
    return c


cases = {}


def store(name, image, *, kind="vector", bin=1, px=(0.127, 0.116), radius=0.0, analyser="",
          sigma_v1=0.0, aberrated=True, xyz=None, depth=None):
    c = coefficients(aberrated)
    if xyz is None:                                       # genpsfmodel: emitter at 0, stage over Zrange
        zr = np.real(np.linspace(-image.shape[0] / 2 + 0.5, image.shape[0] / 2 - 0.5, image.shape[0]))
        xyz = np.zeros((1, image.shape[0], 3))
        xyz[0, :, 2] = -zr * 50.0
        image = image[None]
    cases[name] = dict(image=np.asarray(image, np.float64), kind=kind, bin=bin, pixel_nm=np.array(px) * 1e3,
                       bead_radius_nm=radius * 1e3, analyser=analyser,
                       sigma_nm=np.array([sigma_v1 / np.pi * px[0] * 1e3 / bin, sigma_v1 / np.pi * px[1] * 1e3 / bin]),
                       phase=c[1, :, 0, 0].astype(np.float64), magnitude=c[0, :, 0, 0].astype(np.float64),
                       xyz=np.asarray(xyz, np.float64),
                       depth=np.zeros(xyz.shape[0]) if depth is None else np.asarray(depth, np.float64))


def gen(m, sigma, aberrated, addbead=False):
    s = tf.constant([sigma, sigma], tf.float32)
    image, _ = m.genpsfmodel(s, Zcoeff=tf.constant(coefficients(aberrated)), addbead=addbead)
    return np.real(np.array(image))


store("ideal", gen(model(), 0.0, False), aberrated=False)
store("aberrated", gen(model(), 0.5 * np.pi, True), sigma_v1=0.5 * np.pi)
store("bin2", gen(model(bin=2), 0.0, True), bin=2)
store("bin2_blur", gen(model(bin=2), 0.6 * np.pi, True), bin=2, sigma_v1=0.6 * np.pi)
store("scalar", gen(model("scalar"), 0.5 * np.pi, True), kind="scalar", sigma_v1=0.5 * np.pi)
store("analyser_x", gen(model(analyser="Ex"), 0.0, True), analyser="x")
store("analyser_y", gen(model(analyser="Ey"), 0.0, True), analyser="y")
m = model(px=(0.1, 0.1), radius=0.05)
store("bead", gen(m, 0.5 * np.pi, True, addbead=True), px=(0.1, 0.1), radius=0.05, sigma_v1=0.5 * np.pi)


def forward(m, pos, sigma):
    """calc_forward_images with unit photons and no background: one stack per emitter."""
    n = pos.shape[0]
    m.weight = np.ones(5, np.float32)
    v = [tf.constant(pos, tf.float32), tf.zeros((n, 1, 1, 1)), tf.ones((n, 1, 1, 1)),
         tf.constant(coefficients(True)), tf.constant([sigma, sigma], tf.float32), tf.zeros((n, 2))]
    return np.array(m.calc_forward_images(v))


# emitters off-centre and off-focus: positions (z [slices], y, x [upsampled pixels])
pos = np.array([[2.3, 1.2, -0.7], [-5.1, -2.0, 1.5], [0.0, 0.4, 0.3]], np.float32)
for b in (1, 2):
    m = model(bin=b, n=3)
    img = forward(m, pos, 0.5 * np.pi)
    zr = np.real(m.Zrange[:, 0, 0])
    st = (zr.shape[0] - NZ) // 2
    zr = zr[st:st + NZ]
    px = (0.127, 0.116)
    xyz = np.stack([np.broadcast_to(-pos[:, 2:3] * px[0] * 1e3 / b, (3, NZ)),
                    np.broadcast_to(-pos[:, 1:2] * px[1] * 1e3 / b, (3, NZ)),
                    -(pos[:, 0:1] + zr[None]) * 50.0], -1)
    store(f"shifted_bin{b}", img, bin=b, sigma_v1=0.5 * np.pi, xyz=xyz)

# an emitter 1 µm deep in the sample (with_IMM): positions (z, depth, y, x)
pos = np.array([[3.0, 20.0, 0.5, -0.3], [-4.0, 10.0, 0.0, 0.0]], np.float32)
m = model(n=2, with_imm=True)
img = forward(m, pos, 0.0)
zr = np.real(m.Zrange[:, 0, 0])
st = (zr.shape[0] - NZ) // 2
zr = zr[st:st + NZ]
xyz = np.stack([np.broadcast_to(-pos[:, 3:4] * 127.0, (2, NZ)), np.broadcast_to(-pos[:, 2:3] * 116.0, (2, NZ)),
                -(pos[:, 0:1] + zr[None]) * 50.0], -1)
store("depth", img, xyz=xyz, depth=pos[:, 1] * 50.0)

flat = {"zernike_v1": genZern1(NMAX, PUPIL).astype(np.float64),
        "optics": np.array([NA, LAM * 1e3, NIMM, NMED, NCOV]), "pupil_size": PUPIL}
for name, d in cases.items():
    for k, v in d.items():
        flat[f"{name}/{k}"] = v
flat["cases"] = np.array(list(cases))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez_compressed(args.out, **flat)
print("wrote", os.path.abspath(args.out), "cases:", ", ".join(cases))
