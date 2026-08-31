"""What a built wheel has to be able to do, run inside cibuildwheel.

Importing the extension modules only proves they load.  This fits a frame with
a known emitter in it, which is the cheapest check that the compiler produced
working code -- an optimizer flag that silently did the wrong thing, or a
platform where the fit diverges, fails here rather than in a lab.
"""
import sys

import numpy as np

import smappy

SHAPE = (48, 48)
TRUTH = (23.4, 19.7)          # x, y in pixels


def frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    clean = 3000.0 * np.exp(-((x - TRUTH[0]) ** 2 + (y - TRUTH[1]) ** 2) / 2.0)
    return (rng.poisson(clean) + 100).astype(np.uint16)


def main() -> int:
    frames = np.stack([frame(i) for i in range(4)])
    locs = smappy.fit(frames, camera={"conversion": 1.0, "offset": 100.0,
                                      "pixelsize_um": 0.1},
                      sigma=1.2, cutoff=40.0, roisize=9, units="nm")

    assert len(locs) == 4, f"expected one localization per frame, got {len(locs)}"
    x, y = locs["x_nm"].mean() / 100.0, locs["y_nm"].mean() / 100.0
    assert abs(x - TRUTH[0]) < 0.2 and abs(y - TRUTH[1]) < 0.2, \
        f"fitted ({x:.2f}, {y:.2f}), expected {TRUTH}"
    assert (locs["photons"] > 1000).all(), "photon counts are not plausible"

    # the other two extensions: grouping and rendering
    grouped, _ = smappy.group(locs)
    assert len(grouped) >= 1
    image = smappy.render_locs(
        locs, smappy.FieldOfView.around(locs["x_nm"], locs["y_nm"], pixelsize=10.0))
    assert image.n_locs == len(locs) and image.weight.sum() > 0, \
        "the render produced nothing"

    print(f"smappy {smappy.__version__} on {sys.platform}: "
          f"fitted ({x:.2f}, {y:.2f}) px, {len(locs)} localizations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
