"""Fig. 2c's prefit over a stretch of the movie: the training window's own (`gui.train_window.run_prefit`,
Spline 3D 2C on blocks), its blocks placed in frames ``[start, stop)`` instead of across the movie.

    python fig2c_prefit.py MOVIE START STOP [BLOCKS] [BLOCK_FRAMES] [JOB_FOLDER]

Jonas (2026-11-22): the live cell's first frames are crowded (1 300 -- 2 300 molecules a frame); its
last are sparser, where a prefit should read the photons, the background and the on-time more
truly -- and the model trained from them may fit the movie better.  Prints the estimates; with
JOB_FOLDER, writes the window's job from them there (`write_job`)."""
import json
import sys

import numpy as np
import sets_c2 as S

from smappy.gui.train_window import PrepareSettings, run_prefit
from smappy.plugins.fit import CameraSettings
from smappy.training import prefit as P


def run(movie, start, stop, n_blocks=8, block_frames=50, job=None):
    def blocks(n_frames, n=n_blocks, length=block_frames):
        starts = np.linspace(start, stop - length, n).round().astype(int)
        return [(int(s), int(s) + length) for s in starts]
    P.blocks = blocks                       # prefit_two imports it at the call: the stretch asked for
    c = S.camera_of("biplane_2c")
    camera = CameraSettings(conversion=c["conversion"], offset=c["offset"], em_on=bool(c["em_on"]),
                            emgain=c["emgain"], pixelsize_um=S.PIXEL_NM[0] / 1000, pixelsize_y_um=S.PIXEL_NM[1] / 1000)
    cal = S.SETS_DIR / f"biplane_2c_{S.READING}_2ct.h5"
    s = PrepareSettings(movie=str(movie), calibration=str(cal), blocks=n_blocks, block_frames=block_frames,
                        model="optimised", camera=camera)
    p = run_prefit(s)
    e, two = p.estimates, p.two
    out = {"frames": [start, stop], "blinks": e.n_blinks, "localizations": e.n_localizations,
           "photons_mu_sd": [round(float(v)) for v in e.intensity_mu_sig],
           "background_main": [round(float(v)) for v in e.background],
           "background_per_channel": [[round(float(v)) for v in b] for b in two.background],
           "z_nm": [round(float(v)) for v in e.z_nm], "on_time_frames": round(float(e.lifetime_frames), 2),
           "per_40x40": round(float(e.emitters_per_frame), 1), "kind": two.kind}
    if job:
        from smappy.gui.train_window import write_job
        out["job"] = str(write_job(p, e, job))
    return out


if __name__ == "__main__":
    movie, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    extra = [int(v) for v in sys.argv[4:6]]
    print(json.dumps(run(movie, a, b, *extra, job=sys.argv[6] if len(sys.argv) > 6 else None)), flush=True)
