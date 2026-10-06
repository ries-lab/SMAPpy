"""Run one uiPSF learning job: the half of `smappy.uipsf` that runs *in uiPSF's
environment*, as a script.

It imports nothing of smappy's, on purpose: uiPSF wants its own TensorFlow and
usually its own Python, so smappy starts this file with that interpreter
(`runner.run`) and talks to it through a job directory and its stdout.  The
directory holds

    job.json     {"psftype": ..., "channeltype": ..., "params": {...}}
    images.npy   photons, already split into channels and cut as uiPSF wants

and the learnt model is written there as uiPSF writes it.  Every line meant for
smappy starts with ``SMAPPY_UIPSF`` and a tab -- ``stage``, ``result`` or
``error`` -- and everything else (TensorFlow's chatter, uiPSF's progress bars,
which go to stderr) is passed on as it comes.

The images bypass uiPSF's own loader (`psflearninglib.load_data`): smappy reads
the camera, the z positions, the split and the mirror itself, the same way its
own calibration does, so this only repeats what `load_data` does *after*
reading -- the channel axis first and the multi-channel defocus list.
"""
import json
import os
import sys
import time
import traceback

TAG = "SMAPPY_UIPSF"


def say(kind, text=""):
    print(f"{TAG}\t{kind}\t{text}", flush=True)


def merge(target, overrides):
    """Write ``overrides`` (nested dicts) into an OmegaConf node, key by key."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in target and hasattr(target[key], "keys"):
            merge(target[key], value)
        else:
            target[key] = value


def main(directory):
    os.environ.setdefault("MPLBACKEND", "Agg")      # uiPSF imports pyplot
    import numpy as np
    from psflearning import io
    from psflearning.psflearninglib import psflearninglib

    with open(os.path.join(directory, "job.json")) as f:
        job = json.load(f)
    say("stage", "starting uiPSF")
    L = psflearninglib()
    L.param = io.param.combine("config_base", psftype=job["psftype"],
                               channeltype=job["channeltype"])
    merge(L.param, job.get("params", {}))
    L.param.datapath = directory + os.sep
    L.param.savename = os.path.join(directory, "psfmodel")
    L.param.ref_channel = 0          # smappy puts its main channel first
    L.param.swapxy = False           # numpy (y, x), as smappy hands it over
    L.param.plotall = False
    images = np.load(os.path.join(directory, "images.npy")).astype(np.float32)
    if L.param.channeltype == "multi":
        # what load_data sets for a multi-channel acquisition: no defocus
        # between the channels unless the configuration says so
        n = images.shape[0]
        multi = L.param.option.multi
        L.param.option.multi.defocus = [multi.defocus_offset + i*multi.defocus_delay
                                        for i in range(n)]
    start = time.time()
    L.getpsfclass()
    say("stage", "finding the emitters")
    data = L.prep_data(images)
    say("stage", "learning the PSF")
    if "insitu" in L.param.PSFtype:
        result = L.iterlearn_psf(data, time=0)
    else:
        psf, fitter = L.learn_psf(data, time=0)
        result = L.save_result(psf, data, fitter)
    say("result", os.path.abspath(result))
    say("stage", f"done in {time.time() - start:.0f} s")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as error:            # noqa: BLE001 -- reported, not hidden
        traceback.print_exc()
        say("error", f"{type(error).__name__}: {error}".replace("\n", " "))
        sys.exit(1)
