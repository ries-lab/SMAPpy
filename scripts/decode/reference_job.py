"""The reference trainings' job, as a job folder the port trains: milestone (f)'s comparison.

    python reference_job.py PACKAGE_DIR OUT_FOLDER --seed 0 [--epochs 1000] [--device auto]
                            [--calibration FILE] [--beads-em-on true|false] [--z-range LO HI]

The example package's param_run.yaml as the reference runs used it (scripts/decode/make_params.py:
v0.10.2's StepLR in place of the file's ReduceLROnPlateau), its calibration loaded by SMAPpy and
written into the folder, the paths relative.  The original drew everything without a seed; the
port's runs take one each, so that two runs are two draws, as the original's two were.

``--calibration`` trains on another model of the same microscope (the pixelation study,
benchmarks/decode/pixelation/), the package's settings otherwise.  ``--beads-em-on`` overrides the EM
state the calibration records, which is what the fitter decides the mirror by: an in situ model
learnt on frames flipped into the beads' orientation is in the beads' orientation, whatever camera
it was learnt with.  ``--z-range`` (nm) is the z the training draws its emitters at, in place of the
package's +-800: a model in situ reaches no lower than the coverslip.
"""
import argparse
import pathlib
import sys

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--calibration", type=pathlib.Path, default=None)
    ap.add_argument("--beads-em-on", choices=["true", "false"], default=None)
    ap.add_argument("--z-range", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    a = ap.parse_args()

    from smappy.io.calibration import load_spline_calibration
    from smappy.training import job as jobs

    p = yaml.safe_load((a.package / "param_run.yaml").read_text())
    p["InOut"].update(calibration_file=jobs.CALIBRATION, experiment_out=".", checkpoint_init=None, model_init=None)
    p["Hardware"].update(device=a.device, device_simulation=a.device)
    p["HyperParameter"].update(epochs=a.epochs, learning_rate_scheduler="StepLR")
    p["HyperParameter"].pop("learning_rate_scheduler_param", None)
    if a.z_range is not None:
        p["Simulation"]["emitter_extent"][2] = [float(a.z_range[0]), float(a.z_range[1])]
    from dataclasses import replace
    cal_file = a.calibration or a.package / "spline_calibration_3dcal.mat"
    cal = load_spline_calibration(cal_file)
    if a.beads_em_on is not None:
        cal = replace(cal, em_on=a.beads_em_on == "true")
    job = jobs.Job.new(method="decode-v0.10.2", seed=a.seed, device=a.device,
                       ranges={"from": "the example package's param_run.yaml"}, camera=p["Camera"],
                       calibration={"source": cal_file.name}, data={"package": a.package.name})
    jobs.write(a.out, job, p, cal)
    print(a.out)


if __name__ == "__main__":
    sys.exit(main())
