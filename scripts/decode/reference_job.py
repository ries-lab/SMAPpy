"""The reference trainings' job, as a job folder the port trains: milestone (f)'s comparison.

    python reference_job.py PACKAGE_DIR OUT_FOLDER --seed 0 [--epochs 1000] [--device auto]

The example package's param_run.yaml as the reference runs used it (scripts/decode/make_params.py:
v0.10.2's StepLR in place of the file's ReduceLROnPlateau), its calibration loaded by SMAPpy and
written into the folder, the paths relative.  The original drew everything without a seed; the
port's runs take one each, so that two runs are two draws, as the original's two were.
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
    a = ap.parse_args()

    from smappy.io.calibration import load_spline_calibration
    from smappy.training import job as jobs

    p = yaml.safe_load((a.package / "param_run.yaml").read_text())
    p["InOut"].update(calibration_file=jobs.CALIBRATION, experiment_out=".", checkpoint_init=None, model_init=None)
    p["Hardware"].update(device=a.device, device_simulation=a.device)
    p["HyperParameter"].update(epochs=a.epochs, learning_rate_scheduler="StepLR")
    p["HyperParameter"].pop("learning_rate_scheduler_param", None)
    cal_file = a.package / "spline_calibration_3dcal.mat"
    job = jobs.Job.new(method="decode-v0.10.2", seed=a.seed, device=a.device,
                       ranges={"from": "the example package's param_run.yaml"}, camera=p["Camera"],
                       calibration={"source": cal_file.name}, data={"package": a.package.name})
    jobs.write(a.out, job, p, load_spline_calibration(cal_file))
    print(a.out)


if __name__ == "__main__":
    sys.exit(main())
