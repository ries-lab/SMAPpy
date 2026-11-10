"""The reference trainings' parameters: the example package's param_run.yaml, its paths made local.

The file is from DECODE of 2020-07.  v0.10.2 fills what a file leaves out from its reference.yaml,
and so adds StepLR's step_size to the file's ReduceLROnPlateau, which then fails.  The scheduler
is therefore v0.10.2's own (StepLR, 10 epochs, 0.9, from reference.yaml); everything else is the
package's.
"""
import sys, pathlib, yaml
pkg, out, epochs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
p = yaml.safe_load((pkg / "param_run.yaml").read_text())
p["InOut"]["calibration_file"] = str(pkg / "spline_calibration_3dcal.mat")
p["InOut"]["experiment_out"] = str(out)
p["InOut"]["checkpoint_init"] = None
p["Hardware"]["device"] = "cuda:0"
p["Hardware"]["device_simulation"] = "cuda:0"
p["HyperParameter"]["epochs"] = epochs
p["HyperParameter"]["learning_rate_scheduler"] = "StepLR"
del p["HyperParameter"]["learning_rate_scheduler_param"]
out.mkdir(parents=True, exist_ok=True)
f = out / f"param_epochs{epochs}.yaml"
f.write_text(yaml.safe_dump(p, sort_keys=False))
print(f)
