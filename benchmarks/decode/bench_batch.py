"""The batch a training would choose on this machine's GPU (`decode_smlm.batch`, plan (f2)).

    python bench_batch.py PACKAGE_DIR [--device cuda] [--tf32 on|off]

The example package's network (param_run.yaml), its frames of 40 x 40: the network's step timed at
64 to 1024 frames, as ``batch_size: auto`` measures it before a training, and the batch chosen.
One JSON line to stdout.
"""
import argparse
import json
import pathlib
import platform

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package", type=pathlib.Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tf32", choices=["on", "off"], default=None, help="cuDNN's TF32; torch's default is on")
    a = ap.parse_args()

    from decode_smlm.batch import choose, measure
    from decode_smlm.neuralfitter import models
    from decode_smlm.utils import hardware, param_io

    if a.tf32 is not None:
        torch.backends.cudnn.allow_tf32 = a.tf32 == "on"
    device = a.device or hardware.default_device()
    param = param_io.load_params(a.package / "param_run.yaml")
    model = getattr(models, param.HyperParameter.architecture).parse(param)
    times = measure(model, int(param.HyperParameter.channels_in), (40, 40), device)
    gpu = torch.cuda.get_device_name(0) if torch.device(device).type == "cuda" else platform.processor()
    print(json.dumps({"device": device, "gpu": gpu, "torch": torch.__version__,
                      "tf32": torch.backends.cudnn.allow_tf32,
                      "us_per_frame": {str(b): round(t * 1e6, 1) for b, t in times.items()},
                      "chosen": choose(times)}))


if __name__ == "__main__":
    main()
