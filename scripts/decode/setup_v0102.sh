#!/usr/bin/env bash
# Rebuild the reference environment of DECODE v0.10.2 on linux-64, in the home directory, no sudo.
# Usage: bash setup_v0102.sh [conda root, default ~/miniforge3]
# Then run with:  GLIBC_TUNABLES=glibc.rtld.execstack=2 <root>/envs/decode_fast/bin/python ...
# (glibc >= 2.41 refuses libtorch_cpu.so of torch 1.12, which asks for an executable stack.)
set -euo pipefail
ROOT=${1:-$HOME/miniforge3}
CONDA=$ROOT/bin/conda
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${DECODE_POC_WORK:-$HOME/decode_poc}
mkdir -p "$WORK"

# 1. DECODE 0.10.2, torch 1.12.0 and spline 0.10.0 exactly as recorded.
$CONDA create -y -n decode_fast --file "$HERE/v0102_linux64.explicit.txt"

# 2. SplinePSF 1f85d10 (0.11.0.dev0): the GPU renderer without dynamic parallelism, Python 3.8 still.
# 1f85d10 is on the fork where the fix was made (branch dev_fix_hopper), not on TuragaLab/SplinePSF.
[ -d "$WORK/SplinePSF" ] || git clone https://github.com/Haydnspass/SplinePSF.git "$WORK/SplinePSF"
git -C "$WORK/SplinePSF" checkout -q 1f85d1018756e8d27b858b114620ed25b7e8427a
$CONDA create -y -n spline_build -c conda-forge --override-channels python=3.8 numpy=1.19 \
    "pybind11>=2.8" cmake=3.25 ninja setuptools wheel pip gxx_linux-64=12 cuda-nvcc=12.4 \
    cuda-cudart-dev=12.4 cuda-version=12.4
(
  source "$ROOT/etc/profile.d/conda.sh"
  conda activate spline_build
  cd "$WORK/SplinePSF/python"
  rm -rf build dist ./*.egg-info
  CUDAARCHS="86;89" pip wheel --no-deps --no-build-isolation -w "$WORK/wheels" .
)

# 3. Swap it in.
$CONDA remove -y --force -n decode_fast spline
"$ROOT/envs/decode_fast/bin/pip" install --no-deps "$WORK"/wheels/spline-0.11.0.dev0-cp38-cp38-linux_x86_64.whl
GLIBC_TUNABLES=glibc.rtld.execstack=2 "$ROOT/envs/decode_fast/bin/python" -c \
    "import spline, decode, torch; assert spline.cuda_compiled and torch.cuda.is_available(); print('ok', decode.__version__)"
