#!/usr/bin/env bash
# The reference trainings of milestone (a): DECODE v0.10.2's live engine on the example package,
# its parameters as make_params.py turns them local.  Each run is detached from the shell
# (setsid), so it survives the ssh connection; train.log and the checkpoint's log dict are its record.
# Usage: bash train_reference_v0102.sh RUN_NAME [EPOCHS]
set -euo pipefail
NAME=$1; EPOCHS=${2:-1000}
ROOT=${CONDA_ROOT:-$HOME/miniforge3}
P=$ROOT/envs/decode_fast/bin/python
PKG=${DECODE_SMLM_DATA:-$HOME/.cache/decode_smlm}/experimental_data_workflow
OUT=${DECODE_POC_WORK:-$HOME/decode_poc}/runs_reference/$NAME
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
F=$($P "$HERE/make_params.py" "$PKG" "$OUT" "$EPOCHS")
export GLIBC_TUNABLES=glibc.rtld.execstack=2
cd "$OUT"
setsid nohup nice -n 10 $P -W ignore -m decode.neuralfitter.train.live_engine -p "$F" \
    -l "$OUT/tb" -c "$NAME" > "$OUT/train.log" 2>&1 < /dev/null &
echo "started $NAME, pid $!, in $OUT"
