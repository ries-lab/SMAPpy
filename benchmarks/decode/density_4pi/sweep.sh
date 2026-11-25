#!/bin/bash
# The sweep of §32 on gpuvm: every density, on-times 1 and 3, the fits named (all by default).  Run from the
# folder with density_compare.py, models/ and cal/, the source two levels up (as ~/netfit/smappy_density/bench).
# $1: a suffix for the output files, then the fits.  Movies of 100 frames, enough of them for ~6000 molecules.
cd "$(dirname "$0")"
PY=${PY:-~/miniconda3/envs/decode_port/bin/python}
SUFFIX=$1; shift
FITS=${@:-netfit_4pi spline_4pi netfit_astig spline_3d netfit_astig_half spline_3d_half}
mkdir -p results
for on in 1 3; do
  for dm in "0.5 20" "1 10" "2 6" "5 3" "10 2" "25 1" "50 1"; do
    set -- $dm
    out=results/d$1_on${on}$SUFFIX
    [ -f $out.json ] && continue
    echo "== d$1 on$on $(date +%T)"
    env -u LD_LIBRARY_PATH PYTHONPATH=../../src OMP_NUM_THREADS=8 nice $PY -W ignore density_compare.py \
      --density $1 --on $on --movies $2 --frames 100 --model4pi models/netfit_4pi_A_6M.safetensors \
      --astig models/netfit_astig_full_6M.safetensors --astig-half models/netfit_astig_half_6M.safetensors \
      --fits $FITS --device cuda --out $out 2>&1 | grep -v "^movie"
  done
done
echo ALLDONE
