#!/bin/bash
# $1 python, then job names (jobs_density/<job>.json): each trained into jobs_density/<job> in turn, restarted
# from its checkpoint on a crash or a stall (30 min without a progress line), 3 tries; failures in
# jobs_density/failures.log.  Run from ~/netfit with the source in src_density and the models in cal/
# (docs/4pi.md §33); the same loop as ../netfit_4pi/run.sh.
cd ~/netfit
PY=$1; shift
TRIES=3; STALL=1800
finished() { grep -q '"kind": "finished"' jobs_density/$1/progress.jsonl 2>/dev/null; }
for J in "$@"; do
  for try in $(seq 1 $TRIES); do
    finished $J && break
    env -u LD_LIBRARY_PATH PYTHONPATH=src_density nice $PY -u -W ignore -m smappy.netfit train jobs_density/$J \
      --settings jobs_density/$J.json --device cuda >> jobs_density/$J.log 2>&1 &
    pid=$!; why=""
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      age=$(( $(date +%s) - $(stat -c %Y jobs_density/$J/progress.jsonl 2>/dev/null || date +%s) ))
      if [ $age -gt $STALL ]; then why="stalled ${age}s"; kill $pid; sleep 20; kill -9 $pid 2>/dev/null; break; fi
    done
    wait $pid; code=$?
    finished $J && break
    echo "$(date '+%F %T') $J try $try/$TRIES: ${why:-exit $code}" >> jobs_density/failures.log
    grep -v cudnn jobs_density/$J.log | tail -3 | sed 's/^/    /' >> jobs_density/failures.log
    sleep 30
  done
  finished $J && echo "$(date '+%F %T') $J DONE" >> jobs_density/failures.log
done
