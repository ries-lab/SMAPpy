#!/bin/bash
# $1 python, $2 job name (jobs4pi/<job>.json): trained into jobs4pi/<job>, restarted from its checkpoint on a
# crash or a stall (30 min without a progress line), 3 tries; failures in jobs4pi/failures.log.
# Run from ~/netfit with the source in src_4pi and the models in cal/ (docs/4pi.md §30).
cd ~/netfit
PY=$1; J=$2
TRIES=3; STALL=1800
finished() { grep -q '"kind": "finished"' jobs4pi/$J/progress.jsonl 2>/dev/null; }
for try in $(seq 1 $TRIES); do
  finished $J && break
  env -u LD_LIBRARY_PATH PYTHONPATH=src_4pi nice $PY -u -W ignore -m smappy.netfit train jobs4pi/$J \
    --settings jobs4pi/$J.json --device cuda >> jobs4pi/$J.log 2>&1 &
  pid=$!; why=""
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    age=$(( $(date +%s) - $(stat -c %Y jobs4pi/$J/progress.jsonl 2>/dev/null || date +%s) ))
    if [ $age -gt $STALL ]; then why="stalled ${age}s"; kill $pid; sleep 20; kill -9 $pid 2>/dev/null; break; fi
  done
  wait $pid; code=$?
  finished $J && break
  echo "$(date '+%F %T') $J try $try/$TRIES: ${why:-exit $code}" >> jobs4pi/failures.log
  grep -v cudnn jobs4pi/$J.log | tail -3 | sed 's/^/    /' >> jobs4pi/failures.log
  sleep 30
done
finished $J && echo "$(date '+%F %T') $J DONE" >> jobs4pi/failures.log
