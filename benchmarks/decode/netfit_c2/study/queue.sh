#!/bin/bash
# The C = 2 study's queue on one machine (plex-plan §16): each job of queue.txt trained in turn (a
# stall of 30 minutes or a crash restarts it, three tries; a finished one is skipped), then its model
# fitted on the blinking C = 2 sets.  Run from the study folder:
#   PY=python ENV="env -u LD_LIBRARY_PATH" SRC=... SUITE=... DATA=... [WAIT=progress.jsonl] ./queue.sh
cd "$(dirname "$0")"
log=queue.log
[ -n "$WAIT" ] && until grep -q '"finished"' "$WAIT"; do sleep 60; done
for job in $(cat queue.txt); do
  d=runs/$job
  for try in 1 2 3; do
    grep -q '"finished"' $d/progress.jsonl 2>/dev/null && break
    mkdir -p $d
    echo "$(date '+%F %T') START $job try $try" >> $log
    nice -n 10 $ENV PYTHONPATH=$SRC $PY -m smappy.netfit train $d --settings jobs/$job.json --device cuda >> $d/train.log 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      [ -f $d/progress.jsonl ] || continue
      age=$(( $(date +%s) - $(stat -c %Y $d/progress.jsonl) ))
      if [ $age -gt 1800 ]; then kill $pid; echo "$(date '+%F %T') STALL $job" >> $log; fi
    done
    wait $pid
    grep -q '"finished"' $d/progress.jsonl || echo "$(date '+%F %T') FAIL $job try $try" >> $log
  done
  if grep -q '"finished"' $d/progress.jsonl 2>/dev/null && [ ! -f fits/$job/done ]; then
    case $job in S_biplane_*) ctx=biplane_2c;; *) ctx=colour_2a;; esac
    mkdir -p fits/$job
    root=$(pwd)
    (cd $SUITE && nice -n 10 $ENV DECODE_SMLM_DATA=$DATA PYTHONPATH=$SRC $PY fit_blink_c2.py $ctx "$root/$d/model.safetensors" "$root/fits/$job") >> fits/$job/fit.log 2>&1 \
      && touch fits/$job/done || echo "$(date '+%F %T') FITFAIL $job" >> $log
  fi
  echo "$(date '+%F %T') DONE $job" >> $log
done
echo "$(date '+%F %T') QUEUEDONE" >> $log
