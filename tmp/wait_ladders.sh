#!/bin/bash
# Exit when both pipelines reach a terminal state (complete OR dead without completing).
done_or_dead () {
  local log=$1 pid=$2
  grep -q "Pipeline complete" "$log" && { echo "ok"; return; }
  kill -0 "$pid" 2>/dev/null || { echo "dead"; return; }
  echo "running"
}
while true; do
  u=$(done_or_dead tmp/rerun_unsw2.log 86808)
  t=$(done_or_dead tmp/rerun_ton2.log 86809)
  [ "$u" != "running" ] && [ "$t" != "running" ] && break
  sleep 30
done
echo "TERMINAL unsw=$u ton=$t"
for d in unsw ton; do
  echo "--- $d"; grep -A8 "LADDER (pooled OOF" tmp/rerun_${d}2.log | tail -12
  tail -4 tmp/rerun_${d}2.log
done
