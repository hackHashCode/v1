#!/usr/bin/env bash
# Evaluate every finished model automatically.  [WS, env evalv, tmux eval, repo root]:  bash eval/watch.sh
while true; do
  for d in /data/precode/models/*/; do
    r=$(basename "$d")
    [[ $r == smoke* || $r == sweep* ]] && continue
    [ -f "$d/DONE" ] || continue
    grep -q "^$r,.*,mbppplus224," results/runs.csv 2>/dev/null && continue
    echo "[watch] evaluating $r"; bash eval/run_eval.sh "$r" || echo "[watch] $r failed"
  done
  sleep 300
done
