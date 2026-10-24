#!/usr/bin/env bash
# Final runs: sft / jepa / precode (frozen config) x seeds 42, 7, 1234 on the full train set (mbpp_train.jsonl).
# [vast, tmux finals, repo root]:  PC_ENV='LBD=1 HEAD_LR=3e-3 EXTRA="--h_layer 8"' bash train/finals.sh
# The watcher (eval/watch.sh) test-scores each model as its DONE file appears. HF upload happens later (train/upload_finals.sh),
# so the ~15 min/model upload never blocks training.
set -uo pipefail
cd /workspace/v1; export HF_HOME=/workspace/hf_cache; unset HF_MODEL_REPO TRAIN NAME
source /venv/main/bin/activate
: "${PC_ENV:?set PC_ENV to the frozen PreCode overrides}"
echo "$PC_ENV" > results/frozen_precode_config.txt
M=/data/precode/models
# Old jepa_s42/precode_s42 were trained before the grad-accum fix: keep them (renamed), never delete.
for r in jepa_s42 precode_s42; do
  [ -f $M/$r/PREFIX_ARCHIVED ] && continue
  if [ -d $M/$r ] && [ ! -e $M/${r}_prefix ]; then
    # rename runs.csv rows first, so the watcher sees the archived dir as already scored
    sed -i "s/^$r,/${r}_prefix,/" results/runs.csv
    for f in results/per_task/${r}_*.json; do [ -f "$f" ] && mv "$f" "${f/$r/${r}_prefix}"; done
    [ -d /data/precode/samples/$r ] && mv /data/precode/samples/$r /data/precode/samples/${r}_prefix
    [ -f $M/$r.log ] && mv $M/$r.log $M/${r}_prefix.log
    mv $M/$r $M/${r}_prefix && touch $M/${r}_prefix/PREFIX_ARCHIVED
    echo "[finals] archived $r -> ${r}_prefix"
  fi
done
for s in 42 7 1234; do
  for m in sft jepa precode; do
    r=${m}_s$s
    if [ -f $M/$r/DONE ]; then echo "[finals] $r already DONE, skip"; continue; fi
    echo "[finals] $(date +%H:%M) train $r"
    if [ $m = precode ]; then eval "$PC_ENV bash train/run_all.sh precode $s" > /dev/null 2>&1
    else bash train/run_all.sh $m $s > /dev/null 2>&1; fi
    [ -f $M/$r/DONE ] && echo "[finals] $(date +%H:%M) $r DONE" || echo "[finals] $(date +%H:%M) $r FAILED"
  done
done
echo "[finals] all training finished $(date +%H:%M)"
