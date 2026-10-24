#!/usr/bin/env bash
# Val-only sweep runner. Pops lines from $Q, trains on mbpp_train_noval, scores on val (90 tasks), deletes the model.
# Queue line: <sweep_name> <method> <seed> [ENV=val ...]    e.g.  sweep_pc_h1e-3 precode 42 LBD=1 HEAD_LR=1e-3 EXTRA="--h_pool mean"
# [vast, tmux sweep, repo root]:  bash train/sweep_queue.sh      Results: results/sweep_log.tsv + results/val_sweep.txt
Q=${Q:-/data/precode/sweep_queue.txt}; touch "$Q"
export HF_HOME=/workspace/hf_cache; unset HF_MODEL_REPO   # sweeps never upload
while true; do
  line=$(head -1 "$Q")
  if [ -z "$line" ]; then sleep 30; continue; fi
  sed -i 1d "$Q"
  read -r name method seed rest <<< "$line"
  [[ $name == sweep_* ]] || { echo "[sweep] refusing non-sweep name $name"; continue; }
  echo "[sweep] $(date +%H:%M) start $name ($method $seed $rest)"
  ( source /venv/main/bin/activate && eval "NAME=$name TRAIN=datasets/mbpp_train_noval.jsonl $rest bash train/run_all.sh $method $seed" ) > /dev/null 2>&1
  M=/data/precode/models/$name
  if [ -f "$M/DONE" ]; then
    v=$(HF_HUB_OFFLINE=1 /venv/evalv/bin/python eval/val_mbpp.py "$M" 2>&1 | grep -E " (val|valext) [0-9]" | awk '{print $3}' | paste -sd' ' -)
  else v=FAILED; fi
  cos=$(grep "\[precode\] step" /data/precode/models/$name.log 2>/dev/null | tail -4 | awk '{s+=$9; n++} END {if (n) printf "%.3f", s/n; else print "-"}')
  printf "%s\t%s\t%s %s %s\t%s\t%s\n" "$(date +%H:%M)" "$name" "$method" "$seed" "$rest" "$v" "$cos" | tee -a results/sweep_log.tsv
  rm -rf "$M"
done
