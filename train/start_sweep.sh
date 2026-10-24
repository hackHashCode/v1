#!/usr/bin/env bash
# Start (or restart) the val sweep runner in tmux `sweep`. Optional: score already-trained sweep dirs first.
cd /workspace/v1; export HF_HOME=/workspace/hf_cache; unset HF_MODEL_REPO
for d in "$@"; do
  v=$(HF_HUB_OFFLINE=1 /venv/evalv/bin/python eval/val_mbpp.py /data/precode/models/$d 2>&1 | grep " val " | tail -1 | awk '{print $3}')
  cos=$(grep "\[precode\] step" /data/precode/models/$d.log 2>/dev/null | tail -4 | awk '{s+=$9; n++} END {if (n) printf "%.3f", s/n; else print "-"}')
  printf "%s\t%s\t%s\t%s\t%s\n" "$(date +%H:%M)" "$d" "(val-only rescore)" "$v" "$cos" | tee -a results/sweep_log.tsv
  rm -rf /data/precode/models/$d
done
exec bash train/sweep_queue.sh
