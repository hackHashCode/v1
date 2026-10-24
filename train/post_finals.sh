#!/usr/bin/env bash
# After finals: wait until the watcher has test-scored all 9 final runs, then attempts-to-green, paraphrase, McNemar, table, heroes.
# [vast, tmux post, repo root]:  bash train/post_finals.sh
set -uo pipefail
cd /workspace/v1; export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1; source /venv/evalv/bin/activate
RUNS="sft_s42 jepa_s42 precode_s42 sft_s7 jepa_s7 precode_s7 sft_s1234 jepa_s1234 precode_s1234"
while true; do
  miss=0; for r in $RUNS; do for d in mbppplus224 humanevalplus; do grep -q "^$r,.*,$d," results/runs.csv || miss=$((miss+1)); done; done
  [ $miss = 0 ] && break; echo "[post] $(date +%H:%M) waiting for watcher: $miss rows missing"; sleep 120
done
echo "[post] $(date +%H:%M) all test rows present"
for r in base_s0 $RUNS; do
  [ -f results/attempts/$r.json ] || { echo "[post] attempts $r"; python eval/attempts.py $r 2>&1 | tail -1; }
  grep -q "^$r,.*,para60_para," results/runs.csv || { echo "[post] paraphrase $r"; bash eval/run_paraphrase.sh $r 2>&1 | tail -2; }
done
[ -f results/stats.md ] && mv results/stats.md results/stats_old_$(date +%s).md
for b in sft jepa; do for d in mbppplus224 humanevalplus; do python eval/mcnemar.py precode $b $d; done; done
python train/make_table.py
python eval/pick_heroes.py 2>&1 | tail -3
echo "[post] $(date +%H:%M) DONE"
