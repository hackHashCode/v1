#!/usr/bin/env bash
# Parallel version of post_finals.sh: test evals for unscored finals, then attempts-to-green and paraphrase,
# 3 GPU jobs at a time (each vLLM takes 0.28 of the GPU). Then McNemar, table, heroes.
# [vast, tmux post, repo root]:  bash train/fast_post.sh
set -uo pipefail
cd /workspace/v1; export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1; source /venv/evalv/bin/activate
RUNS="sft_s42 jepa_s42 precode_s42 sft_s7 jepa_s7 precode_s7 sft_s1234 jepa_s1234 precode_s1234"
L=/data/precode/models/post_logs; mkdir -p $L results/attempts
scored() { grep -q "^$1,.*,mbppplus224," results/runs.csv && grep -q "^$1,.*,humanevalplus," results/runs.csv; }
one() {   # $1 = kind, $2 = run
  case $1 in
    eval) scored $2 || bash eval/run_eval.sh $2 ;;
    att)  [ -f results/attempts/$2.json ] || python eval/attempts.py $2 ;;
    para) n=0; until scored $2; do n=$((n+1)); [ $n -gt 90 ] && return 1; sleep 20; done
          grep -q "^$2,.*,para60_para," results/runs.csv || bash eval/run_paraphrase.sh $2 ;;
  esac
}
task() {   # retry: a vLLM engine fails its startup memory check if another job frees GPU memory meanwhile
  for i in 1 2 3 4 5 6 7 8; do
    if one $1 $2 > $L/$1_$2.log 2>&1; then echo "[post] $(date +%H:%M) ok $1 $2 (try $i)"; return 0; fi
    grep -q "Error in memory profiling\|Engine core initialization failed\|KV cache" $L/$1_$2.log || break
    sleep $((15 + RANDOM % 45))
  done
  echo "[post] $(date +%H:%M) FAILED $1 $2 (see $L/$1_$2.log)"
}
export -f task one scored; export L
{
  for r in $RUNS; do scored $r || echo "eval $r"; done
  for r in base_s0 $RUNS; do echo "att $r"; done
  for r in base_s0 $RUNS; do echo "para $r"; done
} | xargs -P 3 -L 1 bash -c 'task "$0" "$1"'
[ -f results/stats.md ] && mv results/stats.md results/stats_old_$(date +%s).md
for b in sft jepa; do for d in mbppplus224 humanevalplus; do python eval/mcnemar.py precode $b $d; done; done
python train/make_table.py
python eval/pick_heroes.py 2>&1 | tail -3
echo "[post] $(date +%H:%M) DONE"
