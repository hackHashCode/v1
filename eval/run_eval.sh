#!/usr/bin/env bash
# usage: bash eval/run_eval.sh <run>   [WS, env evalv, tmux eval, ~/nikhita/precode-jepa]
set -euo pipefail
RUN=$1; MODEL=/data/precode/models/$RUN
[ "$RUN" = base_s0 ] && MODEL=meta-llama/Llama-3.2-1B-Instruct
[ "$RUN" = base_s0 ] || [ -f "$MODEL/DONE" ] || { echo "$RUN not DONE"; exit 1; }
OUT=/data/precode/samples/$RUN; mkdir -p $OUT

# EvalPlus writes <name>_eval_results.json (old) or <name>.eval_results.json (>=0.3), and silently
# reuses an existing one, so delete both before evaluating and pick whichever appears.
res() { local b=${1%.jsonl}; if [ -f "${b}_eval_results.json" ]; then echo "${b}_eval_results.json"; else echo "${b}.eval_results.json"; fi; }
rm -f $OUT/{mbpp,humaneval}_eval_results.json $OUT/{mbpp,humaneval}.eval_results.json

python eval/generate.py --model $MODEL --dataset mbpp --out $OUT/mbpp.jsonl
python eval/generate.py --model $MODEL --dataset humaneval --out $OUT/humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples $OUT/mbpp.jsonl
evalplus.evaluate --dataset humaneval --samples $OUT/humaneval.jsonl
python eval/score.py "$(res $OUT/mbpp.jsonl)" $RUN mbppplus224
python eval/score.py "$(res $OUT/humaneval.jsonl)" $RUN humanevalplus
