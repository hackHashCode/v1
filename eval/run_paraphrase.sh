#!/usr/bin/env bash
# usage: bash eval/run_paraphrase.sh <run>   (after run_eval.sh <run>)  [WS, env evalv]
# Writes para60_para (reworded prompts) and para60_orig (same 60 tasks, original prompts) rows.
set -euo pipefail
RUN=$1; P=eval/paraphrase/mbppplus_paraphrased.jsonl; O=/data/precode/samples/$RUN
MODEL=/data/precode/models/$RUN; [ "$RUN" = base_s0 ] && MODEL=meta-llama/Llama-3.2-1B-Instruct
res() { local b=${1%.jsonl}; if [ -f "${b}_eval_results.json" ]; then echo "${b}_eval_results.json"; else echo "${b}.eval_results.json"; fi; }
rm -f $O/mbpp_para_eval_results.json $O/mbpp_para.eval_results.json
python eval/generate.py --model $MODEL --dataset mbpp --prompts $P --out $O/mbpp_para.jsonl
evalplus.evaluate --dataset mbpp --samples $O/mbpp_para.jsonl
python eval/score.py "$(res $O/mbpp_para.jsonl)" $RUN para60_para --ids $P
python eval/score.py "$(res $O/mbpp.jsonl)"      $RUN para60_orig --ids $P
