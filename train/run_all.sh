#!/usr/bin/env bash
# Train one (method, seed). Methods: sft | jepa | precode | precodence (PreCode with InfoNCE over all cached z). Run on the 4090 workstation, env `train`, from repo root, inside tmux.
#   bash train/run_all.sh <sft|jepa|precode|precodence> <seed>          e.g.  bash train/run_all.sh precode 42
#   LBD=0.5 bash train/run_all.sh precode 7                   (override lambda)
#   CACHE=fallback/behaviour_cache.pt bash train/run_all.sh precode 42
set -euo pipefail
M=$1; S=$2; RUN=${M}_s${S}; OUT=/data/precode/models/$RUN
mkdir -p /data/precode/models
BASE="--train_file datasets/mbpp_train.jsonl --model_name meta-llama/Llama-3.2-1B-Instruct --lora --lora_rank 16 \
 --batch_size 4 --grad_accum 4 --num_epochs 4 --max_length 512 --learning_rate 2e-4 \
 --use_default_data_collator --finetune_seed $S --output_dir $OUT"
case $M in
  sft)     python train/finetune.py $BASE --regular ;;
  jepa)    python train/finetune.py $BASE --additive_mask --predictors 1 --lbd 1.0 ;;
  precode) python train/finetune.py $BASE --regular --precode --behaviour_cache ${CACHE:-cache/behaviour_cache.pt} --lbd ${LBD:-1.0} ;;
  precodence) python train/finetune.py $BASE --regular --precode --precode_loss infonce --infonce_temp ${TEMP:-0.07} \
              --behaviour_cache ${CACHE:-cache/behaviour_cache.pt} --lbd ${LBD:-1.0} ;;
  *) echo "unknown method $M"; exit 1 ;;
esac 2>&1 | tee /data/precode/models/${RUN}.log
