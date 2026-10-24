#!/usr/bin/env bash
# Train one (method, seed). Methods: sft | jepa | precode | precodence (PreCode with InfoNCE over all cached z). Run on the 4090 workstation, env `train`, from repo root, inside tmux.
#   bash train/run_all.sh <sft|jepa|precode|precodence> <seed>          e.g.  bash train/run_all.sh precode 42
#   LBD=0.5 bash train/run_all.sh precode 7                   (override lambda)
#   HEAD_LR=1e-3 (pred_head lr)   EXTRA="--lbd_schedule decay --h_pool mean" (extra finetune.py flags, all methods)
#   CACHE=fallback/behaviour_cache.pt bash train/run_all.sh precode 42
#   MODEL=Qwen/Qwen2.5-Coder-1.5B MODELS_ROOT=/data/precode/qc15/models TRAIN=datasets/mbpp_train_noval.jsonl  (other model / root / train file)
set -euo pipefail
M=$1; S=$2; RUN=${NAME:-${M}_s${S}}; ROOT=${MODELS_ROOT:-/data/precode/models}; OUT=$ROOT/$RUN
mkdir -p $ROOT
BASE="--train_file ${TRAIN:-datasets/mbpp_train.jsonl} --model_name ${MODEL:-unsloth/Llama-3.2-1B-Instruct} --lora --lora_rank 16 \
 --batch_size 4 --grad_accum 4 --num_epochs ${EPOCHS:-2} --max_length 512 --learning_rate ${LR:-5e-5} \
 --use_default_data_collator --unmask_assistant_special_tokens --finetune_seed $S --output_dir $OUT ${EXTRA:-}"
PC="${HEAD_LR:+--head_lr $HEAD_LR}"
case $M in
  sft)     HF_HUB_OFFLINE=1 python train/finetune.py $BASE --regular ;;
  jepa)    HF_HUB_OFFLINE=1 python train/finetune.py $BASE --additive_mask --predictors 1 --lbd 1.0 ;;
  precode) HF_HUB_OFFLINE=1 python train/finetune.py $BASE --regular --precode --behaviour_cache ${CACHE:-cache/behaviour_cache.pt} --lbd ${LBD:-1.0} $PC ;;
  precodence) HF_HUB_OFFLINE=1 python train/finetune.py $BASE --regular --precode --precode_loss infonce --infonce_temp ${TEMP:-0.07} \
              --behaviour_cache ${CACHE:-cache/behaviour_cache.pt} --lbd ${LBD:-1.0} $PC ;;
  *) echo "unknown method $M"; exit 1 ;;
esac 2>&1 | tee $ROOT/${RUN}.log
if [ -n "${HF_MODEL_REPO:-}" ] && [ -f "$OUT/DONE" ]; then
  python train/save_to_hub.py "$RUN" --repo "$HF_MODEL_REPO" 2>&1 | tee -a $ROOT/${RUN}.log || echo "[hub] upload failed for $RUN (model still on local disk)"
fi
