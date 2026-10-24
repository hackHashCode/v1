#!/usr/bin/env bash
# Back up final models to the private HF repo, one at a time (network only; runs alongside eval).
cd /workspace/v1; export HF_HOME=/workspace/hf_cache; source /venv/main/bin/activate
eval "$(grep "^export HF_TOKEN" ~/.bashrc)"
for r in jepa_s42 precode_s42 sft_s7 jepa_s7 precode_s7 sft_s1234 jepa_s1234 precode_s1234; do
  python train/save_to_hub.py $r --repo gaganbuilds/precode-jepa-models 2>&1 | grep "\[hub\]" || echo "[hub] upload failed for $r"
done
echo "[hub] all done"
