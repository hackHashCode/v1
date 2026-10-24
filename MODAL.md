# Running everything on Modal

`modal_app.py` tunes every method on validation data, trains each with 3 seeds, evaluates on MBPP+ and HumanEval+,
computes behaviour metrics and writes the comparison. All jobs run in parallel on Modal GPUs.

## 1. One-time setup (laptop, repo root)

```bash
pip install modal
python -m modal setup
python -m modal secret create huggingface HF_TOKEN=hf_xxx
```
On Windows also run `$env:PYTHONUTF8="1"` in each new PowerShell window before `modal run`.

## 2. Run (default model: Qwen2.5-Coder-1.5B base)

```bash
python -m modal run --detach modal_app.py
```
Pipeline: **sweep → select → train → eval → analyze → compare → download**. Follow it at https://modal.com/apps;
results land in `results_modal_qc15/` (`COMPARISON.md`, `SWEEP.md`, `tsne_*.png`). Finished work is skipped on reruns;
a run is retrained / re-evaluated automatically when its selected config or behaviour cache changes
(each model folder records its `train_config.json`; sweep results record the cache they used).

Single stages: `--stage sweep | select | train | eval | analyze | compare | download`.
Other flags: `--methods sft,precodefull_aux`, `--seeds 42,7,1234,11,99`, `--force` (redo everything),
`--force-methods a,b` (redo the sweep for these only), `--extra "--lbd 0.3"`,
`--model llama1b` (the original Llama-3.2-1B-Instruct runs, kept at their old paths), `--upload-repo you/repo`.
GPU: `PRECODE_GPU=A100-40GB` (default L40S).

## 3. How settings are chosen (no test-set tuning)

For every trained method, a grid of learning rate {1e-5, 2e-5, 5e-5, 1e-4, 2e-4} × epochs {1, 2, 3} is trained on
`mbpp_train_noval.jsonl` (384 tasks, seed 42) and scored on `mbpp_val` + `mbpp_valext` (365 tasks that are in
neither training nor the test sets). The best setting per method (ties → fewer epochs, lower lr) goes to
`results/best_config.json` and is then used for the 3-seed runs on all 474 training tasks. The test sets are
evaluated once, after selection. Note: the behaviour cache contains the 90 `mbpp_val` tasks' z vectors, so during
the sweep they appear as InfoNCE negatives (never as training targets).

## 4. Methods

| method | trainer | what it adds |
|---|---|---|
| `base` | none | zero-shot base model |
| `sft` | `train/finetune.py --regular` | next-token loss only |
| `jepa` | `train/finetune.py --predictors 1 --lbd 1.0` | LLM-JEPA: predict own embedding of the code text |
| `precode` | `train/finetune.py --precode` | + λ·(1 − cos(W·h_prompt, z_behaviour)) |
| `precodence` | `train/finetune.py --precode --precode_loss infonce` | same, InfoNCE against all cached z |
| `sft_loop` | `train/precode_full.py --mode aux --lbd 0` | control: plain SFT through the PreCode-full loop |
| `precodefull_aux` | `train/precode_full.py --mode aux` | warmed-up MLP behaviour head + InfoNCE, λ 0.5 |
| `precodefull_prefix` | `train/precode_full.py --mode prefix` | + 4 soft tokens from predicted z, read while writing |
| `precodefull` | `train/precode_full.py --mode full` | + hard-task NTP weights 1 − cos, detached, mean 1 |

All share LoRA r16, effective batch 16, 474 training tasks; lr and epochs come from the sweep.

## 5. Metrics in `COMPARISON.md`

- pass@1 base and plus on MBPP+ (224) and HumanEval+ (164), mean ± std over seeds, 95% bootstrap CI over tasks
- exact McNemar tests pooled over seeds: every method vs SFT, the ablation ladder, PreCode vs LLM-JEPA
- behaviour metrics on MBPP+ test tasks (`eval/analyze.py`): behaviour match (partial credit per probe),
  exact behaviour, Recall@1 / @5 of the predicted z among 224 unseen tasks (random 0.45% / 2.2%),
  cos(predicted, true behaviour), cos(predicted, behaviour of the code the model wrote), cos split by solved / unsolved
- t-SNE of predicted vs true behaviour for each seed-42 run with a behaviour head (`tsne_<run>.png`)
- end-of-training head fit per PreCode-full run (training tasks)

## 6. Demo with the trained models

```bash
python -m modal volume get precode-data precode/qc15/models/precodefull_aux_s42 ./models/
python -m modal volume get precode-data precode/qc15/models/sft_s42 ./models/       # also jepa_s42, precode_s42
MODELS_DIR=./models BASE_MODEL=Qwen/Qwen2.5-Coder-1.5B uvicorn demo.studio.server:app --port 8010
```
