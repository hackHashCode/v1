# Running everything on Modal

One command trains every method on every seed in parallel, evaluates each run on MBPP+ and HumanEval+, and
writes the comparison. Models, samples and results live on the Modal volume `precode-data` (mounted at `/precode-vol` and linked as `/data/precode`,
so every path in `CONTRACTS.md` still holds).

## 1. One-time setup (your laptop, repo root)

```bash
pip install modal
modal setup                                          # opens a browser to log in
modal secret create huggingface HF_TOKEN=hf_xxx      # read token; a write token if you also want --upload-repo
```

## 2. Run

```bash
modal run --detach modal_app.py                      # everything: 7 methods x 3 seeds + zero-shot, ~20 GPU jobs in parallel
```
`--detach` keeps it running if your laptop sleeps; follow it at https://modal.com/apps.

Smaller runs (do this first to check the pipeline end to end, ~15 min):
```bash
modal run modal_app.py --methods base,sft,precodefull --seeds 42
```

Single stages:
```bash
modal run modal_app.py --stage train   --methods precodefull_prefix,precodefull --seeds 42,7,1234
modal run modal_app.py --stage eval    --methods sft,precodefull --seeds 42,7,1234
modal run modal_app.py --stage compare                         # rebuild COMPARISON.md from whatever is evaluated
modal run modal_app.py --stage download                        # copy results to ./results_modal/
```
Already-trained runs (with a `DONE` file) are skipped; add `--force` to retrain. Other flags:
`--epochs 3`, `--extra "--lbd 0.3"` (passed to the trainer), `--upload-repo you/precode-jepa-models` (HF backup).
GPU type: `PRECODE_GPU=A100-40GB modal run modal_app.py` (default L40S).

## 3. Methods

| method | trainer | what it adds |
|---|---|---|
| `base` | none | zero-shot Llama-3.2-1B-Instruct |
| `sft` | `train/finetune.py --regular` | next-token loss only |
| `jepa` | `train/finetune.py --predictors 1 --lbd 1.0` | LLM-JEPA: predict own embedding of the code text |
| `precode` | `train/finetune.py --precode` | + λ·(1 − cos(W·h_prompt, z_behaviour)) |
| `precodence` | `train/finetune.py --precode --precode_loss infonce` | same, InfoNCE against all 470 cached z |
| `sft_loop` | `train/precode_full.py --mode aux --lbd 0` | control: plain SFT through the PreCode-full training loop |
| `precodefull_aux` | `train/precode_full.py --mode aux` | MLP head (warmed up on frozen prompt states, then joint) + InfoNCE, λ 0.5 |
| `precodefull_prefix` | `train/precode_full.py --mode prefix` | + 4 soft tokens from predicted z, read while writing (on from 34% of training) |
| `precodefull` | `train/precode_full.py --mode full` | + hard-task NTP weights 1 − cos, detached, mean 1 (on from 67%) |

The first five use Gagan's `train/run_all.sh` unchanged, so they reproduce the earlier workstation runs.
All methods share the recipe: LoRA r16, lr 5e-5, effective batch 16, 2 epochs, 474 training tasks.

## 4. Outputs

- `results_modal/COMPARISON.md`: pass@1 table (mean ± std over seeds), exact McNemar tests pooled over seeds for
  every method vs SFT plus the ablation ladder, and end-of-training behaviour stats for the PreCode-full runs.
- `results_modal/rows/`, `results_modal/per_task/`: one file per run and dataset (contracts C9, C10).
- `results_modal/runs.csv`: rebuilt from all runs, so the studio's Results tab shows everything.
- `results_modal/train_logs/`: per-step losses, cos(own z), top-1, stage switches.
- On the volume: `/data/precode/models/<run>/` (merged model, `pred_head.pt`, `prefix_injector.pt`,
  `precode_config.json`), `/data/precode/samples/<run>/` (generations, EvalPlus results, predicted z per task).

Every method is generated with the same script (`eval/generate_hf.py`: HF `generate`, greedy, 384 tokens,
contract C13 prompt), so decoding is identical across methods. Numbers will differ slightly from the earlier
vLLM-generated workstation results; compare methods within one pipeline, never across.

## 5. Demo with the trained models

Download the runs you want to demo, then point the studio at them:
```bash
modal volume get precode-data precode/models/precodefull_s42 ./models/precodefull_s42
modal volume get precode-data precode/models/sft_s42 ./models/sft_s42          # repeat for jepa_s42, precode_s42
MODELS_DIR=./models uvicorn demo.studio.server:app --port 8010
```
The studio picks up `precodefull_s42` automatically and generates with its prefix, exactly as in training.

## 6. Rough cost

About 20 training jobs (5–20 min each on an L40S) and 22 eval jobs (10–20 min each), so roughly 8–12 GPU-hours.
Check current GPU prices at https://modal.com/pricing before launching the full grid; the one-seed smoke run
above costs a small fraction of that.
