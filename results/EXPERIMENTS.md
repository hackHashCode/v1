# PreCode experiments log

Protocol: train on `datasets/mbpp_train_noval.jsonl` (384 tasks), score on `datasets/mbpp_val.jsonl` (90 held-out MBPP tasks, all 3 asserts) with `eval/val_mbpp.py`.
1 task = 1.1 pts; noise is about ±4 tasks. Shared recipe: Llama-3.2-1B-Instruct, LoRA r16, batch 16, lr 5e-5, 2 epochs, seed 42 unless stated.
`cos` = mean cos(W·h, z) over the last 4 logged micro-batches (end of training). Raw rows: `results/sweep_log.tsv`.

## 0. Setup notes (2026-10-24 ~21:20 IST)
- **Why the GPU was idle:** the earlier tmux `sweep` was the 4-config SFT recipe sweep. It finished cleanly (lr 5e-5/2ep chosen, 27/90).
  `train/sweep_precode.sh` was never created and the `--head_lr` patch was not in `train/finetune.py`, so nothing else was queued.
- **Bug found and fixed: grad-accum loss scaling.** `PreCodeTrainer` and `RepresentationTrainer` (LLM-JEPA) return a per-micro-batch mean loss, but
  Trainer (transformers 4.55) skips the `/grad_accum` division because Llama's forward accepts loss kwargs. So PreCode and JEPA gradients were 4× SFT's,
  logged loss was 4× (precode 7.8, jepa 4.6 vs sft 0.93), and grad_norm 5–6 was clipped to 1.0 on every step. SFT was unaffected.
  Fix: both trainers set `model_accepts_loss_kwargs = False` (`--no_ga_fix` restores the old behaviour). Under Adam the 4× scale mostly cancels,
  but constant clipping still changes the dynamics, so all s42 numbers before this fix are not strictly comparable with SFT. Final runs use the fix for every method.
- Added knobs: `--head_lr`, `--lbd_schedule constant|decay|linear`, `--h_pool last|mean`, `--h_layer`, `--pred_mlp`; `train/sweep_queue.sh` runner.

## Val results

| run | config | val /90 | end cos | takeaway |
|---|---|---|---|---|
| zero-shot | – | 25 | – | |
| sft (recipe sweep) | lr 5e-5, 2 ep | 27 | – | chosen recipe |
| sweep_jepa_l1 | LLM-JEPA λ1, GA fix | 23 | – | 4 below SFT (edge of noise) |
| sweep_pc_l1 | PreCode λ1, head lr = 5e-5 | 25 | 0.146 | head barely learns (cos 0.15); ≈ SFT within noise |
| sweep_pc_l1_h1e-3 | λ1, head_lr 1e-3 | 25 | 0.232 | cos up 0.15→0.23, val unchanged |
| sweep_pc_l1_h3e-3 | λ1, head_lr 3e-3 | 28 | 0.226 | +3 over h1e-3 with same cos → likely noise; needs 2nd seed |
| sweep_pc_l0.3_h1e-3 | λ0.3, head_lr 1e-3 | 28 | 0.222 | smaller λ, same cos; ties best |
| sweep_pc_l3_h1e-3 | λ3, head_lr 1e-3 | 25 | 0.255 | higher cos, no val gain |
| sweep_pcnce_l0.1_h1e-3 | precodence λ0.1 τ0.07, head_lr 1e-3 | 26 | 0.014 | InfoNCE barely moves own-z cos; ≈ noise |

**Batch 1 takeaway:** every PreCode/JEPA variant lands at 23–28/90, inside the ±4 noise band of SFT's 27; head cos saturates around 0.22–0.25
whatever head_lr is (1e-3–3e-3) and rises only slightly with λ3. 90 val tasks can't rank these configs, so I added a larger val set (next section).

## Extended val (`datasets/mbpp_valext.jsonl`, 276 tasks)
- MBPP ids 11–510 that are **not** in MBPP+ (EvalPlus dropped them). They're disjoint from training (ids 1–10, 511–974) and from the reported MBPP+ 224 test set (= MBPP+ ∩ 11–510).
  Same prompt format (`train/make_valext.py` asserts it reproduces an existing val prompt byte-for-byte). Selection-only, never reported as a result.
- Caveat: EvalPlus dropped some of these for flawed tests, so absolute scores are lower and noisier per task; only use it for paired comparisons between configs.
- From here on, `eval/val_mbpp.py` scores val (90) and valext (276) together and writes per-task results to `results/val_per_task/`. Selection uses **val+valext (366)**.

(1 valext task with an over-long prompt was dropped by a fixed length rule → 275 tasks, total 365.)

| run | config | val /90 | valext /275 | total /365 | end cos | takeaway |
|---|---|---|---|---|---|---|
| zero-shot | – | 25 | 55 | 80 | – | |
| sweep_sft_s42 | SFT, recipe | 27 | 58 | 85 | – | val reproduces 27 exactly (training is deterministic per seed) |
| sweep_sft_s7 | SFT, recipe, seed 7 | 26 | 59 | 85 | – | SFT seed spread is small (85 / 85) |
| sweep_pc_l1_h3e-3_s7 | PreCode λ1, head_lr 3e-3, seed 7 | 28 | 60 | 88 | 0.233 | +3 vs SFT s7 (85) |
| sweep_pc_l1_h3e-3 (rerun) | same, seed 42 | 26 | 56 | 82 | – | −3 vs SFT s42; same config gave 28/90 an hour earlier → PreCode training is not bit-deterministic (±2 val tasks from that alone). 2-seed mean 85 = SFT |
