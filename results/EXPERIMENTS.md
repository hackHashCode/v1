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
| sweep_pc_l1_h3e-3_decay | + λ decays to 0 over last epoch | 26 | 58 | 84 | 0.227 | ≈ SFT; decay neither helps nor hurts |
| sweep_pc_l1_h3e-3_mean | + h = mean over prompt tokens | 26 | 62 | 88 | 0.197 | +3 vs SFT s42, best valext so far; seed-7 check queued |
| sweep_pc_l1_h3e-3_L8 | + h from layer 8 of 16 | 26 | 64 | 90 | 0.192 | +5 vs SFT s42, best so far; seed-7 check queued |
| sweep_pc_l1_h3e-3_mlp | + 2-layer MLP predictor | 25 | 61 | 86 | – | ≈ SFT (+1) |
| sweep_pc_l1_h3e-3_L8_s7 | layer 8, seed 7 | 25 | 60 | 85 | – | = SFT s7. Layer-8 2-seed mean 87.5 vs SFT 85 → +2.5, within noise |
| sweep_pc_l1_h1e-2 | λ1, head_lr 1e-2 | 28 | 59 | 87 | 0.213 | +2 vs SFT s42; higher head lr does not raise cos further |
| sweep_pc_l1_h3e-3_mean_s7 | mean pool, seed 7 | 25 | 60 | 85 | 0.185 | = SFT s7. Mean-pool 2-seed mean 86.5 vs SFT 85 → noise |
| sweep_jepa_l1 | LLM-JEPA λ1, seed 42 | 22 | 57 | 79 | – | −6 vs SFT, below zero-shot (80) |
| sweep_jepa_l1_s7 | LLM-JEPA λ1, seed 7 | 24 | 57 | 81 | – | JEPA 2-seed mean 80 vs SFT 85 vs PreCode variants 82–90 |
| sweep_pc_l1_h3e-3_hyb | hybrid cache (0.5 mpnet + 0.5 features) | 26 | 58 | 84 | – | ≈ SFT; hybrid target no better than features |
| sweep_pc_l1_h3e-3_hyb_s7 | hybrid cache, seed 7 | 25 | 57 | 82 | – | −3 vs SFT s7. Hybrid 2-seed mean 83 → drop hybrid |
| sweep_pc_l3_h3e-3_L8 | λ3, layer 8 | 26 | 60 | 86 | – | +1 vs SFT s42; λ3 no better than λ1 at layer 8 (90) |
| sweep_pc_l0.3_h3e-3 | λ0.3, head_lr 3e-3 | 24 | 60 | 84 | – | ≈ SFT |
| sweep_pcnce_l0.1_h3e-3_t0.1 | precodence λ0.1, τ0.1, head_lr 3e-3 | 26 | 60 | 86 | – | +1 vs SFT; InfoNCE ≈ cosine |
| sweep_sft_s1234 | SFT, seed 1234 | 25 | 63 | 88 | – | SFT 3-seed: 85 / 85 / 88 → mean 86.0 |
| sweep_pc_l1_h3e-3_L8_s1234 | layer 8, seed 1234 | 25 | 61 | 86 | – | Layer-8 3-seed: 90 / 85 / 86 → mean 87.0 (+1.0 vs SFT) |
| sweep_pc_l1_h3e-3_mean_s1234 | mean pool, seed 1234 | 27 | 58 | 85 | – | Mean-pool 3-seed: 88 / 85 / 85 → mean 86.0 (= SFT) |
| sweep_jepa_l1_s1234 | LLM-JEPA λ1, seed 1234 | 25 | 59 | 84 | – | JEPA 3-seed: 79 / 81 / 84 → mean 81.3 |
| sweep_pc_l1_h3e-3_L8mean | layer 8 + mean pool | 26 | 60 | 86 | – | combining the two doesn't add up |

## Freeze decision (Oct 24 23:30 IST)
3-seed val+valext means (/365): **PreCode λ1 + head_lr 3e-3 + layer 8: 87.0** · mean-pool variant 86.0 · SFT 86.0 · LLM-JEPA 81.3 · zero-shot 80.
- Frozen PreCode config: `LBD=1 HEAD_LR=3e-3 EXTRA="--h_layer 8"` (cosine loss, `features` cache, linear head, constant λ), shared recipe unchanged.
- Expectation going into test: PreCode ≈ SFT (+1 task on 365 is noise), PreCode > LLM-JEPA by about 6 tasks (consistent across all 3 seeds).
- Nothing in the sweep moved PreCode clearly above SFT. Head cos plateaus at about 0.2 regardless of head_lr, λ, pooling, layer, predictor or target encoder.
  The auxiliary loss is learnable but barely changes what the LoRA learns in 48–60 steps.

## What didn't work (val+valext, vs SFT 86.0)
- head_lr 1e-3 / 3e-3 / 1e-2: cos 0.15 → ~0.22, then it plateaus; no val gain.
- λ 0.3 / 1 / 3: no trend.
- λ decay over the last epoch: 84.
- 2-layer MLP predictor: 86.
- hybrid (mpnet + features) behaviour target: 2-seed 83.
- InfoNCE (precodence) τ 0.07 / 0.1: 26/90 and 86/365, ≈ SFT.
- Note: a PreCode config is not bit-reproducible (same seed: 28 then 26/90), so ±2 val tasks per run is pure training noise.
- Note: `results/per_task/sweep_lr2e-4_e4_mbppplus224.json` (a sweep scored on test) predates this session and was not used for any selection.

## Final test runs (Oct 24 23:30 – Oct 25 00:50 IST)
Frozen config, full train set, seeds 42/7/1234. Test pass@1 (plus) means: zero-shot 38.8 / 23.2 · SFT 36.5 / 17.1 · LLM-JEPA 38.2 / 18.9 · PreCode 37.1 / 18.3 (MBPP+ / HumanEval+).
- Val (365 tasks) predicted PreCode > JEPA by about 6 tasks; test has JEPA ahead by 1–3 tasks. Both gaps are inside noise, so val and test agree only on "PreCode ≈ JEPA, ≥ SFT".
- PreCode vs SFT on HumanEval+: 2 PreCode-only tasks at every seed, 0 the other way (pooled McNemar p = 0.031; 6 distinct tasks).
- Attempts-to-green: no PreCode advantage (4.32 vs SFT 4.33, JEPA 4.25, zero-shot 3.88). Paraphrase: fine-tuned models score higher on reworded prompts; no meaningful method differences.
- Infra notes: running 3 vLLM jobs on one GPU makes engine startup fail its memory check whenever another job frees memory; `train/fast_post.sh` retries with random back-off (≤ 8 tries, most passed by try 3).
- Post-hoc ideas, **not run** (test already seen, so any result would be post-hoc): visible-test reranking of the 5 samples (upper bound = solved@5: PreCode 43.5%,
  zero-shot 50.4%, so it lifts every method), WiSE-FT-style interpolation back toward base to recover HumanEval+, and RL with a unit-test reward (doesn't fit the GPU budget and would need all methods RL'd for a fair claim).
