# PreCode-JEPA — final results (HashCode 14.0, team 21Squad)

**Bottom line:** PreCode beats SFT on average on both benchmarks (+0.6 MBPP+, +1.2 HumanEval+; MBPP+ at 2 of 3 seeds, HumanEval+ at all 3), and on HumanEval+ the gain is one-directional
(PreCode solves 2 tasks SFT misses at every seed, never the reverse; pooled McNemar p = 0.031). PreCode does **not** beat LLM-JEPA
(−1.2 MBPP+, −0.6 HumanEval+, both within noise, p > 0.2), and **no fine-tuned method beats zero-shot**: the shared 474-task MBPP fine-tune costs
4–6 HumanEval+ points for every method.

## Final config (frozen on val, Oct 24 23:30 IST)
- Shared recipe (all methods): `unsloth/Llama-3.2-1B-Instruct`, LoRA r16, batch 16, lr 5e-5, 2 epochs (60 steps), max_len 512, train = `datasets/mbpp_train.jsonl` (474 MBPP tasks).
- **PreCode:** loss = NTP + 1.0 · (1 − cos(W·h, z)); h = hidden state at the last prompt token from **layer 8 of 16**; W = Linear(2048→768) with its own **lr 3e-3**; z = cached `features` behaviour embedding.
  `bash train/run_all.sh precode <seed>` with `LBD=1 HEAD_LR=3e-3 EXTRA="--h_layer 8"`.
- **LLM-JEPA:** `--additive_mask --predictors 1 --lbd 1.0` (unchanged). **SFT:** `--regular`.
- PreCode and JEPA losses are now divided by grad_accum like SFT (bug fix, see EXPERIMENTS.md §0). The old pre-fix s42 runs are kept as `jepa_s42_prefix` / `precode_s42_prefix`.
- Selection used only val (90 MBPP tasks) + valext (275 MBPP tasks outside MBPP+ and outside train). Test sets were touched only for these final runs.

## Test results (pass@1 on plus tests, greedy; mean ± std over seeds 42, 7, 1234)

| method | MBPP+ (224) | HumanEval+ (164) |
|---|---|---|
| zero-shot (`base_s0`) | **38.8** | **23.2** |
| SFT | 36.5 ± 0.7 | 17.1 ± 0.0 |
| LLM-JEPA | 38.2 ± 1.6 | 18.9 ± 0.0 |
| **PreCode** | 37.1 ± 0.5 | 18.3 ± 0.0 |

Per seed (MBPP+ / HumanEval+): SFT 36.6/17.1, 35.7/17.1, 37.0/17.1 · JEPA 36.6/18.9, 38.4/18.9, 39.7/18.9 · PreCode 37.5/18.3, 37.0/18.3, 36.6/18.3.
HumanEval+ counts are identical across seeds within each method (28 / 31 / 30 of 164). This was checked: samples and per-task results are regenerated per seed;
for JEPA and PreCode the solved sets differ by seed and only the counts match; SFT solves the same 28 tasks at every seed.

## Significance (exact McNemar, per-task pairs pooled over 3 seeds)
| comparison | benchmark | PreCode-only / other-only (pooled) | per seed 42, 7, 1234 | distinct tasks | p |
|---|---|---|---|---|---|
| PreCode vs SFT | HumanEval+ | 6 / 0 | 2/0, 2/0, 2/0 | 6 / 0 | **0.031** |
| PreCode vs SFT | MBPP+ | 18 / 14 | 6/4, 7/4, 5/6 | 12 / 11 | 0.60 |
| PreCode vs LLM-JEPA | HumanEval+ | 4 / 7 | 1/2, 1/2, 2/3 | 4 / 5 | 0.55 |
| PreCode vs LLM-JEPA | MBPP+ | 14 / 22 | 6/4, 4/7, 4/11 | 10 / 16 | 0.24 |

Caveat: pooling treats the 3 seeds as independent, but seed outcomes are correlated (same data, small LoRA). The HumanEval+ result vs SFT is consistent
in direction (never reversed at any seed, 6 distinct tasks: HumanEval/25, 52, 58, 60, 71, 156), but each single-seed test (2 vs 0) is not significant on its own.
Read it as "consistent small gain", not as a strong effect. Raw lines: `results/stats.md`.

## Attempts-to-green and paraphrase robustness
**Attempts-to-green** (MBPP+ 224, 5 samples at T=0.8, first passing attempt, 6 = none; 3-seed means; `results/attempts/`):

| method | mean attempts ↓ | solved @1 | @3 | @5 |
|---|---|---|---|---|
| zero-shot | **3.88** | **30.8%** | **45.1%** | **50.4%** |
| LLM-JEPA | 4.25 | 21.6% | 35.7% | 45.5% |
| PreCode | 4.32 | 20.4% | 34.4% | 43.5% |
| SFT | 4.33 | 20.1% | 34.4% | 43.6% |

No attempts-to-green advantage for PreCode (ties SFT). All fine-tuned models degrade much more than zero-shot under sampling (@1 at T=0.8 ≈ 20% vs ≈ 37% greedy).
Only 1 hero task met the demo rule (PreCode green on attempt 1, SFT ≥ 3, in ≥ 2 seeds): Mbpp/89 (`demo/ui/hero_tasks.json`).

**Paraphrase robustness** (60 MBPP+ tasks, original vs reworded prompts; 1 task = 1.7 pts; 3-seed means):

| method | original | paraphrased | drop |
|---|---|---|---|
| zero-shot | 36.7 | 36.7 | 0.0 |
| SFT | 27.2 | 32.2 | −5.0 |
| LLM-JEPA | 28.9 | 32.2 | −3.3 |
| PreCode | 28.3 | 30.0 | −1.7 |

No method loses accuracy on reworded prompts; the fine-tuned models actually do *better* on paraphrases, consistent with the fine-tune overfitting to the
literal MBPP prompt style. PreCode changes the least (−1.7), but at 1.7 pts per task these differences are 1–3 tasks and not meaningful.

## What worked, what didn't
- **Worked (small):** the behaviour-prediction loss is a cheap, mostly consistent improvement over plain SFT: higher on HumanEval+ at every seed and on MBPP+ at 2 of 3 seeds (+0.6 on average, not significant), and on
  HumanEval+ it fixes 2 tasks per seed that SFT gets wrong while never breaking one (p = 0.031 pooled). Training cost is the same as SFT and inference is unchanged.
  The behaviour target itself is meaningful: on 120 visible-test-passing mutants it separates correct from buggy code in 85% of pairs (cos 0.86 vs 0.948 for a code-text hash).
- **Didn't work:** PreCode does not beat LLM-JEPA. On val (365 tasks × 3 seeds) it led JEPA by about 6 tasks, but on test JEPA is ahead by 1–3 tasks, so the honest reading is
  "tie within noise". A 25-run val sweep (head lr, λ 0.3–3, λ decay, mean pooling, layer 8, MLP predictor, hybrid mpnet target, InfoNCE) never moved PreCode more than
  about 1 task above SFT: the head's cos plateaus at about 0.2, and the auxiliary gradient barely changes a 60-step LoRA. See `results/EXPERIMENTS.md`.
- **Biggest problem is shared, not PreCode-specific:** fine-tuning on 474 terse MBPP solutions makes every method worse than zero-shot (−4 to −6 HumanEval+,
  about −1 to −2 MBPP+, much worse under sampling). Our MBPP-only val couldn't see this damage. Fixing the shared recipe (less data drift, a HumanEval-style val signal)
  is the main lever for absolute accuracy, and should be applied to all methods equally.

## Notes
- Bug fixed during this round: PreCode and LLM-JEPA losses were not divided by grad_accum (gradients 4× SFT's, clipped every step). All final runs use the fix;
  pre-fix s42 numbers are kept as `*_prefix` rows (method `jepa_prefix` / `precode_prefix`) in `results/runs.csv`.
- "Meta reports 38% HumanEval for Llama-3.2-1B-Instruct" is unverified (the official model card lists no HumanEval/MBPP numbers); we don't use it.
- Models: private HF repo `gaganbuilds/precode-jepa-models` (one folder per run). Per-task results: `results/per_task/`. Table: `results/table.md`.
