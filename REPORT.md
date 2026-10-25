# PreCode-JEPA: final report

HashCode 14.0, Track 02 (LLM Training). Team 21Squad: Gagan P, Shryes, Nikhita U.

## Claim in one paragraph

Adding a training objective that makes a code model **predict what its program will do** (an embedding of the
reference solution's execution trace) from the task description, before writing any code, improves MBPP+ pass@1 of
Qwen2.5-Coder-1.5B from **55.0 to 60.0**: +5.3 points over an identically trained control with the objective switched
off (exact McNemar p < 0.0001, 5 seeds). The model genuinely learns behaviour: on 224 **unseen** test tasks its
predicted vector identifies the right task **5.4%** of the time (Recall@1; chance 0.45%, ~12x) with function names
removed from the target. On HumanEval+ the gain over SFT is small and not significant, and LLM-JEPA is best there.
Feeding the prediction back into generation as soft prefix tokens **hurt**; we report it as a negative result.

## Method

```
h      = final-layer hidden state at the last prompt token (prompt only)
ẑ      = normalize(MLP(h))                        MLP: 1536 -> 768 -> 768 (warmed up on frozen prompt states first)
z      = frozen behaviour vector of the task's reference solution
loss   = NTP(code tokens) + λ · InfoNCE(ẑ vs all 470 cached z, τ = 0.07)        λ = 0.5
```

- **Behaviour vector z** (`cache/behaviour.py`): the reference solution runs in a sandbox on ~17 probe inputs (the
  task's test inputs plus one-argument mutations: empty, negative, reversed, repeated…; exceptions are recorded as
  behaviour). Trace lines `f(args) -> output` are feature-hashed into 768 dimensions. **The function is renamed to `f`
  before encoding**: function names are near-unique per task (447 / 470) and visible in the prompt, so a name-bearing
  target could be "predicted" by reading the name (only ~half of a name-bearing vector survives renaming, cos 0.52).
- **Inference is unchanged**: code is generated exactly as with SFT. The head is only used for analysis and the demo.

## Setup

| | |
|---|---|
| Model | Qwen2.5-Coder-1.5B (base), LoRA r16 on all attention + MLP projections |
| Training data | 474 MBPP tasks outside the test split (ids 1–10, 511–974) |
| Hyper-parameters | chosen **per method on validation only**: lr {1e-5, 2e-5, 5e-5, 1e-4, 2e-4} × epochs {1, 2, 3}, trained on 384 tasks, scored on 365 held-out MBPP tasks (`SWEEP.md`) |
| Test sets (evaluated once, after selection) | MBPP+ (224 MBPP test-split tasks, EvalPlus tests) and HumanEval+ (164) |
| Decoding | greedy, 384 tokens, identical script for every method (`eval/generate_hf.py`) |
| Statistics | 5 seeds per method; exact McNemar on per-task pairs pooled over seeds; 95% bootstrap CI over tasks |
| Compute | Modal, L40S GPUs |

Selected settings: SFT lr 2e-5 / 2 ep · SFT control lr 2e-5 / 2 ep · LLM-JEPA lr 2e-5 / 3 ep · PreCode lr 5e-5 / 3 ep.

## Results (pass@1 plus, %, mean ± std over 5 seeds [95% CI])

| Method | MBPP+ | HumanEval+ |
|---|---|---|
| Zero-shot (base model) | 42.4 [36.2–49.1] | 1.8 [0.0–4.3] |
| SFT | 55.0 ± 1.5 [48.8–61.3] | 16.8 ± 0.7 [11.5–22.4] |
| SFT control (PreCode's training loop, λ = 0) | 54.7 ± 1.4 [48.5–61.1] | 16.2 ± 1.5 [11.1–21.7] |
| LLM-JEPA | 46.9 ± 0.8 [40.7–52.8] | **26.1 ± 2.6** [20.1–32.2] |
| **PreCode** (behaviour objective) | **60.0 ± 1.9** [53.9–65.8] | 18.0 ± 1.4 [12.6–23.8] |

### Paired tests (exact McNemar, tasks pooled over 5 seeds)

| A vs B | MBPP+ (A only / B only, p) | HumanEval+ (A only / B only, p) |
|---|---|---|
| SFT vs zero-shot | 288 / 147, p < 0.0001 | 129 / 6, p < 0.0001 |
| **PreCode vs SFT** | **101 / 45, p < 0.0001** | 48 / 38, p = 0.33 |
| **PreCode vs SFT control** | **107 / 48, p < 0.0001** | 51 / 36, p = 0.13 |
| SFT control vs SFT | 39 / 42, p = 0.82 | 11 / 16, p = 0.44 |
| PreCode vs LLM-JEPA | 191 / 44, p < 0.0001 | 46 / 112, p < 0.0001 (LLM-JEPA better) |
| LLM-JEPA vs SFT | 48 / 139, p < 0.0001 (SFT better) | 88 / 12, p < 0.0001 |

The SFT control ties SFT on both benchmarks, so the two training implementations are equivalent and PreCode's MBPP+
gain is attributable to the behaviour objective, not to the loop, schedule or loss averaging.

### Does the model learn behaviour? (224 unseen MBPP+ tasks, name-free vectors)

| Method | Recall@1 | Recall@5 | cos(pred, true) | behaviour match | exact behaviour |
|---|---|---|---|---|---|
| PreCode | **5.4 ± 0.5%** | **23.0 ± 2.6%** | 0.176 | **69.0%** | **59.8%** |
| SFT control (untrained head) | 0.5% | 2.0% | 0.010 | 63.9% | 54.6% |
| SFT | — | — | — | 65.5% | 55.1% |
| LLM-JEPA | — | — | — | 55.9% | 46.9% |
| Zero-shot | — | — | — | 49.2% | 42.0% |
| *chance* | *0.45%* | *2.25%* | | | |

Behaviour match = share of probe inputs where the generated code's output equals the canonical solution's (partial
credit). The predicted vector is only weakly better on solved than unsolved tasks (cos 0.180 vs 0.171).

## Negative and secondary results

- **Prefix conditioning hurts.** Turning the predicted vector into 4 soft tokens that the model reads while writing
  lowered MBPP+ from 58.2 to 45.4 (p < 0.0001, 3 seeds, earlier run); adding hard-task loss weighting did not recover
  it. Same finding on Llama-3.2-1B. Using the behaviour prediction as a training signal helps; feeding it back into
  generation does not.
- **LLM-JEPA is hyper-parameter sensitive.** With lr 2e-5 / 1 ep (earlier run) it scored 59.2 MBPP+ / 11.4 HumanEval+;
  with the validation-selected 2e-5 / 3 ep it scored 46.9 / 26.1. PreCode was stable across both runs
  (58.2 → 60.0 MBPP+, 19.7 → 18.0 HumanEval+).
- **The original cosine / single-linear-head PreCode was unstable** on Qwen (32.0 ± 4.8 MBPP+; InfoNCE variant
  43.8 ± 10.2). The working version adds a head warm-up on frozen prompt states and a small MLP head.
- **The behaviour objective acts as a regulariser**: PreCode trains stably at lr 1e-4 (val 44.4) where SFT collapses
  (val 1.6).
- **Replication on Llama-3.2-1B-Instruct** (earlier run, name-bearing target, 3 seeds): PreCode 48.8 vs SFT control
  45.4 on MBPP+ (p = 0.0095); HumanEval+ tie. There every fine-tuned model stayed below zero-shot (instruct model).

## Limitations

- The pass@1 gain is significant on MBPP+ (same distribution as training) but not on HumanEval+.
- 474 training tasks and a 1.5B model; larger data and models are untested.
- Behaviour vectors are feature-hashed traces on ~17 probes; richer probes or learned encoders may do better.
- Recall@1 of 5.4% is far above chance but low in absolute terms; the prediction is coarse.
- Hyper-parameters were selected on MBPP validation, which matches MBPP+ better than HumanEval+.

## Files

| Path | What |
|---|---|
| `results/qc15/COMPARISON.md`, `SWEEP.md` | full tables, all paired tests, the validation sweep |
| `results/qc15/rows/`, `per_task/`, `metrics/` | per-run, per-task and behaviour-metric files |
| `results/qc15/tsne_*.png` | predicted vs true behaviour on test tasks |
| `train/precode_full.py` | PreCode trainer (aux / prefix / full modes) |
| `train/finetune.py`, `train/run_all.sh` | SFT and LLM-JEPA (and original PreCode) trainer |
| `cache/behaviour.py`, `cache/behaviour_cache_anon.pt` | probes, sandbox, encoder; name-free behaviour cache |
| `eval/generate_hf.py`, `eval/analyze.py`, `eval/compare.py` | generation, behaviour metrics, comparison |
| `modal_app.py`, `MODAL.md` | the whole pipeline on Modal |
| `demo/studio/` | live side-by-side demo |

Reproduce: `python -m modal run modal_app.py --methods base,sft,jepa,sft_loop,precodefull_aux --seeds 42,7,1234,11,99`.
