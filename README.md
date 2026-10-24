# PreCode-JEPA

Execution-derived behaviour representations for code generation, trained with an LLM-JEPA-style objective.
HashCode 14.0, Track 02 (LLM Training). Team 21Squad.

**Idea.** LLM-JEPA adds a loss that makes an LLM's embedding of a task description predict the embedding of the
reference code *text*. PreCode-JEPA keeps the objective but swaps the target: an embedding of what the reference
code *does* when executed on ~20 probe inputs (exceptions included). Training costs the same as SFT (the prompt
embedding comes from the SFT forward pass; the target is precomputed). Inference is unchanged.

```
loss = NTP + lambda * (1 - cos(W h_prompt, z_behaviour))      W: linear 2048 -> 768, z frozen
# ablation `precodence`: InfoNCE of W h_prompt against ALL cached z (own task = positive), tau = 0.07
```

## Layout (one owner per folder)

| Path | Owner | What |
|---|---|---|
| `train/` | Gagan | `finetune.py` (forked from [llm-jepa](https://github.com/rbalestr-lab/llm-jepa), + `PreCodeTrainer`), `build_mbpp.py`, `run_all.sh`, `make_table.py`, `mcnemar_pooled.py` |
| `datasets/` | Gagan | `mbpp_train.jsonl` (474 non-test MBPP tasks) |
| `cache/` | Shryes | sandbox, probes, behaviour cache, mutant test, Recall@5 |
| `eval/` | Nikhita | generation (vLLM), EvalPlus scoring, attempts-to-green, McNemar |
| `eval/paraphrase/` | Shryes | 60 paraphrased MBPP+ prompts |
| `demo/server.py` / `demo/ui/` | Gagan / Nikhita | live "attempts to green" demo |
| `fallback/` | Gagan | minimal versions of every track, used only if a track is blocked |
| `tests/` | Gagan | CPU unit tests |

Interfaces between tracks: see `CONTRACTS.md`.

## Reproduce

```bash
python train/build_mbpp.py                                   # datasets/mbpp_train.jsonl
python cache/build_cache.py                                  # cache/behaviour_cache.pt
for S in 42 7 1234; do for M in sft jepa precode; do bash train/run_all.sh $M $S; done; done
bash eval/run_eval.sh precode_s42                            # MBPP+ (224 test-split tasks) and HumanEval+
python train/make_table.py
```

Eval: MBPP+ v0.2.0 filtered to task_ids 11-510 (224 tasks; 154 MBPP+ tasks overlap MBPP's non-test splits and are
dropped) and HumanEval+ (164). 3 seeds, greedy decoding, exact paired McNemar test.

Tests: `python tests/test_precode_trainer.py` (CPU, no downloads).
