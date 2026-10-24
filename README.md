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
| `cache/` | Shryes (built by Gagan) | `behaviour.py` (probes, sandbox, encoders), `build_cache.py`, `behaviour_cache.pt` (470 tasks), `traces.jsonl`, `mutant_test.py` |
| `eval/` | Nikhita (+ Gagan) | `generate.py`, `score.py`, `run_eval.sh`, `sanity_fake.sh` (Nikhita); `attempts.py`, `mcnemar.py`, `watch.sh`, `run_paraphrase.sh`, `pick_heroes.py` |
| `eval/paraphrase/` | Gagan | 60 paraphrased MBPP+ prompts (`mbppplus_paraphrased.jsonl`) |
| `demo/` | Gagan | `server.py` (serves UI + `/solve`), `ui/index.html`, `specs.json` (10 judge specs, verified) |
| `fallback/` | Gagan | minimal versions of every track, used only if a track is blocked |
| `tests/` | Gagan | CPU unit tests |

Interfaces between tracks: see `CONTRACTS.md`.

## Behaviour target (z)

Each training task's reference solution runs in a subprocess on ~17 probes (test inputs + edge cases;
1 s per probe; run twice, keep deterministic lines). The trace lines `f(args) -> repr(output)` are encoded to
768-d with `--encoder`:

- `features` (default, no downloads): signed feature hashing of exact lines, output values, output types and
  edge-input behaviour (e.g. `empty -> IndexError`), sqrt-damped.
- `hybrid` (needs Hugging Face): `normalize(0.5 * all-mpnet-base-v2 + 0.5 * features)`.

**Mutant test** (`cache/mutant_test.py`, `features`): 120 automatic bugs that still pass the visible test but
change behaviour. Cosine(correct, buggy) = **0.86** for our behaviour vector vs **0.95** for a code-text
embedding; ours separates the bug more in **85%** of pairs. Cross-task mean cosine 0.12.

## Reproduce

```bash
python train/build_mbpp.py                                   # datasets/mbpp_train.jsonl
python cache/build_cache.py [--encoder features|hybrid]     # cache/behaviour_cache.pt
python tests/check_cache.py                                  # 4 handoff checks
python cache/mutant_test.py                                  # results/plots/mutant_separation_features.png
for S in 42 7 1234; do for M in sft jepa precode; do bash train/run_all.sh $M $S; done; done
bash eval/run_eval.sh precode_s42                            # MBPP+ (224 test-split tasks) and HumanEval+
python eval/attempts.py precode_s42 && bash eval/run_paraphrase.sh precode_s42
python eval/mcnemar.py precode sft mbppplus224
python train/make_table.py
```

Eval: MBPP+ v0.2.0 filtered to task_ids 11-510 (224 tasks; 154 MBPP+ tasks overlap MBPP's non-test splits and are
dropped) and HumanEval+ (164). 3 seeds, greedy decoding, exact paired McNemar test.

Tests: `python tests/test_precode_trainer.py` (CPU, no downloads).
