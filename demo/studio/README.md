# PreCode Studio

Live side-by-side demo: one prompt, every method, then proof of what each program actually does.

```bash
pip install fastapi uvicorn scikit-learn transformers torch
# from the repo root
MOCK=1 uvicorn demo.studio.server:app --port 8010          # UI only, no GPU or checkpoints
uvicorn demo.studio.server:app --host 0.0.0.0 --port 8010  # real models from /data/precode/models/<run>_s42
```
Open http://localhost:8010. Point the page at another server with `?api=http://host:8010`.

## What the page shows
| Panel | Where it comes from |
|---|---|
| 4 streamed columns (zero-shot, SFT, LLM-JEPA, PreCode; PreCode-NCE if its dir exists) | `/api/generate` (SSE), greedy, prompt format = contract C13 |
| "What PreCode imagined" | `pred_head.pt` on the last-prompt-token hidden state, nearest cached tasks + their traces |
| Pass/fail per check | your asserts, run in a sandboxed subprocess |
| Probe grid | the same probe builder as the cache (`cache/behaviour.py: make_probes`); free text with no asserts falls back to generic inputs matched to the function's arity |
| Imagined vs actual | `encode_features(trace of generated code)` vs `pred_head` output |
| Behaviour map | t-SNE of `behaviour_cache.pt` (cached in `map_cache.json`); new points placed by nearest-neighbour weighting |
| Results tab | `results/runs.csv`, `results/per_task/*` (exact McNemar), `results/plots/mutant_numbers_features.json` |

## Env
`MODELS_DIR` (/data/precode/models), `DEMO_SEED` (42), `RUNS` (base,sft,jepa,precode), `BASE_MODEL`,
`GPU_MODE` (auto | all | swap), `H_POOL` (last | mean; must match training), `MAX_NEW` (384), `MOCK`.

GPU memory: each merged 1B model is ~2.5 GB in fp16. `auto` keeps all of them on the GPU only if they fit;
otherwise `swap` keeps them in CPU RAM (~10 GB) and moves one at a time to the GPU (adds ~1 s per method).
Cards without bf16 (RTX 20-series) load in fp16 automatically.
