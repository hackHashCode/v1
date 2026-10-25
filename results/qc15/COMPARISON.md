# PreCode-JEPA: method comparison

pass@1 in %, greedy decoding, mean ± std over seeds (n) [95% bootstrap CI over tasks]. MBPP+ = 224 test-split tasks, HumanEval+ = 164.
"plus" = passes EvalPlus's extended tests too (the headline number).

| method | humanevalplus plus | humanevalplus base | mbppplus224 plus | mbppplus224 base |
|---|---|---|---|---|
| Zero-shot | 1.8 ± 0.0 (1) [0.0–4.3] | 1.8 ± 0.0 (1) | 42.4 ± 0.0 (1) [36.2–49.1] | 49.5 ± 0.0 (1) |
| SFT | 16.8 ± 0.7 (5) [11.5–22.4] | 19.8 ± 1.0 (5) | 55.0 ± 1.5 (5) [48.8–61.3] | 63.8 ± 1.9 (5) |
| LLM-JEPA | **26.1 ± 2.6 (5) [20.1–32.2]** | 31.8 ± 2.8 (5) | 46.9 ± 0.8 (5) [40.7–52.8] | 56.2 ± 1.1 (5) |
| PreCode (cosine) | 7.1 ± 3.4 (3) [4.1–10.6] | 9.3 ± 3.6 (3) | 32.0 ± 4.8 (3) [26.8–37.2] | 38.7 ± 4.3 (3) |
| PreCode (InfoNCE) | 14.8 ± 13.1 (3) [11.4–18.5] | 18.3 ± 15.8 (3) | 43.8 ± 10.2 (3) [38.2–49.1] | 52.8 ± 12.7 (3) |
| SFT (PreCode-full loop, λ=0) | 16.2 ± 1.5 (5) [11.1–21.7] | 18.4 ± 1.5 (5) | 54.7 ± 1.4 (5) [48.5–61.1] | 63.7 ± 1.4 (5) |
| PreCode-full: aux only | 18.0 ± 1.4 (5) [12.6–23.8] | 20.4 ± 2.5 (5) | **60.0 ± 1.9 (5) [53.9–65.8]** | 68.2 ± 1.7 (5) |
| PreCode-full: + prefix | 15.6 ± 3.9 (3) [11.8–19.9] | 18.5 ± 5.2 (3) | 45.4 ± 2.5 (3) [40.0–50.6] | 53.4 ± 2.7 (3) |
| PreCode-full: + prefix + weighting | 16.7 ± 2.8 (3) [12.2–21.3] | 18.9 ± 2.4 (3) | 46.7 ± 2.3 (3) [41.2–52.1] | 54.2 ± 1.7 (3) |

## Paired tests (exact McNemar, tasks pooled over seeds)

"A only" = tasks A solves and B doesn't. A gap only counts if p < 0.05 **and** the sign agrees across seeds.

| A | B | dataset | A only | B only | both | p | seeds |
|---|---|---|---|---|---|---|---|
| Zero-shot | SFT | humanevalplus | 6 | 129 | 9 | 0.0 ✱ | 7,11,42,99,1234 |
| Zero-shot | SFT | mbppplus224 | 147 | 288 | 328 | 0.0 ✱ | 7,11,42,99,1234 |
| LLM-JEPA | SFT | humanevalplus | 88 | 12 | 126 | 0.0 ✱ | 7,11,42,99,1234 |
| LLM-JEPA | SFT | mbppplus224 | 48 | 139 | 477 | 0.0 ✱ | 7,11,42,99,1234 |
| PreCode (cosine) | SFT | humanevalplus | 5 | 55 | 30 | 0.0 ✱ | 7,42,1234 |
| PreCode (cosine) | SFT | mbppplus224 | 14 | 162 | 201 | 0.0 ✱ | 7,42,1234 |
| PreCode (InfoNCE) | SFT | humanevalplus | 38 | 50 | 35 | 0.2408 | 7,42,1234 |
| PreCode (InfoNCE) | SFT | mbppplus224 | 46 | 115 | 248 | 0.0 ✱ | 7,42,1234 |
| SFT (PreCode-full loop, λ=0) | SFT | humanevalplus | 11 | 16 | 122 | 0.4421 | 7,11,42,99,1234 |
| SFT (PreCode-full loop, λ=0) | SFT | mbppplus224 | 39 | 42 | 574 | 0.8243 | 7,11,42,99,1234 |
| PreCode-full: aux only | SFT | humanevalplus | 48 | 38 | 100 | 0.3318 | 7,11,42,99,1234 |
| PreCode-full: aux only | SFT | mbppplus224 | 101 | 45 | 571 | 0.0 ✱ | 7,11,42,99,1234 |
| PreCode-full: + prefix | SFT | humanevalplus | 38 | 46 | 39 | 0.4452 | 7,42,1234 |
| PreCode-full: + prefix | SFT | mbppplus224 | 46 | 104 | 259 | 0.0 ✱ | 7,42,1234 |
| PreCode-full: + prefix + weighting | SFT | humanevalplus | 35 | 38 | 47 | 0.8151 | 7,42,1234 |
| PreCode-full: + prefix + weighting | SFT | mbppplus224 | 51 | 100 | 263 | 0.0001 ✱ | 7,42,1234 |
| PreCode-full: aux only | SFT (PreCode-full loop, λ=0) | humanevalplus | 51 | 36 | 97 | 0.1329 | 7,11,42,99,1234 |
| PreCode-full: aux only | SFT (PreCode-full loop, λ=0) | mbppplus224 | 107 | 48 | 565 | 0.0 ✱ | 7,11,42,99,1234 |
| PreCode-full: + prefix + weighting | SFT (PreCode-full loop, λ=0) | humanevalplus | 38 | 33 | 44 | 0.6353 | 7,42,1234 |
| PreCode-full: + prefix + weighting | SFT (PreCode-full loop, λ=0) | mbppplus224 | 51 | 105 | 263 | 0.0 ✱ | 7,42,1234 |
| PreCode-full: aux only | LLM-JEPA | humanevalplus | 46 | 112 | 102 | 0.0 ✱ | 7,11,42,99,1234 |
| PreCode-full: aux only | LLM-JEPA | mbppplus224 | 191 | 44 | 481 | 0.0 ✱ | 7,11,42,99,1234 |
| PreCode-full: + prefix + weighting | LLM-JEPA | humanevalplus | 26 | 74 | 56 | 0.0 ✱ | 7,42,1234 |
| PreCode-full: + prefix + weighting | LLM-JEPA | mbppplus224 | 83 | 81 | 231 | 0.9378 | 7,42,1234 |
| PreCode (cosine) | LLM-JEPA | humanevalplus | 0 | 95 | 35 | 0.0 ✱ | 7,42,1234 |
| PreCode (cosine) | LLM-JEPA | mbppplus224 | 32 | 129 | 183 | 0.0 ✱ | 7,42,1234 |
| PreCode (InfoNCE) | PreCode (cosine) | humanevalplus | 63 | 25 | 10 | 0.0001 ✱ | 7,42,1234 |
| PreCode (InfoNCE) | PreCode (cosine) | mbppplus224 | 124 | 45 | 170 | 0.0 ✱ | 7,42,1234 |
| PreCode-full: aux only | PreCode (InfoNCE) | humanevalplus | 56 | 42 | 31 | 0.1888 | 7,42,1234 |
| PreCode-full: aux only | PreCode (InfoNCE) | mbppplus224 | 137 | 27 | 267 | 0.0 ✱ | 7,42,1234 |
| PreCode-full: + prefix | PreCode-full: aux only | humanevalplus | 31 | 41 | 46 | 0.2888 | 7,42,1234 |
| PreCode-full: + prefix | PreCode-full: aux only | mbppplus224 | 17 | 116 | 288 | 0.0 ✱ | 7,42,1234 |
| PreCode-full: + prefix + weighting | PreCode-full: + prefix | humanevalplus | 23 | 18 | 59 | 0.5327 | 7,42,1234 |
| PreCode-full: + prefix + weighting | PreCode-full: + prefix | mbppplus224 | 56 | 47 | 258 | 0.4307 | 7,42,1234 |

## Behaviour prediction at the end of training (PreCode-full runs)

cos_own = cos(predicted z, own task's z); top1 = fraction where own z is the nearest of all cached tasks (training tasks, so this measures fit, not generalisation).

| run | cos_own | top1 | NTP | L_beh | steps |
|---|---|---|---|---|---|
| precodefull_aux_s11 | 0.456 | 0.928 | 0.799 | 1.476 | 90 |
| precodefull_aux_s1234 | 0.443 | 0.929 | 0.796 | 1.572 | 90 |
| precodefull_aux_s42 | 0.461 | 0.955 | 0.805 | 1.389 | 90 |
| precodefull_aux_s7 | 0.454 | 0.912 | 0.817 | 1.502 | 90 |
| precodefull_aux_s99 | 0.449 | 0.909 | 0.814 | 1.516 | 90 |
| precodefull_prefix_s1234 | 0.394 | 0.797 | 0.907 | 2.126 | 60 |
| precodefull_prefix_s42 | 0.415 | 0.857 | 0.845 | 1.799 | 60 |
| precodefull_prefix_s7 | 0.399 | 0.783 | 0.83 | 2.078 | 60 |
| precodefull_s1234 | 0.395 | 0.802 | 0.907 | 2.12 | 60 |
| precodefull_s42 | 0.415 | 0.863 | 0.852 | 1.802 | 60 |
| precodefull_s7 | 0.399 | 0.78 | 0.839 | 2.077 | 60 |
| sft_loop_s11 | 0.009 | 0.002 | 0.801 | 6.29 | 60 |
| sft_loop_s1234 | 0.02 | 0.002 | 0.815 | 6.279 | 60 |
| sft_loop_s42 | 0.004 | 0.002 | 0.801 | 6.266 | 60 |
| sft_loop_s7 | -0.003 | 0.003 | 0.798 | 6.268 | 60 |
| sft_loop_s99 | 0.011 | 0.002 | 0.811 | 6.251 | 60 |

## Behaviour metrics on the 224 MBPP+ test tasks (eval/analyze.py)

behaviour match = % of probe inputs where the generated code's output equals the canonical solution's (partial credit). exact = % of tasks matching on every probe. Recall@k / cos only exist for methods with a behaviour head; test-task behaviour vectors are built for this analysis only, training never sees them.

| method | behaviour match % | exact behaviour % | recall@1 % | recall@5 % | cos(pred, true) | cos(pred, own code) | cos_own solved / unsolved |
|---|---|---|---|---|---|---|---|
| Zero-shot | 49.2 ± 0.0 | 42.0 ± 0.0 | — | — | — | — | — |
| SFT | 65.5 ± 1.4 | 55.1 ± 1.3 | — | — | — | — | — |
| LLM-JEPA | 55.9 ± 1.0 | 46.9 ± 0.7 | — | — | — | — | — |
| PreCode (cosine) † | 38.4 ± 4.2 | 32.1 ± 4.4 | 0.7 ± 0.5 | 2.4 ± 0.9 | 0.070 ± 0.020 | 0.069 ± 0.019 | 0.072 / 0.068 |
| PreCode (InfoNCE) † | 53.1 ± 12.7 | 43.2 ± 10.4 | 1.3 ± 0.4 | 5.1 ± 1.8 | 0.016 ± 0.006 | 0.014 ± 0.005 | 0.016 / 0.016 |
| SFT (PreCode-full loop, λ=0) | 63.9 ± 1.4 | 54.6 ± 1.6 | 0.5 ± 0.2 | 2.0 ± 0.2 | 0.010 ± 0.008 | 0.009 ± 0.008 | 0.010 / 0.010 |
| PreCode-full: aux only | 69.0 ± 1.2 | 59.8 ± 1.5 | 5.4 ± 0.5 | 23.0 ± 2.6 | 0.176 ± 0.004 | 0.173 ± 0.006 | 0.180 / 0.171 |
| PreCode-full: + prefix † | 54.4 ± 3.5 | 45.7 ± 3.0 | 4.6 ± 1.7 | 22.6 ± 1.6 | 0.179 ± 0.006 | 0.171 ± 0.006 | 0.183 / 0.175 |
| PreCode-full: + prefix + weighting † | 55.7 ± 1.5 | 47.3 ± 1.9 | 4.3 ± 1.6 | 21.6 ± 1.4 | 0.178 ± 0.006 | 0.173 ± 0.012 | 0.185 / 0.174 |

† measured against name-bearing test vectors (older analysis); recall/cos for these rows can be inflated by reading the function name.

Random recall@1 = 0.45%, recall@5 = 2.25%.
Figures: tsne_precode_s42.png, tsne_precodefull_aux_s42.png, tsne_precodefull_prefix_s42.png, tsne_precodefull_s42.png, tsne_precodence_s42.png
