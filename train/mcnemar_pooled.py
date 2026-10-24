"""python train/mcnemar_pooled.py precode sft mbppplus224  (exact McNemar, pooled over seeds)"""
import json, os, sys
from scipy.stats import binomtest
A, B, ds = sys.argv[1:4]
n01 = n10 = 0; used = []
def find(m, s):
    for d in ("results/per_task", "fallback/per_task"):
        p = f"{d}/{m}_s{s}_{ds}.json"
        if os.path.exists(p):
            return p
for s in (42, 7, 1234):
    pa, pb = find(A, s), find(B, s)
    if not (pa and pb):
        continue
    a, b = json.load(open(pa)), json.load(open(pb)); used.append(s)
    for k in set(a) & set(b):
        n01 += a[k] and not b[k]; n10 += b[k] and not a[k]
p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
print(f"{A} vs {B} on {ds}, seeds {used}: {A} only {n01}, {B} only {n10}, exact McNemar p = {p:.4f}")
