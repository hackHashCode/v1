"""Name-free behaviour cache from the existing traces (no re-execution, deterministic).
python cache/anonymize_cache.py  ->  cache/behaviour_cache_anon.pt (+ _stats.json)
Same probes and outputs as cache/behaviour_cache.pt, encoded with the function renamed to f (see behaviour.anonymize)."""
import json, sys
sys.path.insert(0, "cache")
import torch
from behaviour import encode_features

traces = [json.loads(l) for l in open("cache/traces.jsonl")]
cache = {int(t["task_id"]): encode_features(t["trace"], anonymize_names=True) for t in traces}
assert not any(11 <= k <= 510 for k in cache), "test-split leak"
torch.save(cache, "cache/behaviour_cache_anon.pt")
Z = torch.stack(list(cache.values())); S = Z @ Z.T; n = len(Z)
off = ((S.sum() - S.diag().sum()) / (n * (n - 1))).item()
old = torch.load("cache/behaviour_cache.pt")
same = sum(float(old[k].float() @ cache[k]) for k in cache) / len(cache)
stats = {"encoder": "features", "anonymized": True, "cached": n, "mean_offdiag_cos": round(off, 4),
         "mean_cos_to_named_cache": round(same, 4)}
json.dump(stats, open("cache/behaviour_cache_anon_stats.json", "w"), indent=1)
print(stats)
