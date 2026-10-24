"""Build the behaviour cache (contracts C2, C3).
python cache/build_cache.py [--encoder features|mpnet|hybrid] [--out cache/behaviour_cache.pt]
Default encoder `features` needs no downloads. On the 4090 (HF reachable) you can build `hybrid`."""
import argparse, json, sys, time
sys.path.insert(0, "cache")
import torch
from behaviour import make_probes, trace, encode, load_mbpp_train

ap = argparse.ArgumentParser()
ap.add_argument("--encoder", default="features", choices=["features", "mpnet", "hybrid"])
ap.add_argument("--out", default="cache/behaviour_cache.pt")
ap.add_argument("--traces", default="cache/traces.jsonl")
a = ap.parse_args()

t0 = time.time()
cache, traces, failed = {}, [], []
for ex in load_mbpp_train():
    tr = trace(ex["code"], ex.get("test_setup_code", ""), make_probes(ex["test_list"]))
    if not tr:
        failed.append(ex["task_id"]); continue
    cache[int(ex["task_id"])] = encode(tr, a.encoder)
    traces.append({"task_id": int(ex["task_id"]), "trace": tr})
assert not any(11 <= k <= 510 for k in cache), "test-split leak"
torch.save(cache, a.out)
with open(a.traces, "w") as f:
    for t in traces: f.write(json.dumps(t) + "\n")
Z = torch.stack(list(cache.values())); S = Z @ Z.T; n = len(Z)
off = ((S.sum() - S.diag().sum()) / (n * (n - 1))).item()
stats = {"encoder": a.encoder, "cached": n, "failed": failed,
         "mean_probes": round(sum(len(t["trace"]) for t in traces) / len(traces), 2),
         "mean_offdiag_cos": round(off, 4), "seconds": round(time.time() - t0)}
json.dump(stats, open(a.out.replace(".pt", "_stats.json"), "w"), indent=1)
print(stats)
