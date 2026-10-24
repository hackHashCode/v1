"""Verify a behaviour cache before handing it to training (contract C2).
python tests/check_cache.py [cache/behaviour_cache.pt]"""
import json, sys
import torch
path = sys.argv[1] if len(sys.argv) > 1 else "cache/behaviour_cache.pt"
cache = torch.load(path)
print("1. loaded OK:", path, "entries:", len(cache))
v = next(iter(cache.values()))
assert v.shape == (768,), v.shape; print("2. shape:", tuple(v.shape))
assert all(isinstance(k, int) for k in cache), "keys must be int"; print("3. keys: int")
ids = {json.loads(l)["task_id"] for l in open("datasets/mbpp_train.jsonl")}
missing = sorted(ids - set(cache)); extra = sorted(set(cache) - ids)
print(f"4. missing {len(missing)} of {len(ids)}: {missing}")
assert len(missing) <= 15, "too many missing tasks"
assert not extra, f"cache has task_ids not in training data: {extra[:10]}"
assert not any(11 <= k <= 510 for k in cache), "test-split leak"
Z = torch.stack([t.float() for t in cache.values()])
assert (Z.norm(dim=1) - 1).abs().max() < 1e-3, "vectors must be L2-normalised"
S = Z @ Z.T; n = len(Z); off = ((S.sum() - S.diag().sum()) / (n * (n - 1))).item()
print(f"5. unit norm OK, mean off-diagonal cosine {off:.3f} (want < 0.6)")
print("CACHE OK" if off < 0.6 else "WARNING: vectors too similar")
