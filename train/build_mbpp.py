"""Build datasets/mbpp_train.jsonl: all MBPP tasks outside the test split (474 rows).

Run from repo root:  python train/build_mbpp.py
Format (contract C1): {"task_id": int, "messages": [system, user, assistant]}
Prompt (contract C13): user = description + "\n" + first test assert.
"""
import json
import os

SYSTEM = "You are an expert Python programmer. Write a correct Python function for the task. Output only code."

MBPP_RAW = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"


def load_mbpp_non_test():
    """All MBPP tasks outside the test split (task_ids 11-510). Hugging Face first, GitHub raw as fallback."""
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "full")
        return [ex for split in ["train", "validation", "prompt"] for ex in ds[split]]
    except Exception as e:  # e.g. HF blocked on this network
        print("HF load failed, using GitHub raw:", type(e).__name__)
        import urllib.request
        lines = urllib.request.urlopen(MBPP_RAW, timeout=60).read().decode().splitlines()
        return [ex for ex in map(json.loads, lines) if not 11 <= ex["task_id"] <= 510]


rows = []
for ex in load_mbpp_non_test():
    user = ex["text"].strip() + "\n" + ex["test_list"][0]
    code = ex["code"].replace("\r\n", "\n").strip()
    rows.append({"task_id": int(ex["task_id"]), "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": code},
    ]})

assert len(rows) == 474, len(rows)
assert not any(11 <= r["task_id"] <= 510 for r in rows), "test-split leak"
os.makedirs("datasets", exist_ok=True)
with open("datasets/mbpp_train.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", len(rows), "rows to datasets/mbpp_train.jsonl")
