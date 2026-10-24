import json, urllib.request
RAW = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
raw = {e["task_id"]: e for e in map(json.loads, urllib.request.urlopen(RAW, timeout=60).read().decode().splitlines())}
train = [json.loads(l) for l in open("datasets/mbpp_train.jsonl")]
val_ids = {t for t in raw if 511 <= t <= 600}
with open("datasets/mbpp_train_noval.jsonl", "w") as f:
    for r in train:
        if r["task_id"] not in val_ids: f.write(json.dumps(r) + "\n")
with open("datasets/mbpp_val.jsonl", "w") as f:
    for r in train:
        if r["task_id"] in val_ids:
            e = raw[r["task_id"]]; f.write(json.dumps({**r, "tests": e["test_list"], "setup": e["test_setup_code"]}) + "\n")
print(sum(1 for _ in open("datasets/mbpp_train_noval.jsonl")), "train /", sum(1 for _ in open("datasets/mbpp_val.jsonl")), "val")
