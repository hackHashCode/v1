"""python train/make_valext.py -> datasets/mbpp_valext.jsonl
Extra selection-only val set: MBPP tasks 11-510 that are NOT in MBPP+ (EvalPlus dropped them), so they are disjoint from
both training data (ids 1-10, 511-974) and the reported MBPP+ 224 test set (MBPP+ ∩ 11-510). Same prompt format as training."""
import json, urllib.request
RAW = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
raw = {e["task_id"]: e for e in map(json.loads, urllib.request.urlopen(RAW, timeout=60).read().decode().splitlines())}
test = {int(t.split("/")[1]) for t in json.load(open("eval/mbppplus_test_ids.json"))}
train = {json.loads(l)["task_id"] for l in open("datasets/mbpp_train.jsonl")}
ref = json.loads(open("datasets/mbpp_val.jsonl").readline())
SYS = ref["messages"][0]["content"]
n = 0
with open("datasets/mbpp_valext.jsonl", "w") as f:
    for t in range(11, 511):
        if t in test or t in train: continue
        e = raw[t]
        if len(e["text"] + e["test_list"][0]) > 1200: continue   # fits the 1024-token eval context (model-independent rule)
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": e["text"] + "\n" + e["test_list"][0]},
                {"role": "assistant", "content": e["code"]}]
        f.write(json.dumps({"task_id": t, "messages": msgs, "tests": e["test_list"], "setup": e["test_setup_code"]}) + "\n"); n += 1
# format check: rebuilding a known val row the same way must reproduce it exactly
e = raw[ref["task_id"]]
assert ref["messages"][1]["content"] == e["text"] + "\n" + e["test_list"][0], "prompt format mismatch"
print(n, "valext tasks")
