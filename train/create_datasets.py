from datasets import load_dataset
import json, os
 
os.makedirs("datasets", exist_ok=True)
 
print("Loading MBPP datasets...")
dataset_full = load_dataset("google-research-datasets/mbpp", "full")
dataset_sanitized = load_dataset("google-research-datasets/mbpp", "sanitized")
 
# training: full split (train + validation + prompt) = 474 tasks
train_data = (list(dataset_full["train"]) +
              list(dataset_full["validation"]) +
              list(dataset_full["prompt"]))
 
# eval: sanitized test split = clean, verified test cases
eval_data = list(dataset_sanitized["test"])
 
print(f"Train size: {len(train_data)} (expected 474)")
print(f"Eval size:  {len(eval_data)} (expected 257)")
 
# write train
with open("datasets/mbpp_train.jsonl", "w") as f:
    for task in train_data:
        line = {
            "instruction": task["text"],
            "output": task["code"],
            "task_id": task["task_id"]
        }
        f.write(json.dumps(line) + "\n")
print("Train file written.")
 
# write eval
with open("datasets/mbpp_eval.jsonl", "w") as f:
    for task in eval_data:
        line = {
            "instruction": task["prompt"],
            "output": task["code"],
            "task_id": task["task_id"]
        }
        f.write(json.dumps(line) + "\n")
print("Eval file written.")
 
# verify line counts
train_count = sum(1 for _ in open("datasets/mbpp_train.jsonl"))
eval_count  = sum(1 for _ in open("datasets/mbpp_eval.jsonl"))
print(f"\nLine counts: train={train_count}, eval={eval_count}")
 
# collect task IDs
train_ids = {json.loads(l)["task_id"] for l in open("datasets/mbpp_train.jsonl")}
eval_ids  = {json.loads(l)["task_id"] for l in open("datasets/mbpp_eval.jsonl")}
 
# check 1: no overlap
overlap = train_ids & eval_ids
print(f"Overlap between train and eval: {len(overlap)} (must be 0)")
if overlap:
    print(f"  PROBLEM — overlapping IDs: {sorted(overlap)[:10]}")
 
# check 2: no sanitized eval IDs leaked into training
# sanitized test split uses task IDs 11-510
leaked = [tid for tid in train_ids if 11 <= tid <= 510]
print(f"Eval-range IDs (11-510) in train: {len(leaked)} (must be 0)")
if leaked:
    print(f"  PROBLEM — leaked IDs: {sorted(leaked)[:10]}")
 
# check 3: print ID ranges so we know what we're working with
print(f"\nTrain ID range: {min(train_ids)} to {max(train_ids)}")
print(f"Eval ID range:  {min(eval_ids)} to {max(eval_ids)}")
 
# print one sample from each
print("\nSample train example:")
with open("datasets/mbpp_train.jsonl") as f:
    s = json.loads(f.readline())
    print(f"  task_id: {s['task_id']}")
    print(f"  instruction: {s['instruction'][:80]}...")
    print(f"  output: {s['output'][:60]}...")
 
print("\nSample eval example:")
with open("datasets/mbpp_eval.jsonl") as f:
    s = json.loads(f.readline())
    print(f"  task_id: {s['task_id']}")
    print(f"  instruction: {s['instruction'][:80]}...")
 
print("\nDone.")