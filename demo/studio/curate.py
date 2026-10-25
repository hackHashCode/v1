"""python demo/studio/curate.py  ->  adds a "stage" list to demo/studio/showcase.json

The stage page cycles only the showcase tasks where the benchmark gap reproduces in the recorded live run
(seed-42 models, MBPP's public asserts): PreCode passes every assert, SFT doesn't. Tasks where SFT's code runs
but gives wrong answers come first (a logic bug is the clearer story than output that never stops).
The page states how many of the 13 these are; the full studio still lists all of them."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, "showcase.json")
show = json.load(open(SC))
logic, other = [], []
for t in show["tasks"]:
    p = os.path.join(HERE, "recordings", t["task_id"].replace("/", "_") + ".json")
    if not os.path.exists(p):
        continue
    res = json.load(open(p))["exec"]["results"]
    ours, sft = res.get("precodefull_aux", {}).get("tests", []), res.get("sft", {}).get("tests", [])
    if ours and all(x["ok"] for x in ours) and not all(x["ok"] for x in sft):
        (other if any("does not run" in x["why"] for x in sft) else logic).append((sum(x["ok"] for x in sft), t["task_id"]))
show["stage"] = [t for _, t in sorted(logic)] + [t for _, t in sorted(other)]   # biggest live gap first
json.dump(show, open(SC, "w"), indent=1)
print("stage:", show["stage"])
