"""python demo/studio/make_showcase.py  ->  demo/studio/showcase.json

Showcase tasks for the demo: MBPP+ test tasks where PreCode (precodefull_aux) passed in >= 4 of 5 seeds and SFT
passed in <= 1 of 5, read from results/qc15/per_task. Also records the reverse count, so the demo can say
how often it goes the other way. Prompts, asserts and reference code come from the public MBPP release
(the MBPP+ prompt is the same description + first assert; MBPP+ adds extra hidden tests on top)."""
import json, os, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PT = os.path.join(ROOT, "results", "qc15", "per_task")
SEEDS = [42, 7, 1234, 11, 99]
RAW = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"


def per_task(run):
    return json.load(open(os.path.join(PT, f"{run}_mbppplus224.json")))


def wins(method):
    return [per_task(f"{method}_s{s}") for s in SEEDS]


pc, sft, jepa, ctrl = wins("precodefull_aux"), wins("sft"), wins("jepa"), wins("sft_loop")
count = lambda runs, t: sum(bool(r.get(t)) for r in runs)
tasks = list(pc[0])
picked = [t for t in tasks if count(pc, t) >= 4 and count(sft, t) <= 1]
reverse = [t for t in tasks if count(sft, t) >= 4 and count(pc, t) <= 1]

raw = {e["task_id"]: e for e in map(json.loads, urllib.request.urlopen(RAW, timeout=60).read().decode().splitlines())}
out = []
for t in sorted(picked, key=lambda t: (-count(pc, t), count(sft, t), count(jepa, t))):
    e = raw[int(t.split("/")[1])]
    label = e["text"].removeprefix("Write a python function to ").removeprefix("Write a function to ").rstrip(".")
    out.append({"label": label[:1].upper() + label[1:48] + ("…" if len(label) > 48 else ""), "task_id": t,
                "task": e["text"], "tests": e["test_list"], "setup": e["test_setup_code"], "reference": e["code"],
                "record": {"precode": count(pc, t), "sft": count(sft, t), "jepa": count(jepa, t), "sft_loop": count(ctrl, t),
                           "seeds": len(SEEDS)}})
json.dump({"source": "results/qc15/per_task (MBPP+, 5 seeds)", "rule": "PreCode passes >= 4/5 seeds, SFT <= 1/5",
           "n_picked": len(picked), "n_reverse": len(reverse), "n_tasks": len(tasks), "tasks": out},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "showcase.json"), "w"), indent=1)
print(f"{len(picked)} showcase tasks (reverse: {len(reverse)} of {len(tasks)})")
