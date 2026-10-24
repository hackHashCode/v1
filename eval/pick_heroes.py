# usage: python eval/pick_heroes.py   -> demo/ui/hero_tasks.json
# Tasks where PreCode is green on attempt 1 and SFT needs >= 3 attempts, in >= 2 of the 3 seeds.
# Hidden tests: the MBPP+ base inputs, turned into asserts using the canonical solution's outputs.
import json, os
from evalplus.data import get_mbpp_plus
P = get_mbpp_plus(); score = {}
for s in (42, 7, 1234):
    fp, fs = f"results/attempts/precode_s{s}.json", f"results/attempts/sft_s{s}.json"
    if not (os.path.exists(fp) and os.path.exists(fs)): continue
    p, q = json.load(open(fp)), json.load(open(fs))
    for t in p:
        if p[t] == 1 and q.get(t, 6) >= 3: score[t] = score.get(t, 0) + 1
heroes = []
for t, c in sorted(score.items(), key=lambda x: -x[1]):
    if c < 2 or len(heroes) >= 6: continue
    prob = P[t]; desc, first = prob["prompt"].strip().strip('"').strip().rsplit("\n", 1)
    ns = {}
    try:
        exec(prob["canonical_solution"], ns); f = ns[prob["entry_point"]]
        hidden = [f"assert {prob['entry_point']}(*{inp!r}) == {f(*inp)!r}" for inp in prob["base_input"][:7]]
    except Exception:
        hidden = []
    heroes.append({"label": t, "task": desc, "tests": [first] + hidden})
json.dump(heroes, open("demo/ui/hero_tasks.json", "w"), indent=1)
print(len(heroes), "hero tasks ->", [h["label"] for h in heroes])
