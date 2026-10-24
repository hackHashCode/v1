"""python fallback/attempts_min.py RESULTS_JSON(n=5 samples) RUN  -> fallback/attempts_<RUN>.json (contract C11)"""
import json, sys
res, run = json.load(open(sys.argv[1]))["eval"], sys.argv[2]
ids = json.load(open("fallback/mbppplus_test_ids.json"))
att = {}
for t in ids:
    ok = [r["base_status"] == "pass" and r["plus_status"] == "pass" for r in res[t]]
    att[t] = next((i + 1 for i, o in enumerate(ok) if o), 6)
json.dump(att, open(f"fallback/attempts_{run}.json", "w"))
v = list(att.values())
print(run, "mean attempts", round(sum(v) / len(v), 3), "solved by 1/3/5:", [round(sum(x <= k for x in v) / len(v), 3) for k in (1, 3, 5)])
