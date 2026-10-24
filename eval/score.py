# usage: python eval/score.py <eval_results.json> <run> <dataset> [--ids FILE]   [WS or Laptop, repo root]
# contracts C9, C10. dataset: mbppplus224 | humanevalplus | para60_orig | para60_para
import argparse, csv, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("results"); ap.add_argument("run"); ap.add_argument("dataset")
ap.add_argument("--ids")                                       # jsonl with task_id, e.g. the paraphrase file
a = ap.parse_args()

res = json.load(open(a.results))["eval"]
if a.ids:
    ids = [json.loads(l)["task_id"] for l in open(a.ids)]
elif a.dataset == "mbppplus224":
    ids = json.load(open("eval/mbppplus_test_ids.json"))
else:
    ids = list(res)

per, nb, npl = {}, 0, 0
for t in ids:
    r = res[t][0]
    b = r["base_status"] == "pass"; p = b and r["plus_status"] == "pass"
    nb += b; npl += p; per[t] = bool(p)

os.makedirs("results/per_task", exist_ok=True)
json.dump(per, open(f"results/per_task/{a.run}_{a.dataset}.json", "w"))
method, seed = a.run.rsplit("_s", 1)
new = not os.path.exists("results/runs.csv")
with open("results/runs.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(["run", "method", "seed", "dataset", "pass1_base", "pass1_plus", "n_tasks", "timestamp"])
    w.writerow([a.run, method, seed, a.dataset, round(nb / len(ids), 4), round(npl / len(ids), 4),
                len(ids), int(time.time())])
print(a.run, a.dataset, len(ids), "base", round(nb / len(ids), 4), "plus", round(npl / len(ids), 4))
