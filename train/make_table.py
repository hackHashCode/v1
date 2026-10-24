"""python train/make_table.py  -> results/table.md (reads results/ and fallback/)"""
import glob, json, os
import pandas as pd
csvs = [p for p in ["results/runs.csv", "fallback/runs.csv"] if os.path.exists(p)]
df = pd.concat([pd.read_csv(p) for p in csvs]).sort_values("timestamp").drop_duplicates(["run", "dataset"], keep="last")
df["pass1_plus"] *= 100
t = df.pivot_table(index="method", columns="dataset", values="pass1_plus", aggfunc=["mean", "std", "count"]).round(1)
att = {}
for p in glob.glob("results/attempts/*.json") + glob.glob("fallback/attempts_*.json"):
    run = os.path.basename(p).removeprefix("attempts_").removesuffix(".json")
    d = json.load(open(p)); att.setdefault(run.rsplit("_s", 1)[0], []).append(sum(d.values()) / len(d))
t[("mean", "attempts_to_green")] = pd.Series({k: round(sum(v) / len(v), 2) for k, v in att.items()})
os.makedirs("results", exist_ok=True)
open("results/table.md", "w").write(t.to_markdown() + "\n")
print(t)
