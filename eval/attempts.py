# usage: python eval/attempts.py <run>   [WS, env evalv]  -> results/attempts/<run>.json  (contract C11)
# Attempts-to-green: 5 independent samples at T=0.8; first passing attempt (base AND plus), 6 if none.
import glob, json, os, subprocess, sys
run = sys.argv[1]; out = f"/data/precode/samples/{run}"; os.makedirs(out, exist_ok=True)
model = "meta-llama/Llama-3.2-1B-Instruct" if run == "base_s0" else f"/data/precode/models/{run}"
samples = f"{out}/mbpp_n5.jsonl"
for f in glob.glob(f"{out}/mbpp_n5*eval_results.json"): os.remove(f)
subprocess.run(["python", "eval/generate.py", "--model", model, "--dataset", "mbpp", "--n", "5",
                "--temperature", "0.8", "--out", samples], check=True)
subprocess.run(["evalplus.evaluate", "--dataset", "mbpp", "--samples", samples], check=True)
resf = next(iter(glob.glob(f"{out}/mbpp_n5*eval_results.json")))
res = json.load(open(resf))["eval"]
ids = json.load(open("eval/mbppplus_test_ids.json"))
att = {}
for t in ids:
    ok = [r["base_status"] == "pass" and r["plus_status"] == "pass" for r in res[t]]
    assert len(ok) == 5, (t, len(ok))
    att[t] = next((i + 1 for i, o in enumerate(ok) if o), 6)       # samples are iid: order doesn't bias
os.makedirs("results/attempts", exist_ok=True)
json.dump(att, open(f"results/attempts/{run}.json", "w"))
v = list(att.values())
print(run, "mean attempts", round(sum(v) / len(v), 3), "| solved by 1/3/5:",
      [round(sum(x <= k for x in v) / len(v), 3) for k in (1, 3, 5)])
