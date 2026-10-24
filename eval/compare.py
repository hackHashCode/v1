"""Cross-method comparison -> results/COMPARISON.md (+ results/comparison.json, results/runs.csv rebuilt).

python eval/compare.py [--results results] [--models /data/precode/models]

Reads per-run scores written by the Modal pipeline (results/rows/<run>_<dataset>.json) and per-task pass/fail
(results/per_task/<run>_<dataset>.json, contract C10). Reports:
  1. pass@1 (base and plus) per method and dataset: mean, std and n over seeds
  2. exact McNemar tests, per-task pairs pooled over seeds: every method vs SFT, plus the ablation ladder
     sft -> jepa -> precode -> precodence -> precodefull_aux -> precodefull_prefix -> precodefull
  3. end-of-training behaviour stats per run from train_log.jsonl (PreCode-full family): cos(own z), top-1
"""
import argparse, csv, glob, json, math, os, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--models", default="/data/precode/models")
a = ap.parse_args()

ORDER = ["base", "sft", "jepa", "precode", "precodence", "sft_loop", "precodefull_aux", "precodefull_prefix", "precodefull"]
NAMES = {"base": "Zero-shot", "sft": "SFT", "jepa": "LLM-JEPA", "precode": "PreCode (cosine)",
         "precodence": "PreCode (InfoNCE)", "sft_loop": "SFT (PreCode-full loop, λ=0)", "precodefull_aux": "PreCode-full: aux only",
         "precodefull_prefix": "PreCode-full: + prefix", "precodefull": "PreCode-full: + prefix + weighting"}
LADDER = [("sft_loop", "sft"), ("precodefull_aux", "sft_loop"), ("precodefull", "sft_loop"), ("precodefull_aux", "jepa"), ("precodefull", "jepa"), ("jepa", "sft"), ("precode", "jepa"), ("precodence", "precode"), ("precodefull_aux", "precodence"),
          ("precodefull_prefix", "precodefull_aux"), ("precodefull", "precodefull_prefix"), ("precodefull", "sft")]

rows = []
for p in sorted(glob.glob(os.path.join(a.results, "rows", "*.json"))):
    rows.append(json.load(open(p)))
if not rows and os.path.exists(os.path.join(a.results, "runs.csv")):     # older single-file format
    rows = [dict(r, pass1_base=float(r["pass1_base"]), pass1_plus=float(r["pass1_plus"])) for r in csv.DictReader(open(os.path.join(a.results, "runs.csv")))]
latest = {}
for r in rows:
    latest[(r["run"], r["dataset"])] = r
rows = list(latest.values())
datasets = sorted({r["dataset"] for r in rows})
methods = [m for m in ORDER if any(r["method"] == m for r in rows)] + sorted({r["method"] for r in rows} - set(ORDER))


def agg(m, ds, key):
    v = [float(r[key]) * 100 for r in rows if r["method"] == m and r["dataset"] == ds]
    if not v:
        return None
    return {"mean": st.mean(v), "std": st.stdev(v) if len(v) > 1 else 0.0, "n": len(v)}


def binom_p(k, n):
    """two-sided exact binomial (McNemar) p-value without scipy."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def per_task(m, ds):
    out = {}
    for p in glob.glob(os.path.join(a.results, "per_task", f"{m}_s*_{ds}.json")):
        seed = os.path.basename(p)[len(m) + 2:-len(ds) - 6]
        if seed.isdigit():
            out[int(seed)] = json.load(open(p))
    return out


def mcnemar(A, B, ds):
    pa, pb = per_task(A, ds), per_task(B, ds)
    if A == "base" and pa:      # zero-shot has one run (seed 0): pair it with every seed of the other method
        pa = {s: next(iter(pa.values())) for s in pb}
    if B == "base" and pb:
        pb = {s: next(iter(pb.values())) for s in pa}
    seeds = sorted(s for s in pa if pb.get(s) and pa.get(s))
    a_only = b_only = both = 0
    for s in seeds:
        x, y = pa[s], pb[s]
        for t in set(x) & set(y):
            a_only += x[t] and not y[t]; b_only += y[t] and not x[t]; both += x[t] and y[t]
    return {"a": A, "b": B, "dataset": ds, "seeds": seeds, "a_only": a_only, "b_only": b_only, "both": both,
            "p": round(binom_p(min(a_only, b_only), a_only + b_only), 4)}


def train_stats(run_dir):
    p = os.path.join(run_dir, "train_log.jsonl")
    if not os.path.exists(p):
        return None
    recs = [json.loads(l) for l in open(p) if l.strip()]
    tail = recs[-4:]
    f = lambda k: round(st.mean([r[k] for r in tail if k in r]), 3) if any(k in r for r in tail) else None
    return {"cos_own": f("cos_own"), "top1": f("top1"), "ntp": f("ntp"), "beh": f("beh"), "steps": recs[-1]["step"] if recs else 0}


lines = ["# PreCode-JEPA: method comparison", "",
         "pass@1 in %, greedy decoding, mean ± std over seeds (n). MBPP+ = 224 test-split tasks, HumanEval+ = 164.",
         "\"plus\" = passes EvalPlus's extended tests too (the headline number).", ""]
hdr = "| method | " + " | ".join(f"{d} plus | {d} base" for d in datasets) + " |"
lines += [hdr, "|" + "---|" * (1 + 2 * len(datasets))]
best = {d: max((agg(m, d, "pass1_plus") or {"mean": -1})["mean"] for m in methods) for d in datasets}
table = {}
for m in methods:
    cells = []
    for d in datasets:
        pl, ba = agg(m, d, "pass1_plus"), agg(m, d, "pass1_base")
        table.setdefault(m, {})[d] = {"plus": pl, "base": ba}
        fmt = lambda x: "—" if not x else f"{x['mean']:.1f} ± {x['std']:.1f} ({x['n']})"
        c = fmt(pl)
        if pl and abs(pl["mean"] - best[d]) < 1e-9:
            c = f"**{c}**"
        cells += [c, fmt(ba)]
    lines.append(f"| {NAMES.get(m, m)} | " + " | ".join(cells) + " |")

tests = []
lines += ["", "## Paired tests (exact McNemar, tasks pooled over seeds)", "",
          "\"A only\" = tasks A solves and B doesn't. A gap only counts if p < 0.05 **and** the sign agrees across seeds.", "",
          "| A | B | dataset | A only | B only | both | p | seeds |", "|---|---|---|---|---|---|---|---|"]
pairs = [(m, "sft") for m in methods if m not in ("sft",)] + [p for p in LADDER if p not in [(m, "sft") for m in methods]]
seen = set()
for A, B in pairs:
    if A not in methods or B not in methods or (A, B) in seen:
        continue
    seen.add((A, B))
    for d in datasets:
        r = mcnemar(A, B, d)
        if not r["seeds"]:
            continue
        tests.append(r)
        sig = " ✱" if r["p"] < 0.05 else ""
        lines.append(f"| {NAMES.get(A, A)} | {NAMES.get(B, B)} | {d} | {r['a_only']} | {r['b_only']} | {r['both']} | {r['p']}{sig} | {','.join(map(str, r['seeds']))} |")

stats = {}
if os.path.isdir(a.models):
    for d in sorted(glob.glob(os.path.join(a.models, "*_s*"))):
        s = train_stats(d)
        if s:
            stats[os.path.basename(d)] = s
if stats:
    lines += ["", "## Behaviour prediction at the end of training (PreCode-full runs)", "",
              "cos_own = cos(predicted z, own task's z); top1 = fraction where own z is the nearest of all cached tasks"
              " (training tasks, so this measures fit, not generalisation).", "",
              "| run | cos_own | top1 | NTP | L_beh | steps |", "|---|---|---|---|---|---|"]
    for k, s in stats.items():
        lines.append(f"| {k} | {s['cos_own']} | {s['top1']} | {s['ntp']} | {s['beh']} | {s['steps']} |")

metrics = {}
for p in glob.glob(os.path.join(a.results, "metrics", "*.json")):
    d = json.load(open(p)); d.pop("per_task_match", None)
    metrics[d["run"]] = d
if metrics:
    def magg(m, k):
        v = [d[k] for r, d in metrics.items() if r.rsplit("_s", 1)[0] == m and k in d]
        return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)) if v else None
    fm = lambda x, pct=True: "—" if not x else (f"{100 * x[0]:.1f} ± {100 * x[1]:.1f}" if pct else f"{x[0]:.3f} ± {x[1]:.3f}")
    lines += ["", "## Behaviour metrics on the 224 MBPP+ test tasks (eval/analyze.py)", "",
              "behaviour match = % of probe inputs where the generated code's output equals the canonical solution's (partial credit). "
              "exact = % of tasks matching on every probe. Recall@k / cos only exist for methods with a behaviour head; "
              "test-task behaviour vectors are built for this analysis only, training never sees them.", "",
              "| method | behaviour match % | exact behaviour % | recall@1 % | recall@5 % | cos(pred, true) | cos(pred, own code) | cos_own solved / unsolved |",
              "|---|---|---|---|---|---|---|---|"]
    mmeth = methods + [m for m in ORDER if m not in methods and any(r.rsplit("_s", 1)[0] == m for r in metrics)]
    for m in mmeth:
        if not any(r.rsplit("_s", 1)[0] == m for r in metrics):
            continue
        cp, cf = magg(m, "cos_own_pass"), magg(m, "cos_own_fail")
        split = "—" if not cp else f"{cp[0]:.3f} / {cf[0]:.3f}"
        lines.append(f"| {NAMES.get(m, m)} | {fm(magg(m, 'behaviour_match'))} | {fm(magg(m, 'exact_behaviour'))} | "
                     f"{fm(magg(m, 'recall@1'))} | {fm(magg(m, 'recall@5'))} | {fm(magg(m, 'cos_own'), False)} | "
                     f"{fm(magg(m, 'cos_generated'), False)} | {split} |")
    r1 = next((d["random@1"] for d in metrics.values() if "random@1" in d), None)
    if r1:
        lines.append(f"\nRandom recall@1 = {100 * r1:.2f}%, recall@5 = {100 * min(1.0, 5 * r1):.2f}%.")
    figs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.results, "tsne_*.png")))
    if figs:
        lines.append("Figures: " + ", ".join(figs))

os.makedirs(a.results, exist_ok=True)
open(os.path.join(a.results, "COMPARISON.md"), "w").write("\n".join(lines) + "\n")
json.dump({"table": table, "tests": tests, "train_stats": stats, "metrics": metrics}, open(os.path.join(a.results, "comparison.json"), "w"), indent=1)
with open(os.path.join(a.results, "runs.csv"), "w", newline="") as f:      # rebuilt so demo/studio's Results tab sees every run
    w = csv.writer(f)
    w.writerow(["run", "method", "seed", "dataset", "pass1_base", "pass1_plus", "n_tasks", "timestamp"])
    for r in sorted(rows, key=lambda r: (ORDER.index(r["method"]) if r["method"] in ORDER else 99, r["run"], r["dataset"])):
        w.writerow([r["run"], r["method"], r["seed"], r["dataset"], r["pass1_base"], r["pass1_plus"], r["n_tasks"], r.get("timestamp", 0)])
print("\n".join(lines))
