"""Behaviour metrics beyond pass@1, on the 224 MBPP+ test tasks (run on Linux: uses cache/behaviour.py's sandbox).

  python eval/analyze.py build-test --results R            # canonical behaviour of every test task -> R/test_behaviour_anon.json
  python eval/analyze.py run --run sft_s42 --samples S --results R      # -> R/metrics/sft_s42.json
  python eval/analyze.py figure --run precodefull_aux_s42 --samples S --results R   # -> R/tsne_<run>.png

Test-task behaviour vectors are built ONLY for this analysis (training never sees them): probes are the EvalPlus
base inputs then plus inputs (first 10 of each, long reprs skipped), run on the canonical solution, traced and
encoded exactly like the training cache (cache/behaviour.py: trace + encode_features, function renamed to f).

Per run:
  behaviour_match   mean over tasks of the fraction of probes where the generated code's output equals the
                    canonical output (partial credit; 1.0 = behaves identically on every probe)
  exact_behaviour   fraction of tasks with behaviour_match == 1
For runs with a behaviour head (PreCode family; samples/<run>/mbpp_z.json from eval/generate_hf.py --zout):
  recall@1, @5      the predicted z's nearest test-task z (out of 224) is the true task (random: 0.45%, 2.2%)
  cos_own           cos(predicted z, true task's z)        cos_generated  cos(predicted z, z of the code it wrote)
  cos_own_pass / cos_own_fail   cos_own split by whether the task was solved (does a better guess go with success?)
"""
import argparse, ast, json, os, sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cache"))
import behaviour as B  # noqa: E402

MAX_PROBES_PER_KIND, MAX_REPR = 10, 300


def probes_for(prob):
    name, calls = prob["entry_point"], []
    for kind in ("base_input", "plus_input"):
        n = 0
        for inp in prob.get(kind) or []:
            if n >= MAX_PROBES_PER_KIND:
                break
            s = f"{name}({', '.join(repr(x) for x in inp)})"
            if len(s) <= MAX_REPR and s not in calls:
                calls.append(s); n += 1
    return calls


def with_entry(code, entry):
    """If the model named its function differently, alias the last top-level def to the expected entry point."""
    try:
        defs = [n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)]
    except SyntaxError:
        return code
    if entry in defs or not defs:
        return code
    return code + f"\n{entry} = {defs[-1]}\n"


def outputs(code, probes):
    return {l.partition(" -> ")[0]: l.partition(" -> ")[2] for l in B.trace(code, "", probes)}


def _canon(item):
    tid, prob = item
    probes = probes_for(prob)
    out = outputs(prob["canonical_solution"], probes)
    lines = [f"{c} -> {out[c]}" for c in probes if c in out]
    return tid, {"probes": probes, "out": out, "z": B.encode_features(lines, anonymize_names=True).tolist() if lines else None}


def build_test(results):
    from evalplus.data import get_mbpp_plus
    P = get_mbpp_plus()
    ids = json.load(open(os.path.join(ROOT, "eval", "mbppplus_test_ids.json")))
    with Pool(min(16, os.cpu_count() or 4)) as pool:
        res = dict(pool.map(_canon, [(t, P[t]) for t in ids]))
    entry = {t: P[t]["entry_point"] for t in ids}
    for t in res:
        res[t]["entry"] = entry[t]
    os.makedirs(results, exist_ok=True)
    json.dump(res, open(os.path.join(results, "test_behaviour_anon.json"), "w"))
    ok = sum(1 for v in res.values() if v["z"] is not None)
    print(f"[test] behaviour vectors for {ok}/{len(ids)} test tasks, mean probes {sum(len(v['probes']) for v in res.values()) / len(res):.1f}")


def _one(args):
    tid, code, tb = args
    o = outputs(with_entry(code, tb["entry"]), tb["probes"])
    common = [c for c in tb["probes"] if c in tb["out"]]
    match = sum(1 for c in common if o.get(c) == tb["out"][c]) / len(common) if common else 0.0
    lines = [f"{c} -> {o[c]}" for c in tb["probes"] if c in o]
    return tid, match, (B.encode_features(lines, anonymize_names=True).tolist() if lines else None)


def analyze_run(run, samples, results):
    import torch
    import torch.nn.functional as F
    tb = json.load(open(os.path.join(results, "test_behaviour_anon.json")))
    sol = {}
    for l in open(os.path.join(samples, run, "mbpp.jsonl")):
        r = json.loads(l); sol[r["task_id"]] = r["solution"]
    ids = [t for t in tb if tb[t]["z"] is not None and t in sol]
    with Pool(min(16, os.cpu_count() or 4)) as pool:
        rows = pool.map(_one, [(t, sol[t], tb[t]) for t in ids])
    match = {t: m for t, m, _ in rows}
    zgen = {t: z for t, _, z in rows}
    out = {"run": run, "n_tasks": len(ids), "name_free": True,
           "behaviour_match": round(sum(match.values()) / len(match), 4),
           "exact_behaviour": round(sum(1 for m in match.values() if m == 1.0) / len(match), 4)}
    zp_path = os.path.join(samples, run, "mbpp_z.json")
    if os.path.exists(zp_path):
        zp = json.load(open(zp_path))
        ids_z = [t for t in ids if t in zp]
        Zt = F.normalize(torch.tensor([tb[t]["z"] for t in ids_z]), dim=-1)
        Zp = F.normalize(torch.tensor([zp[t] for t in ids_z]), dim=-1)
        sims = Zp @ Zt.T                                            # [n, n]
        rank = (sims > sims.diag()[:, None]).sum(1)                 # 0 = own task is the nearest
        cos_own = sims.diag()
        cos_gen = [float(F.cosine_similarity(Zp[i], F.normalize(torch.tensor(zgen[t]), dim=0), dim=0))
                   for i, t in enumerate(ids_z) if zgen.get(t)]
        pt = os.path.join(results, "per_task", f"{run}_mbppplus224.json")
        passed = json.load(open(pt)) if os.path.exists(pt) else {}
        cp = [float(cos_own[i]) for i, t in enumerate(ids_z) if passed.get(t)]
        cf = [float(cos_own[i]) for i, t in enumerate(ids_z) if t in passed and not passed[t]]
        out.update({"recall@1": round(float((rank == 0).float().mean()), 4), "recall@5": round(float((rank < 5).float().mean()), 4),
                    "random@1": round(1 / len(ids_z), 4), "cos_own": round(float(cos_own.mean()), 4),
                    "cos_generated": round(sum(cos_gen) / max(1, len(cos_gen)), 4),
                    "cos_own_pass": round(sum(cp) / max(1, len(cp)), 4), "cos_own_fail": round(sum(cf) / max(1, len(cf)), 4)})
    os.makedirs(os.path.join(results, "metrics"), exist_ok=True)
    json.dump({**out, "per_task_match": match}, open(os.path.join(results, "metrics", f"{run}.json"), "w"))
    print(json.dumps(out))


def figure(run, samples, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    tb = json.load(open(os.path.join(results, "test_behaviour_anon.json")))
    zp = json.load(open(os.path.join(samples, run, "mbpp_z.json")))
    ids = [t for t in tb if tb[t]["z"] is not None and t in zp]
    Zt = np.array([tb[t]["z"] for t in ids]); Zp = np.array([zp[t] for t in ids])
    Zt /= np.linalg.norm(Zt, axis=1, keepdims=True); Zp /= np.linalg.norm(Zp, axis=1, keepdims=True)
    nearest_ok = (Zp @ Zt.T).argmax(1) == np.arange(len(ids))
    xy = TSNE(n_components=2, metric="cosine", init="random", perplexity=min(30, max(2, (2 * len(ids) - 1) // 3)), random_state=0).fit_transform(np.vstack([Zt, Zp]))
    t_xy, p_xy = xy[:len(ids)], xy[len(ids):]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    for i in range(len(ids)):
        ax.plot([t_xy[i, 0], p_xy[i, 0]], [t_xy[i, 1], p_xy[i, 1]], lw=0.6, alpha=0.5,
                color="#2a9d5c" if nearest_ok[i] else "#c9c9c9")
    ax.scatter(*t_xy.T, s=14, c="#3a3a3a", label="true behaviour (canonical solution)")
    ax.scatter(*p_xy.T, s=18, marker="*", c="#d98a00", label="PreCode prediction from the description")
    ax.set_title(f"{run}: predicted vs true behaviour on 224 unseen MBPP+ tasks\n"
                 f"green line = prediction's nearest true vector is its own task ({nearest_ok.mean():.0%})", fontsize=10)
    ax.legend(loc="lower left", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    p = os.path.join(results, f"tsne_{run}.png")
    fig.savefig(p, bbox_inches="tight"); print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-test", "run", "figure"])
    ap.add_argument("--run")
    ap.add_argument("--samples", default="/data/precode/samples")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    if a.cmd == "build-test":
        build_test(a.results)
    elif a.cmd == "run":
        analyze_run(a.run, a.samples, a.results)
    else:
        figure(a.run, a.samples, a.results)
