"""Mutant-separation test: does z encode behaviour, not surface form?

For MBPP training tasks we make small automatic bugs (operator swaps, off-by-one constants, min<->max,
dropping a set()/sorted() wrapper), keep only mutants that STILL PASS the first visible test but behave
differently on the probes ("looks right, is wrong"), and compare:
  behaviour z (ours)            : cos(z_ref, z_mutant)
  code-text embedding (baseline): cos(text_ref, text_mutant)
python cache/mutant_test.py [--encoder features|mpnet|hybrid] [--n 120]
-> results/plots/mutant_numbers.json, results/plots/mutant_separation.png, cache/mutants.jsonl"""
import argparse, ast, json, os, random, subprocess, sys, tempfile
sys.path.insert(0, "cache")
import torch.nn.functional as F
from behaviour import make_probes, trace, encode, encode_code_text, load_mbpp_train

ap = argparse.ArgumentParser()
ap.add_argument("--encoder", default="features", choices=["features", "mpnet", "hybrid"])
ap.add_argument("--n", type=int, default=120)
a = ap.parse_args()

SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Add: ast.Sub, ast.Sub: ast.Add,
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}

class Mut(ast.NodeTransformer):
    """Apply the k-th applicable mutation site."""
    def __init__(self, k): self.k, self.i, self.kind = k, 0, None
    def _hit(self, kind):
        self.i += 1
        if self.i == self.k: self.kind = kind; return True
        return False
    def visit_Compare(self, n):
        self.generic_visit(n)
        n.ops = [SWAP[type(o)]() if type(o) in SWAP and self._hit("cmp-swap") else o for o in n.ops]; return n
    def visit_BinOp(self, n):
        self.generic_visit(n)
        if type(n.op) in SWAP and self._hit("arith-swap"): n.op = SWAP[type(n.op)]()
        return n
    def visit_Constant(self, n):
        if isinstance(n.value, int) and not isinstance(n.value, bool) and self._hit("off-by-one"):
            return ast.copy_location(ast.Constant(n.value + 1), n)
        return n
    def visit_Call(self, n):
        self.generic_visit(n)
        if isinstance(n.func, ast.Name):
            if n.func.id in ("min", "max") and self._hit("min-max"):
                n.func.id = "max" if n.func.id == "min" else "min"
            elif n.func.id in ("set", "sorted") and len(n.args) == 1 and self._hit("drop-" + n.func.id):
                return n.args[0]
        return n

def passes(code, setup, test):
    try:
        return subprocess.run([sys.executable, "-c", f"{setup}\n{code}\n{test}"], capture_output=True,
                              timeout=10, cwd=tempfile.gettempdir()).returncode == 0
    except subprocess.TimeoutExpired:
        return False

random.seed(0)
tasks = load_mbpp_train(); random.shuffle(tasks)
beh, txt, other, kinds, refs, rows = [], [], [], {}, [], []
for ex in tasks:
    if len(rows) >= a.n: break
    setup = ex.get("test_setup_code", ""); pr = make_probes(ex["test_list"])
    ref = trace(ex["code"], setup, pr)
    if not ref or not passes(ex["code"], setup, ex["test_list"][0]): continue
    for k in range(1, 12):
        m = Mut(k)
        try: mcode = ast.unparse(m.visit(ast.parse(ex["code"])))
        except Exception: break
        if m.kind is None: break
        if not passes(mcode, setup, ex["test_list"][0]): continue          # must still look right
        mt = trace(mcode, setup, pr)
        if not mt or mt == ref: continue                                     # must behave differently
        zr, zm = encode(ref, a.encoder), encode(mt, a.encoder)
        tr_, tm = encode_code_text(ex["code"], a.encoder), encode_code_text(mcode, a.encoder)
        beh.append(F.cosine_similarity(zr, zm, dim=0).item()); txt.append(F.cosine_similarity(tr_, tm, dim=0).item())
        kinds[m.kind] = kinds.get(m.kind, 0) + 1; refs.append(zr)
        rows.append({"task_id": ex["task_id"], "kind": m.kind, "code": ex["code"], "mutant": mcode,
                     "differing_probes": sum(x != y for x, y in zip(ref, mt)), "n_probes": len(ref)})
        break
for x, y in zip(refs, refs[1:]):
    other.append(F.cosine_similarity(x, y, dim=0).item())
avg = lambda v: round(sum(v) / len(v), 3)
res = {"encoder": a.encoder, "n_mutants": len(beh), "mutation_kinds": kinds,
       "behaviour_ref_vs_mutant": avg(beh), "codetext_ref_vs_mutant": avg(txt),
       "behaviour_ref_vs_other_task": avg(other),
       "frac_mutants_where_behaviour_separates_more": round(sum(b < t for b, t in zip(beh, txt)) / len(beh), 3),
       "mean_differing_probes": avg([r["differing_probes"] for r in rows])}
os.makedirs("results/plots", exist_ok=True)
json.dump({**res, "pairs_behaviour": beh, "pairs_codetext": txt}, open(f"results/plots/mutant_numbers_{a.encoder}.json", "w"), indent=1)
with open("cache/mutants.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(json.dumps(res, indent=1))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(7.5, 4.2))
    lo = max(0.0, (min(beh + txt) // 0.05) * 0.05)
    bins = [lo + i * 0.025 for i in range(int((1.0 - lo) / 0.025) + 2)]
    plt.hist(txt, bins=bins, alpha=.65, label=f"code-text embedding (mean {res['codetext_ref_vs_mutant']})")
    plt.hist(beh, bins=bins, alpha=.65, label=f"behaviour embedding, ours (mean {res['behaviour_ref_vs_mutant']})")
    plt.xlabel("cosine similarity: correct solution vs. buggy mutant that passes the visible test")
    plt.ylabel("mutant pairs"); plt.title(f"{len(beh)} 'looks right, is wrong' bugs: our target separates {res['frac_mutants_where_behaviour_separates_more']:.0%} of them more")
    plt.legend(loc="upper left"); plt.tight_layout()
    plt.savefig(f"results/plots/mutant_separation_{a.encoder}.png", dpi=200)
    print("saved plot")
except ImportError:
    print("matplotlib not installed; skipped plot")
