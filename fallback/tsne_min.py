"""Corrected t-SNE: the SAME tasks and their buggy mutants, in two spaces.
Left: code-text embedding (the kind of target LLM-JEPA uses). Right: behaviour embedding z (our target).
Grey lines join each correct solution to its mutant: short on the left, long on the right = the point.
python fallback/tsne_min.py -> results/plots/tsne_comparison.png   (CPU; encoder download needs HF access)
pip install scikit-learn matplotlib"""
import ast, json, os, random, sys, urllib.request
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
sys.path.insert(0, "fallback")
from build_cache_min import calls, probes, run, embed

SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Add: ast.Sub, ast.Sub: ast.Add}
class Mut(ast.NodeTransformer):
    def __init__(self, k): self.k, self.i = k, 0
    def _m(self, op):
        if type(op) in SWAP:
            self.i += 1
            if self.i == self.k: return SWAP[type(op)]()
        return op
    def visit_Compare(self, n): self.generic_visit(n); n.ops = [self._m(o) for o in n.ops]; return n
    def visit_BinOp(self, n): self.generic_visit(n); n.op = self._m(n.op); return n

raw = urllib.request.urlopen("https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl").read().decode().splitlines()
tasks = [e for e in map(json.loads, raw) if not 11 <= e["task_id"] <= 510]
enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
random.seed(0)
code_vecs, beh_vecs, pairs = [], [], []          # pairs: (index of correct, index of mutant)
for ex in random.sample(tasks, 200):
    if len(pairs) >= 50: break
    pr = probes(calls(ex["test_list"])); ref = run(ex["code"], ex["test_setup_code"], pr)
    if not ref: continue
    for k in (1, 2, 3):
        try: mcode = ast.unparse(Mut(k).visit(ast.parse(ex["code"])))
        except Exception: continue
        mt = run(mcode, ex["test_setup_code"], pr)
        if mt and mt != ref:
            i = len(code_vecs)
            cv = enc.encode([ex["code"], mcode], normalize_embeddings=True)
            code_vecs += [cv[0], cv[1]]
            beh_vecs += [embed(enc, ref).numpy(), embed(enc, mt).numpy()]
            pairs.append((i, i + 1)); break

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, vecs, title in [(axes[0], code_vecs, "Code-text embedding (LLM-JEPA-style target)"),
                        (axes[1], beh_vecs, "Behaviour embedding z (PreCode target)")]:
    X = TSNE(n_components=2, perplexity=15, random_state=0, init="pca").fit_transform(np.stack(vecs))
    for a, b in pairs:
        ax.plot([X[a, 0], X[b, 0]], [X[a, 1], X[b, 1]], color="grey", lw=0.8, alpha=0.6)
    ax.scatter(X[[a for a, _ in pairs], 0], X[[a for a, _ in pairs], 1], s=22, label="correct solution")
    ax.scatter(X[[b for _, b in pairs], 0], X[[b for _, b in pairs], 1], s=22, marker="x", label="buggy mutant")
    d = np.mean([np.linalg.norm(X[a] - X[b]) for a, b in pairs])
    ax.set_title(f"{title}\nmean correct-to-buggy distance: {d:.1f}"); ax.set_xticks([]); ax.set_yticks([]); ax.legend()
plt.suptitle(f"{len(pairs)} MBPP tasks: same bug, two embedding spaces"); plt.tight_layout()
os.makedirs("results/plots", exist_ok=True); plt.savefig("results/plots/tsne_comparison.png", dpi=200)
print("saved results/plots/tsne_comparison.png with", len(pairs), "pairs")
