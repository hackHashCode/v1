"""Fallback mutant-separation test with automatic operator-swap mutants.
python fallback/mutant_test_min.py  -> fallback/mutant_scores.json
"""
import ast, json, random, sys
import torch.nn.functional as F
sys.path.insert(0, "fallback")
from build_cache_min import calls, probes, run, embed
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Add: ast.Sub, ast.Sub: ast.Add}

class Mut(ast.NodeTransformer):
    def __init__(self, k):
        self.k, self.i = k, 0
    def _m(self, op):
        if type(op) in SWAP:
            self.i += 1
            if self.i == self.k:
                return SWAP[type(op)]()
        return op
    def visit_Compare(self, n):
        self.generic_visit(n); n.ops = [self._m(o) for o in n.ops]; return n
    def visit_BinOp(self, n):
        self.generic_visit(n); n.op = self._m(n.op); return n

enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Optional: also score behaviour vectors built with CodeBERT (mean-pooled, not [CLS]).  --codebert
cb = None
if "--codebert" in sys.argv:
    from transformers import AutoTokenizer, AutoModel
    import torch
    cb_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base"); cb_m = AutoModel.from_pretrained("microsoft/codebert-base").eval()
    def cb_embed(lines):
        with torch.no_grad():
            x = cb_tok(lines, return_tensors="pt", truncation=True, max_length=128, padding=True)
            hs = cb_m(**x).last_hidden_state; msk = x["attention_mask"].unsqueeze(-1)
            v = F.normalize((hs * msk).sum(1) / msk.sum(1), dim=-1).mean(0)
        return F.normalize(v, dim=0)
    cb = cb_embed
ds = load_dataset("google-research-datasets/mbpp", "full")["train"]
random.seed(0)
beh, txt, other, refs, beh_cb, other_cb, refs_cb = [], [], [], [], [], [], []
for ex in random.sample(list(ds), 120):
    pr = probes(calls(ex["test_list"]))
    ref = run(ex["code"], ex["test_setup_code"], pr)
    if not ref:
        continue
    zr = embed(enc, ref); refs.append(zr)
    if cb: zr_cb = cb(ref); refs_cb.append(zr_cb)
    for k in (1, 2, 3):
        try:
            mcode = ast.unparse(Mut(k).visit(ast.parse(ex["code"])))
        except Exception:
            continue
        mt = run(mcode, ex["test_setup_code"], pr)
        if mt and mt != ref:
            beh.append(F.cosine_similarity(zr, embed(enc, mt), dim=0).item())
            if cb: beh_cb.append(F.cosine_similarity(zr_cb, cb(mt), dim=0).item())
            ct = enc.encode([ex["code"], mcode], convert_to_tensor=True, normalize_embeddings=True)
            txt.append(F.cosine_similarity(ct[0], ct[1], dim=0).item())
            break
for a, b in zip(refs, refs[1:]):
    other.append(F.cosine_similarity(a, b, dim=0).item())
for a, b in zip(refs_cb, refs_cb[1:]):
    other_cb.append(F.cosine_similarity(a, b, dim=0).item())
m = lambda x: round(sum(x) / max(len(x), 1), 3)
res = {"n_mutants": len(beh), "behaviour_ref_vs_mutant": m(beh), "codetext_ref_vs_mutant": m(txt),
       "behaviour_ref_vs_other_task": m(other)}
if cb:
    res.update({"codebert_behaviour_ref_vs_mutant": m(beh_cb), "codebert_behaviour_ref_vs_other_task": m(other_cb)})
# Better encoder = larger gap between (ref vs other task) and (ref vs mutant) relative to its own scale: compare both.
print(res)
json.dump({**res, "behaviour": beh, "codetext": txt, "other": other}, open("fallback/mutant_scores.json", "w"))
