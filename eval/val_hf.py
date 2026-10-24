# usage: python eval/val_hf.py --model DIR --out FILE.json [--sets datasets/mbpp_val.jsonl,datasets/mbpp_valext.jsonl]
# Greedy pass@1 on the held-out MBPP validation splits (all asserts), used ONLY to choose hyper-parameters.
# Same rule as eval/val_mbpp.py (setup + code + all tests must exit 0) but generated with eval/generate_hf.py's
# loader so it matches the test-time decoding exactly (works for prefix runs and base checkpoints).
import argparse, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_hf import load, run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sets", default="datasets/mbpp_val.jsonl,datasets/mbpp_valext.jsonl")
ap.add_argument("--batch", type=int, default=32)
a = ap.parse_args()

tok, model, head, inj, cfg = load(a.model)
rows = []
for vf in a.sets.split(","):
    name = os.path.basename(vf).removeprefix("mbpp_").removesuffix(".jsonl")
    rows += [dict(json.loads(l), _set=name) for l in open(vf)]
import generate_hf as G                                        # noqa: E402
raw_store = []
_orig = G.extract_code
def _keep(text):                                               # record raw completions alongside extracted code
    raw_store.append(text); return _orig(text)
G.extract_code = _keep
codes, _ = run(tok, model, head, inj, cfg, [r["messages"][1]["content"] for r in rows], a.batch)


def check(args):
    r, code = args
    src = (r.get("setup") or "") + "\n" + code + "\n" + "\n".join(r["tests"])
    try:
        p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=10, cwd=tempfile.gettempdir())
        return p.returncode == 0, (p.stderr or "").strip().splitlines()[-1:] or [""]
    except subprocess.TimeoutExpired:
        return False, ["Timeout"]


with ThreadPoolExecutor(8) as ex:
    checked = list(ex.map(check, zip(rows, codes)))
ok = [c[0] for c in checked]
per = {f"{r['_set']}/{r['task_id']}": bool(o) for r, o in zip(rows, ok)}
res = {"model": a.model, "n": len(rows), "passed": int(sum(ok)), "pass1": round(sum(ok) / len(rows), 4),
       "samples": [{"task": r["task_id"], "prompt": r["messages"][1]["content"][:300], "raw": raw[:1500], "code": c[:1500],
                 "ok": o, "error": e[0][:300]} for r, raw, c, (o, e) in list(zip(rows, raw_store, codes, checked))[:40]],
       "by_set": {s: round(sum(o for r, o in zip(rows, ok) if r["_set"] == s) / max(1, sum(r["_set"] == s for r in rows)), 4)
                  for s in sorted({r["_set"] for r in rows})}, "per_task": per}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"))
print(json.dumps({k: v for k, v in res.items() if k not in ("per_task", "samples")}))
